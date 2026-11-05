# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import copy as _copy
import functools
import inspect
import threading
import weakref
from copy import deepcopy
from typing import Any, Callable, Dict, Hashable, Optional, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from torch_harmonics import utils as _utils
from torch_harmonics.utils import recast_buffers_for


# copying LRU cache decorator a la:
# https://stackoverflow.com/questions/54909357/how-to-get-functools-lru-cache-to-return-new-instances
def lru_cache(maxsize=20, typed=False, copy=False):
    """
    Least Recently Used (LRU) cache decorator with optional deep copying.

    This is a wrapper around functools.lru_cache that adds the ability to return
    deep copies of cached results to prevent unintended modifications to cached objects.

    Parameters
    ----------
    maxsize : int, optional
        Maximum number of items to cache, by default 20
    typed : bool, optional
        Whether to cache different types separately, by default False
    copy : bool, optional
        Whether to return deep copies of cached results, by default False

    Returns
    -------
    function
        Decorated function with LRU caching. Metadata of the wrapped function is
        preserved via :func:`functools.wraps`, and the ``cache_info``,
        ``cache_clear`` and ``cache_parameters`` accessors of the underlying
        :func:`functools.lru_cache` are forwarded onto the wrapper.

    Examples
    --------
    >>> @lru_cache(maxsize=10, copy=True)
    ... def expensive_function(x):
    ...     return [x, x*2, x*3]
    >>> expensive_function(2)
    [2, 4, 6]
    >>> expensive_function(2)
    [2, 4, 6]
    >>> expensive_function.cache_info().hits
    1
    >>> expensive_function.cache_clear()
    >>> expensive_function.cache_info().currsize
    0
    """

    def decorator(f):
        cached_func = functools.lru_cache(maxsize=maxsize, typed=typed)(f)

        # functools.wraps carries the name, docstring and signature of `f` across.
        # Without it every decorated function is documented as "wrapper(*args,
        # **kwargs)" with no docstring, which silently dropped the documentation of
        # every cached routine in the package.
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            res = cached_func(*args, **kwargs)
            if copy:
                return deepcopy(res)
            else:
                return res

        # Forward the introspection and invalidation hooks. functools.lru_cache
        # attaches these to the object it returns, which the copying wrapper would
        # otherwise hide, leaving callers with no way to inspect hit rates or drop
        # stale entries.
        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear
        wrapper.cache_parameters = cached_func.cache_parameters

        return wrapper

    return decorator


# -- shared tables -------------------------------------------------------------------------
#
# The layers precompute large tables -- Legendre polynomials, quadrature-weighted projection
# rows, DISCO filter tensors, attention neighbourhoods -- that depend only on a handful of
# hashable arguments. A model routinely holds many layers with the same arguments, and a
# copying cache gave each of them its own copy, on the host and again on every device, while
# keeping up to `maxsize` further copies alive for the lifetime of the process.
#
# The registry below keeps exactly one copy of each table per (arguments, device, dtype) and
# holds it weakly: a table lives as long as some layer uses it, and identical layers that
# exist at the same time share it. Deleting the last layer that uses a table frees it.
#
# Shared tables are read-only by convention. Nothing in the library writes to one in place,
# and a layer that needs a modified table derives a new tensor; a user writing into a layer's
# table buffer in place changes it for every layer sharing it.

_MISSING = object()

#: a member's rounding history: the dtypes its values were narrowed to, in order, on the way
#: from the table as built. Empty for a member holding the exact values
History = Tuple[torch.dtype, ...]


def _kind(dtype: torch.dtype) -> int:
    if dtype == torch.bool:
        return 0
    if dtype.is_complex:
        return 3
    if dtype.is_floating_point:
        return 2
    return 1


def _lossless(source: torch.dtype, target: torch.dtype) -> bool:
    """Whether every value of dtype ``source`` is exactly representable in dtype ``target``."""
    if source == target:
        return True
    if torch.promote_types(source, target) != target:
        return False
    # promotion alone would call int64 -> float32 safe; across kinds only widening is
    return _kind(source) == _kind(target) or _kind(source) == 0 or (_kind(source) == 2 and _kind(target) == 3)


class _Strong:
    """A non-tensor leaf of a shared value, held strongly; called like a weakref."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class _Record:
    """
    How the value under one key was built: the builder, the tree structure and the non-tensor leaves.

    Outlives the entry, which goes as soon as any of its tensors dies: every family of the
    value holds its record, so a value can be put back together, or built again, while any
    copy of any of its tensors is in use.
    """

    __slots__ = ("key", "build", "spec", "strong", "families", "__weakref__")

    def __init__(self, key: Hashable, build: Optional[Callable[[], Any]], spec, strong: Dict[int, Any]):
        self.key = key
        self.build = build
        self.spec = spec
        self.strong = strong
        # leaf index -> weakref to its family; the families hold the record, not the reverse
        self.families: Dict[int, weakref.ref] = {}

    def family(self, index: int) -> Optional["_Family"]:
        ref = self.families.get(index)
        return None if ref is None else ref()


class _Family:
    """
    One tensor of a shared value and all its copies, keyed by ``(device, dtype, history)``.

    Alive while any member is: the registry holds a family only through its members' entries.
    """

    __slots__ = ("record", "index", "root_device", "root_dtype", "members", "__weakref__")

    def __init__(self, record: _Record, index: int, root: torch.Tensor):
        self.record = record
        self.index = index
        # where and in what dtype the builder produces this tensor
        self.root_device = root.device
        self.root_dtype = root.dtype
        # (device, dtype, history) -> (weakref, token)
        self.members: Dict[Tuple[torch.device, torch.dtype, History], Tuple[weakref.ref, object]] = {}

    def member(self, device: torch.device, dtype: torch.dtype, history: History) -> Optional[torch.Tensor]:
        entry = self.members.get((device, dtype, history))
        return None if entry is None else entry[0]()

    def live_members(self):
        """``(history, tensor)`` of every live member."""
        for (_, _, history), (ref, _) in list(self.members.items()):
            tensor = ref()
            if tensor is not None:
                yield history, tensor


def _replay(source: torch.Tensor, done: int, device: torch.device, dtype: torch.dtype, history: History) -> torch.Tensor:
    """``source``, which went through ``history[:done]``, taken through the rest of ``history`` to ``device`` in ``dtype``."""
    out = source
    for narrowed in history[done:]:
        out = out.to(dtype=narrowed)
    return out.to(device=device, dtype=dtype)


class TableRegistry:
    """
    Process-wide, weak-valued store of precomputed tables.

    An entry maps a hashable key to a value -- a tensor or a nested tuple, list, dict or
    named tuple of them, possibly with non-tensor leaves -- whose tensors are held weakly.
    The entry disappears when any of its tensors is garbage collected, so a lookup either
    returns the complete value as it was stored or misses.

    Every tensor the registry hands out belongs to a *family*: the tensor as first built,
    and its copies on other devices and in other dtypes. A member is identified by its
    device, its dtype and its *history*, the dtypes its values were narrowed to on the way
    from the table as built, so a copy that went through half precision is never handed to
    a layer that asked for a float32 copy of the exact values. :meth:`resolve` returns the
    family's member for a target, creating it on a miss from a live member that holds the
    values it needs, or else by building the table again. That is also how a value is put
    back together after the tensors first built have died: a layer constructed while
    identical layers live on another device gets its table from their copy, not by
    building it again.

    Not meant to be used directly: :func:`shared_table` keys builders by their arguments,
    and :class:`SharedBufferMixin` and the backend selection in ``torch_harmonics._backend``
    move shared state between devices.
    """

    def __init__(self):
        # builds and the creation of members are serialized, so two threads asking for the
        # same table build it once; re-entrant, since a builder may call other shared builders
        self._lock = threading.RLock()
        self._entries: Dict[Hashable, Tuple[Any, list, object]] = {}
        # key -> record, for as long as some family of the key is alive
        self._records: "weakref.WeakValueDictionary[Hashable, _Record]" = weakref.WeakValueDictionary()
        # id(tensor) -> (weakref, family, history, token); ids are recycled, so a hit is confirmed by the weakref
        self._members: Dict[int, Tuple[weakref.ref, _Family, History, object]] = {}

    # -- entries -------------------------------------------------------------------------

    def lookup(self, key: Hashable) -> Any:
        """The value stored under ``key`` if all of its tensors are alive, else ``_MISSING``."""
        entry = self._entries.get(key)
        if entry is None:
            return _MISSING
        spec, refs, _ = entry
        leaves = []
        for ref in refs:
            value = ref()
            if value is None and not isinstance(ref, _Strong):
                return _MISSING
            leaves.append(value)
        return tree_unflatten(leaves, spec)

    def store(self, key: Hashable, value: Any, build: Optional[Callable[[], Any]] = None) -> Any:
        """
        Store ``value`` under ``key``, replacing any entry, and return it.

        ``build`` rebuilds the value; without it a family of the value can only be served
        from its live members.
        """
        leaves, spec = tree_flatten(value)
        with self._lock:
            record = self._records.get(key)
            if record is None or record.spec != spec:
                strong = {index: leaf for index, leaf in enumerate(leaves) if not isinstance(leaf, torch.Tensor)}
                record = _Record(key, build, spec, strong)
                self._records[key] = record
            elif build is not None:
                record.build = build
            token = object()
            refs = []
            for index, leaf in enumerate(leaves):
                if isinstance(leaf, torch.Tensor):
                    refs.append(weakref.ref(leaf, _drop_callback(self._entries, key, token)))
                    if self.membership(leaf) is None:
                        family = record.family(index)
                        if family is None:
                            family = _Family(record, index, leaf)
                            record.families[index] = weakref.ref(family)
                        # stored as built, or put back together from exact copies: no history
                        self._adopt(leaf, family, ())
                else:
                    refs.append(_Strong(leaf))
            self._entries[key] = (spec, refs, token)
        return value

    def get_or_build(self, key: Hashable, build: Callable[[], Any]) -> Any:
        """
        The value under ``key``, built and stored on a miss.

        A miss is first answered from the families of an earlier value under ``key``: if a
        copy with the exact values of each of its tensors is alive anywhere, the value is put
        back together from those, on the device and in the dtype the builder produces.
        """
        value = self.lookup(key)
        if value is not _MISSING:
            return value
        with self._lock:
            value = self.lookup(key)
            if value is _MISSING:
                value = self._reassemble(key)
            if value is _MISSING:
                value = build()
            value = self.store(key, value, build)
        return value

    def _reassemble(self, key: Hashable) -> Any:
        record = self._records.get(key)
        if record is None:
            return _MISSING
        leaves = []
        for index in range(record.spec.num_leaves):
            if index in record.strong:
                leaves.append(record.strong[index])
                continue
            family = record.family(index)
            tensor = None if family is None else self._from_members(family, family.root_device, family.root_dtype, ())
            if tensor is None:
                return _MISSING
            leaves.append(tensor)
        return tree_unflatten(leaves, record.spec)

    def __len__(self) -> int:
        """Number of live entries."""
        return sum(self.lookup(key) is not _MISSING for key in list(self._entries))

    # -- families ------------------------------------------------------------------------

    def _adopt(self, tensor: torch.Tensor, family: _Family, history: History) -> None:
        """Make ``tensor`` the member of ``family`` with ``history``, unless it already belongs to one."""
        if self.membership(tensor) is not None:
            return
        token = object()
        tid = id(tensor)
        self._members[tid] = (weakref.ref(tensor, _drop_callback(self._members, tid, token)), family, history, token)
        mkey = (tensor.device, tensor.dtype, history)
        family.members[mkey] = (weakref.ref(tensor, _drop_callback(family.members, mkey, token)), token)

    def membership(self, tensor: torch.Tensor) -> Optional[Tuple[_Family, History]]:
        """The family ``tensor`` belongs to and its history, or ``None`` if the registry did not hand it out."""
        entry = self._members.get(id(tensor))
        if entry is None or entry[0]() is not tensor:
            return None
        return entry[1], entry[2]

    def _from_members(self, family: _Family, device: torch.device, dtype: torch.dtype, history: History, held: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        The member for a target, made from a live member if one holds the values it needs.

        A source must hold data -- not be on the meta device -- and its history must be a
        prefix of the target's: the rest of the history is replayed on it before the move.
        Of several, one already on ``device`` and with the longest history is preferred.
        """
        existing = family.member(device, dtype, history)
        if existing is not None:
            return existing
        candidates = list(family.live_members())
        if held is not None:
            membership = self.membership(held)
            if membership is not None:
                candidates.append((membership[1], held))
        best = None
        for source_history, source in candidates:
            if source.is_meta or source_history != history[: len(source_history)]:
                continue
            rank = (source.device == device, len(source_history))
            if best is None or rank > best[0]:
                best = (rank, source_history, source)
        if best is None:
            return None
        _, source_history, source = best
        out = _replay(source, len(source_history), device, dtype, history)
        self._adopt(out, family, history)
        return out

    def resolve(self, family: _Family, device: torch.device, dtype: torch.dtype, history: History, held: torch.Tensor, fallback: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        The member of ``family`` on ``device`` in ``dtype`` with ``history``.

        Made on a miss from a live member that holds the values it needs, else from the table
        built again; ``held`` is the member asking. A family that cannot be built again (state
        stored without a builder) and has no such member is served by ``fallback``, whose
        result does not join the family.
        """
        with self._lock:
            existing = family.member(device, dtype, history)
            if existing is not None:
                return existing
            if device.type == "meta":
                # no values needed: any member gives the shape
                out = held.to(device=device, dtype=dtype)
                self._adopt(out, family, history)
                return out
            out = self._from_members(family, device, dtype, history, held)
            if out is not None:
                return out
            record = family.record
            if record.build is None:
                return fallback()
            value = self.get_or_build(record.key, record.build)
            root = tree_flatten(value)[0][family.index]
            if self.membership(root)[0] is not family:
                # a rebuild after the record was replaced; serve the copy, as unshared
                return _replay(root, 0, device, dtype, history)
            return self._from_members(family, device, dtype, history, root)


def _drop_callback(table: dict, key: Hashable, token: object):
    """
    A weakref callback that removes ``table[key]`` if it is still the entry ``token`` marks.

    Entries are tuples ending in their token, which guards against removing a newer entry
    stored under the same key after the old one died. Callbacks take no lock: they can run
    inside any allocation, and a single dict operation is atomic under the GIL.
    """

    def drop(_ref):
        entry = table.get(key)
        if entry is not None and entry[-1] is token:
            table.pop(key, None)

    return drop


#: the process-wide registry
_REGISTRY = TableRegistry()


def _typed(value: Any) -> Any:
    """A key component that tells ``1``, ``1.0`` and ``True`` apart, as ``lru_cache(typed=True)`` does."""
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_typed(v) for v in value))
    return (type(value), value)


def shared_table(f: Callable) -> Callable:
    """
    Share the tables a builder returns, keyed by its arguments.

    The decorated function is called at most once per distinct set of arguments while its
    result is alive anywhere: as long as some layer holds the returned tensors, a call with
    the same arguments returns the very same tensors. Once they have all been released the
    next call builds anew. The arguments are normalized through the signature, so
    positional and keyword spellings of one call share an entry, and typed, so ``1`` and
    ``1.0`` do not. They must be hashable. Each decorated function has entries of its own:
    two builders never share, even when they have the same name, as closures made by one
    factory do.

    The returned tensors are shared and must not be modified in place.

    Examples
    --------
    >>> import torch
    >>> @shared_table
    ... def ramp(n, scale=1.0):
    ...     return scale * torch.arange(n, dtype=torch.float64)
    >>> a = ramp(4)
    >>> b = ramp(n=4, scale=1.0)
    >>> a is b
    True
    >>> ramp(4, scale=2.0) is a
    False
    """
    signature = inspect.signature(f)
    # identifies this builder: the name alone does not, closures from one factory share it
    token = object()

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = []
        for name, value in bound.arguments.items():
            kind = signature.parameters[name].kind
            if kind is inspect.Parameter.VAR_KEYWORD:
                value = tuple(sorted(value.items()))
            arguments.append((name, _typed(value)))
        key = (f.__module__, f.__qualname__, token, tuple(arguments))
        return _REGISTRY.get_or_build(key, lambda: f(*args, **kwargs))

    return wrapper


def shared_value(key: Hashable, build: Callable[[], Any]) -> Any:
    """The value stored under ``key`` in the shared registry, built by ``build`` on a miss."""
    return _REGISTRY.get_or_build(key, build)


def lookup_shared(key: Hashable) -> Any:
    """The value stored under ``key``, or ``None`` on a miss."""
    value = _REGISTRY.lookup(key)
    return None if value is _MISSING else value


def store_shared(key: Hashable, value: Any) -> Any:
    """Store ``value`` under ``key`` in the shared registry and return it."""
    return _REGISTRY.store(key, value)


def is_shared(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` was handed out by the shared registry."""
    return _REGISTRY.membership(tensor) is not None


def _conversion_target(fn: Callable, tensor: torch.Tensor) -> Tuple[torch.device, torch.dtype]:
    """
    Where ``module._apply(fn)`` would send ``tensor``: its device and dtype afterwards.

    Found by applying ``fn`` to empty probes on the tensor's device, so nothing is copied. A
    float64 tensor bound for a device without float64 lands as float32, as
    :func:`~torch_harmonics.utils.recast_buffers_for` arranges for unshared buffers.
    """
    probe = fn(torch.empty(0, dtype=torch.float32, device=tensor.device))
    device = probe.device
    dtype = tensor.dtype
    # read at call time, so a patched set (as the tests use to emulate MPS) takes effect
    if dtype == torch.float64 and device.type in _utils._NO_FLOAT64_DEVICE_TYPES:
        dtype = torch.float32
    if dtype != torch.float32:
        probe = fn(torch.empty(0, dtype=dtype, device=tensor.device))
    return device, probe.dtype


def move_shared(tensor: torch.Tensor, fn: Callable) -> torch.Tensor:
    """
    ``fn(tensor)`` for a shared tensor, as the family's copy on the target.

    Only the device and dtype ``fn`` produces are honoured; a conversion that changes
    neither returns ``tensor`` itself. The copy holds the values ``tensor`` would convert
    to: a narrowing conversion adds to the history it carries, so the float32 copy of a
    table that went through half precision is not the float32 copy of the exact table.
    It need not be made from ``tensor`` itself, though, which is what lets a table on the
    meta device come back, with its values, under ``to_empty``.
    """
    device, dtype = _conversion_target(fn, tensor)
    if device == tensor.device and dtype == tensor.dtype:
        return tensor

    def fallback():
        # what an unshared buffer would get; a meta tensor has no values to convert
        if tensor.is_meta and device.type != "meta":
            return fn(tensor).to(dtype=dtype)
        return tensor.to(device=device, dtype=dtype)

    membership = _REGISTRY.membership(tensor)
    if membership is None:
        return fallback()
    family, history = membership
    if not _lossless(tensor.dtype, dtype):
        history = history + (dtype,)
    return _REGISTRY.resolve(family, device, dtype, history, tensor, fallback)


def _deepcopy_sharing(module: torch.nn.Module, memo: dict, shared: list) -> torch.nn.Module:
    """``copy.deepcopy(module)``, with the objects in ``shared`` referenced rather than copied."""
    for obj in shared:
        memo[id(obj)] = obj
    reductor = module.__reduce_ex__(4)
    return _copy._reconstruct(module, memo, *reductor)


class SharedBufferMixin:
    """
    Buffers shared between identical layers, on every device.

    Mix in ahead of :class:`torch.nn.Module`. :meth:`register_shared_buffer` registers a
    non-persistent buffer holding a shared table. On every conversion -- ``.to()``,
    ``.cuda()``, ``.half()`` and the rest all end in ``_apply`` -- the buffer is replaced by
    the family's copy on the target rather than converted per layer, so N identical layers
    moved to a device hold one copy there, and only the first of them pays the transfer.
    A deep copy shares the tables too.

    Unshared buffers are converted as usual, float64 ones downcast for a device without
    float64 as :func:`~torch_harmonics.utils.recast_buffers_for` does; shared ones follow
    the same rule.
    """

    #: names of the buffers holding shared tables
    _shared_buffers: Tuple[str, ...] = ()

    def register_shared_buffer(self, name: str, tensor: Optional[torch.Tensor]) -> None:
        """Register ``tensor``, a shared table, as the non-persistent buffer ``name``."""
        self.register_buffer(name, tensor, persistent=False)
        if name not in self._shared_buffers:
            self._shared_buffers = (*self._shared_buffers, name)

    def _shared_objects(self) -> list:
        return [self._buffers[name] for name in self._shared_buffers if self._buffers.get(name) is not None]

    def _apply(self, fn, recurse=True):
        # take the shared buffers out, so the default conversion does not copy them per layer
        held = {}
        for name in self._shared_buffers:
            tensor = self._buffers.get(name)
            if tensor is not None:
                held[name] = tensor
                self._buffers[name] = None
        try:
            # float64 buffers cannot move to MPS; see recast_buffers_for
            recast_buffers_for(self, fn)
            out = super()._apply(fn, recurse)
        except BaseException:
            self._buffers.update(held)
            raise
        for name, tensor in held.items():
            self._buffers[name] = move_shared(tensor, fn)
        return out

    def __deepcopy__(self, memo):
        return _deepcopy_sharing(self, memo, self._shared_objects())
