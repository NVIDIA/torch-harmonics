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


class _Strong:
    """A non-tensor leaf of a shared value, held strongly; called like a weakref."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class TableRegistry:
    """
    Process-wide, weak-valued store of precomputed tables.

    An entry maps a hashable key to a value -- a tensor or a nested tuple, list, dict or
    named tuple of them, possibly with non-tensor leaves -- whose tensors are held weakly.
    The entry disappears when any of its tensors is garbage collected, so a lookup either
    returns the complete value as it was stored or misses.

    Every tensor the registry hands out belongs to a *family*: the tensor as first built,
    and the copies of it on other devices or in other dtypes, at most one per
    ``(device, dtype)``. :meth:`variant` returns the family's copy for a target, creating it
    from the tensor at hand on a miss, which is what lets layers moved with ``.to()`` keep
    sharing.

    Not meant to be used directly: :func:`shared_table` keys builders by their arguments,
    and :class:`SharedBufferMixin` and the backend selection in ``torch_harmonics._backend``
    move shared state between devices.
    """

    def __init__(self):
        # builds are serialized, so two threads asking for the same table build it once;
        # re-entrant, since a builder may call other shared builders
        self._build_lock = threading.RLock()
        self._entries: Dict[Hashable, Tuple[Any, list, object]] = {}
        # id(tensor) -> (weakref, family, token); ids are recycled, so a hit is confirmed by the weakref
        self._families: Dict[int, Tuple[weakref.ref, Hashable, object]] = {}
        # (family, device, dtype) -> (weakref, token)
        self._variants: Dict[Tuple[Hashable, torch.device, torch.dtype], Tuple[weakref.ref, object]] = {}

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

    def store(self, key: Hashable, value: Any) -> Any:
        """Store ``value`` under ``key``, replacing any entry, and return it."""
        leaves, spec = tree_flatten(value)
        token = object()
        refs = []
        for index, leaf in enumerate(leaves):
            if isinstance(leaf, torch.Tensor):
                refs.append(weakref.ref(leaf, _drop_callback(self._entries, key, token)))
                self._adopt(leaf, (key, index))
            else:
                refs.append(_Strong(leaf))
        self._entries[key] = (spec, refs, token)
        return value

    def get_or_build(self, key: Hashable, build: Callable[[], Any]) -> Any:
        """The value under ``key``, built and stored on a miss."""
        value = self.lookup(key)
        if value is not _MISSING:
            return value
        with self._build_lock:
            value = self.lookup(key)
            if value is _MISSING:
                value = self.store(key, build())
        return value

    def __len__(self) -> int:
        """Number of live entries."""
        return sum(self.lookup(key) is not _MISSING for key in list(self._entries))

    # -- families ------------------------------------------------------------------------

    def _adopt(self, tensor: torch.Tensor, family: Hashable) -> None:
        """Make ``tensor`` a member of ``family``, unless it already belongs to one."""
        if self.family_of(tensor) is not None:
            return
        token = object()
        tid = id(tensor)
        self._families[tid] = (weakref.ref(tensor, _drop_callback(self._families, tid, token)), family, token)
        vkey = (family, tensor.device, tensor.dtype)
        self._variants[vkey] = (weakref.ref(tensor, _drop_callback(self._variants, vkey, token)), token)

    def family_of(self, tensor: torch.Tensor) -> Optional[Hashable]:
        """The family ``tensor`` belongs to, or ``None`` if the registry did not hand it out."""
        entry = self._families.get(id(tensor))
        if entry is None or entry[0]() is not tensor:
            return None
        return entry[1]

    def variant(self, tensor: torch.Tensor, device: torch.device, dtype: torch.dtype, convert: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        The member of ``tensor``'s family on ``device`` in ``dtype``.

        Created by ``convert`` on a miss and adopted into the family. A tensor the registry
        did not hand out is simply converted.
        """
        family = self.family_of(tensor)
        if family is None:
            return convert()
        entry = self._variants.get((family, device, dtype))
        existing = None if entry is None else entry[0]()
        if existing is not None:
            return existing
        converted = convert()
        self._adopt(converted, family)
        return converted


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
    return _REGISTRY.family_of(tensor) is not None


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
    neither returns ``tensor`` itself.
    """
    device, dtype = _conversion_target(fn, tensor)
    if device == tensor.device and dtype == tensor.dtype:
        return tensor
    return _REGISTRY.variant(tensor, device, dtype, lambda: tensor.to(device=device, dtype=dtype))


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
