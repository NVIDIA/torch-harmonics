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

import functools
from copy import deepcopy


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
