# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
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
# Balanced partitioning of a dimension across ranks.
#
# Deliberately free of any dependency on torch.distributed, so that both the grid
# descriptors and the distributed primitives can share one implementation. A second
# copy of this arithmetic in torch_harmonics.grid would be a bug waiting to happen:
# the two would have to agree exactly for a sharded grid to line up with the tensors
# the collectives move.

from typing import List

from torch_harmonics.utils import check


def compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    r"""
    Compute balanced chunk sizes for distributing a dimension across ranks.

    Divides ``size`` elements into ``num_chunks`` pieces that differ by at most
    one element.  The first ``size % num_chunks`` chunks receive one extra
    element; the remaining chunks get the base size ``size // num_chunks``.

    This is used internally by every distributed module to determine how
    latitudes, longitudes, and spectral modes are partitioned across process
    groups.

    Parameters
    ----------
    size : int
        Total number of elements to split (e.g.\ ``nlat`` or ``nlon``).
    num_chunks : int
        Number of chunks (typically the process-group size).

    Returns
    -------
    List[int]
        Per-rank chunk sizes, ordered by rank.

    Raises
    ------
    RuntimeError
        If ``size < num_chunks`` (every chunk must be non-empty).

    Examples
    --------
    >>> from torch_harmonics.partition import compute_split_shapes
    >>> compute_split_shapes(256, 4)
    [64, 64, 64, 64]
    >>> compute_split_shapes(128, 3)
    [43, 43, 42]
    >>> compute_split_shapes(10, 4)
    [3, 3, 2, 2]
    """

    check(size >= num_chunks, lambda: f"Cannot split {size} elements into {num_chunks} chunks; every chunk must be non-empty.")

    base, remainder = divmod(size, num_chunks)
    return [base + 1] * remainder + [base] * (num_chunks - remainder)
