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

# we need this in order to enable distributed
import os

import torch.distributed as dist

# those need to be global
_POLAR_PARALLEL_GROUP = None
_AZIMUTH_PARALLEL_GROUP = None
_IS_INITIALIZED = False


class _DistributedConfig:
    """
    Module-level configuration for torch_harmonics.distributed.
    Env vars are used as defaults but can be overridden programmatically, e.g.:

        from torch_harmonics.distributed import config
        config.debug = True
    """

    def __init__(self):
        self._debug = None

    @property
    def debug(self):
        if self._debug is None:
            return os.getenv("TORCH_HARMONICS_DISTRIBUTED_DEBUG", "0") == "1"
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = bool(value)

    def __repr__(self):
        return f"_DistributedConfig(debug={self.debug})"


# set up debug configuration
config = _DistributedConfig()


def polar_group():
    """Return the polar (latitudinal) process group registered by :func:`init`, or ``None``."""
    return _POLAR_PARALLEL_GROUP


def azimuth_group():
    """Return the azimuth (longitudinal) process group registered by :func:`init`, or ``None``."""
    return _AZIMUTH_PARALLEL_GROUP


def init(polar_process_group, azimuth_process_group):
    """
    Initialize the torch-harmonics distributed backend.

    This must be called before any distributed SHT, convolution, or other
    distributed module is used.  It registers two orthogonal process groups that
    define a 2-D process grid over the sphere: one group for the **polar**
    (latitudinal) dimension and one for the **azimuthal** (longitudinal)
    dimension.

    The two groups are typically created from a single
    :func:`torch.distributed.new_subgroups_by_enumeration` (or equivalent) call
    so that every global rank belongs to exactly one polar group and one azimuth
    group.  See the :doc:`distributed guide </guide/distributed>` for a
    complete example of how to build the orthogonal communicator grid.

    Parameters
    ----------
    polar_process_group : torch.distributed.ProcessGroup
        Process group whose members share the same azimuthal index and
        collectively own all latitude chunks.
    azimuth_process_group : torch.distributed.ProcessGroup
        Process group whose members share the same polar index and
        collectively own all longitude chunks.

    Examples
    --------
    Setting up a 2 x 4 process grid on 8 GPUs (2 polar ranks, 4 azimuth
    ranks)::

        import torch.distributed as dist
        import torch_harmonics.distributed as thd

        dist.init_process_group(backend="nccl")
        world_rank = dist.get_rank()
        world_size = dist.get_world_size()   # 8

        num_polar, num_azimuth = 2, 4

        # ranks in the same row share a polar index -> azimuth group
        azimuth_group = dist.new_group(
            ranks=[r for r in range(world_size)
                   if r // num_azimuth == world_rank // num_azimuth]
        )
        # ranks in the same column share an azimuth index -> polar group
        polar_group = dist.new_group(
            ranks=[r for r in range(world_size)
                   if r % num_azimuth == world_rank % num_azimuth]
        )

        thd.init(polar_group, azimuth_group)
    """
    global _POLAR_PARALLEL_GROUP
    global _AZIMUTH_PARALLEL_GROUP
    global _IS_INITIALIZED
    _POLAR_PARALLEL_GROUP = polar_process_group
    _AZIMUTH_PARALLEL_GROUP = azimuth_process_group
    _IS_INITIALIZED = True


def finalize():
    """
    Tear down the torch-harmonics distributed backend.

    Destroys the polar and azimuth process groups that were registered by
    :func:`init` and resets the internal state.  After calling this function,
    :func:`is_initialized` returns ``False`` and distributed modules can no
    longer be used until :func:`init` is called again.
    """
    global _POLAR_PARALLEL_GROUP
    global _AZIMUTH_PARALLEL_GROUP
    global _IS_INITIALIZED
    if is_initialized():
        if is_distributed_polar():
            dist.destroy_process_group(_POLAR_PARALLEL_GROUP)
            _POLAR_PARALLEL_GROUP = None
        if is_distributed_azimuth():
            dist.destroy_process_group(_AZIMUTH_PARALLEL_GROUP)
            _AZIMUTH_PARALLEL_GROUP = None
    _IS_INITIALIZED = False


def is_initialized() -> bool:
    """Return ``True`` if :func:`init` has been called and :func:`finalize` has not."""
    return _IS_INITIALIZED


def is_distributed_polar() -> bool:
    """Return ``True`` if a polar process group has been registered."""
    return _POLAR_PARALLEL_GROUP is not None


def is_distributed_azimuth() -> bool:
    """Return ``True`` if an azimuth process group has been registered."""
    return _AZIMUTH_PARALLEL_GROUP is not None


def polar_group_size() -> int:
    """Return the number of ranks in the polar group (1 if not distributed)."""
    if not is_distributed_polar():
        return 1
    else:
        return dist.get_world_size(group=_POLAR_PARALLEL_GROUP)


def azimuth_group_size() -> int:
    """Return the number of ranks in the azimuth group (1 if not distributed)."""
    if not is_distributed_azimuth():
        return 1
    else:
        return dist.get_world_size(group=_AZIMUTH_PARALLEL_GROUP)


def polar_group_rank() -> int:
    """Return this rank's index within the polar group (0 if not distributed)."""
    if not is_distributed_polar():
        return 0
    else:
        return dist.get_rank(group=_POLAR_PARALLEL_GROUP)


def azimuth_group_rank() -> int:
    """Return this rank's index within the azimuth group (0 if not distributed)."""
    if not is_distributed_azimuth():
        return 0
    else:
        return dist.get_rank(group=_AZIMUTH_PARALLEL_GROUP)
