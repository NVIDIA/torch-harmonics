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
import torch
import torch.distributed as dist

# those need to be global
_POLAR_PARALLEL_GROUP = None
_AZIMUTH_PARALLEL_GROUP = None
_IS_INITIALIZED = False

def polar_group():
    return _POLAR_PARALLEL_GROUP

def azimuth_group():
    return _AZIMUTH_PARALLEL_GROUP

def init(polar_process_group, azimuth_process_group):
    global _POLAR_PARALLEL_GROUP
    global _AZIMUTH_PARALLEL_GROUP
    _POLAR_PARALLEL_GROUP = polar_process_group
    _AZIMUTH_PARALLEL_GROUP = azimuth_process_group
    _IS_INITIALIZED = True

def finalize():
    if is_initialized():
        if is_distributed_polar():
            dist.destroy_process_group(_POLAR_PARALLEL_GROUP)
        if is_distributed_azimuth():
            ist.destroy_process_group(_AZIMUTH_PARALLEL_GROUP)

def is_initialized() -> bool:
    return _IS_INITIALIZED

def is_distributed_polar() -> bool:
    return (_POLAR_PARALLEL_GROUP is not None)

def is_distributed_azimuth() -> bool:
    return (_AZIMUTH_PARALLEL_GROUP is not None)

def polar_group_size() -> int:
    if not is_distributed_polar():
        return 1
    else:
        return dist.get_world_size(group = _POLAR_PARALLEL_GROUP)

def azimuth_group_size() -> int:
    if not is_distributed_azimuth():
        return 1
    else:
        return dist.get_world_size(group = _AZIMUTH_PARALLEL_GROUP)

def polar_group_rank() -> int:
    if not is_distributed_polar():
        return 0
    else:
        return dist.get_rank(group = _POLAR_PARALLEL_GROUP)

def azimuth_group_rank() -> int:
    if not is_distributed_azimuth():
        return 0
    else:
        return dist.get_rank(group = _AZIMUTH_PARALLEL_GROUP)
