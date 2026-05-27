# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
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

"""
Pytest hooks shared across the test suite.

Surfaces the GRID_H x GRID_W decomposition used by the distributed tests when
the suite is launched in distributed mode (GRID_H or GRID_W > 1):
  * `pytest_report_header`: one line at the top of the pytest session output.
  * `pytest_collection_modifyitems`: appends `[h=H,w=W]` to each distributed test's
    node-id so the grid is visible alongside every test name in `-v` output (e.g.
    `test_distributed_neighborhood_attention_00[h=2,w=4]`).

Both hooks no-op when GRID_H == GRID_W == 1, so plain non-distributed pytest
invocations see zero additional output. The per-test suffix additionally
filters on filenames starting with `test_distributed_`.
"""

import os


def _grid():
    return int(os.getenv("GRID_H", 1)), int(os.getenv("GRID_W", 1))


def pytest_report_header(config):
    h, w = _grid()
    if h == 1 and w == 1:
        return None
    return f"distributed grid: GRID_H x GRID_W = {h} x {w}"


def pytest_collection_modifyitems(config, items):
    h, w = _grid()
    if h == 1 and w == 1:
        return
    suffix = f"[h={h},w={w}]"
    for item in items:
        if item.fspath.basename.startswith("test_distributed_"):
            item._nodeid = item.nodeid + suffix
