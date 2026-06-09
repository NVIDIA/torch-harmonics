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

It also keeps rank 0 the single source of console output:
  * `pytest_configure`: on any non-zero WORLD_RANK, unregisters pytest's terminal
    reporter so only rank 0 prints the session header, progress, per-test results
    and summary. This is safe because distributed comparisons all-reduce their
    verdict (see testutils.reduce_success), so rank 0 fails whenever any rank does
    -- the silenced ranks can never hide a failure. Each rank still returns its own
    process exit code to the launcher.

The grid hooks no-op when GRID_H == GRID_W == 1, and the reporter is only silenced
on non-zero ranks, so plain non-distributed pytest invocations see zero change in
output. The per-test suffix additionally filters on filenames starting with
`test_distributed_`.
"""

import os

import pytest


def _grid():
    return int(os.getenv("GRID_H", 1)), int(os.getenv("GRID_W", 1))


def _world_rank():
    # Prefer WORLD_RANK; fall back to RANK (both are exported by the launch
    # scripts, but RANK is the more common convention for torchrun-style launches).
    rank = os.getenv("WORLD_RANK")
    if rank is None:
        rank = os.getenv("RANK", 0)
    return int(rank)


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # Silence pytest's own console output on every rank except rank 0.
    #
    # trylast=True is required: conftest hooks run before the builtin terminal
    # plugin's pytest_configure (LIFO order), so without it the "terminalreporter"
    # plugin is not registered yet and get_plugin() returns None -- letting every
    # rank keep reporting (the duplicate-output bug). Running last ensures the
    # reporter exists by the time we unregister it.
    if _world_rank() == 0:
        return
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        config.pluginmanager.unregister(reporter)


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
