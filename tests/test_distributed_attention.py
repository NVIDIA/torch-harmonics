# coding=utf-8

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
#

import math
import os
import unittest

import torch
import torch.distributed as dist
from parameterized import parameterized
from testutils import (
    amp_ulps,
    compare_tensors,
    gather_tensor_hw,
    maybe_autocast,
    param_grad_ulps,
    reduce_success,
    set_seed,
    setup_class_from_context,
    setup_module,
    split_tensor_hw,
    teardown_module,
)

import torch_harmonics as th
import torch_harmonics.distributed as thd
from torch_harmonics.distributed.distributed_attention import _RING_GATHER_DEVICES, _RING_UPSAMPLE_DEVICES
from torch_harmonics.radial import split_radial

# Opt-in gate for slow / large-grid parameterized cases (e.g. 721x1440 ERA5-like
# shapes, whose polar rows have the longest neighbourhoods).
_run_slow_tests = os.getenv("TORCH_HARMONICS_RUN_SLOW_TESTS", "0") == "1"

# (nlat_in, nlon_in, nlat_out, nlon_out) shapes whose parameterized cases are
# gated behind TORCH_HARMONICS_RUN_SLOW_TESTS=1.
_SLOW_ATTN_SHAPES = frozenset(
    {
        (721, 1440, 721, 1440),
        (721, 1440, 360, 720),
    }
)

# shared state
_DIST_CTX = {}

# the radial factor of the product-grid rows
_RADIAL = th.as_radial_grid("legendre-gauss", nr=2, r_inner=1.0, r_outer=2.0)


def _make_grid(name, shape):
    """
    A grid from a row's grid name and shape: ``(nside,)`` for HEALPix, ``(nlat, nlon)`` otherwise.

    A ``"+radial"`` suffix on the name makes it the product with ``_RADIAL``.
    """
    angular, _, radial = name.partition("+")
    grid = th.HealpixGrid(nside=shape[0]) if angular == "healpix" else th.as_grid(angular, nlat=shape[0], nlon=shape[1])
    if radial:
        grid = th.SphericalProductGrid(grid, _RADIAL)
    return grid


def _ulps(dtype):
    """
    ULP budget, of max|ref| in ``dtype``, for the output and input gradients. fp32: the
    distributed path reassociates a dot product, a softmax and a weighted neighborhood sum
    (measured: up to 29 ULPs on GB200). fp16/bf16: see amp_ulps. Parameter gradients use
    param_grad_ulps.
    """
    return amp_ulps(dtype) or 128.0


def setUpModule():
    setup_module(_DIST_CTX)


def tearDownModule():
    teardown_module(_DIST_CTX)


class TestDistributedNeighborhoodAttention(unittest.TestCase):
    """
    Compare serial NeighborhoodAttentionS2 against DistributedNeighborhoodAttentionS2.

    The ring kernels exist for CUDA only. Nothing here is skipped for want of them: on
    another device the comparisons assert that the layer refuses to run, and the tests
    that never reach a kernel -- backend state, shape checks -- run as they are.
    """

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)

    def _refuses_device(self, attn_dist):
        """
        On a device without the ring kernels, assert that the layer refuses to run and
        says why, rather than failing in the op dispatcher, and report that there is
        nothing to compare. On a device with them, report that there is.

        Only a pair of regular grids takes the ring kernels: a ragged layer runs the serial
        ragged backends, compiled for CPU and CUDA, with the reference everywhere else.
        """
        devices = _RING_UPSAMPLE_DEVICES if attn_dist.upsample else _RING_GATHER_DEVICES
        refuses = not attn_dist.ragged and self.device.type not in devices

        if refuses:
            C = attn_dist.in_channels
            q = torch.randn(1, C, *attn_dist.shard_out.shape, device=self.device)
            k = torch.randn(1, C, *attn_dist.shard_in.shape, device=self.device)
            with self.assertRaisesRegex(RuntimeError, "ring kernels run on CUDA only"):
                attn_dist(q, k, k)

        return refuses

    def _skip_unless_decomposable(self, *grids):
        """HEALPix decomposes along latitude only; a layer on it refuses an azimuth group, see test_healpix_refuses_an_azimuthal_decomposition."""
        if self.grid_size_w > 1 and any(isinstance(split_radial(g)[0], th.HealpixGrid) for g in grids):
            self.skipTest("HEALPix decomposes along latitude only")

    def _sync(self):
        """Wait for the device, where there is one to wait for."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _split_helper(self, tensor, shard):
        """This rank's piece of a global field, as ``shard`` describes it: a band of pixels, or a block of latitudes and longitudes."""
        if isinstance(shard.grid, th.HealpixGrid):
            tensor_local = torch.split(tensor, list(shard.pixel_shapes), dim=-1)[self.hrank].contiguous()
        else:
            tensor_local = split_tensor_hw(tensor, hdim=-2, wdim=-1, hsize=self.grid_size_h, wsize=self.grid_size_w, hrank=self.hrank, wrank=self.wrank)
        return tensor_local

    def _gather_helper(self, tensor, shard):
        """The global field from every rank's piece, as ``shard`` describes them."""
        if isinstance(shard.grid, th.HealpixGrid):
            hdim, hshapes, wshapes, wsize = -1, list(shard.pixel_shapes), [], 1
        else:
            hdim, hshapes, wshapes, wsize = -2, list(shard.lat_shapes), list(shard.lon_shapes), self.grid_size_w
        return gather_tensor_hw(
            tensor,
            hdim=hdim,
            wdim=-1,
            hshapes=hshapes,
            wshapes=wshapes,
            hsize=self.grid_size_h,
            wsize=wsize,
            hrank=self.hrank,
            wrank=self.wrank,
            hgroup=self.h_group,
            wgroup=self.w_group,
        )

    def _allreduce_param_grad(self, tensor):
        """
        Sum a per-rank parameter gradient over the polar and azimuth groups.

        The distributed layers leave parameter gradients partial: each rank's holds the
        contribution of its own spatial chunk, and reducing them is the caller's job, as
        for the distributed convolution (see test_distributed_convolution). Returns a
        clone, since all_reduce is in place.
        """
        out = tensor.clone()
        if self.grid_size_h > 1:
            dist.all_reduce(out, group=self.h_group)
        if self.grid_size_w > 1:
            dist.all_reduce(out, group=self.w_group)
        return out

    @parameterized.expand(
        [
            # grid_in, in_shape, grid_out, out_shape, batch_size, in_channels, num_heads, k_channels, out_channels, use_qknorm, dtype[, trainable_scale, optimized_kernel, theta_cutoff]
            # shapes are (nlat, nlon), or (nside,) for HEALPix
            # same shape tests
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 1, None, None, False, torch.float32],
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 2, None, None, False, torch.float32],
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 1, 8, 8, False, torch.float32],
            ["equiangular", (65, 128), "equiangular", (65, 128), 2, 16, 1, None, None, False, torch.float32],
            # Realistic ERA5-like grid with strongly unbalanced rows: near the poles the
            # neighbourhood disk covers the whole longitude ring, so pole rows hold
            # thousands of neighbours while mid-latitude rows hold tens, and pole arcs
            # span several ring chunks of the azimuth split. Memory note:
            # 721x1440 x B=2 x C=16 fp32 ~250 MB per major tensor; expect a few GB
            # working-set including halos and gradient buffers. Splittable up to 2x4
            # (uneven polar shard for odd nlat is handled by the test framework).
            ["equiangular", (721, 1440), "equiangular", (721, 1440), 2, 16, 1, None, None, False, torch.float32],
            # downsampling tests, pscale=2 (lat+lon)
            ["equiangular", (64, 128), "equiangular", (32, 64), 2, 16, 1, None, None, False, torch.float32],
            ["equiangular", (65, 128), "equiangular", (33, 64), 2, 16, 1, None, None, False, torch.float32],
            # pscale=3 downsampling (exercises pscale*wo kernel arithmetic)
            ["equiangular", (64, 96), "equiangular", (32, 32), 2, 16, 1, None, None, False, torch.float32],
            # pscale=4 downsampling
            ["equiangular", (64, 128), "equiangular", (32, 32), 2, 16, 1, None, None, False, torch.float32],
            # lon-only downsampling (same nlat; isolates azimuth ring with pscale_lon=2, no lat halo)
            ["equiangular", (64, 128), "equiangular", (64, 64), 2, 16, 1, None, None, False, torch.float32],
            # lat-only downsampling (same nlon, pscale_lon=1; isolates polar halo exchange)
            ["equiangular", (64, 128), "equiangular", (32, 128), 2, 16, 1, None, None, False, torch.float32],
            # odd nlat_in -> even nlat_out, pscale_lon=2
            ["equiangular", (65, 128), "equiangular", (32, 64), 2, 16, 1, None, None, False, torch.float32],
            # legendre-gauss grid (same-shape + downsampling)
            ["legendre-gauss", (64, 128), "legendre-gauss", (64, 128), 2, 16, 1, None, None, False, torch.float32],
            ["legendre-gauss", (64, 128), "legendre-gauss", (32, 64), 2, 16, 1, None, None, False, torch.float32],
            # mixed grid: equiangular input -> legendre-gauss output
            ["equiangular", (64, 128), "legendre-gauss", (64, 128), 2, 16, 1, None, None, False, torch.float32],
            ["equiangular", (64, 128), "legendre-gauss", (32, 64), 2, 16, 1, None, None, False, torch.float32],
            # Realistic ERA5-like downsample (equi -> LG, ~2x lat/lon). The default cutoff
            # spans a few input latitudes, so pole rows hold several x nlon_in entries
            # while equator rows stay short -- the same imbalance as above, combined with
            # pscale>1 and a mixed grid. Same memory caveat as the 721x1440 case above.
            ["equiangular", (721, 1440), "legendre-gauss", (360, 720), 2, 16, 1, None, None, False, torch.float32],
            # heads=4 with asymmetric channels (k=32, out=16; in=16)
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 4, 32, 16, False, torch.float32],
            ["equiangular", (64, 128), "equiangular", (32, 64), 2, 16, 4, 32, 16, False, torch.float32],
            # same cases with QK norm enabled
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 1, None, None, True, torch.float32],
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 2, None, None, True, torch.float32],
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 1, 8, 8, True, torch.float32],
            ["equiangular", (65, 128), "equiangular", (65, 128), 2, 16, 1, None, None, True, torch.float32],
            # downsampling tests with QK norm enabled, pscale=2
            ["equiangular", (64, 128), "equiangular", (32, 64), 2, 16, 1, None, None, True, torch.float32],
            ["equiangular", (65, 128), "equiangular", (33, 64), 2, 16, 1, None, None, True, torch.float32],
            # pscale=3 / pscale=4 downsampling with QK norm
            ["equiangular", (64, 96), "equiangular", (32, 32), 2, 16, 1, None, None, True, torch.float32],
            ["equiangular", (64, 128), "equiangular", (32, 32), 2, 16, 1, None, None, True, torch.float32],
            # lon-only / lat-only downsampling with QK norm
            ["equiangular", (64, 128), "equiangular", (64, 64), 2, 16, 1, None, None, True, torch.float32],
            ["equiangular", (64, 128), "equiangular", (32, 128), 2, 16, 1, None, None, True, torch.float32],
            # odd nlat_in -> even nlat_out with QK norm
            ["equiangular", (65, 128), "equiangular", (32, 64), 2, 16, 1, None, None, True, torch.float32],
            # legendre-gauss grid with QK norm
            ["legendre-gauss", (64, 128), "legendre-gauss", (64, 128), 2, 16, 1, None, None, True, torch.float32],
            ["legendre-gauss", (64, 128), "legendre-gauss", (32, 64), 2, 16, 1, None, None, True, torch.float32],
            # scalar (non-vectorized) path: per-head channels not divisible by 4
            # same-shape, CHOUT_AS_IN=True (nchans_in=nchans_out=10)
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 10, 1, None, None, False, torch.float32],
            # downsampling pscale=2, CHOUT_AS_IN=True
            ["equiangular", (64, 128), "equiangular", (32, 64), 2, 10, 1, None, None, False, torch.float32],
            # CHOUT_AS_IN=False, scalar path (nchans_in=5, nchans_out=3 per head)
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 4, 20, 12, False, torch.float32],
            # downsampling pscale=2, CHOUT_AS_IN=False, scalar path
            ["equiangular", (64, 128), "equiangular", (32, 64), 2, 16, 4, 20, 12, False, torch.float32],
            # BDIM_X dispatch coverage. The kernel picks BDIM_X = next_pow2(max(32, ceil(nchans/16))),
            # where FWD uses nchans_out and BWD pass1/pass2 use nchans_in. With heads=1 and default
            # k_channels/out_channels, all three dispatches land in the same bucket. Spatial dims are
            # kept small to bound memory; only one row per bucket since the same kernel template
            # is exercised regardless of grid/pscale.
            # BDIM_X=64   (per-head 513..1024)
            ["equiangular", (32, 64), "equiangular", (32, 64), 2, 576, 1, None, None, False, torch.float32],
            # BDIM_X=128  (per-head 1025..2048)
            ["equiangular", (32, 64), "equiangular", (32, 64), 2, 1280, 1, None, None, False, torch.float32],
            # BDIM_X=256  (per-head 2049..4096)
            ["equiangular", (16, 32), "equiangular", (16, 32), 2, 2560, 1, None, None, False, torch.float32],
            # BDIM_X=512  (per-head 4097..8192)
            ["equiangular", (16, 32), "equiangular", (16, 32), 2, 4608, 1, None, None, False, torch.float32],
            # top of the BDIM_X=512 bucket: the ring bwd pass-2 request,
            # ``sizeof(float) * (nchans_in + nchans_out)``, is 64 KiB here, above the default
            # 48 KiB, so this also covers the opt-in (launch_dyn_shmem). Per-head counts
            # past 8192 take the generic kernels, whose pass-1 request (32 B per channel)
            # exceeds every device's per-block maximum.
            ["equiangular", (8, 16), "equiangular", (8, 16), 2, 8192, 1, None, None, False, torch.float32],
            # upsampling tests (scatter ring kernels), pscale_out=2 (lat+lon)
            ["equiangular", (32, 64), "equiangular", (64, 128), 2, 16, 1, None, None, False, torch.float32],
            ["equiangular", (33, 64), "equiangular", (65, 128), 2, 16, 1, None, None, False, torch.float32],
            # pscale_out=3 upsampling (exercises pscale_out*wi kernel arithmetic)
            ["equiangular", (32, 32), "equiangular", (64, 96), 2, 16, 1, None, None, False, torch.float32],
            # pscale_out=4 upsampling
            ["equiangular", (32, 32), "equiangular", (64, 128), 2, 16, 1, None, None, False, torch.float32],
            # lon-only upsampling (same nlat; isolates the azimuth ring scatter, no lat halo)
            ["equiangular", (64, 64), "equiangular", (64, 128), 2, 16, 1, None, None, False, torch.float32],
            # odd nlat_in -> even nlat_out, pscale_out=2
            ["equiangular", (33, 64), "equiangular", (64, 128), 2, 16, 1, None, None, False, torch.float32],
            # legendre-gauss / mixed grid upsampling
            ["legendre-gauss", (32, 64), "legendre-gauss", (64, 128), 2, 16, 1, None, None, False, torch.float32],
            ["legendre-gauss", (32, 64), "equiangular", (64, 128), 2, 16, 1, None, None, False, torch.float32],
            # heads=4 with asymmetric channels (k=32, out=16; in=16), upsampling
            ["equiangular", (32, 64), "equiangular", (64, 128), 2, 16, 4, 32, 16, False, torch.float32],
            # upsampling with QK norm
            ["equiangular", (32, 64), "equiangular", (64, 128), 2, 16, 1, None, None, True, torch.float32],
            ["equiangular", (32, 64), "equiangular", (64, 128), 2, 16, 2, None, None, True, torch.float32],
            # upsampling AMP coverage (see the AMP note below)
            ["equiangular", (32, 64), "equiangular", (64, 128), 2, 16, 1, None, None, False, torch.float16],
            ["equiangular", (32, 64), "equiangular", (64, 128), 2, 16, 1, None, None, False, torch.bfloat16],
            # AMP coverage — one row per dtype on a downsample config. The bf16/fp16
            # rounding of the einsum operands and outputs dominates, see _ulps.
            ["equiangular", (64, 128), "equiangular", (32, 64), 2, 16, 1, None, None, False, torch.float16],
            ["equiangular", (64, 128), "equiangular", (32, 64), 2, 16, 1, None, None, False, torch.bfloat16],
            # same-resolution gather (self-attention) — the path the downsample rows skip
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 1, None, None, False, torch.float16],
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 1, None, None, False, torch.bfloat16],
            # multi-head same-resolution gather
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 2, None, None, False, torch.float16],
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 2, None, None, False, torch.bfloat16],
            # a trainable scale: its gradient is partial per rank like every other parameter's,
            # on the gather (same-shape, downsample) and scatter (upsample) paths, and with QK norm
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 2, None, None, False, torch.float32, True],
            ["equiangular", (64, 128), "equiangular", (32, 64), 2, 16, 1, None, None, False, torch.float32, True],
            ["equiangular", (32, 64), "equiangular", (64, 128), 2, 16, 1, None, None, False, torch.float32, True],
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 2, None, None, True, torch.float32, True],
            # HEALPix, on one side or both: polar bands of whole rings, a halo planned from the
            # neighbourhood, and the serial ragged backends, which run on CPU and CUDA
            ["healpix", (8,), "healpix", (8,), 2, 8, 1, None, None, False, torch.float32],
            ["healpix", (8,), "healpix", (8,), 2, 8, 2, None, None, False, torch.float64, False, False],  # the reference, in float64
            ["healpix", (8,), "healpix", (4,), 2, 8, 2, None, None, False, torch.float32],  # downsampling
            ["healpix", (4,), "healpix", (8,), 2, 8, 2, None, None, False, torch.float32],  # upsampling: still a gather on the ragged path
            ["healpix", (8,), "equiangular", (17, 32), 2, 8, 2, None, None, False, torch.float32],  # onto a regular grid, which reaches the poles
            ["legendre-gauss", (16, 32), "healpix", (8,), 2, 8, 2, None, None, False, torch.float64, False, False],  # from a regular grid
            ["healpix", (8,), "healpix", (8,), 2, 8, 1, None, None, False, torch.float32, False, True, 0.5],  # a wide cutoff: a halo of several rings
            ["healpix+radial", (8,), "healpix+radial", (8,), 2, 8, 2, None, None, False, torch.float32],  # a product grid, shells folded into the batch
            ["healpix", (8,), "healpix", (8,), 2, 16, 4, 32, 16, True, torch.float32, True],  # QK norm, asymmetric channels, a trainable scale
            ["healpix", (8,), "healpix", (4,), 2, 16, 1, None, None, False, torch.bfloat16],  # AMP
        ],
        skip_on_empty=True,
    )
    def test_distributed_neighborhood_attention(
        self,
        grid_in,
        in_shape,
        grid_out,
        out_shape,
        batch_size,
        in_channels,
        num_heads,
        k_channels,
        out_channels,
        use_qknorm,
        dtype,
        trainable_scale=False,
        optimized_kernel=True,
        theta_cutoff=None,
        verbose=True,
    ):
        if (*in_shape, *out_shape) in _SLOW_ATTN_SHAPES and not _run_slow_tests:
            self.skipTest("slow test; set TORCH_HARMONICS_RUN_SLOW_TESTS=1 to run")

        # Without the ring kernels these cases only check the refusal, which the small ones
        # already do; the large channel counts exist for the kernels' tiling, and building
        # their layers on every rank of an 8-rank CPU run exhausts the runner's memory.
        if in_channels > 64 and self.device.type not in _RING_GATHER_DEVICES & _RING_UPSAMPLE_DEVICES:
            self.skipTest("large-channel case needs the ring kernels; the refusal is covered by the small cases")

        set_seed(333)

        B, C = batch_size, in_channels
        gin, gout = _make_grid(grid_in, in_shape), _make_grid(grid_out, out_shape)
        self._skip_unless_decomposable(gin, gout)

        attn_args = dict(
            in_channels=C,
            grid_in=gin,
            grid_out=gout,
            num_heads=num_heads,
            bias=True,
            use_qknorm=use_qknorm,
            k_channels=k_channels,
            out_channels=out_channels,
            theta_cutoff=theta_cutoff,
            optimized_kernel=optimized_kernel,
        )

        # modules and inputs stay fp32 under autocast, see below; float64 is not autocast
        # but the layer's own precision, which only the ragged path accepts
        param_dtype = torch.float64 if dtype == torch.float64 else torch.float32

        # A trainable scale, one per layer, so the two backward passes do not accumulate into
        # the same .grad; it starts at the default value, so only its gradient is new
        def scale():
            if not trainable_scale:
                return {}
            per_head = (C if k_channels is None else k_channels) // num_heads
            return {"scale": torch.nn.Parameter(torch.tensor(per_head**-0.5))}

        # build serial and distributed modules with identical weights
        attn_serial = th.NeighborhoodAttentionS2(**attn_args, **scale()).to(self.device, param_dtype)
        attn_dist = thd.DistributedNeighborhoodAttentionS2(**attn_args, **scale()).to(self.device, param_dtype)
        if self._refuses_device(attn_dist):
            return

        with torch.no_grad():
            attn_dist.k_weights.copy_(attn_serial.k_weights)
            attn_dist.v_weights.copy_(attn_serial.v_weights)
            attn_dist.q_weights.copy_(attn_serial.q_weights)
            attn_dist.proj_weights.copy_(attn_serial.proj_weights)
            if attn_args["bias"]:
                attn_dist.v_bias.copy_(attn_serial.v_bias)
                attn_dist.q_bias.copy_(attn_serial.q_bias)
                attn_dist.proj_bias.copy_(attn_serial.proj_bias)
            if attn_args["bias"] and use_qknorm:
                attn_dist.k_bias.copy_(attn_serial.k_bias)
            if use_qknorm:
                attn_dist.q_norm_weights.copy_(attn_serial.q_norm_weights)
                attn_dist.k_norm_weights.copy_(attn_serial.k_norm_weights)

        # Helper: create inputs
        inp_full = {
            "k": torch.randn(B, C, *gin.shape, requires_grad=True, device=self.device, dtype=param_dtype),
            "v": torch.randn(B, C, *gin.shape, requires_grad=True, device=self.device, dtype=param_dtype),
            "q": torch.randn(B, C, *gout.shape, requires_grad=True, device=self.device, dtype=param_dtype),
        }
        # the shard each input is split by: queries live on the output grid
        shards = {"k": attn_dist.shard_in, "v": attn_dist.shard_in, "q": attn_dist.shard_out}

        # ---- serial forward ----
        # Modules + inputs stay fp32; maybe_autocast wraps fwd in autocast(dtype)
        # for fp16/bf16, no-op for fp32. Backward runs outside autocast, per the
        # standard PyTorch AMP pattern.
        with maybe_autocast(self.device.type, dtype):
            out_full = attn_serial(inp_full["q"], inp_full["k"], inp_full["v"])

        self._sync()

        # ---- serial backward ----
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)

        igrad_full = {}
        for inp in ["q", "k", "v"]:
            igrad_full[inp] = inp_full[inp].grad.clone()

        self._sync()

        # ---- distributed forward ----
        inp_local = {}
        for inp in ["q", "k", "v"]:
            inp_local[inp] = self._split_helper(inp_full[inp].detach(), shards[inp])
            inp_local[inp].requires_grad_(True)
        with maybe_autocast(self.device.type, dtype):
            out_local = attn_dist(inp_local["q"], inp_local["k"], inp_local["v"])

        self._sync()

        # ---- distributed backward ----
        ograd_local = self._split_helper(ograd_full, attn_dist.shard_out)
        out_local.backward(ograd_local)

        igrad_local = {}
        for inp in ["q", "k", "v"]:
            igrad_local[inp] = inp_local[inp].grad.clone()

        self._sync()

        # Print diagnostics from rank 0 only; assert the all-reduced verdict on every
        # rank so a failure on any rank fails the test consistently (see reduce_success).
        verbose = verbose and self.world_rank == 0

        # ---- compare forward ----
        out_gather = self._gather_helper(out_local, attn_dist.shard_out)
        ok = compare_tensors("forward output", out_gather, out_full, atol=0.0, rtol=0.0, ulps=_ulps(dtype), eps_dtype=dtype, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "forward output")

        # ---- compare backward ----
        for inp in ["q", "k", "v"]:
            igrad_gather = self._gather_helper(igrad_local[inp], shards[inp])
            ok = compare_tensors(f"input gradient {inp}", igrad_gather, igrad_full[inp], atol=0.0, rtol=0.0, ulps=_ulps(dtype), eps_dtype=dtype, verbose=verbose)
            self.assertTrue(reduce_success(ok, self.device), f"input gradient {inp}")

        # ---- compare parameter gradients ----
        # summed over the spatial ranks first, which is the caller's job (see
        # _allreduce_param_grad). A parameter gradient sums over every point, so the two
        # sides add the same terms in different orders: atol scales with the gradient's
        # range, see param_grad_ulps.
        pg_ulps = param_grad_ulps(B * max(math.prod(gin.shape), math.prod(gout.shape)), dtype)
        named_serial = dict(attn_serial.named_parameters())
        named_dist = dict(attn_dist.named_parameters())
        self.assertEqual(sorted(named_serial), sorted(named_dist), "the two layers have different parameters")
        self.assertEqual("scale" in named_dist, trainable_scale)
        for name, p_serial in named_serial.items():
            pgrad = self._allreduce_param_grad(named_dist[name].grad)
            ok = compare_tensors(f"parameter gradient {name}", pgrad, p_serial.grad, atol=0.0, rtol=0.0, verbose=verbose, ulps=pg_ulps, eps_dtype=dtype)
            self.assertTrue(reduce_success(ok, self.device), f"parameter gradient {name}")

    @parameterized.expand(
        [
            # (grid_in, in_shape, grid_out, out_shape, batch, in_chans, heads, k_chans, out_chans, frozen)
            # one same-shape config and one downsample config (pscale=2), each with each branch frozen in turn.
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 1, None, None, "k"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 1, None, None, "v"],
            ["equiangular", (64, 128), "equiangular", (64, 128), 2, 16, 1, None, None, "q"],
            ["equiangular", (64, 128), "equiangular", (32, 64), 2, 16, 1, None, None, "k"],
            ["equiangular", (64, 128), "equiangular", (32, 64), 2, 16, 1, None, None, "v"],
            ["equiangular", (64, 128), "equiangular", (32, 64), 2, 16, 1, None, None, "q"],
            # upsample config (pscale_out=2), each branch frozen in turn
            ["equiangular", (32, 64), "equiangular", (64, 128), 2, 16, 1, None, None, "k"],
            ["equiangular", (32, 64), "equiangular", (64, 128), 2, 16, 1, None, None, "v"],
            ["equiangular", (32, 64), "equiangular", (64, 128), 2, 16, 1, None, None, "q"],
            # HEALPix: the ragged kernels' backward, through the halo exchange's
            ["healpix", (8,), "healpix", (8,), 2, 16, 1, None, None, "k"],
            ["healpix", (8,), "healpix", (8,), 2, 16, 1, None, None, "v"],
            ["healpix", (8,), "healpix", (8,), 2, 16, 1, None, None, "q"],
        ],
        skip_on_empty=True,
    )
    def test_distributed_neighborhood_attention_selective_requires_grad(
        self,
        grid_in,
        in_shape,
        grid_out,
        out_shape,
        batch_size,
        in_channels,
        num_heads,
        k_channels,
        out_channels,
        frozen,
        verbose=True,
    ):
        """Freezes one of {k, v, q} at a time and verifies:
        - frozen input's `.grad` is None on both serial and distributed,
        - frozen projection's weight/bias `.grad` is None on both modules,
        - the remaining (unfrozen) input gradients match between serial and distributed.

        Both the raw module input AND the corresponding projection (weight + bias) are
        frozen, so the custom op's input tensor (kw / vw / qw) is genuinely a
        non-requires_grad leaf and `ctx.needs_input_grad[0:3]` reflects the frozen branch.
        This is what exercises the per-branch gates added to _RingNeighborhoodAttentionFn.backward.
        """
        set_seed(333)

        B, C = batch_size, in_channels
        gin, gout = _make_grid(grid_in, in_shape), _make_grid(grid_out, out_shape)
        self._skip_unless_decomposable(gin, gout)

        attn_args = dict(
            in_channels=C,
            grid_in=gin,
            grid_out=gout,
            num_heads=num_heads,
            bias=True,
            use_qknorm=False,
            k_channels=k_channels,
            out_channels=out_channels,
        )

        attn_serial = th.NeighborhoodAttentionS2(**attn_args).to(self.device)
        attn_dist = thd.DistributedNeighborhoodAttentionS2(**attn_args).to(self.device)
        if self._refuses_device(attn_dist):
            return

        with torch.no_grad():
            attn_dist.k_weights.copy_(attn_serial.k_weights)
            attn_dist.v_weights.copy_(attn_serial.v_weights)
            attn_dist.q_weights.copy_(attn_serial.q_weights)
            attn_dist.proj_weights.copy_(attn_serial.proj_weights)
            attn_dist.v_bias.copy_(attn_serial.v_bias)
            attn_dist.q_bias.copy_(attn_serial.q_bias)
            attn_dist.proj_bias.copy_(attn_serial.proj_bias)

        # Freeze the chosen branch's projection (weights + bias) on BOTH modules.
        for m in (attn_serial, attn_dist):
            getattr(m, f"{frozen}_weights").requires_grad_(False)
            bias = getattr(m, f"{frozen}_bias")
            if bias is not None:
                bias.requires_grad_(False)

        # Build full inputs; the frozen branch's input has requires_grad=False.
        inp_full = {}
        for name, shape in (("k", (B, C, *gin.shape)), ("v", (B, C, *gin.shape)), ("q", (B, C, *gout.shape))):
            t = torch.randn(*shape, device=self.device, dtype=torch.float32)
            t.requires_grad_(name != frozen)
            inp_full[name] = t

        # ---- serial forward / backward ----
        out_full = attn_serial(inp_full["q"], inp_full["k"], inp_full["v"])
        self._sync()
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = {n: (inp_full[n].grad.clone() if inp_full[n].grad is not None else None) for n in ("k", "v", "q")}
        self._sync()

        # ---- distributed forward / backward ----
        # the shard each input is split by: queries live on the output grid
        shards = {"k": attn_dist.shard_in, "v": attn_dist.shard_in, "q": attn_dist.shard_out}
        inp_local = {}
        for n in ("k", "v", "q"):
            inp_local[n] = self._split_helper(inp_full[n].detach(), shards[n])
            inp_local[n].requires_grad_(n != frozen)
        out_local = attn_dist(inp_local["q"], inp_local["k"], inp_local["v"])
        self._sync()

        ograd_local = self._split_helper(ograd_full, attn_dist.shard_out)
        out_local.backward(ograd_local)
        igrad_local = {n: (inp_local[n].grad.clone() if inp_local[n].grad is not None else None) for n in ("k", "v", "q")}
        self._sync()

        # Print diagnostics from rank 0 only; assert the all-reduced verdict on every
        # rank so a failure on any rank fails the test consistently (see reduce_success).
        verbose = verbose and self.world_rank == 0

        # ---- compare forward ----
        out_gather = self._gather_helper(out_local, attn_dist.shard_out)
        ok = compare_tensors("forward output", out_gather, out_full, atol=0.0, rtol=0.0, ulps=_ulps(torch.float32), verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "forward output")

        # ---- contract: frozen input grads must be None on both ----
        self.assertIsNone(igrad_full[frozen], f"serial: frozen input {frozen} must have None grad")
        self.assertIsNone(igrad_local[frozen], f"distributed: frozen input {frozen} must have None grad")

        # ---- contract: frozen projection's weight + bias grads must be None on both ----
        for m, label in ((attn_serial, "serial"), (attn_dist, "distributed")):
            w = getattr(m, f"{frozen}_weights")
            b = getattr(m, f"{frozen}_bias")
            self.assertIsNone(w.grad, f"{label}: {frozen}_weights frozen, grad must be None")
            if b is not None:
                self.assertIsNone(b.grad, f"{label}: {frozen}_bias frozen, grad must be None")

        # ---- unfrozen input grads must match between serial and distributed ----
        for n in ("k", "v", "q"):
            if n == frozen:
                continue
            self.assertIsNotNone(igrad_full[n], f"serial: unfrozen input {n} should have a grad")
            self.assertIsNotNone(igrad_local[n], f"distributed: unfrozen input {n} should have a grad")
            igrad_gather = self._gather_helper(igrad_local[n], shards[n])
            ok = compare_tensors(
                f"input grad {n} (frozen={frozen})",
                igrad_gather,
                igrad_full[n],
                atol=0.0,
                rtol=0.0,
                ulps=_ulps(torch.float32),
                verbose=verbose,
            )
            self.assertTrue(reduce_success(ok, self.device), f"input grad {n} (frozen={frozen})")

    def _check_backend_state(self, attn, backend):
        """
        The layer holds exactly the state of ``backend``, sized to this rank's block.

        The ring backends hold the regular arcs: rows are this rank's output latitudes when
        gathering, its halo-padded input latitudes when scattering. The ragged ones hold the
        serial ragged state of the band: arcs for this rank's output points, over ring tables
        cut to the halo-padded input band and rebased to its first point.
        """
        names = {
            "ring-gather": {"ring_weights", "psi_seg", "psi_seg_off"},
            "ring-upsample": {"ring_weights", "psi_seg", "psi_seg_off"},
            "ragged-halo-optimized": {"ring_weights", "psi_seg", "psi_seg_off", "psi_ring_base", "psi_ring_size"},
            "ragged-halo-reference": {"point_weights", "psi_col_idx", "psi_roff_idx"},
        }[backend]
        self.assertEqual(attn.backend.name, backend)
        self.assertEqual({name for name, _ in attn.named_buffers()}, names)
        self.assertEqual(set(attn._backend_state), names)

        if attn.ragged:
            npoints_band = attn.band_hi - attn.band_lo
            self.assertEqual(npoints_band, attn.npoints_in_local + attn.halo.before + attn.halo.after)
            if backend == "ragged-halo-optimized":
                self.assertEqual(attn.psi_seg_off.numel(), attn.npoints_out_local + 1)
                self.assertEqual(int(attn.psi_seg_off[-1]), attn.psi_seg.shape[0])
                self.assertEqual(attn.ring_weights.numel(), attn.band_ring_hi - attn.band_ring_lo)
                self.assertEqual(int(attn.psi_ring_base[0]), 0)
                self.assertEqual(int(attn.psi_ring_size.sum()), npoints_band)
            else:
                self.assertEqual(attn.psi_roff_idx.numel(), attn.npoints_out_local + 1)
                self.assertEqual(attn.point_weights.numel(), npoints_band)
                self.assertTrue(attn.psi_col_idx.numel() == 0 or int(attn.psi_col_idx.max()) < npoints_band)
        else:
            nrows = attn.nlat_in_local + 2 * attn.r_lat if attn.upsample else attn.nlat_out_local
            self.assertEqual(attn.psi_seg_off.numel(), nrows + 1)
            self.assertEqual(int(attn.psi_seg_off[-1]), attn.psi_seg.shape[0])
            self.assertEqual(attn.psi_seg.device, attn.ring_weights.device)

    @parameterized.expand(
        [
            # grid_in, in_shape, grid_out, out_shape, backend[, optimized_kernel]
            ["equiangular", (64, 128), "equiangular", (64, 128), "ring-gather"],  # self-attention
            ["equiangular", (64, 128), "equiangular", (32, 64), "ring-gather"],  # downsampling
            ["equiangular", (32, 64), "equiangular", (64, 128), "ring-upsample"],  # upsampling
            ["healpix", (8,), "healpix", (8,), "ragged-halo-optimized"],
            ["healpix", (8,), "healpix", (8,), "ragged-halo-reference", False],
            ["healpix", (8,), "equiangular", (17, 32), "ragged-halo-optimized"],  # a regular side takes the ragged path too
        ],
        skip_on_empty=True,
    )
    def test_backend_state(self, grid_in, in_shape, grid_out, out_shape, backend, optimized_kernel=True):
        """
        The layer selects the backend for its grids and direction and holds exactly that
        backend's state, sized to this rank's shard -- on construction and after a move.

        Built on CPU and moved, as a model normally is: construction must not refuse for
        want of a device, and the move reselects the backend and prepares its state on
        the new device, rather than carrying the old one along.
        """
        gin, gout = _make_grid(grid_in, in_shape), _make_grid(grid_out, out_shape)
        self._skip_unless_decomposable(gin, gout)
        attn = thd.DistributedNeighborhoodAttentionS2(grid_in=gin, grid_out=gout, in_channels=8, num_heads=2, bias=False, optimized_kernel=optimized_kernel)

        # a build without the compiled ragged kernels falls back to the reference, as the serial layer does
        if backend == "ragged-halo-optimized" and not attn.optimized_kernel:
            backend = "ragged-halo-reference"

        self._check_backend_state(attn, backend)

        # .to() moves the module in place and reselects the backend on the new device
        attn.to(self.device)
        self._check_backend_state(attn, backend)
        self.assertEqual(next(attn.buffers()).device.type, self.device.type)

    def test_float64_is_refused(self):
        """
        The ring kernels compute in float32 and the layer has no reference to hand float64
        to, so it refuses it, on any device and before anything else is checked.
        """
        attn = thd.DistributedNeighborhoodAttentionS2(
            grid_in=th.as_grid("equiangular", nlat=32, nlon=64),
            grid_out=th.as_grid("equiangular", nlat=32, nlon=64),
            in_channels=8,
            num_heads=1,
        ).to(self.device, torch.float64)

        x = torch.randn(1, 8, attn.nlat_in_local, attn.nlon_in_local, device=self.device, dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, "does not support float64"):
            attn(x)

    def test_wrong_shape_assertions(self):
        """Verify that forward raises RuntimeError on spatial-shape mismatches."""
        B, C = 2, 16
        in_shape = (64, 128)
        out_shape = (32, 64)

        attn = thd.DistributedNeighborhoodAttentionS2(
            grid_in=th.as_grid("equiangular", nlat=in_shape[0], nlon=in_shape[1]),
            grid_out=th.as_grid("equiangular", nlat=out_shape[0], nlon=out_shape[1]),
            in_channels=C,
            num_heads=1,
            bias=False,
        ).to(self.device)

        # Build correctly-shaped local tensors using the module's own local extents.
        q_local = torch.randn(B, C, attn.nlat_out_local, attn.nlon_out_local, device=self.device)
        k_local = torch.randn(B, C, attn.nlat_in_local, attn.nlon_in_local, device=self.device)

        # 1. Self-attention on an up/downsampling module: a single tensor cannot
        #    simultaneously satisfy in_shape (for k/v) and out_shape (for q).
        # the regex keeps a missing ring kernel (NotImplementedError is a RuntimeError)
        # from passing for a shape check
        with self.assertRaisesRegex(RuntimeError, "Expected"):
            attn(q_local)  # key defaults to query, but key must have in_shape

        # 2. q_shape == k_shape != v_shape: key carries out_shape instead of in_shape.
        with self.assertRaisesRegex(RuntimeError, "Expected"):
            attn(q_local, q_local, k_local)

        # 3. q_shape == v_shape != k_shape: value carries out_shape instead of in_shape.
        with self.assertRaisesRegex(RuntimeError, "Expected"):
            attn(q_local, k_local, q_local)

    @parameterized.expand(
        [
            # nlat_in, nlon_in, nlat_out, nlon_out
            [32, 64, 32, 64],
            [16, 32, 32, 64],
        ],
        skip_on_empty=True,
    )
    def test_double_backward_raises(self, nlat_in, nlon_in, nlat_out, nlon_out):
        """Under create_graph=True the ring backward refuses to run, rather than return gradients that cannot be differentiated."""
        set_seed(333)
        B, C = 1, 4
        attn = thd.DistributedNeighborhoodAttentionS2(
            in_channels=C,
            grid_in=th.as_grid("equiangular", nlat=nlat_in, nlon=nlon_in),
            grid_out=th.as_grid("equiangular", nlat=nlat_out, nlon=nlon_out),
            num_heads=1,
        ).to(self.device)
        if self._refuses_device(attn):
            return

        q = torch.randn(B, C, attn.nlat_out_local, attn.nlon_out_local, device=self.device, requires_grad=True)
        k = torch.randn(B, C, attn.nlat_in_local, attn.nlon_in_local, device=self.device, requires_grad=True)
        out = attn(q, k, k)

        # first order is unaffected
        torch.autograd.grad(torch.sum(out**2), (q, k), retain_graph=True)
        with self.assertRaisesRegex(RuntimeError, "does not support double backward"):
            torch.autograd.grad(torch.sum(out**2), (q, k), create_graph=True)

    # -----------------------------------------------------------------------
    # HEALPix-only behaviour: the halo plan and the decompositions it refuses. The
    # comparisons with the serial layer and the backend state are rows of the tests above.
    # -----------------------------------------------------------------------

    def test_healpix_halo_is_planned_per_side(self):
        """
        Every rank borrows exactly the whole rings its output points reach, which differ per
        side and per rank: the outermost ranks receive nothing beyond the poles, and the plans
        of two neighbours agree on what crosses between them.
        """
        grid = th.HealpixGrid(nside=8)
        self._skip_unless_decomposable(grid)
        attn = thd.DistributedNeighborhoodAttentionS2(grid, grid, in_channels=4)
        halo = attn.halo
        if self.hrank == 0:
            self.assertEqual(halo.recv_prev, 0)
        if self.hrank == self.grid_size_h - 1:
            self.assertEqual(halo.recv_next, 0)
        self.assertEqual((halo.pad_prev, halo.pad_next), (0, 0))

        # what this rank receives from its next neighbour is what that neighbour sends back
        sent = torch.tensor([halo.send_prev, halo.recv_next], device=self.device)
        gathered = [torch.zeros_like(sent) for _ in range(self.grid_size_h)]
        dist.all_gather(gathered, sent, group=self.h_group)
        for r in range(self.grid_size_h - 1):
            self.assertEqual(int(gathered[r][1]), int(gathered[r + 1][0]))

    def test_healpix_cutoff_too_wide_for_the_decomposition(self):
        """A neighbourhood reaching past the neighbouring bands is refused on every rank, at construction."""
        if self.grid_size_h < 3:
            self.skipTest("with fewer than three polar ranks every band is a neighbour")
        grid = th.HealpixGrid(nside=4)
        self._skip_unless_decomposable(grid)
        with self.assertRaisesRegex(ValueError, "too wide for this polar decomposition"):
            thd.DistributedNeighborhoodAttentionS2(grid, grid, in_channels=4, theta_cutoff=1.5)

    @parameterized.expand([["healpix", "healpix"], ["healpix", "equiangular"], ["equiangular", "healpix"]], skip_on_empty=True)
    def test_healpix_refuses_an_azimuthal_decomposition(self, grid_in, grid_out):
        """A HEALPix grid on either side cannot be cut along the azimuth; the layer says so at construction."""
        if self.grid_size_w == 1:
            self.skipTest("needs an azimuth group of more than one rank")
        gin = _make_grid(grid_in, (8,) if grid_in == "healpix" else (17, 32))
        gout = _make_grid(grid_out, (8,) if grid_out == "healpix" else (17, 32))
        with self.assertRaises(NotImplementedError):
            thd.DistributedNeighborhoodAttentionS2(gin, gout, in_channels=4)
