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


import unittest

import torch
from parameterized import parameterized
from testutils import accepted_leadings, rejected_shapes, setup_class_from_context, setup_module, shape_name, teardown_module

import torch_harmonics.distributed as thd
from torch_harmonics import LegendreGaussRadialGrid, SphericalProductGrid, as_grid

# shared state
_DIST_CTX = {}


def setUpModule():
    setup_module(_DIST_CTX)


def tearDownModule():
    teardown_module(_DIST_CTX)


_C = 8
_C_OUT = 12

# 33 x 66 splits unevenly over 2 and over 4 ranks in both directions, so neighbouring
# ranks hold different local extents and a layer checking any other rank's extent, or
# the global one, is caught. 8 or more rows per polar rank keep the default DISCO and
# attention cutoffs within one neighbour's halo.
_GRID = as_grid("legendre-gauss", nlat=33, nlon=66)
_GRID_OUT = as_grid("equiangular", nlat=35, nlon=66)


# The distributed counterpart of test_field_contract: each case builds a layer and reports
# the trailing shape of this rank's input, the output shape this rank should get back as
# a function of the input's leading axes, the input dtype, and the global trailing shape a
# caller might wrongly hand in. A free leading extent is filled with at least `free`.
#
# Each factory takes a radial grid, None for the angular grid alone; with one, the layer
# runs on the product grid, and every field gains the radial axis in front of the angular
# (or spectral) ones.
_RADIAL = LegendreGaussRadialGrid(nr=3, r_inner=1.0, r_outer=2.0)


def _on(grid, radial):
    return grid if radial is None else SphericalProductGrid(grid, radial)


def _r(radial):
    return () if radial is None else (radial.nr,)


def _sht(radial):
    layer = thd.DistributedRealSHT(_on(_GRID, radial))
    return layer, (*_r(radial), *layer.shard.shape), lambda lead: (*lead, *_r(radial), layer.lmax_local, layer.mmax_local), torch.float32, (*_r(radial), *_GRID.shape)


def _isht(radial):
    layer = thd.DistributedInverseRealSHT(_on(_GRID, radial))
    return layer, (*_r(radial), layer.lmax_local, layer.mmax_local), lambda lead: (*lead, *_r(radial), *layer.shard.shape), torch.complex64, (*_r(radial), layer.lmax, layer.mmax)


def _vsht(radial):
    layer = thd.DistributedRealVectorSHT(_on(_GRID, radial))
    # the vector transform keeps only the per-rank list of degree counts
    lmax_local = layer.l_shapes[thd.polar_group_rank()]
    return layer, (*_r(radial), *layer.shard.shape), lambda lead: (*lead, *_r(radial), lmax_local, layer.mmax_local), torch.float32, (*_r(radial), *_GRID.shape)


def _ivsht(radial):
    layer = thd.DistributedInverseRealVectorSHT(_on(_GRID, radial))
    return layer, (*_r(radial), layer.lmax_local, layer.mmax_local), lambda lead: (*lead, *_r(radial), *layer.shard.shape), torch.complex64, (*_r(radial), layer.lmax, layer.mmax)


def _quadrature(radial):
    layer = thd.DistributedQuadratureS2(_on(_GRID, radial))
    return layer, (*_r(radial), *layer.shard.shape), lambda lead: (*lead, *_r(radial)), torch.float32, (*_r(radial), *_GRID.shape)


def _quadrature_volume(radial):
    layer = thd.DistributedQuadratureVolume(_on(_GRID, radial))
    return layer, (*_r(radial), *layer.shard.shape), lambda lead: lead, torch.float32, (*_r(radial), *_GRID.shape)


def _resample(radial):
    layer = thd.DistributedResampleS2(_on(_GRID, radial), _on(_GRID_OUT, radial))
    return layer, (*_r(radial), *layer.shard_in.shape), lambda lead: (*lead, *_r(radial), *layer.shard_out.shape), torch.float32, (*_r(radial), *_GRID.shape)


def _spectral_conv(radial):
    layer = thd.DistributedSpectralConvS2(_on(_GRID, radial), _on(_GRID, radial), in_channels=_C, out_channels=_C_OUT)
    return (
        layer,
        (*_r(radial), *layer.sht.shard.shape),
        lambda lead: (lead[0], _C_OUT, *_r(radial), *layer.isht.shard.shape),
        torch.float32,
        (*_r(radial), *_GRID.shape),
    )


def _disco(radial):
    layer = thd.DistributedDiscreteContinuousConvS2(_on(_GRID, radial), _on(_GRID, radial), in_channels=_C, out_channels=_C_OUT, kernel_shape=3)
    return layer, (*_r(radial), *layer.shard_in.shape), lambda lead: (lead[0], _C_OUT, *_r(radial), *layer.shard_out.shape), torch.float32, (*_r(radial), *_GRID.shape)


def _disco_transpose(radial):
    layer = thd.DistributedDiscreteContinuousConvTransposeS2(_on(_GRID, radial), _on(_GRID, radial), in_channels=_C, out_channels=_C_OUT, kernel_shape=3)
    return layer, (*_r(radial), *layer.shard_in.shape), lambda lead: (lead[0], _C_OUT, *_r(radial), *layer.shard_out.shape), torch.float32, (*_r(radial), *_GRID.shape)


def _attention(radial):
    layer = thd.DistributedNeighborhoodAttentionS2(_on(_GRID, radial), _on(_GRID, radial), in_channels=_C, out_channels=_C_OUT)
    return layer, (*_r(radial), *layer.shard_in.shape), lambda lead: (lead[0], _C_OUT, *_r(radial), *layer.shard_out.shape), torch.float32, (*_r(radial), *_GRID.shape)


def _resample_free():
    # the resampling transposes split the channel axis without padding, so it needs at
    # least one channel per rank of either group
    return max(thd.polar_group_size(), thd.azimuth_group_size())


# name, factory, leading contract, smallest free extent (int, or a callable evaluated once
# the process groups exist), whether running it needs CUDA
_CASES = [
    # the SHTs pad their leading axis, so a single channel works on any rank grid
    ["sht", _sht, (..., None), 1, False],
    ["inverse_sht", _isht, (..., None), 1, False],
    ["vector_sht", _vsht, (..., None, 2), 1, False],
    ["inverse_vector_sht", _ivsht, (..., None, 2), 1, False],
    ["quadrature", _quadrature, (...,), 1, False],
    ["resample", _resample, (..., None), _resample_free, False],
    ["spectral_conv", _spectral_conv, (None, _C), 1, False],
    ["disco", _disco, (None, _C), 1, False],
    ["disco_transpose", _disco_transpose, (None, _C), 1, False],
    # the ring kernels exist for CUDA only; the shape checks run before them on any device
    ["neighborhood_attention", _attention, (None, _C), 1, True],
]

# name, factory, radial grid, leading contract, free extent, needs CUDA: every case on the
# angular grid alone and again on its product with a radial grid
_CASES = [[name, factory, None, *rest] for name, factory, *rest in _CASES] + [[f"{name}_product", factory, _RADIAL, *rest] for name, factory, *rest in _CASES]
# and the layers that exist only on a product grid
_CASES += [["quadrature_volume", _quadrature_volume, _RADIAL, (...,), 1, False]]


def _free(free):
    return free() if callable(free) else free


class TestDistributedFieldContract(unittest.TestCase):
    """Every distributed layer checks its input against this rank's block of the grid."""

    @classmethod
    def setUpClass(cls):
        setup_class_from_context(cls, _DIST_CTX)

    @parameterized.expand(_CASES)
    def test_shapes(self, name, factory, radial, spec, free, needs_cuda):
        """Every shape the contract admits runs, from the minimum up, and gives this rank its share of the output."""
        if needs_cuda and self.device.type != "cuda":
            self.skipTest(f"{name} runs on CUDA only")
        layer, trailing, out_shape, dtype, _ = factory(radial)
        layer = layer.to(self.device)
        # every rank feeds the same leading shapes, so the collectives stay matched
        for leading in accepted_leadings(spec, _free(free)):
            with self.subTest(leading=shape_name(leading)):
                out = layer(torch.randn((*leading, *trailing), device=self.device, dtype=dtype))
                self.assertEqual(tuple(out.shape), out_shape(leading))

    @parameterized.expand(_CASES)
    def test_rejects_every_violation(self, name, factory, radial, spec, free, needs_cuda):
        """
        Each violation is rejected before any collective runs.

        Every rank breaks the contract against its own block, so every rank raises and
        none is left waiting in a collective for the others.
        """
        layer, trailing, _, dtype, global_trailing = factory(radial)
        layer = layer.to(self.device)
        leading = accepted_leadings(spec, _free(free))[-1]
        bad = rejected_shapes(spec, leading, trailing)
        # the global grid in place of this rank's block; on a 1 x 1 rank grid they coincide
        if tuple(global_trailing) != tuple(trailing):
            bad.append(("global trailing shape", (*leading, *global_trailing)))
        for how, shape in bad:
            with self.subTest(how=how, shape=shape):
                with self.assertRaisesRegex(RuntimeError, r"Expected .* of shape"):
                    layer(torch.randn(shape, device=self.device, dtype=dtype))

    @parameterized.expand([["angular", None], ["product", _RADIAL]])
    def test_attention_rejects_a_shared_key_value_batch(self, _, radial):
        """Like the serial layers, a key/value batch of one is refused, on every rank and before any collective."""
        layer, trailing, _, dtype, _ = _attention(radial)
        layer = layer.to(self.device)
        query = torch.randn(2, _C, *trailing, device=self.device, dtype=dtype)
        shared = torch.randn(1, _C, *trailing, device=self.device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "query's batch size 2, got 1"):
            layer(query, shared, shared)

    def test_resample_rejects_too_few_channels(self):
        """The resampling transposes do not pad, so fewer channels than ranks is an error, not a hang."""
        need = _resample_free()
        if need == 1:
            self.skipTest("a single rank in each group splits nothing")
        layer = thd.DistributedResampleS2(_GRID, _GRID_OUT).to(self.device)
        x = torch.randn(2, need - 1, *layer.shard_in.shape, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "channels"):
            layer(x)

    @parameterized.expand([["forward", thd.DistributedDiscreteContinuousConvS2, "in_channels"], ["transpose", thd.DistributedDiscreteContinuousConvTransposeS2, "out_channels"]])
    def test_disco_rejects_channels_the_azimuth_split_cannot_share(self, _, cls, name):
        """
        The azimuth a2a gives every azimuth rank a share of the channels and does not pad,
        so a layer with fewer channels than azimuth ranks is refused at construction.
        """
        azimuth = thd.azimuth_group_size()
        if azimuth == 1:
            self.skipTest("a single azimuth rank splits nothing")
        channels = {"in_channels": _C, "out_channels": _C_OUT}
        channels[name] = azimuth - 1
        with self.assertRaisesRegex(ValueError, name):
            cls(_GRID, _GRID, kernel_shape=3, **channels)
        # exactly one channel per rank is enough, even though the split is then as thin as it gets
        channels[name] = azimuth
        cls(_GRID, _GRID, kernel_shape=3, **channels)


if __name__ == "__main__":
    unittest.main()
