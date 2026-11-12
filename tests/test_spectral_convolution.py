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

import subprocess
import sys
import tempfile
import textwrap
import unittest
import unittest.mock

import torch
from parameterized import parameterized, parameterized_class
from testutils import compare_tensors, requires_torch_compile, set_seed

from torch_harmonics import HealpixGrid, as_grid
from torch_harmonics.spectral_convolution import SpectralConvS2
from torch_harmonics.utils import compile_if_supported, torch_compile_supported

# Error budget in ULPs of the largest reference entry (see testutils.range_atol): the layer
# contracts over channels and spectral modes in float32, and the checks below only
# reassociate that contraction (measured: up to 2).
_CONTRACTION_ULPS = 16


def _test_grid(name):
    """A grid of the tests above: HEALPix, or a regular grid with nlon a multiple of 4."""
    return HealpixGrid(nside=8) if name == "healpix" else as_grid(name, nlat=32, nlon=64)


_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device",), _devices)
class TestSpectralConvS2(unittest.TestCase):

    # -----------------------------------------------------------------------
    # Test 1: Zonal rotation equivariance
    #
    # A cyclic shift of the input along the longitude axis (dim=-1) by k steps
    # must produce an identically shifted output.  This holds because:
    #   - equiangular grids have uniform longitude spacing, so a cyclic shift
    #     is a rotation by an exact grid multiple
    #   - the forward SHT decomposes via DFT in longitude, so a shift by k
    #     introduces a phase factor exp(i*m*2π*k/nlon) on each order m
    #   - SpectralConvS2 weights depend only on degree l, not on order m, so
    #     the phase factor passes through the contraction unchanged
    #   - the inverse SHT maps the phase-shifted coefficients back to a
    #     cyclically shifted spatial field
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, in_channels, out_channels, num_groups, lon_shift
            [32, 64, 4, 4, 1, 8],
            [32, 64, 4, 4, 2, 16],
            [32, 64, 1, 1, 1, 7],
            [16, 32, 8, 4, 2, 3],
        ],
        skip_on_empty=True,
    )
    def test_zonal_rotation_equivariance(self, nlat, nlon, in_channels, out_channels, num_groups, lon_shift, verbose=False):
        """Conv(Roll(x, k)) ≈ Roll(Conv(x), k) for any longitude shift k."""

        # set seed
        set_seed(333)

        conv = SpectralConvS2(
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
        ).to(self.device)
        conv.eval()

        x = torch.randn(2, in_channels, nlat, nlon, device=self.device)

        with torch.no_grad():
            y = conv(x)
            x_shifted = torch.roll(x, lon_shift, dims=-1)
            y_of_shifted = conv(x_shifted)
            shift_of_y = torch.roll(y, lon_shift, dims=-1)

        self.assertTrue(
            compare_tensors(
                "zonal rotation equivariance",
                y_of_shifted,
                shift_of_y,
                atol=0.0,
                rtol=0.0,
                ulps=_CONTRACTION_ULPS,
                verbose=verbose,
            )
        )

    # -----------------------------------------------------------------------
    # HEALPix: quarter-turn equivariance
    #
    # A HEALPix grid is invariant under a quarter turn about the polar axis: every ring
    # of n_r pixels maps onto itself, shifted by n_r / 4. The SHT of such a ring turns
    # the shift into the phase exp(-i m pi / 2) for the order m it was sampled at, and for
    # the order it aliases onto, since n_r is a multiple of 4. The weights do not depend
    # on m, so the layer commutes with the turn, on a HEALPix grid on either side or both.
    # A regular grid is invariant too whenever nlon is a multiple of 4. (A spectral bias is
    # not covered: it depends on m.)
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # grid_in, grid_out
            ["healpix", "healpix"],
            ["equiangular", "healpix"],
            ["healpix", "equiangular"],
            ["legendre-gauss", "healpix"],
            ["healpix", "legendre-gauss"],
            ["lobatto", "healpix"],
            ["healpix", "lobatto"],
            ["equiangular", "lobatto"],
        ],
        skip_on_empty=True,
    )
    def test_healpix_quarter_turn_equivariance(self, grid_in, grid_out, verbose=False):
        """Conv(Turn(x)) == Turn(Conv(x)) for a quarter turn about the polar axis."""

        set_seed(333)

        grid_in, grid_out = _test_grid(grid_in), _test_grid(grid_out)

        def turn(x, grid):
            if isinstance(grid, HealpixGrid):
                rings = x.split(grid.nlon_per_lat.tolist(), dim=-1)
                return torch.cat([torch.roll(r, r.shape[-1] // 4, dims=-1) for r in rings], dim=-1)
            return torch.roll(x, grid.nlon // 4, dims=-1)

        conv = SpectralConvS2(grid_in, grid_out, in_channels=4, out_channels=6, num_groups=2).to(self.device)
        conv.eval()

        x = torch.randn(2, 4, *grid_in.shape, device=self.device)

        with torch.no_grad():
            y_of_turned = conv(turn(x, grid_in))
            turn_of_y = turn(conv(x), grid_out)

        self.assertTrue(
            compare_tensors(
                "healpix quarter-turn equivariance",
                y_of_turned,
                turn_of_y,
                atol=0.0,
                rtol=0.0,
                ulps=_CONTRACTION_ULPS,
                verbose=verbose,
            )
        )

    # -----------------------------------------------------------------------
    # Identity weights reduce the layer to its transforms
    # -----------------------------------------------------------------------
    @parameterized.expand([["healpix"], ["equiangular"], ["legendre-gauss"], ["lobatto"]], skip_on_empty=True)
    def test_identity_weights_reduce_to_sht_roundtrip(self, grid, verbose=False):
        """With the identity as weights the layer is synthesis after analysis."""

        set_seed(333)

        grid = _test_grid(grid)
        conv = SpectralConvS2(grid, grid, in_channels=3, out_channels=3).to(self.device)
        conv.eval()
        with torch.no_grad():
            conv.weight.copy_(torch.eye(3, dtype=conv.weight.dtype, device=self.device).reshape(1, 3, 3, 1).expand_as(conv.weight))

        x = torch.randn(2, 3, *grid.shape, device=self.device)

        with torch.no_grad():
            y = conv(x)
            expected = conv.isht(conv.sht(x))

        self.assertTrue(compare_tensors("identity weights", y, expected, atol=0.0, rtol=0.0, ulps=_CONTRACTION_ULPS, verbose=verbose))

    # -----------------------------------------------------------------------
    # Test 4: Grouped convolution consistency
    #
    # A SpectralConvS2 with num_groups=G must produce the same result as
    # running G independent SpectralConvS2 layers (each with num_groups=1)
    # on the corresponding channel slices, using the same weight values.
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, in_channels, out_channels, num_groups
            [32, 64, 4, 4, 2],
            [32, 64, 4, 4, 4],  # depthwise (num_groups == in_channels)
            [16, 32, 6, 6, 3],
        ],
        skip_on_empty=True,
    )
    def test_grouped_convolution_consistency(self, nlat, nlon, in_channels, out_channels, num_groups, verbose=False):
        """num_groups > 1 equals running separate num_groups=1 convs on channel subsets."""

        # set seed
        set_seed(333)

        conv_grouped = SpectralConvS2(
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
        ).to(self.device)
        conv_grouped.eval()

        ic_per_group = in_channels // num_groups
        oc_per_group = out_channels // num_groups

        # Build one num_groups=1 conv per group and copy the matching weight slice
        group_convs = []
        for g in range(num_groups):
            gc = SpectralConvS2(
                as_grid("equiangular", nlat=nlat, nlon=nlon),
                as_grid("equiangular", nlat=nlat, nlon=nlon),
                in_channels=ic_per_group,
                out_channels=oc_per_group,
                num_groups=1,
            ).to(self.device)
            # weight shape: [num_groups, ic_per_group, oc_per_group, lmax]
            gc.weight.data = conv_grouped.weight[g : g + 1].clone()
            gc.eval()
            group_convs.append(gc)

        x = torch.randn(2, in_channels, nlat, nlon, device=self.device)

        with torch.no_grad():
            y_grouped = conv_grouped(x)
            y_manual = torch.cat(
                [gc(x[:, g * ic_per_group : (g + 1) * ic_per_group]) for g, gc in enumerate(group_convs)],
                dim=1,
            )

        self.assertTrue(
            compare_tensors(
                "grouped convolution consistency",
                y_grouped,
                y_manual,
                atol=0.0,
                rtol=0.0,
                ulps=_CONTRACTION_ULPS,
                verbose=verbose,
            )
        )

    # -----------------------------------------------------------------------
    # Test 6: Dtype preservation
    #
    # The module casts inputs to float32 internally for the SHT, but must
    # restore the original dtype on output via the final .to(dtype=dtype) call.
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, in_channels, out_channels, dtype
            [32, 64, 4, 4, torch.bfloat16],
            [32, 64, 4, 4, torch.float16],
        ],
        skip_on_empty=True,
    )
    def test_dtype_preservation(self, nlat, nlon, in_channels, out_channels, dtype, verbose=False):
        """Output dtype must match input dtype after the internal float32 cast."""

        # set seed
        set_seed(333)

        conv = SpectralConvS2(
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            in_channels=in_channels,
            out_channels=out_channels,
        ).to(self.device)
        conv.eval()

        x = torch.randn(2, in_channels, nlat, nlon, device=self.device, dtype=dtype)
        with torch.no_grad():
            y = conv(x)

        self.assertEqual(
            y.dtype,
            dtype,
            msg=f"Expected output dtype {dtype}, got {y.dtype}",
        )

    # -----------------------------------------------------------------------
    # Test 7: Batch independence
    #
    # Stacking two identical inputs must yield identical outputs up to float32
    # non-determinism from @torch.compile.  The compiled _contract_lwise einsum
    # parallelises the reduction over threads; different batch positions can be
    # assigned to different threads, so partial sums accumulate in a different
    # order.  Differences are about one ULP — not a bug, just float32
    # non-associativity. _CONTRACTION_ULPS stays safely above this noise floor while
    # still catching any real cross-sample mixing.
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, in_channels, out_channels, num_groups
            [32, 64, 4, 4, 1],
            [32, 64, 4, 4, 2],
        ],
        skip_on_empty=True,
    )
    def test_batch_independence(self, nlat, nlon, in_channels, out_channels, num_groups, verbose=True):
        """Two identical samples in a batch must produce bitwise-identical outputs."""

        # set seed
        set_seed(333)

        conv = SpectralConvS2(
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
        ).to(self.device)
        conv.eval()

        x_single = torch.randn(1, in_channels, nlat, nlon, device=self.device)
        # two identical samples
        x_batch = x_single.expand(2, -1, -1, -1).contiguous()

        with torch.no_grad():
            y_single = conv(x_single)
            y_batch = conv(x_batch)

        # the two batched outputs must be identical to each other …
        self.assertTrue(compare_tensors("batch independence [0 vs 1]", y_batch[0], y_batch[1], atol=0.0, rtol=0.0, ulps=_CONTRACTION_ULPS, verbose=verbose))
        # … and identical to running the single sample alone
        self.assertTrue(compare_tensors("batch independence [single vs batched]", y_single[0], y_batch[0], atol=0.0, rtol=0.0, ulps=_CONTRACTION_ULPS, verbose=verbose))

    # -----------------------------------------------------------------------
    # Test 8: Zero weights → zero output
    #
    # With all weights set to zero the output must be exactly zero for any
    # input (the bias-free path is purely linear in the weights).
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, in_channels, out_channels, num_groups
            [32, 64, 4, 4, 1],
            [32, 64, 4, 4, 2],
            [16, 32, 1, 1, 1],
        ],
        skip_on_empty=True,
    )
    def test_zero_weights_zero_output(self, nlat, nlon, in_channels, out_channels, num_groups, verbose=False):
        """Zeroing all weights must produce an exactly-zero output."""

        # set seed
        set_seed(333)

        conv = SpectralConvS2(
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
        ).to(self.device)
        conv.eval()

        conv.weight.data.zero_()

        x = torch.randn(2, in_channels, nlat, nlon, device=self.device)
        with torch.no_grad():
            y = conv(x)

        self.assertTrue(
            compare_tensors(
                "zero weights zero output",
                y,
                torch.zeros_like(y),
                atol=0.0,
                rtol=0.0,
                verbose=verbose,
            )
        )

    # -----------------------------------------------------------------------
    # Test 8b: Spectral truncation (lmax, mmax, lmmax)
    #
    # The layer must act as the masked spectral operator
    #   v_hat[l, m] = mask[l, m] * sum_i K[l] (u_hat[l, m] + I * b[l, m])
    # with mask[l, m] = (m < mmax) & (m <= l < lmax) & (l - m < lmmax). The reference
    # does not use the layer's truncated transforms: it analyses with an SHT at a full
    # bound lfull >= lmax, applies the mask and the zero-padded weight by hand, and
    # synthesises with the full inverse SHT, all in float64. Agreement in space also
    # pins the output spectrum to the mask, which is checked explicitly on the
    # exact (Gauss) output grid.
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # grid_in, grid_out, lfull, lmax, mmax, lmmax, bias
            [("legendre-gauss", 32, 64), ("legendre-gauss", 32, 64), 32, 20, None, None, False],  # triangular, only lmax
            [("legendre-gauss", 32, 64), ("legendre-gauss", 32, 64), 32, None, 12, None, False],  # triangular, only mmax
            [("legendre-gauss", 32, 64), ("legendre-gauss", 32, 64), 32, 24, 12, None, False],  # rectangular
            [("legendre-gauss", 32, 64), ("legendre-gauss", 32, 64), 32, 24, 12, 12, True],  # rhomboidal
            [("legendre-gauss", 32, 64), ("legendre-gauss", 32, 64), 32, 32, 16, 10, True],  # lmmax clipped by lmax
            [("legendre-gauss", 32, 64), ("legendre-gauss", 32, 64), 32, None, None, 8, True],  # default bounds + lmmax
            [("equiangular", 33, 64), ("legendre-gauss", 24, 48), 17, 16, 8, 6, True],  # different grids
            [("legendre-gauss", 24, 48), ("equiangular", 41, 80), 24, 24, 10, None, False],  # upsampling
        ],
        skip_on_empty=True,
    )
    def test_spectral_truncation(self, grid_in, grid_out, lfull, lmax, mmax, lmmax, bias, verbose=False):
        """The truncated layer equals the full transforms with the (lmax, mmax, lmmax) mask applied by hand."""
        from torch_harmonics import InverseRealSHT, RealSHT
        from torch_harmonics.integration import QuadratureS2

        set_seed(333)

        in_channels, out_channels, num_groups = 4, 6, 2
        gin = as_grid(grid_in[0], nlat=grid_in[1], nlon=grid_in[2])
        gout = as_grid(grid_out[0], nlat=grid_out[1], nlon=grid_out[2])

        conv = SpectralConvS2(gin, gout, in_channels, out_channels, num_groups=num_groups, bias=bias, lmax=lmax, mmax=mmax, lmmax=lmmax).to(self.device)
        conv.eval()

        # resolved bounds: an omitted bound follows the other one, both omitted is the triangular default
        exp_lmax = lmax if lmax is not None else (mmax if mmax is not None else lfull)
        exp_mmax = mmax if mmax is not None else exp_lmax
        self.assertEqual((conv.lmax, conv.mmax, conv.lmmax), (exp_lmax, exp_mmax, lmmax))
        self.assertEqual(conv.sht.grid_out, conv.isht.grid_in)
        self.assertEqual(conv.weight.shape[-1], exp_lmax)

        if bias:
            with torch.no_grad():
                conv.spectral_bias.copy_(torch.randn_like(conv.spectral_bias))

        # mask over the full (lfull, lfull) coefficient array
        l = torch.arange(lfull, device=self.device).reshape(-1, 1)
        m = torch.arange(lfull, device=self.device).reshape(1, -1)
        mask = (m < exp_mmax) & (m <= l) & (l < exp_lmax)
        if lmmax is not None:
            mask = mask & (l - m < lmmax)

        x = torch.randn(2, in_channels, *gin.shape, device=self.device)
        with torch.no_grad():
            y = conv(x)

            # float64 reference from the untruncated transforms
            sht_full = RealSHT(gin, lmax=lfull, mmax=lfull).to(self.device).double()
            isht_full = InverseRealSHT(gout, lmax=lfull, mmax=lfull).to(self.device).double()
            xd = x.double()
            u = sht_full(xd)
            if bias:
                integral = QuadratureS2(gin, normalize=False).to(self.device).double()(xd)
                b = torch.zeros(1, in_channels, lfull, lfull, dtype=torch.complex128, device=self.device)
                b[..., :exp_lmax, :exp_mmax] = conv.spectral_bias.to(torch.complex128)
                u = u + integral.reshape(*integral.shape, 1, 1) * b
            u = u * mask
            w = torch.zeros(*conv.weight.shape[:-1], lfull, dtype=torch.complex128, device=self.device)
            w[..., :exp_lmax] = conv.weight.to(torch.complex128)
            u = u.reshape(u.shape[0], num_groups, in_channels // num_groups, lfull, lfull)
            v = torch.einsum("bgilm,giol->bgolm", u, w).reshape(u.shape[0], out_channels, lfull, lfull)
            y_ref = isht_full(v)

        self.assertTrue(compare_tensors("truncated spectral conv output", y, y_ref.float(), atol=1e-5, rtol=0.0, ulps=64, verbose=verbose))

        # the output spectrum lives on the mask. Only checkable where analysis is exact on the output grid
        if gout.grid_type == "legendre-gauss":
            with torch.no_grad():
                spec = RealSHT(gout, lmax=lfull, mmax=lfull).to(self.device).double()(y.double())
            outside = spec.abs() * (~mask)
            self.assertTrue(compare_tensors("output spectrum outside the support", outside, torch.zeros_like(outside), atol=1e-5, rtol=0.0, verbose=verbose))

    @parameterized.expand(
        [
            # grid, nlat, nlon, lmax, mmax, lmmax
            ["legendre-gauss", 32, 64, 40, None, None],  # lmax alone sets mmax above max_azimuthal_order
            ["legendre-gauss", 32, 64, 16, 20, None],  # mmax > lmax
            ["legendre-gauss", 32, 64, 16, 8, 0],  # lmmax must be positive
            ["legendre-gauss", 32, 64, 0, None, None],  # no degrees
            ["legendre-gauss", 32, 64, 16, 0, None],  # no orders
        ],
        skip_on_empty=True,
    )
    def test_spectral_truncation_rejects_invalid_bounds(self, grid, nlat, nlon, lmax, mmax, lmmax):
        g = as_grid(grid, nlat=nlat, nlon=nlon)
        with self.assertRaises(ValueError):
            SpectralConvS2(g, g, 2, 2, lmax=lmax, mmax=mmax, lmmax=lmmax)

    # -----------------------------------------------------------------------
    # Test 9: Spectral bias
    #
    # With bias=True the layer computes:
    #   x_spectral += integral(x_spatial) * spectral_bias
    # before the weight contraction, where integral is the quadrature-weighted
    # spatial sum (a scalar per channel per batch element).
    #
    # Three properties to verify:
    #  a) zero spectral_bias (the initialised state) is a no-op: the output
    #     matches a bias=False layer with identical weights.
    #  b) setting spectral_bias to a nonzero value changes the output.
    #  c) gradients flow back through spectral_bias after a backward pass.
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, channels, num_groups
            [32, 64, 4, 1],
            [32, 64, 4, 2],
        ],
        skip_on_empty=True,
    )
    def test_spectral_bias_zero_equals_no_bias(self, nlat, nlon, channels, num_groups, verbose=False):
        """Zero spectral_bias must produce the same output as bias=False."""
        set_seed(333)

        conv_no_bias = SpectralConvS2(
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            bias=False,
        ).to(self.device)

        set_seed(333)
        conv_bias = SpectralConvS2(
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            bias=True,
        ).to(self.device)
        # spectral_bias is zero-initialised; weights must match
        conv_bias.weight.data.copy_(conv_no_bias.weight.data)
        # confirm bias is zero
        self.assertTrue(conv_bias.spectral_bias.data.abs().max().item() == 0.0)

        x = torch.randn(2, channels, nlat, nlon, device=self.device)
        with torch.no_grad():
            y_no_bias = conv_no_bias(x)
            y_bias = conv_bias(x)

        self.assertTrue(compare_tensors("zero bias equals no bias", y_bias, y_no_bias, atol=0.0, rtol=0.0, verbose=verbose))

    @parameterized.expand(
        [
            # nlat, nlon, channels, num_groups
            [32, 64, 4, 1],
            [32, 64, 4, 2],
        ],
        skip_on_empty=True,
    )
    def test_spectral_bias_nonzero_changes_output(self, nlat, nlon, channels, num_groups, verbose=False):
        """A nonzero spectral_bias must change the output relative to zero bias."""
        set_seed(333)

        conv = SpectralConvS2(
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            bias=True,
        ).to(self.device)

        x = torch.randn(2, channels, nlat, nlon, device=self.device)

        with torch.no_grad():
            y_zero_bias = conv(x)
            # set a clearly nonzero bias and re-run
            conv.spectral_bias.data.fill_(1.0)
            y_nonzero_bias = conv(x)

        self.assertFalse(
            torch.allclose(y_zero_bias, y_nonzero_bias, atol=1e-6),
            msg="Expected output to change when spectral_bias is nonzero",
        )

    @parameterized.expand(
        [
            # nlat, nlon, channels, num_groups
            [32, 64, 4, 1],
            [32, 64, 4, 2],
        ],
        skip_on_empty=True,
    )
    def test_spectral_bias_gradient(self, nlat, nlon, channels, num_groups, verbose=False):
        """Gradients must flow back to spectral_bias after a backward pass."""
        set_seed(333)

        conv = SpectralConvS2(
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            bias=True,
        ).to(self.device)

        with torch.no_grad():
            # give the bias a nonzero value so it has an effect on the output
            conv.spectral_bias.data.fill_(0.1)

        x = torch.randn(2, channels, nlat, nlon, device=self.device)
        loss = conv(x).sum()
        loss.backward()

        self.assertIsNotNone(conv.spectral_bias.grad, msg="spectral_bias.grad should not be None")
        self.assertTrue(
            conv.spectral_bias.grad.abs().max().item() > 0.0,
            msg="spectral_bias.grad should contain non-zero entries",
        )

    @parameterized.expand(
        [
            # nlat, nlon, in_channels, out_channels, num_groups
            [16, 32, 4, 8, 1],
            [16, 32, 8, 4, 1],
            [16, 32, 4, 8, 2],
        ],
        skip_on_empty=True,
    )
    def test_spectral_bias_supports_channel_projection(self, nlat, nlon, in_channels, out_channels, num_groups, verbose=False):
        """Spectral bias works if input and output channel counts differ."""
        set_seed(333)

        conv = SpectralConvS2(
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            bias=True,
        ).to(self.device)

        self.assertEqual(conv.spectral_bias.shape, (1, in_channels, conv.lmax, conv.mmax))

        with torch.no_grad():
            conv.spectral_bias.data.fill_(0.1)

        x = torch.randn(2, in_channels, nlat, nlon, device=self.device)
        y = conv(x)

        self.assertEqual(y.shape, (2, out_channels, nlat, nlon))
        self.assertTrue(torch.isfinite(y).all(), "output contains non-finite values")

        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(conv.spectral_bias.grad, msg="spectral_bias.grad should not be None")
        self.assertTrue(
            conv.spectral_bias.grad.abs().max().item() > 0.0,
            msg="spectral_bias.grad should contain non-zero entries",
        )

    # -----------------------------------------------------------------------
    # Test 12: torch.compile
    #
    # The layer holds complex spectral coefficients throughout, and inductor
    # cannot generate a kernel that reads or writes a complex buffer: triton has
    # no complex type, so the kernel signature fails with KeyError: 'complex64'.
    # Every pointwise step here (bias, contiguity, reshape) therefore has to run
    # on the real view.  The bias path is parameterized separately because it is
    # the only complex arithmetic in the layer.
    # -----------------------------------------------------------------------
    @parameterized.expand(
        [
            # nlat, nlon, in_channels, out_channels, num_groups, bias
            [16, 32, 4, 4, 1, False],
            [16, 32, 4, 8, 2, False],
            [16, 32, 4, 4, 1, True],
            [16, 32, 4, 8, 2, True],
        ],
        skip_on_empty=True,
    )
    @requires_torch_compile
    def test_compile(self, nlat, nlon, in_channels, out_channels, num_groups, bias, verbose=False):
        """The layer compiles and matches eager, forward and backward."""

        set_seed(333)

        conv = SpectralConvS2(
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            as_grid("equiangular", nlat=nlat, nlon=nlon),
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            bias=bias,
        ).to(self.device)
        conv.eval()

        if bias:
            with torch.no_grad():
                conv.spectral_bias.data.fill_(0.1)

        x = torch.randn(2, in_channels, nlat, nlon, device=self.device, requires_grad=True)
        gradient = torch.randn(2, out_channels, nlat, nlon, device=self.device)

        expected = conv(x)
        (expected_grad,) = torch.autograd.grad(expected, x, grad_outputs=gradient)

        # fullgraph is deliberately not requested: _contract_lwise carries its own
        # @torch.compile, and the point here is that inductor can generate code for
        # every complex-valued step, not that the layer traces into a single graph.
        compiled = torch.compile(conv, dynamic=False)
        actual = compiled(x)
        (actual_grad,) = torch.autograd.grad(actual, x, grad_outputs=gradient)

        self.assertTrue(compare_tensors("compiled forward", actual, expected, atol=0.0, rtol=0.0, ulps=_CONTRACTION_ULPS, verbose=verbose))
        self.assertTrue(compare_tensors("compiled backward", actual_grad, expected_grad, atol=0.0, rtol=0.0, ulps=_CONTRACTION_ULPS, verbose=verbose))


class TestCompileIfSupported(unittest.TestCase):
    """compile_if_supported must not fail where torch.compile refuses to run at all."""

    # torch_compile_supported caches its answer; each test patches torch.compile, so it
    # must see a fresh check, and must not leave a patched answer behind.
    def setUp(self):
        torch_compile_supported.cache_clear()

    def tearDown(self):
        torch_compile_supported.cache_clear()

    def test_falls_back_when_torch_compile_refuses(self):
        # what torch 2.9 does on Python 3.14, raised when torch.compile is called
        def refuse(fn, *args, **kwargs):
            raise RuntimeError("torch.compile is not supported on Python 3.14+")

        def fn(x):
            return x + 1

        with unittest.mock.patch.object(torch, "compile", refuse):
            self.assertIs(compile_if_supported(fn), fn)

    def test_other_errors_are_raised(self):
        # anything but the refusal is a real problem, not a reason to run uncompiled
        def broken(fn, *args, **kwargs):
            raise RuntimeError("inductor misconfigured")

        with unittest.mock.patch.object(torch, "compile", broken):
            with self.assertRaisesRegex(RuntimeError, "inductor misconfigured"):
                compile_if_supported(lambda x: x)

    def test_import_with_torch_compile_refused(self):
        # The failure this guards against happened at import, when the class bodies applied
        # the decorator, so import both layers in a fresh interpreter in which torch.compile
        # already refuses. Run outside the repository, so the installed package is imported.
        code = textwrap.dedent(
            """
            import torch

            def refuse(*args, **kwargs):
                raise RuntimeError("torch.compile is not supported on Python 3.14+")

            torch.compile = refuse

            from torch_harmonics.spectral_convolution import SpectralConvS2
            from torch_harmonics.distributed.distributed_spectral_convolution import DistributedSpectralConvS2

            for cls in (SpectralConvS2, DistributedSpectralConvS2):
                fn = cls.__dict__["_contract_lwise"]
                assert not hasattr(fn, "_torchdynamo_orig_callable"), f"{cls.__name__} was compiled"
            """
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tempfile.gettempdir())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_compiles_where_supported(self):
        def fn(x):
            return x + 1

        sentinel = object()
        with unittest.mock.patch.object(torch, "compile", lambda f, *a, **k: sentinel):
            self.assertIs(compile_if_supported(fn), sentinel)
