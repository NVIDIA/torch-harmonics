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
import unittest

import torch
from parameterized import parameterized, parameterized_class
from testutils import compare_tensors, set_seed

from torch_harmonics import ResampleS2, as_grid
from torch_harmonics.quadrature import precompute_latitudes, precompute_longitudes

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device"), _devices)
class TestResampleS2(unittest.TestCase):
    """Tests for the ResampleS2 spherical resampling module."""

    @parameterized.expand(
        [
            [32, 64, "equiangular"],
            [32, 64, "legendre-gauss"],
            [32, 64, "lobatto"],
        ]
    )
    def test_identity(self, nlat, nlon, grid, verbose=False):
        """Identical input/output grid → skip_resampling=True and output is the same object."""
        set_seed(333)

        resample = ResampleS2(as_grid(grid, nlat=nlat, nlon=nlon), as_grid(grid, nlat=nlat, nlon=nlon)).to(self.device)

        self.assertTrue(resample.skip_resampling, "skip_resampling should be True for identical grids")

        data = torch.randn(2, 3, nlat, nlon, dtype=torch.float32, device=self.device)
        out = resample(data)

        # forward() does `return x` so the same tensor object must come back
        self.assertTrue(out is data, "identity resample must return the exact same tensor object")

    @parameterized.expand(
        [
            [32, 64, 16, 32, "equiangular", "equiangular", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "equiangular", "legendre-gauss", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "equiangular", "lobatto", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "legendre-gauss", "equiangular", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "legendre-gauss", "legendre-gauss", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "legendre-gauss", "lobatto", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "lobatto", "equiangular", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "lobatto", "legendre-gauss", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "lobatto", "lobatto", "bilinear", 1e-5, 1e-5],
            [32, 64, 16, 32, "equiangular", "equiangular", "bilinear-spherical", 1e-5, 1e-5],
        ]
    )
    def test_constant_field(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out, mode, atol, rtol, verbose=False):
        """A constant field f=1 must be reproduced exactly under any resampling."""
        set_seed(333)

        resample = ResampleS2(as_grid(grid_in, nlat=nlat_in, nlon=nlon_in), as_grid(grid_out, nlat=nlat_out, nlon=nlon_out), mode=mode).to(self.device)

        data = torch.ones(2, 3, nlat_in, nlon_in, dtype=torch.float32, device=self.device)
        out = resample(data)

        expected = torch.ones(2, 3, nlat_out, nlon_out, dtype=torch.float32, device=self.device)

        self.assertTrue(compare_tensors("constant field", out, expected, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # Only grid pairs where output latitudes lie strictly within the input latitude range
            # (expand_poles=False), so that bilinear interpolation is exact for linear-in-θ functions.
            [32, 64, 16, 32, "equiangular", "equiangular", 1e-5, 1e-5],
            [32, 64, 16, 32, "equiangular", "legendre-gauss", 1e-5, 1e-5],
            [32, 64, 16, 32, "legendre-gauss", "equiangular", 1e-5, 1e-5],
            [32, 64, 16, 32, "legendre-gauss", "legendre-gauss", 1e-5, 1e-5],
            [32, 64, 16, 32, "lobatto", "equiangular", 1e-5, 1e-5],
            [32, 64, 16, 32, "lobatto", "legendre-gauss", 1e-5, 1e-5],
        ]
    )
    def test_linear_latitude_exactness(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out, atol, rtol, verbose=False):
        """Bilinear interpolation is exact for f(θ,φ)=θ (linear in latitude)."""
        set_seed(333)

        resample = ResampleS2(as_grid(grid_in, nlat=nlat_in, nlon=nlon_in), as_grid(grid_out, nlat=nlat_out, nlon=nlon_out)).to(self.device)

        # self.assertFalse(resample.expand_poles,
        #                 f"expand_poles must be False for this test ({grid_in}→{grid_out}), "
        #                  f"otherwise pole extrapolation breaks linear exactness")

        lats_in, _ = precompute_latitudes(nlat_in, grid=grid_in)
        lats_out, _ = precompute_latitudes(nlat_out, grid=grid_out)

        # f(θ, φ) = θ — constant across longitude, linear in latitude
        data = lats_in.float().to(self.device).unsqueeze(-1).expand(nlat_in, nlon_in).contiguous()
        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, nlat_in, nlon_in)

        out = resample(data)

        expected = lats_out.float().to(self.device).unsqueeze(-1).expand(nlat_out, nlon_out).contiguous()
        expected = expected.unsqueeze(0).unsqueeze(0)  # (1, 1, nlat_out, nlon_out)

        # the pole value would differ if this is true
        if resample.expand_poles:
            out = out[..., 1:-1, :]
            expected = expected[..., 1:-1, :]

        self.assertTrue(compare_tensors("linear-in-theta", out, expected, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # Upsample in longitude (nlon_out > nlon_in) so that output nodes near φ=2π
            # require the wrap-around lon_idx_right=0 branch in _upscale_longitudes.
            # Tolerance reflects the O((Δφ)²) bilinear error for sin(φ) with Δφ=2π/32.
            [16, 32, 16, 64, "equiangular", "equiangular", 1e-2, 1e-2],
            [16, 32, 16, 64, "legendre-gauss", "legendre-gauss", 1e-2, 1e-2],
        ]
    )
    def test_longitude_periodicity(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out, atol, rtol, verbose=True):
        """Upsampling in longitude handles the 2π→0 periodic wrap-around correctly."""
        set_seed(333)

        resample = ResampleS2(as_grid(grid_in, nlat=nlat_in, nlon=nlon_in), as_grid(grid_out, nlat=nlat_out, nlon=nlon_out)).to(self.device)

        lons_in = precompute_longitudes(nlon_in)
        lons_out = precompute_longitudes(nlon_out)

        # f(θ, φ) = sin(φ) — smooth, periodic; tests that wrap-around is handled correctly
        data = torch.sin(lons_in).float().to(self.device).unsqueeze(0).expand(nlat_in, nlon_in).contiguous()
        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, nlat_in, nlon_in)

        out = resample(data)

        expected = torch.sin(lons_out).float().to(self.device).unsqueeze(0).expand(nlat_out, nlon_out).contiguous()
        expected = expected.unsqueeze(0).unsqueeze(0)  # (1, 1, nlat_out, nlon_out)

        self.assertTrue(compare_tensors("sin(phi) periodic", out, expected, atol=atol, rtol=rtol, verbose=verbose))

    # The pre-existing coverage of "bilinear-spherical" only used constant fields. For a
    # constant field every neighbouring difference is zero, so the shortest-arc branch was
    # never actually executed. These tests exercise it on non-constant data.

    @parameterized.expand(
        [
            [32, 64, 48, 96, "equiangular", "equiangular"],
            [32, 64, 16, 32, "equiangular", "legendre-gauss"],
        ]
    )
    def test_spherical_agrees_with_bilinear_without_wraps(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out):
        """Without a phase wrap, slerp must reduce exactly to linear interpolation.

        ``wrap_pi`` is the identity for ``|f1 - f0| <= pi``, so on a field whose
        neighbouring samples differ by much less than pi the two modes agree up to
        floating point (``torch.lerp(a, b, t)`` vs ``a + t * (b - a)``).
        """
        set_seed(444)
        grid_in, grid_out = as_grid(grid_in, nlat=nlat_in, nlon=nlon_in), as_grid(grid_out, nlat=nlat_out, nlon=nlon_out)
        lin = ResampleS2(grid_in, grid_out, mode="bilinear").to(self.device)
        sph = ResampleS2(grid_in, grid_out, mode="bilinear-spherical").to(self.device)

        # small amplitude keeps every neighbouring difference far below pi
        data = 0.2 * torch.randn(2, 3, nlat_in, nlon_in, dtype=torch.float64, device=self.device)
        self.assertTrue(compare_tensors("slerp == lerp off the branch cut", sph(data.double()), lin(data.double()), atol=1e-12, rtol=1e-12))

    @parameterized.expand([[32, 64, 48, 96, "equiangular", "equiangular"]])
    def test_spherical_takes_shortest_arc(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out):
        """Across a 2*pi wrap the spherical mode must beat linear on angular error.

        A sawtooth phase field is smooth on the circle but discontinuous in value.
        Linear interpolation runs the wrong way around the circle at each branch cut;
        shortest-arc interpolation does not.
        """
        set_seed(555)
        grid_in, grid_out = as_grid(grid_in, nlat=nlat_in, nlon=nlon_in), as_grid(grid_out, nlat=nlat_out, nlon=nlon_out)
        lin = ResampleS2(grid_in, grid_out, mode="bilinear").to(self.device)
        sph = ResampleS2(grid_in, grid_out, mode="bilinear-spherical").to(self.device)

        wrap = lambda t: torch.remainder(t + math.pi, 2 * math.pi) - math.pi
        lons_in = precompute_longitudes(nlon_in).to(self.device)
        lons_out = precompute_longitudes(nlon_out).to(self.device)
        data = wrap(3.0 * lons_in).float().expand(nlat_in, nlon_in)[None, None].contiguous()
        ref = wrap(3.0 * lons_out).float().expand(nlat_out, nlon_out)[None, None].contiguous()

        # compare as directions on the circle, i.e. modulo 2*pi
        err_lin = wrap(lin(data) - ref).abs().max().item()
        err_sph = wrap(sph(data) - ref).abs().max().item()
        self.assertLess(err_sph, 0.25, f"shortest-arc angular error too large: {err_sph}")
        self.assertLess(err_sph, 0.2 * err_lin, f"shortest-arc ({err_sph}) should beat linear ({err_lin}) across the wrap")

    @parameterized.expand([[32, 64, 48, 96, "equiangular", "equiangular"]])
    def test_spherical_shift_equivariance(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out):
        """Resampling must commute with adding a constant: f -> f + c gives out -> out + c.

        The previous slerp-weights-on-scalars formula violated this badly (a shift of 10
        moved the output by ~11.7), because those weights are not a partition of unity.
        """
        set_seed(666)
        sph = ResampleS2(as_grid(grid_in, nlat=nlat_in, nlon=nlon_in), as_grid(grid_out, nlat=nlat_out, nlon=nlon_out), mode="bilinear-spherical").to(self.device)

        data = (torch.rand(2, 3, nlat_in, nlon_in, dtype=torch.float64, device=self.device) * 2 - 1) * math.pi
        base = sph(data)
        for c in (1.0, 10.0):
            self.assertTrue(compare_tensors(f"shift equivariance c={c}", sph(data + c), base + c, atol=1e-10, rtol=1e-10))

    @parameterized.expand([[32, 64, 48, 96, "equiangular", "equiangular"]])
    def test_spherical_stays_bounded(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out):
        """Interpolation must not amplify: the old formula turned inputs in [-pi, pi] into ~1e4.

        The result is a continuous lift from the left sample rather than a re-wrapped
        angle, so it may exceed the input range by at most pi per interpolation pass
        (latitude then longitude) -- but never by more.
        """
        set_seed(777)
        sph = ResampleS2(as_grid(grid_in, nlat=nlat_in, nlon=nlon_in), as_grid(grid_out, nlat=nlat_out, nlon=nlon_out), mode="bilinear-spherical").to(self.device)

        data = (torch.rand(2, 3, nlat_in, nlon_in, dtype=torch.float32, device=self.device) * 2 - 1) * math.pi
        out = sph(data)
        bound = data.abs().max().item() + 2 * math.pi + 1e-5
        self.assertLess(out.abs().max().item(), bound, "shortest-arc interpolation must not amplify beyond one wrap per pass")

    @parameterized.expand([[48, 96, 64, 128, "legendre-gauss", "equiangular"]])
    def test_spherical_pole_uses_circular_mean(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out):
        """The pole value of an angle field is the mean *direction*, not the mean value.

        ``_expand_poles`` reduces the adjacent ring over longitude to get a single,
        longitude-independent pole value. For an angle field that reduction has to
        average unit vectors: the arithmetic mean of ``+pi`` and ``-pi``, which are the
        same direction, is ``0`` -- the opposite one.
        """
        set_seed(888)
        sph = ResampleS2(as_grid(grid_in, nlat=nlat_in, nlon=nlon_in), as_grid(grid_out, nlat=nlat_out, nlon=nlon_out), mode="bilinear-spherical").to(self.device)
        self.assertTrue(sph.expand_poles, "this grid pair must exercise the pole extension")

        wrap = lambda t: torch.remainder(t + math.pi, 2 * math.pi) - math.pi
        lons_in = precompute_longitudes(nlon_in).to(self.device)
        # a field pointing (almost) due pi everywhere, i.e. straddling the branch cut
        data = wrap(math.pi + 0.2 * torch.cos(lons_in)).expand(nlat_in, nlon_in)[None, None].contiguous().float()

        out = sph(data)
        # the pole row must be a single direction, and it must be pi (not 0)
        pole = out[0, 0, 0, :]
        self.assertLess((pole - pole[0]).abs().max().item(), 1e-6, "pole value must not depend on longitude")
        self.assertLess(wrap(pole - math.pi).abs().max().item(), 1e-3, f"pole must point at pi, got {pole[0].item()}")

    # In "bilinear" mode resampling is a linear operator: the gather indices and the
    # interpolation weights are fixed at construction, and the pole extension is a mean.
    # That admits exact gradient checks instead of finite differences -- the Jacobian is
    # the operator itself, so backward has to be precisely its transpose.

    @parameterized.expand(
        [
            [32, 64, 16, 32, "equiangular", "equiangular"],
            [16, 32, 24, 48, "legendre-gauss", "equiangular"],  # expand_poles=True: covers _expand_poles
            [32, 64, 16, 32, "lobatto", "legendre-gauss"],
        ]
    )
    def test_bilinear_adjoint(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out):
        """``<R x, y> == <x, R^T y>``: backward must be the exact adjoint of forward.

        A transposed operator is pinned down by this identity for random ``x`` and ``y``,
        so it catches any mis-mapped index or weight in the backward pass without needing
        a finite-difference tolerance.
        """
        set_seed(999)
        resample = ResampleS2(as_grid(grid_in, nlat=nlat_in, nlon=nlon_in), as_grid(grid_out, nlat=nlat_out, nlon=nlon_out)).to(self.device)

        x = torch.randn(1, 1, nlat_in, nlon_in, dtype=torch.float64, device=self.device, requires_grad=True)
        y = torch.randn(1, 1, nlat_out, nlon_out, dtype=torch.float64, device=self.device)

        lhs = (resample(x) * y).sum()  # <R x, y>
        lhs.backward()
        rhs = (x.grad * x.detach()).sum()  # <x, R^T y>, since x.grad == R^T y for linear R

        scale = max(1.0, abs(lhs.item()))
        self.assertLess(abs(lhs.item() - rhs.item()) / scale, 1e-12, f"adjoint mismatch: <Rx,y>={lhs.item()} vs <x,R^T y>={rhs.item()}")

    @parameterized.expand(
        [
            [32, 64, 16, 32, "equiangular", "equiangular"],
            [16, 32, 24, 48, "legendre-gauss", "equiangular"],
            [32, 64, 16, 32, "lobatto", "legendre-gauss"],
        ]
    )
    def test_bilinear_jacobian_is_constant(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out):
        """The Jacobian may not depend on the input, and forward must obey superposition.

        Probed through a vector-Jacobian product rather than the full Jacobian: for a fixed
        cotangent, ``R^T y`` has to come out identical at unrelated inputs. Any data
        dependence sneaking into forward -- a value-dependent branch or index -- breaks
        one of the two checks.
        """
        set_seed(1001)
        resample = ResampleS2(as_grid(grid_in, nlat=nlat_in, nlon=nlon_in), as_grid(grid_out, nlat=nlat_out, nlon=nlon_out)).to(self.device)
        shape_in = (1, 1, nlat_in, nlon_in)
        kw = dict(dtype=torch.float64, device=self.device)

        y = torch.randn(1, 1, nlat_out, nlon_out, **kw)

        def vjp(x0):
            x = x0.clone().requires_grad_(True)
            (resample(x) * y).sum().backward()
            return x.grad

        xa = torch.randn(*shape_in, **kw)
        xb = torch.randn(*shape_in, **kw) * 7.0 + 3.0
        self.assertTrue(compare_tensors("constant Jacobian", vjp(xa), vjp(xb), atol=1e-14, rtol=1e-14))

        # superposition: R(a*x1 + b*x2) == a*R(x1) + b*R(x2)
        a, b = 2.5, -1.75
        x1, x2 = torch.randn(*shape_in, **kw), torch.randn(*shape_in, **kw)
        self.assertTrue(compare_tensors("superposition", resample(a * x1 + b * x2), a * resample(x1) + b * resample(x2), atol=1e-12, rtol=1e-12))


if __name__ == "__main__":
    unittest.main()
