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

import math
import unittest
import warnings

import numpy as np
import torch
from parameterized import parameterized, parameterized_class
from scipy.special import gammaln, lpmv
from testutils import compare_tensors, disable_tf32, requires_torch_compile, set_seed
from torch.autograd import gradcheck

import torch_harmonics as th
from torch_harmonics.legendre import _precompute_dlegpoly, _precompute_legpoly
from torch_harmonics.quadrature import precompute_latitudes
from torch_harmonics.sht import _fold_resampled_latitude, _precompute_cc_resampling, _precompute_f1_resampling, _precompute_folded_projection

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


def random_sht_coeffs(batch_size, lmax, mmax, device, zero_l0=False, lmmax=None):
    """Random scalar SHT coefficients with proper structure:
    m=0 column real (real-valued field), triangular support (m <= l),
    and optionally l=0 row zeroed (needed when testing gradient/curl)."""
    c = torch.randn(batch_size, lmax, mmax, dtype=torch.complex128, device=device)
    c[:, :, 0] = c[:, :, 0].real
    for l in range(lmax):
        if l + 1 < mmax:
            c[:, l, l + 1 :] = 0.0
    if zero_l0:
        c[:, 0, :] = 0.0
    if lmmax is not None:
        l = torch.arange(lmax, device=device)[:, None]
        m = torch.arange(mmax, device=device)[None, :]
        c = c * (l - m < lmmax)
    return c


# The largest band each grid recovers: its native band, max_exact_degree, and on equiangular
# and Fejer-1 also the extended band of the folded analysis.
_EXTENDED_DEGREE = {"equiangular": lambda nlat: nlat - 1, "fejer1": lambda nlat: nlat}
_RECOVERY_BANDS = [("legendre-gauss", False), ("lobatto", False), ("equiangular", False), ("fejer1", False), ("equiangular", True), ("fejer1", True)]

# nlat, grid, extended, mmax (None: lmax), lmmax, norm, dtype
_RECOVERY_CASES = (
    [[nlat, grid, extended, None, None, norm, torch.float64] for nlat in (32, 33) for grid, extended in _RECOVERY_BANDS for norm in ("ortho", "four-pi", "schmidt")]
    + [[nlat, grid, extended, None, None, "ortho", torch.float32] for nlat in (32, 33) for grid, extended in _RECOVERY_BANDS]
    # fewer orders and an l - m bound on top of the extended band
    + [[33, "equiangular", True, None, 8, "four-pi", torch.float64], [32, "fejer1", True, 12, 6, "schmidt", torch.float64]]
)


def _band(grid, nlat, extended):
    """The descriptor of an nlat x 2 nlat grid and the degree bound of its native or extended band."""
    descriptor = th.as_grid(grid, nlat=nlat, nlon=2 * nlat)
    return descriptor, _EXTENDED_DEGREE[grid](nlat) if extended else descriptor.max_exact_degree


def _check_coefficient_recovery(test, nlat, grid, extended, mmax, lmmax, norm, dtype, vector, verbose=False):
    """Assert sht(isht(c)) == c for band-limited c, relative to the largest coefficient."""
    set_seed(333)
    descriptor, lmax = _band(grid, nlat, extended)
    mmax = lmax if mmax is None else mmax

    forward_cls, inverse_cls = (th.RealVectorSHT, th.InverseRealVectorSHT) if vector else (th.RealSHT, th.InverseRealSHT)
    forward = forward_cls(descriptor, lmax=lmax, mmax=mmax, lmmax=lmmax, norm=norm).to(device=test.device, dtype=dtype)
    inverse = inverse_cls(descriptor, lmax=lmax, mmax=mmax, lmmax=lmmax, norm=norm).to(device=test.device, dtype=dtype)

    # there is no vector harmonic of degree 0
    coeffs = random_sht_coeffs(2, lmax, mmax, test.device, zero_l0=vector, lmmax=lmmax)
    if vector:
        coeffs = torch.stack([coeffs, random_sht_coeffs(2, lmax, mmax, test.device, zero_l0=True, lmmax=lmmax)], dim=-3)
    coeffs = coeffs.to(torch.complex128 if dtype == torch.float64 else torch.complex64)

    recovered = forward(inverse(coeffs))

    if dtype == torch.float64:
        tol = 1e-11 if vector else 1e-12
    else:
        tol = 1e-4 if vector else 1e-5
    atol = tol * coeffs.abs().max().item()
    test.assertTrue(compare_tensors("recovered coefficients", recovered, coeffs, atol=atol, rtol=0.0, verbose=verbose))


# nlat, grid, extended
_GRADIENT_CASES = [[nlat, grid, extended] for nlat in (12, 15) for grid, extended in _RECOVERY_BANDS]


def _check_gradients(test, nlat, grid, extended, vector):
    """gradcheck the forward and the inverse transform themselves, in float64.

    Both are linear, so central differences are exact up to roundoff and the comparison can
    be tight. Fast mode compares v^T J u for random u and v instead of every Jacobian entry:
    for a linear map, a wrong entry changes that product with probability one, at the cost
    of a few transforms rather than one per input and output element. The product is about
    5e-5 at these sizes, so atol stays well below it; at 1e-8 a backward scaled by 1 + 1e-4
    still passed. Complex coefficients are checked as real and imaginary parts.
    """
    set_seed(333)
    descriptor, lmax = _band(grid, nlat, extended)
    forward_cls, inverse_cls = (th.RealVectorSHT, th.InverseRealVectorSHT) if vector else (th.RealSHT, th.InverseRealSHT)
    forward = forward_cls(descriptor, lmax=lmax, mmax=lmax).to(test.device)
    inverse = inverse_cls(descriptor, lmax=lmax, mmax=lmax).to(test.device)

    leading = (2, 2) if vector else (2,)
    x = torch.randn(*leading, *descriptor.shape, dtype=torch.float64, device=test.device, requires_grad=True)
    test.assertTrue(gradcheck(forward, (x,), eps=1e-6, atol=1e-10, rtol=1e-6, fast_mode=True))
    c = torch.randn(*leading, lmax, lmax, dtype=torch.complex128, device=test.device, requires_grad=True)
    test.assertTrue(gradcheck(inverse, (c,), eps=1e-6, atol=1e-10, rtol=1e-6, fast_mode=True))


# nlat, grid, norm
_GRAM_CASES = [[nlat, grid, norm] for nlat in (12, 15) for grid in ("legendre-gauss", "lobatto", "equiangular", "fejer1") for norm in ("ortho", "four-pi", "schmidt")]


def _check_gram(test, nlat, grid, norm, vector, verbose=False):
    """Assert that synthesis maps the coefficient basis to fields with a diagonal Gram matrix.

    One field is synthesized per real basis function: c[l, m] = 1 and, for m > 0, also
    c[l, m] = 1j, on the spheroidal and the toroidal channel for vector fields. Their inner
    products under the grid's quadrature form the Gram matrix. Its diagonal holds
    w_m * W_l, with w_m = 1 for m = 0 and 2 for m > 0 from the real irfft folding +m and -m,
    and

      ortho:   W_l = 1
      four-pi: W_l = 1 / (4 pi)
      schmidt: W_l = (2l + 1) / (4 pi)

    times l(l + 1) = ||grad Y_l^m||^2 for vector fields, which have no degree 0. The quadrature
    integrates these products exactly up to max_exact_degree, so every off-diagonal entry,
    including those between spheroidal and toroidal fields, vanishes to roundoff.
    """
    descriptor = th.as_grid(grid, nlat=nlat, nlon=2 * nlat)
    lmax = mmax = descriptor.max_exact_degree
    inverse_cls = th.InverseRealVectorSHT if vector else th.InverseRealSHT
    inverse = inverse_cls(descriptor, lmax=lmax, mmax=mmax, norm=norm).to(test.device)

    channels = (0, 1) if vector else (None,)
    basis, diagonal = [], []
    for channel in channels:
        for l in range(1 if vector else 0, lmax):
            weight = {"ortho": 1.0, "four-pi": 1.0 / (4.0 * math.pi), "schmidt": (2 * l + 1) / (4.0 * math.pi)}[norm]
            if vector:
                weight *= l * (l + 1)
            for m in range(min(l + 1, mmax)):
                for value in (1.0, 1.0j) if m > 0 else (1.0,):
                    c = torch.zeros((2, lmax, mmax) if vector else (lmax, mmax), dtype=torch.complex128)
                    c[(channel, l, m) if vector else (l, m)] = value
                    basis.append(c)
                    diagonal.append(weight * (2.0 if m > 0 else 1.0))

    with torch.no_grad():
        fields = inverse(torch.stack(basis).to(test.device))

    _, w_lat = precompute_latitudes(nlat, grid=grid)
    measure = (2.0 * math.pi / descriptor.nlon) * w_lat.to(device=test.device, dtype=torch.float64)
    fields = fields.reshape(len(basis), -1, nlat, descriptor.nlon)
    gram = torch.einsum("ivkn,k,jvkn->ij", fields, measure, fields)

    expected = torch.diag(torch.tensor(diagonal, dtype=torch.float64, device=test.device))
    atol = 1e-12 * max(diagonal)
    test.assertTrue(compare_tensors("Gram matrix", gram, expected, atol=atol, rtol=0.0, verbose=verbose))


@parameterized_class(("device"), _devices)
class TestPentagonalSHT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disable_tf32()

    @parameterized.expand(
        [
            (f"{truncation}_{dtype_name}_{'vector' if vector else 'scalar'}", lmax, mmax, lmmax, dtype, vector)
            for truncation, lmax, mmax, lmmax in [
                ("triangular", 16, 16, None),
                ("trapezoidal", 20, 9, None),
                ("rhomboidal", 17, 9, 9),
                ("rhomboidal_capped", 12, 9, 9),
                ("pentagonal", 14, 9, 6),
                ("r42", 85, 43, 43),
                ("r42_capped", 64, 43, 43),
            ]
            for dtype_name, dtype in (("fp32", torch.float32), ("fp64", torch.float64))
            for vector in (False, True)
        ]
    )
    def test_scalar_and_vector_forward_inverse(self, case, lmax, mmax, lmmax, dtype, vector, verbose=False):
        set_seed(333)
        nlat = max(32, lmax + 1)
        grid = th.as_grid("legendre-gauss", nlat=nlat, nlon=2 * nlat)
        kwargs = dict(lmax=lmax, mmax=mmax, lmmax=lmmax)
        l = torch.arange(lmax, device=self.device)[:, None]
        m = torch.arange(mmax, device=self.device)[None, :]
        keep = m <= l
        if lmmax is not None:
            keep &= l - m < lmmax
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        tol = 1e-7 if dtype == torch.float64 else 1e-5

        forward_cls, inverse_cls = (th.RealVectorSHT, th.InverseRealVectorSHT) if vector else (th.RealSHT, th.InverseRealSHT)
        forward = forward_cls(grid, **kwargs).to(device=self.device, dtype=dtype)
        inverse = inverse_cls(grid, **kwargs).to(device=self.device, dtype=dtype)
        forward_ref = forward_cls(grid, lmax=lmax, mmax=mmax).to(device=self.device, dtype=dtype)
        inverse_ref = inverse_cls(grid, lmax=lmax, mmax=mmax).to(device=self.device, dtype=dtype)
        shape = (1, 2, lmax, mmax) if vector else (1, lmax, mmax)
        coeffs = torch.randn(shape, dtype=complex_dtype, device=self.device)
        coeffs[..., 0] = coeffs[..., 0].real
        if vector:
            coeffs[..., 0, :] = 0
        expected = coeffs * keep
        coeffs.requires_grad_(True)
        ref_coeffs = coeffs.detach().clone().requires_grad_(True)
        signal = inverse(coeffs)
        ref_signal = inverse_ref(ref_coeffs * keep)
        self.assertTrue(compare_tensors("inverse", ref_signal, signal, atol=tol, rtol=tol, verbose=verbose))
        recovered = forward(signal)
        self.assertTrue(compare_tensors("round trip", expected, recovered, atol=tol, rtol=tol, verbose=verbose))
        self.assertTrue(torch.equal(recovered[..., ~keep], torch.zeros_like(recovered[..., ~keep])))
        spatial_grad = torch.randn_like(signal)
        signal.backward(spatial_grad)
        ref_signal.backward(spatial_grad)
        self.assertTrue(compare_tensors("inverse gradient", ref_coeffs.grad, coeffs.grad, atol=tol, rtol=tol, verbose=verbose))

        inp = torch.randn_like(signal, requires_grad=True)
        ref_inp = inp.detach().clone().requires_grad_(True)
        output = forward(inp)
        ref_output = forward_ref(ref_inp) * keep
        self.assertTrue(compare_tensors("forward", ref_output, output, atol=tol, rtol=tol, verbose=verbose))
        spectral_grad = torch.randn_like(output)
        output.backward(spectral_grad)
        ref_output.backward(spectral_grad)
        self.assertTrue(compare_tensors("forward gradient", ref_inp.grad, inp.grad, atol=tol, rtol=tol, verbose=verbose))


@parameterized_class(("device"), _devices)
class TestSphericalHarmonicTransform(unittest.TestCase):
    """Test the spherical harmonic transform (CPU/CUDA if available)."""

    def setUp(self):
        disable_tf32()

    @parameterized.expand(_RECOVERY_CASES, skip_on_empty=True)
    def test_forward_inverse(self, nlat, grid, extended, mmax, lmmax, norm, dtype, verbose=False):
        """sht(isht(c)) == c at the largest band each grid recovers.

        This implies the spatial round trip isht(sht(f)) == f for every band-limited f, and
        compares the coefficients themselves, so an error in a single mode cannot hide in
        the synthesized field. The extended rows exercise the folded analysis of the
        equiangular and Fejer-1 grids beyond their native band.
        """
        _check_coefficient_recovery(self, nlat, grid, extended, mmax, lmmax, norm, dtype, vector=False, verbose=verbose)

    @parameterized.expand(_GRADIENT_CASES, skip_on_empty=True)
    def test_grads(self, nlat, grid, extended, verbose=False):
        """Autograd of sht and isht matches finite differences of the transforms themselves.

        The normalization only rescales the weight buffer, not the backward path, so it is
        not varied here; test_forward_inverse covers every normalization.
        """
        _check_gradients(self, nlat, grid, extended, vector=False)

    @parameterized.expand(
        [
            [32, 64, 16, "equiangular", 1e-9, 1e-9],
            [32, 64, 16, "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 16, "lobatto", 1e-9, 1e-9],
            [32, 64, 16, "fejer1", 1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_cross_norm_consistency(self, nlat, nlon, batch_size, grid, atol, rtol, verbose=False):
        """The three normalizations applied to the same band-limited signal must satisfy
        known per-coefficient scaling relations:
          c_four_pi[l,m] = c_ortho[l,m] * sqrt(4*pi)
          c_schmidt[l,m] = c_ortho[l,m] * sqrt(4*pi / (2*l+1))
        This catches bugs where a norm is internally self-consistent but scaled wrongly
        relative to the standard conventions.
        """
        # set seed
        set_seed(333)

        sht_ortho = th.RealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm="ortho").to(self.device)
        sht_four_pi = th.RealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm="four-pi").to(self.device)
        sht_schmidt = th.RealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm="schmidt").to(self.device)
        isht_ortho = th.InverseRealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm="ortho").to(self.device)

        lmax = sht_ortho.lmax
        mmax = sht_ortho.mmax

        with torch.no_grad():
            c = random_sht_coeffs(batch_size, lmax, mmax, self.device)
            signal = isht_ortho(c)  # band-limited real spatial field

            c_ortho = sht_ortho(signal)
            c_four_pi = sht_four_pi(signal)
            c_schmidt = sht_schmidt(signal)

        # four-pi: each coefficient is sqrt(4*pi) larger than ortho
        c_four_pi_ref = c_ortho * math.sqrt(4.0 * math.pi)
        self.assertTrue(compare_tensors("four-pi vs ortho scaling", c_four_pi, c_four_pi_ref, atol=atol, rtol=rtol, verbose=verbose))

        # schmidt: coefficient at degree l is sqrt(4*pi/(2*l+1)) * c_ortho[l,m]
        l_vals = torch.arange(lmax, dtype=torch.float64, device=self.device)
        schmidt_scale = torch.sqrt(4.0 * math.pi / (2.0 * l_vals + 1.0))  # (lmax,)
        c_schmidt_ref = c_ortho * schmidt_scale[:, None]  # broadcast over mmax
        self.assertTrue(compare_tensors("schmidt vs ortho scaling", c_schmidt, c_schmidt_ref, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            [32, 64, "ortho", "equiangular", 1e-9, 1e-9],
            [32, 64, "ortho", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, "ortho", "lobatto", 1e-9, 1e-9],
            [32, 64, "four-pi", "equiangular", 1e-9, 1e-9],
            [32, 64, "four-pi", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, "four-pi", "lobatto", 1e-9, 1e-9],
            [32, 64, "schmidt", "equiangular", 1e-9, 1e-9],
            [32, 64, "schmidt", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, "schmidt", "lobatto", 1e-9, 1e-9],
            [32, 64, "ortho", "fejer1", 1e-9, 1e-9],
            [32, 64, "four-pi", "fejer1", 1e-9, 1e-9],
            [32, 64, "schmidt", "fejer1", 1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_known_function(self, nlat, nlon, norm, grid, atol, rtol, verbose=False):
        """The SHT of analytically known functions must produce exact spectral coefficients.

        f(θ,φ) = 1:
          ortho   → c[0,0] = sqrt(4*pi),  all others = 0
          four-pi → c[0,0] = 4*pi,        all others = 0
          schmidt → c[0,0] = 4*pi,        all others = 0

        f(θ,φ) = cos θ:
          ortho   → c[1,0] = sqrt(4*pi/3),  all others = 0
          four-pi → c[1,0] = 4*pi/sqrt(3),  all others = 0
          schmidt → c[1,0] = 4*pi/3,        all others = 0

        These are strict value checks that will fail even when the forward/inverse
        transforms are mutually consistent but carry a wrong overall scale.
        """
        sht = th.RealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm=norm).to(self.device)
        lmax = sht.lmax
        mmax = sht.mmax

        # colatitude nodes θ in [0, π] (north pole to south pole)
        lats, _ = precompute_latitudes(nlat, grid=grid)
        lats = lats.to(device=self.device, dtype=torch.float64)
        cost = torch.cos(lats)  # shape (nlat,)

        # ---- f = 1 (constant field) ----
        f_const = torch.ones(nlat, nlon, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            c_const = sht(f_const)  # (lmax, mmax), complex

        if norm == "ortho":
            c00_ref = math.sqrt(4.0 * math.pi)
        else:  # four-pi and schmidt agree for l=0
            c00_ref = 4.0 * math.pi

        self.assertTrue(
            compare_tensors(
                "f=1: c[0,0]",
                c_const[0:1, 0:1].real,
                torch.tensor([[c00_ref]], dtype=torch.float64, device=self.device),
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )
        c_rest = c_const.clone()
        c_rest[0, 0] = 0.0
        self.assertTrue(
            compare_tensors(
                "f=1: all other coeffs vanish",
                c_rest.abs(),
                torch.zeros_like(c_rest.abs()),
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )

        # ---- f = cos θ ----
        f_costheta = cost.unsqueeze(-1).expand(nlat, nlon).contiguous()
        with torch.no_grad():
            c_cos = sht(f_costheta)  # (lmax, mmax), complex

        if norm == "ortho":
            c10_ref = math.sqrt(4.0 * math.pi / 3.0)
        elif norm == "four-pi":
            c10_ref = 4.0 * math.pi / math.sqrt(3.0)
        else:  # schmidt
            c10_ref = 4.0 * math.pi / 3.0

        self.assertTrue(
            compare_tensors(
                "f=cosθ: c[1,0]",
                c_cos[1:2, 0:1].real,
                torch.tensor([[c10_ref]], dtype=torch.float64, device=self.device),
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )
        c_rest = c_cos.clone()
        c_rest[1, 0] = 0.0
        self.assertTrue(
            compare_tensors(
                "f=cosθ: all other coeffs vanish",
                c_rest.abs(),
                torch.zeros_like(c_rest.abs()),
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )

        # ---- batched: both signals together ----
        # Exercises the leading batch dimensions (..., nlat, nlon) that the
        # single-sample checks above never reach.
        f_batch = torch.stack([f_const, f_costheta], dim=0)  # (2, nlat, nlon)
        with torch.no_grad():
            c_batch = sht(f_batch)  # (2, lmax, mmax)

        self.assertTrue(
            compare_tensors(
                "batch f=1 matches unbatched",
                c_batch[0],
                c_const,
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )
        self.assertTrue(
            compare_tensors(
                "batch f=cosθ matches unbatched",
                c_batch[1],
                c_cos,
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )

    @parameterized.expand(
        [
            [32, 64, 16, "ortho", "equiangular", 1e-9, 1e-9],
            [32, 64, 16, "ortho", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 16, "ortho", "lobatto", 1e-9, 1e-9],
            [32, 64, 16, "four-pi", "equiangular", 1e-9, 1e-9],
            [32, 64, 16, "four-pi", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 16, "four-pi", "lobatto", 1e-9, 1e-9],
            [32, 64, 16, "schmidt", "equiangular", 1e-9, 1e-9],
            [32, 64, 16, "schmidt", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 16, "schmidt", "lobatto", 1e-9, 1e-9],
            [32, 64, 16, "ortho", "fejer1", 1e-9, 1e-9],
            [32, 64, 16, "four-pi", "fejer1", 1e-9, 1e-9],
            [32, 64, 16, "schmidt", "fejer1", 1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_csphase(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        """Toggling csphase only flips the sign of odd-m spectral columns.

        csphase=True (default) multiplies the Legendre weight rows at odd m by -1.
        Applied to the same spatial signal, the two transforms must satisfy:

          sht(csphase=True) [l, m] = -sht(csphase=False)[l, m]  for odd  m
          sht(csphase=True) [l, m] =  sht(csphase=False)[l, m]  for even m

        This catches a bug where csphase is applied in both the forward and
        inverse transforms, causing the signs to cancel and hiding the error.
        """
        # set seed
        set_seed(333)

        sht_cs = th.RealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm=norm, csphase=True).to(self.device)
        sht_no_cs = th.RealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm=norm, csphase=False).to(self.device)
        isht = th.InverseRealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm=norm).to(self.device)
        lmax, mmax = sht_cs.lmax, sht_cs.mmax

        with torch.no_grad():
            c = random_sht_coeffs(batch_size, lmax, mmax, self.device)
            signal = isht(c)  # band-limited spatial field

            c_cs = sht_cs(signal)
            c_no_cs = sht_no_cs(signal)

        # build the expected sign pattern: +1 for even m, -1 for odd m
        sign = torch.ones(mmax, dtype=torch.float64, device=self.device)
        sign[1::2] = -1.0

        self.assertTrue(
            compare_tensors(
                "csphase sign flip",
                c_cs,
                c_no_cs * sign,
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )

    @parameterized.expand(
        [
            # even-even
            [12, 24, "ortho", "equiangular", 1e-5, 1e-5],
            [12, 24, "ortho", "legendre-gauss", 1e-5, 1e-5],
            [12, 24, "ortho", "lobatto", 1e-5, 1e-5],
            [12, 24, "ortho", "fejer1", 1e-5, 1e-5],
        ],
        skip_on_empty=True,
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_device_instantiation(self, nlat, nlon, norm, grid, atol, rtol, verbose=False):
        # set seed
        set_seed(333)

        # init on cpu
        sht_host = th.RealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm=norm)
        isht_host = th.InverseRealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm=norm)

        # init on device
        with torch.device(self.device):
            sht_device = th.RealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm=norm)
            isht_device = th.InverseRealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm=norm)

        self.assertTrue(compare_tensors("sht weights", sht_host.weights.cpu(), sht_device.weights.cpu(), atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("isht weights", isht_host.pct.cpu(), isht_device.pct.cpu(), atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            [12, 24, 2, "equiangular"],
            [12, 24, 2, "legendre-gauss"],
            [11, 22, 2, "equiangular"],
            [12, 24, 2, "fejer1"],
        ],
        skip_on_empty=True,
    )
    @requires_torch_compile
    def test_compile(self, nlat, nlon, batch_size, grid, verbose=False):
        """The scalar round trip compiles into a single graph and matches eager.

        The round trip is compiled as one function so the complex spectral coefficients
        are an *intermediate* buffer rather than a graph output -- that is the case
        inductor has to generate code for.  Triton has no complex type, so any pointwise
        kernel over a complex buffer fails codegen with ``KeyError: 'complex64'``; on CPU
        the C++ backend is used instead, where the exposure is a layout mismatch on
        ``aten.complex`` caught by ``assert_size_stride``.  Both are worth covering, which
        the CPU/CUDA parameterization of this class does.

        Backward is included deliberately: ``_EnsureContiguous.backward`` copies a complex
        gradient, and that copy only appears in the joint graph.
        """

        set_seed(333)

        sht = th.RealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon)).to(self.device)
        isht = th.InverseRealSHT(th.as_grid(grid, nlat=nlat, nlon=nlon)).to(self.device)

        def fn(t):
            return isht(sht(t))

        x = torch.randn(batch_size, nlat, nlon, device=self.device, dtype=torch.float32, requires_grad=True)
        gradient = torch.randn_like(x)

        expected = fn(x)
        (expected_grad,) = torch.autograd.grad(expected, x, grad_outputs=gradient)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        actual = compiled(x)
        (actual_grad,) = torch.autograd.grad(actual, x, grad_outputs=gradient)

        self.assertTrue(compare_tensors("compiled forward", actual, expected, atol=1e-5, rtol=1e-5, verbose=verbose))
        self.assertTrue(compare_tensors("compiled backward", actual_grad, expected_grad, atol=1e-5, rtol=1e-5, verbose=verbose))


@parameterized_class(("device"), _devices)
class TestSphericalHarmonicsFunctions(unittest.TestCase):
    """Test fundamental properties of the real spherical harmonic basis functions.

    InverseRealSHT with norm="ortho" synthesizes orthonormal basis functions on
    the sphere.  Setting a single complex coefficient c_{l,m} = 1 synthesizes

      f_{l,0}      = Y_l^0(theta, phi)                   for m = 0
      f_{l,m,cos}  ~ P_l^m(cos theta) * cos(m*phi)       for m > 0  (c_{l,m} = 1+0j)
      f_{l,m,sin}  ~ P_l^m(cos theta) * sin(m*phi)       for m > 0  (c_{l,m} = 0+1j)

    With ortho normalization the sphere inner products satisfy, and the other
    normalizations rescale them per degree (see _check_gram):
      <f_{l,0},     f_{l',0}    > = delta_{ll'}
      <f_{l,m,cos}, f_{l',m',*}> = 2 * delta_{ll'} * delta_{mm'}   for m, m' > 0
      <f_{l,m,sin}, f_{l',m',*}> = 2 * delta_{ll'} * delta_{mm'}   for m, m' > 0

    The factor of 2 for m > 0 arises because the real irfft folds the +m and -m
    modes together, doubling the amplitude of each mode.
    """

    def setUp(self):
        disable_tf32()

    @parameterized.expand(_GRAM_CASES, skip_on_empty=True)
    def test_orthogonality(self, nlat, grid, norm, verbose=False):
        """isht synthesizes mutually orthogonal basis functions with the norms of each convention."""
        _check_gram(self, nlat, grid, norm, vector=False, verbose=verbose)


@parameterized_class(("device"), _devices)
class TestVectorSphericalHarmonicTransform(unittest.TestCase):
    """Tests for the consistency between the scalar SHT and the vector SHT.

    RealVectorSHT includes a 1/(l*(l+1)) normalization in its quadrature weights
    so that the spheroidal/toroidal spectral coefficients relate directly to the
    scalar SHT coefficients of the generating potential:

      Gradient:  vsht(ivsht([c, 0]))[spheroidal] = c,  [toroidal] = 0  (l > 0)
      Curl:      vsht(ivsht([0, c]))[spheroidal] = 0,  [toroidal] = c  (l > 0)

    The l = 0 mode is zero in both vsht and ivsht because the gradient and curl
    of a constant field (Y_0^0) vanish identically on the sphere.

    These tests catch swapped spheroidal/toroidal channels, wrong signs in the
    dP/dtheta or P/sin(theta) terms, and incorrect l*(l+1) normalization.
    """

    def setUp(self):
        disable_tf32()

    @parameterized.expand(
        [
            [32, 64, 16, "ortho", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "ortho", "equiangular", 1e-7, 1e-7],
            [32, 64, 16, "ortho", "lobatto", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "equiangular", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "lobatto", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "equiangular", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "lobatto", 1e-7, 1e-7],
            [32, 64, 16, "ortho", "fejer1", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "fejer1", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "fejer1", 1e-7, 1e-7],
        ],
        skip_on_empty=True,
    )
    def test_gradient_consistency(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        """ivsht([c, 0]) synthesizes the surface gradient ∇_S f of a scalar field
        f = isht(c).  Applying vsht to this gradient field must recover c in the
        spheroidal channel and zero in the toroidal channel, because a gradient
        field is curl-free (purely spheroidal).
        """
        # set seed
        set_seed(333)

        vsht = th.RealVectorSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm=norm).to(self.device)
        ivsht = th.InverseRealVectorSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm=norm).to(self.device)
        lmax, mmax = vsht.lmax, vsht.mmax

        with torch.no_grad():
            c = random_sht_coeffs(batch_size, lmax, mmax, self.device, zero_l0=True)
            zeros = torch.zeros_like(c)

            # synthesize gradient field: ivsht([c, 0]) = ∇_S f
            grad_f = ivsht(torch.stack([c, zeros], dim=-3))  # (batch, 2, nlat, nlon)

            # analyse: vsht(∇_S f) must give [c, 0]
            st = vsht(grad_f)  # (batch, 2, lmax, mmax)
            s = st[..., 0, :, :]  # spheroidal
            t = st[..., 1, :, :]  # toroidal

        self.assertTrue(compare_tensors("spheroidal coefficients", s, c, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("toroidal coefficients", t, zeros, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            [32, 64, 16, "ortho", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "ortho", "equiangular", 1e-7, 1e-7],
            [32, 64, 16, "ortho", "lobatto", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "equiangular", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "lobatto", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "equiangular", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "lobatto", 1e-7, 1e-7],
            [32, 64, 16, "ortho", "fejer1", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "fejer1", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "fejer1", 1e-7, 1e-7],
        ],
        skip_on_empty=True,
    )
    def test_curl_consistency(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        """ivsht([0, c]) synthesizes the surface curl ê_r × ∇_S f of a scalar field
        f = isht(c).  Applying vsht to this curl field must recover c in the
        toroidal channel and zero in the spheroidal channel, because a surface
        curl field is divergence-free (purely toroidal).
        """
        # set seed
        set_seed(333)

        vsht = th.RealVectorSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm=norm).to(self.device)
        ivsht = th.InverseRealVectorSHT(th.as_grid(grid, nlat=nlat, nlon=nlon), norm=norm).to(self.device)
        lmax, mmax = vsht.lmax, vsht.mmax

        with torch.no_grad():
            c = random_sht_coeffs(batch_size, lmax, mmax, self.device, zero_l0=True)
            zeros = torch.zeros_like(c)

            # synthesize curl field: ivsht([0, c]) = ê_r × ∇_S f
            curl_f = ivsht(torch.stack([zeros, c], dim=-3))  # (batch, 2, nlat, nlon)

            # analyse: vsht(ê_r × ∇_S f) must give [0, c]
            st = vsht(curl_f)  # (batch, 2, lmax, mmax)
            s = st[..., 0, :, :]  # spheroidal
            t = st[..., 1, :, :]  # toroidal

        self.assertTrue(compare_tensors("spheroidal coefficients", s, zeros, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("toroidal coefficients", t, c, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(_GRADIENT_CASES, skip_on_empty=True)
    def test_vector_grads(self, nlat, grid, extended, verbose=False):
        """Autograd of vsht and ivsht matches finite differences, as for the scalar transforms."""
        _check_gradients(self, nlat, grid, extended, vector=True)

    @parameterized.expand(_RECOVERY_CASES, skip_on_empty=True)
    def test_vector_forward_inverse(self, nlat, grid, extended, mmax, lmmax, norm, dtype, verbose=False):
        """vsht(ivsht(c)) == c for spheroidal and toroidal coefficients, as for the scalar transform.

        The vector analysis is exact to the same band as the scalar one: with the
        (1 - x^2)^(m/2) factor of P_l^m, the products of dP/dtheta and m P/sin(theta) that it
        integrates are polynomials in cos(theta) of degree l + l', like P_l^m P_l'^m.
        """
        _check_coefficient_recovery(self, nlat, grid, extended, mmax, lmmax, norm, dtype, vector=True, verbose=verbose)

    @parameterized.expand(_GRAM_CASES, skip_on_empty=True)
    def test_vector_orthogonality(self, nlat, grid, norm, verbose=False):
        """ivsht synthesizes mutually orthogonal spheroidal and toroidal fields, with norm l(l+1)
        times that of the scalar harmonic."""
        _check_gram(self, nlat, grid, norm, vector=True, verbose=verbose)

    @parameterized.expand(
        [
            [12, 24, 2, "equiangular"],
            [12, 24, 2, "legendre-gauss"],
            [11, 22, 2, "equiangular"],
            [12, 24, 2, "fejer1"],
        ],
        skip_on_empty=True,
    )
    @requires_torch_compile
    def test_compile(self, nlat, nlon, batch_size, grid, verbose=False):
        """The vector round trip compiles into a single graph and matches eager.

        Same rationale as the scalar case, see
        ``TestSphericalHarmonicTransform.test_compile``.  The vector transforms assemble
        their spheroidal and toroidal components separately, so they exercise a code path
        the scalar test does not reach.
        """

        set_seed(333)

        vsht = th.RealVectorSHT(th.as_grid(grid, nlat=nlat, nlon=nlon)).to(self.device)
        ivsht = th.InverseRealVectorSHT(th.as_grid(grid, nlat=nlat, nlon=nlon)).to(self.device)

        def fn(t):
            return ivsht(vsht(t))

        x = torch.randn(batch_size, 2, nlat, nlon, device=self.device, dtype=torch.float32, requires_grad=True)
        gradient = torch.randn_like(x)

        expected = fn(x)
        (expected_grad,) = torch.autograd.grad(expected, x, grad_outputs=gradient)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        actual = compiled(x)
        (actual_grad,) = torch.autograd.grad(actual, x, grad_outputs=gradient)

        self.assertTrue(compare_tensors("compiled forward", actual, expected, atol=1e-5, rtol=1e-5, verbose=verbose))
        self.assertTrue(compare_tensors("compiled backward", actual_grad, expected_grad, atol=1e-5, rtol=1e-5, verbose=verbose))


@parameterized_class(("device"), _devices)
class TestUnsuitableGridWarning(unittest.TestCase):
    """
    The SHT must object loudly when handed a grid whose quadrature cannot support it.

    ``trapezoidal`` is the only such grid today: it converges as
    :math:`O(h^2)` rather than spectrally, so the associated Legendre polynomials are
    not discretely orthogonal under its quadrature and the transform does not
    round-trip. Its numerical behaviour is deliberately *not* covered here -- it is not
    a supported configuration, and pinning its error would suggest otherwise. What is
    covered is that a user cannot stumble into it silently.
    """

    def setUp(self):
        disable_tf32()

    @parameterized.expand([[cls_name] for cls_name in ["RealSHT", "InverseRealSHT", "RealVectorSHT", "InverseRealVectorSHT"]])
    def test_unsuitable_grid_warns(self, cls_name):
        with self.assertWarns(UserWarning) as ctx:
            getattr(th, cls_name)(th.as_grid("trapezoidal", nlat=32, nlon=64))
        self.assertIn("must not be used", str(ctx.warning))

    @parameterized.expand([[cls_name, grid] for cls_name in ["RealSHT", "InverseRealSHT"] for grid in ["equiangular", "fejer1", "legendre-gauss", "lobatto"]])
    def test_supported_grids_do_not_warn(self, cls_name, grid):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            getattr(th, cls_name)(th.as_grid(grid, nlat=32, nlon=64), lmax=8)


def _transforms(grid, lmax, mmax, vector, **kwargs):
    analysis = th.RealVectorSHT if vector else th.RealSHT
    synthesis = th.InverseRealVectorSHT if vector else th.InverseRealSHT
    return analysis(grid, lmax=lmax, mmax=mmax, **kwargs), synthesis(grid, lmax=lmax, mmax=mmax, **kwargs)


def _scipy_mode_field(grid, l, m, coefficient, vector=False, component=0):
    """Evaluate one normalized positive-m harmonic (or vector derivative) with SciPy."""
    theta = grid.colats.numpy()[:, None]
    phi = grid.lons().numpy()[None, :]
    normalization = np.sqrt((2 * l + 1) / (4 * np.pi)) * np.exp(0.5 * (gammaln(l - m + 1) - gammaln(l + m + 1)))
    p = normalization * lpmv(m, l, np.cos(theta))
    harmonic = p * np.exp(1j * m * phi)
    if not vector:
        basis = harmonic
    else:
        derivative = ((l * np.cos(theta) * p - (l + m) * normalization * lpmv(m, l - 1, np.cos(theta))) / np.sin(theta)) * np.exp(1j * m * phi)
        longitude = 1j * m * harmonic / np.sin(theta)
        basis = np.stack((derivative, longitude) if component == 0 else (longitude, -derivative))
    return (1 if m == 0 else 2) * (coefficient * basis).real


@parameterized_class(("device",), _devices)
class TestSharedFoldedSHTAnalysis(unittest.TestCase):
    """Shared extended-analysis coverage for Clenshaw-Curtis and Fejer-1 grids."""

    @classmethod
    def setUpClass(cls):
        disable_tf32()

    @parameterized.expand(
        [(vector, "equiangular", nlat, 2 * nlat - 1, nlat - 1, nlat - 1, lmmax) for vector in (False, True) for nlat in (8, 9, 34) for lmmax in (None, 3)]
        + [(vector, "fejer1", nlat, 2 * nlat - 1, nlat, nlat, lmmax) for vector in (False, True) for nlat in (8, 9) for lmmax in (None, 3)]
        + [(vector, "equiangular", 257, 33, 256, 16, None) for vector in (False, True)],
        skip_on_empty=True,
    )
    def test_effective_projection(self, vector, grid_type, nlat, nlon, lmax, mmax, lmmax):
        """Compare every row, complex contractions and gradients with runtime folding."""
        set_seed(417)
        grid = th.as_grid(grid_type, nlat=nlat, nlon=nlon)
        analysis, _ = _transforms(grid, lmax, mmax, vector, lmmax=lmmax)
        if grid_type == "fejer1":
            b = {name: value.to(self.device) for name, value in _precompute_f1_resampling(nlat, analysis.mmax, vector).items()}
            native_weights = b["_f1_native_weights"]
            complementary_weights = b["_f1_complementary_weights"]
            phase = torch.complex(*b["_f1_phase"])
            parity = b["_f1_parity"]
            include_poles = False
        else:
            b = {name: value.to(self.device) for name, value in _precompute_cc_resampling(nlat, analysis.mmax, vector).items()}
            native_weights = b["_cc_weights"]
            complementary_weights = b["_cc_midpoint_weights"]
            phase = torch.complex(*b["_cc_phase"])
            parity = b["_cc_parity"]
            include_poles = True
        if vector:
            basis = _precompute_dlegpoly(analysis.mmax, analysis.lmax, grid, truncation=analysis.grid_out)
            degrees = torch.arange(analysis.lmax, dtype=torch.float64)
            basis *= (1 / (degrees * (degrees + 1)).clamp(min=1))[None, None, :, None]
            basis[1].neg_()
        else:
            basis = _precompute_legpoly(analysis.mmax, analysis.lmax, grid, truncation=analysis.grid_out)
        basis = basis.to(self.device)
        folded = _fold_resampled_latitude(basis.transpose(-3, -2), parity, native_weights, complementary_weights, phase, include_poles)
        self.assertLess(folded.imag.abs().max().item(), 1e-13)
        effective = analysis.weights.to(self.device)
        torch.testing.assert_close(effective, folded.real.transpose(-3, -2), atol=1e-13, rtol=1e-13)
        # Local order offsets must use global parity, including odd starting orders.
        for mmin in (1, 2, analysis.mmax):
            local = _precompute_folded_projection(grid, analysis.grid_out, "ortho", True, vector, mmin=mmin, kmin=2, kmax=nlat - 1)
            torch.testing.assert_close(local.to(self.device), effective[..., mmin:, :, 2:-1], atol=1e-13, rtol=1e-13)
        x = torch.randn(2, analysis.mmax, nlat, device=self.device, dtype=torch.complex128, requires_grad=True)
        folded_x = _fold_resampled_latitude(x, parity, native_weights, complementary_weights, phase, include_poles)
        equation = "...mk,dmlk->...dlm" if vector else "...mk,mlk->...lm"
        expected = torch.einsum(equation, folded_x, basis.to(x.dtype))
        actual = torch.einsum(equation, x, effective.to(x.dtype))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        gradient = torch.randn_like(actual)
        expected_grad = torch.autograd.grad(expected, x, gradient, retain_graph=True)[0]
        for _ in range(2):
            actual_grad = torch.autograd.grad(actual, x, gradient, retain_graph=True)[0]
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)
        self.assertEqual(list(dict(analysis.named_buffers())), ["weights"])
        # A warm constructor must reuse numerical work without sharing mutable buffers.
        another, _ = _transforms(grid, analysis.lmax, analysis.mmax, vector, lmmax=lmmax)
        torch.testing.assert_close(analysis.weights, another.weights, atol=0, rtol=0)
        self.assertNotEqual(analysis.weights.data_ptr(), another.weights.data_ptr())
        another.weights.zero_()
        self.assertGreater(analysis.weights.abs().max().item(), 0)
        self.assertEqual(_precompute_folded_projection.cache_parameters()["maxsize"], 2)

    @parameterized.expand(
        [
            (vector, l, m, component, norm, csphase)
            for vector in (False, True)
            for l, m in [(1, 0), (35, 0), (35, 7), (70, 69), (71, 1), (71, 35), (71, 71)]
            for component in (range(2) if vector else range(1))
            for norm, csphase in ([("ortho", True)] if (l, m) != (71, 35) else [(norm, csphase) for norm in ("ortho", "four-pi", "schmidt", "unnorm") for csphase in (True, False)])
        ],
        skip_on_empty=True,
    )
    def test_f1_scipy_harmonics_through_maximum_degree(self, vector, l, m, component, norm, csphase, verbose=False):
        """Analyze independent SciPy harmonics, including F1 sectoral T71 modes."""
        grid = th.as_grid("fejer1", nlat=72, nlon=144)
        coefficient = 0.7 + (0.3j if m else 0j)
        spatial = torch.tensor(_scipy_mode_field(grid, l, m, coefficient, vector, component), dtype=torch.float64, device=self.device)
        analysis = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=72, mmax=72, norm=norm, csphase=csphase).to(self.device, torch.float64)
        actual = analysis(spatial)
        expected = torch.zeros_like(actual)
        factor = 1.0 if norm == "ortho" else np.sqrt(4 * np.pi)
        if norm == "schmidt":
            factor /= np.sqrt(2 * l + 1)
        if not csphase:
            factor *= (-1) ** m
        if vector:
            expected[component, l, m] = factor * coefficient
        else:
            expected[l, m] = factor * coefficient
        self.assertTrue(compare_tensors("SciPy F1 harmonic", expected, actual, atol=3e-11, rtol=3e-11, verbose=verbose))
        self.assertLess((actual - expected).abs().max().item(), 3e-11)
        synthesis = (th.InverseRealVectorSHT if vector else th.InverseRealSHT)(grid, lmax=72, mmax=72, norm=norm, csphase=csphase).to(self.device, torch.float64)
        synthesized = synthesis(expected)
        self.assertTrue(compare_tensors("SciPy F1 synthesis", spatial, synthesized, atol=3e-11, rtol=3e-11, verbose=verbose))
        self.assertLess((synthesized - spatial).abs().max().item(), 3e-11)

    @parameterized.expand(
        [
            (vector, l, m, component, norm, csphase)
            for vector in (False, True)
            for l, m in [(70, 0), (70, 69), (71, 0), (71, 1), (71, 35), (71, 70), (71, 71)]
            for component in (range(2) if vector else range(1))
            for norm, csphase in [("ortho", True), ("four-pi", False), ("schmidt", False), ("unnorm", True)]
        ],
        skip_on_empty=True,
    )
    def test_independent_harmonics(self, vector, l, m, component, norm, csphase, verbose=False):
        """Analyze SciPy harmonics and derivatives without using our inverse transform."""
        grid = th.as_grid("equiangular", nlat=73, nlon=144)
        theta = grid.colats.numpy()[:, None]
        phi = grid.lons().numpy()[None, :]
        # SciPy's associated Legendre function uses its own implementation.
        normalization = np.sqrt((2 * l + 1) / (4 * np.pi)) * np.exp(0.5 * (gammaln(l - m + 1) - gammaln(l + m + 1)))
        p = normalization * lpmv(m, l, np.cos(theta))
        y = p * np.exp(1j * m * phi)
        derivative = np.zeros_like(y)
        derivative[1:-1] = ((l * np.cos(theta[1:-1]) * p[1:-1] - (l + m) * normalization * lpmv(m, l - 1, np.cos(theta[1:-1]))) / np.sin(theta[1:-1])) * np.exp(1j * m * phi)
        if m == 1:
            derivative[0] = -0.5 * l * (l + 1) * normalization * np.exp(1j * phi)
            derivative[-1] = (-1) ** (l + 1) * 0.5 * l * (l + 1) * normalization * np.exp(1j * phi)
        c = 0.7 + (0.3j if m else 0j)
        if vector:
            longitude = np.zeros_like(y)
            longitude[1:-1] = 1j * m * y[1:-1] / np.sin(theta[1:-1])
            if m == 1:
                longitude[[0, -1]] = 1j * derivative[[0, -1]] / np.cos(theta[[0, -1]])
            basis = np.stack((derivative, longitude) if component == 0 else (longitude, -derivative))
        else:
            basis = y
        field = torch.tensor((1 if m == 0 else 2) * (c * basis).real, device=self.device)
        analysis = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=72, mmax=72, norm=norm, csphase=csphase).to(self.device)
        actual = analysis(field)
        expected = torch.zeros_like(actual)
        factor = 1.0 if norm == "ortho" else np.sqrt(4 * np.pi)
        if norm == "schmidt":
            factor /= np.sqrt(2 * l + 1)
        if not csphase:
            factor *= (-1) ** m
        if vector:
            expected[component, l, m] = factor * c
        else:
            expected[l, m] = factor * c
        self.assertTrue(compare_tensors("SciPy harmonic analysis", expected, actual, atol=3e-12, rtol=3e-12, verbose=verbose))
        self.assertLess((actual - expected).abs().max().item(), 3e-12)

    def test_f1_extended_selection_and_oversized_bounds(self):
        for nlat, nlon in [(8, 16), (9, 19)]:
            grid = th.as_grid("fejer1", nlat=nlat, nlon=nlon)
            supported = th.RealSHT(grid, lmax=nlat, mmax=nlat)
            self.assertTrue(supported._extended_folded)
            self.assertEqual(supported.grid_out, th.SpectralGrid(nlat, nlat))
            oversized = th.RealSHT(grid, lmax=nlat + 1, mmax=nlat + 1)
            self.assertFalse(oversized._extended_folded)
            self.assertEqual(oversized.grid_out, th.SpectralGrid(nlat + 1, nlat + 1))
        # With even nlon=2*nlat-2, lmax=nlat is latitude-supported but mmax
        # would include the unrecoverable complex longitude Nyquist order.
        nyquist_limited = th.RealSHT(th.as_grid("fejer1", nlat=8, nlon=14), lmax=8, mmax=8)
        self.assertFalse(nyquist_limited._extended_folded)

    @parameterized.expand(
        [
            (vector, dtype, *case)
            for vector in (False, True)
            for dtype in (torch.float32, torch.float64)
            for case in [
                (32, 64, 64, 64, None),
                (33, 64, 65, 65, None),
                (32, 64, 64, None, 11),
                (33, 64, 65, 65, 20),
                (73, 144, 73, 72, 18),
                (25, 32, 24, None, None),
                (25, 32, 24, 17, 6),
                (25, 32, 24, 18, None),
                (73, 144, 74, None, None),
            ]
        ],
        skip_on_empty=True,
    )
    def test_oversized_dimensions(self, vector, dtype, nlat, nlon, lmax, mmax, lmmax, verbose=False):
        set_seed(333)
        grid = th.as_grid("equiangular", nlat=nlat, nlon=nlon)
        analysis = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=lmax, mmax=mmax, lmmax=lmmax)
        self.assertFalse(analysis._extended_folded)
        self.assertEqual(analysis.grid_out, th.SpectralGrid(lmax, min(lmax, nlon // 2 + 1) if mmax is None else mmax, lmmax))
        # Native projection formula, with the degree factor in float64 like the transform.
        q = 2 * torch.pi * grid.colat_weights
        if vector:
            p = _precompute_dlegpoly(analysis.mmax, analysis.lmax, grid, truncation=analysis.grid_out)
            l = torch.arange(analysis.lmax, dtype=torch.float64)
            factor = 1.0 / l / (l + 1)
            factor[0] = 1
            weights = torch.einsum("dmlk,k,l->dmlk", p, q, factor).contiguous()
            weights[1] *= -1
        else:
            p = _precompute_legpoly(analysis.mmax, analysis.lmax, grid, truncation=analysis.grid_out)
            weights = torch.einsum("mlk,k->mlk", p, q).contiguous()
        analysis = analysis.to(device=self.device, dtype=dtype)
        self.assertTrue(compare_tensors("oversized native weights", weights.to(device=self.device, dtype=dtype), analysis.weights, atol=0, rtol=0, verbose=verbose))
        x = torch.randn((2,) + grid.shape if vector else grid.shape, device=self.device, dtype=dtype, requires_grad=True)
        modes = torch.fft.rfft(x, dim=-1, norm="forward")
        if analysis.mmax > modes.shape[-1]:
            modes = torch.nn.functional.pad(modes, (0, analysis.mmax - modes.shape[-1]))
        modes = modes[..., : analysis.mmax].transpose(-1, -2)
        # Evaluate the discrete quadrature in complex arithmetic independently of
        # the implementation's separate real/imaginary contractions.
        w = weights.to(device=self.device, dtype=modes.dtype)
        if vector:
            s0 = torch.einsum("mk,mlk->lm", modes[0], w[0]) + 1j * torch.einsum("mk,mlk->lm", modes[1], w[1])
            t0 = 1j * torch.einsum("mk,mlk->lm", modes[0], w[1]) - torch.einsum("mk,mlk->lm", modes[1], w[0])
            expected = torch.stack((s0, t0))
        else:
            expected = torch.einsum("mk,mlk->lm", modes, w)
        actual = analysis(x)
        if mmax is None:
            explicit = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=lmax, mmax=analysis.mmax, lmmax=lmmax).to(device=self.device, dtype=dtype)
            self.assertFalse(explicit._extended_folded)
            self.assertTrue(compare_tensors("inferred and explicit orders", actual, explicit(x), atol=0, rtol=0, verbose=verbose))
        self.assertEqual(actual.shape, ((2,) if vector else ()) + analysis.grid_out.shape)
        tol = 1e-5 if dtype == torch.float32 else 1e-12
        self.assertTrue(compare_tensors("oversized direct quadrature", expected, actual, atol=tol, rtol=tol, verbose=verbose))
        gradient = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, x, gradient)[0]
        expected_grad = torch.autograd.grad(expected, x, gradient)[0]
        self.assertTrue(compare_tensors("oversized input gradient", expected_grad, actual_grad, atol=tol, rtol=tol, verbose=verbose))
        l = torch.arange(analysis.lmax, device=self.device)[:, None]
        m = torch.arange(analysis.mmax, device=self.device)[None, :]
        mask = m <= l
        if lmmax is not None:
            mask &= l - m < lmmax
        self.assertTrue(compare_tensors("oversized mask", actual[..., ~mask], torch.zeros_like(actual[..., ~mask]), atol=0, rtol=0, verbose=verbose))

    @parameterized.expand(
        [
            ("equiangular", vector, dtype, nlat, profile)
            for vector in (False, True)
            for dtype in (torch.float32, torch.float64)
            for nlat in (8, 9, 73, 74)
            for profile in ("spatial", "latitude", "localized", "alternating")
        ]
        + [
            ("fejer1", vector, dtype, nlat, profile)
            for vector in (False, True)
            for dtype in (torch.float32, torch.float64)
            for nlat in (8, 9)
            for profile in ("spatial", "latitude", "localized", "alternating")
        ],
        skip_on_empty=True,
    )
    def test_arbitrary_input_zero_order_reality(self, geometry, vector, dtype, nlat, profile, verbose=False):
        set_seed(333)
        nlon = 2 * (nlat - 1) if geometry == "equiangular" else 2 * nlat
        grid = th.as_grid(geometry, nlat=nlat, nlon=nlon)
        lmax = nlat - 1 if geometry == "equiangular" else nlat
        analysis = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=lmax, mmax=lmax).to(device=self.device, dtype=dtype)
        self.assertTrue(analysis._extended_folded)
        leading = (2,) if vector else ()
        if profile == "spatial":
            x = torch.randn(leading + grid.shape, device=self.device, dtype=dtype)
        else:
            if profile == "latitude":
                latitude = torch.randn(leading + (nlat,), device=self.device, dtype=dtype)
            elif profile == "localized":
                theta = grid.colats.to(device=self.device, dtype=dtype)
                latitude = torch.exp(-(((theta - 0.27 * torch.pi) / 0.2) ** 2))
            else:
                latitude = (1 - 2 * torch.arange(nlat, device=self.device).remainder(2)).to(dtype)
            x = latitude[..., None].expand(leading + grid.shape).contiguous()
        actual = analysis(x)
        tol = 64 * torch.finfo(dtype).eps * max(1.0, actual.abs().max().item())
        self.assertTrue(compare_tensors("arbitrary-input zero-order reality", actual[..., 0].imag, torch.zeros_like(actual[..., 0].imag), atol=tol, rtol=0, verbose=verbose))

    @parameterized.expand(
        [(geometry, vector, lmax) for geometry in ("equiangular", "fejer1") for vector in (False, True) for lmax in (0, 8)],
        skip_on_empty=True,
    )
    def test_empty_analysis_synthesis(self, geometry, vector, lmax):
        grid = th.as_grid(geometry, nlat=9, nlon=16)
        analysis, synthesis = _transforms(grid, lmax, 0, vector)
        analysis, synthesis = analysis.to(self.device), synthesis.to(self.device)
        x = torch.randn(((2,) if vector else ()) + grid.shape, device=self.device, dtype=torch.float64, requires_grad=True)
        coefficients = analysis(x)
        actual = synthesis(coefficients)
        torch.testing.assert_close(actual, torch.zeros_like(x), atol=0, rtol=0)
        actual.sum().backward()
        torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0, rtol=0)

    @parameterized.expand([(geometry, vector) for geometry in ("equiangular", "fejer1") for vector in (False, True)], skip_on_empty=True)
    def test_empty_orders(self, geometry, vector, verbose=False):
        set_seed(333)
        grid = th.as_grid(geometry, nlat=9, nlon=16)
        analysis, _ = _transforms(grid, 8, 0, vector)
        analysis = analysis.to(self.device)
        x = torch.randn((2,) + grid.shape if vector else grid.shape, device=self.device, requires_grad=True)
        result = analysis(x)
        expected = torch.zeros((2, 8, 0) if vector else (8, 0), device=self.device, dtype=torch.complex64)
        self.assertTrue(compare_tensors("empty spectrum", expected, result, atol=0, rtol=0, verbose=verbose))
        result.real.sum().backward()
        self.assertTrue(compare_tensors("empty gradient", torch.zeros_like(x), x.grad, atol=0, rtol=0, verbose=verbose))

    @parameterized.expand(
        [
            (vector, dtype, grid_type, nlat, nlon, lmax)
            for vector in (False, True)
            for dtype in (torch.float32, torch.float64)
            for grid_type, nlat, nlon, lmax in [
                ("equiangular", 9, 16, 8),
                ("equiangular", 73, 144, 72),
                ("fejer1", 9, 18, 9),
                ("fejer1", 72, 144, 72),
            ]
        ],
        skip_on_empty=True,
    )
    @requires_torch_compile
    def test_compile(self, vector, dtype, grid_type, nlat, nlon, lmax, verbose=False):
        set_seed(333)
        grid = th.as_grid(grid_type, nlat=nlat, nlon=nlon)
        analysis, synthesis = _transforms(grid, lmax, lmax, vector)
        analysis = analysis.to(device=self.device, dtype=dtype)
        synthesis = synthesis.to(device=self.device, dtype=dtype)

        def fn(t):
            return synthesis(analysis(t))

        x = torch.randn((2, 2) + grid.shape if vector else (2,) + grid.shape, device=self.device, dtype=dtype, requires_grad=True)
        gradient = torch.randn_like(x)
        expected = fn(x)
        expected_grad = torch.autograd.grad(expected, x, gradient)[0]
        actual = torch.compile(fn, fullgraph=True, dynamic=False)(x)
        actual_grad = torch.autograd.grad(actual, x, gradient)[0]
        tolerance = 1e-5 if dtype == torch.float32 else 1e-12
        self.assertTrue(compare_tensors("compiled forward", expected, actual, atol=tolerance, rtol=tolerance, verbose=verbose))
        self.assertTrue(compare_tensors("compiled backward", expected_grad, actual_grad, atol=tolerance, rtol=tolerance, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
