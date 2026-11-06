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

import gc
import math
import unittest
import warnings
import weakref

import numpy as np
import torch
from parameterized import parameterized, parameterized_class
from scipy.special import gammaln, lpmv
from testutils import compare_tensors, range_atol, requires_torch_compile, set_seed
from torch.autograd import gradcheck

import torch_harmonics as th
from torch_harmonics.fft import RingFFT
from torch_harmonics.healpix import healpix_ring_weights
from torch_harmonics.legendre import _precompute_dlegpoly, _precompute_legpoly, dlegpoly, legpoly
from torch_harmonics.quadrature import precompute_latitudes
from torch_harmonics.radial import LegendreGaussRadialGrid, SphericalProductGrid

# healpy is in the dev extra, so CI runs the HEALPix cross-checks; a local install without
# it skips them
try:
    import healpy
except ImportError:
    healpy = None

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

# The spectral truncations truncate_sht supports, as lmax, mmax, lmmax. The capped variants
# clip the rhomboid with lmax.
_TRUNCATION_SHAPES = {
    "triangular": (16, 16, None),
    "trapezoidal": (20, 9, None),
    "rhomboidal": (17, 9, 9),
    "rhomboidal_capped": (12, 9, 9),
    "pentagonal": (14, 9, 6),
    "r42": (85, 43, 43),
    "r42_capped": (64, 43, 43),
}

# nlat, grid, extended, mmax (None: lmax), lmmax, norm, dtype
_RECOVERY_CASES = (
    [[nlat, grid, extended, None, None, norm, torch.float64] for nlat in (32, 33) for grid, extended in _RECOVERY_BANDS for norm in ("ortho", "four-pi", "schmidt")]
    + [[nlat, grid, extended, None, None, "ortho", torch.float32] for nlat in (32, 33) for grid, extended in _RECOVERY_BANDS]
    # fewer orders and an l - m bound on top of the extended band
    + [[33, "equiangular", True, None, 8, "four-pi", torch.float64], [32, "fejer1", True, 12, 6, "schmidt", torch.float64]]
    # every truncation shape at the edge of the Gauss band, which is nlat
    + [[lmax, "legendre-gauss", False, mmax, lmmax, "ortho", dtype] for lmax, mmax, lmmax in _TRUNCATION_SHAPES.values() for dtype in (torch.float64, torch.float32)]
)


def _band(grid, nlat, extended):
    """The descriptor of an nlat x 2 nlat grid and the degree bound of its native or extended band."""
    descriptor = th.as_grid(grid, nlat=nlat, nlon=2 * nlat)
    return descriptor, _EXTENDED_DEGREE[grid](nlat) if extended else descriptor.max_exact_degree


# Error budgets in ULPs of the largest reference entry (see testutils.range_atol): every
# check below is a quadrature or Legendre contraction, whose rounding error scales with
# the size of the summed terms rather than with the entry it lands on.
# Round trip, measured 3 / 13 ULPs (float32 scalar / vector) and 954 / 3856 (float64). In
# float64 the precomputed nodes, weights and Legendre recursions, accurate to ~1e-13, dominate
# rather than the transform's own rounding, so the budget is still under 4e-12 relative.
_ROUNDTRIP_ULPS = {(torch.float32, False): 16, (torch.float32, True): 64, (torch.float64, False): 4096, (torch.float64, True): 16384}
# Gram: float64 inner products over the nlat * nlon grid, ~sqrt(N) growth (measured: 36).
_GRAM_ULPS = 128
# compile vs eager: the same contraction, possibly reassociated by inductor (measured: 0).
_COMPILE_ULPS = 64
# oversized analysis vs an einsum over the latitudes (measured: 1.5).
_QUADRATURE_ULPS = 16
# four-pi and schmidt weights are separate float64 recursions from ortho (measured: 13).
_NORM_ULPS = 64


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

    ulps = _ROUNDTRIP_ULPS[(dtype, vector)]
    test.assertTrue(compare_tensors(f"recovered coefficients [{dtype}, vector={vector}]", recovered, coeffs, atol=0.0, rtol=0.0, ulps=ulps, verbose=verbose))


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


def _transforms(grid, lmax, mmax, vector, **kwargs):
    analysis = th.RealVectorSHT if vector else th.RealSHT
    synthesis = th.InverseRealVectorSHT if vector else th.InverseRealSHT
    return analysis(grid, lmax=lmax, mmax=mmax, **kwargs), synthesis(grid, lmax=lmax, mmax=mmax, **kwargs)


# nlat, nlon, grid, lmax (None: the grid's default band), dtype
_COMPILE_CASES = [
    [12, 24, "equiangular", None, torch.float32],
    [12, 24, "legendre-gauss", None, torch.float32],
    [11, 22, "equiangular", None, torch.float32],
    [12, 24, "fejer1", None, torch.float32],
] + [
    # the extended band runs the folded analysis
    [nlat, nlon, grid, lmax, dtype]
    for grid, nlat, nlon, lmax in [("equiangular", 9, 16, 8), ("equiangular", 73, 144, 72), ("fejer1", 9, 18, 9), ("fejer1", 72, 144, 72)]
    for dtype in (torch.float32, torch.float64)
]


def _check_compile(test, nlat, nlon, grid, lmax, dtype, vector, verbose=False):
    """Compile the round trip as one graph, forward and backward, and compare it to eager."""
    set_seed(333)
    descriptor = th.as_grid(grid, nlat=nlat, nlon=nlon)
    forward, inverse = _transforms(descriptor, lmax, lmax, vector)
    forward = forward.to(device=test.device, dtype=dtype)
    inverse = inverse.to(device=test.device, dtype=dtype)

    def fn(t):
        return inverse(forward(t))

    x = torch.randn((2, 2) + descriptor.shape if vector else (2,) + descriptor.shape, device=test.device, dtype=dtype, requires_grad=True)
    gradient = torch.randn_like(x)

    expected = fn(x)
    (expected_grad,) = torch.autograd.grad(expected, x, grad_outputs=gradient)

    compiled = torch.compile(fn, fullgraph=True, dynamic=False)
    actual = compiled(x)
    (actual_grad,) = torch.autograd.grad(actual, x, grad_outputs=gradient)

    test.assertTrue(compare_tensors("compiled forward", actual, expected, atol=0.0, rtol=0.0, ulps=_COMPILE_ULPS, verbose=verbose))
    test.assertTrue(compare_tensors("compiled backward", actual_grad, expected_grad, atol=0.0, rtol=0.0, ulps=_COMPILE_ULPS, verbose=verbose))


def _scipy_mode_field(grid, l, m, coefficient, vector=False, component=0):
    """Evaluate one normalized positive-m harmonic, or its gradient (component 0) or curl
    (component 1), with SciPy, independently of our Legendre tables and FFTs.

    Vector components are ordered (colatitude, longitude). At a pole, where sin(theta)
    vanishes, the derivatives take their limits: zero unless m = 1.
    """
    theta = grid.colats.numpy()[:, None]
    phi = grid.lons().numpy()[None, :]
    normalization = np.sqrt((2 * l + 1) / (4 * np.pi)) * np.exp(0.5 * (gammaln(l - m + 1) - gammaln(l + m + 1)))
    p = normalization * lpmv(m, l, np.cos(theta))
    harmonic = p * np.exp(1j * m * phi)
    if not vector:
        basis = harmonic
    else:
        sin_theta = np.sin(theta)
        interior = np.abs(sin_theta) > 1e-6
        safe_sin = np.where(interior, sin_theta, 1.0)
        derivative = np.where(interior, (l * np.cos(theta) * p - (l + m) * normalization * lpmv(m, l - 1, np.cos(theta))) / safe_sin, 0.0) * np.exp(1j * m * phi)
        longitude = np.where(interior, 1j * m * harmonic / safe_sin, 0.0)
        if m == 1:
            # d_theta P_l^1 -> -l(l+1)/2 at the north pole and +(-1)^(l+1) l(l+1)/2 at the south pole
            pole = 0.5 * l * (l + 1) * normalization * np.where(np.cos(theta) > 0, -1.0, (-1.0) ** (l + 1)) * np.exp(1j * phi)
            derivative = np.where(interior, derivative, pole)
            longitude = np.where(interior, longitude, 1j * pole / np.cos(theta))
        basis = np.stack((derivative, longitude) if component == 0 else (longitude, -derivative))
    return (1 if m == 0 else 2) * (coefficient * basis).real


# grid, extended, nlat: every band of _RECOVERY_BANDS, near degree 72 where the grid allows it
_SCIPY_BANDS = [("legendre-gauss", False, 72), ("lobatto", False, 73), ("equiangular", False, 73), ("fejer1", False, 72), ("equiangular", True, 73), ("fejer1", True, 72)]
_SCIPY_NORMS = [(norm, csphase) for norm in ("ortho", "four-pi", "schmidt", "unnorm") for csphase in (True, False)]


def _scipy_modes(lmax):
    """Low degrees, a mid-band order and the edge of the band: zonal, tesseral and sectoral."""
    top, mid = lmax - 1, lmax // 2
    return [(0, 0), (1, 0), (1, 1), (mid, 0), (mid, 7), (top - 1, top - 2), (top, 0), (top, 1), (top, mid), (top, top)]


# grid, extended, nlat, vector, l, m, component, norm, csphase; every normalization on (1, 0) and (top, mid)
_SCIPY_CASES = [
    [grid, extended, nlat, vector, l, m, component, norm, csphase]
    for grid, extended, nlat in _SCIPY_BANDS
    for vector in (False, True)
    for l, m in _scipy_modes(_band(grid, nlat, extended)[1])
    if not (vector and l == 0)
    for component in ((0, 1) if vector else (0,))
    for norm, csphase in (_SCIPY_NORMS if (l, m) in ((1, 0), (_band(grid, nlat, extended)[1] - 1, _band(grid, nlat, extended)[1] // 2)) else [("ortho", True)])
]


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
    test.assertTrue(compare_tensors("Gram matrix", gram, expected, atol=0.0, rtol=0.0, ulps=_GRAM_ULPS, eps_dtype=gram.dtype, verbose=verbose))


@parameterized_class(("device"), _devices)
class TestSphericalHarmonicTransform(unittest.TestCase):
    """Test the spherical harmonic transform (CPU/CUDA if available)."""

    @parameterized.expand(_SCIPY_CASES, skip_on_empty=True)
    def test_scipy_harmonics(self, grid, extended, nlat, vector, l, m, component, norm, csphase, verbose=False):
        """Analysis and synthesis agree with harmonics, gradients and curls evaluated by SciPy.

        The reference fields come from SciPy's associated Legendre functions with exp(i m phi)
        evaluated directly, so they share neither our Legendre tables nor our FFTs. This pins
        down what recovery and the Gram matrix cannot: the absolute normalization of every
        convention, the Condon-Shortley phase, and for vector fields which component is the
        gradient and which the curl, their signs and the order of their components.
        """
        descriptor, lmax = _band(grid, nlat, extended)
        coefficient = 0.7 + (0.3j if m else 0j)
        spatial = torch.tensor(_scipy_mode_field(descriptor, l, m, coefficient, vector, component), dtype=torch.float64, device=self.device)
        forward_cls, inverse_cls = (th.RealVectorSHT, th.InverseRealVectorSHT) if vector else (th.RealSHT, th.InverseRealSHT)
        forward, inverse = (cls(descriptor, lmax=lmax, mmax=lmax, norm=norm, csphase=csphase).to(self.device, torch.float64) for cls in (forward_cls, inverse_cls))

        factor = 1.0 if norm == "ortho" else np.sqrt(4 * np.pi)
        if norm == "schmidt":
            factor /= np.sqrt(2 * l + 1)
        if not csphase:
            factor *= (-1) ** m
        expected = torch.zeros((2, lmax, lmax) if vector else (lmax, lmax), dtype=torch.complex128, device=self.device)
        expected[(component, l, m) if vector else (l, m)] = factor * coefficient

        self.assertTrue(compare_tensors("SciPy harmonic analysis", forward(spatial), expected, atol=3e-11, rtol=0.0, verbose=verbose))
        self.assertTrue(compare_tensors("SciPy harmonic synthesis", inverse(expected), spatial, atol=3e-11, rtol=0.0, verbose=verbose))

    @parameterized.expand(_RECOVERY_CASES, skip_on_empty=True)
    def test_forward_inverse(self, nlat, grid, extended, mmax, lmmax, norm, dtype, verbose=False):
        """sht(isht(c)) == c at the largest band each grid recovers.

        This implies the spatial round trip isht(sht(f)) == f for every band-limited f, and
        compares the coefficients themselves, so an error in a single mode cannot hide in
        the synthesized field. The extended rows exercise the folded analysis of the
        equiangular and Fejer-1 grids beyond their native band.
        """
        _check_coefficient_recovery(self, nlat, grid, extended, mmax, lmmax, norm, dtype, vector=False, verbose=verbose)

    @parameterized.expand([[shape, vector] for shape in _TRUNCATION_SHAPES for vector in (False, True)], skip_on_empty=True)
    def test_truncation_is_a_mask(self, shape, vector, verbose=False):
        """A truncated transform is the untruncated one with the coefficients outside the shape zeroed.

        test_forward_inverse recovers band-limited coefficients in every shape. This checks
        the rest of the contract on arbitrary inputs: synthesis ignores coefficients outside
        the shape, analysis of any field, band-limited or not, equals the untruncated analysis
        inside the shape and is exactly zero outside it, and the same holds for the gradients.
        """
        set_seed(333)
        lmax, mmax, lmmax = _TRUNCATION_SHAPES[shape]
        nlat = max(32, lmax + 1)
        grid = th.as_grid("legendre-gauss", nlat=nlat, nlon=2 * nlat)
        l = torch.arange(lmax, device=self.device)[:, None]
        m = torch.arange(mmax, device=self.device)[None, :]
        keep = m <= l
        if lmmax is not None:
            keep &= l - m < lmmax

        forward_cls, inverse_cls = (th.RealVectorSHT, th.InverseRealVectorSHT) if vector else (th.RealSHT, th.InverseRealSHT)
        forward, inverse = (cls(grid, lmax=lmax, mmax=mmax, lmmax=lmmax).to(device=self.device, dtype=torch.float64) for cls in (forward_cls, inverse_cls))
        forward_ref, inverse_ref = (cls(grid, lmax=lmax, mmax=mmax).to(device=self.device, dtype=torch.float64) for cls in (forward_cls, inverse_cls))

        # coefficients everywhere, also outside the shape
        coeffs = torch.randn((2, 2, lmax, mmax) if vector else (2, lmax, mmax), dtype=torch.complex128, device=self.device)
        coeffs[..., 0] = coeffs[..., 0].real
        coeffs.requires_grad_(True)
        ref_coeffs = coeffs.detach().clone().requires_grad_(True)
        signal = inverse(coeffs)
        ref_signal = inverse_ref(ref_coeffs * keep)
        self.assertTrue(compare_tensors("truncated synthesis", signal, ref_signal, atol=1e-13, rtol=1e-13, verbose=verbose))
        spatial_grad = torch.randn_like(signal)
        signal.backward(spatial_grad)
        ref_signal.backward(spatial_grad)
        self.assertTrue(compare_tensors("truncated synthesis gradient", coeffs.grad, ref_coeffs.grad, atol=1e-13, rtol=1e-13, verbose=verbose))

        # an arbitrary field, not band-limited
        x = torch.randn_like(signal, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_(True)
        output = forward(x)
        ref_output = forward_ref(ref_x) * keep
        self.assertTrue(compare_tensors("truncated analysis", output, ref_output, atol=1e-13, rtol=1e-13, verbose=verbose))
        self.assertTrue(torch.equal(output[..., ~keep], torch.zeros_like(output[..., ~keep])))
        spectral_grad = torch.randn_like(output)
        output.backward(spectral_grad)
        ref_output.backward(spectral_grad)
        self.assertTrue(compare_tensors("truncated analysis gradient", x.grad, ref_x.grad, atol=1e-13, rtol=1e-13, verbose=verbose))

    @parameterized.expand(_GRADIENT_CASES, skip_on_empty=True)
    def test_grads(self, nlat, grid, extended, verbose=False):
        """Autograd of sht and isht matches finite differences of the transforms themselves.

        The normalization only rescales the weight buffer, not the backward path, so it is
        not varied here; test_forward_inverse covers every normalization.
        """
        _check_gradients(self, nlat, grid, extended, vector=False)

    @parameterized.expand(
        [
            [32, 64, 16, "equiangular"],
            [32, 64, 16, "legendre-gauss"],
            [32, 64, 16, "lobatto"],
            [32, 64, 16, "fejer1"],
        ],
        skip_on_empty=True,
    )
    def test_cross_norm_consistency(self, nlat, nlon, batch_size, grid, verbose=False):
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
        self.assertTrue(compare_tensors("four-pi vs ortho scaling", c_four_pi, c_four_pi_ref, atol=0.0, rtol=0.0, ulps=_NORM_ULPS, verbose=verbose))

        # schmidt: coefficient at degree l is sqrt(4*pi/(2*l+1)) * c_ortho[l,m]
        l_vals = torch.arange(lmax, dtype=torch.float64, device=self.device)
        schmidt_scale = torch.sqrt(4.0 * math.pi / (2.0 * l_vals + 1.0))  # (lmax,)
        c_schmidt_ref = c_ortho * schmidt_scale[:, None]  # broadcast over mmax
        self.assertTrue(compare_tensors("schmidt vs ortho scaling", c_schmidt, c_schmidt_ref, atol=0.0, rtol=0.0, ulps=_NORM_ULPS, verbose=verbose))

    @parameterized.expand(
        [
            [32, 64, 16, "ortho", "equiangular"],
            [32, 64, 16, "ortho", "legendre-gauss"],
            [32, 64, 16, "ortho", "lobatto"],
            [32, 64, 16, "four-pi", "equiangular"],
            [32, 64, 16, "four-pi", "legendre-gauss"],
            [32, 64, 16, "four-pi", "lobatto"],
            [32, 64, 16, "schmidt", "equiangular"],
            [32, 64, 16, "schmidt", "legendre-gauss"],
            [32, 64, 16, "schmidt", "lobatto"],
            [32, 64, 16, "ortho", "fejer1"],
            [32, 64, 16, "four-pi", "fejer1"],
            [32, 64, 16, "schmidt", "fejer1"],
        ],
        skip_on_empty=True,
    )
    def test_csphase(self, nlat, nlon, batch_size, norm, grid, verbose=False):
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
                atol=0.0,
                rtol=0.0,
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

    @parameterized.expand(_COMPILE_CASES, skip_on_empty=True)
    @requires_torch_compile
    def test_compile(self, nlat, nlon, grid, lmax, dtype, verbose=False):
        """The scalar round trip compiles into a single graph and matches eager.

        The round trip is compiled as one function so the complex spectral coefficients
        are an *intermediate* buffer rather than a graph output -- that is the case
        inductor has to generate code for.  Triton has no complex type, so any pointwise
        kernel over a complex buffer fails codegen with ``KeyError: 'complex64'``; on CPU
        the C++ backend is used instead, where the exposure is a layout mismatch on
        ``aten.complex`` caught by ``assert_size_stride``.  Both are worth covering, which
        the CPU/CUDA parameterization of this class does. The extended rows compile the
        folded analysis of the equiangular and Fejer-1 grids.

        Backward is included deliberately: ``_EnsureContiguous.backward`` copies a complex
        gradient, and that copy only appears in the joint graph.
        """
        _check_compile(self, nlat, nlon, grid, lmax, dtype, vector=False, verbose=verbose)


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

    @parameterized.expand(_GRAM_CASES, skip_on_empty=True)
    def test_orthogonality(self, nlat, grid, norm, verbose=False):
        """isht synthesizes mutually orthogonal basis functions with the norms of each convention."""
        _check_gram(self, nlat, grid, norm, vector=False, verbose=verbose)


@parameterized_class(("device"), _devices)
class TestVectorSphericalHarmonicTransform(unittest.TestCase):
    """Tests for the vector SHT and its relation to the scalar SHT.

    The spheroidal and toroidal coefficients of a vector field are the scalar
    coefficients of its potentials: ivsht([c, 0]) is the gradient and ivsht([0, c])
    the curl of isht(c), and RealVectorSHT includes a 1/(l*(l+1)) normalization in
    its quadrature weights so that vsht recovers c from either. The l = 0 mode is
    zero in both directions because a constant field has no gradient or curl.

    Recovery and the Gram matrix are blind to a convention changed consistently in
    both directions, such as the sign of the toroidal channel or the component order.
    Those are fixed by test_scipy_harmonics.
    """

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

    @parameterized.expand(_COMPILE_CASES, skip_on_empty=True)
    @requires_torch_compile
    def test_compile(self, nlat, nlon, grid, lmax, dtype, verbose=False):
        """The vector round trip compiles into a single graph and matches eager.

        Same rationale as the scalar case, see
        ``TestSphericalHarmonicTransform.test_compile``.  The vector transforms assemble
        their spheroidal and toroidal components separately, so they exercise a code path
        the scalar test does not reach.
        """
        _check_compile(self, nlat, nlon, grid, lmax, dtype, vector=True, verbose=verbose)


@parameterized_class(("device"), _devices)
class TestUnsuitableGridWarning(unittest.TestCase):
    """
    The SHT must object loudly when handed a grid whose quadrature cannot support it,
    or a degree its analysis cannot recover.

    ``trapezoidal`` is the only such grid today: it converges as
    :math:`O(h^2)` rather than spectrally, so the associated Legendre polynomials are
    not discretely orthogonal under its quadrature and the transform does not
    round-trip. Its numerical behaviour is deliberately *not* covered here -- it is not
    a supported configuration, and pinning its error would suggest otherwise. What is
    covered is that a user cannot stumble into it silently.
    """

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

    @parameterized.expand(
        [
            [cls_name, grid]
            for cls_name in ["RealSHT", "RealVectorSHT", "InverseRealSHT", "InverseRealVectorSHT"]
            for grid in ["legendre-gauss", "lobatto", "equiangular", "fejer1", "trapezoidal"]
        ]
    )
    def test_forward_degree_limit(self, cls_name, grid):
        """
        One degree past max_recoverable_degree, analysis raises on Gauss and Lobatto and
        warns on equiangular and Fejer-1; at the limit it is silent. Synthesis may upsample,
        so the inverse is silent on both sides. Trapezoidal keeps its single unsuitable-grid
        warning either way.
        """
        desc = th.as_grid(grid, nlat=8, nlon=16)
        forward = not cls_name.startswith("Inverse")
        for lmax in (desc.max_recoverable_degree, desc.max_recoverable_degree + 1):
            past = lmax > desc.max_recoverable_degree
            kwargs = dict(lmax=lmax, mmax=min(lmax, desc.max_azimuthal_order))
            with self.subTest(lmax=lmax):
                if forward and past and grid in ("legendre-gauss", "lobatto"):
                    with self.assertRaisesRegex(ValueError, "max_recoverable_degree"):
                        getattr(th, cls_name)(desc, **kwargs)
                    continue
                expected = ["must not be used"] if grid == "trapezoidal" else []
                if forward and past and grid in ("equiangular", "fejer1"):
                    expected.append("max_recoverable_degree")
                with warnings.catch_warnings(record=True) as emitted:
                    warnings.simplefilter("always", UserWarning)
                    getattr(th, cls_name)(desc, **kwargs)
                messages = [str(w.message) for w in emitted]
                self.assertEqual(len(messages), len(expected), messages)
                for message, fragment in zip(messages, expected):
                    self.assertIn(fragment, message)


@parameterized_class(("device",), _devices)
class TestSharedFoldedSHTAnalysis(unittest.TestCase):
    """Shared extended-analysis coverage for Clenshaw-Curtis and Fejer-1 grids."""

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
                # mmax=17 includes the Nyquist order 16 of nlon=32, so folding stays off
                (25, 32, 24, 17, None),
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
        # lmax-only resolution requires an explicit order above the recoverable
        # longitude bound, so spell out the largest order the longitude sampling
        # recovers.
        if mmax is None:
            mmax = min(lmax, grid.max_azimuthal_order)
        analysis = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=lmax, mmax=mmax, lmmax=lmmax)
        self.assertFalse(analysis._extended_folded)
        self.assertEqual(analysis.grid_out, th.SpectralGrid(lmax, mmax, lmmax))
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
        self.assertEqual(actual.shape, ((2,) if vector else ()) + analysis.grid_out.shape)
        self.assertTrue(compare_tensors("oversized direct quadrature", actual, expected, atol=0.0, rtol=0.0, ulps=_QUADRATURE_ULPS, verbose=verbose))
        gradient = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, x, gradient)[0]
        expected_grad = torch.autograd.grad(expected, x, gradient)[0]
        self.assertTrue(compare_tensors("oversized input gradient", actual_grad, expected_grad, atol=0.0, rtol=0.0, ulps=_QUADRATURE_ULPS, verbose=verbose))
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
        # the reference is zero, so the floor comes from the transform's own output range
        atol = range_atol(actual, 64 * torch.finfo(dtype).eps, 64)
        self.assertTrue(compare_tensors("arbitrary-input zero-order reality", actual[..., 0].imag, torch.zeros_like(actual[..., 0].imag), atol=atol, rtol=0, verbose=verbose))

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


# -- HEALPix ---------------------------------------------------------------------------
#
# The scalar SHT on HEALPix. The references are independent of the ring machinery under
# test: synthesis and analysis are compared with a direct evaluation of every harmonic at
# every pixel, which aliases nothing away because it never forms a ring spectrum, and
# RingFFT is compared with a plain rfft on a regular grid. healpy, pinned in the dev extra,
# serves as an external cross-check.

# the resolution of the truncation tests: 3 * nside = 24 fits all but the r42 shapes
_HEALPIX_TRUNCATION_NSIDE = 8

# both regimes: the default band, and the full 3 * nside band in which the polar rings
# alias most orders
_HEALPIX_BANDS = [(nside, factor) for nside in (1, 2, 4, 8) for factor in (2, 3)]


def _pixel_harmonics(grid, lmax, mmax, inverse):
    r"""
    Every harmonic evaluated at every pixel, ``(mmax, lmax, npix)`` complex, as
    :math:`P_l^m(\cos\theta_p)\, e^{\pm i m \lambda_p}`: ``+`` for synthesis, ``-`` with the
    analysis normalization for ``inverse=False``.
    """
    colat = grid.all_colats()
    lon = grid.all_lons()
    p = legpoly(mmax, lmax, torch.cos(colat), inverse=inverse)
    sign = 1.0 if inverse else -1.0
    phase = torch.exp(sign * 1j * torch.arange(mmax, dtype=torch.float64)[:, None] * lon[None, :])
    return p * phase[:, None, :]


def _direct_synthesis(grid, coeffs):
    """f(p) = sum_l Re c_l0 Y_l0(p) + 2 sum_{m>0} Re(c_lm Y_lm(p))."""
    lmax, mmax = coeffs.shape[-2:]
    y = _pixel_harmonics(grid, lmax, mmax, inverse=True)
    terms = torch.einsum("...lm,mlp->...mp", coeffs, y).real
    scale = torch.full((mmax, 1), 2.0, dtype=torch.float64)
    scale[0] = 1.0
    return (terms * scale).sum(-2)


def _direct_analysis(grid, field, lmax, mmax):
    """a_lm = sum_p v_p f(p) conj Y_lm(p), with v_p the ring weight of the pixel's ring."""
    v = 2.0 * torch.pi * healpix_ring_weights(grid.nside) / grid.nlon_per_lat.to(torch.float64)
    v = torch.repeat_interleave(v, grid.nlon_per_lat)
    y = _pixel_harmonics(grid, lmax, mmax, inverse=False)
    return torch.einsum("...p,mlp->...lm", (field * v).to(torch.complex128), y)


def _random_vector_coeffs(batch, lmax, mmax, device, lmmax=None):
    """Spheroidal and toroidal coefficients, ``(batch, 2, lmax, mmax)``, without the l = 0 row a tangent field lacks."""
    return torch.stack([random_sht_coeffs(batch, lmax, mmax, device, zero_l0=True, lmmax=lmmax) for _ in range(2)], dim=-3)


def _pixel_vector_tables(grid, lmax, mmax, inverse):
    r"""
    :math:`\partial_\theta P_l^m` and :math:`m P_l^m / \sin\theta` at every pixel, each
    ``(mmax, lmax, npix)``, times the azimuthal phase :math:`e^{\pm i m \lambda_p}` as in
    :func:`_pixel_harmonics`.
    """
    d = dlegpoly(mmax, lmax, grid.all_colats(), inverse=inverse)
    sign = 1.0 if inverse else -1.0
    phase = torch.exp(sign * 1j * torch.arange(mmax, dtype=torch.float64)[:, None] * grid.all_lons()[None, :])
    return d[0] * phase[:, None, :], d[1] * phase[:, None, :]


def _direct_vector_synthesis(grid, coeffs):
    r"""
    The tangent field of spheroidal coefficients :math:`s` and toroidal :math:`t`:
    :math:`u = s \nabla Y - t\, \hat r \times \nabla Y`, i.e. per order
    :math:`u_\theta = s\, \partial_\theta Y + i t\, m Y / \sin\theta` and
    :math:`u_\lambda = i s\, m Y / \sin\theta - t\, \partial_\theta Y`, summed like
    :func:`_direct_synthesis` over the real field's orders.
    """
    lmax, mmax = coeffs.shape[-2:]
    dtheta, dlambda = _pixel_vector_tables(grid, lmax, mmax, inverse=True)
    s, t = coeffs[..., 0, :, :], coeffs[..., 1, :, :]

    def contract(c, table):
        return torch.einsum("...lm,mlp->...mp", c, table)

    u_theta = contract(s, dtheta) + 1j * contract(t, dlambda)
    u_lambda = 1j * contract(s, dlambda) - contract(t, dtheta)
    scale = torch.full((mmax, 1), 2.0, dtype=torch.float64)
    scale[0] = 1.0
    return torch.stack([(u_theta.real * scale).sum(-2), (u_lambda.real * scale).sum(-2)], dim=-2)


def _direct_vector_analysis(grid, field, lmax, mmax):
    r"""
    The projection of a tangent field onto the vector harmonics with the ring weights,
    normalized by :math:`1 / (l (l + 1))` (1 at :math:`l = 0`, where both are zero).
    """
    v = 2.0 * torch.pi * healpix_ring_weights(grid.nside) / grid.nlon_per_lat.to(torch.float64)
    v = torch.repeat_interleave(v, grid.nlon_per_lat)
    dtheta, dlambda = _pixel_vector_tables(grid, lmax, mmax, inverse=False)
    u_theta = (field[..., 0, :] * v).to(torch.complex128)
    u_lambda = (field[..., 1, :] * v).to(torch.complex128)

    def contract(u, table):
        return torch.einsum("...p,mlp->...lm", u, table)

    s = contract(u_theta, dtheta) - 1j * contract(u_lambda, dlambda)
    t = -1j * contract(u_theta, dlambda) - contract(u_lambda, dtheta)
    l = torch.arange(lmax, dtype=torch.float64)
    norm = 1.0 / torch.where(l > 0, l * (l + 1), torch.ones_like(l))
    return torch.stack([s, t], dim=-3) * norm[:, None]


class TestHealpixRingWeights(unittest.TestCase):
    """The ring weights are a quadrature rule in their own right; check it as one."""

    @parameterized.expand([[n] for n in (1, 2, 4, 16, 64)])
    def test_they_integrate_zonal_harmonics_below_three_nside(self, nside):
        g = th.HealpixGrid(nside=nside)
        w = healpix_ring_weights(nside)
        self.assertEqual(w.shape, (g.nrings,))
        self.assertAlmostEqual(float(w.sum()), 2.0, places=13)
        self.assertTrue(torch.equal(w, w.flip(0)))

        # every P_l with l < 3N integrates exactly, odd ones by symmetry
        p = legpoly(1, 3 * nside, torch.cos(g.colats), norm="ortho")[0]
        integral = 2.0 * torch.pi * (p * w).sum(-1)
        expected = torch.zeros_like(integral)
        expected[0] = 2.0 * torch.pi * float(p[0, 0]) * 2.0
        self.assertTrue(compare_tensors(f"ring-weight integrals, nside={nside}", integral, expected, atol=1e-12, rtol=0.0))

    @parameterized.expand([[n] for n in (2, 4, 16, 64, 256)])
    def test_they_stay_close_to_the_equal_area_weights(self, nside):
        """
        The min-norm correction at degree 3N is well conditioned; at 4N it would not be.
        The ratio to the equal-area weights converges to about [0.97, 1.14].
        """
        g = th.HealpixGrid(nside=nside)
        ratio = healpix_ring_weights(nside) / g.colat_weights
        self.assertGreater(float(ratio.min()), 0.9)
        self.assertLess(float(ratio.max()), 1.17)


class TestRingFFT(unittest.TestCase):
    """RingFFT on a regular grid, where a plain rfft is the reference."""

    def test_a_regular_grid_needs_no_pixel_permutation(self):
        """One batch of rings already in pixel order is transformed without a gather."""
        self.assertFalse(RingFFT(th.as_grid("equiangular", nlat=5, nlon=16), mmax=9)._permuted)
        self.assertTrue(RingFFT(th.HealpixGrid(nside=2), mmax=4)._permuted)

    def test_a_regular_grid_shares_its_tables(self):
        """Without a gather the layer holds every table it is given, so the shared entry survives."""
        g = th.as_grid("equiangular", nlat=5, nlon=16)
        a, b = RingFFT(g, mmax=9), RingFFT(g, mmax=9)
        for name in ("_gather", "_fold_pos", "_fold_neg", "_imag_mask", "_phase_re"):
            self.assertIs(getattr(a, name), getattr(b, name), msg=name)

    def test_gradients_on_healpix(self):
        """The batch order is an internal layout: both directions differentiate exactly."""
        set_seed(333)
        ring_fft = RingFFT(th.HealpixGrid(nside=2), mmax=6)
        x = torch.randn(2, 48, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda t: torch.view_as_real(ring_fft(t)), (x,)))
        c = torch.randn(2, ring_fft.nrings, 6, dtype=torch.complex128, requires_grad=True)
        self.assertTrue(gradcheck(ring_fft.inverse, (c,)))

    def test_it_matches_rfft_on_a_regular_grid(self):
        set_seed(333)
        g = th.as_grid("equiangular", nlat=5, nlon=16)
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        ours = RingFFT(g, mmax=9)(x.reshape(3, -1))
        ref = torch.fft.rfft(x, norm="forward")
        self.assertTrue(compare_tensors("ring fft", ours, ref, atol=1e-14, rtol=0.0))

        # below the Nyquist order the inverse is irfft
        ref = ref[..., :8]
        ref[..., 0] = ref[..., 0].real
        back = RingFFT(g, mmax=8).inverse(ref)
        self.assertTrue(compare_tensors("ring ifft", back, torch.fft.irfft(ref, n=16, norm="forward").reshape(3, -1), atol=1e-14, rtol=0.0))

    def test_the_inverse_matches_irfft_on_odd_and_even_rings(self):
        """Every bin of an odd ring, the last one included, keeps its imaginary part."""
        set_seed(333)
        for nlon in (7, 8, 15, 16):
            with self.subTest(nlon=nlon):
                g = th.as_grid("equiangular", nlat=3, nlon=nlon)
                mmax = (nlon + 1) // 2
                c = torch.randn(2, 3, mmax, dtype=torch.complex128)
                c[..., 0] = c[..., 0].real
                expected = torch.fft.irfft(c, n=nlon, norm="forward").reshape(2, -1)
                self.assertTrue(compare_tensors("ring ifft", RingFFT(g, mmax=mmax).inverse(c), expected, atol=1e-13, rtol=0.0))

    def test_the_last_bin_of_an_odd_ring_keeps_its_sine_term(self):
        g = th.as_grid("equiangular", nlat=2, nlon=5)
        c = torch.zeros(2, 3, dtype=torch.complex128)
        c[:, 2] = 1j
        expected = (-2.0 * torch.sin(2 * g.lons())).repeat(2)
        self.assertTrue(compare_tensors("odd ring", RingFFT(g, mmax=3).inverse(c), expected, atol=1e-14, rtol=0.0))

    def test_the_nyquist_order_is_a_two_sided_harmonic(self):
        """
        Order n / 2 synthesizes 2 Re(G e^{i m lambda}), like every m > 0, not irfft's
        one-sided Nyquist bin: on HEALPix it is a genuine order of the expansion.
        """
        g = th.as_grid("equiangular", nlat=2, nlon=8)
        c = torch.zeros(2, 5, dtype=torch.complex128)
        c[:, 4] = 0.7 + 0.3j
        lon = g.lons()
        expected = (2.0 * 0.7 * torch.cos(4 * lon) - 2.0 * 0.3 * torch.sin(4 * lon)).repeat(2)
        self.assertTrue(compare_tensors("nyquist", RingFFT(g, mmax=5).inverse(c), expected, atol=1e-14, rtol=0.0))

    def test_orders_beyond_nyquist_alias_instead_of_vanishing(self):
        """Order m on a ring of n points reads bin m mod n, conjugated past n / 2."""
        set_seed(333)
        g = th.as_grid("equiangular", nlat=3, nlon=8)
        x = torch.randn(3, 8, dtype=torch.float64)
        ours = RingFFT(g, mmax=20)(x.reshape(-1))
        full = torch.fft.fft(x, norm="forward")
        for m in range(20):
            self.assertTrue(compare_tensors(f"m={m}", ours[:, m], full[:, m % 8], atol=1e-14, rtol=0.0))

    def test_the_inverse_is_the_adjoint_up_to_ring_length(self):
        """<F x, c> = <x, F^H c> with F^H = inverse / n, the defining property of the fold."""
        set_seed(333)
        g = th.HealpixGrid(nside=4)
        mmax = 13
        ring_fft = RingFFT(g, mmax)
        x = torch.randn(g.npoints, dtype=torch.float64)
        c = torch.randn(g.nrings, mmax, dtype=torch.complex128)
        # the Hermitian pairing of a real field counts every m > 0 twice
        scale = torch.full((mmax,), 2.0, dtype=torch.float64)
        scale[0] = 1.0
        lhs = (ring_fft(x) * c.conj() * scale).real.sum()
        y = ring_fft.inverse(c) / torch.repeat_interleave(g.nlon_per_lat.to(torch.float64), g.nlon_per_lat)
        # inverse drops Im c at m = 0, like irfft, and so must the pairing
        self.assertAlmostEqual(float(lhs), float((x * y).sum()), places=11)


@parameterized_class(("device"), _devices)
class TestHealpixSHT(unittest.TestCase):

    def setUp(self):
        set_seed(333)

    @parameterized.expand(_HEALPIX_BANDS)
    def test_synthesis_is_exact(self, nside, factor):
        g = th.HealpixGrid(nside=nside)
        lmax = factor * nside
        isht = th.InverseRealSHT(g, lmax=lmax, mmax=lmax).to(self.device)
        c = random_sht_coeffs(2, lmax, lmax, "cpu")
        ours = isht(c.to(self.device)).cpu()
        ref = _direct_synthesis(g, c)
        self.assertTrue(compare_tensors(f"synthesis nside={nside} lmax={lmax}", ours, ref, atol=1e-12, rtol=0.0, ulps=64))

    @parameterized.expand(_HEALPIX_BANDS)
    def test_analysis_is_the_ring_weight_quadrature(self, nside, factor):
        g = th.HealpixGrid(nside=nside)
        lmax = factor * nside
        sht = th.RealSHT(g, lmax=lmax, mmax=lmax).to(self.device)
        x = torch.randn(2, g.npoints, dtype=torch.float64)
        ours = sht(x.to(self.device)).cpu()
        ref = _direct_analysis(g, x, lmax, lmax)
        self.assertTrue(compare_tensors(f"analysis nside={nside} lmax={lmax}", ours, ref, atol=1e-12, rtol=0.0, ulps=64))

    @parameterized.expand([[name, *shape] for name, shape in _TRUNCATION_SHAPES.items() if shape[0] <= 3 * _HEALPIX_TRUNCATION_NSIDE])
    def test_spectral_truncations(self, name, lmax, mmax, lmmax):
        """
        The truncations of the regular grids, on HEALPix: both directions agree with the
        direct reference on the retained support, and the forward transform is exactly
        zero outside it. Shapes with lmax beyond 3 * nside are left out.
        """
        g = th.HealpixGrid(nside=_HEALPIX_TRUNCATION_NSIDE)
        sht = th.RealSHT(g, lmax, mmax, lmmax).to(self.device)
        isht = th.InverseRealSHT(g, lmax, mmax, lmmax).to(self.device)
        self.assertEqual((sht.lmax, sht.mmax, sht.lmmax), (lmax, mmax, lmmax))

        c = random_sht_coeffs(2, lmax, mmax, "cpu", lmmax=lmmax)
        ours = isht(c.to(self.device)).cpu()
        self.assertTrue(compare_tensors(f"{name} synthesis", ours, _direct_synthesis(g, c), atol=1e-12, rtol=0.0, ulps=64))

        l = torch.arange(lmax)[:, None]
        m = torch.arange(mmax)[None, :]
        support = (m <= l) & (l - m < (lmmax if lmmax is not None else lmax))
        x = torch.randn(2, g.npoints, dtype=torch.float64)
        coeffs = sht(x.to(self.device)).cpu()
        self.assertEqual(coeffs.shape, (2, lmax, mmax))
        ref = _direct_analysis(g, x, lmax, mmax) * support
        self.assertTrue(compare_tensors(f"{name} analysis", coeffs, ref, atol=1e-12, rtol=0.0, ulps=64))
        self.assertTrue(torch.equal(coeffs[..., ~support], torch.zeros_like(coeffs[..., ~support])))

    @parameterized.expand([[n] for n in (4, 8, 16)])
    def test_iteration_recovers_band_limited_fields(self, nside):
        """
        The quoted accuracy of the default band: about 1e-2 without iteration and 1e-5
        or better with three Jacobi steps.
        """
        g = th.HealpixGrid(nside=nside)
        isht = th.InverseRealSHT(g).to(self.device)
        c = random_sht_coeffs(4, isht.lmax, isht.mmax, self.device)
        x = isht(c)
        errors = []
        for niter in (0, 1, 3):
            sht = th.RealSHT(g, iter=niter).to(self.device)
            errors.append(float((sht(x) - c).abs().max() / c.abs().max()))
        self.assertLess(errors[0], 5e-2)
        self.assertLess(errors[2], 1e-4)
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])

    def test_the_default_band_and_shapes(self):
        g = th.HealpixGrid(nside=4)
        sht = th.RealSHT(g).to(self.device)
        isht = th.InverseRealSHT(g).to(self.device)
        self.assertEqual((sht.lmax, sht.mmax), (8, 8))
        x = torch.randn(2, 3, g.npoints, device=self.device)
        c = sht(x)
        self.assertEqual(c.shape, (2, 3, 8, 8))
        self.assertEqual(c.dtype, torch.complex64)
        self.assertEqual(isht(c).shape, x.shape)

    def test_a_radial_axis_is_carried_along(self):
        g = SphericalProductGrid(th.HealpixGrid(nside=2), LegendreGaussRadialGrid(nr=3, r_inner=1.0, r_outer=2.0))
        sht = th.RealSHT(g).to(self.device)
        isht = th.InverseRealSHT(g).to(self.device)
        x = torch.randn(2, 3, 48, dtype=torch.float64, device=self.device)
        c = sht.double()(x)
        self.assertEqual(c.shape, (2, 3, 4, 4))
        self.assertEqual(isht.double()(c).shape, x.shape)

    def test_float32_agrees_with_float64(self):
        g = th.HealpixGrid(nside=8)
        x = torch.randn(2, g.npoints, dtype=torch.float64)
        sht = th.RealSHT(g, iter=1).to(self.device)
        isht = th.InverseRealSHT(g).to(self.device)
        c64 = sht(x.to(self.device)).cpu()
        c32 = sht(x.float().to(self.device)).cpu()
        self.assertTrue(compare_tensors("fp32 analysis", c32.to(torch.complex128), c64, atol=0.0, rtol=0.0, ulps=64, eps_dtype=torch.float32))
        y64 = isht(c64.to(self.device)).cpu()
        y32 = isht(c64.to(torch.complex64).to(self.device)).cpu()
        self.assertTrue(compare_tensors("fp32 synthesis", y32.double(), y64, atol=0.0, rtol=0.0, ulps=64, eps_dtype=torch.float32))

    def test_gradients(self):
        g = th.HealpixGrid(nside=2)
        sht = th.RealSHT(g, iter=2).to(self.device)
        isht = th.InverseRealSHT(g).to(self.device)
        x = torch.randn(2, g.npoints, dtype=torch.float64, device=self.device, requires_grad=True)
        self.assertTrue(gradcheck(lambda t: torch.view_as_real(sht(t)), (x,)))
        c = random_sht_coeffs(2, isht.lmax, isht.mmax, self.device).requires_grad_(True)
        self.assertTrue(gradcheck(isht, (c,)))

    @requires_torch_compile
    def test_compile(self):
        g = th.HealpixGrid(nside=4)
        sht = th.RealSHT(g, iter=1).to(self.device)
        isht = th.InverseRealSHT(g).to(self.device)

        def fn(t):
            return isht(sht(t))

        x = torch.randn(2, g.npoints, device=self.device, requires_grad=True)
        gradient = torch.randn_like(x)
        expected = fn(x)
        (expected_grad,) = torch.autograd.grad(expected, x, grad_outputs=gradient)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        actual = compiled(x)
        (actual_grad,) = torch.autograd.grad(actual, x, grad_outputs=gradient)
        self.assertTrue(compare_tensors("compiled forward", actual, expected, atol=0.0, rtol=0.0, ulps=64))
        self.assertTrue(compare_tensors("compiled backward", actual_grad, expected_grad, atol=0.0, rtol=0.0, ulps=64))


@parameterized_class(("device"), _devices)
class TestHealpixVectorSHT(unittest.TestCase):
    """The vector SHT on HEALPix, against the same kind of direct per-pixel reference."""

    def setUp(self):
        set_seed(333)

    @parameterized.expand(_HEALPIX_BANDS)
    def test_synthesis_is_exact(self, nside, factor):
        g = th.HealpixGrid(nside=nside)
        lmax = factor * nside
        isht = th.InverseRealVectorSHT(g, lmax, lmax).to(self.device)
        c = _random_vector_coeffs(2, lmax, lmax, "cpu")
        ours = isht(c.to(self.device)).cpu()
        self.assertEqual(ours.shape, (2, 2, g.npoints))
        self.assertTrue(compare_tensors(f"vector synthesis nside={nside} lmax={lmax}", ours, _direct_vector_synthesis(g, c), atol=1e-12, rtol=0.0, ulps=64))

    @parameterized.expand(_HEALPIX_BANDS)
    def test_analysis_is_the_ring_weight_projection(self, nside, factor):
        g = th.HealpixGrid(nside=nside)
        lmax = factor * nside
        sht = th.RealVectorSHT(g, lmax, lmax).to(self.device)
        x = torch.randn(2, 2, g.npoints, dtype=torch.float64)
        ours = sht(x.to(self.device)).cpu()
        self.assertTrue(compare_tensors(f"vector analysis nside={nside} lmax={lmax}", ours, _direct_vector_analysis(g, x, lmax, lmax), atol=1e-12, rtol=0.0, ulps=64))

    @parameterized.expand([[n] for n in (4, 8, 16)])
    def test_iteration_recovers_band_limited_fields(self, nside):
        g = th.HealpixGrid(nside=nside)
        isht = th.InverseRealVectorSHT(g).to(self.device)
        c = _random_vector_coeffs(4, isht.lmax, isht.mmax, self.device)
        x = isht(c)
        errors = []
        for niter in (0, 1, 3):
            sht = th.RealVectorSHT(g, iter=niter).to(self.device)
            errors.append(float((sht(x) - c).abs().max() / c.abs().max()))
        self.assertLess(errors[0], 5e-2)
        self.assertLess(errors[2], 1e-4)
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])

    @parameterized.expand([[name, *shape] for name, shape in _TRUNCATION_SHAPES.items() if shape[0] <= 3 * _HEALPIX_TRUNCATION_NSIDE])
    def test_spectral_truncations(self, name, lmax, mmax, lmmax):
        g = th.HealpixGrid(nside=_HEALPIX_TRUNCATION_NSIDE)
        sht = th.RealVectorSHT(g, lmax, mmax, lmmax).to(self.device)
        isht = th.InverseRealVectorSHT(g, lmax, mmax, lmmax).to(self.device)
        c = _random_vector_coeffs(2, lmax, mmax, "cpu", lmmax=lmmax)
        ours = isht(c.to(self.device)).cpu()
        self.assertTrue(compare_tensors(f"{name} vector synthesis", ours, _direct_vector_synthesis(g, c), atol=1e-12, rtol=0.0, ulps=64))

        l = torch.arange(lmax)[:, None]
        m = torch.arange(mmax)[None, :]
        support = (m <= l) & (l - m < (lmmax if lmmax is not None else lmax))
        x = torch.randn(2, 2, g.npoints, dtype=torch.float64)
        coeffs = sht(x.to(self.device)).cpu()
        ref = _direct_vector_analysis(g, x, lmax, mmax) * support
        self.assertTrue(compare_tensors(f"{name} vector analysis", coeffs, ref, atol=1e-12, rtol=0.0, ulps=64))
        self.assertTrue(torch.equal(coeffs[..., ~support], torch.zeros_like(coeffs[..., ~support])))

    def test_a_radial_axis_is_carried_along_through_the_iteration(self):
        """The component axis sits in front of the radial one, which the Jacobi synthesis must respect."""
        g = SphericalProductGrid(th.HealpixGrid(nside=4), LegendreGaussRadialGrid(nr=3, r_inner=1.0, r_outer=2.0))
        sht = th.RealVectorSHT(g, iter=3).to(self.device).double()
        isht = th.InverseRealVectorSHT(g).to(self.device).double()
        c = torch.stack([_random_vector_coeffs(2, 8, 8, self.device) for _ in range(3)], dim=-3)
        x = isht(c)
        self.assertEqual(x.shape, (2, 2, 3, 192))
        self.assertLess(float((sht(x) - c).abs().max() / c.abs().max()), 1e-4)

    def test_gradients(self):
        g = th.HealpixGrid(nside=2)
        sht = th.RealVectorSHT(g, iter=2).to(self.device)
        isht = th.InverseRealVectorSHT(g).to(self.device)
        x = torch.randn(2, 2, g.npoints, dtype=torch.float64, device=self.device, requires_grad=True)
        self.assertTrue(gradcheck(lambda t: torch.view_as_real(sht(t)), (x,)))
        c = _random_vector_coeffs(2, isht.lmax, isht.mmax, self.device).requires_grad_(True)
        self.assertTrue(gradcheck(isht, (c,)))

    @requires_torch_compile
    def test_compile(self):
        g = th.HealpixGrid(nside=4)
        sht = th.RealVectorSHT(g, iter=1).to(self.device)
        isht = th.InverseRealVectorSHT(g).to(self.device)

        def fn(t):
            return isht(sht(t))

        x = torch.randn(2, 2, g.npoints, device=self.device, requires_grad=True)
        gradient = torch.randn_like(x)
        expected = fn(x)
        (expected_grad,) = torch.autograd.grad(expected, x, grad_outputs=gradient)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        actual = compiled(x)
        (actual_grad,) = torch.autograd.grad(actual, x, grad_outputs=gradient)
        self.assertTrue(compare_tensors("compiled forward", actual, expected, atol=0.0, rtol=0.0, ulps=64))
        self.assertTrue(compare_tensors("compiled backward", actual_grad, expected_grad, atol=0.0, rtol=0.0, ulps=64))


_RING_FFT_TABLES = ("_pixel_order", "_pixel_inverse", "_gather", "_gather_sign", "_fold_pos", "_fold_neg", "_imag_mask", "_phase_re", "_phase_im")


@parameterized_class(("device"), _devices)
class TestHealpixSharedTables(unittest.TestCase):
    """HEALPix transforms share their tables like the regular ones: see torch_harmonics.cache."""

    def _assert_ring_fft_shared(self, a, b):
        for name in _RING_FFT_TABLES:
            self.assertIs(getattr(a, name), getattr(b, name), msg=name)

    @parameterized.expand([["RealSHT", "weights"], ["InverseRealSHT", "pct"], ["RealVectorSHT", "weights"], ["InverseRealVectorSHT", "dpct"]])
    def test_identical_transforms_share_their_tables(self, cls_name, name):
        cls = getattr(th, cls_name)
        g = th.HealpixGrid(nside=4)
        a, b = cls(g).to(self.device), cls(g).to(self.device)
        self.assertIs(getattr(a, name), getattr(b, name))
        self._assert_ring_fft_shared(a.ring_fft, b.ring_fft)
        self.assertEqual(getattr(a, name).device.type, torch.device(self.device).type)
        # a different band is a different table
        self.assertIsNot(getattr(cls(g, lmax=10, mmax=10).to(self.device), name), getattr(a, name))

    def test_forward_and_inverse_share_the_ring_fft(self):
        """The ring tables depend on the grid and mmax only, not on the direction."""
        g = th.HealpixGrid(nside=4)
        self._assert_ring_fft_shared(th.RealSHT(g).to(self.device).ring_fft, th.InverseRealSHT(g).to(self.device).ring_fft)

    def test_the_jacobi_synthesis_shares_with_a_standalone_inverse(self):
        g = th.HealpixGrid(nside=4)
        sht = th.RealSHT(g, iter=3).to(self.device)
        isht = th.InverseRealSHT(g).to(self.device)
        self.assertIs(sht._synthesis.pct, isht.pct)
        self._assert_ring_fft_shared(sht._synthesis.ring_fft, isht.ring_fft)
        self._assert_ring_fft_shared(sht.ring_fft, isht.ring_fft)

    def test_the_tables_are_freed_with_the_last_transform(self):
        g = th.HealpixGrid(nside=3)
        layers = [th.RealSHT(g).to(self.device) for _ in range(2)]
        refs = [weakref.ref(layers[0].weights), weakref.ref(layers[0].ring_fft._gather)]
        layers.clear()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs), "a table outlived every transform using it")

    def test_sharing_changes_no_result(self):
        """Shared tables compute exactly what a transform with its own copies does."""
        g = th.HealpixGrid(nside=4)
        sht = th.RealSHT(g, iter=1).to(self.device)
        unshared = th.RealSHT(g, iter=1).to(self.device)
        unshared.weights = sht.weights.clone()
        for name in _RING_FFT_TABLES:
            setattr(unshared.ring_fft, name, getattr(sht.ring_fft, name).clone())
        x = torch.randn(2, g.npoints, dtype=torch.float64, device=self.device)
        self.assertTrue(torch.equal(sht.double()(x), unshared.double()(x)))


class TestJacobiIteration(unittest.TestCase):

    def test_it_is_a_no_op_on_an_exact_grid(self):
        """On a Gauss grid the analysis is exact, so each correction is round-off."""
        set_seed(333)
        g = th.as_grid("legendre-gauss", nlat=16, nlon=32)
        x = th.InverseRealSHT(g).double()(random_sht_coeffs(2, 16, 16, "cpu"))
        a0 = th.RealSHT(g).double()(x)
        a2 = th.RealSHT(g, iter=2).double()(x)
        self.assertTrue(compare_tensors("iter=2 vs iter=0", a2, a0, atol=1e-13, rtol=0.0))

    @parameterized.expand([[-1], [1.5], [True], ["3"]])
    def test_invalid_counts_are_rejected(self, niter):
        with self.assertRaises(ValueError):
            th.RealSHT(th.HealpixGrid(nside=2), iter=niter)

    def test_the_count_is_in_the_repr(self):
        self.assertIn("iter=3", repr(th.RealSHT(th.HealpixGrid(nside=2), iter=3)))
        self.assertNotIn("iter=", repr(th.RealSHT(th.HealpixGrid(nside=2))))

    def test_other_ragged_grids_are_rejected(self):
        with self.assertRaises(TypeError):
            th.RealSHT(th.as_grid("legendre-gauss", nlat=4, nlon=8).shard(polar=(0, 2)))


@unittest.skipIf(healpy is None, "healpy is not installed")
class TestAgainstHealpy(unittest.TestCase):

    @parameterized.expand(_HEALPIX_BANDS[2:])
    def test_synthesis_matches_alm2map(self, nside, factor):
        """healpy's alm order is m-major with an inclusive lmax; its convention is ours."""
        g = th.HealpixGrid(nside=nside)
        lmax = factor * nside
        c = random_sht_coeffs(1, lmax, lmax, "cpu")[0]
        alm = torch.cat([c[m:, m] for m in range(lmax)]).numpy()
        ref = torch.from_numpy(healpy.alm2map(alm, nside, lmax=lmax - 1))
        ours = th.InverseRealSHT(g, lmax=lmax, mmax=lmax).double()(c)
        self.assertTrue(compare_tensors(f"vs healpy nside={nside} lmax={lmax}", ours, ref, atol=1e-12, rtol=0.0, ulps=64))

    @parameterized.expand([[8], [16]])
    def test_iterated_analysis_is_as_accurate_as_map2alm(self, nside):
        g = th.HealpixGrid(nside=nside)
        lmax = 2 * nside
        c = random_sht_coeffs(1, lmax, lmax, "cpu")[0]
        x = th.InverseRealSHT(g).double()(c)
        ours = th.RealSHT(g, iter=3).double()(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alm = healpy.map2alm(x.numpy(), lmax=lmax - 1, iter=3, use_weights=False)
        theirs = torch.zeros_like(c)
        start = 0
        for m in range(lmax):
            theirs[m:, m] = torch.from_numpy(alm[start : start + lmax - m])
            start += lmax - m
        err_ours = float((ours - c).abs().max())
        err_theirs = float((theirs - c).abs().max())
        self.assertLess(err_ours, 2.0 * err_theirs)

    @staticmethod
    def _to_healpy(coeffs, factor):
        """Our (lmax, mmax) layout as healpy's m-major alm array, each degree scaled by factor[l]."""
        lmax, mmax = coeffs.shape
        return torch.cat([coeffs[m:, m] * factor[m:] for m in range(mmax)]).numpy()

    @parameterized.expand(_HEALPIX_BANDS[2:])
    def test_vector_synthesis_matches_alm2map_spin(self, nside, factor):
        """
        healpy's spin-1 gradient and curl coefficients are ours scaled by sqrt(l (l + 1)),
        with the opposite sign on the curl; the two map components agree as they are.
        """
        g = th.HealpixGrid(nside=nside)
        lmax = factor * nside
        c = _random_vector_coeffs(1, lmax, lmax, "cpu")[0]
        l = torch.arange(lmax, dtype=torch.float64)
        scale = torch.sqrt(l * (l + 1))
        maps = healpy.alm2map_spin([self._to_healpy(c[0], scale), self._to_healpy(c[1], -scale)], nside, 1, lmax - 1)
        ref = torch.from_numpy(np.stack(maps))
        ours = th.InverseRealVectorSHT(g, lmax, lmax).double()(c)
        self.assertTrue(compare_tensors(f"vector vs healpy nside={nside} lmax={lmax}", ours, ref, atol=1e-12, rtol=0.0, ulps=64))

    @parameterized.expand([[8], [16]])
    def test_vector_analysis_is_as_accurate_as_map2alm_spin(self, nside):
        """map2alm_spin has no iteration; ours without one is as accurate, with three far more."""
        g = th.HealpixGrid(nside=nside)
        lmax = 2 * nside
        c = _random_vector_coeffs(1, lmax, lmax, "cpu")[0]
        x = th.InverseRealVectorSHT(g).double()(c)
        grad, curl = healpy.map2alm_spin([x[0].numpy(), x[1].numpy()], 1, lmax - 1)
        l = torch.arange(lmax, dtype=torch.float64)
        scale = torch.where(l > 0, torch.sqrt(l * (l + 1)), torch.ones_like(l))
        theirs = torch.zeros_like(c)
        start = 0
        for m in range(lmax):
            theirs[0, m:, m] = torch.from_numpy(grad[start : start + lmax - m]) / scale[m:]
            theirs[1, m:, m] = -torch.from_numpy(curl[start : start + lmax - m]) / scale[m:]
            start += lmax - m
        err_theirs = float((theirs - c).abs().max())
        self.assertLess(float((th.RealVectorSHT(g).double()(x) - c).abs().max()), 1.5 * err_theirs)
        self.assertLess(float((th.RealVectorSHT(g, iter=3).double()(x) - c).abs().max()), 1e-2 * err_theirs)
