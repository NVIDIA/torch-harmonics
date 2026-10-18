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

    tol = 1e-5 if dtype == torch.float32 else 1e-12
    test.assertTrue(compare_tensors("compiled forward", actual, expected, atol=tol, rtol=tol, verbose=verbose))
    test.assertTrue(compare_tensors("compiled backward", actual_grad, expected_grad, atol=tol, rtol=tol, verbose=verbose))


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
    atol = 1e-12 * max(diagonal)
    test.assertTrue(compare_tensors("Gram matrix", gram, expected, atol=atol, rtol=0.0, verbose=verbose))


@parameterized_class(("device"), _devices)
class TestSphericalHarmonicTransform(unittest.TestCase):
    """Test the spherical harmonic transform (CPU/CUDA if available)."""

    def setUp(self):
        disable_tf32()

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

    def setUp(self):
        disable_tf32()

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

    def setUp(self):
        disable_tf32()

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


@parameterized_class(("device",), _devices)
class TestSharedFoldedSHTAnalysis(unittest.TestCase):
    """Shared extended-analysis coverage for Clenshaw-Curtis and Fejer-1 grids."""

    @classmethod
    def setUpClass(cls):
        disable_tf32()

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
        requested_mmax = mmax
        # lmax-only resolution now requires an explicit order above the recoverable
        # longitude bound. Preserve these direct-quadrature cases by spelling out the
        # same Nyquist order that the old inference selected.
        if mmax is None and lmax > grid.max_recoverable_order:
            mmax = min(lmax, grid.max_azimuthal_order)
        analysis = (th.RealVectorSHT if vector else th.RealSHT)(grid, lmax=lmax, mmax=mmax, lmmax=lmmax)
        self.assertFalse(analysis._extended_folded)
        self.assertEqual(analysis.grid_out, th.SpectralGrid(lmax, min(lmax, nlon // 2 + 1) if requested_mmax is None else requested_mmax, lmmax))
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
        if requested_mmax is None and mmax is None:
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


if __name__ == "__main__":
    unittest.main()
