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
"""Associated Legendre polynomials, their derivatives and the spectral tables built from them."""

import math
import unittest

import torch
from parameterized import parameterized, parameterized_class
from testutils import compare_tensors, disable_tf32

import torch_harmonics as th
from torch_harmonics.legendre import _precompute_dlegpoly, _precompute_legpoly
from torch_harmonics.quadrature import precompute_latitudes

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device"), _devices)
class TestLegendrePolynomials(unittest.TestCase):
    """Test the associated Legendre polynomials (CPU/CUDA if available)."""

    def setUp(self):
        disable_tf32()
        self.cml = lambda m, l: math.sqrt((2 * l + 1) / 4 / math.pi) * math.sqrt(math.factorial(l - m) / math.factorial(l + m))
        self.pml = dict()

        # preparing associated Legendre Polynomials (These include the Condon-Shortley phase)
        # for reference see e.g. https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
        self.pml[(0, 0)] = lambda x: torch.ones_like(x)
        self.pml[(0, 1)] = lambda x: x
        self.pml[(1, 1)] = lambda x: -torch.sqrt(1.0 - x**2)
        self.pml[(0, 2)] = lambda x: 0.5 * (3 * x**2 - 1)
        self.pml[(1, 2)] = lambda x: -3 * x * torch.sqrt(1.0 - x**2)
        self.pml[(2, 2)] = lambda x: 3 * (1 - x**2)
        self.pml[(0, 3)] = lambda x: 0.5 * (5 * x**3 - 3 * x)
        self.pml[(1, 3)] = lambda x: 1.5 * (1 - 5 * x**2) * torch.sqrt(1.0 - x**2)
        self.pml[(2, 3)] = lambda x: 15 * x * (1 - x**2)
        self.pml[(3, 3)] = lambda x: -15 * torch.sqrt(1.0 - x**2) ** 3

        self.lmax = self.mmax = 4

        self.tol = 1e-9

    def test_legendre(self, verbose=False):
        t = torch.linspace(0, 1, 100, dtype=torch.float64, device=self.device)
        vdm = th.legendre.legpoly(self.mmax, self.lmax, t)

        for l in range(self.lmax):
            for m in range(l + 1):
                diff = vdm[m, l] / self.cml(m, l) - self.pml[(m, l)](t)
                self.assertTrue(diff.abs().max() <= self.tol, f"P_{l}^{m}: max |diff| = {float(diff.abs().max())}")

    @parameterized.expand(
        [
            # lmax, norm, csphase
            [16, "ortho", True],
            [16, "ortho", False],
            [16, "four-pi", True],
            [16, "schmidt", True],
            [48, "ortho", True],
            [48, "schmidt", False],
        ],
        skip_on_empty=True,
    )
    def test_addition_theorem(self, lmax, norm, csphase, verbose=False):
        r"""The degree-l harmonics sum to a constant at every latitude.

        :math:`\sum_{m=-l}^{l} |Y_l^m|^2 = (2l+1)/4\pi` holds pointwise in :math:`\theta`,
        so this needs no quadrature, no transform and no reference table -- and at each
        degree it constrains every order at once. That makes it the one cheap check that
        reaches arbitrary ``lmax``, where the closed forms in ``test_legendre`` stop at
        degree 3. A drifted recurrence, a mis-seeded diagonal or a lost normalization all
        break it.

        The Condon-Shortley phase is squared away, so the identity is insensitive to it.
        """

        t, _ = precompute_latitudes(2 * lmax, grid="legendre-gauss")
        t = t.to(self.device)
        # the tensor-based core, so the nodes can stay on the test device; the cached
        # grid-keyed wrapper is exercised by test_range_restriction
        pct = th.legendre.legpoly(lmax, lmax, torch.cos(t), norm=norm, csphase=csphase)

        # the normalization scales every entry, so it scales the sum by a known constant
        for l in range(lmax):
            if norm == "ortho":
                expected = (2 * l + 1) / (4 * math.pi)
            elif norm == "four-pi":
                expected = 2.0 * l + 1.0
            else:  # schmidt carries an extra 1/sqrt(2l+1) on top of the four-pi factor
                expected = 1.0

            # m and -m contribute equally, so orders above zero are counted twice
            total = pct[0, l] ** 2 + 2.0 * (pct[1 : l + 1, l] ** 2).sum(dim=0)
            ok = compare_tensors(f"addition theorem at l={l}", total, torch.full_like(total, expected), atol=1e-12, rtol=1e-12, verbose=verbose)
            self.assertTrue(ok, msg=f"addition theorem violated at l={l}")

    @parameterized.expand(
        [
            # nlat, lmax, grid
            [48, 24, "equiangular"],
            [48, 24, "legendre-gauss"],
            [48, 24, "lobatto"],
            [96, 48, "legendre-gauss"],
            [48, 24, "fejer1"],
        ],
        skip_on_empty=True,
    )
    def test_discrete_orthogonality(self, nlat, lmax, grid, verbose=False):
        r"""The polynomials are discretely orthonormal under the grid's own quadrature.

        :math:`2\pi \sum_k w_k P_l^m P_{l'}^m = \delta_{l l'}` for each fixed order. This
        is the property the transform actually relies on for round-tripping, checked here
        as a full Gram matrix per order rather than a handful of entries, and without
        going through the SHT -- so a failure points at the polynomials, not the transform.
        """

        t, w = precompute_latitudes(nlat, grid=grid)
        t, w = t.to(self.device), w.to(self.device)
        pct = th.legendre.legpoly(lmax, lmax, torch.cos(t), norm="ortho", csphase=False)

        for m in [0, 1, lmax // 2, lmax - 1]:
            # only degrees l >= m are supported; the rest are identically zero
            block = pct[m, m:lmax]
            gram = 2.0 * math.pi * torch.einsum("lk,jk,k->lj", block, block, w)
            eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            ok = compare_tensors(f"gram matrix at m={m}", gram, eye, atol=1e-11, rtol=1e-11, verbose=verbose)
            self.assertTrue(ok, msg=f"orthogonality violated at m={m}")

    @parameterized.expand(
        [
            # lmax, norm, inverse, csphase
            [20, "ortho", False, True],
            [20, "ortho", True, False],
            [20, "four-pi", False, True],
            [20, "schmidt", False, True],
            [20, "schmidt", True, True],
        ],
        skip_on_empty=True,
    )
    def test_dlegpoly_consistency(self, lmax, norm, inverse, csphase, verbose=False):
        r"""The derivative table agrees with the base table it is supposed to differentiate.

        Two independent checks on ``_precompute_dlegpoly``, which is computed from its own
        recurrences rather than by differentiating anything:

        * ``dpct[1]`` must equal :math:`m P_l^m / \sin\theta` exactly (to roundoff). It is
          built from a different three-term relation, so agreement is real corroboration
          rather than a restatement of the same arithmetic.
        * ``dpct[0]`` must equal :math:`d P_l^m / d\theta`, checked against a central
          difference of the base table. Truncation dominates here, hence the looser
          tolerance -- it is still orders of magnitude tighter than any sign or index error.
        """

        t, _ = precompute_latitudes(2 * lmax, grid="legendre-gauss")
        t = t.to(self.device)

        kwargs = dict(norm=norm, inverse=inverse, csphase=csphase)
        pct = th.legendre.legpoly(lmax, lmax, torch.cos(t), **kwargs)
        dpct = th.legendre.dlegpoly(lmax, lmax, t, **kwargs)

        # entries outside the triangular support are zero by construction and carry no information
        m_g = torch.arange(lmax, device=self.device).view(lmax, 1, 1)
        l_g = torch.arange(lmax, device=self.device).view(1, lmax, 1)
        support = (m_g <= l_g).expand_as(pct)

        m_f = torch.arange(lmax, dtype=torch.float64, device=self.device).view(lmax, 1, 1)
        expected = m_f * pct / torch.sin(t).view(1, 1, -1)
        ok = compare_tensors("dpct[1] vs m P/sin(theta)", dpct[1][support], expected[support], atol=1e-10, rtol=1e-10, verbose=verbose)
        self.assertTrue(ok, msg="dpct[1] does not match m P/sin(theta)")

        # central difference; truncation dominates the comparison, hence the looser tolerance
        h = 1e-5
        fd = (th.legendre.legpoly(lmax, lmax, torch.cos(t + h), **kwargs) - th.legendre.legpoly(lmax, lmax, torch.cos(t - h), **kwargs)) / (2 * h)
        ok = compare_tensors("dpct[0] vs central difference", dpct[0][support], fd[support], atol=1e-5, rtol=1e-6, verbose=verbose)
        self.assertTrue(ok, msg="dpct[0] does not match d/dtheta P")

    @parameterized.expand(
        [
            # mmax, lmax
            [12, 12],
            [16, 10],
            [10, 16],
            [1, 1],
        ],
        skip_on_empty=True,
    )
    def test_range_restriction(self, mmax, lmax, verbose=False):
        """Restricting the stored range returns exactly the corresponding slice.

        ``mmin``/``lmin`` control which orders and degrees are *stored*, not which are
        *computed*: both recurrences still have to be walked from the start, since
        ``P^m_m`` is reached from ``P^{m-1}_{m-1}`` and ``P^m_l`` from ``P^m_{l-1}``. The
        distributed transforms rely on this to build only the block they keep, so the
        restricted result has to be indistinguishable from slicing the full one.

        Agreement is exact, not approximate -- the same elementwise operations are applied
        to the same values either way -- so this asserts bitwise equality. Note this also
        pins the ``mmin=lmin=0`` default as the serial path: every other test in this file
        goes through the same code.
        """

        nlat = 2 * max(mmax, lmax)
        t, _ = precompute_latitudes(nlat, grid="legendre-gauss")
        t = t.to(self.device)

        # legpoly evaluates at cos(theta), dlegpoly at theta itself
        nodes = lambda fn: torch.cos(t) if fn is th.legendre.legpoly else t

        for fn in (th.legendre.legpoly, th.legendre.dlegpoly):
            for norm in ["ortho", "four-pi", "schmidt"]:
                for inverse in [False, True]:
                    for csphase in [False, True]:
                        kwargs = dict(norm=norm, inverse=inverse, csphase=csphase)
                        full = fn(mmax, lmax, nodes(fn), **kwargs)

                        for mmin in sorted({0, 1, mmax // 2, mmax - 1} & set(range(mmax))):
                            for lmin in sorted({0, 1, lmax // 2, lmax - 1} & set(range(lmax))):
                                block = fn(mmax, lmax, nodes(fn), mmin=mmin, lmin=lmin, **kwargs)
                                ref = full[..., mmin:, lmin:, :]
                                case = f"{fn.__name__} mmax={mmax} lmax={lmax} mmin={mmin} lmin={lmin} {norm} inverse={inverse} csphase={csphase}"
                                self.assertEqual(tuple(block.shape), tuple(ref.shape), msg=f"shape mismatch: {case}")
                                # atol=rtol=0: bitwise, see the docstring
                                ok = compare_tensors(case, block, ref.contiguous(), atol=0.0, rtol=0.0, verbose=verbose)
                                self.assertTrue(ok, msg=f"values differ from the full-table slice: {case}")

        # the latitude range is only reachable through the cached, grid-keyed wrappers, since
        # the tensor-based core restricts latitudes by simply being handed fewer nodes
        for fn in (th.legendre._precompute_legpoly, th.legendre._precompute_dlegpoly):
            full = fn(mmax, lmax, th.as_grid("legendre-gauss", nlat=nlat, nlon=2 * nlat))
            for kmin, kmax in [(0, None), (1, None), (0, nlat - 1), (2, nlat - 2), (nlat // 2, nlat)]:
                block = fn(mmax, lmax, th.as_grid("legendre-gauss", nlat=nlat, nlon=2 * nlat), kmin=kmin, kmax=kmax)
                ref = full[..., kmin : (nlat if kmax is None else kmax)]
                case = f"{fn.__name__} mmax={mmax} lmax={lmax} kmin={kmin} kmax={kmax}"
                self.assertEqual(tuple(block.shape), tuple(ref.shape), msg=f"shape mismatch: {case}")
                ok = compare_tensors(case, block, ref.contiguous(), atol=0.0, rtol=0.0, verbose=verbose)
                self.assertTrue(ok, msg=f"values differ from the full-table slice: {case}")


@parameterized_class(("device"), _devices)
class TestSpectralLegendreMask(unittest.TestCase):
    @parameterized.expand(
        [
            ("scalar_forward_zero_offsets", _precompute_legpoly, False, 0, 0),
            ("scalar_forward_order_offset", _precompute_legpoly, False, 3, 0),
            ("scalar_forward_order_degree_offsets", _precompute_legpoly, False, 3, 5),
            ("scalar_inverse_zero_offsets", _precompute_legpoly, True, 0, 0),
            ("scalar_inverse_order_offset", _precompute_legpoly, True, 3, 0),
            ("scalar_inverse_order_degree_offsets", _precompute_legpoly, True, 3, 5),
            ("derivative_forward_zero_offsets", _precompute_dlegpoly, False, 0, 0),
            ("derivative_forward_order_offset", _precompute_dlegpoly, False, 3, 0),
            ("derivative_forward_order_degree_offsets", _precompute_dlegpoly, False, 3, 5),
            ("derivative_inverse_zero_offsets", _precompute_dlegpoly, True, 0, 0),
            ("derivative_inverse_order_offset", _precompute_dlegpoly, True, 3, 0),
            ("derivative_inverse_order_degree_offsets", _precompute_dlegpoly, True, 3, 5),
        ]
    )
    def test_global_offset_blocks(self, case, builder, inverse, mmin, lmin):
        grid = th.as_grid("legendre-gauss", nlat=32, nlon=64)
        trunc = th.truncate_sht(grid, lmax=20, mmax=9, lmmax=6)
        args = (9, 16, grid)
        kwargs = dict(mmin=mmin, lmin=lmin, inverse=inverse)
        original = builder(*args, **kwargs).to(self.device)
        masked = builder(*args, truncation=trunc, **kwargs).to(self.device)
        m = torch.arange(mmin, 9, device=self.device)[:, None]
        l = torch.arange(lmin, 16, device=self.device)[None, :]
        keep = (m <= l) & (l - m < 6)
        if builder is _precompute_legpoly:
            self.assertTrue(torch.equal(masked[keep], original[keep]))
            self.assertTrue(torch.equal(masked[~keep], torch.zeros_like(masked[~keep])))
        else:
            self.assertTrue(torch.equal(masked[:, keep], original[:, keep]))
            self.assertTrue(torch.equal(masked[:, ~keep], torch.zeros_like(masked[:, ~keep])))
        self.assertTrue((original[..., (l - m == 6), :].abs() > 0).any())

    @parameterized.expand([("scalar", _precompute_legpoly), ("derivative", _precompute_dlegpoly)])
    def test_cached_tables_distinguish_lmmax(self, case, builder):
        grid = th.as_grid("legendre-gauss", nlat=32, nlon=64)
        args = (9, 16, grid)
        narrow = builder(*args, truncation=th.SpectralGrid(16, 9, 6)).to(self.device)
        wide = builder(*args, truncation=th.SpectralGrid(16, 9, 7)).to(self.device)

        narrow_edge = narrow[..., 0, 6, :]
        wide_edge = wide[..., 0, 6, :]
        self.assertTrue(torch.equal(narrow_edge, torch.zeros_like(narrow_edge)))
        self.assertTrue((wide_edge.abs() > 0).any())
        self.assertTrue(torch.equal(narrow[..., 0, 5, :], wide[..., 0, 5, :]))

    @parameterized.expand([("scalar", _precompute_legpoly), ("derivative", _precompute_dlegpoly)])
    def test_two_sided_truncation_is_rejected(self, case, builder):
        grid = th.as_grid("legendre-gauss", nlat=32, nlon=64)
        with self.assertRaises(NotImplementedError):
            builder(9, 16, grid, truncation=th.SpectralGrid(16, 9, onesided=False))

    def test_r42_noninclusive_edge(self):
        grid = th.as_grid("legendre-gauss", nlat=96, nlon=192)
        trunc = th.truncate_sht(grid, lmax=85, mmax=43, lmmax=43)
        unmasked = _precompute_legpoly(43, 85, grid, mmin=41, lmin=82)
        masked = _precompute_legpoly(43, 85, grid, mmin=41, lmin=82, truncation=trunc)
        self.assertTrue(torch.equal(masked[0, 1], unmasked[0, 1]))  # (m, l) = (41, 83)
        self.assertTrue(torch.equal(masked[0, 2], torch.zeros_like(masked[0, 2])))  # (41, 84): l-m=43
        self.assertTrue(torch.equal(masked[1, 2], unmasked[1, 2]))  # (42, 84): l-m=42
        self.assertTrue((unmasked[0, 2].abs() > 0).any())


if __name__ == "__main__":
    unittest.main()
