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

r"""
Longitudinal Fourier transforms on a ragged :class:`~torch_harmonics.grid.GridS2`.

On a regular grid every ring has ``nlon`` points, so the longitudinal step of an SHT is
one batched ``rfft`` over the last axis of an ``(nlat, nlon)`` field. On a ragged grid
such as HEALPix the rings differ in length and are offset by a fraction of a point, and
the field is stored flat as ``(npoints,)`` in ring order. :class:`RingFFT` provides the
same step for such a grid: it maps a flat field to the Fourier coefficients
:math:`F_m(\theta_k)`, ``m < mmax``, of every ring, shaped ``(nrings, mmax)``, and back.

**Analysis.** Ring :math:`k` holds :math:`n_k` points at
:math:`\lambda_j = \lambda_{0,k} + 2\pi j / n_k`. Its coefficient of order :math:`m` is

.. math::

    F_m(\theta_k) = \frac{1}{n_k} \sum_{j=0}^{n_k - 1} f_{kj}\, e^{-i m \lambda_j}
                  = e^{-i m \lambda_{0,k}}\, \hat{f}_{k,\, m \bmod n_k},

where :math:`\hat f_k` is the forward-normalized DFT of the ring. Orders at or above
:math:`n_k / 2` are not dropped but *aliased*: the bin ``m mod n_k`` is read from the
``rfft`` output directly when it lies in its half spectrum, and as the conjugate of bin
``n_k - (m mod n_k)`` otherwise. On HEALPix the short polar rings carry far fewer points
than the transform has orders, so dropping the aliased orders, as a zero-padded ``rfft``
would, is wrong there.

**Synthesis** is the reverse. A real field with coefficients :math:`G_m`,
:math:`0 \le m < m_{\max}`, has the two-sided expansion :math:`H_{\pm m}` with
:math:`H_{-m} = \overline{G_m}`, and the ring's DFT bin :math:`b` receives every
:math:`H_m` with :math:`m \equiv b \pmod{n_k}`. Each order is therefore *folded* into
the half-spectrum bins it aliases to -- the positive frequency into ``m mod n_k`` and the
negative one into ``(-m) mod n_k``, whichever of them lies in ``0 ... n_k / 2`` -- after
the phase :math:`e^{i m \lambda_{0,k}}`. An ``irfft`` of each ring finishes the step.

Both directions are exact. The ring FFTs are batched over the rings that share a length:
one call per distinct length, which on HEALPix is a single call for the equatorial belt
(two thirds of the pixels) and one per pair of mirrored polar-cap rings. The gather and the
fold are precomputed index tables, so a forward pass has no host synchronization.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from torch_harmonics.grid import GridS2
from torch_harmonics.utils import recast_buffers_for


def _ring_groups(nlon_per_lat: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
    """Rings grouped by length, as ``(length, ring indices)`` in order of first appearance."""
    lengths = nlon_per_lat.tolist()
    groups = {}
    for ring, n in enumerate(lengths):
        groups.setdefault(n, []).append(ring)
    return [(n, torch.tensor(rings, dtype=torch.int64)) for n, rings in groups.items()]


class RingFFT(nn.Module):
    r"""
    Exact longitudinal Fourier transform of a field on the rings of a :class:`GridS2`.

    See the module docstring for the definition. :meth:`forward` maps ``(..., npoints)``
    to complex ``(..., nrings, mmax)`` coefficients, normalized like
    ``torch.fft.rfft(norm="forward")``, and :meth:`inverse` maps them back, normalized
    like ``torch.fft.irfft(norm="forward")``. On a regular grid :meth:`forward` equals
    ``rfft`` for orders up to ``nlon / 2`` and aliases the ones above instead of
    zeroing them; :meth:`inverse` equals ``irfft`` for orders below ``nlon / 2``. At and
    above it, every order is a two-sided harmonic :math:`2\,\mathrm{Re}(G_m e^{i m
    \lambda})`, the Nyquist order included, where ``irfft`` would take its bin
    one-sided. On HEALPix those are genuine orders of the expansion on the short rings.

    Parameters
    ----------
    grid : GridS2
        The grid. Its :attr:`~torch_harmonics.grid.GridS2.nlon_per_lat`,
        :attr:`~torch_harmonics.grid.GridS2.lon_offsets` and
        :attr:`~torch_harmonics.grid.GridS2.lon_shifts` describe the rings.
    mmax : int
        Number of orders, non-inclusive.
    """

    def __init__(self, grid: GridS2, mmax: int):
        super().__init__()

        nlon_per_lat = grid.nlon_per_lat.to(torch.int64)
        offsets = grid.lon_offsets.to(torch.int64)
        self.nrings = int(nlon_per_lat.numel())
        self.npoints = int(offsets[-1])
        self.mmax = int(mmax)

        # every ring's half spectrum fits into nbins bins
        self.nbins = int(nlon_per_lat.max()) // 2 + 1

        # -- ring batches ----------------------------------------------------
        # One batch per distinct ring length. A batch whose rings are consecutive is a
        # contiguous pixel range, read with a slice; any other one with a gather.
        self._lengths = []
        self._contiguous = []
        ring_order = []
        pixel_order = []
        for i, (n, rings) in enumerate(_ring_groups(nlon_per_lat)):
            pixels = offsets[rings].unsqueeze(-1) + torch.arange(n, dtype=torch.int64)
            contiguous = bool((rings[1:] - rings[:-1] == 1).all()) if rings.numel() > 1 else True
            self._lengths.append(n)
            self._contiguous.append((int(pixels[0, 0]), rings.numel()) if contiguous else None)
            self.register_buffer(f"_pixels_{i}", pixels, persistent=False)
            self.register_buffer(f"_rings_{i}", rings, persistent=False)
            ring_order.append(rings)
            pixel_order.append(pixels.reshape(-1))

        # the batches come out concatenated; these permutations restore ring order
        ring_order = torch.cat(ring_order)
        pixel_order = torch.cat(pixel_order)
        self.register_buffer("_ring_inverse", torch.argsort(ring_order), persistent=False)
        self.register_buffer("_pixel_inverse", torch.argsort(pixel_order), persistent=False)

        # -- alias tables ----------------------------------------------------
        n = nlon_per_lat.unsqueeze(-1)
        m = torch.arange(self.mmax, dtype=torch.int64).unsqueeze(0)
        ring = torch.arange(self.nrings, dtype=torch.int64).unsqueeze(-1)
        half = n // 2

        # analysis: order m reads bin m mod n, or the conjugate of bin n - (m mod n)
        k = torch.remainder(m, n)
        conj = k > half
        k = torch.where(conj, n - k, k)
        self.register_buffer("_gather", (ring * self.nbins + k).reshape(-1), persistent=False)
        self.register_buffer("_gather_sign", torch.where(conj, -1.0, 1.0).to(torch.float64), persistent=False)

        # synthesis: positive frequency m to bin m mod n, negative frequency -m to bin
        # (-m) mod n, each dropped into a trailing dump bin when it falls outside the half
        # spectrum, and -0 dropped altogether since m = 0 has no separate negative part
        stride = self.nbins + 1
        dump = ring * stride + self.nbins
        k_pos = torch.remainder(m, n)
        k_neg = torch.remainder(-m, n)
        pos = torch.where(k_pos <= half, ring * stride + k_pos, dump)
        neg = torch.where((k_neg <= half) & (m > 0), ring * stride + k_neg, dump)
        self.register_buffer("_fold_pos", pos.reshape(-1), persistent=False)
        self.register_buffer("_fold_neg", neg.reshape(-1), persistent=False)

        # the DC and Nyquist bins of a real ring are real. Folding makes them so up to
        # round-off, and the imaginary part of m = 0 is dropped by definition, so both are
        # zeroed explicitly rather than left to the irfft backend. A precomputed mask
        # rather than fft.irfft's per-call one, which writes to the device from the host
        # and would synchronize once per ring length.
        imag_mask = torch.ones(self.nrings, stride, dtype=torch.float64)
        imag_mask[:, 0] = 0.0
        imag_mask[torch.arange(self.nrings), nlon_per_lat // 2] = 0.0
        self.register_buffer("_imag_mask", imag_mask.reshape(-1), persistent=False)

        # ring phases e^{-i m lambda_0}, lambda_0 = 2 pi shift / n
        lam0 = 2.0 * torch.pi * grid.lon_shifts.to(torch.float64) / nlon_per_lat.to(torch.float64)
        angle = -m.to(torch.float64) * lam0.unsqueeze(-1)
        self.register_buffer("_phase_re", torch.cos(angle), persistent=False)
        self.register_buffer("_phase_im", torch.sin(angle), persistent=False)

    def _apply(self, fn, recurse=True):
        # float64 buffers cannot move to MPS; see recast_buffers_for
        recast_buffers_for(self, fn)
        return super()._apply(fn, recurse)

    def _batch(self, i: int, name: str) -> torch.Tensor:
        return getattr(self, f"_{name}_{i}")

    def extra_repr(self) -> str:
        return f"nrings={self.nrings}, npoints={self.npoints}, mmax={self.mmax}, nbatches={len(self._lengths)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fourier coefficients of every ring.

        Parameters
        ----------
        x : torch.Tensor
            Real field of shape ``(..., npoints)``, in ring order.

        Returns
        -------
        torch.Tensor
            Complex coefficients of shape ``(..., nrings, mmax)``.
        """
        lead = x.shape[:-1]

        # rfft each batch of equal-length rings and pad its half spectrum to nbins
        spectra = []
        for i, n in enumerate(self._lengths):
            contiguous = self._contiguous[i]
            if contiguous is not None:
                start, count = contiguous
                rings = x.narrow(-1, start, count * n).reshape(*lead, count, n)
            else:
                rings = x[..., self._batch(i, "pixels")]
            spectrum = torch.fft.rfft(rings, dim=-1, norm="forward")
            spectra.append(nn.functional.pad(spectrum, (0, self.nbins - spectrum.shape[-1])))
        spectrum = torch.cat(spectra, dim=-2).index_select(-2, self._ring_inverse)

        # alias gather, in real arithmetic: F = Re X + i * sign * Im X at the gathered bin
        spectrum = spectrum.reshape(*lead, self.nrings * self.nbins)
        dtype = spectrum.real.dtype
        re = spectrum.real[..., self._gather].reshape(*lead, self.nrings, self.mmax)
        im = spectrum.imag[..., self._gather].reshape(*lead, self.nrings, self.mmax) * self._gather_sign.to(dtype)

        # ring phase
        p_re = self._phase_re.to(dtype)
        p_im = self._phase_im.to(dtype)
        return torch.complex(re * p_re - im * p_im, re * p_im + im * p_re)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Synthesize the rings from their Fourier coefficients.

        Parameters
        ----------
        x : torch.Tensor
            Complex coefficients of shape ``(..., nrings, mmax)``. The imaginary part of
            ``m = 0`` is ignored, as by ``irfft``.

        Returns
        -------
        torch.Tensor
            Real field of shape ``(..., npoints)``, in ring order.
        """
        lead = x.shape[:-2]
        dtype = x.real.dtype

        # undo the ring phase: multiply by e^{+i m lambda_0}, the conjugate of the stored phase
        p_re = self._phase_re.to(dtype)
        p_im = self._phase_im.to(dtype)
        re = x.real * p_re + x.imag * p_im
        im = x.imag * p_re - x.real * p_im
        re = re.reshape(*lead, self.nrings * self.mmax)
        im = im.reshape(*lead, self.nrings * self.mmax)

        # fold: H_m into bin m mod n and conj(H_m) into bin (-m) mod n
        size = self.nrings * (self.nbins + 1)
        out_re = re.new_zeros(*lead, size).index_add(-1, self._fold_pos, re).index_add(-1, self._fold_neg, re)
        out_im = im.new_zeros(*lead, size).index_add(-1, self._fold_pos, im).index_add(-1, self._fold_neg, -im)
        out_im = out_im * self._imag_mask.to(dtype)
        spectrum = torch.complex(out_re, out_im).reshape(*lead, self.nrings, self.nbins + 1)

        # irfft each batch of equal-length rings and restore pixel order
        fields = []
        for i, n in enumerate(self._lengths):
            rings = spectrum.index_select(-2, self._batch(i, "rings")).narrow(-1, 0, n // 2 + 1)
            fields.append(torch.fft.irfft(rings, n=n, dim=-1, norm="forward").reshape(*lead, -1))
        return torch.cat(fields, dim=-1).index_select(-1, self._pixel_inverse)
