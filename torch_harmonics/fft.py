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
Longitudinal Fourier transforms.

:func:`rfft` and :func:`irfft` wrap ``torch.fft`` with the padding and truncation an SHT
needs on a regular grid. :class:`RingFFT` is the same step on a ragged
:class:`~torch_harmonics.grid.GridS2` such as HEALPix.

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

from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn

from torch_harmonics.cache import SharedBufferMixin, shared_table
from torch_harmonics.grid import GridS2
from torch_harmonics.utils import ensure_contiguous


def _pad_dim_right(x: torch.Tensor, dim: int, target_size: int, value: float = 0.0) -> torch.Tensor:
    """Pad tensor along a single dimension to target_size (right-side only)."""
    ndim = x.ndim
    dim = dim if dim >= 0 else ndim + dim
    pad_amount = target_size - x.shape[dim]
    # F.pad expects (left, right) for last dim, then second-to-last, etc.
    pad_spec = [0] * (2 * ndim)
    pad_spec[(ndim - 1 - dim) * 2 + 1] = pad_amount
    return nn.functional.pad(x, tuple(pad_spec), value=value)


def rfft(x: torch.Tensor, nmodes: Optional[int] = None, dim: int = -1, **kwargs) -> torch.Tensor:
    """
    Real FFT with the correct padding behavior.
    If nmodes is given and larger than x.size(dim), x is zero-padded along dim before FFT.
    """

    if "n" in kwargs:
        raise ValueError("The 'n' argument is not allowed. Use 'nmodes' instead.")

    x = ensure_contiguous(fft.rfft(x, dim=dim, **kwargs))

    if nmodes is not None and nmodes > x.shape[dim]:
        x = _pad_dim_right(x, dim, nmodes, value=0.0)
    elif nmodes is not None and nmodes < x.shape[dim]:
        x = x.narrow(dim, 0, nmodes)

    return x


def irfft(x: torch.Tensor, n: Optional[int] = None, dim: int = -1, **kwargs) -> torch.Tensor:
    """
    Torch version of IRFFT handles paddign and truncation correctly.
    This routine only applies Hermitian symmetry to avoid artifacts which occur depending on the backend.
    """

    if n is None:
        n = 2 * (x.size(dim) - 1)

    # An explicitly sized empty spectrum synthesizes zero, retaining autograd.
    # FFT backends require at least one input bin even when n is provided.
    if x.size(dim) == 0 and n > 0:
        x = _pad_dim_right(x, dim, 1)

    # Zero the imaginary part of the DC bin (and Nyquist bin if present) to
    # enforce Hermitian symmetry. We do this functionally via torch.complex()
    # rather than in-place assignment (x[..., k].imag = 0.0) because the
    # in-place write on a strided complex view breaks autograd's backward pass
    # on CPU+MKL (DftiCommitDescriptor rejects the resulting stride pattern).
    imag_scale = torch.ones(x.size(dim), dtype=x.real.dtype, device=x.device)
    imag_scale[0] = 0.0
    if (n % 2 == 0) and (n // 2 < x.size(dim)):
        imag_scale[n // 2] = 0.0
    shape = [1] * x.ndim
    shape[dim] = -1
    imag_scale = imag_scale.reshape(*shape)
    x = torch.complex(x.real, x.imag * imag_scale)

    x = ensure_contiguous(fft.irfft(x, n=n, dim=dim, **kwargs))

    return x


def _ring_groups(nlon_per_lat: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
    """Rings grouped by length, as ``(length, ring indices)`` in order of first appearance."""
    lengths = nlon_per_lat.tolist()
    groups = {}
    for ring, n in enumerate(lengths):
        groups.setdefault(n, []).append(ring)
    return [(n, torch.tensor(rings, dtype=torch.int64)) for n, rings in groups.items()]


@shared_table
@torch.no_grad()
def _ring_fft_tables(grid: GridS2, mmax: int) -> Dict[str, object]:
    """
    The index and phase tables of a :class:`RingFFT`, and the layout of its ring batches.

    Depend on the grid's rings and ``mmax`` only, so every ring FFT of the same grid and
    order count -- forward and inverse SHTs alike -- shares one set, see
    :func:`~torch_harmonics.cache.shared_table`. The tensors are read-only.
    """
    nlon_per_lat = grid.nlon_per_lat.to(torch.int64)
    offsets = grid.lon_offsets.to(torch.int64)
    nrings = int(nlon_per_lat.numel())

    # every ring's half spectrum fits into nbins bins
    nbins = int(nlon_per_lat.max()) // 2 + 1

    # -- ring batches --------------------------------------------------------
    # One batch per distinct ring length. A batch whose rings are consecutive is a
    # contiguous pixel range, read with a slice; any other one with a gather.
    lengths = []
    contiguous = []
    batch_pixels = []
    batch_rings = []
    for n, rings in _ring_groups(nlon_per_lat):
        pixels = offsets[rings].unsqueeze(-1) + torch.arange(n, dtype=torch.int64)
        is_contiguous = bool((rings[1:] - rings[:-1] == 1).all()) if rings.numel() > 1 else True
        lengths.append(n)
        contiguous.append((int(pixels[0, 0]), rings.numel()) if is_contiguous else None)
        batch_pixels.append(pixels)
        batch_rings.append(rings)

    # the batches come out concatenated; these permutations restore ring order
    ring_order = torch.cat(batch_rings)
    pixel_order = torch.cat([pixels.reshape(-1) for pixels in batch_pixels])

    # -- alias tables --------------------------------------------------------
    n = nlon_per_lat.unsqueeze(-1)
    m = torch.arange(mmax, dtype=torch.int64).unsqueeze(0)
    ring = torch.arange(nrings, dtype=torch.int64).unsqueeze(-1)
    half = n // 2

    # analysis: order m reads bin m mod n, or the conjugate of bin n - (m mod n)
    k = torch.remainder(m, n)
    conj = k > half
    k = torch.where(conj, n - k, k)

    # synthesis: positive frequency m to bin m mod n, negative frequency -m to bin
    # (-m) mod n, each dropped into a trailing dump bin when it falls outside the half
    # spectrum, and -0 dropped altogether since m = 0 has no separate negative part
    stride = nbins + 1
    dump = ring * stride + nbins
    k_pos = torch.remainder(m, n)
    k_neg = torch.remainder(-m, n)
    pos = torch.where(k_pos <= half, ring * stride + k_pos, dump)
    neg = torch.where((k_neg <= half) & (m > 0), ring * stride + k_neg, dump)

    # the DC bin of a real ring, and the Nyquist bin of an even one, are real. Folding
    # makes them so up to
    # round-off, and the imaginary part of m = 0 is dropped by definition, so both are
    # zeroed explicitly rather than left to the irfft backend. A precomputed mask
    # rather than irfft's per-call one, which writes to the device from the host
    # and would synchronize once per ring length.
    imag_mask = torch.ones(nrings, stride, dtype=torch.float64)
    imag_mask[:, 0] = 0.0
    # an odd ring has no Nyquist bin: its last bin n // 2 is an ordinary frequency
    even = nlon_per_lat % 2 == 0
    imag_mask[torch.arange(nrings)[even], nlon_per_lat[even] // 2] = 0.0

    # ring phases e^{-i m lambda_0}, lambda_0 = 2 pi shift / n
    lam0 = 2.0 * torch.pi * grid.lon_shifts.to(torch.float64) / nlon_per_lat.to(torch.float64)
    angle = -m.to(torch.float64) * lam0.unsqueeze(-1)

    return {
        "nbins": nbins,
        "lengths": tuple(lengths),
        "contiguous": tuple(contiguous),
        "pixels": tuple(batch_pixels),
        "rings": tuple(batch_rings),
        "ring_inverse": torch.argsort(ring_order),
        "pixel_inverse": torch.argsort(pixel_order),
        "gather": (ring * nbins + k).reshape(-1),
        "gather_sign": torch.where(conj, -1.0, 1.0).to(torch.float64),
        "fold_pos": pos.reshape(-1),
        "fold_neg": neg.reshape(-1),
        "imag_mask": imag_mask.reshape(-1),
        "phase_re": torch.cos(angle),
        "phase_im": torch.sin(angle),
    }


class RingFFT(SharedBufferMixin, nn.Module):
    r"""
    Exact longitudinal Fourier transform of a field on the rings of a :class:`GridS2`.

    See the :mod:`torch_harmonics.fft` module docstring for the definition. :meth:`forward` maps ``(..., npoints)``
    to complex ``(..., nrings, mmax)`` coefficients, normalized like
    ``torch.fft.rfft(norm="forward")``, and :meth:`inverse` maps them back, normalized
    like ``torch.fft.irfft(norm="forward")``. On a regular grid :meth:`forward` equals
    ``rfft`` for orders up to ``nlon / 2`` and aliases the ones above instead of
    zeroing them; :meth:`inverse` equals ``irfft`` for orders below ``nlon / 2``. At and
    above it, every order is a two-sided harmonic :math:`2\,\mathrm{Re}(G_m e^{i m
    \lambda})`, the Nyquist order included, where ``irfft`` would take its bin
    one-sided. On HEALPix those are genuine orders of the expansion on the short rings.

    The index and phase tables are shared between all ring FFTs of the same grid and
    ``mmax``, on every device; see :func:`~torch_harmonics.cache.shared_table`.

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

        self.nrings = int(grid.nlon_per_lat.numel())
        self.npoints = int(grid.lon_offsets[-1])
        self.mmax = int(mmax)

        tables = _ring_fft_tables(grid, self.mmax)
        self.nbins = tables["nbins"]
        self._lengths = list(tables["lengths"])
        self._contiguous = list(tables["contiguous"])
        for i, (pixels, rings) in enumerate(zip(tables["pixels"], tables["rings"])):
            self.register_shared_buffer(f"_pixels_{i}", pixels)
            self.register_shared_buffer(f"_rings_{i}", rings)
        for name in ("ring_inverse", "pixel_inverse", "gather", "gather_sign", "fold_pos", "fold_neg", "imag_mask", "phase_re", "phase_im"):
            self.register_shared_buffer(f"_{name}", tables[name])

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
