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

"""
Helpers for constructing the discrete-continuous (DISCO) convolution tensor on
the 2-sphere. Shared between the disco convolution modules and the attention
modules; kept at the top level so neither needs to depend on the other.

Three public-internal helpers:

  * `_precompute_convolution_tensor_s2` — build the COO sparse psi (forward
    direction) for an (in_shape, out_shape) pair, including normalization and
    optional quadrature merging.

  * `_normalize_convolution_tensor_s2` — value-normalization step factored out
    of `_precompute_convolution_tensor_s2` so it can be exercised directly by
    tests / alternate callers.

  * `_transpose_convolution_tensor_s2` — given a forward psi (output of
    `_precompute_convolution_tensor_s2` plus its preprocessed forms), produce
    the row-transposed CSR view used by the gather-based backward kernel.
"""

import math
import warnings
from typing import Tuple, Optional

import torch

from torch_harmonics.cache import lru_cache
from torch_harmonics.quadrature import precompute_latitudes, precompute_longitudes
from torch_harmonics.filter_basis import FilterBasis


def _normalize_convolution_tensor_s2(
    psi_idx,
    psi_vals,
    in_shape,
    out_shape,
    kernel_size,
    quad_weights,
    theta_cutoff,
    transpose_normalization=False,
    basis_norm_mode="mean",
    merge_quadrature=False,
    isotropic_mask=None,
    eps=1e-9,
):
    """Normalizes convolution tensor values based on specified normalization mode.

    This function applies different normalization strategies to the convolution tensor
    values based on the basis_norm_mode parameter. It can normalize individual basis
    functions, compute mean normalization across all basis functions, or use support
    weights. The function also optionally merges quadrature weights into the tensor.

    Parameters
    -----------
    psi_idx: torch.Tensor
        Index tensor for the sparse convolution tensor.
    psi_vals: torch.Tensor
        Value tensor for the sparse convolution tensor.
    in_shape: Tuple[int]
        Tuple of (nlat_in, nlon_in) representing input grid dimensions.
    out_shape: Tuple[int]
        Tuple of (nlat_out, nlon_out) representing output grid dimensions.
    kernel_size: int
        Number of kernel basis functions.
    quad_weights: torch.Tensor
        Quadrature weights for numerical integration.
    theta_cutoff: float
        Angular cutoff of the filter support (radians). Required by the "geometric" mode,
        which normalizes by the theoretical area measure of the spherical cap of half-angle
        theta_cutoff; unused by other modes.
    transpose_normalization: bool
        If True, applies normalization in transpose direction.
    basis_norm_mode: str
        Normalization mode, one of ["none", "nodal", "modal", "mean", "support", "geometric"].
        The legacy names "individual" and "area ratio" are accepted as deprecated aliases
        for "nodal" and "geometric" respectively; each emits a DeprecationWarning.
    merge_quadrature: bool
        If True, multiplies values by quadrature weights.
    isotropic_mask: Optional[Sequence[bool]]
        Per-kernel-index boolean mask; True marks an axisymmetric (m=0) basis function.
        Used by the "modal" mode to decide which kernels get a weighted-mean bias
        subtraction (anisotropic only). If None, only kernel index 0 is treated as isotropic.
    eps: float
        Small epsilon value to prevent division by zero.

    Returns
    -------
    torch.Tensor
        Normalized convolution tensor values.

    Raises
    ------
    ValueError
        If basis_norm_mode is not one of the supported modes.
    """

    if basis_norm_mode == "individual":
        warnings.warn(
            'basis_norm_mode="individual" is deprecated, use "nodal" instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        basis_norm_mode = "nodal"
    elif basis_norm_mode == "area ratio":
        warnings.warn(
            'basis_norm_mode="area ratio" is deprecated, use "geometric" instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        basis_norm_mode = "geometric"

    # reshape the indices implicitly to be ikernel, out_shape[0], in_shape[0], in_shape[1]
    idx = torch.stack([psi_idx[0], psi_idx[1], psi_idx[2] // in_shape[1], psi_idx[2] % in_shape[1]], dim=0)

    # getting indices for adressing kernels, input and output latitudes
    ikernel = idx[0]

    if transpose_normalization:
        ilat_out = idx[2]
        ilat_in = idx[1]
        # here we are deliberately swapping input and output shapes to handle transpose normalization with the same code
        nlat_out = in_shape[0]
        correction_factor = out_shape[1] / in_shape[1]
    else:
        ilat_out = idx[1]
        ilat_in = idx[2]
        nlat_out = out_shape[0]

    # get the quadrature weights
    q = quad_weights[ilat_in].reshape(-1)

    # buffer to store intermediate values
    bias = torch.zeros(kernel_size, nlat_out, dtype=psi_vals.dtype, device=psi_vals.device)
    scale = torch.zeros(kernel_size, nlat_out, dtype=psi_vals.dtype, device=psi_vals.device)
    support = torch.zeros(kernel_size, nlat_out, dtype=psi_vals.dtype, device=psi_vals.device)

    # loop through dimensions to compute the norms
    for ik in range(kernel_size):
        for ilat in range(nlat_out):

            # find indices corresponding to the given output latitude and kernel basis function
            iidx = torch.argwhere((ikernel == ik) & (ilat_out == ilat))

            # compute the support
            support[ik, ilat] = torch.sum(q[iidx])

            # for modal normalization, subtract the weighted mean from anisotropic modes
            # so that directional modes integrate to zero over their support
            is_isotropic = isotropic_mask[ik] if isotropic_mask is not None else (ik == 0)
            if basis_norm_mode == "modal" and not is_isotropic and support[ik, ilat].abs() > eps:
                bias[ik, ilat] = torch.sum(psi_vals[iidx] * q[iidx]) / support[ik, ilat]

            scale[ik, ilat] = torch.sum((psi_vals[iidx] - bias[ik, ilat]).abs() * q[iidx])

    # precompute the per-ik mean for "mean" mode so we don't rely on Python function-scope
    # reuse of b/s across ilat iterations inside the loop below
    if basis_norm_mode == "mean":
        bias_per_ik = bias.mean(dim=1)
        scale_per_ik = scale.mean(dim=1)

    # precompute the "geometric" scalar once; it's ik/ilat-independent
    if basis_norm_mode == "geometric":
        geometric_scale = (1.0 - math.cos(theta_cutoff)) / 2.0 / 2.0

    # loop over values and renormalize
    for ik in range(kernel_size):
        for ilat in range(nlat_out):

            iidx = torch.argwhere((ikernel == ik) & (ilat_out == ilat))

            if basis_norm_mode in ["nodal", "modal"]:
                b = bias[ik, ilat]
                s = scale[ik, ilat]
            elif basis_norm_mode == "mean":
                b = bias_per_ik[ik]
                s = scale_per_ik[ik]
            elif basis_norm_mode == "support":
                b = 0.0
                s = support[ik, ilat]
            elif basis_norm_mode == "geometric":
                b = 0.0
                s = geometric_scale
            elif basis_norm_mode == "none":
                b = 0.0
                s = 1.0
            else:
                raise ValueError(f"Unknown basis normalization mode {basis_norm_mode}.")

            psi_vals[iidx] = (psi_vals[iidx] - b) / max(s, eps)

            if merge_quadrature:
                psi_vals[iidx] = psi_vals[iidx] * q[iidx]

    if transpose_normalization and merge_quadrature:
        psi_vals = psi_vals / correction_factor

    return psi_vals


@lru_cache(typed=True, copy=True)
def _precompute_convolution_tensor_s2(
    in_shape: Tuple[int],
    out_shape: Tuple[int],
    filter_basis: FilterBasis,
    grid_in: Optional[str] = "equiangular",
    grid_out: Optional[str] = "equiangular",
    theta_cutoff: Optional[float] = 0.01 * math.pi,
    theta_eps: Optional[float] = 1e-3,
    transpose_normalization: Optional[bool] = False,
    basis_norm_mode: Optional[str] = "nodal",
    merge_quadrature: Optional[bool] = False,
):
    r"""
    Precomputes the rotated filters at positions $R^{-1}_j \omega_i = R^{-1}_j R_i \nu = Y(-\theta_j)Z(\phi_i - \phi_j)Y(\theta_j)\nu$.
    Assumes a tensorized grid on the sphere with an equidistant sampling in longitude as described in Ocampo et al.
    The output tensor has shape kernel_shape x nlat_out x (nlat_in * nlon_in).

    The rotation of the Euler angles uses the YZY convention, which applied to the northpole $(0,0,1)^T$ yields
    $$
    Y(\alpha) Z(\beta) Y(\gamma) n =
        {\begin{bmatrix}
            \cos(\gamma)\sin(\alpha) + \cos(\alpha)\cos(\beta)\sin(\gamma) \\
            \sin(\beta)\sin(\gamma) \\
            \cos(\alpha)\cos(\gamma)-\cos(\beta)\sin(\alpha)\sin(\gamma)
        \end{bmatrix}}
    $$

    Parameters
    -----------
    in_shape: Tuple[int]
        Input shape of the convolution tensor
    out_shape: Tuple[int]
        Output shape of the convolution tensor
    filter_basis: FilterBasis
        Filter basis functions
    grid_in: str
        Input grid type
    grid_out: str
        Output grid type
    theta_cutoff: float
        Theta cutoff for the filter basis functions
    theta_eps: float
        Epsilon for the theta cutoff
    transpose_normalization: bool
        Whether to normalize the convolution tensor in the transpose direction
    basis_norm_mode: str
        Mode for basis normalization
    merge_quadrature: bool
        Whether to merge the quadrature weights into the convolution tensor

    Returns
    -------
    out_idx: torch.Tensor
        Index tensor of the convolution tensor
    out_vals: torch.Tensor
        Values tensor of the convolution tensor

    """

    assert len(in_shape) == 2
    assert len(out_shape) == 2

    kernel_size = filter_basis.kernel_size

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    # precompute input and output grids
    lats_in, win = precompute_latitudes(nlat_in, grid=grid_in)
    lats_out, wout = precompute_latitudes(nlat_out, grid=grid_out)

    # compute the phi differences
    # It's imporatant to not include the 2 pi point in the longitudes, as it is equivalent to lon=0
    lons_in = precompute_longitudes(nlon_in)

    # compute quadrature weights and merge them into the convolution tensor.
    # These quadrature integrate to 1 over the sphere.
    if transpose_normalization:
        quad_weights = wout.reshape(-1, 1) / nlon_in / 2.0
    else:
        quad_weights = win.reshape(-1, 1) / nlon_in / 2.0

    # effective theta cutoff if multiplied with a fudge factor to avoid aliasing with grid width (especially near poles)
    theta_cutoff_eff = (1.0 + theta_eps) * theta_cutoff

    out_idx = []
    out_vals = []

    beta = lons_in
    gamma = lats_in.reshape(-1, 1)

    # compute trigs
    cbeta = torch.cos(beta)
    sbeta = torch.sin(beta)
    cgamma = torch.cos(gamma)
    sgamma = torch.sin(gamma)

    # compute row offsets
    out_roff = torch.zeros(nlat_out + 1, dtype=torch.int64, device=lons_in.device)
    out_roff[0] = 0
    for t in range(nlat_out):
        # the last angle has a negative sign as it is a passive rotation, which rotates the filter around the y-axis
        alpha = -lats_out[t]

        # compute cartesian coordinates of the rotated position
        # This uses the YZY convention of Euler angles, where the last angle (alpha) is a passive rotation,
        # and therefore applied with a negative sign
        x = torch.cos(alpha) * cbeta * sgamma + cgamma * torch.sin(alpha)
        y = sbeta * sgamma
        z = -cbeta * torch.sin(alpha) * sgamma + torch.cos(alpha) * cgamma

        # normalization is important to avoid NaNs when arccos and atan are applied
        # this can otherwise lead to spurious artifacts in the solution
        norm = torch.sqrt(x * x + y * y + z * z)
        x = x / norm
        y = y / norm
        z = z / norm

        # compute spherical coordinates, where phi needs to fall into the [0, 2pi) range
        theta = torch.arccos(z)
        phi = torch.arctan2(y, x)
        phi = torch.where(phi < 0.0, phi + 2 * torch.pi, phi)

        # find the indices where the rotated position falls into the support of the kernel
        iidx, vals = filter_basis.compute_support_vals(theta, phi, r_cutoff=theta_cutoff_eff)

        # add the output latitude and reshape such that psi has dimensions kernel_shape x nlat_out x (nlat_in*nlon_in)
        idx = torch.stack([iidx[:, 0], t * torch.ones_like(iidx[:, 0]), iidx[:, 1] * nlon_in + iidx[:, 2]], dim=0)

        # append indices and values to the COO datastructure, compute row offsets
        out_idx.append(idx)
        out_vals.append(vals)
        out_roff[t + 1] = out_roff[t] + iidx.shape[0]

    # concatenate the indices and values
    out_idx = torch.cat(out_idx, dim=-1)
    out_vals = torch.cat(out_vals, dim=-1)

    out_vals = _normalize_convolution_tensor_s2(
        out_idx,
        out_vals,
        in_shape,
        out_shape,
        kernel_size,
        quad_weights,
        theta_cutoff,
        transpose_normalization=transpose_normalization,
        basis_norm_mode=basis_norm_mode,
        merge_quadrature=merge_quadrature,
        isotropic_mask=filter_basis.isotropic_mask,
    )

    out_idx = out_idx.contiguous()
    out_vals = out_vals.contiguous()

    return out_idx, out_vals, out_roff


def _transpose_convolution_tensor_s2(
    ker_idx: torch.Tensor,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    vals: torch.Tensor,
    in_shape: Tuple[int, int],
    out_shape: Tuple[int, int],
):
    """
    Construct the row-transposed CSR view (psi_T) used by the gather-based
    backward kernel.

    Forward psi maps (k, ho, nz) -> (hi, wi_offset) with
        col_idx[z] = hi * Wi + wi_offset,   wi_offset in [0, Wi).
    The forward kernel then produces
        out[b,c,k,ho,wo] += val * inp[b,c,hi,(wi_offset + pscale*wo) mod Wi],   pscale = Wi/Wo.

    For the backward (gradient w.r.t. inp), we need a structure indexed by
    `hi` (the latitude on the *input* grid) so each output cell
    grad_inp[b,c,hi,wi] gathers its contributions without atomicAdd.

    Bucketing by `wi_offset % pscale`: a forward entry only contributes to
    backward output cells whose `wi % pscale` equals `wi_offset % pscale`,
    so we group entries into `pscale` buckets per `hi`. The backward kernel
    then iterates only the relevant bucket per output cell.

    Output layout (sorted by row_T, ker secondary):
        row_T   = hi * pscale + (wi_offset % pscale)   in [0, Hi*pscale)
        col_T   = ho * Wi + wi_offset                  in [0, Ho*Wi)
        ker_T   = ker (kept per-entry; rows mix kernel indices since the
                  backward contracts over k_kern)
        roff_T  = row offsets, length Hi*pscale + 1

    Returns (ker_idx_T, col_idx_T, vals_T, roff_idx_T).
    """
    Hi, Wi = in_shape
    Ho, Wo = out_shape
    assert Wi % Wo == 0, f"Wi ({Wi}) must be a multiple of Wo ({Wo}) for psi_T construction"
    pscale = Wi // Wo

    ho = row_idx
    hi = col_idx // Wi
    wi_offset = col_idx % Wi
    r = wi_offset % pscale

    row_T = hi * pscale + r
    col_T = ho * Wi + wi_offset

    # stable sort by row_T (primary). ker_idx ordering within a row_T bucket
    # is irrelevant for correctness; the backward kernel sums over all entries
    # in a bucket regardless of ker order.
    perm = torch.argsort(row_T, stable=True)

    ker_idx_T = ker_idx[perm].contiguous()
    col_idx_T = col_T[perm].contiguous()
    vals_T = vals[perm].contiguous()
    row_T_sorted = row_T[perm]

    Nrow_T = Hi * pscale
    counts = torch.bincount(row_T_sorted, minlength=Nrow_T).to(torch.int64)
    roff_idx_T = torch.zeros(Nrow_T + 1, dtype=torch.int64, device=ker_idx.device)
    roff_idx_T[1:] = torch.cumsum(counts, dim=0)

    return ker_idx_T, col_idx_T, vals_T, roff_idx_T


def _transpose_psi_kpacked(
    psi_kpacked_idx: torch.Tensor,    # [Ho, NBR_PAD, 2]   -> (hi, wi_offset)
    psi_kpacked_vals: torch.Tensor,   # [Ho, NBR_PAD, K_pad]
    psi_kpacked_count: torch.Tensor,  # [Ho]
    in_shape: Tuple[int, int],        # (Hi, Wi)  — bigger grid
    out_shape: Tuple[int, int],       # (Ho, Wo)  — smaller grid
):
    """
    Build K-packed psi_T from K-packed forward psi.

    For bases where all K basis functions share the same (hi, wi_offset)
    support per ho (harmonic, morlet, zernike, fourier-bessel — NOT
    piecewise-linear), the K-packed forward layout stores K values per
    (ho, nz) neighbor slot. This routine produces the analogous structure
    indexed by `hi*pscale + (wi_offset%pscale)` for the backward gather:

      Output layout (K-packed psi_T):
        psi_T_kpacked_idx   : [Hi*pscale, NBR_PAD_T, 2]      int64
                              -> (ho, wi_offset)
        psi_T_kpacked_vals  : [Hi*pscale, NBR_PAD_T, K_pad]  same dtype as forward
        psi_T_kpacked_count : [Hi*pscale]                    int64

    The split-by-K gather backward kernel issues one CTA per (bc, ho_kernel, k)
    and reads `vals[row_T, nz, k]` (one column of the K-packed vector) per
    entry — this keeps the parallelism count high (K× more CTAs than the
    plain gather) while paying only mild K-way atomic contention on the
    output. Returns None when the input isn't actually K-shared (e.g.
    piecewise-linear bases), so the caller can fall back.
    """
    Hi, Wi = in_shape
    Ho, Wo = out_shape
    assert Wi % Wo == 0, f"Wi ({Wi}) must be a multiple of Wo ({Wo})"
    pscale = Wi // Wo

    Ho_in, NBR_PAD, two = psi_kpacked_idx.shape
    assert two == 2 and Ho_in == Ho, "psi_kpacked_idx shape mismatch"
    K_pad = psi_kpacked_vals.shape[2]
    assert psi_kpacked_vals.shape[:2] == (Ho, NBR_PAD), "psi_kpacked_vals shape mismatch"
    assert psi_kpacked_count.shape == (Ho,), "psi_kpacked_count shape mismatch"

    device = psi_kpacked_idx.device

    # Per-(ho, nz_slot) validity mask: nz_slot < count[ho].
    nz_axis  = torch.arange(NBR_PAD, device=device)
    valid    = nz_axis.view(1, NBR_PAD) < psi_kpacked_count.view(Ho, 1)   # [Ho, NBR_PAD]

    # Decode (hi, wi_offset) per slot.
    hi_all        = psi_kpacked_idx[:, :, 0]                              # [Ho, NBR_PAD]
    wi_offset_all = psi_kpacked_idx[:, :, 1]                              # [Ho, NBR_PAD]
    ho_axis       = torch.arange(Ho, device=device).view(Ho, 1).expand(Ho, NBR_PAD)

    # Bucket: row_T = hi*pscale + r, where r = wi_offset % pscale.
    r_all     = wi_offset_all % pscale
    row_T_all = hi_all * pscale + r_all                                   # [Ho, NBR_PAD]

    # Flatten valid entries.
    valid_flat        = valid.reshape(-1)
    row_T_flat        = row_T_all.reshape(-1)[valid_flat]
    ho_flat           = ho_axis.reshape(-1)[valid_flat]
    wi_offset_flat    = wi_offset_all.reshape(-1)[valid_flat]
    vals_flat         = psi_kpacked_vals.reshape(Ho * NBR_PAD, K_pad)[valid_flat]    # [N_valid, K_pad]

    # Sort by row_T to bucket entries.
    perm              = torch.argsort(row_T_flat, stable=True)
    row_T_sorted      = row_T_flat[perm]
    ho_sorted         = ho_flat[perm]
    wi_offset_sorted  = wi_offset_flat[perm]
    vals_sorted       = vals_flat[perm]                                   # [N_valid, K_pad]

    # Counts per row_T → NBR_PAD_T = max.
    Nrows_T  = Hi * pscale
    counts_T = torch.bincount(row_T_sorted, minlength=Nrows_T).to(torch.int64)
    NBR_PAD_T = int(counts_T.max().item())
    if NBR_PAD_T == 0:
        # Degenerate: no entries at all (very small theta_cutoff).
        return (
            torch.zeros(Nrows_T, 0, 2, dtype=torch.int64, device=device),
            torch.zeros(Nrows_T, 0, K_pad, dtype=psi_kpacked_vals.dtype, device=device),
            counts_T,
        )

    # Position of each sorted entry within its row_T bucket = z - cumcount_prefix.
    roff_T   = torch.zeros(Nrows_T + 1, dtype=torch.int64, device=device)
    roff_T[1:] = torch.cumsum(counts_T, dim=0)
    z_axis   = torch.arange(row_T_sorted.numel(), device=device)
    pos_in_row = z_axis - roff_T[row_T_sorted]

    # Scatter into dense [Nrows_T, NBR_PAD_T, ...] tensors.
    idx_T = torch.zeros(Nrows_T, NBR_PAD_T, 2, dtype=torch.int64, device=device)
    vals_T = torch.zeros(Nrows_T, NBR_PAD_T, K_pad,
                         dtype=psi_kpacked_vals.dtype, device=device)
    idx_T[row_T_sorted, pos_in_row, 0]   = ho_sorted
    idx_T[row_T_sorted, pos_in_row, 1]   = wi_offset_sorted
    vals_T[row_T_sorted, pos_in_row, :]  = vals_sorted

    return idx_T, vals_T, counts_T
