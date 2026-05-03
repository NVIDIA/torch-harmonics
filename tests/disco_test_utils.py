# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Shared test-only helpers for disco tests:
#   - dense reference construction of the psi convolution tensor (used as a
#     ground-truth oracle in test_convolution.py)
#   - reference Python implementations of the dense-packed psi repacker and
#     a slow forward consuming the packed layout (oracle for the C++ packer
#     and the CUDA kernels in test_convolution_tensor.py)

import math

import torch

from torch_harmonics.quadrature import precompute_latitudes, precompute_longitudes


# ---------------------------------------------------------------------------
# Dense reference construction of the psi convolution tensor
# ---------------------------------------------------------------------------

def normalize_convolution_tensor_dense(
    psi,
    quad_weights,
    transpose_normalization=False,
    basis_norm_mode="none",
    merge_quadrature=False,
    isotropic_mask=None,
    theta_cutoff=None,
    in_support=None,
    eps=1e-9,
):
    """Discretely normalizes the convolution tensor.

    Mirrors the normalization logic in _normalize_convolution_tensor_s2
    for all supported normalization modes.
    """

    kernel_size, nlat_out, nlon_out, nlat_in, nlon_in = psi.shape
    correction_factor = nlon_out / nlon_in

    if transpose_normalization:
        n_olat = nlat_in
    else:
        n_olat = nlat_out

    bias_arr = torch.zeros(kernel_size, n_olat, dtype=psi.dtype, device=psi.device)
    scale_arr = torch.zeros(kernel_size, n_olat, dtype=psi.dtype, device=psi.device)
    support_arr = torch.zeros(kernel_size, n_olat, dtype=psi.dtype, device=psi.device)

    for ik in range(kernel_size):
        for ilat in range(n_olat):
            if transpose_normalization:
                entries = psi[ik, :, 0, ilat, :]
                q = quad_weights[:nlat_out, 0].unsqueeze(1).expand_as(entries)
                smask = in_support[ik, :, 0, ilat, :] if in_support is not None else (entries.abs() > 0)
            else:
                entries = psi[ik, ilat, 0, :, :]
                q = quad_weights[:nlat_in, 0].unsqueeze(1).expand_as(entries)
                smask = in_support[ik, ilat, 0, :, :] if in_support is not None else (entries.abs() > 0)

            q_masked = q * smask
            support_arr[ik, ilat] = q_masked.sum()

            is_isotropic = isotropic_mask[ik] if isotropic_mask is not None else (ik == 0)
            if basis_norm_mode == "modal" and not is_isotropic and support_arr[ik, ilat].abs() > eps:
                bias_arr[ik, ilat] = (entries * q_masked).sum() / support_arr[ik, ilat]

            scale_arr[ik, ilat] = ((entries - bias_arr[ik, ilat]).abs() * q_masked).sum()

    # The sparse implementation stores one longitude slice and reuses it for all
    # output longitudes via rolling during contraction. We mirror this: normalize
    # only the r=0 slice, then fill other slices with cyclic shifts. Normalizing
    # each slice independently would amplify floating-point noise at near-zero
    # entries (e.g. anisotropic modes at the poles where scale ≈ 0).
    pscale = nlon_in // nlon_out

    # precompute the per-ik mean for "mean" mode so we don't rely on Python function-scope
    # reuse of b/s across ilat iterations inside the loop below
    if basis_norm_mode == "mean":
        bias_per_ik = bias_arr.mean(dim=1)
        scale_per_ik = scale_arr.mean(dim=1)

    # precompute the "geometric" scalar once; it's ik/ilat-independent
    if basis_norm_mode == "geometric":
        geometric_scale = (1.0 - math.cos(theta_cutoff)) / 2.0 / 2.0

    for ik in range(kernel_size):
        for ilat in range(n_olat):
            if basis_norm_mode in ["nodal", "modal"]:
                b = bias_arr[ik, ilat]
                s = scale_arr[ik, ilat]
            elif basis_norm_mode == "mean":
                b = bias_per_ik[ik]
                s = scale_per_ik[ik]
            elif basis_norm_mode == "support":
                b = 0.0
                s = support_arr[ik, ilat]
            elif basis_norm_mode == "geometric":
                b = 0.0
                s = geometric_scale
            elif basis_norm_mode == "none":
                b = 0.0
                s = 1.0
            else:
                raise ValueError(f"Unknown basis normalization mode {basis_norm_mode}.")

            if transpose_normalization:
                slc0 = psi[ik, :, 0, ilat, :]
                mask0 = in_support[ik, :, 0, ilat, :] if in_support is not None else (slc0 != 0)
                psi[ik, :, 0, ilat, :] = torch.where(mask0, (slc0 - b) / max(s, eps), slc0)
                for r in range(1, nlon_out):
                    psi[ik, :, r, ilat, :] = torch.roll(psi[ik, :, 0, ilat, :], r * pscale, dims=-1)
            else:
                slc0 = psi[ik, ilat, 0, :, :]
                mask0 = in_support[ik, ilat, 0, :, :] if in_support is not None else (slc0 != 0)
                psi[ik, ilat, 0, :, :] = torch.where(mask0, (slc0 - b) / max(s, eps), slc0)
                for r in range(1, nlon_out):
                    psi[ik, ilat, r, :, :] = torch.roll(psi[ik, ilat, 0, :, :], r * pscale, dims=-1)

    if transpose_normalization:
        if merge_quadrature:
            psi = quad_weights.reshape(1, -1, 1, 1, 1) * psi / correction_factor
    else:
        if merge_quadrature:
            psi = quad_weights.reshape(1, 1, 1, -1, 1) * psi

    return psi


def precompute_convolution_tensor_dense(
    in_shape,
    out_shape,
    filter_basis,
    grid_in="equiangular",
    grid_out="equiangular",
    theta_cutoff=0.01 * math.pi,
    theta_eps=1e-3,
    transpose_normalization=False,
    basis_norm_mode="none",
    merge_quadrature=False,
):
    """Helper routine to compute the convolution Tensor in a dense fashion."""
    assert len(in_shape) == 2
    assert len(out_shape) == 2

    kernel_size = filter_basis.kernel_size

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    lats_in, win = precompute_latitudes(nlat_in, grid=grid_in)
    lats_out, wout = precompute_latitudes(nlat_out, grid=grid_out)

    # compute the phi differences.
    lons_in = precompute_longitudes(nlon_in)
    lons_out = precompute_longitudes(nlon_out)

    # effective theta cutoff if multiplied with a fudge factor to avoid aliasing with grid width (especially near poles)
    theta_cutoff_eff = (1.0 + theta_eps) * theta_cutoff

    # compute quadrature weights that will be merged into the Psi tensor
    if transpose_normalization:
        quad_weights = wout.reshape(-1, 1) / nlon_in / 2.0
    else:
        quad_weights = win.reshape(-1, 1) / nlon_in / 2.0

    # array for accumulating non-zero indices and tracking filter support
    out = torch.zeros(kernel_size, nlat_out, nlon_out, nlat_in, nlon_in, dtype=torch.float64, device=lons_in.device)
    in_support = torch.zeros_like(out, dtype=torch.bool)

    for t in range(nlat_out):
        for p in range(nlon_out):
            alpha = -lats_out[t]
            beta = lons_in - lons_out[p]
            gamma = lats_in.reshape(-1, 1)

            # compute latitude of the rotated position
            z = -torch.cos(beta) * torch.sin(alpha) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma)

            # compute cartesian coordinates of the rotated position
            x = torch.cos(alpha) * torch.cos(beta) * torch.sin(gamma) + torch.cos(gamma) * torch.sin(alpha)
            y = torch.sin(beta) * torch.sin(gamma) * torch.ones_like(alpha)

            # normalize instead of clipping to ensure correct range
            norm = torch.sqrt(x * x + y * y + z * z)
            x = x / norm
            y = y / norm
            z = z / norm

            # compute spherical coordinates
            theta = torch.arccos(z)
            phi = torch.arctan2(y, x)
            phi = torch.where(phi < 0.0, phi + 2 * torch.pi, phi)

            # find the indices where the rotated position falls into the support of the kernel
            iidx, vals = filter_basis.compute_support_vals(theta, phi, r_cutoff=theta_cutoff_eff)
            out[iidx[:, 0], t, p, iidx[:, 1], iidx[:, 2]] = vals
            in_support[iidx[:, 0], t, p, iidx[:, 1], iidx[:, 2]] = True

    # take care of normalization
    out = normalize_convolution_tensor_dense(
        out,
        quad_weights=quad_weights,
        transpose_normalization=transpose_normalization,
        basis_norm_mode=basis_norm_mode,
        merge_quadrature=merge_quadrature,
        isotropic_mask=filter_basis.isotropic_mask,
        theta_cutoff=theta_cutoff,
        in_support=in_support,
    )

    return out


# ---------------------------------------------------------------------------
# Reference implementations for the dense-packed psi format
# ---------------------------------------------------------------------------

def python_pack_psi(K, Ho, Wi, nbr_pad, ker_idx, row_idx, col_idx, vals, roff_idx):
    """Reference packer matching disco_helpers.pack_psi_dense.

    Inputs (already produced by preprocess_psi):
        K, Ho, Wi: int     — kernel-basis count, output-lat count, input-lon count.
        nbr_pad:   int     — target padding along the neighbor dim. If <= 0, auto-set
                              to the maximum row length.
        ker_idx, row_idx, col_idx: int64 [nnz]
        vals:      float [nnz]
        roff_idx:  int64 [K*Ho + 1]

    Returns:
        idx_out:   int64 [K, Ho, NBR_PAD, 2]   — last dim is (hi, wi_base)
        val_out:   vals.dtype [K, Ho, NBR_PAD]
        count_out: int64 [K, Ho]
    """
    nrows = roff_idx.numel() - 1
    if nrows != K * Ho:
        raise ValueError(f"expected nrows == K*Ho ({K*Ho}), got {nrows}")

    diffs = roff_idx[1:] - roff_idx[:-1]
    max_nbr = int(diffs.max().item()) if nrows > 0 else 0

    if nbr_pad <= 0:
        nbr_pad = max_nbr
    elif nbr_pad < max_nbr:
        raise ValueError(f"nbr_pad ({nbr_pad}) smaller than max row length ({max_nbr})")

    idx_out = torch.zeros(K, Ho, nbr_pad, 2, dtype=torch.int64)
    val_out = torch.zeros(K, Ho, nbr_pad, dtype=vals.dtype)
    count_out = torch.zeros(K, Ho, dtype=torch.int64)

    ker_h = ker_idx.cpu()
    row_h = row_idx.cpu()
    col_h = col_idx.cpu()
    val_h = vals.cpu()
    roff_h = roff_idx.cpu()

    for i in range(nrows):
        soff = int(roff_h[i].item())
        eoff = int(roff_h[i + 1].item())
        if eoff == soff:
            # row has no entries; nothing to fill, count stays 0
            continue
        k = int(ker_h[soff].item())
        ho = int(row_h[soff].item())
        cnt = eoff - soff

        cols_slice = col_h[soff:eoff]
        hi_slice = cols_slice // Wi
        wi_slice = cols_slice % Wi

        idx_out[k, ho, :cnt, 0] = hi_slice
        idx_out[k, ho, :cnt, 1] = wi_slice
        val_out[k, ho, :cnt] = val_h[soff:eoff]
        count_out[k, ho] = cnt

    return idx_out, val_out, count_out


def python_disco_fwd_from_packed(inp, idx_packed, val_packed, count_packed, K, Ho, Wo):
    """Reference disco forward consuming the packed psi.

    Computes  out[b, c, k, ho, wo] = sum_nz val_packed[k, ho, nz]
                                          * inp[b, c, hi, (wi_base + pscale*wo) % Wi]
    where (hi, wi_base) = idx_packed[k, ho, nz].

    Inputs:
        inp:           [B, C, Hi, Wi]
        idx_packed:    int64 [K, Ho, NBR_PAD, 2]
        val_packed:    [K, Ho, NBR_PAD]
        count_packed:  int64 [K, Ho]
        K, Ho, Wo:     ints (Wo must divide Wi)

    Returns:
        out:           [B, C, K, Ho, Wo] in inp.dtype/device
    """
    B, C, Hi, Wi = inp.shape
    if Wi % Wo != 0:
        raise ValueError(f"Wi ({Wi}) must be a multiple of Wo ({Wo})")
    pscale = Wi // Wo

    device = inp.device
    out_dtype = inp.dtype

    # do the accumulation in val_packed.dtype to match the CUDA kernel's compute dtype path
    accum_dtype = val_packed.dtype
    out = torch.zeros(B, C, K, Ho, Wo, dtype=accum_dtype, device=device)

    wo_idx = torch.arange(Wo, device=device, dtype=torch.long)

    idx_packed_dev = idx_packed.to(device)
    val_packed_dev = val_packed.to(device)
    count_packed_dev = count_packed.to(device)

    inp_acc = inp.to(accum_dtype)

    for k in range(K):
        for ho in range(Ho):
            cnt = int(count_packed_dev[k, ho].item())
            for nz in range(cnt):
                hi = int(idx_packed_dev[k, ho, nz, 0].item())
                wi_base = int(idx_packed_dev[k, ho, nz, 1].item())
                v = val_packed_dev[k, ho, nz]
                wi_full = (wi_base + pscale * wo_idx) % Wi  # [Wo]
                gathered = inp_acc[:, :, hi, wi_full]        # [B, C, Wo]
                out[:, :, k, ho, :] += v * gathered

    return out.to(out_dtype)
