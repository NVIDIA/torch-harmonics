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

import abc
import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from disco_helpers import optimized_kernels_is_available

from torch_harmonics._backend import BackendSelectionMixin
from torch_harmonics.cache import lru_cache
from torch_harmonics.filter_basis import FilterBasis, get_filter_basis
from torch_harmonics.grid import GridS2, RegularGridS2, require_grid, require_regular_grid
from torch_harmonics.neighborhood import _row_chunks, precompute_neighborhood_csr_s2
from torch_harmonics.quadrature import THETA_CUTOFF_EPS, effective_theta_cutoff, latitude_support_band
from torch_harmonics.truncation import truncate_support
from torch_harmonics.utils import check

from ._disco_utils import _get_psi, _get_psi_ragged
from .backends import BACKENDS
from .optimized.disco_optimized import _use_spatial_first_dgrad


def _normalize_psi_vals(
    psi_vals,
    ikernel,
    igroup,
    q,
    kernel_size,
    ngroups,
    theta_cutoff,
    basis_norm_mode="mean",
    merge_quadrature=False,
    isotropic_mask=None,
    eps=1e-9,
):
    r"""Normalizes convolution tensor values, given each nonzero's group and quadrature weight.

    The grid-independent core of the normalization, shared by the regular and the ragged
    precompute. A *group* is what one basis function is normalized over: a (basis
    function, output latitude) pair on a regular grid, where the p-shift makes every
    point of a latitude alike, and a (basis function, output point) pair on a ragged one.
    The callers differ only in how they assign each nonzero its group and its quadrature
    weight.

    The implementation is fully vectorized: each nonzero is assigned a flat group id
    ``gid = ikernel * ngroups + igroup`` and all per-group sums (support, bias
    numerator, scale) are accumulated in a single ``scatter_add_`` per reduced
    quantity.

    Parameters
    ----------
    psi_vals : torch.Tensor
        Value tensor for the sparse convolution tensor.
    ikernel : torch.Tensor
        Basis function of each nonzero.
    igroup : torch.Tensor
        Group of each nonzero, in ``[0, ngroups)``.
    q : torch.Tensor
        Quadrature weight of each nonzero, normalized so that the weights of a grid
        integrate to 1 over the sphere.
    kernel_size : int
        Number of kernel basis functions.
    ngroups : int
        Number of groups per basis function.
    theta_cutoff : float
        Angular cutoff of the filter support (radians). Required by the "geometric" mode,
        which normalizes by the theoretical area measure of the spherical cap of half-angle
        theta_cutoff; unused by other modes.
    basis_norm_mode : str
        Normalization mode, one of ["none", "nodal", "modal", "mean", "support", "geometric"].
        The legacy names "individual" and "area ratio" are accepted as deprecated aliases
        for "nodal" and "geometric" respectively; each emits a DeprecationWarning.
    merge_quadrature : bool
        If True, multiplies values by quadrature weights.
    isotropic_mask : Optional[Sequence[bool]]
        Per-kernel-index boolean mask; True marks an axisymmetric (m=0) basis function.
        Used by the "modal" mode to decide which kernels get a weighted-mean bias
        subtraction (anisotropic only). If None, only kernel index 0 is treated as isotropic.
    eps : float
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
            stacklevel=3,
        )
        basis_norm_mode = "nodal"
    elif basis_norm_mode == "area ratio":
        warnings.warn(
            'basis_norm_mode="area ratio" is deprecated, use "geometric" instead.',
            DeprecationWarning,
            stacklevel=3,
        )
        basis_norm_mode = "geometric"

    # group id per nonzero: (ikernel, igroup) -> flat index in [0, kernel_size * ngroups)
    n_groups = kernel_size * ngroups
    gid = ikernel * ngroups + igroup

    # support[ik, group] = sum_{nonzeros in group} q
    support_flat = torch.zeros(n_groups, dtype=psi_vals.dtype, device=psi_vals.device)
    support_flat.scatter_add_(0, gid, q)
    support = support_flat.view(kernel_size, ngroups)

    # bias[ik, group] -- only nonzero for "modal" mode on anisotropic kernels.
    # Quadrature-weighted mean of psi_vals over the (ik, group) neighborhood.
    bias = torch.zeros(kernel_size, ngroups, dtype=psi_vals.dtype, device=psi_vals.device)
    if basis_norm_mode == "modal":
        if isotropic_mask is not None:
            iso = torch.as_tensor(isotropic_mask, dtype=torch.bool, device=psi_vals.device)
        else:
            iso = torch.zeros(kernel_size, dtype=torch.bool, device=psi_vals.device)
            iso[0] = True
        aniso_per_nz = (~iso)[ikernel].to(psi_vals.dtype)
        bias_num_flat = torch.zeros(n_groups, dtype=psi_vals.dtype, device=psi_vals.device)
        bias_num_flat.scatter_add_(0, gid, psi_vals * q * aniso_per_nz)
        # divide; isotropic kernels have bias_num=0 so result stays 0; clamp protects empty groups
        bias = (bias_num_flat / support_flat.clamp(min=eps)).view(kernel_size, ngroups)
        # zero the bias on empty groups
        bias = torch.where(support.abs() > eps, bias, torch.zeros_like(bias))

    # scale[ik, group] = sum |psi_vals - bias[ik, group]| * q over the neighborhood
    bias_per_nz = bias.view(-1)[gid]
    scale_flat = torch.zeros(n_groups, dtype=psi_vals.dtype, device=psi_vals.device)
    scale_flat.scatter_add_(0, gid, (psi_vals - bias_per_nz).abs() * q)
    scale = scale_flat.view(kernel_size, ngroups)

    # per-mode (b, s) selection per nonzero, then renormalize in a single elementwise pass
    if basis_norm_mode in ("nodal", "modal"):
        scale_per_nz = scale.view(-1)[gid]
        psi_vals = (psi_vals - bias_per_nz) / scale_per_nz.clamp(min=eps)
    elif basis_norm_mode == "mean":
        # average over groups per kernel; bias is zero in this mode
        bias_per_ik = bias.mean(dim=1)
        scale_per_ik = scale.mean(dim=1)
        bias_per_nz_per_ik = bias_per_ik[ikernel]
        scale_per_nz_per_ik = scale_per_ik[ikernel]
        psi_vals = (psi_vals - bias_per_nz_per_ik) / scale_per_nz_per_ik.clamp(min=eps)
    elif basis_norm_mode == "support":
        support_scale_per_nz = support.view(-1)[gid]
        psi_vals = psi_vals / support_scale_per_nz.clamp(min=eps)
    elif basis_norm_mode == "geometric":
        geometric_scale = (1.0 - math.cos(theta_cutoff)) / 2.0 / 2.0
        psi_vals = psi_vals / max(geometric_scale, eps)
    elif basis_norm_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown basis normalization mode {basis_norm_mode}.")

    if merge_quadrature:
        psi_vals = psi_vals * q

    return psi_vals


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
    r"""Normalizes convolution tensor values based on specified normalization mode.

    This function applies different normalization strategies to the convolution tensor
    values based on the basis_norm_mode parameter. It can normalize individual basis
    functions, compute mean normalization across all basis functions, or use support
    weights. The function also optionally merges quadrature weights into the tensor.

    On a regular grid a group is a (basis function, output latitude) pair; this assigns
    every nonzero its group and quadrature weight and leaves the arithmetic to
    :func:`_normalize_psi_vals`, which the ragged precompute shares.

    Parameters
    ----------
    psi_idx : torch.Tensor
        Index tensor for the sparse convolution tensor.
    psi_vals : torch.Tensor
        Value tensor for the sparse convolution tensor.
    in_shape : Tuple[int]
        Tuple of (nlat_in, nlon_in) representing input grid dimensions.
    out_shape : Tuple[int]
        Tuple of (nlat_out, nlon_out) representing output grid dimensions.
    kernel_size : int
        Number of kernel basis functions.
    quad_weights : torch.Tensor
        Per-latitude normalization weights of shape ``(nlat, 1)``, built by the caller as
        ``colat_weights / nlon_in / 2`` so that they integrate to 1 over the sphere. Not
        :attr:`~torch_harmonics.grid.PointSetS2.quad_weights`, which is per point and
        integrates to :math:`4\pi`.
    theta_cutoff : float
        Angular cutoff of the filter support (radians). Required by the "geometric" mode,
        which normalizes by the theoretical area measure of the spherical cap of half-angle
        theta_cutoff; unused by other modes.
    transpose_normalization : bool
        If True, applies normalization in transpose direction.
    basis_norm_mode : str
        Normalization mode, one of ["none", "nodal", "modal", "mean", "support", "geometric"].
        The legacy names "individual" and "area ratio" are accepted as deprecated aliases
        for "nodal" and "geometric" respectively; each emits a DeprecationWarning.
    merge_quadrature : bool
        If True, multiplies values by quadrature weights.
    isotropic_mask : Optional[Sequence[bool]]
        Per-kernel-index boolean mask; True marks an axisymmetric (m=0) basis function.
        Used by the "modal" mode to decide which kernels get a weighted-mean bias
        subtraction (anisotropic only). If None, only kernel index 0 is treated as isotropic.
    eps : float
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

    # reshape the indices implicitly to be ikernel, out_shape[0], in_shape[0], in_shape[1]
    idx = torch.stack([psi_idx[0], psi_idx[1], psi_idx[2] // in_shape[1], psi_idx[2] % in_shape[1]], dim=0)

    ikernel = idx[0]
    if transpose_normalization:
        ilat_out = idx[2]
        ilat_in = idx[1]
        # deliberately swap input/output shapes to handle transpose normalization with the same code
        nlat_out = in_shape[0]
        correction_factor = out_shape[1] / in_shape[1]
    else:
        ilat_out = idx[1]
        ilat_in = idx[2]
        nlat_out = out_shape[0]

    # quadrature weight per nonzero
    q = quad_weights[ilat_in].reshape(-1)

    psi_vals = _normalize_psi_vals(
        psi_vals,
        ikernel,
        ilat_out,
        q,
        kernel_size,
        nlat_out,
        theta_cutoff,
        basis_norm_mode=basis_norm_mode,
        merge_quadrature=merge_quadrature,
        isotropic_mask=isotropic_mask,
        eps=eps,
    )

    if transpose_normalization and merge_quadrature:
        psi_vals = psi_vals / correction_factor

    return psi_vals


@lru_cache(typed=True, copy=True)
def _precompute_convolution_tensor_s2(
    grid_in: RegularGridS2,
    grid_out: RegularGridS2,
    filter_basis: FilterBasis,
    theta_cutoff: float,
    theta_eps: Optional[float] = THETA_CUTOFF_EPS,
    transpose_normalization: Optional[bool] = False,
    basis_norm_mode: Optional[str] = "nodal",
    merge_quadrature: Optional[bool] = False,
):
    r"""
    Precomputes the rotated filters at positions $R^{-1}_j \omega_i = R^{-1}_j R_i \nu = Y(-\theta_j)Z(\phi_i - \phi_j)Y(\theta_j)\nu$.
    Assumes a tensorized grid on the sphere with equiangular sampling in longitude -- a periodic trapezoidal rule -- as described in Ocampo et al.
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
    ----------
    grid_in : RegularGridS2
        Descriptor of the input grid
    grid_out : RegularGridS2
        Descriptor of the output grid
    filter_basis : FilterBasis
        Filter basis functions
    theta_cutoff : float
        Theta cutoff for the filter basis functions
    theta_eps : float
        Epsilon for the theta cutoff
    transpose_normalization : bool
        Whether to normalize the convolution tensor in the transpose direction
    basis_norm_mode : str
        Mode for basis normalization
    merge_quadrature : bool
        Whether to merge the quadrature weights into the convolution tensor

    Returns
    -------
    out_idx : torch.Tensor
        Index tensor of the convolution tensor
    out_vals : torch.Tensor
        Values tensor of the convolution tensor

    """

    # the descriptors carry the shapes, so the old 2-tuple validation is gone; what
    # is still worth rejecting is a shard, whose latitudes are only part of the sphere
    input_grid = require_regular_grid(grid_in, "grid_in")
    output_grid = require_regular_grid(grid_out, "grid_out")
    in_shape, out_shape = input_grid.shape, output_grid.shape

    kernel_size = filter_basis.kernel_size

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape
    colats_in, win = input_grid.colats, input_grid.colat_weights
    colats_out, wout = output_grid.colats, output_grid.colat_weights

    # compute the phi differences
    # It's imporatant to not include the 2 pi point in the longitudes, as it is equivalent to lon=0
    lons_in = input_grid.lons()

    # compute quadrature weights and merge them into the convolution tensor.
    # These quadrature integrate to 1 over the sphere.
    if transpose_normalization:
        quad_weights = wout.reshape(-1, 1) / nlon_in / 2.0
    else:
        quad_weights = win.reshape(-1, 1) / nlon_in / 2.0

    # effective theta cutoff if multiplied with a fudge factor to avoid aliasing with grid width (especially near poles)
    theta_cutoff_eff = effective_theta_cutoff(theta_cutoff, theta_eps)

    out_idx = []
    out_vals = []

    beta = lons_in

    # compute trigs
    cbeta = torch.cos(beta)
    sbeta = torch.sin(beta)
    cgamma_all = torch.cos(colats_in).reshape(-1, 1)
    sgamma_all = torch.sin(colats_in).reshape(-1, 1)

    # only input latitudes within the cutoff of an output latitude can land in the support, so
    # the rotation is evaluated on that band alone rather than on the whole input grid. Without
    # this the cost is O(nlat_out * nlat_in * nlon_in) to produce a result that is O(nlat_out *
    # band * nlon_in) -- at the default cutoff the band is a handful of rings wide regardless of
    # resolution, so almost all of that work was discarded. The band is a superset of the
    # support, so the sparsity pattern is unchanged, entry for entry.
    band_lo, band_hi = latitude_support_band(colats_in, colats_out, theta_cutoff_eff)

    # compute row offsets
    out_roff = torch.zeros(nlat_out + 1, dtype=torch.int64, device=lons_in.device)
    out_roff[0] = 0
    for t in range(nlat_out):
        # the last angle has a negative sign as it is a passive rotation, which rotates the filter around the y-axis
        alpha = -colats_out[t]

        lo = int(band_lo[t])
        hi = int(band_hi[t])
        if hi < lo:
            # no input latitude is close enough to this output latitude
            out_roff[t + 1] = out_roff[t]
            continue
        cgamma = cgamma_all[lo : hi + 1]
        sgamma = sgamma_all[lo : hi + 1]

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

        # add the output latitude and reshape such that psi has dimensions kernel_shape x nlat_out x (nlat_in*nlon_in).
        # iidx[:, 1] indexes the band, so it is shifted back onto the global input latitude axis
        idx = torch.stack([iidx[:, 0], t * torch.ones_like(iidx[:, 0]), (iidx[:, 1] + lo) * nlon_in + iidx[:, 2]], dim=0)

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


# ceiling on the candidate (output point, input point) pairs rotated at once, which bounds
# the ragged precompute's peak memory independently of the grid size
_PAIR_BUDGET = 1 << 20


@lru_cache(typed=True, copy=True)
def _precompute_convolution_tensor_s2_ragged(
    grid_in: GridS2,
    grid_out: GridS2,
    filter_basis: FilterBasis,
    theta_cutoff: float,
    theta_eps: Optional[float] = THETA_CUTOFF_EPS,
    transpose_normalization: Optional[bool] = False,
    basis_norm_mode: Optional[str] = "nodal",
    merge_quadrature: Optional[bool] = False,
):
    r"""
    Precomputes the rotated filters as :func:`_precompute_convolution_tensor_s2` does, on any ring grid.

    The counterpart of the regular precompute for grids whose rings differ in length, such
    as HEALPix, and for a regular grid paired with one. The regular precompute rotates the
    filter to one output longitude per latitude and lets the kernels reach the rest by the
    p-shift; on a ragged grid there is no shift that carries one output point's
    neighbourhood onto the next one's (see :mod:`torch_harmonics.neighborhood`), so psi is
    keyed by output *point*: the output tensor has shape
    kernel_shape x npoints_out x npoints_in, both indexed in the grids' flat order.

    The candidates come from the geodesic neighbourhood that neighborhood attention uses,
    taken a little wider than the cutoff so that the basis evaluation, and not the arc
    arithmetic, decides each point on the boundary -- as it does on the regular path.
    Each candidate is rotated into its output point's frame with the same YZY formula,
    the longitude difference standing in for the input longitude.

    Normalization is the regular one with a group per (basis function, output point) and
    each nonzero weighted by its point's quadrature weight; with
    ``transpose_normalization`` a group is a (basis function, column point) and the
    weight is that of the row's point. On a regular grid this reproduces the regular psi
    unfolded over longitudes, except for the transpose with ``nlon`` differing between the
    grids, where the regular normalization pools the ``nlon_out / nlon_in`` distinct
    column phases into one latitude's group and this one does not.

    Parameters
    ----------
    grid_in : GridS2
        Descriptor of the input grid
    grid_out : GridS2
        Descriptor of the output grid
    filter_basis : FilterBasis
        Filter basis functions
    theta_cutoff : float
        Theta cutoff for the filter basis functions
    theta_eps : float
        Epsilon for the theta cutoff
    transpose_normalization : bool
        Whether to normalize the convolution tensor in the transpose direction
    basis_norm_mode : str
        Mode for basis normalization
    merge_quadrature : bool
        Whether to merge the quadrature weights into the convolution tensor

    Returns
    -------
    out_idx : torch.Tensor
        Index tensor of the convolution tensor, rows ``(ker, output point, input point)``
    out_vals : torch.Tensor
        Values tensor of the convolution tensor
    """
    grid_in = require_grid(grid_in, "grid_in")
    grid_out = require_grid(grid_out, "grid_out")

    kernel_size = filter_basis.kernel_size
    theta_cutoff_eff = effective_theta_cutoff(theta_cutoff, theta_eps)

    # candidate pairs, keyed by output point: the neighbourhood at the effective cutoff,
    # widened once more by THETA_CUTOFF_EPS so that it is a strict superset of the support
    col_idx, row_off = precompute_neighborhood_csr_s2(grid_in, grid_out, theta_cutoff=theta_cutoff_eff)

    coords_in = grid_in.coords.to(torch.float64)
    coords_out = grid_out.coords.to(torch.float64)

    out_idx = []
    out_vals = []

    for p0, p1 in _row_chunks(row_off, _PAIR_BUDGET):
        e0, e1 = int(row_off[p0]), int(row_off[p1])
        if e1 == e0:
            continue
        cols = col_idx[e0:e1]
        rows = torch.repeat_interleave(torch.arange(p0, p1, dtype=torch.int64), row_off[p0 + 1 : p1 + 1] - row_off[p0:p1])

        # the regular precompute's rotation, with the output point at longitude phi_o
        # rather than 0: alpha = -theta_o, beta = phi_i - phi_o, gamma = theta_i
        alpha = -coords_out[rows, 0]
        beta = coords_in[cols, 1] - coords_out[rows, 1]
        gamma = coords_in[cols, 0]

        x = torch.cos(alpha) * torch.cos(beta) * torch.sin(gamma) + torch.cos(gamma) * torch.sin(alpha)
        y = torch.sin(beta) * torch.sin(gamma)
        z = -torch.cos(beta) * torch.sin(alpha) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma)

        # normalization is important to avoid NaNs when arccos and atan are applied
        norm = torch.sqrt(x * x + y * y + z * z)
        x = x / norm
        y = y / norm
        z = z / norm

        theta = torch.arccos(z)
        phi = torch.arctan2(y, x)
        phi = torch.where(phi < 0.0, phi + 2 * torch.pi, phi)

        # the bases evaluate on a 2-D grid of positions; the pairs are its single row
        iidx, vals = filter_basis.compute_support_vals(theta.unsqueeze(0), phi.unsqueeze(0), r_cutoff=theta_cutoff_eff)
        pair = iidx[:, 2]

        out_idx.append(torch.stack([iidx[:, 0], rows[pair], cols[pair]], dim=0))
        out_vals.append(vals)

    if out_idx:
        out_idx = torch.cat(out_idx, dim=-1)
        out_vals = torch.cat(out_vals, dim=-1)
    else:
        out_idx = torch.empty((3, 0), dtype=torch.int64)
        out_vals = torch.empty(0, dtype=torch.float64)

    # quadrature weights per point, normalized to integrate to 1 over the sphere like the
    # regular path's colat_weights / nlon / 2
    if transpose_normalization:
        q = grid_out.point_weights(torch.float64)[out_idx[1]] / (4.0 * math.pi)
        igroup, ngroups = out_idx[2], grid_in.npoints
    else:
        q = grid_in.point_weights(torch.float64)[out_idx[2]] / (4.0 * math.pi)
        igroup, ngroups = out_idx[1], grid_out.npoints

    out_vals = _normalize_psi_vals(
        out_vals.to(q.dtype),
        out_idx[0],
        igroup,
        q,
        kernel_size,
        ngroups,
        theta_cutoff,
        basis_norm_mode=basis_norm_mode,
        merge_quadrature=merge_quadrature,
        isotropic_mask=filter_basis.isotropic_mask,
    )

    return out_idx.contiguous(), out_vals.contiguous()


def _setup_grids(layer: "DiscreteContinuousConv", grid_in: GridS2, grid_out: GridS2) -> None:
    """
    The grid attributes of a serial DISCO layer, as NeighborhoodAttentionS2 sets them.

    Raggedness is a property of each side on its own. ``ragged`` picks the computation, and
    one ragged grid is enough to force it for both sides, because the ragged path is the
    general one and a regular grid is a case it admits. ``ragged_in`` and ``ragged_out``
    pick only the layout each side shows the caller, which is what lets a HEALPix field be
    convolved onto a lat/lon one and come back shaped like a lat/lon field.
    """
    layer.grid_in = require_grid(grid_in, "grid_in")
    layer.grid_out = require_grid(grid_out, "grid_out")

    # decided by type: only a RegularGridS2 has the (nlat, nlon) layout and uniform stride
    # the regular kernels assume, whatever its ring lengths happen to be
    layer.ragged_in = not isinstance(layer.grid_in, RegularGridS2)
    layer.ragged_out = not isinstance(layer.grid_out, RegularGridS2)
    layer.ragged = layer.ragged_in or layer.ragged_out

    layer.npoints_in = layer.grid_in.npoints
    layer.npoints_out = layer.grid_out.npoints

    # nlat/nlon exist only on a regular grid; keep them where they are defined so that a
    # consumer reaching for them on a ragged side fails rather than silently taking the
    # widest ring for a stride
    if not layer.ragged_in:
        layer.nlat_in, layer.nlon_in = layer.grid_in.shape
    if not layer.ragged_out:
        layer.nlat_out, layer.nlon_out = layer.grid_out.shape


def _to_flat(layer: "DiscreteContinuousConv", x: torch.Tensor) -> torch.Tensor:
    """The input as the computation takes it: unchanged on the regular path, ``(..., npoints_in)`` on the ragged one."""
    if not layer.ragged:
        return x
    if not layer.ragged_in:
        # A regular grid entering the ragged path. Its two spatial axes collapse into the
        # one flat axis psi's columns index, and ring-major order is already that flat order.
        x = x.flatten(-2, -1)
    # the ragged kernels read the field by flat index, so a wrong extent would read out of bounds
    check(x.shape[-1] == layer.npoints_in, lambda: f"Expected {layer.npoints_in} input points, got {x.shape[-1]} (shape {tuple(x.shape)})")
    return x


def _from_flat(layer: "DiscreteContinuousConv", x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_to_flat`, on the output grid."""
    if layer.ragged and not layer.ragged_out:
        # restore the two spatial axes of a regular output grid
        x = x.unflatten(-1, (layer.nlat_out, layer.nlon_out))
    return x


class DiscreteContinuousConv(BackendSelectionMixin, nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for discrete-continuous convolutions

    Holds the filter basis and the weights, and selects the backend that evaluates the
    psi contraction -- see :mod:`torch_harmonics.disco.backends` for what a subclass
    describes about its psi so that any backend can serve it.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_shape : Union[int, Tuple[int], Tuple[int, int]]
        Shape of the kernel
    basis_type : Optional[str]
        Type of the basis functions
    groups : Optional[int]
        Number of groups
    bias : Optional[bool]
        Whether to use bias
    optimized_kernel : Optional[bool]
        Whether to use the optimized kernel (if available)

    Returns
    -------
    torch.Tensor
        Output tensor
    """

    #: The implementations this layer chooses from, in order; see
    #: :data:`.backends.BACKENDS`.
    _backends = BACKENDS

    #: whether psi is applied in the scatter direction
    transpose = False

    #: whether psi is keyed per point rather than per latitude, see DiscreteContinuousConvS2;
    #: only the serial layers can be, the distributed ones shard regular grids
    ragged = False

    #: whether the layer contracts through the fused node with the spatial-first input
    #: gradient in play; a subclass that does sets this before selecting its backend
    _needs_split = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
        basis_type: Optional[str] = "piecewise linear",
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        optimized_kernel: Optional[bool] = True,
    ):
        super().__init__()

        self.kernel_shape = kernel_shape
        self.optimized_kernel = optimized_kernel and optimized_kernels_is_available()

        # get the filter basis functions
        self.filter_basis = get_filter_basis(kernel_shape=kernel_shape, basis_type=basis_type)

        # groups
        self.groups = groups

        # weight tensor
        if in_channels % self.groups != 0:
            raise ValueError("Error, the number of input channels has to be an integer multiple of the group size")
        if out_channels % self.groups != 0:
            raise ValueError("Error, the number of output channels has to be an integer multiple of the group size")
        self.groupsize = in_channels // self.groups
        self.out_per_group = out_channels // self.groups
        scale = math.sqrt(1.0 / self.groupsize) * self.filter_basis.get_init_factors().reshape(1, 1, -1)
        self.weight = nn.Parameter(scale * torch.randn(out_channels, self.groupsize, self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # psi belongs to the backend, which registers exactly what it reads. Selection is
        # the last step of a subclass's __init__, once psi can be described.
        self._backend_state = ()
        self.backend = None

    @property
    def kernel_size(self):
        return self.filter_basis.kernel_size

    @property
    def device(self) -> torch.device:
        """
        The device this module is on.

        ``nn.Module`` has no public equivalent. Every buffer belongs to the backend and can
        come and go with it, so the answer is a parameter: the weight exists on every layer
        and follows every move.
        """
        return self.weight.device

    @abc.abstractmethod
    def _psi_coo(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The psi entries ``(ker_idx, row_idx, col_idx, vals)``, as fresh tensors.

        A row is a latitude of the grid psi is keyed by and a column a flat index
        ``ring * _psi_nlon + lon`` into the other grid. On a ragged layer a row is a point
        of the grid psi is keyed by and a column a flat index into ``_psi_col_grid``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _reference_psi(self, ker_idx, row_idx, col_idx, vals) -> torch.Tensor:
        """The sparse psi the torch reference contracts with."""
        raise NotImplementedError

    def _weight_r(self) -> torch.Tensor:
        """The weight as ``(groups, out_per_group, groupsize, kernel_size)``."""
        return self.weight.reshape(self.groups, self.out_per_group, self.weight.shape[1], self.weight.shape[2])

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class DiscreteContinuousConvS2(DiscreteContinuousConv):
    r"""
    Discrete-continuous (DISCO) convolution on the 2-sphere, as described in :cite:`Ocampo2023`.

    The layer evaluates a spherical convolution with a compactly supported
    filter of angular radius ``theta_cutoff``.  The filter is parameterised as
    a learnable linear combination of fixed basis functions
    :math:`\{\phi_k\}`, and the integral is computed by sparse quadrature over
    the input grid, giving :math:`O(N)` cost in the number of grid points.
    The forward pass is

    .. math::

        g^{c_o}(\theta'_j, \lambda'_q)
            = \sum_{c_i} \sum_k w_k^{c_o,c_i}
              \sum_{i,\,p} \Psi_{k,\,j,\,(i,p)}\;
              f^{c_i}(\theta_i, \lambda'_q + \lambda_p)

    where :math:`\Psi` is a precomputed sparse convolution tensor that
    encodes the basis function values at rotated input grid positions,
    weighted by the quadrature weights.  Because the grid is equispaced in
    longitude, :math:`\Psi` is independent of the output longitude
    (p-shift symmetry).

    Either grid may also be ragged, such as HEALPix, whose rings differ in length.
    There is no p-shift then, so :math:`\Psi` is keyed per output point, and fields
    on a ragged grid are flat: ``(batch, channels, npoints)``. A regular grid paired
    with a ragged one keeps its ``(nlat, nlon)`` layout at the interface.

    .. seealso::
        :doc:`/guide/disco_convolutions`
            User guide with the full mathematical derivation, filter basis
            visualisations, and worked examples.

    Parameters
    ----------
    grid_in : GridS2
        Descriptor of the input grid; it carries the resolution as well as the
        quadrature rule. Any ring grid, ragged ones such as HEALPix included.
    grid_out : GridS2
        Descriptor of the output grid, likewise.
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_shape : Union[int, Tuple[int], Tuple[int, int]]
        Shape of the kernel
    basis_type : Optional[str]
        Type of the basis functions
    basis_norm_mode : Optional[str]
        Mode for basis normalization
    groups : Optional[int]
        Number of groups
    bias : Optional[bool]
        Whether to use bias
    theta_cutoff : Optional[float]
        Theta cutoff for the filter basis functions
    optimized_kernel : Optional[bool]
        Whether to use the optimized kernel (if available)
    fused : Optional[bool]
        When True, recomputes the K-expanded intermediate ``(B, C, K, H, W)`` in backward
        instead of storing it: K times less activation memory for one extra sparse
        contraction. Has no effect with the torch reference.

    References
    ----------
    :cite:`Ocampo2023`
    """

    def __init__(
        self,
        grid_in: GridS2,
        grid_out: GridS2,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
        basis_type: Optional[str] = "piecewise linear",
        basis_norm_mode: Optional[str] = "nodal",
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
        optimized_kernel: Optional[bool] = True,
        fused: Optional[bool] = False,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, basis_type, groups, bias, optimized_kernel)

        self.fused = bool(fused)
        self.basis_norm_mode = basis_norm_mode
        _setup_grids(self, grid_in, grid_out)

        # make sure the p-shift works by checking that longitudes are divisible
        if not self.ragged and self.nlon_in % self.nlon_out != 0:
            raise ValueError(f"nlon_in ({self.nlon_in}) must be an integer multiple of nlon_out ({self.nlon_out}) for the DISCO p-shift to be exact")

        # heuristic to compute theta cutoff based on the bandlimit of the input field and overlaps of the basis functions
        self.theta_cutoff = truncate_support(self.grid_out, theta_cutoff)

        # psi is keyed by output latitude -- output point if ragged -- and its columns index the input grid
        self._psi_col_grid = self.grid_in
        if not self.ragged:
            self._psi_nlon = self.nlon_in
            self._contract_shape = (self.nlat_out, self.nlon_out)
        self._needs_split = _use_spatial_first_dgrad(self.out_per_group, self.groupsize, self.kernel_size)

        self._select_backend()

    def extra_repr(self):
        return f"grid_in={self.grid_in!r},\ngrid_out={self.grid_out!r},\nin_channels={self.groupsize * self.groups}, out_channels={self.weight.shape[0]}, filter_basis={self.filter_basis}, kernel_shape={self.kernel_shape}, theta_cutoff={self.theta_cutoff}, groups={self.groups}"

    def _psi_coo(self):
        precompute = _precompute_convolution_tensor_s2_ragged if self.ragged else _precompute_convolution_tensor_s2
        idx, vals = precompute(
            self.grid_in,
            self.grid_out,
            self.filter_basis,
            theta_cutoff=self.theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode=self.basis_norm_mode,
            merge_quadrature=True,
        )[:2]
        return idx[0].contiguous(), idx[1].contiguous(), idx[2].contiguous(), vals.contiguous()

    def _reference_psi(self, ker_idx, row_idx, col_idx, vals):
        idx = torch.stack([ker_idx, row_idx, col_idx], dim=0)
        if self.ragged:
            return _get_psi_ragged(self.kernel_size, idx, vals, self.npoints_out, self.npoints_in)
        return _get_psi(self.kernel_size, idx, vals, self.nlat_in, self.nlon_in, self.nlat_out, self.nlon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the discrete-continuous convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape ``(batch, in_channels, nlat_in, nlon_in)`` on a
            :class:`~torch_harmonics.grid.RegularGridS2`, ``(batch, in_channels, npoints_in)``
            on any other grid.

        Returns
        -------
        torch.Tensor
            Convolved signal of shape ``(batch, out_channels, nlat_out, nlon_out)`` on a
            :class:`~torch_harmonics.grid.RegularGridS2`, ``(batch, out_channels, npoints_out)``
            on any other grid.
        """
        x = _to_flat(self, x)

        # the backend is fixed before tracing, see torch_harmonics._backend
        out = self.backend.conv(self, x, self._weight_r(), self.groups, self.groupsize, recompute=self.fused)

        if self.bias is not None:
            out = out + self.bias.reshape(1, self.bias.shape[0], *([1] * (out.dim() - 2)))

        return _from_flat(self, out)


class DiscreteContinuousConvTransposeS2(DiscreteContinuousConv):
    r"""
    Discrete-continuous (DISCO) transpose convolution on the 2-sphere, as described in :cite:`Ocampo2023`.

    This is the transpose (adjoint) of
    :class:`~torch_harmonics.DiscreteContinuousConvS2`.  It uses the same
    continuous-filter and quadrature construction but applies the
    :math:`\Psi` tensor in the reverse direction -- typically to map a coarser
    grid to a finer one (upsampling), analogous to a transposed/strided
    convolution in the planar case.  It shares the compact-support filter and
    sparse, linearly scaling evaluation, and the same approximate
    :math:`SO(3)` equivariance. Like the forward layer it accepts ragged grids
    such as HEALPix, on which fields are ``(batch, channels, npoints)``.

    .. seealso::
        :doc:`/guide/disco_convolutions`
            User guide with the full mathematical derivation, filter basis
            visualisations, and worked examples.

    Parameters
    ----------
    grid_in : GridS2
        Descriptor of the input grid; it carries the resolution as well as the
        quadrature rule. Any ring grid, ragged ones such as HEALPix included.
    grid_out : GridS2
        Descriptor of the output grid, likewise.
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_shape : Union[int, Tuple[int], Tuple[int, int]]
        Shape of the kernel
    basis_type : Optional[str]
        Type of the basis functions
    basis_norm_mode : Optional[str]
        Mode for basis normalization
    groups : Optional[int]
        Number of groups
    bias : Optional[bool]
        Whether to use bias
    theta_cutoff : Optional[float]
        Theta cutoff for the filter basis functions
    optimized_kernel : Optional[bool]
        Whether to use the optimized kernel (if available)

    References
    ----------
    :cite:`Ocampo2023`
    """

    transpose = True

    def __init__(
        self,
        grid_in: GridS2,
        grid_out: GridS2,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
        basis_type: Optional[str] = "piecewise linear",
        basis_norm_mode: Optional[str] = "nodal",
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
        optimized_kernel: Optional[bool] = True,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, basis_type, groups, bias, optimized_kernel)

        self.basis_norm_mode = basis_norm_mode
        _setup_grids(self, grid_in, grid_out)

        # make sure the p-shift works by checking that longitudes are divisible
        if not self.ragged and self.nlon_out % self.nlon_in != 0:
            raise ValueError(f"nlon_out ({self.nlon_out}) must be an integer multiple of nlon_in ({self.nlon_in}) for the DISCO transpose p-shift to be exact")

        # bandlimit
        self.theta_cutoff = truncate_support(self.grid_in, theta_cutoff)

        # psi is that of the forward convolution from grid_out to grid_in, so it is keyed
        # by *input* latitude -- input point if ragged -- and its columns index the output
        # grid it scatters onto
        self._psi_col_grid = self.grid_out
        if not self.ragged:
            self._psi_nlon = self.nlon_out
            self._contract_shape = (self.nlat_out, self.nlon_out)

        self._select_backend()

    def extra_repr(self):
        return f"grid_in={self.grid_in!r},\ngrid_out={self.grid_out!r},\nin_channels={self.groupsize * self.groups}, out_channels={self.weight.shape[0]}, filter_basis={self.filter_basis}, kernel_shape={self.kernel_shape}, theta_cutoff={self.theta_cutoff}, groups={self.groups}"

    def _psi_coo(self):
        # switch in_shape and out_shape since we want the transpose convolution
        precompute = _precompute_convolution_tensor_s2_ragged if self.ragged else _precompute_convolution_tensor_s2
        idx, vals = precompute(
            self.grid_out,
            self.grid_in,
            self.filter_basis,
            theta_cutoff=self.theta_cutoff,
            transpose_normalization=True,
            basis_norm_mode=self.basis_norm_mode,
            merge_quadrature=True,
        )[:2]
        return idx[0].contiguous(), idx[1].contiguous(), idx[2].contiguous(), vals.contiguous()

    def _reference_psi(self, ker_idx, row_idx, col_idx, vals):
        idx = torch.stack([ker_idx, row_idx, col_idx], dim=0)
        if self.ragged:
            return _get_psi_ragged(self.kernel_size, idx, vals, self.npoints_in, self.npoints_out, transposed=True)
        return _get_psi(self.kernel_size, idx, vals, self.nlat_in, self.nlon_in, self.nlat_out, self.nlon_out, semi_transposed=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the transpose discrete-continuous convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape ``(batch, in_channels, nlat_in, nlon_in)`` on a
            :class:`~torch_harmonics.grid.RegularGridS2`, ``(batch, in_channels, npoints_in)``
            on any other grid.

        Returns
        -------
        torch.Tensor
            Convolved signal of shape ``(batch, out_channels, nlat_out, nlon_out)`` on a
            :class:`~torch_harmonics.grid.RegularGridS2`, ``(batch, out_channels, npoints_out)``
            on any other grid.
        """
        x = _to_flat(self, x)

        # extract shape; one spatial axis on a ragged layer, two on a regular one
        B = x.shape[0]
        spatial = x.shape[2:]
        x = x.reshape(B, self.groups, self.groupsize, *spatial)

        # do weight multiplication
        x = torch.einsum("bgc...,gock->bgok...", x, self._weight_r()).contiguous()
        x = x.reshape(B, self.weight.shape[0], self.kernel_size, *spatial)

        # the backend is fixed before tracing, see torch_harmonics._backend
        out = self.backend.transpose(self, x)

        if self.bias is not None:
            out = out + self.bias.reshape(1, self.bias.shape[0], *([1] * (out.dim() - 2)))

        return _from_flat(self, out)
