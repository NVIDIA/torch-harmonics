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
from typing import List, Tuple, Union, Optional
from warnings import warn

import math

import torch
import torch.nn as nn

from functools import partial

from torch_harmonics.cache import lru_cache
from torch_harmonics.quadrature import _precompute_grid, _precompute_latitudes, _precompute_longitudes
from ._disco_utils import _get_psi, _disco_s2_contraction_torch, _disco_s2_transpose_contraction_torch
from ._disco_utils import _disco_s2_contraction_optimized, _disco_s2_transpose_contraction_optimized
from torch_harmonics.filter_basis import FilterBasis, get_filter_basis
from disco_helpers import optimized_kernels_is_available, preprocess_psi



def _normalize_convolution_tensor_s2(
    psi_idx, psi_vals, in_shape, out_shape, kernel_size, quad_weights, transpose_normalization=False, basis_norm_mode="mean", merge_quadrature=False, eps=1e-9
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
    transpose_normalization: bool
        If True, applies normalization in transpose direction.
    basis_norm_mode: str
        Normalization mode, one of ["none", "individual", "mean", "support"].
    merge_quadrature: bool
        If True, multiplies values by quadrature weights.
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

    # exit here if no normalization is needed
    if basis_norm_mode == "none":
        return psi_vals

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
    vnorm = torch.zeros(kernel_size, nlat_out, device=psi_vals.device)
    support = torch.zeros(kernel_size, nlat_out, device=psi_vals.device)

    # loop through dimensions to compute the norms
    for ik in range(kernel_size):
        for ilat in range(nlat_out):

            # find indices corresponding to the given output latitude and kernel basis function
            iidx = torch.argwhere((ikernel == ik) & (ilat_out == ilat))

            # compute the 1-norm
            # vnorm[ik, ilat] = torch.sqrt(torch.sum(psi_vals[iidx].abs().pow(2) * q[iidx]))
            vnorm[ik, ilat] = torch.sum(psi_vals[iidx].abs() * q[iidx])

            # compute the support
            support[ik, ilat] = torch.sum(q[iidx])

    # loop over values and renormalize
    for ik in range(kernel_size):
        for ilat in range(nlat_out):

            iidx = torch.argwhere((ikernel == ik) & (ilat_out == ilat))

            if basis_norm_mode == "individual":
                val = vnorm[ik, ilat]
            elif basis_norm_mode == "mean":
                val = vnorm[ik, :].mean()
            elif basis_norm_mode == "support":
                val = support[ik, ilat]
            elif basis_norm_mode == "none":
                val = 1.0
            else:
                raise ValueError(f"Unknown basis normalization mode {basis_norm_mode}.")

            psi_vals[iidx] = psi_vals[iidx] / (val + eps)

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
    grid_in: Optional[str]="equiangular",
    grid_out: Optional[str]="equiangular",
    theta_cutoff: Optional[float]=0.01 * math.pi,
    theta_eps: Optional[float]=1e-3,
    transpose_normalization: Optional[bool]=False,
    basis_norm_mode: Optional[str]="mean",
    merge_quadrature: Optional[bool]=False,
):
    """
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
    lats_in, win = _precompute_latitudes(nlat_in, grid=grid_in)
    lats_out, wout = _precompute_latitudes(nlat_out, grid=grid_out)

    # compute the phi differences
    # It's imporatant to not include the 2 pi point in the longitudes, as it is equivalent to lon=0
    lons_in = _precompute_longitudes(nlon_in)

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
        transpose_normalization=transpose_normalization,
        basis_norm_mode=basis_norm_mode,
        merge_quadrature=merge_quadrature,
    )

    out_idx = out_idx.contiguous()
    out_vals = out_vals.to(dtype=torch.float32).contiguous()

    return out_idx, out_vals, out_roff


class DiscreteContinuousConv(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for discrete-continuous convolutions

    Parameters
    -----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    kernel_shape: Union[int, Tuple[int], Tuple[int, int]]
        Shape of the kernel
    basis_type: Optional[str]
        Type of the basis functions
    groups: Optional[int]
        Number of groups
    bias: Optional[bool]
        Whether to use bias

    Returns
    -------
    out: torch.Tensor
        Output tensor
    """

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
        scale = math.sqrt(1.0 / self.groupsize / self.kernel_size)
        self.weight = nn.Parameter(scale * torch.randn(out_channels, self.groupsize, self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    @property
    def kernel_size(self):
        return self.filter_basis.kernel_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class DiscreteContinuousConvS2(DiscreteContinuousConv):
    """
    Discrete-continuous (DISCO) convolutions on the 2-Sphere as described in [1].

    Parameters
    -----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    in_shape: Tuple[int]
        Input shape of the convolution tensor
    out_shape: Tuple[int]
        Output shape of the convolution tensor
    kernel_shape: Union[int, Tuple[int], Tuple[int, int]]
        Shape of the kernel
    basis_type: Optional[str]
        Type of the basis functions
    basis_norm_mode: Optional[str]
        Mode for basis normalization
    groups: Optional[int]
        Number of groups
    grid_in: Optional[str]
        Input grid type
    grid_out: Optional[str]
        Output grid type
    bias: Optional[bool]
        Whether to use bias
    theta_cutoff: Optional[float]
        Theta cutoff for the filter basis functions

    Returns
    -------
    out: torch.Tensor
        Output tensor

    References
    ----------
    [1] Ocampo, Price, McEwen, Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
        basis_type: Optional[str] = "piecewise linear",
        basis_norm_mode: Optional[str] = "mean",
        groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, basis_type, groups, bias, optimized_kernel)

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        # make sure the p-shift works by checking that longitudes are divisible
        assert self.nlon_in % self.nlon_out == 0

        # heuristic to compute theta cutoff based on the bandlimit of the input field and overlaps of the basis functions
        if theta_cutoff is None:
            theta_cutoff = torch.pi / float(self.nlat_out - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        idx, vals, _ = _precompute_convolution_tensor_s2(
            in_shape,
            out_shape,
            self.filter_basis,
            grid_in=grid_in,
            grid_out=grid_out,
            theta_cutoff=theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )

        # sort the values
        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        if self.optimized_kernel:
            # preprocessed data-structure for GPU kernel
            roff_idx = preprocess_psi(self.kernel_size, self.nlat_out, ker_idx, row_idx, col_idx, vals).contiguous()
            self.register_buffer("psi_roff_idx", roff_idx, persistent=False)

        # save all datastructures
        self.register_buffer("psi_ker_idx", ker_idx, persistent=False)
        self.register_buffer("psi_row_idx", row_idx, persistent=False)
        self.register_buffer("psi_col_idx", col_idx, persistent=False)
        self.register_buffer("psi_vals", vals, persistent=False)

        # also store psi as COO matrix just in case for torch input
        self.psi = _get_psi(self.kernel_size, self.psi_idx, self.psi_vals, self.nlat_in, self.nlon_in, self.nlat_out, self.nlon_out)

    def extra_repr(self):
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_chans={self.groupsize * self.groups}, out_chans={self.weight.shape[0]}, filter_basis={self.filter_basis}, kernel_shape={self.kernel_shape}, groups={self.groups}"

    @property
    def psi_idx(self):
        return torch.stack([self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx], dim=0).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.optimized_kernel:
            x = _disco_s2_contraction_optimized(
                x, self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx, self.psi_vals, self.kernel_size, self.nlat_out, self.nlon_out
            )
        else:
            if x.is_cuda:
                warn("couldn't find CUDA extension, falling back to slow PyTorch implementation")
            x = _disco_s2_contraction_torch(x, self.psi.to(x.device), self.nlon_out)

        # extract shape
        B, C, K, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, K, H, W)

        # do weight multiplication
        out = torch.einsum("bgckxy,gock->bgoxy", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2])).contiguous()
        out = out.reshape(B, -1, H, W)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out


class DiscreteContinuousConvTransposeS2(DiscreteContinuousConv):
    """
    Discrete-continuous (DISCO) transpose convolutions on the 2-Sphere as described in [1].

    Parameters
    -----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    in_shape: Tuple[int]
        Input shape of the convolution tensor
    out_shape: Tuple[int]
        Output shape of the convolution tensor
    kernel_shape: Union[int, Tuple[int], Tuple[int, int]]
        Shape of the kernel
    basis_type: Optional[str]
        Type of the basis functions
    basis_norm_mode: Optional[str]
        Mode for basis normalization
    groups: Optional[int]
        Number of groups
    grid_in: Optional[str]
        Input grid type
    grid_out: Optional[str]
        Output grid type
    bias: Optional[bool]
        Whether to use bias
    theta_cutoff: Optional[float]
        Theta cutoff for the filter basis functions

    Returns
    --------
    out: torch.Tensor
        Output tensor

    References
    ----------
    [1] Ocampo, Price, McEwen, Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        kernel_shape: Union[int, Tuple[int], Tuple[int, int]],
        basis_type: Optional[str] = "piecewise linear",
        basis_norm_mode: Optional[str] = "mean",
        groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        bias: Optional[bool] = True,
        theta_cutoff: Optional[float] = None,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, basis_type, groups, bias, optimized_kernel)

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        # make sure the p-shift works by checking that longitudes are divisible
        assert self.nlon_out % self.nlon_in == 0

        # bandlimit
        if theta_cutoff is None:
            theta_cutoff = torch.pi / float(self.nlat_in - 1)

        if theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        # switch in_shape and out_shape since we want the transpose convolution
        idx, vals, _ = _precompute_convolution_tensor_s2(
            out_shape,
            in_shape,
            self.filter_basis,
            grid_in=grid_out,
            grid_out=grid_in,
            theta_cutoff=theta_cutoff,
            transpose_normalization=True,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )

        # sort the values
        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        if self.optimized_kernel:
            # preprocessed data-structure for GPU kernel
            roff_idx = preprocess_psi(self.kernel_size, self.nlat_in, ker_idx, row_idx, col_idx, vals).contiguous()
            self.register_buffer("psi_roff_idx", roff_idx, persistent=False)

        # save all datastructures
        self.register_buffer("psi_ker_idx", ker_idx, persistent=False)
        self.register_buffer("psi_row_idx", row_idx, persistent=False)
        self.register_buffer("psi_col_idx", col_idx, persistent=False)
        self.register_buffer("psi_vals", vals, persistent=False)

        # also store psi just in case
        if not self.optimized_kernel:
            self.psi_st = _get_psi(self.kernel_size, self.psi_idx, self.psi_vals, self.nlat_in, self.nlon_in, self.nlat_out, self.nlon_out, semi_transposed=True)

    def extra_repr(self):
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_chans={self.groupsize * self.groups}, out_chans={self.weight.shape[0]}, filter_basis={self.filter_basis}, kernel_shape={self.kernel_shape}, groups={self.groups}"

    @property
    def psi_idx(self):
        return torch.stack([self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx], dim=0).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # extract shape
        B, C, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, H, W)

        # do weight multiplication
        x = torch.einsum("bgcxy,gock->bgokxy", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2])).contiguous()
        x = x.reshape(B, -1, x.shape[-3], H, W)

        if self.optimized_kernel:
            out = _disco_s2_transpose_contraction_optimized(
                x, self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx, self.psi_vals, self.kernel_size, self.nlat_out, self.nlon_out
            )
        else:
            if x.is_cuda:
                warn("couldn't find CUDA extension, falling back to slow PyTorch implementation")
            out = _disco_s2_transpose_contraction_torch(x, self.psi_st.to(x.device), self.nlon_out)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out
