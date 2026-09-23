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

"""
Tests for :class:`~torch_harmonics.NeighborhoodAttentionS2` on a ragged grid.

The oracle is a dense masked softmax: the same attention written out as one
``(npoints_out, npoints_in)`` matrix per head, with the neighbourhood applied as an
additive mask and the quadrature weights as log-weights, differentiated by autograd
rather than by hand. It shares no code with the module under test -- not the arc
encoding, not the neighbour walk, not the gradient algebra -- so agreement between
the two is evidence about all three.

The mask itself comes from the brute-force great-circle reference in
:mod:`test_neighborhood`, imported rather than copied so the two test modules cannot
drift into disagreeing about what a neighbourhood is.
"""

import math
import unittest

import torch
import torch.nn.functional as F
from attention_helpers import optimized_kernels_is_available
from parameterized import parameterized, parameterized_class
from test_neighborhood import _brute_force_neighborhood
from testutils import compare_tensors, disable_tf32, set_seed
from torch.library import opcheck

from torch_harmonics import HealpixGrid, NeighborhoodAttentionS2, as_grid, precompute_neighborhood_arcs_s2

# imported for the side effect of registering the op that opcheck looks up by name
from torch_harmonics.attention.kernels_torch.attention_ragged_torch import _neighborhood_s2_attention_ragged_torch  # noqa: F401
from torch_harmonics.attention.optimized.attention_optimized import _op_is_declared

try:
    from torch_harmonics.attention.optimized.attention_optimized import _neighborhood_s2_attention_ragged_optimized
except ImportError:
    # defined only when the extension declares both halves of the ragged pair; the
    # tests that use it are skipped in that case
    _neighborhood_s2_attention_ragged_optimized = None

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


def _dense_masked_attention(model, query, key, value, mask):
    r"""
    The oracle: :class:`NeighborhoodAttentionS2`'s mathematics as a dense masked
    softmax, using ``model``'s parameters but none of its machinery.

    Computes, for each head,

    .. math::
        y_p = \sum_{j \in D(p)} \frac{e^{s\, q_p \cdot k_j} w_j}{\sum_{j' \in D(p)} e^{s\, q_p \cdot k_{j'}} w_{j'}} v_j

    with the neighbourhood :math:`D(p)` and the quadrature weights :math:`w_j` both
    folded into a single additive pre-softmax mask -- the weights as
    :math:`\log w_j`, which the softmax exponential turns back into factors, and the
    neighbourhood as :math:`-\infty` outside the disk.

    Note the weights are kept in the formula even though HEALPix is equal-area and
    they therefore cancel: the point is to check the module against the general
    expression, not against a simplification that happens to hold on this grid.

    Parameters
    ----------
    model : NeighborhoodAttentionS2
        Supplies the projection weights, biases, head count and scale.
    query : torch.Tensor
        ``(batch, in_channels, npoints_out)``.
    key, value : torch.Tensor
        ``(batch, in_channels, npoints_in)``.
    mask : torch.Tensor
        Boolean ``(npoints_out, npoints_in)``; ``True`` where the input point is in
        the output point's neighbourhood.

    Returns
    -------
    torch.Tensor
        ``(batch, out_channels, npoints_out)``.
    """
    heads = model.num_heads

    def project(signal, weights, bias):
        # the stored weights are (C_out, C_in, 1, 1) convolution kernels
        return F.conv1d(signal, weights.reshape(*weights.shape[:2], 1), bias=bias)

    q = project(query, model.q_weights, model.q_bias)
    k = project(key, model.k_weights, model.k_bias)
    v = project(value, model.v_weights, model.v_bias)

    # (batch, channels, npoints) -> (batch, heads, npoints, channels per head)
    def split_heads(signal):
        batch, channels, npoints = signal.shape
        return signal.reshape(batch, heads, channels // heads, npoints).transpose(-1, -2)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    if model.q_norm_weights is not None:
        q = F.rms_norm(q, normalized_shape=model.q_norm_weights.shape, weight=1 + model.q_norm_weights)
    if model.k_norm_weights is not None:
        k = F.rms_norm(k, normalized_shape=model.k_norm_weights.shape, weight=1 + model.k_norm_weights)

    logits = model.scale * (q @ k.transpose(-1, -2))

    log_weights = torch.log(model.point_weights.to(logits.dtype))
    logits = logits + log_weights.reshape(1, 1, 1, -1)
    logits = logits.masked_fill(~mask.reshape(1, 1, *mask.shape), float("-inf"))

    out = torch.softmax(logits, dim=-1) @ v

    # (batch, heads, npoints, channels per head) -> (batch, out_channels, npoints)
    batch, _, npoints, _ = out.shape
    out = out.transpose(-1, -2).reshape(batch, model.out_channels, npoints)

    return project(out, model.proj_weights, model.proj_bias)


@parameterized_class(("device"), _devices)
class TestRaggedNeighborhoodAttentionS2(unittest.TestCase):
    """NeighborhoodAttentionS2 on HEALPix, against a dense masked-softmax oracle."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _build(self, nside_in, nside_out, channels, out_channels, heads, use_qknorm, bias, theta_cutoff=None):
        grid_in = HealpixGrid(nside=nside_in)
        grid_out = HealpixGrid(nside=nside_out)
        model = NeighborhoodAttentionS2(
            grid_in=grid_in,
            grid_out=grid_out,
            in_channels=channels,
            out_channels=out_channels,
            num_heads=heads,
            use_qknorm=use_qknorm,
            bias=bias,
            theta_cutoff=theta_cutoff,
        ).to(self.device)
        return grid_in, grid_out, model

    def _inputs(self, batch, channels, npoints_in, npoints_out):
        make = lambda npoints: torch.randn(batch, channels, npoints, device=self.device, dtype=torch.float32, requires_grad=True)
        return {"q": make(npoints_out), "k": make(npoints_in), "v": make(npoints_in)}

    @parameterized.expand(
        [
            # Format: [nside_in, nside_out, batch, channels, out_channels, heads, use_qknorm, bias]
            [4, 4, 2, 4, 4, 1, False, True],
            [4, 4, 2, 4, 4, 2, False, True],
            [4, 4, 2, 4, 8, 4, False, True],
            [4, 4, 2, 8, 4, 4, False, True],
            [4, 4, 2, 4, 4, 2, True, True],
            [4, 4, 2, 4, 4, 1, False, False],
            [8, 8, 1, 4, 4, 2, False, True],
            # resampling in both directions. The ragged path has no p-shift, so unlike
            # the regular one it does not need the point counts to divide each other
            [8, 4, 1, 4, 4, 2, False, True],
            [4, 8, 1, 4, 4, 2, False, True],
            [2, 4, 2, 4, 4, 1, False, True],
            # Channel counts that reach the register-blocked CUDA kernel.
            #
            # Every case above has at most 32 channels per head, so NLOC, which is
            # DIV_UP(channels per head, 32), is 1 for all of them. At NLOC == 1 the
            # kernel's unrolled loops run from 0 to NLOC-1 == 0, i.e. not at all, and
            # only the bounds-guarded tail iteration executes. So none of those cases
            # touch the unguarded register indexing that the whole kernel rests on --
            # it is sound only because i <= NLOC-2 implies i*32 + tidx < nchan, and an
            # off-by-one there reads a neighbouring channel block and still returns a
            # smooth, plausible field.
            #
            # 96 per head is what healda's dit-5B runs, and is an exact multiple of the
            # warp, so it never exercises the guard on the final register; 80 and 40
            # leave that one partial (only 16 and 8 lanes of the last register are in
            # range) and are here for the boundary. 192 over 2 heads puts NLOC > 1 and
            # a head offset together, since the head stride is what the unrolled
            # indexing is added to.
            [4, 4, 1, 96, 96, 1, False, True],  # NLOC 3, exact
            [4, 4, 1, 80, 80, 1, False, True],  # NLOC 3, last register partial
            [4, 4, 1, 40, 40, 1, False, True],  # NLOC 2, last register partial
            [4, 4, 1, 192, 192, 2, False, True],  # NLOC 3, two heads
            # Unequal counts at a large channel width, which the register-blocked
            # kernel refuses (it needs one NLOC to serve both), so this pins the
            # generic fallback on the sizes that now bypass it.
            [4, 4, 1, 96, 64, 1, False, True],
        ],
        skip_on_empty=True,
    )
    def test_it_matches_a_dense_masked_softmax(self, nside_in, nside_out, batch, channels, out_channels, heads, use_qknorm, bias, atol=1e-5, rtol=1e-4):
        """Forward and every gradient, against the dense oracle."""
        grid_in, grid_out, model = self._build(nside_in, nside_out, channels, out_channels, heads, use_qknorm, bias)

        mask = _brute_force_neighborhood(grid_in, grid_out, model.theta_cutoff).to(self.device)

        inputs = self._inputs(batch, channels, grid_in.npoints, grid_out.npoints)
        inputs_ref = {name: tensor.detach().clone().requires_grad_() for name, tensor in inputs.items()}

        out = model(inputs["q"], inputs["k"], inputs["v"])
        out_ref = _dense_masked_attention(model, inputs_ref["q"], inputs_ref["k"], inputs_ref["v"], mask)

        self.assertTrue(compare_tensors("output", out, out_ref, atol=atol, rtol=rtol))

        # one backward per graph, from the same upstream gradient
        grad = torch.randn_like(out)
        grads_ref = torch.autograd.grad(out_ref, list(inputs_ref.values()) + list(model.parameters()), grad_outputs=grad, retain_graph=True)
        grads = torch.autograd.grad(out, list(inputs.values()) + list(model.parameters()), grad_outputs=grad)

        names = list(inputs.keys()) + [name for name, _ in model.named_parameters()]
        for name, got, expected in zip(names, grads, grads_ref):
            self.assertTrue(compare_tensors(f"grad {name}", got, expected, atol=atol, rtol=rtol))

    @parameterized.expand([[2], [4]], skip_on_empty=True)
    def test_a_global_cutoff_attends_to_the_whole_sphere(self, nside, atol=1e-5, rtol=1e-4):
        r"""
        With :math:`\theta_\mathrm{cutoff} = 2\pi` every input point must be in every
        neighbourhood, and the layer must then agree with an *unmasked* dense softmax.

        The degenerate end of the cutoff range, where the local operator becomes the
        global one. It is worth its own case because it is the one radius at which
        every arc wraps a full ring, so an off-by-one in the arc encoding that the
        interior cases tolerate has nowhere to hide.
        """
        grid = HealpixGrid(nside=nside)
        channels, batch = 4, 2

        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=channels, num_heads=2, bias=False, theta_cutoff=2 * math.pi).to(self.device)

        # the pattern is complete: every output point sees every input point, exactly once
        self.assertEqual(model.psi_col_idx.numel(), grid.npoints * grid.npoints)

        inputs = self._inputs(batch, channels, grid.npoints, grid.npoints)
        inputs_ref = {name: tensor.detach().clone().requires_grad_() for name, tensor in inputs.items()}

        unmasked = torch.ones(grid.npoints, grid.npoints, dtype=torch.bool, device=self.device)

        out = model(inputs["q"], inputs["k"], inputs["v"])
        out_ref = _dense_masked_attention(model, inputs_ref["q"], inputs_ref["k"], inputs_ref["v"], unmasked)

        self.assertTrue(compare_tensors("output", out, out_ref, atol=atol, rtol=rtol))

    def test_the_quadrature_weights_integrate_to_the_sphere(self):
        """A constant field must integrate to the sphere's area, at every resolution."""
        for nside in (1, 2, 4, 8):
            with self.subTest(nside=nside):
                grid = HealpixGrid(nside=nside)
                model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=1).to(self.device)
                self.assertAlmostEqual(float(model.point_weights.sum()), 4.0 * math.pi, places=4)

    def test_a_constant_field_is_reproduced(self):
        """
        Attention over a constant field returns that constant, whatever the weights.

        The normalized attention weights of each output point sum to one by
        construction, so this is really a check that the neighbour list, the weight
        gather and the normalization all agree about which points a row contains --
        a row that gathered a weight it did not gather a value for would break it.
        """
        grid_in, grid_out, model = self._build(4, 4, 4, 4, 1, False, False)

        # with the projections set to the identity and no bias, attention over a
        # constant input is the only thing left that could change the value
        with torch.no_grad():
            for weights in (model.q_weights, model.k_weights, model.v_weights, model.proj_weights):
                weights.zero_()
                weights[:, :, 0, 0] = torch.eye(weights.shape[0], weights.shape[1])

        constant = torch.full((2, 4, grid_in.npoints), 0.375, device=self.device)
        out = model(constant)

        self.assertTrue(compare_tensors("output", out, torch.full_like(out, 0.375), atol=1e-5, rtol=1e-5))

    def test_the_op_satisfies_its_schema(self):
        """
        opcheck: schema, fake tensor and autograd registration of the ragged op.

        ``test_aot_dispatch_dynamic`` is deliberately excluded. It traces the backward,
        which -- like the regular reference's backward -- walks the neighbour list in
        Python and so reads ``row_off`` with ``int()``. Under AOT autograd those reads
        are guards on unbacked symints and raise, because the values of an index tensor
        are not known at trace time. The forward does not hit this only because a
        ``custom_op`` is opaque to tracing; a backward registered through
        ``register_autograd`` is not.

        This is a property of every neighbour-list-walking reference in the library,
        which is why the optimized ops are the ones checked under AOT dispatch and the
        references are not (see the opcheck calls in test_attention.py). The remaining
        three utilities are the ones that mean something for an op whose whole purpose
        is to be an opaque, readable specification.
        """
        grid = HealpixGrid(nside=2)
        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=4, num_heads=2).to(self.device)

        batch, npoints = 2, grid.npoints
        make = lambda channels: torch.randn(batch, npoints, channels, device=self.device, dtype=torch.float32, requires_grad=True)
        args = (
            make(model.k_channels),
            make(model.out_channels),
            make(model.k_channels),
            model.point_weights,
            model.psi_col_idx,
            model.psi_roff_idx,
            model.num_heads,
            npoints,
        )
        opcheck(
            torch.ops.attention_kernels._neighborhood_s2_attention_ragged_torch,
            args,
            test_utils=("test_schema", "test_autograd_registration", "test_faketensor"),
        )

    def test_a_mixed_pair_takes_the_ragged_path_and_keeps_each_layout(self):
        """
        A mixed pair is computed by the ragged path, because keying the neighbourhood
        by output point is the general choice and a regular grid admits it too. What
        each side must not lose is its own layout: the ragged side stays flat and the
        regular side keeps its two spatial axes.
        """
        hpx, eqa = HealpixGrid(nside=2), as_grid("equiangular", (6, 12))

        decode = NeighborhoodAttentionS2(grid_in=hpx, grid_out=eqa, in_channels=4)
        self.assertTrue(decode.ragged)
        self.assertTrue(decode.ragged_in)
        self.assertFalse(decode.ragged_out)

        encode = NeighborhoodAttentionS2(grid_in=eqa, grid_out=hpx, in_channels=4)
        self.assertTrue(encode.ragged)
        self.assertFalse(encode.ragged_in)
        self.assertTrue(encode.ragged_out)

    def test_it_rejects_inputs_of_the_wrong_rank_or_extent(self):
        """A ragged field is flat, so the module wants 3 dims and the right point count."""
        grid = HealpixGrid(nside=2)
        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=4).to(self.device)

        with self.assertRaises(RuntimeError):
            model(torch.randn(2, 4, grid.nlat, 4 * grid.nside, device=self.device))

        with self.assertRaises(RuntimeError):
            model(torch.randn(2, 4, grid.npoints + 1, device=self.device))

    def test_every_output_point_attends_to_itself(self):
        """
        Self-attention on the same grid must include the diagonal, or an output point
        would be built without reference to its own value.
        """
        grid = HealpixGrid(nside=4)
        model = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=1)

        col_idx, row_off = model.psi_col_idx, model.psi_roff_idx
        for ipoint in range(grid.npoints):
            neighbors = col_idx[row_off[ipoint] : row_off[ipoint + 1]]
            self.assertIn(ipoint, neighbors.tolist(), f"output point {ipoint} does not attend to itself")

    def test_the_neighbourhood_matches_the_brute_force_reference(self):
        """
        The module's expanded neighbour list is the brute-force one.

        The dense test above builds its mask from the reference, so on its own it
        would not notice the module and the reference disagreeing about the
        neighbourhood in a way that happened to be self-consistent. This compares the
        two directly.
        """
        for nside_in, nside_out in ((4, 4), (8, 4), (4, 8)):
            with self.subTest(nside_in=nside_in, nside_out=nside_out):
                grid_in, grid_out, model = self._build(nside_in, nside_out, 1, 1, 1, False, False)
                expected = _brute_force_neighborhood(grid_in, grid_out, model.theta_cutoff)

                got = torch.zeros_like(expected)
                for ipoint in range(grid_out.npoints):
                    got[ipoint, model.psi_col_idx[model.psi_roff_idx[ipoint] : model.psi_roff_idx[ipoint + 1]]] = True

                self.assertTrue(torch.equal(got, expected))


@parameterized_class(("device"), _devices)
class TestMixedGridNeighborhoodAttentionS2(unittest.TestCase):
    """
    Attention between a ragged grid and a product grid: the encoder and decoder case.

    The neighbourhood is keyed by output point whichever way the resampling goes, so
    the ragged path serves both and there is no new mathematics here. What is new is
    that the two sides carry different layouts, and the thing that can go wrong is the
    flattening: a regular grid enters the ragged path as ``ilat * nlon + ilon``, and if
    that disagreed with the order the neighbourhood indexes, the result would be a
    plausible-looking field built from the wrong neighbours. The oracle below is the
    same dense masked softmax the pure-ragged tests use, addressed in flat indices
    throughout, so it pins the ordering rather than assuming it.
    """

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @parameterized.expand(
        [
            # Format: [name, grid_in, grid_out]. Both directions, and both families
            # in the output role, since only the output side gets unflattened again.
            ["decode", HealpixGrid(nside=4), as_grid("equiangular", (16, 32))],
            ["encode", as_grid("equiangular", (16, 32)), HealpixGrid(nside=4)],
            ["decode_to_lobatto", HealpixGrid(nside=4), as_grid("lobatto", (15, 30))],
            ["decode_coarser", HealpixGrid(nside=8), as_grid("equiangular", (12, 24))],
        ],
        skip_on_empty=True,
    )
    def test_it_matches_a_dense_masked_softmax(self, name, grid_in, grid_out, atol=1e-5, rtol=1e-4):
        """Forward and every gradient, against the oracle, in flat index space."""
        batch, channels, heads = 2, 8, 2
        model = NeighborhoodAttentionS2(grid_in=grid_in, grid_out=grid_out, in_channels=channels, num_heads=heads).to(self.device)

        mask = _brute_force_neighborhood(grid_in, grid_out, model.theta_cutoff).to(self.device)
        # an output point with an empty neighbourhood would make the oracle's softmax
        # divide by zero, which would look like a module bug rather than a test one
        self.assertTrue(bool(mask.any(dim=-1).all()), f"{name}: some output point has no neighbours")

        # The leaves are flat for both the module and the oracle, and the module's
        # view is a reshape of the same leaf. That way the gradients come back in one
        # layout and comparing them needs no reinterpretation of its own.
        def leaf(npoints):
            return torch.randn(batch, channels, npoints, device=self.device, requires_grad=True)

        flat = {"q": leaf(grid_out.npoints), "k": leaf(grid_in.npoints), "v": leaf(grid_in.npoints)}
        flat_ref = {name_: t.detach().clone().requires_grad_() for name_, t in flat.items()}

        def as_grid_shape(tensor, grid):
            return tensor if not grid.is_regular else tensor.unflatten(-1, grid.shape)

        out = model(as_grid_shape(flat["q"], grid_out), as_grid_shape(flat["k"], grid_in), as_grid_shape(flat["v"], grid_in))

        expected_shape = (batch, model.out_channels, *grid_out.spatial_shape)
        self.assertEqual(tuple(out.shape), expected_shape, f"{name}: output must be laid out on grid_out")

        out_ref = _dense_masked_attention(model, flat_ref["q"], flat_ref["k"], flat_ref["v"], mask)

        out_flat = out.reshape(batch, model.out_channels, grid_out.npoints)
        self.assertTrue(compare_tensors("output", out_flat, out_ref, atol=atol, rtol=rtol))

        # one backward per graph, from the same upstream gradient
        grad = torch.randn_like(out_ref)
        grads_ref = torch.autograd.grad(out_ref, list(flat_ref.values()) + list(model.parameters()), grad_outputs=grad, retain_graph=True)
        grads = torch.autograd.grad(out_flat, list(flat.values()) + list(model.parameters()), grad_outputs=grad)

        names = list(flat.keys()) + [pname for pname, _ in model.named_parameters()]
        for pname, got, expected in zip(names, grads, grads_ref):
            self.assertTrue(compare_tensors(f"grad {pname}", got, expected, atol=atol, rtol=rtol))

    def test_the_flattening_is_ring_major(self):
        """
        The ordering assumption, on its own so that a violation of it is not diagnosed
        as an attention bug.

        ``GridS2.lon_offsets`` places ``(ilat, ilon)`` at ``lon_offsets[ilat] + ilon``,
        which on a regular grid is ``ilat * nlon + ilon``. That is what the layer
        relies on when it reshapes a regular field into the flat axis the
        neighbourhood indexes.
        """
        grid = as_grid("equiangular", (6, 12))
        offsets = grid.lon_offsets
        for ilat in range(grid.nlat):
            for ilon in (0, 1, grid.nlon - 1):
                self.assertEqual(int(offsets[ilat]) + ilon, ilat * grid.nlon + ilon)

        field = torch.arange(grid.npoints, dtype=torch.float32).reshape(grid.nlat, grid.nlon)
        self.assertTrue(torch.equal(field.flatten(-2, -1), torch.arange(grid.npoints, dtype=torch.float32)))

    def test_a_constant_field_decodes_to_a_constant_field(self):
        """
        The decoder's sanity check. Attention weights are a partition of unity, so a
        constant input must come back constant on the output grid whatever the
        neighbourhoods look like -- including at the poles, where an equiangular grid
        stacks many output points onto nearly the same place.
        """
        grid_in, grid_out = HealpixGrid(nside=4), as_grid("equiangular", (16, 32))
        model = NeighborhoodAttentionS2(grid_in=grid_in, grid_out=grid_out, in_channels=4, num_heads=1, bias=False).to(self.device)

        # with no biases and a constant input, every value vector is the same, so the
        # output is that vector projected -- independent of the softmax entirely
        const = torch.full((1, 4, grid_in.npoints), 0.75, device=self.device)
        query = torch.full((1, 4, *grid_out.shape), 0.75, device=self.device)
        out = model(query, const, const)

        self.assertEqual(tuple(out.shape), (1, 4, *grid_out.shape))
        spread = (out - out.amin(dim=(-2, -1), keepdim=True)).abs().max().detach()
        self.assertLess(float(spread), 1e-5, "a constant field did not decode to a constant field")


class TestRaggedKernelAgainstRegularKernel(unittest.TestCase):
    r"""
    The ragged kernels against the regular ones on a product grid.

    A product grid is the special case of a ragged grid in which every ring happens to
    have the same length, so the two kernels must agree on it exactly. That makes the
    established regular reference an oracle for the ragged one covering the parts no
    HEALPix test can reach on its own: it is the same softmax, the same quadrature and
    the same gradient algebra, checked without a second implementation of any of them.

    The bridge between the two is the p-shift. The regular kernels store one neighbour
    list per output *latitude* and recover the neighbours of output longitude ``wo`` by
    advancing each stored input longitude by ``pscale * wo``; the ragged kernels want
    the result of that, one list per output *point*. Doing the shift here, in the test,
    from the definition in the regular kernels' own docstring, is what makes the
    comparison meaningful -- neither implementation is asked to agree with the other
    about the pattern, only about the arithmetic on it.
    """

    def _expand_pshift(self, col_idx, row_off, nlat_out, nlon_out, nlon_in):
        """The regular kernels' per-latitude psi as a per-output-point column list."""
        pscale = nlon_in // nlon_out

        per_point = []
        for ho in range(nlat_out):
            stored = col_idx[int(row_off[ho]) : int(row_off[ho + 1])].to(torch.int64)
            ring, lon = stored // nlon_in, stored % nlon_in
            for wo in range(nlon_out):
                per_point.append(ring * nlon_in + (lon + pscale * wo) % nlon_in)

        counts = torch.tensor([c.numel() for c in per_point], dtype=torch.int64)
        return torch.cat(per_point), torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(dim=0)])

    @parameterized.expand(
        [
            # Format: [nlat_in, nlon_in, nlat_out, nlon_out, grid, batch, k_channels, v_channels]
            [6, 12, 6, 12, "equiangular", 2, 4, 4],
            [6, 12, 6, 12, "equiangular", 2, 3, 5],
            [6, 12, 6, 12, "legendre-gauss", 2, 4, 4],
            [6, 12, 6, 12, "lobatto", 1, 4, 4],
            # downsampling, where the p-shift is a genuine shift rather than the identity
            [12, 24, 6, 12, "equiangular", 1, 4, 4],
            [12, 24, 6, 8, "equiangular", 1, 4, 4],
            [7, 12, 5, 6, "equiangular", 1, 4, 4],
        ],
        skip_on_empty=True,
    )
    def test_the_two_kernels_agree(self, nlat_in, nlon_in, nlat_out, nlon_out, grid_type, batch, k_channels, v_channels, atol=1e-5, rtol=1e-4):
        from torch_harmonics.attention.kernels_torch.attention_ragged_torch import (
            _neighborhood_s2_attention_ragged_bwd_dk_torch,
            _neighborhood_s2_attention_ragged_bwd_dq_torch,
            _neighborhood_s2_attention_ragged_bwd_dv_torch,
            _neighborhood_s2_attention_ragged_fwd_torch,
        )
        from torch_harmonics.attention.kernels_torch.attention_torch import (
            _neighborhood_s2_attention_bwd_dk_torch,
            _neighborhood_s2_attention_bwd_dq_torch,
            _neighborhood_s2_attention_bwd_dv_torch,
            _neighborhood_s2_attention_fwd_torch,
        )
        from torch_harmonics.disco.convolution import _precompute_convolution_tensor_s2
        from torch_harmonics.filter_basis import get_filter_basis

        set_seed(333)

        grid_in = as_grid(grid_type, (nlat_in, nlon_in))
        grid_out = as_grid(grid_type, (nlat_out, nlon_out))
        theta_cutoff = grid_out.theta_cutoff()

        idx, _, roff = _precompute_convolution_tensor_s2(
            grid_in, grid_out, get_filter_basis(kernel_shape=1, basis_type="zernike"), theta_cutoff=theta_cutoff, basis_norm_mode="none", merge_quadrature=True
        )
        col_idx, row_off = idx[2].contiguous(), roff.contiguous()

        # what the regular kernels index by latitude, and the same thing per point
        quad_weights = (2.0 * math.pi * grid_in.quad_weights / nlon_in).to(torch.float32)
        point_weights = quad_weights.repeat_interleave(nlon_in)
        ragged_col_idx, ragged_row_off = self._expand_pshift(col_idx, row_off, nlat_out, nlon_out, nlon_in)

        kx = torch.randn(batch, k_channels, nlat_in, nlon_in, dtype=torch.float32)
        vx = torch.randn(batch, v_channels, nlat_in, nlon_in, dtype=torch.float32)
        qy = torch.randn(batch, k_channels, nlat_out, nlon_out, dtype=torch.float32)
        dy = torch.randn(batch, v_channels, nlat_out, nlon_out, dtype=torch.float32)

        flat = lambda tensor: tensor.reshape(*tensor.shape[:2], -1)
        npoints_out = nlat_out * nlon_out

        regular_args = (kx, vx, qy, quad_weights, col_idx, row_off, nlon_in, nlat_out, nlon_out)
        ragged_args = (flat(kx), flat(vx), flat(qy), point_weights, ragged_col_idx, ragged_row_off, npoints_out)

        got = _neighborhood_s2_attention_ragged_fwd_torch(*ragged_args)
        expected = _neighborhood_s2_attention_fwd_torch(*regular_args)
        self.assertTrue(compare_tensors("forward", got, flat(expected), atol=atol, rtol=rtol))

        regular_bwd_args = (kx, vx, qy, dy, quad_weights, col_idx, row_off, nlon_in, nlat_out, nlon_out)
        ragged_bwd_args = (flat(kx), flat(vx), flat(qy), flat(dy), point_weights, ragged_col_idx, ragged_row_off, npoints_out)

        for name, ragged_fn, regular_fn in (
            ("dv", _neighborhood_s2_attention_ragged_bwd_dv_torch, _neighborhood_s2_attention_bwd_dv_torch),
            ("dk", _neighborhood_s2_attention_ragged_bwd_dk_torch, _neighborhood_s2_attention_bwd_dk_torch),
            ("dq", _neighborhood_s2_attention_ragged_bwd_dq_torch, _neighborhood_s2_attention_bwd_dq_torch),
        ):
            with self.subTest(gradient=name):
                got = ragged_fn(*ragged_bwd_args)
                expected = regular_fn(*regular_bwd_args)
                self.assertTrue(compare_tensors(name, got, flat(expected), atol=atol, rtol=rtol))


class TestRaggedArcsToCsr(unittest.TestCase):
    """The CSR expansion of an arc pattern, against its per-point counterpart."""

    def test_it_agrees_with_the_per_point_expansion(self):
        for nside_in, nside_out in ((2, 2), (4, 2), (2, 4)):
            with self.subTest(nside_in=nside_in, nside_out=nside_out):
                grid_in = HealpixGrid(nside=nside_in)
                grid_out = HealpixGrid(nside=nside_out)
                arcs = precompute_neighborhood_arcs_s2(grid_in, grid_out, 0.5)

                col_idx, row_off = arcs.to_csr()

                self.assertEqual(row_off.numel(), grid_out.npoints + 1)
                self.assertEqual(int(row_off[0]), 0)
                self.assertEqual(int(row_off[-1]), col_idx.numel())
                self.assertEqual(col_idx.numel(), arcs.nnz)

                for ipoint in range(grid_out.npoints):
                    expected = arcs.columns(ipoint)
                    got = col_idx[row_off[ipoint] : row_off[ipoint + 1]]
                    self.assertTrue(torch.equal(got, expected))


@unittest.skipUnless(
    torch.cuda.is_available() and optimized_kernels_is_available() and _op_is_declared("forward_ragged"),
    "needs a compiled extension that declares forward_ragged",
)
class TestRaggedForwardCudaKernel(unittest.TestCase):
    """
    The ragged CUDA forward kernel, against the torch reference.

    The reference is the thing already checked against a dense masked softmax and
    against the regular kernels, so pinning the CUDA kernel to it is what carries
    that validation across. The kernel is a rewrite of the addressing only -- the
    online softmax and the quadrature weighting are copied unchanged from the
    product-grid kernel -- so a disagreement points at the arc walk, which is the
    part that is new.
    """

    def setUp(self):
        set_seed(333)
        disable_tf32()
        self.device = torch.device("cuda")

    def _run_both(self, nside, batch, num_heads, channels, dtype):
        grid = HealpixGrid(nside=nside)
        layer = NeighborhoodAttentionS2(
            in_channels=channels,
            num_heads=num_heads,
            grid_in=grid,
            grid_out=grid,
        ).to(self.device)

        npix = grid.npoints
        packed = num_heads * (channels // num_heads)
        shape = (batch, npix, packed)
        kx = torch.randn(shape, device=self.device, dtype=dtype)
        vx = torch.randn(shape, device=self.device, dtype=dtype)
        qy = torch.randn(shape, device=self.device, dtype=dtype)

        # the op returns its softmax statistics alongside the output, plus an fp32 copy
        # of the output in bf16 only; they exist for the backward's benefit and are
        # checked for shape and dtype below
        got, y_hi, alpha_sum, qdotk_max = torch.ops.attention_kernels.forward_ragged(
            kx.contiguous(),
            vx.contiguous(),
            qy.contiguous(),
            layer.ring_weights,
            layer.psi_seg,
            layer.psi_seg_off,
            layer.psi_ring_base,
            layer.psi_ring_size,
            num_heads,
            npix,
        )

        for name, stat in (("alpha_sum", alpha_sum), ("qdotk_max", qdotk_max)):
            self.assertEqual(stat.shape, (batch, num_heads, npix), name)
            self.assertEqual(stat.dtype, torch.float32, name)
        # alpha_sum is a sum of positive terms, and every output point has neighbours
        self.assertTrue((alpha_sum > 0).all())
        self.assertTrue(torch.isfinite(qdotk_max).all())

        # y_hi carries the output again at full precision, and only where the backward
        # needs it: in bf16, whose 8 mantissa bits cannot form integral = dy . out
        # accurately enough for the single-pass form. Empty for every other dtype, so
        # they pay nothing. Asserted because "silently absent" and "silently empty"
        # would both leave bf16 quietly back on two passes.
        self.assertEqual(y_hi.dtype, torch.float32, "y_hi")
        if dtype == torch.bfloat16:
            self.assertEqual(y_hi.shape, got.shape, "y_hi")
            self.assertTrue(torch.allclose(y_hi.to(dtype), got, atol=0, rtol=0), "y_hi must equal y once narrowed")
        else:
            self.assertEqual(y_hi.numel(), 0, "y_hi should be empty except in bf16")

        # the reference shares this op's channels-last ABI, so it takes the same
        # tensors; it differs only in consuming the CSR expansion instead of the arcs
        expected = _neighborhood_s2_attention_ragged_torch(
            kx.contiguous(),
            vx.contiguous(),
            qy.contiguous(),
            layer.point_weights,
            layer.psi_col_idx,
            layer.psi_roff_idx,
            num_heads,
            npix,
        )

        return got, expected

    @parameterized.expand([(2, 1, 1, 8), (4, 2, 4, 32), (8, 1, 8, 64)])
    def test_it_matches_the_torch_reference(self, nside, batch, num_heads, channels):
        got, expected = self._run_both(nside, batch, num_heads, channels, torch.float32)
        self.assertTrue(compare_tensors("forward", got, expected, rtol=1e-5, atol=1e-5))

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_it_matches_the_torch_reference_in_reduced_precision(self, dtype):
        got, expected = self._run_both(4, 2, 4, 32, dtype)
        self.assertEqual(got.dtype, dtype)
        self.assertTrue(compare_tensors(f"forward {dtype}", got.float(), expected.float(), rtol=2e-2, atol=2e-2))

    def test_it_rejects_shapes_it_cannot_serve(self):
        grid = HealpixGrid(nside=2)
        layer = NeighborhoodAttentionS2(in_channels=8, num_heads=1, grid_in=grid, grid_out=grid).to(self.device)
        npix = grid.npoints
        good = torch.randn(1, npix, 8, device=self.device)

        args = (layer.ring_weights, layer.psi_seg, layer.psi_seg_off, layer.psi_ring_base, layer.psi_ring_size)

        # a 4-D activation is the product-grid ABI, which this op does not accept
        with self.assertRaises(RuntimeError):
            torch.ops.attention_kernels.forward_ragged(good.unsqueeze(1), good, good, *args, 1, npix)

        # npoints_out that disagrees with qy
        with self.assertRaises(RuntimeError):
            torch.ops.attention_kernels.forward_ragged(good, good, good, *args, 1, npix + 1)

        # a channel count that does not divide by num_heads
        with self.assertRaises(RuntimeError):
            torch.ops.attention_kernels.forward_ragged(good, good, good, *args, 3, npix)


@unittest.skipUnless(
    torch.cuda.is_available() and optimized_kernels_is_available() and _op_is_declared("forward_ragged") and _op_is_declared("backward_ragged"),
    "requires a CUDA device and an extension built with the ragged attention kernels",
)
class TestRaggedBackwardCudaKernel(unittest.TestCase):
    """
    The ragged CUDA backward kernel, against autograd through the torch reference.

    Differentiating the reference is a stronger check than differentiating a
    hand-written formula: the reference's own forward is already pinned to a dense
    masked softmax, so autograd through it is a gradient of something independently
    known to be right. The kernel recomputes q.k in a second pass rather than
    storing the per-neighbour alphas, so a disagreement concentrated in dk or dv
    points at that replay, and one in dq points at the three shared reductions.
    """

    def setUp(self):
        set_seed(444)
        disable_tf32()
        self.device = torch.device("cuda")

    def _grads_from_both(self, nside, batch, num_heads, channels, dtype):
        grid = HealpixGrid(nside=nside)
        layer = NeighborhoodAttentionS2(
            in_channels=channels,
            num_heads=num_heads,
            grid_in=grid,
            grid_out=grid,
        ).to(self.device)

        npix = grid.npoints
        packed = num_heads * (channels // num_heads)
        shape = (batch, npix, packed)

        base = [torch.randn(shape, device=self.device, dtype=dtype) for _ in range(3)]
        dy = torch.randn(shape, device=self.device, dtype=dtype)

        def run(fn, weights, *pattern):
            # fresh leaves per side so the two backward passes cannot accumulate
            # into each other's .grad
            kx, vx, qy = (t.clone().detach().requires_grad_(True) for t in base)
            out = fn(kx, vx, qy, weights, *pattern, num_heads, npix)
            # the optimized op also returns the softmax statistics its backward needs;
            # the reference returns the output alone
            if isinstance(out, tuple):
                out = out[0]
            out.backward(dy)
            return kx.grad, vx.grad, qy.grad

        got = run(
            _neighborhood_s2_attention_ragged_optimized,
            layer.ring_weights,
            layer.psi_seg,
            layer.psi_seg_off,
            layer.psi_ring_base,
            layer.psi_ring_size,
        )
        expected = run(
            _neighborhood_s2_attention_ragged_torch,
            layer.point_weights,
            layer.psi_col_idx,
            layer.psi_roff_idx,
        )

        return got, expected

    @parameterized.expand([(2, 1, 1, 8), (4, 2, 4, 32), (8, 1, 8, 64)])
    def test_it_matches_autograd_through_the_reference(self, nside, batch, num_heads, channels):
        got, expected = self._grads_from_both(nside, batch, num_heads, channels, torch.float32)
        for name, g, e in zip(("dk", "dv", "dq"), got, expected):
            with self.subTest(grad=name):
                self.assertTrue(compare_tensors(f"grad {name}", g, e, rtol=1e-4, atol=1e-4))

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_it_matches_autograd_through_the_reference_in_reduced_precision(self, dtype):
        got, expected = self._grads_from_both(4, 2, 4, 32, dtype)
        for name, g, e in zip(("dk", "dv", "dq"), got, expected):
            with self.subTest(grad=name):
                self.assertEqual(g.dtype, dtype)
                self.assertTrue(compare_tensors(f"grad {name}", g.float(), e.float(), rtol=3e-2, atol=3e-2))

    def test_the_layer_selects_the_optimized_path_and_stays_differentiable(self):
        # the point of the pair: with both halves present the module should pick the
        # kernel rather than the reference, and still produce a gradient
        grid = HealpixGrid(nside=4)
        layer = NeighborhoodAttentionS2(in_channels=32, num_heads=4, grid_in=grid, grid_out=grid).to(self.device)

        self.assertTrue(layer.optimized_kernel)

        x = torch.randn(2, 32, grid.npoints, device=self.device, requires_grad=True)
        layer(x).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_a_cpu_module_falls_back_to_the_reference(self):
        # The ragged kernels are CUDA-only -- there is no CPU implementation to fall
        # back on, unlike the product-grid ops. Selecting the optimized handle once
        # at setup therefore breaks any module used on CPU, and the device is not
        # even known at setup, since building on CPU and moving with .to() is normal.
        grid = HealpixGrid(nside=2)
        layer = NeighborhoodAttentionS2(in_channels=8, num_heads=1, grid_in=grid, grid_out=grid)

        x = torch.randn(1, 8, grid.npoints, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        self.assertTrue(torch.isfinite(out).all())
        self.assertIsNotNone(x.grad)

        # and the same module, once moved, must take the kernel
        layer = layer.to(self.device)
        xc = x.detach().to(self.device).requires_grad_(True)
        out_cuda = layer(xc)
        out_cuda.sum().backward()

        self.assertTrue(torch.isfinite(out_cuda).all())
        self.assertTrue(compare_tensors("cpu vs cuda forward", out_cuda.cpu(), out, atol=1e-4, rtol=1e-4))

    def test_the_op_survives_torch_compile(self):
        """
        A fake whose signature has drifted from its schema is invisible until
        something traces the graph.

        Eager execution calls the CUDA implementation directly and never consults a
        fake, so a stale one passes every other test here. Only tracing reads them,
        and only tracing a *backward* reads the backward's -- which is how
        `backward_ragged` came to be traced with one argument more than its fake
        accepted, after y_hi was added to its schema. Every gate in
        rebuild_and_validate_ragged.sh passed on that build; the first thing to
        notice was a benchmark whose compiled column had quietly gone empty.

        So this compiles a forward and a backward, which is the cheapest thing that
        reads both fakes, and asserts the gradients arrive. It is a registration
        test, not a numerics test -- the accuracy of the compiled path is the same
        kernel the other tests already check.
        """
        grid = HealpixGrid(nside=4)
        layer = NeighborhoodAttentionS2(in_channels=8, num_heads=2, grid_in=grid, grid_out=grid).to(self.device)

        # channels-first, which is the layer's ABI. The raw op takes the transpose --
        # (batch, npoints, packed channels) -- and mixing the two up gets caught by a
        # shape check naming the point count, which reads like a channel error.
        x = torch.randn(1, 8, grid.npoints, device=self.device, requires_grad=True)

        compiled = torch.compile(layer, dynamic=False)
        compiled(x, x, x).sum().backward()

        self.assertIsNotNone(x.grad, "no gradient came back through the compiled op")
        self.assertTrue(torch.isfinite(x.grad).all(), "compiled backward produced non-finite gradients")

    def test_dk_and_dv_accumulate_over_overlapping_neighborhoods(self):
        # dk/dv are scatter-accumulated with atomicAdd because neighbourhoods overlap.
        # If the buffers were allocated with empty() instead of zeros(), or a
        # contribution were dropped, the gradient of a point that many neighbourhoods
        # touch would be wrong -- so require every input point to receive one.
        grid = HealpixGrid(nside=4)
        layer = NeighborhoodAttentionS2(in_channels=8, num_heads=1, grid_in=grid, grid_out=grid).to(self.device)

        npix = grid.npoints
        kx, vx, qy = (torch.randn(1, npix, 8, device=self.device, requires_grad=True) for _ in range(3))

        out, y_hi, alpha_sum, qdotk_max = _neighborhood_s2_attention_ragged_optimized(
            kx,
            vx,
            qy,
            layer.ring_weights,
            layer.psi_seg,
            layer.psi_seg_off,
            layer.psi_ring_base,
            layer.psi_ring_size,
            1,
            npix,
        )

        # the statistics are the backward's own bookkeeping, so nothing must be able to
        # route a gradient through them -- see _setup_context_attention_ragged_backward.
        # y_hi is the same for the same reason and one more: it is the output again, so
        # a gradient through it would be counted twice.
        self.assertFalse(alpha_sum.requires_grad)
        self.assertFalse(qdotk_max.requires_grad)
        self.assertFalse(y_hi.requires_grad)

        out.backward(torch.ones_like(out))

        for name, g in (("dk", kx.grad), ("dv", vx.grad)):
            with self.subTest(grad=name):
                self.assertTrue(torch.isfinite(g).all())
                touched = (g.abs().sum(dim=-1) > 0).sum().item()
                self.assertEqual(touched, npix, f"{name}: only {touched} of {npix} input points received a gradient")


if __name__ == "__main__":
    unittest.main()
