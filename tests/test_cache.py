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

import unittest

import torch


class TestCacheConsistency(unittest.TestCase):
    def test_consistency(self, verbose=False):
        if verbose:
            print("Testing that cache values does not get modified externally")
        from torch_harmonics.legendre import _precompute_legpoly

        with torch.no_grad():
            leg1 = _precompute_legpoly(10, 10, 10, "legendre-gauss")
            # perform in-place modification of leg1
            leg1 *= -1.0
            leg2 = _precompute_legpoly(10, 10, 10, "legendre-gauss")
            self.assertFalse(torch.allclose(leg1, leg2))

    def test_cache_tensor(self, verbose=False):
        from torch_harmonics.legendre import _precompute_legpoly

        with torch.no_grad():
            # the grid is part of the key, so two grids of equal size must not collide
            leg1 = _precompute_legpoly(10, 10, 10, "legendre-gauss")
            leg2 = _precompute_legpoly(10, 10, 10, "equiangular")
            self.assertFalse(torch.allclose(leg1, leg2))

    def test_cache_hits(self, verbose=False):
        """Repeated setup at one resolution must reuse the table, not rebuild it.

        Stacked models construct many layers at the same resolution, and each rebuild of the
        Legendre tables is the most expensive part of that setup. Nothing else in the suite
        would notice if caching silently stopped -- the other tests here only check that a
        cached value cannot be wrong, not that it is ever reused.

        The historical failure this guards against: these tables were once keyed on a tensor
        of colatitudes, and tensors hash by identity, so every layer built a fresh node tensor
        and therefore missed. The key is now ``(nlat, grid)`` and friends, all scalars.
        """

        import torch_harmonics.legendre as legendre

        def count_builds(inner_name, call, repeats=5):
            """Invocations of the uncached core while ``call`` runs ``repeats`` times."""
            calls = []
            original = getattr(legendre, inner_name)

            def counting(*args, **kwargs):
                calls.append(1)
                return original(*args, **kwargs)

            setattr(legendre, inner_name, counting)
            try:
                for _ in range(repeats):
                    call()
            finally:
                setattr(legendre, inner_name, original)
            return len(calls)

        with torch.no_grad():
            # distinct (nlat, grid) per assertion so a warm cache from another test cannot
            # mask a miss, and so the two assertions cannot warm each other
            n = count_builds("legpoly", lambda: legendre._precompute_legpoly(8, 8, 14, "legendre-gauss"))
            self.assertEqual(n, 1, msg=f"_precompute_legpoly rebuilt the table {n} times instead of caching it")

            n = count_builds("dlegpoly", lambda: legendre._precompute_dlegpoly(8, 8, 18, "legendre-gauss"))
            self.assertEqual(n, 1, msg=f"_precompute_dlegpoly rebuilt the table {n} times instead of caching it")

            # ...and the key must still discriminate: a different resolution is a real miss,
            # so a cache that returned everything unconditionally would fail here
            calls = iter((22, 22, 26, 26, 22))
            n = count_builds("legpoly", lambda: legendre._precompute_legpoly(8, 8, next(calls), "lobatto"), repeats=5)
            self.assertEqual(n, 2, msg=f"expected one build per distinct nlat, got {n}")


if __name__ == "__main__":
    unittest.main()
