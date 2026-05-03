// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "disco.h"

#include <algorithm>

template <typename REAL_T>
void pack_psi_dense_kernel(int64_t K, int64_t Ho, int64_t Wi, int64_t nbr_pad, int64_t nrows,
                           const int64_t *ker_h, const int64_t *row_h, const int64_t *col_h,
                           const REAL_T *val_h, const int64_t *roff_h,
                           int64_t *idx_out_h, REAL_T *val_out_h, int64_t *count_out_h)
{
    // idx_out: [K, Ho, nbr_pad, 2]   -> (hi, wi_base)
    // val_out: [K, Ho, nbr_pad]
    // count_out: [K, Ho]
    // outputs are assumed pre-zeroed by the caller.

    for (int64_t i = 0; i < nrows; i++) {

        const int64_t soff = roff_h[i];
        const int64_t eoff = roff_h[i + 1];
        const int64_t k    = ker_h[soff];
        const int64_t ho   = row_h[soff];
        const int64_t cnt  = eoff - soff;

        if (cnt > nbr_pad) {
            fprintf(stderr,
                    "%s:%d: error, row (k=%ld, ho=%ld) has %ld entries, exceeds nbr_pad=%ld\n",
                    __FILE__, __LINE__, (long)k, (long)ho, (long)cnt, (long)nbr_pad);
            exit(EXIT_FAILURE);
        }

        const int64_t row_base = (k * Ho + ho) * nbr_pad;

        for (int64_t off = 0; off < cnt; off++) {
            const int64_t col = col_h[soff + off];
            const int64_t hi  = col / Wi;
            const int64_t wi  = col % Wi;
            idx_out_h[(row_base + off) * 2 + 0] = hi;
            idx_out_h[(row_base + off) * 2 + 1] = wi;
            val_out_h[row_base + off]           = val_h[soff + off];
        }

        count_out_h[k * Ho + ho] = cnt;
    }
}

template <typename REAL_T>
void preprocess_psi_kernel(int64_t nnz, int64_t K, int64_t Ho, int64_t *ker_h, int64_t *row_h, int64_t *col_h,
                           int64_t *roff_h, REAL_T *val_h, int64_t &nrows)
{

    int64_t *Koff = new int64_t[K];
    for (int i = 0; i < K; i++) { Koff[i] = 0; }

    for (int64_t i = 0; i < nnz; i++) { Koff[ker_h[i]]++; }

    int64_t prev = Koff[0];
    Koff[0] = 0;
    for (int i = 1; i < K; i++) {
        int64_t save = Koff[i];
        Koff[i] = prev + Koff[i - 1];
        prev = save;
    }

    int64_t *ker_sort = new int64_t[nnz];
    int64_t *row_sort = new int64_t[nnz];
    int64_t *col_sort = new int64_t[nnz];
    REAL_T *val_sort = new REAL_T[nnz];

    for (int64_t i = 0; i < nnz; i++) {

        const int64_t ker = ker_h[i];
        const int64_t off = Koff[ker]++;

        ker_sort[off] = ker;
        row_sort[off] = row_h[i];
        col_sort[off] = col_h[i];
        val_sort[off] = val_h[i];
    }
    for (int64_t i = 0; i < nnz; i++) {
        ker_h[i] = ker_sort[i];
        row_h[i] = row_sort[i];
        col_h[i] = col_sort[i];
        val_h[i] = val_sort[i];
    }

    delete[] Koff;
    delete[] ker_sort;
    delete[] row_sort;
    delete[] col_sort;
    delete[] val_sort;

    // compute rows offsets
    nrows = 1;
    roff_h[0] = 0;
    for (int64_t i = 1; i < nnz; i++) {

        if (row_h[i - 1] == row_h[i]) continue;
        roff_h[nrows++] = i;

        if (nrows > Ho * K) {
            fprintf(stderr, "%s:%d: error, found more rows in the K COOs than Ho*K (%ld)\n", __FILE__, __LINE__,
                    int64_t(Ho) * K);
            exit(EXIT_FAILURE);
        }
    }
    roff_h[nrows] = nnz;

    return;
}
