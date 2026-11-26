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

// =====================================================================================================================
// The DISCO forward contraction on the tensor cores of sm_100a (B200 / GB200), as GEMMs of a Hankel matrix of input
// longitudes and a Toeplitz matrix of filter values; float16 or bfloat16 activations, psi in the kpacked layout.
//
// Provenance: Kernel Factory campaign disco-fwd-b200-v2 (dssag5rgmx2036sgydnadrbwsr), delivered candidate 2c7ddc19
// (20.82x the PyTorch reference, geometric mean over its workloads). The device code below is the candidate's as
// delivered, with these changes, each marked "torch-harmonics":
//
// * fp16 and bf16: the kernels move 16-bit bit patterns; the MMA's operand format and the epilogue's rounding follow
//   a runtime flag, so bf16 adds no kernel variants;
// * output offsets 64-bit to the channel (the delivered kernel bounded the whole output to 2^32 elements and fell
//   back to a scalar kernel above);
// * tcgen05.ld results are read after tcgen05.wait::ld;
// * the input-extension blocks of the Toeplitz builder are a kernel of their own (xpre_kernel), the builder being
//   part of the plan; the per-call words of the co-resident protocol are set by init_kernel at the start of every
//   call, instead of being kept across calls and told apart by an epoch, which a CUDA graph freezes at capture;
// * the unused extension kernel and the scalar fallback are gone (sizes the tensor cores do not take run the
//   shipped kpacked forward).
//
// The host side at the end of the file is torch-harmonics' own. The campaign's final review left open that NaN or
// Inf in an input longitude the stencil does not reach may still reach outputs (the Toeplitz zeros multiply it);
// accepted.
//
// The candidate's description:
//
// DISCO S2 forward contraction (K-packed psi), fp16 in / fp32 accumulate / fp16 out.
//
// Formulation.  For every output row ho the support is split into "pieces": runs of points on
// one input latitude hi with consecutive longitudes (mod Wi).  For a piece with start S, n points
// and filter h_k[j] = val[point j][k]:
//   out[c,k,ho,wo] += sum_j h_k[j] * x[c, hi, S + j + wo*ps].
// Write wo*ps = 8*b + ps*dwo with dwo in [0, DW), DW = 8/ps, and S = B + delta (B a multiple of 8).
// Then x[c, hi, S + j + wo*ps] = x[c, hi, B + 8*b + u] with u = delta + j + ps*dwo, so
//   D[(c,b), (k,dwo)] = sum_u A[(c,b), u] * T[u, (k,dwo)],
//   A[(c,b), u] = x[c, hi, B + 8*b + u]      (Hankel in (b, u) at 8-element granularity)
//   T[u, (k,dwo)] = h_k[u - delta - ps*dwo]  (Toeplitz, zero outside the piece).
// A is a UMMA operand straight from a contiguous longitude run in shared memory: in the K-major
// no-swizzle core-matrix layout row r sits at 16*r bytes and the next 8-element K chunk at
// LBO = 16 bytes, so one copy of the longitude run serves all 128 rows (blocks) of an M tile.
// GEMM rows are (channel, block) pairs, so the tensor-core M dimension is fully used for any
// channel count; every block holds DW outputs, so outputs leave as DW-wide vectors.
// Rows of a channel are padded with HB halo blocks so that every block's window stays inside the
// channel's (wrapped) longitude run.

#include "disco_cuda_fwd_hankel_common.cuh"

// ------------------------------------------------------------------------------------------
// tcgen05 helpers
// ------------------------------------------------------------------------------------------
__device__ __forceinline__ uint64_t umma_desc(unsigned saddr, unsigned lbo, unsigned sbo)
{
    // K-major, no swizzle: core matrices of 8 rows x 16 B; lbo = K-direction stride,
    // sbo = M/N-direction stride (between 8-row groups)
    return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16)
        | ((uint64_t)((sbo >> 4) & 0x3FFFu) << 32) | (1ull << 46);
}
// K-major no-swizzle descriptor from 16-byte-unit start address and K-direction stride
__device__ __forceinline__ uint64_t mkdesc(unsigned start16, unsigned lbo16)
{
    return (uint64_t)(start16 | (lbo16 << 16)) | (8ull << 32) | (1ull << 46);
}
__device__ __forceinline__ void umma_f16(unsigned dtmem, uint64_t adesc, uint64_t bdesc, unsigned idesc, unsigned acc)
{
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
                 " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(dtmem),
                 "l"(adesc), "l"(bdesc), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void umma_commit(unsigned bar)
{
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tmem_ld16(unsigned taddr, unsigned *r)
{
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,"
                 "%15}, [%16];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld32(unsigned taddr, unsigned *r)
{
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                 "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                   "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                   "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
                 : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld8(unsigned taddr, unsigned *r)
{
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "r"(taddr));
}
__device__ __forceinline__ bool elect_one()
{
    unsigned pred = 0;
    asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(pred));
    return pred != 0;
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }

// ------------------------------------------------------------------------------------------
// Main kernel.  Item = (output row ho, group of G channels); GEMM row g = (c - c0) * RPC + b
// (block b of channel c; b >= nb are halo rows, discarded), M tiles of 128 rows, G * RPC <= 128 MT.
// Shared-memory slot g = (c - c0) * RPC + i holds x[c, hi, B + 8 i (mod Wi)] for the current
// piece, so the A operand of M tile m starts at slot 128 m.  The slots come from one 4-D TMA box
// of the longitude-extended input copy (TX) or from per-channel bulk copies (two runs at the seam).
// Warp roles: 0 producer, 1 MMA issuer (+TMEM), 2..5 epilogue (TMEM -> fp16 / bf16 -> global).
// TMEM holds two accumulator sets (MT x N columns each).
// ------------------------------------------------------------------------------------------
// warp 0 producer, warp 1 MMA, then 8 (one CTA per SM) or 4 (two CTAs per SM) epilogue warps

// SB: one TMEM accumulator set with per-tile release (more M tiles per item, used for wide N);
// otherwise two accumulator sets (the next item never waits for the epilogue).
template <int DW, bool SB_, int CPS = 1, int RG = 1> struct HkCfg {
    static constexpr int MN1 = KB * DW;                 // logical columns per output row
    static constexpr int MN = ((RG * MN1 + 7) / 8) * 8; // exact MMA width across row groups
    static constexpr int N = ((MN + 15) / 16) * 16;     // TMEM columns per M tile
    static constexpr bool SB = SB_;
    static constexpr int TCOLS = CPS == 3 ? 128 : 512 / CPS; // power-of-two TMEM allocation
    // one-CTA row pairs: a single epilogue quartet drains all M tiles (fewer resident warps)
    static constexpr int EPIW = CPS == 1 && RG == 2 ? 4 : (CPS == 1 ? 8 : 4); // epilogue warps
    static constexpr int MT0 = SB ? TCOLS / N : TCOLS / (2 * N);
    static constexpr int MT = MT0 > (SB ? 10 : 8) ? (SB ? 10 : 8) : MT0; // M tiles per item
    // MMA-issuing warps (tile m is issued by warp m % NMW): a single issuer spends most of a piece on
    // bookkeeping while the tensor core drains its few queued MMAs, so the tiles are spread over
    // several issuers that feed the tensor core in parallel
    // A one-CTA-per-SM configuration can spend the resident warp budget on one issuer per tile;
    // two- and three-CTA configurations retain three issuers per CTA.
    static constexpr int NMW_CAP = CPS == 1 ? 6 : 3;
    static constexpr int NMW = MT < NMW_CAP ? MT : NMW_CAP;
    // One-CTA kernels use a second producer so adjacent pieces can issue their TMA loads in
    // parallel; other configurations keep a single producer to preserve occupancy.
    static constexpr int NPW = CPS == 1 ? 2 : 1;
    static constexpr int THREADS = 32 * (NPW + NMW + EPIW);
    // Register budget: two-CTA configurations would fill the register file at 128 registers; they
    // are capped at 120 so one co-resident extension block (one 32-register warp per SM
    // sub-partition) still fits next to them.
    static constexpr bool RCAP = CPS == 2;
    static constexpr int NACC = SB ? MT : 2; // acc_empty barriers
    static constexpr int NACF = SB ? MT : 2; // acc_full barriers (SB: one per M tile)
};

template <int DW, bool TX, bool SB_, int CPS, int RG, bool LP, int NV, bool Z>
__device__ __forceinline__ void hk_body(const TmapX<NV> &tmap_x, const HkParams &p)
{
    static_assert(NV == 8 || NV == NVMAX);
    static_assert(!LP || RG == 1);
    using Cfg = HkCfg<DW, SB_, CPS, RG>;
    constexpr int N = Cfg::N;
    constexpr int MN = Cfg::MN;
    constexpr int MN1 = Cfg::MN1;
    constexpr int MT = Cfg::MT;
    constexpr bool SB = SB_;
    constexpr int NACC = Cfg::NACC;
    constexpr int NACF = Cfg::NACF;
    constexpr int EPIW = Cfg::EPIW;
    constexpr int NTH = Cfg::THREADS;
    constexpr int TCOLS = Cfg::TCOLS;
    constexpr int EG = EPIW / 4; // epilogue warps per TMEM lane quadrant
    constexpr int NMW = Cfg::NMW;
    constexpr int NPW = Cfg::NPW;
    extern __shared__ __align__(1024) unsigned char smem[];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (TX)
        if (warp == 0 && lane < (1 << p.nvs)) prefetch_tmap(&tmap_x.m[lane]);
    const int nst = p.nst;
    unsigned char *bar_area = smem + p.bar_off;
    uint64_t *bars = reinterpret_cast<uint64_t *>(bar_area);
    // [0,nst) full, [nst,2nst) empty, 2nst+b acc_full (b < NACF), 2nst+NACF+j acc_empty (j < NACC),
    // then item queue full/empty
    int *sks = reinterpret_cast<int *>(bar_area + 8 * (2 * nst + NACF + NACC + 2 * QN));
    int2 *qbuf = reinterpret_cast<int2 *>(bar_area + 8 * (2 * nst + NACF + NACC + 2 * QN) + 4 * 16);
    unsigned *tmem_slot = reinterpret_cast<unsigned *>(qbuf + QN);
    long long *qpb = reinterpret_cast<long long *>(bar_area + 512);
    const unsigned bar0 = smem_u32(bars);
    auto FULL = [&](int s) { return bar0 + 8u * s; };
    auto EMPTY = [&](int s) { return bar0 + 8u * (nst + s); };
    auto ACCF = [&](int b) { return bar0 + 8u * (2 * nst + b); };
    auto ACCE = [&](int b) { return bar0 + 8u * (2 * nst + NACF + b); };
    auto QFULL = [&](int q) { return bar0 + 8u * (2 * nst + NACF + NACC + q); };
    auto QEMPTY = [&](int q) { return bar0 + 8u * (2 * nst + NACF + NACC + QN + q); };
    const int grows = p.G * p.RPC; // GEMM rows filled from the slots

    if (tid == 0) {
        for (int s = 0; s < nst; ++s) {
            mbar_init(FULL(s), 1);
            // per MMA warp: thread arrival (stage metadata read) + tcgen05.commit (MMAs done)
            mbar_init(EMPTY(s), 2 * NMW);
        }
        for (int b = 0; b < NACF; ++b) mbar_init(ACCF(b), SB ? 1 : NMW);
        // SB: tile m is drained by the 4 epilogue warps of half (m & 1); else all 8 warps per set
        for (int j = 0; j < NACC; ++j) mbar_init(ACCE(j), SB ? 4 : EPIW);
        for (int q = 0; q < QN; ++q) {
            mbar_init(QFULL(q), 1);
            mbar_init(QEMPTY(q), NMW + EPIW + NPW - 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (warp == 1) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(tmem_slot)),
                     "n"(TCOLS)
                     : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
    }
    if constexpr (Z)
        for (int e = tid; e < MN; e += NTH) reinterpret_cast<uint4 *>(smem + p.z_off)[e] = make_uint4(0, 0, 0, 0);
    // slots past the loaded rows are read only by discarded GEMM rows: keep them finite
    for (int s = 0; s < nst; ++s) {
        uint4 *xs = reinterpret_cast<uint4 *>(smem + (size_t)s * p.stage_bytes);
        for (int e = grows + tid; e < p.xs_bytes / 16; e += NTH) xs[e] = make_uint4(0, 0, 0, 0);
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const unsigned tmem = *tmem_slot;
    const int ps = p.ps;
    pdl_wait(); // pieces, Toeplitz tiles, extended input (or its builder part) and the item counter
    // co-resident extension: the extension kernel is this kernel's programmatic dependent; it can
    // only become resident once every CTA of this grid is, so it never displaces one
    if (p.co) pdl_trigger();

    if (warp < NPW) {
        // ---------------- producer(s) ----------------
        const uint64_t tpolicy = reuse_policy();
        // co-resident extension: before a window of latitude hi is loaded, its channel rows must have
        // been extended.  Until the extension kernel has finished every task, each item's latitudes
        // are checked (one relaxed poll per lane, then an acquire fence and a proxy fence so that the
        // TMA reads observe the generic-proxy writes).
        // (compiled only into variants the host can run in co-resident mode: TMA input, CPS < 3)
        constexpr bool COK = TX && CPS < 3;
        bool xall = !COK || p.co == 0;
        bool xhelp_done = false;
        auto lat_check = [&](int h, bool valid) { // warp-collective
            if constexpr (COK) {
                if (xall) return;
                const unsigned xd = ld_relaxed_u32(p.stamps + ST_XSTAMP);
                // extension position of latitude h (alternately from both poles); the builder's prefix is
                // complete once the main kernel has started
                const int lpos = 2 * h < p.Hi ? 2 * h : 2 * (p.Hi - 1 - h) + 1;
                bool ok = !valid || lpos < p.xpre || ld_relaxed_u32(p.stamps + ST_LSTAMP + h) == p.epoch;
                // every lane observed the final value itself, so its acquire fence below synchronizes
                const bool fin = __all_sync(0xffffffffu, xd == p.epoch);
                if (!fin) {
                    while (!__all_sync(0xffffffffu, ok)) {
                        // help: claim and copy an extension task (claims run in priority order, so the missing
                        // latitude is reached); once every task is claimed, the claimants are running and finish
                        if (xhelp_done
                            || !xcopy_help(reinterpret_cast<const uint4 *>(p.x), p.xe, p.nb, p.Lc, p.Hi, p.BC, p.tpl,
                                           p.xpre, p.stamps, p.epoch, p.R, p.hrot, p.xres)) {
                            xhelp_done = true;
                            __nanosleep(128);
                        }
                        if (!ok) ok = ld_relaxed_u32(p.stamps + ST_LSTAMP + h) == p.epoch;
                    }
                }
                // each lane acquires what it observed; the warp barrier then orders every lane's acquire
                // before any lane's TMA issue (lane 0 issues windows of other lanes' pieces)
                fence_acq_rel_gpu();
                __syncwarp();
                fence_proxy_async_global();
                if (fin) xall = true;
            }
        };
        if constexpr (LP) {
            // For short input rows a single lane can issue each piece's independent TMA/bulk-copy
            // transaction.  Issue a prefix of ready stages in parallel across the producer warp;
            // scratch-overflow pieces still fall back to a whole-warp Toeplitz build.
            auto meta = [&](int it_, int &np_, int64_t &pb_) {
                np_ = 0;
                pb_ = 0;
                if (it_ < p.nitems) {
                    int h_, t_;
                    decode_item(p, it_, h_, t_);
                    np_ = p.npieces[h_];
                    pb_ = p.off[h_];
                }
            };
            int ticket_next = 0, ticket_left = 0;
            auto claim_one = [&]() {
                if (ticket_left == 0) {
                    int base = 0;
                    if (lane == 0) base = atomicAdd(p.ictr, p.ticket);
                    ticket_next = __shfl_sync(0xffffffffu, base, 0);
                    ticket_left = p.ticket;
                }
                --ticket_left;
                return ticket_next++;
            };

            int s = 0, P = 0, qs = 0, nq = 0;
            unsigned ph = 0, qph = 0;
            int item = (int)blockIdx.x;
            int item1 = claim_one();
            int np, np1;
            int64_t pbase, pb1;
            meta(item, np, pbase);
            meta(item1, np1, pb1);
            Piece mine;
            if (lane < np) mine = p.pieces[pbase + lane];
            int fut = claim_one();
            const unsigned xb = (unsigned)grows * 16u;
            const unsigned rpcb = (unsigned)p.RPC * 16u;

            while (true) {
                if (nq >= QN) mbar_wait(QEMPTY(qs), qph ^ 1u);
                if (lane == 0) {
                    qbuf[qs] = make_int2(item, np);
                    mbar_arrive(QFULL(qs));
                }
                __syncwarp();
                if (++qs == QN) {
                    qs = 0;
                    qph ^= 1u;
                }
                ++nq;
                if (item >= p.nitems) break;
                lat_check(lane < np ? mine.hi : 0, lane < np);

                int ho, cg;
                decode_item(p, item, ho, cg);
                const int c0 = cg * p.G;
                Piece nmine;
                if (lane < np1) nmine = p.pieces[pb1 + lane];
                const int item2 = fut;
                int np2;
                int64_t pb2;
                meta(item2, np2, pb2);
                fut = claim_one();

                for (int q0 = 0; q0 < np; q0 += 32) {
                    const int npass = min(32, np - q0);
                    if (q0 > 0) {
                        if (lane < npass) mine = p.pieces[pbase + q0 + lane];
                        lat_check(lane < npass ? mine.hi : 0, lane < npass);
                    }
                    int sl = s + lane;
                    unsigned phl = ph;
                    while (sl >= nst) {
                        sl -= nst;
                        phl ^= 1u;
                    }
                    const bool reused = P + lane >= nst;
                    const unsigned bad = __ballot_sync(0xffffffffu, lane < npass && mine.toff < 0);
                    int done = 0;
                    while (done < npass) {
                        const unsigned remaining_bad = bad & (0xffffffffu << done);
                        if (remaining_bad & (1u << done)) {
                            const int sd = __shfl_sync(0xffffffffu, sl, done);
                            const unsigned phd = __shfl_sync(0xffffffffu, phl, done);
                            if (__shfl_sync(0xffffffffu, (int)reused, done)) mbar_wait(EMPTY(sd), phd ^ 1u);

                            Piece pc;
                            pc.hi = __shfl_sync(0xffffffffu, mine.hi, done);
                            pc.S = __shfl_sync(0xffffffffu, mine.S, done);
                            pc.n = __shfl_sync(0xffffffffu, mine.n, done);
                            pc.p1 = __shfl_sync(0xffffffffu, mine.p1, done);
                            pc.n1 = __shfl_sync(0xffffffffu, mine.n1, done);
                            pc.p2 = __shfl_sync(0xffffffffu, mine.p2, done);
                            pc.jb = __shfl_sync(0xffffffffu, mine.jb, done);
                            pc.nc = __shfl_sync(0xffffffffu, mine.nc, done);
                            unsigned char *st = smem + (size_t)sd * p.stage_bytes;
                            const unsigned xa = smem_u32(st);
                            uint4 *ts = reinterpret_cast<uint4 *>(st + p.xs_bytes);
                            for (int e = lane; e < (pc.nc + (Z ? 0 : 1)) * MN; e += 32) {
                                const int kc = e / MN;
                                const int n = e - kc * MN;
                                const int k = n / DW;
                                ts[e] = kc < pc.nc ? toeplitz8(p.val, pc, 8 * kc, k, n - k * DW, ps, p.K, p.kpad) :
                                                     make_uint4(0, 0, 0, 0);
                            }
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            __syncwarp();
                            if (lane == 0) {
                                sks[sd] = pc.nc;
                                mbar_expect_tx(FULL(sd), xb);
                            }
                            __syncwarp();
                            const int B8 = pc.S >> 3;
                            if constexpr (TX) {
                                int q0 = B8 - p.hrot;
                                if (q0 < 0) q0 += p.nb;
                                if (lane == 0)
                                    tma_load_4d(xa, &tmap_x.m[q0 & ((1 << p.nvs) - 1)], 0, q0 >> p.nvs, pc.hi, c0,
                                                FULL(sd));
                            } else {
                                for (int cl = lane; cl < p.G; cl += 32) {
                                    const int c = c0 + cl;
                                    int ch = B8;
                                    const uint16_t *src = p.x + ((size_t)(c < p.BC ? c : p.BC - 1) * p.Hi + pc.hi) * p.Wi;
                                    unsigned dst = xa + (unsigned)cl * rpcb;
                                    for (int rem = p.RPC; rem;) {
                                        const int len = min(rem, p.nb - ch);
                                        bulk_load(dst, src + 8 * ch, (unsigned)len * 16u, FULL(sd));
                                        dst += (unsigned)len * 16u;
                                        rem -= len;
                                        ch = 0;
                                    }
                                }
                            }
                            __syncwarp();
                            ++done;
                            continue;
                        }

                        int lim = remaining_bad ? (__ffs(remaining_bad) - 1) : npass;
                        lim = min(lim, done + nst);
                        bool ready = false;
                        if (lane >= done && lane < lim) ready = !reused || mbar_test(EMPTY(sl), phl ^ 1u);
                        const unsigned ready_mask = __ballot_sync(0xffffffffu, ready);
                        const unsigned first_not_ready = ~(ready_mask >> done);
                        const int count = first_not_ready ? (__ffs(first_not_ready) - 1) : (32 - done);
                        if (count == 0) {
                            if (lane == done) mbar_wait(EMPTY(sl), phl ^ 1u);
                            __syncwarp();
                            continue;
                        }

                        if (lane >= done && lane < done + count) {
                            const unsigned tb = (unsigned)(mine.nc + (Z ? 0 : 1)) * MN * 16u;
                            const unsigned xa = smem_u32(smem + (size_t)sl * p.stage_bytes);
                            sks[sl] = mine.nc;
                            mbar_expect_tx(FULL(sl), xb + tb);
                            bulk_load_reuse(xa + (unsigned)p.xs_bytes, p.T + mine.toff, tb, FULL(sl), tpolicy);
                            const int B8 = mine.S >> 3;
                            if constexpr (TX) {
                                int q0 = B8 - p.hrot;
                                if (q0 < 0) q0 += p.nb;
                                tma_load_4d(xa, &tmap_x.m[q0 & ((1 << p.nvs) - 1)], 0, q0 >> p.nvs, mine.hi, c0,
                                            FULL(sl));
                            } else {
                                for (int cl = 0; cl < p.G; ++cl) {
                                    const int c = c0 + cl;
                                    int ch = B8;
                                    const uint16_t *src
                                        = p.x + ((size_t)(c < p.BC ? c : p.BC - 1) * p.Hi + mine.hi) * p.Wi;
                                    unsigned dst = xa + (unsigned)cl * rpcb;
                                    for (int rem = p.RPC; rem;) {
                                        const int len = min(rem, p.nb - ch);
                                        bulk_load(dst, src + 8 * ch, (unsigned)len * 16u, FULL(sl));
                                        dst += (unsigned)len * 16u;
                                        rem -= len;
                                        ch = 0;
                                    }
                                }
                            }
                        }
                        __syncwarp();
                        done += count;
                    }
                    P += npass;
                    s += npass;
                    while (s >= nst) {
                        s -= nst;
                        ph ^= 1u;
                    }
                }

                item = item1;
                np = np1;
                pbase = pb1;
                mine = nmine;
                item1 = item2;
                np1 = np2;
                pb1 = pb2;
            }
        } else {
            auto meta = [&](int it_, int &np_, int64_t &pb_) {
                np_ = 0;
                pb_ = 0;
                if (it_ < p.nitems) {
                    int h_, t_;
                    decode_item(p, it_, h_, t_);
                    np_ = p.npieces[h_];
                    pb_ = p.off[h_ * RG];
                }
            };
            int s = 0;
            unsigned ph = 0;
            int P = 0;
            int qs = 0, nq = 0;
            unsigned qph = 0;
            const unsigned xbytes = (unsigned)grows * 16u;
            // Both producer warps walk the same piece stream, but each issues alternating stages.
            auto issue_item = [&](int np, int64_t pbase, int c0, Piece &mine) {
                for (int q = 0; q < np; ++q, ++P) {
                    if (q > 0 && (q & 31) == 0) {
                        if (q + lane < np) mine = p.pieces[pbase + q + lane];
                        lat_check(q + lane < np ? mine.hi : 0, q + lane < np);
                    }
                    if (NPW > 1 && (P & 1) != warp) {
                        if (++s == nst) {
                            s = 0;
                            ph ^= 1u;
                        }
                        continue;
                    }
                    const int hi = __shfl_sync(0xffffffffu, mine.hi, q & 31);
                    const int S = __shfl_sync(0xffffffffu, mine.S, q & 31);
                    const int toff = __shfl_sync(0xffffffffu, mine.toff, q & 31);
                    const int nc = __shfl_sync(0xffffffffu, mine.nc, q & 31);
                    if (P >= nst) mbar_wait(EMPTY(s), ph ^ 1u);
                    unsigned char *st = smem + (size_t)s * p.stage_bytes;
                    const unsigned xa = smem_u32(st);
                    const unsigned ta = xa + (unsigned)p.xs_bytes;
                    const unsigned tbytes = (unsigned)((nc + (Z ? 0 : 1)) * MN * 16);
                    if (toff < 0) {
                        Piece pc;
                        pc.hi = hi;
                        pc.S = S;
                        pc.n = __shfl_sync(0xffffffffu, mine.n, q & 31);
                        pc.p1 = __shfl_sync(0xffffffffu, mine.p1, q & 31);
                        pc.n1 = __shfl_sync(0xffffffffu, mine.n1, q & 31);
                        pc.p2 = __shfl_sync(0xffffffffu, mine.p2, q & 31);
                        pc.jb = __shfl_sync(0xffffffffu, mine.jb, q & 31);
                        uint4 *ts = reinterpret_cast<uint4 *>(st + p.xs_bytes);
                        if (RG == 1) {
                            for (int e = lane; e < (nc + (Z ? 0 : 1)) * MN; e += 32) {
                                const int kc = e / MN;
                                const int n = e - kc * MN;
                                const int k = n / DW;
                                ts[e] = kc < nc ? toeplitz8(p.val, pc, 8 * kc, k, n - k * DW, ps, p.K, p.kpad) :
                                                  make_uint4(0, 0, 0, 0);
                            }
                        } else {
                            const PieceX px = p.pxs[pbase + q];
                            for (int e = lane; e < (nc + (Z ? 0 : 1)) * MN; e += 32) {
                                const int kc = e / MN;
                                const int n0 = e - kc * MN;
                                const int r = n0 / MN1;
                                const int n = n0 - r * MN1;
                                const int k = n / DW;
                                ts[e] = kc < nc ? toeplitz8x(p.val, pc.S, pc.n, pc.jb, px, r, 8 * kc, k, n - k * DW, ps,
                                                             p.K, p.kpad, p.Wi) :
                                                  make_uint4(0, 0, 0, 0);
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    }
                    if (toff < 0) __syncwarp();
                    const int B8 = S >> 3;
                    if (elect_one()) {
                        sks[s] = nc;
                        mbar_expect_tx(FULL(s), xbytes + (toff >= 0 ? tbytes : 0u));
                        if (toff >= 0) bulk_load_reuse(ta, p.T + toff, tbytes, FULL(s), tpolicy);
                        if (TX) {
                            int q0 = B8 - p.hrot;
                            if (q0 < 0) q0 += p.nb;
                            tma_load_4d(xa, &tmap_x.m[q0 & ((1 << p.nvs) - 1)], 0, q0 >> p.nvs, hi, c0, FULL(s));
                        }
                    }
                    if (!TX) {
                        for (int cl = lane; cl < p.G; cl += 32) {
                            const int c = c0 + cl;
                            int ch = B8;
                            const uint16_t *src = p.x + ((size_t)(c < p.BC ? c : p.BC - 1) * p.Hi + hi) * p.Wi;
                            unsigned d = xa + (unsigned)(cl * p.RPC) * 16u;
                            int rem = p.RPC;
                            while (rem > 0) {
                                const int l = min(rem, p.nb - ch);
                                bulk_load(d, src + 8 * ch, (unsigned)l * 16u, FULL(s));
                                d += (unsigned)l * 16u;
                                rem -= l;
                                ch = 0;
                            }
                        }
                    }
                    if (!TX) __syncwarp();
                    if (++s == nst) {
                        s = 0;
                        ph ^= 1u;
                    }
                }
            };
            if (NPW > 1 && warp == 1) {
                for (;;) {
                    mbar_wait(QFULL(qs), qph);
                    const int2 e = qbuf[qs];
                    const int64_t pbase = qpb[qs];
                    __syncwarp();
                    if (lane == 0) mbar_arrive(QEMPTY(qs));
                    if (++qs == QN) {
                        qs = 0;
                        qph ^= 1u;
                    }
                    const int item = e.x;
                    if (item >= p.nitems) break;
                    const int np = e.y;
                    int ho, cg;
                    decode_item(p, item, ho, cg);
                    Piece mine;
                    if (lane < np) mine = p.pieces[pbase + lane];
                    lat_check(lane < np ? mine.hi : 0, lane < np);
                    issue_item(np, pbase, cg * p.G, mine);
                }
            } else {
                int ticket_next = 0, ticket_left = 0;
                auto claim = [&]() {
                    if (ticket_left == 0) {
                        int base = 0;
                        if (lane == 0) base = atomicAdd(p.ictr, p.ticket);
                        ticket_next = __shfl_sync(0xffffffffu, base, 0);
                        ticket_left = p.ticket;
                    }
                    --ticket_left;
                    return ticket_next++;
                };
                // Item pipeline: while item i is issued, item i+1's pieces, item i+2's row metadata and the
                // claim of item i+3 are in flight (no dependent global round trips at item boundaries).
                int item = (int)blockIdx.x;
                int item1 = claim();
                int np, np1;
                int64_t pbase, pb1;
                meta(item, np, pbase);
                meta(item1, np1, pb1);
                Piece mine;
                if (lane < np) mine = p.pieces[pbase + lane];
                int fut = claim();
                while (true) {
                    // publish the item to the MMA, epilogue and second-producer warps
                    if (nq >= QN) mbar_wait(QEMPTY(qs), qph ^ 1u);
                    if (lane == 0) {
                        qbuf[qs] = make_int2(item, np);
                        if (NPW > 1) qpb[qs] = pbase;
                        mbar_arrive(QFULL(qs));
                    }
                    __syncwarp();
                    if (++qs == QN) {
                        qs = 0;
                        qph ^= 1u;
                    }
                    ++nq;
                    if (item >= p.nitems) break;
                    lat_check(lane < np ? mine.hi : 0, lane < np);
                    int ho, cg;
                    decode_item(p, item, ho, cg);
                    const int c0 = cg * p.G;
                    Piece nmine;
                    if (lane < np1) nmine = p.pieces[pb1 + lane];
                    const int item2 = fut;
                    int np2;
                    int64_t pb2;
                    meta(item2, np2, pb2);
                    fut = claim();
                    issue_item(np, pbase, c0, mine);
                    item = item1;
                    np = np1;
                    pbase = pb1;
                    mine = nmine;
                    item1 = item2;
                    np1 = np2;
                    pb1 = pb2;
                }
            }
        }
    } else if (warp < NPW + NMW) {
        // ---------------- MMA issuers: one elected lane per warp owns the schedule for tiles
        // m = mw (mod NMW).  Allocation/deallocation remain warp-collective outside this region.
        const int mw = warp - NPW;
        // operand format (bits 7-9 A, 10-12 B): 0 f16, 1 bf16 (torch-harmonics)
        const unsigned fmt = p.bf16 ? ((1u << 7) | (1u << 10)) : 0u;
        const unsigned idesc = (1u << 4) | fmt | ((unsigned)(MN >> 3) << 17) | ((128u >> 4) << 24);
        const unsigned idesc64 = (1u << 4) | fmt | ((unsigned)(MN >> 3) << 17) | ((64u >> 4) << 24);
        const uint64_t adesc0 = umma_desc(smem_u32(smem), 16u, 128u);
        const uint64_t bdesc0 = umma_desc(smem_u32(smem) + (unsigned)p.xs_bytes, (unsigned)(MN * 16), 128u);
        const unsigned z16 = (smem_u32(smem) + (unsigned)p.z_off) >> 4;
        const bool leader = elect_one();
        if (leader) {
            int s = 0;
            unsigned ph = 0;
            int qs = 0;
            unsigned qph = 0;
            for (int it = 0;; ++it) {
                mbar_wait(QFULL(qs), qph);
                const int2 e = qbuf[qs];
                mbar_arrive(QEMPTY(qs));
                if (++qs == QN) {
                    qs = 0;
                    qph ^= 1u;
                }
                const int item = e.x;
                if (item >= p.nitems) break;
                const int np = e.y;
                int ho, cg;
                decode_item(p, item, ho, cg);
                const int nch = min(p.G, p.BC - cg * p.G);
                // The last channel's halo rows are never stored, so exclude them from the exact tile count.
                const int nrow = (nch - 1) * p.RPC + p.nb;
                const int mtc = (nrow + 127) >> 7;
                const int mlast = nrow - 128 * (mtc - 1) <= 64 ? mtc - 1 : -1;
                const int ab = SB ? 0 : (it & 1);
                if (!SB && it >= 2) {
                    mbar_wait(ACCE(ab), ((it >> 1) - 1) & 1);
                    tc_fence_after();
                }
                if (np == 0) {
                    if (SB && it > 0)
                        for (int m = mw; m < MT; m += NMW) mbar_wait(ACCE(m), (it - 1) & 1);
                    if (SB)
                        for (int m = mw; m < MT; m += NMW) mbar_arrive(ACCF(m));
                    else
                        mbar_arrive(ACCF(ab));
                    continue;
                }
                const unsigned dbase = tmem + (unsigned)(ab * MT * N);
                // With one issuer per M tile, keep its item-invariant descriptor/address state once rather
                // than rebuilding the same select and multiply for every piece MMA.
                constexpr bool ONE = NMW == MT;
                const unsigned d_mw = dbase + (unsigned)(mw * N);
                const unsigned id_mw = mw == mlast ? idesc64 : idesc;
                const bool act_mw = mw < mtc;
                const unsigned off_mw = (unsigned)(mw * 128);
                // Chunk-packed issue: a k-step covers two 8-longitude chunks.  A piece's odd last chunk is
                // paired with the next piece's first chunk (both operands via an LBO spanning the two
                // stages) when the next stage lies at a higher shared-memory address.
                int pend = -1;           // stage whose last chunk is still pending
                unsigned pa = 0, pb = 0; // its A / B start (16-byte units)
                for (int q = 0; q < np; ++q) {
                    mbar_wait(FULL(s), ph); // TMA-written operands: the mbarrier completion orders them
                    const int nc = sks[s];
                    mbar_arrive(EMPTY(s));
                    const unsigned xs16 = (smem_u32(smem) + (unsigned)(s * p.stage_bytes)) >> 4;
                    const unsigned ts16 = xs16 + (unsigned)(p.xs_bytes >> 4);
                    auto bdesc = [&](unsigned c) -> uint64_t {
                        const unsigned a = ts16 + c * (unsigned)MN;
                        if constexpr (Z)
                            return (int)c + 1 == nc ? mkdesc(a, z16 - a) : mkdesc(a, (unsigned)MN);
                        else
                            return mkdesc(a, (unsigned)MN);
                    };
                    const bool comb = pend >= 0;
                    const int cs = comb ? 1 : 0;
                    const int rem = nc - cs;
                    // Pair an odd terminal chunk with the next piece instead of multiplying it by the
                    // stored zero chunk.  On the first piece, keep at least one pair so D is initialized
                    // before the cross-piece MMA accumulates into it.  Ring wrap is handled below by
                    // reversing both K halves, which leaves A*B unchanged and keeps descriptor LBO positive.
                    const bool carry = (rem & 1) && (q > 0 || nc > 1) && (q + 1 < np);
                    const int npair = carry ? (rem >> 1) : ((rem + 1) >> 1);
                    // SB, last piece: tile-major with a completion signal per tile, so the epilogue drains
                    // tile m while the later tiles still compute
                    const bool lastq = SB && q + 1 == np;
                    if (SB && q == 0) {
                        // first piece: tile-major, so tile m starts as soon as the epilogue released it
                        for (int m = mw; m < MT; m += NMW) {
                            if (it > 0) {
                                mbar_wait(ACCE(m), (it - 1) & 1);
                                tc_fence_after();
                            }
                            if (ONE ? act_mw : m < mtc)
                                for (int j = 0; j < npair; ++j)
                                    umma_f16(ONE ? d_mw : dbase + (unsigned)(m * N),
                                             mkdesc(xs16 + (unsigned)(2 * j) + (ONE ? off_mw : (unsigned)(m * 128)), 1u),
                                             bdesc((unsigned)(2 * j)), ONE ? id_mw : (m == mlast ? idesc64 : idesc),
                                             j > 0 ? 1u : 0u);
                            if (lastq) umma_commit(ACCF(m));
                        }
                    } else {
                        if (comb) {
                            const bool fwd = ts16 > pb;
                            const uint64_t bd = fwd ? mkdesc(pb, ts16 - pb) : mkdesc(ts16, pb - ts16);
#pragma unroll
                            for (int m = mw; m < MT; m += NMW)
                                if (ONE ? act_mw : m < mtc)
                                    umma_f16(ONE ? d_mw : dbase + (unsigned)(m * N),
                                             fwd ? mkdesc(pa + (ONE ? off_mw : (unsigned)(m * 128)), xs16 - pa) :
                                                   mkdesc(xs16 + (ONE ? off_mw : (unsigned)(m * 128)), pa - xs16),
                                             bd, ONE ? id_mw : (m == mlast ? idesc64 : idesc), 1u);
                            umma_commit(EMPTY(pend));
                        }
                        if (lastq) {
                            for (int m = mw; m < MT; m += NMW) {
                                if (ONE ? act_mw : m < mtc)
                                    for (int j = 0; j < npair; ++j) {
                                        const unsigned c = (unsigned)(cs + 2 * j);
                                        umma_f16(ONE ? d_mw : dbase + (unsigned)(m * N),
                                                 mkdesc(xs16 + c + (ONE ? off_mw : (unsigned)(m * 128)), 1u), bdesc(c),
                                                 ONE ? id_mw : (m == mlast ? idesc64 : idesc), 1u);
                                    }
                                umma_commit(ACCF(m));
                            }
                        } else {
                            for (int j = 0; j < npair; ++j) {
                                const unsigned c = (unsigned)(cs + 2 * j);
                                const uint64_t bd = bdesc(c);
                                const unsigned acc = (q > 0 || j > 0) ? 1u : 0u;
#pragma unroll
                                for (int m = mw; m < MT; m += NMW)
                                    if (ONE ? act_mw : m < mtc)
                                        umma_f16(ONE ? d_mw : dbase + (unsigned)(m * N),
                                                 mkdesc(xs16 + c + (ONE ? off_mw : (unsigned)(m * 128)), 1u), bd,
                                                 ONE ? id_mw : (m == mlast ? idesc64 : idesc), acc);
                            }
                        }
                    }
                    if (carry) {
                        pend = s;
                        pa = xs16 + (unsigned)(nc - 1);
                        pb = ts16 + (unsigned)((nc - 1) * MN);
                    } else {
                        pend = -1;
                        umma_commit(EMPTY(s));
                    }
                    if (++s == nst) {
                        s = 0;
                        ph ^= 1u;
                    }
                }
                if (!SB) umma_commit(ACCF(ab));
            }
        }
    } else {
        // ---------------- epilogue ----------------
        const int quad = warp & 3;                // TMEM lane quadrant accessible to this warp
        const int half = (warp - NPW - NMW) >> 2; // this warp drains tiles m = half (mod EG)
        const int lrow = quad * 32 + lane;
        // Output offsets: 64-bit to the channel, 32-bit within it (the host checks K * Ho * Wo < 2^32), so the
        // epilogue keeps mad.wide.u32 address generation while the whole output may exceed 2^32 elements
        // (torch-harmonics; the delivered kernel bounded the whole output to 32 bits).
        const unsigned kstride = (unsigned)p.Ho * (unsigned)p.Wo;
        int qs = 0;
        unsigned qph = 0;
        for (int it = 0;; ++it) {
            mbar_wait(QFULL(qs), qph);
            const int2 e = qbuf[qs];
            __syncwarp();
            if (lane == 0) mbar_arrive(QEMPTY(qs));
            if (++qs == QN) {
                qs = 0;
                qph ^= 1u;
            }
            const int item = e.x;
            if (item >= p.nitems) break;
            const int np = e.y;
            int ho, cg;
            decode_item(p, item, ho, cg);
            const int c0 = cg * p.G;
            const int nch = min(p.G, p.BC - c0);
            const int nrow = (nch - 1) * p.RPC + p.nb;
            const int mtc = (nrow + 127) >> 7;
            const int mlast = nrow - 128 * (mtc - 1) <= 64 ? mtc - 1 : -1;
            const int ab = SB ? 0 : (it & 1);
            if (!SB) {
                mbar_wait(ACCF(ab), (it >> 1) & 1);
                tc_fence_after();
            }
            // Each of this warp's tiles (m = half + EG i) is drained into packed fp16 registers and
            // released to the MMA warps (SB) before its global stores.
            constexpr int TPH = (MT + EG - 1) / EG;
            constexpr int NWT = N / 2;
            // half-block shift: GEMM row b holds outputs DW b + DW / 2 + dwo (mod Wo), stored as two
            // naturally aligned half vectors (the second one wraps at the seam for the last block)
            const bool hshift = DW >= 2 && p.rowsig[ho] != 0;
#pragma unroll
            for (int i = 0; i < TPH; ++i) {
                const int m = half + EG * i;
                if (m < MT) {
                    // Odd logical widths use this same register array first for FP32 TMEM data and then,
                    // in place, for packed fp16 pairs.  Packing low-to-high never overwrites an unread pair.
                    constexpr int HPREG = (MN & 15) ? MN : (MN == 144 ? 96 : NWT);
                    unsigned hp[HPREG];
                    if (SB) {
                        mbar_wait(ACCF(m), it & 1);
                        tc_fence_after();
                    }
                    if (np > 0 && m < mtc) {
                        const unsigned ta = tmem + ((unsigned)(quad * 32) << 16) + (unsigned)(ab * MT * N + m * N);
                        if constexpr ((MN & 15) != 0) {
                            static_assert((MN & 15) == 8);
                            constexpr int N32 = MN / 32;
                            constexpr int REM = MN & 31;
                            static_assert(REM == 8 || REM == 24);
#pragma unroll
                            for (int g = 0; g < N32; ++g) tmem_ld32(ta + (unsigned)(32 * g), hp + 32 * g);
                            if constexpr (REM == 24) {
                                tmem_ld16(ta + (unsigned)(MN - 24), hp + MN - 24);
                                tmem_ld8(ta + (unsigned)(MN - 8), hp + MN - 8);
                            } else {
                                tmem_ld8(ta + (unsigned)(MN - 8), hp + MN - 8);
                            }
                            tmem_wait_ld(); // torch-harmonics: before the registers are read
#pragma unroll
                            for (int w = 0; w < MN / 2; ++w) {
                                hp[w] = pack2(__uint_as_float(hp[2 * w]), __uint_as_float(hp[2 * w + 1]), p.bf16);
                            }
                            tmem_wait_ld();
                        } else {
                            if constexpr (MN == 144) {
                                // Drain the decoder's 144 columns in two waves.  The first 96 FP32 values are
                                // packed in place; their upper half is then reused for the final 48-column wave.
                                tmem_ld32(ta, hp);
                                tmem_ld32(ta + 32u, hp + 32);
                                tmem_ld32(ta + 64u, hp + 64);
                                tmem_wait_ld(); // torch-harmonics: before the registers are read
#pragma unroll
                                for (int w = 0; w < 48; ++w) {
                                    hp[w] = pack2(__uint_as_float(hp[2 * w]), __uint_as_float(hp[2 * w + 1]), p.bf16);
                                }
                                tmem_wait_ld();
                                tmem_ld32(ta + 96u, hp + 48);
                                tmem_ld16(ta + 128u, hp + 80);
                                tmem_wait_ld(); // torch-harmonics: before the registers are read
#pragma unroll
                                for (int w = 0; w < 24; ++w) {
                                    hp[48 + w] = pack2(__uint_as_float(hp[48 + 2 * w]),
                                                       __uint_as_float(hp[48 + 2 * w + 1]), p.bf16);
                                }
                                tmem_wait_ld();
                            } else {
#pragma unroll
                                for (int g0 = 0; g0 < MN / 16; g0 += 3) {
                                    unsigned r[48];
                                    if (g0 + 1 < MN / 16)
                                        tmem_ld32(ta + (unsigned)(16 * g0), r);
                                    else
                                        tmem_ld16(ta + (unsigned)(16 * g0), r);
                                    if (g0 + 2 < MN / 16) tmem_ld16(ta + (unsigned)(16 * (g0 + 2)), r + 32);
                                    tmem_wait_ld(); // torch-harmonics: before the registers are read
#pragma unroll
                                    for (int g = 0; g < 3; ++g)
                                        if (g0 + g < MN / 16)
#pragma unroll
                                            for (int w = 0; w < 8; ++w) {
                                                hp[8 * (g0 + g) + w]
                                                    = pack2(__uint_as_float(r[16 * g + 2 * w]),
                                                            __uint_as_float(r[16 * g + 2 * w + 1]), p.bf16);
                                            }
                                    tmem_wait_ld();
                                }
                            }
                        }
                    } else {
#pragma unroll
                        for (int w = 0; w < NWT; ++w) hp[w] = 0u;
                    }
                    if (SB) {
                        tc_fence_before();
                        __syncwarp();
                        if (lane == 0) mbar_arrive(ACCE(m));
                    }
                    if (m < mtc) do {
                            if (m == mlast && lane >= 16) break;
                            const int g = m * 128 + (m == mlast ? quad * 16 + lane : lrow);
                            const int cl = p.G == 2 ? (int)(g >= p.RPC) : g / p.RPC;
                            const int b = g - cl * p.RPC;
                            const int c = c0 + cl;
                            if constexpr (DW >= 2) {
                                if (hshift) {
                                    if (cl < nch && b < p.nb) {
                                        const int w1 = DW * b + DW / 2;
                                        const int w2 = (DW * b + DW == p.Wo) ? 0 : DW * b + DW;
#pragma unroll
                                        for (int r = 0; r < RG; ++r) {
                                            const int hr = ho * RG + r;
                                            if (hr >= p.Ho) break;
                                            uint16_t *rowp = p.out + (size_t)c * ((size_t)p.K * kstride)
                                                + (unsigned)hr * (unsigned)p.Wo;
#pragma unroll
                                            for (int k = 0; k < KB; ++k) {
                                                const int c0w = r * MN1 + k * DW;
                                                if (k < p.K) {
                                                    uint16_t *dk = rowp + k * kstride;
                                                    if constexpr (DW == 8) {
                                                        *reinterpret_cast<uint2 *>(dk + w1)
                                                            = make_uint2(hp[c0w / 2], hp[c0w / 2 + 1]);
                                                        *reinterpret_cast<uint2 *>(dk + w2)
                                                            = make_uint2(hp[c0w / 2 + 2], hp[c0w / 2 + 3]);
                                                    } else if constexpr (DW == 4) {
                                                        *reinterpret_cast<unsigned *>(dk + w1) = hp[c0w / 2];
                                                        *reinterpret_cast<unsigned *>(dk + w2) = hp[c0w / 2 + 1];
                                                    } else {
                                                        const unsigned w = hp[c0w / 2];
                                                        dk[w1] = (uint16_t)(w & 0xFFFFu);
                                                        dk[w2] = (uint16_t)(w >> 16);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    break;
                                }
                            }
                            if (cl < nch && b < p.nb) {
#pragma unroll
                                for (int r = 0; r < RG; ++r) {
                                    const int hr = ho * RG + r;
                                    if (hr >= p.Ho) break;
                                    uint16_t *dst = p.out + (size_t)c * ((size_t)p.K * kstride)
                                        + ((unsigned)hr * (unsigned)p.Wo + (unsigned)(DW * b));
#pragma unroll
                                    for (int k = 0; k < KB; ++k) {
                                        const int c0w = r * MN1 + k * DW; // first column of (row r, basis k)
                                        if (k < p.K) {
                                            if (DW == 8) {
                                                *reinterpret_cast<uint4 *>(dst + k * kstride) = make_uint4(
                                                    hp[c0w / 2], hp[c0w / 2 + 1], hp[c0w / 2 + 2], hp[c0w / 2 + 3]);
                                            } else if (DW == 4) {
                                                *reinterpret_cast<uint2 *>(dst + k * kstride)
                                                    = make_uint2(hp[c0w / 2], hp[c0w / 2 + 1]);
                                            } else if (DW == 2) {
                                                *reinterpret_cast<unsigned *>(dst + k * kstride) = hp[c0w / 2];
                                            } else {
                                                const unsigned w = hp[c0w >> 1];
                                                dst[k * kstride] = (uint16_t)((c0w & 1) ? (w >> 16) : (w & 0xFFFFu));
                                            }
                                        }
                                    }
                                }
                            }
                        } while (0);
                }
            }
            if (!SB) {
                tc_fence_before();
                __syncwarp();
                if (lane == 0) mbar_arrive(ACCE(ab));
            }
        }
    }

    __syncthreads();
    if (warp == 1) {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem), "n"(TCOLS) : "memory");
    }
}

template <int DW, bool TX, bool SB_, int CPS, int RG, bool LP, int NV, bool Z>
__global__ void __launch_bounds__(HkCfg<DW, SB_, CPS, RG>::THREADS, CPS)
    disco_hk_lb(const __grid_constant__ TmapX<NV> tmap_x, const HkParams p)
{
    hk_body<DW, TX, SB_, CPS, RG, LP, NV, Z>(tmap_x, p);
}
template <int DW, bool TX, bool SB_, int CPS, int RG, bool LP, int NV, bool Z>
__global__ void __maxnreg__(120) disco_hk_mr(const __grid_constant__ TmapX<NV> tmap_x, const HkParams p)
{
    hk_body<DW, TX, SB_, CPS, RG, LP, NV, Z>(tmap_x, p);
}
// the main-kernel entry of a configuration (register-capped where HkCfg::RCAP)
template <int DW, bool TX, bool SB_, int CPS, int RG, bool LP = false, int NV = NVMAX, bool Z = false>
static constexpr auto disco_hk_kernel()
{
    if constexpr (HkCfg<DW, SB_, CPS, RG>::RCAP)
        return &disco_hk_mr<DW, TX, SB_, CPS, RG, LP, NV, Z>;
    else
        return &disco_hk_lb<DW, TX, SB_, CPS, RG, LP, NV, Z>;
}

// =====================================================================================================================
// Host (torch-harmonics)
//
// The delivered host side (one call: preprocess psi, extend the input, run the main kernel, all on scratch from a
// process-wide workspace cache told apart by an epoch) is replaced by two ops:
//
// forward_hankel_sm100_plan
//     Everything the forward derives from psi for B * C planes of a dtype: the filter values in that dtype, the
//     pieces (flag pass and row / row-group planner), the Toeplitz tiles and the latitude ranges of the extended
//     copy. A byte buffer, built once per key by the plan of the DISCO backend and kept (disco_plan.py).
// forward_hankel_sm100_run
//     The forward through such a plan: per-call stamps and the extended input copy from torch's allocator on the
//     current stream, zeroed / published by init_kernel, so the call is safe to capture in a CUDA graph and to
//     replay; then the input extension, the main kernel and the co-resident extension, chained by programmatic
//     dependent launch as delivered.
//
// The launch configuration is a function of the sizes and the device alone (make_config): the plan and the run
// compute the same one. Sizes the tensor-core path does not take (pscale 3 or above 8, kernel_size above 9,
// nlon_out not a multiple of 8, ...) run the shipped kpacked forward instead, as does a plan of zero bytes.
// =====================================================================================================================

#if defined(DISCO_TARGET_SM90A) && DISCO_TARGET_SM90A
#error "disco_cuda_fwd_hankel_sm100.cu is compiled for sm_100a / sm_103a only, in torch_harmonics.disco._C_sm100"
#endif

// the family fields of a configuration: SB (one TMEM accumulator set, see HkCfg)
struct HkFamily {
    bool SB = false;
    static constexpr bool LANE_PARALLEL = true;
    template <class Cfg, auto F> void set_family() { SB = F; }
};
#define HK_NAME "forward_hankel_sm100"

#include "disco_cuda_fwd_hankel_host.cuh"

namespace
{

    // The main kernel of a configuration (as the delivered launch_hk2 chose it).
    template <int DW, auto SB, int CPS, int RG> const void *main_kernel(const HkConfig &c)
    {
        constexpr bool ZCFG = CPS != 3;
        switch (c.variant) {
        case MV_LP_NV:
        case MV_LP_8:
            if constexpr (RG == 1) {
                if (c.variant == MV_LP_NV)
                    return (const void *)disco_hk_kernel<DW, true, SB, CPS, RG, true, NVMAX, ZCFG>();
                return (const void *)disco_hk_kernel<DW, true, SB, CPS, RG, true, 8, ZCFG>();
            }
            return nullptr;
        case MV_TX_NV: return (const void *)disco_hk_kernel<DW, true, SB, CPS, RG, false, NVMAX, ZCFG>();
        case MV_TX_8: return (const void *)disco_hk_kernel<DW, true, SB, CPS, RG, false, 8, ZCFG>();
        default: return (const void *)disco_hk_kernel<DW, false, SB, CPS, RG, false, 8>();
        }
    }

    // One-CTA-per-SM single-buffered accumulators let an item hold twice the M tiles; the family choice of the
    // delivered launch_hk.
    template <int DW, int RG = 1> void configure_hk(HkConfig &c, const cudaDeviceProp &prop)
    {
        const int BC = c.BC, Wi = c.Wi, ps = c.ps;
        int pm, ksm;
        // Small channel counts expose too little independent work per CTA.  Three single-buffered CTAs
        // retain the same two-M-tile item geometry while increasing the number of resident pipelines.
        if (BC <= 32 && choose_group(Wi / 8, ps, HkCfg<DW, true, 3, RG>::MT, BC, pm, ksm) >= 1)
            return configure<DW, true, 3, RG>(c, prop);
        // Two CTAs per SM overlap one CTA's pipeline bubbles with the other's MMAs; used when channels are
        // plentiful.  With few channels the items are few and long (polar rows), and one CTA per SM is better.
        bool capped = true;
        const int g2 = choose_group(Wi / 8, ps, HkCfg<DW, true, 2, RG>::MT, BC, pm, ksm, &capped);
        if constexpr (RG == 1) {
            int pm1, ksm1;
            const int g1 = choose_group(Wi / 8, ps, HkCfg<DW, true, 1, RG>::MT, BC, pm1, ksm1);
            const int rpc1 = ((Wi / 8 + 2 * ksm1 - 1 + 7) & ~7);
            // Fuse the two resident narrow-row pipelines when doubling the M tiles also doubles the
            // channel group.
            if (g2 >= 4 && !capped && g1 >= 2 * g2 && rpc1 <= 128) return configure<DW, true, 1, RG>(c, prop);
        }
        if (g2 >= 1 && !capped) return configure<DW, true, 2, RG>(c, prop);
        if (choose_group(Wi / 8, ps, HkCfg<DW, false, 2, RG>::MT, BC, pm, ksm) >= 1)
            return configure<DW, false, 2, RG>(c, prop);
        const int gs = choose_group(Wi / 8, ps, HkCfg<DW, true, 1, RG>::MT, BC, pm, ksm);
        const int gd = choose_group(Wi / 8, ps, HkCfg<DW, false, 1, RG>::MT, BC, pm, ksm);
        if (gs >= 1 && 2 * gs >= 3 * gd) return configure<DW, true, 1, RG>(c, prop);
        return configure<DW, false, 1, RG>(c, prop);
    }

    // The configuration for (BC, K, Hi, Wi) -> (BC, K, Ho, Wo) through NNZ points padded to kpad values: the delivered
    // Run's dispatch.
    HkConfig make_config(int64_t BC64, int64_t K64, int64_t kpad64, int64_t Hi64, int64_t Wi64, int64_t Ho64,
                         int64_t Wo64, int64_t NNZ, const cudaDeviceProp &prop)
    {
        HkConfig c;
        if (!init_config(c, BC64, K64, kpad64, Hi64, Wi64, Ho64, Wo64, NNZ)) return c;
        const int BC = c.BC, Wi = c.Wi, Ho = c.Ho, ps = c.ps;
        const int nsm = prop.multiProcessorCount;
        if (ps == 1) {
            // Row pairs (N = 144): half the windows and MMAs per output; they pay when there are enough
            // row-pair items to balance the long polar ones over the SMs.  Rings that fit one 128-row tile
            // stay on single rows, whose producer issues the pieces lane-parallel.
            int pm, ksm;
            const int g1 = choose_group(Wi / 8, ps, HkCfg<8, true, 1, 2>::MT, BC, pm, ksm);
            const long long items2 = g1 >= 1 ? (long long)((Ho + 1) / 2) * ((BC + g1 - 1) / g1) : 0;
            const int gs = choose_group(Wi / 8, ps, HkCfg<8, true, 2, 1>::MT, BC, pm, ksm);
            const bool lane_par = gs >= 4 && ((Wi / 8 + 2 * ksm - 1 + 7) & ~7) <= 128;
            if (!lane_par && items2 >= 8LL * nsm)
                configure<8, true, 1, 2>(c, prop);
            else
                configure_hk<8>(c, prop);
        } else if (ps == 2) {
            int pm, ksm;
            const int g2 = choose_group(Wi / 8, ps, HkCfg<4, true, 2, 2>::MT, BC, pm, ksm);
            const long long items2 = g2 >= 1 ? (long long)((Ho + 1) / 2) * ((BC + g2 - 1) / g2) : 0;
            if (items2 >= 8LL * nsm)
                configure_hk<4, 2>(c, prop);
            else
                configure_hk<4, 1>(c, prop);
        } else if (ps == 4) {
            configure_hk<2>(c, prop);
        } else {
            configure_hk<1>(c, prop);
        }
        return c;
    }

    // f(std::integral_constant...) for the family of c: every family make_config can choose.
    template <class Fn> void with_family(const HkConfig &c, Fn &&f)
    {
#define DISCO_HK_FAMILY(DW_, SB_, CPS_, RG_)                                                                           \
    if (c.DW == DW_ && c.SB == SB_ && c.CPS == CPS_ && c.RG == RG_) return f.template operator()<DW_, SB_, CPS_, RG_>();
#define DISCO_HK_FAMILIES(DW_, RG_)                                                                                    \
    DISCO_HK_FAMILY(DW_, true, 3, RG_)                                                                                 \
    DISCO_HK_FAMILY(DW_, true, 1, RG_)                                                                                 \
    DISCO_HK_FAMILY(DW_, true, 2, RG_)                                                                                 \
    DISCO_HK_FAMILY(DW_, false, 2, RG_)                                                                                \
    DISCO_HK_FAMILY(DW_, false, 1, RG_)
        DISCO_HK_FAMILIES(8, 1)
        DISCO_HK_FAMILIES(8, 2)
        DISCO_HK_FAMILIES(4, 1)
        DISCO_HK_FAMILIES(4, 2)
        DISCO_HK_FAMILIES(2, 1)
        DISCO_HK_FAMILIES(1, 1)
#undef DISCO_HK_FAMILIES
#undef DISCO_HK_FAMILY
        TORCH_CHECK(false, "forward_hankel_sm100: no main kernel for DW ", c.DW, ", SB ", c.SB, ", CPS ", c.CPS,
                    ", RG ", c.RG);
    }

    void check_device(int device)
    {
        const cudaDeviceProp *prop = at::cuda::getDeviceProperties(device);
        TORCH_CHECK(prop->major == 10 && (prop->minor == 0 || prop->minor == 3),
                    "forward_hankel_sm100 requires a compute capability 10.0 or 10.3 device (Blackwell); got ",
                    prop->major, ".", prop->minor);
    }

} // namespace

namespace disco_kernels
{

    // the shipped tcgen05 kpacked forward (disco_cuda_fwd_dense_kpacked_sm100.cu): the sizes this kernel does not take
    torch::Tensor disco_cuda_fwd_kpacked_sm100(torch::Tensor inp, torch::Tensor pack_idx, torch::Tensor pack_val,
                                               torch::Tensor pack_offset, int64_t K, int64_t Ho, int64_t Wo);

    // The plan of the forward of BC planes of dtype (fp16 / bf16) on (Hi, Wi) through psi in the kpacked layout; zero
    // bytes where the shipped kpacked forward runs instead.
    at::Tensor forward_hankel_sm100_plan(at::Tensor pack_idx, at::Tensor pack_val, at::Tensor pack_offset, int64_t BC,
                                         int64_t K, int64_t Hi, int64_t Wi, int64_t Wo, at::ScalarType dtype)
    {
        return hankel_plan(pack_idx, pack_val, pack_offset, BC, K, Hi, Wi, Wo, dtype);
    }

    // The forward of the contiguous fp16 / bf16 (B, C, Hi, Wi) inp to (B, C, K, Ho, Wo) through the plan built for
    // B*C planes of its dtype.
    at::Tensor forward_hankel_sm100_run(at::Tensor inp, at::Tensor plan, at::Tensor pack_idx, at::Tensor pack_val,
                                        at::Tensor pack_offset, int64_t K, int64_t Ho, int64_t Wo)
    {
        return hankel_run(inp, plan, pack_idx, pack_val, pack_offset, K, Ho, Wo, &disco_cuda_fwd_kpacked_sm100);
    }

    // The configuration the forward takes for these sizes on the current device, for tests:
    // [ok, DW, SB, CPS, RG, variant, co, plan_bytes, scratch_bytes]
    std::vector<int64_t> forward_hankel_sm100_config(int64_t BC, int64_t K, int64_t kpad, int64_t Hi, int64_t Wi,
                                                     int64_t Ho, int64_t Wo, int64_t nnz)
    {
        const cudaDeviceProp &prop = *at::cuda::getCurrentDeviceProperties();
        const HkConfig c = make_config(BC, K, kpad, Hi, Wi, Ho, Wo, nnz, prop);
        return {c.ok, c.DW, c.SB, c.CPS, c.RG, c.variant, c.co, (int64_t)c.plan_bytes, (int64_t)c.scratch_bytes};
    }

    TORCH_LIBRARY_FRAGMENT(disco_kernels, m)
    {
        m.def(
            "forward_hankel_sm100_plan(Tensor pack_idx, Tensor pack_val, Tensor pack_offset, int bc, int kernel_size, "
            "int nlat_in, int nlon_in, int nlon_out, ScalarType dtype) -> Tensor");
        m.def("forward_hankel_sm100_run(Tensor inp, Tensor plan, Tensor pack_idx, Tensor pack_val, Tensor pack_offset, "
              "int kernel_size, int nlat_out, int nlon_out) -> Tensor",
              {at::Tag::pt2_compliant_tag});
        m.def("forward_hankel_sm100_config(int bc, int kernel_size, int kpad, int nlat_in, int nlon_in, int nlat_out, "
              "int nlon_out, int nnz) -> int[]");
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CUDA, m)
    {
        m.impl("forward_hankel_sm100_plan", &forward_hankel_sm100_plan);
        m.impl("forward_hankel_sm100_run", &forward_hankel_sm100_run);
    }

    TORCH_LIBRARY_IMPL(disco_kernels, CompositeExplicitAutograd, m)
    {
        m.impl("forward_hankel_sm100_config", &forward_hankel_sm100_config);
    }

} // namespace disco_kernels
