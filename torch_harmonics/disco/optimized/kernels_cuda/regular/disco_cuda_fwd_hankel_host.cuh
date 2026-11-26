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
// The host side that the sm_100a and sm_90a Hankel/Toeplitz forwards share (disco_cuda_fwd_hankel_sm100.cu,
// disco_cuda_fwd_hankel_sm90.cu): the launch configuration of a main-kernel family, the plan layout, the launches and
// the bodies of the plan and run ops.
//
// The including arch file provides, ahead of the include, the device code (disco_cuda_fwd_hankel_common.cuh and its
// main kernels with their HkCfg<DW, F, CPS, RG>), HK_NAME (the ops' name prefix, for messages) and
//
//     struct HkFamily {
//         // the arch's fields of a configuration's main-kernel family
//         static constexpr bool LANE_PARALLEL; // whether the arch has the lane-parallel producer variants
//         template <class Cfg, auto F> void set_family(); // sets the fields for the family parameter F
//     };
//
// and, after it, main_kernel, make_config, with_family and check_device (declared below).
// =====================================================================================================================

#pragma once

namespace
{

    constexpr int STAMP_WORDS = STAMP_WORDS_BYTES / 4;
    constexpr size_t MAIN_SMEM_ATTR = 227 * 1024;

    inline size_t align256(size_t v) { return ((v + 255) / 256) * 256; }

    // Piece length: the longest of {128, 96, 80, 64, 48, 32} points whose halo lets the channel groups
    // fill >= 95% of the M tiles (best fill otherwise).  Longer pieces waste less window per point.
    // Returns the channel group size G (0 if none fits) and the chosen pmax / k-step bound. (As delivered.)
    int choose_group(int nb, int ps, int MT, int BC, int &pmax, int &kstep_max, bool *capped = nullptr)
    {
        const int cands[6] = {128, 96, 80, 64, 48, 32};
        double best = -1.0;
        pmax = 0;
        kstep_max = 0;
        for (int ci = 0; ci < 6; ++ci) {
            const int nc = (7 + cands[ci] + 8 - ps + 7) / 8;
            const int ksm = (nc + 1) / 2;
            const int rpc = (nb + 2 * ksm - 1 + 7) & ~7;
            const int g = (128 * MT) / rpc;
            if (g < 1) continue;
            const double fill = (double)(g * rpc) / (128.0 * MT);
            if (fill >= 0.95) {
                pmax = cands[ci];
                kstep_max = ksm;
                break;
            }
            if (fill > best + 0.02) {
                best = fill;
                pmax = cands[ci];
                kstep_max = ksm;
            }
        }
        if (pmax == 0) return 0;
        int G = (128 * MT) / ((nb + 2 * kstep_max - 1 + 7) & ~7);
        // keep at least ~6 channel groups per row when channels are few, so that the long polar rows
        // are spread over several items (dynamic scheduling then balances them)
        const int gsmall = (BC + 5) / 6;
        if (capped) *capped = G > gsmall;
        if (G > gsmall) G = gsmall;
        return G;
    }

    // which main kernel of a family <DW, F, CPS, RG> runs (LP: the lane-parallel producer, sm_100a only)
    enum MainVariant : int { MV_LP_NV = 0, MV_LP_8 = 1, MV_TX_NV = 2, MV_TX_8 = 3, MV_BULK = 4 };

    struct HkConfig : HkFamily {
        bool ok = false;
        // sizes
        int BC = 0, Hi = 0, Wi = 0, Ho = 0, Wo = 0, K = 0, kpad = 0, ps = 0;
        int64_t NNZ = 0;
        int nsm = 0;
        // main kernel family and variant
        int DW = 0, CPS = 0, RG = 0;
        // (and the arch's fields, HkFamily)
        int variant = MV_BULK;
        bool tx = false, shared_z = false, pdl = false, co = false, use_R = false;
        // launch parameters (HkParams and the preprocessing)
        int nb = 0, HB = 0, RPC = 0, G = 0, NCG = 0, NG = 0, nitems = 0, ticket = 1, nvs = 3;
        unsigned ncg_recip = 0;
        int nst = 0, xs_bytes = 0, stage_bytes = 0, bar_off = 0, z_off = 0;
        int nc_cap = 0, nunit = 0, grid = 0, TY = 0, tpl = 0, xpre = 0, xres = 0, hrot = 0, Lc = 0, threads = 0;
        long long cap = 0;
        size_t smem = 0;
        // plan layout (bytes)
        size_t o_val = 0, o_pieces = 0, o_sub = 0, o_np = 0, o_flags = 0, o_cnt = 0, o_T = 0, o_flat = 0, o_arcs = 0,
               o_pxs = 0, o_sig = 0, o_R = 0, plan_bytes = 0;
        // per-call scratch: the stamp words, then the extended input copy
        size_t xe_bytes = 0, scratch_bytes = 0;
    };

    // provided by the arch file:
    // the main kernel of a configuration of the family <DW, F, CPS, RG>
    template <int DW, auto F, int CPS, int RG> const void *main_kernel(const HkConfig &c);
    // the configuration for these sizes on a device (c.ok false: the shipped kpacked forward runs instead)
    HkConfig make_config(int64_t BC64, int64_t K64, int64_t kpad64, int64_t Hi64, int64_t Wi64, int64_t Ho64,
                         int64_t Wo64, int64_t NNZ, const cudaDeviceProp &prop);
    // f.operator()<DW, F, CPS, RG>() for the family of c
    template <class Fn> void with_family(const HkConfig &c, Fn &&f);
    // a TORCH_CHECK that the device is one the arch file's kernels run on
    void check_device(int device);

    // Can one extension block share an SM with CPS main-kernel CTAs?  (Only a performance question:
    // the extension never waits for the main kernel, so without room the main CTAs simply start later.)
    // As delivered, with the device's limits from torch's cached properties.
    bool co_fits(const void *kfn, int nth, int cps, size_t smem, const cudaDeviceProp &prop)
    {
        cudaFuncAttributes fa, fx;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kfn));
        C10_CUDA_CHECK(cudaFuncGetAttributes(&fx, (const void *)xcopy_kernel));
        const int smem_sm = (int)prop.sharedMemPerMultiprocessor, regs_sm = prop.regsPerMultiprocessor;
        const int thr_sm = prop.maxThreadsPerMultiProcessor, res = (int)prop.reservedSharedMemPerBlock;
        // registers are allocated per warp (multiples of 256) from the four sub-partitions' files, the
        // warps of a CTA spread across them
        auto wregs = [](int r) { return ((r * 32 + 255) / 256) * 256; };
        const int mw = (nth + 31) / 32, xw = (XC_THREADS + 31) / 32;
        const long long regs = 4LL
            * ((long long)cps * ((mw + 3) / 4) * wregs(fa.numRegs) + (long long)((xw + 3) / 4) * wregs(fx.numRegs));
        const long long sm
            = (long long)cps * ((long long)smem + fa.sharedSizeBytes + res) + (long long)fx.sharedSizeBytes + res;
        return regs <= regs_sm && sm <= smem_sm && cps * nth + XC_THREADS <= thr_sm;
    }

    // The configuration of the family <DW, F, CPS, RG> (the delivered launch_hk2 up to its launches; F: the arch's
    // family parameter, see HkFamily). Leaves
    // c.ok false where the delivered kernel would have fallen back.
    template <int DW, auto F, int CPS, int RG> void configure(HkConfig &c, const cudaDeviceProp &prop)
    {
        using Cfg = HkCfg<DW, F, CPS, RG>;
        constexpr int MN = Cfg::MN;
        constexpr int MT = Cfg::MT;
        const int BC = c.BC, Hi = c.Hi, Wi = c.Wi, Ho = c.Ho, ps = c.ps;
        const int64_t NNZ = c.NNZ;
        c.DW = DW;
        c.set_family<Cfg, F>();
        c.CPS = CPS;
        c.RG = RG;
        c.threads = Cfg::THREADS;
        c.nunit = 2 * MN; // 16-byte units per k-step tile
        // The paired-row planner can reserve up to three MN-wide records per input point.  Avoid
        // signed reservation wrap on unusually large but otherwise valid sparse layouts.
        if (3ull * (uint64_t)MN * (uint64_t)NNZ > (uint64_t)INT32_MAX) return;
        c.nb = Wi / 8;
        int pmax = 0, kstep_max = 0;
        c.G = choose_group(c.nb, ps, MT, BC, pmax, kstep_max);
        if (c.G < 1) return;
        // The deep-ring trim needs kstep_max >= 4 so that pieces keep nc_cap >= 2 chunks: with
        // nc_cap >= 2 a run of L >= 1 points emits at most ceil(((L + 21) / 8) / 2) <= L pieces, which
        // bounds the planner's Piece / PieceX / flat arrays by NNZ.
        const bool deep_wide_ps1 = CPS == 1 && RG == 2 && ps == 1 && c.nb >= 128 && kstep_max >= 4;
        c.HB = 2 * kstep_max - 1 - (deep_wide_ps1 ? 5 : 0);
        c.RPC = (c.nb + c.HB + 7) & ~7; // per-channel slot stride (whole 128-byte TMA rows)
        // One-CTA row pairs trade two terminal chunks for a seventh producer/MMA stage.
        c.nc_cap = 2 * kstep_max - ((CPS == 1 && RG == 2 && kstep_max >= 2) ? (deep_wide_ps1 ? 5 : 2) : 0);
        if (c.nc_cap < 2 || c.HB < c.nc_cap - 1) return; // planner capacity / halo invariants
        c.NCG = (BC + c.G - 1) / c.G;
        c.ncg_recip = c.NCG == 1 ? UINT32_MAX : (unsigned)((1ull << 32) / (unsigned)c.NCG);
        c.NG = (Ho + RG - 1) / RG;
        const long long nitems64 = (long long)c.NG * c.NCG;
        if (nitems64 > INT32_MAX) return;
        c.nitems = (int)nitems64;
        c.tx = c.RPC / 8 <= 256 && c.G <= 256;
        // 512-byte TMA rows when the channel pitch is a whole number of them, else 128-byte rows
        c.nvs = (c.RPC % 32 == 0) ? 5 : 3;
        constexpr bool ZCFG = CPS != 3;
        c.shared_z = c.tx && ZCFG;
        const int nv = 1 << c.nvs;
        c.Lc = ((c.nb + nv - 1) & ~(nv - 1)) + c.RPC; // minimal whole-TMA-row extension

        const size_t nn = (size_t)(NNZ > 0 ? NNZ : 1);
        c.cap = (long long)c.nunit * ((long long)nn / 8 + (long long)Ho * 64 + 4096);
        if (c.cap > INT32_MAX) return; // Piece::toff and the planner's allocation counter are int32
        const uint64_t xrows = (uint64_t)BC * (uint64_t)Hi;
        if (c.tx && xrows > SIZE_MAX / ((size_t)c.Lc * 16)) return;

        c.nsm = prop.multiProcessorCount;
        c.grid = c.nitems < CPS * c.nsm ? c.nitems : CPS * c.nsm;
        const size_t limit = (CPS == 1 ? 227 : (CPS == 2 ? 113 : (CPS == 3 ? 75 : 56))) * 1024;
        c.xs_bytes = (((128 * MT + c.HB) * 16 + 127) / 128) * 128;
        const int ts_bytes = (c.nc_cap + (c.shared_z ? 0 : 1)) * MN * 16;
        c.stage_bytes = ((c.xs_bytes + ts_bytes + 127) / 128) * 128;
        const int bar_bytes = 1024;
        const int z_bytes = c.shared_z ? ((MN * 16 + 127) / 128) * 128 : 0;
        int nst = (int)((limit - bar_bytes - z_bytes) / c.stage_bytes);
        if (nst > 16) nst = 16;
        if (nst < 2) return;
        c.nst = nst;
        c.bar_off = nst * c.stage_bytes;
        c.z_off = c.bar_off + bar_bytes;
        c.smem = (size_t)c.bar_off + bar_bytes + z_bytes;
        c.ticket = c.nitems >= 64 * c.grid ? 4 : 1;

        // the lane-parallel producer (where the arch has it) for short rows of enough channel groups
        const bool lp = HkFamily::LANE_PARALLEL && RG == 1 && CPS != 1 && c.tx && c.G >= 4 && c.RPC <= 128;
        if (lp)
            c.variant = c.nvs == 5 ? MV_LP_NV : MV_LP_8;
        else if (c.tx && c.nvs == 5)
            c.variant = MV_TX_NV;
        else if (c.tx)
            c.variant = MV_TX_8;
        else
            c.variant = MV_BULK;

        // Overlapping the planner, the Toeplitz build and the main kernel's prologue with their
        // predecessors pays on the regular paths; the three-CTA small-channel path runs strictly serialized.
        c.pdl = CPS != 3;
        // Co-resident extension: the longitude-extended input is written by a separate kernel that runs
        // next to the main kernel instead of in front of it.
        c.co = c.pdl && c.tx && Hi <= CO_MAXHI
            && co_fits(main_kernel<DW, F, CPS, RG>(c), Cfg::THREADS, CPS, c.smem, prop);
        c.TY = CPS == 3 ? 8 : 6;
        c.tpl = (BC + XC_RPT - 1) / XC_RPT;
        // co-resident mode: the input extension ahead of the main kernel covers the first fifth of the latitude
        // order (the polar rows the first items need), the extension kernel the rest while the main kernel runs
        // (half when the support is sparse relative to the input grid)
        const double xdens = (double)NNZ * (double)c.Wo / ((double)Hi * (double)Wi);
        c.xpre = c.co ? (xdens < 256.0 ? (Hi + 1) / 2 : (Hi + 4) / 5) : 0;
        c.xres = c.co ? std::min(std::min(c.nsm, XC_MAXRES), (Hi - c.xpre) * c.tpl) : 0;
        c.hrot = c.tx ? c.nb / 2 : 0;
        // extended copy restricted to the windows' latitude ranges (rotated by half a turn)
        c.use_R = c.tx && c.pdl && Hi <= CO_MAXHI;

        // the plan
        size_t o = 0;
        auto put = [&](size_t bytes) {
            const size_t at = o;
            o += align256(bytes);
            return at;
        };
        c.o_val = put((size_t)NNZ * c.kpad * 2);
        c.o_pieces = put(nn * sizeof(Piece));
        c.o_sub = put(nn * 4);
        c.o_np = put((size_t)Ho * 4);
        c.o_flags = put(nn);
        c.o_cnt = put(256);
        c.o_T = put((size_t)c.cap * 16);
        c.o_flat = put(nn * 4);
        c.o_arcs = put(RG > 1 ? nn * sizeof(Arc) : 0);
        c.o_pxs = put(RG > 1 ? nn * sizeof(PieceX) : 0);
        c.o_sig = put((size_t)Ho * 4);
        c.o_R = put(c.use_R ? (size_t)16 * Hi : 0);
        c.plan_bytes = o;
        c.xe_bytes = c.tx ? align256((size_t)BC * Hi * c.Lc * 16) : 0;
        c.scratch_bytes = (size_t)STAMP_WORDS_BYTES + c.xe_bytes;
        c.ok = true;
    }

    // The sizes of a configuration (BC, K, Hi, Wi) -> (BC, K, Ho, Wo) through NNZ points padded to kpad values; false
    // (c.ok false) for sizes the tensor-core path does not take.
    bool init_config(HkConfig &c, int64_t BC64, int64_t K64, int64_t kpad64, int64_t Hi64, int64_t Wi64, int64_t Ho64,
                     int64_t Wo64, int64_t NNZ)
    {
        if (Ho64 <= 0 || BC64 <= 0 || Wo64 <= 0 || K64 <= 0) return false;
        const int64_t ps64 = Wi64 / Wo64;
        const bool fits_i32 = BC64 <= INT32_MAX && Hi64 <= INT32_MAX && Wi64 <= INT32_MAX && Ho64 <= INT32_MAX
            && Wo64 <= INT32_MAX && NNZ <= INT32_MAX;
        // offsets within one channel stay 32-bit (see the epilogue)
        const bool chan_i32 = (uint64_t)K64 * (uint64_t)Ho64 * (uint64_t)Wo64 <= (uint64_t)UINT32_MAX;
        const bool hk_ok = fits_i32 && chan_i32 && ps64 >= 1 && Wi64 == ps64 * Wo64 && Wo64 % 8 == 0 && 8 % ps64 == 0
            && K64 <= KB && kpad64 >= K64 && (kpad64 == 8 || kpad64 == 16) && NNZ > 0;
        if (!hk_ok) return false;
        c.BC = (int)BC64;
        c.Hi = (int)Hi64;
        c.Wi = (int)Wi64;
        c.Ho = (int)Ho64;
        c.Wo = (int)Wo64;
        c.K = (int)K64;
        c.kpad = (int)kpad64;
        c.ps = (int)ps64;
        c.NNZ = NNZ;
        return true;
    }

    // launch with programmatic stream serialization when pdl is set (the kernel calls pdl_wait before
    // consuming the predecessor's results; without the attribute pdl_wait/pdl_trigger are no-ops)
    template <typename... KArgs, typename... Args>
    void launch_pdl(bool pdl, void (*k)(KArgs...), unsigned grid, unsigned block, size_t smem, cudaStream_t st,
                    Args... args)
    {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(grid);
        cfg.blockDim = dim3(block);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = st;
        cudaLaunchAttribute a[1];
        a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        a[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = a;
        cfg.numAttrs = pdl ? 1 : 0;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, args...));
    }

    // cuTensorMapEncodeTiled, from the driver the runtime uses (no link against libcuda)
    using EncodeTiledFn = CUresult (*)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
                                       const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave,
                                       CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

    EncodeTiledFn encode_tiled()
    {
        static const EncodeTiledFn fn = [] {
            void *p = nullptr;
            cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
            C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
#else
            C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q));
#endif
            TORCH_CHECK(q == cudaDriverEntryPointSuccess && p != nullptr,
                        HK_NAME ": the driver has no cuTensorMapEncodeTiled");
            return reinterpret_cast<EncodeTiledFn>(p);
        }();
        return fn;
    }

    void check_plan_inputs(const at::Tensor &pack_idx, const at::Tensor &pack_val, const at::Tensor &pack_offset)
    {
        TORCH_CHECK(pack_idx.is_cuda(), HK_NAME ": psi must be on a CUDA device, got ", pack_idx.device());
        disco_kernels::check_same_device(pack_val, pack_idx, "pack_val");
        disco_kernels::check_same_device(pack_offset, pack_idx, "pack_offset");
        TORCH_CHECK(pack_idx.scalar_type() == at::kLong && pack_idx.dim() == 2 && pack_idx.size(1) == 2,
                    "pack_idx must be an int64 (nnz, 2) tensor, got ", pack_idx.scalar_type(), " of shape ",
                    pack_idx.sizes());
        TORCH_CHECK(pack_val.dim() == 2 && pack_val.size(0) == pack_idx.size(0)
                        && at::isFloatingType(pack_val.scalar_type()),
                    "pack_val must be a floating (nnz, K_pad) tensor with one row per pack_idx entry, got ",
                    pack_val.scalar_type(), " of shape ", pack_val.sizes());
        disco_kernels::check_index_vector(pack_offset, at::kLong, "pack_offset");
        disco_kernels::check_dense(pack_idx, "pack_idx");
        disco_kernels::check_dense(pack_val, "pack_val");
    }

    // the Toeplitz builder of a family (plan)
    struct TfillLaunch {
        const HkConfig &c;
        cudaStream_t st;
        const uint16_t *val;
        const Piece *pieces;
        const PieceX *pxs;
        uint4 *T;
        const unsigned *counter;
        const int *flat;

        template <int RG, int DW, int KPAD, bool Z> void go() const
        {
            tfill_kernel<RG, DW, KPAD, Z><<<(unsigned)(c.nsm * c.TY), 128, 0, st>>>(val, pieces, pxs, c.Wi, c.ps, c.K,
                                                                                    T, c.nsm, c.TY, counter, flat);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        template <int DW, auto F, int CPS, int RG> void operator()() const
        {
            if (c.kpad == 16) {
                if (c.shared_z)
                    go<RG, DW, 16, true>();
                else
                    go<RG, DW, 16, false>();
            } else {
                if (c.shared_z)
                    go<RG, DW, 8, true>();
                else
                    go<RG, DW, 8, false>();
            }
        }
    };

    // the main kernel of a family (run)
    struct MainLaunch {
        const HkConfig &c;
        cudaStream_t st;
        TmapX<NVMAX> *tmap;
        HkParams *prm;

        template <int DW, auto F, int CPS, int RG> void operator()() const
        {
            const void *kfn = main_kernel<DW, F, CPS, RG>(c);
            C10_CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)MAIN_SMEM_ATTR));
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3((unsigned)c.grid);
            cfg.blockDim = dim3((unsigned)c.threads);
            cfg.dynamicSmemBytes = c.smem;
            cfg.stream = st;
            cudaLaunchAttribute a[1];
            a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            a[0].val.programmaticStreamSerializationAllowed = 1;
            cfg.attrs = a;
            cfg.numAttrs = c.pdl ? 1 : 0;
            void *args[2] = {(void *)tmap, (void *)prm};
            C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg, kfn, args));
        }
    };

    // The plan of the forward of BC planes of dtype (fp16 / bf16) on (Hi, Wi) through psi in the kpacked layout; zero
    // bytes where the shipped kpacked forward runs instead.
    at::Tensor hankel_plan(at::Tensor pack_idx, at::Tensor pack_val, at::Tensor pack_offset, int64_t BC, int64_t K,
                           int64_t Hi, int64_t Wi, int64_t Wo, at::ScalarType dtype)
    {
        check_plan_inputs(pack_idx, pack_val, pack_offset);
        TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, HK_NAME " takes fp16 or bf16, got ", dtype);
        const c10::cuda::CUDAGuard guard(pack_idx.device());
        check_device(pack_idx.get_device());
        const cudaDeviceProp &prop = *at::cuda::getDeviceProperties(pack_idx.get_device());
        const int64_t Ho = pack_offset.size(0) - 1;
        const HkConfig c = make_config(BC, K, pack_val.size(1), Hi, Wi, Ho, Wo, pack_idx.size(0), prop);
        auto plan = at::empty({c.ok ? (int64_t)c.plan_bytes : 0}, pack_idx.options().dtype(at::kByte));
        if (!c.ok) return plan;

        cudaStream_t st = at::cuda::getCurrentCUDAStream();
        unsigned char *base = static_cast<unsigned char *>(plan.data_ptr());
        uint16_t *val = reinterpret_cast<uint16_t *>(base + c.o_val);
        Piece *pieces = reinterpret_cast<Piece *>(base + c.o_pieces);
        int *substart = reinterpret_cast<int *>(base + c.o_sub);
        int *npieces = reinterpret_cast<int *>(base + c.o_np);
        unsigned char *flags = base + c.o_flags;
        unsigned *counter = reinterpret_cast<unsigned *>(base + c.o_cnt);
        uint4 *T = reinterpret_cast<uint4 *>(base + c.o_T);
        int *flat = reinterpret_cast<int *>(base + c.o_flat);
        Arc *arcs = reinterpret_cast<Arc *>(base + c.o_arcs);
        PieceX *pxs = reinterpret_cast<PieceX *>(base + c.o_pxs);
        int *rowsig = reinterpret_cast<int *>(base + c.o_sig);
        unsigned long long *R = c.use_R ? reinterpret_cast<unsigned long long *>(base + c.o_R) : nullptr;
        const int64_t *idx = pack_idx.data_ptr<int64_t>();
        const int64_t *off = pack_offset.data_ptr<int64_t>();

        // the filter values in the activations' dtype
        const at::Tensor vals = pack_val.to(dtype).contiguous();
        C10_CUDA_CHECK(cudaMemcpyAsync(val, vals.data_ptr(), (size_t)c.NNZ * c.kpad * 2, cudaMemcpyDeviceToDevice, st));

        const unsigned fgrid = (unsigned)((c.NNZ + 255) / 256);
        if (R) {
            // the latitude ranges of the extended copy (a call's epoch is 1, see init_kernel)
            C10_CUDA_CHECK(cudaMemsetAsync(R, 0, (size_t)16 * c.Hi, st));
            range_kernel<<<fgrid, 256, 0, st>>>(idx, c.NNZ, R, c.nb, c.hrot, c.nc_cap, c.RPC, 1u);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        flag_kernel<<<fgrid, 256, 0, st>>>(idx, c.NNZ, c.Wi, flags, counter);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (c.RG == 1)
            row_kernel<<<(unsigned)c.Ho, 256, 0, st>>>(idx, off, flags, c.Wi, c.ps, c.nc_cap, c.nunit, substart, pieces,
                                                       npieces, counter, c.cap, flat, rowsig);
        else
            group_kernel<<<(unsigned)c.NG, GT, 0, st>>>(idx, off, flags, c.Wi, c.ps, c.nc_cap, c.nunit, c.Ho, substart,
                                                        arcs, pieces, pxs, npieces, counter, c.cap, flat, rowsig);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // the Toeplitz tiles
        with_family(c, TfillLaunch {c, st, val, pieces, pxs, T, counter, flat});
        return plan;
    }

    // The forward of the contiguous fp16 / bf16 (B, C, Hi, Wi) inp to (B, C, K, Ho, Wo) through the plan built for
    // B*C planes of its dtype; kpacked: the arch's shipped kpacked forward, for the sizes this kernel does not take.
    using KpackedForward
        = torch::Tensor (*)(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
    at::Tensor hankel_run(at::Tensor inp, at::Tensor plan, at::Tensor pack_idx, at::Tensor pack_val,
                          at::Tensor pack_offset, int64_t K, int64_t Ho, int64_t Wo, KpackedForward kpacked)
    {
        TORCH_CHECK(inp.is_cuda(), "inp must be a CUDA tensor, got ", inp.device());
        disco_kernels::check_kpacked_inputs(inp, pack_idx, pack_val, pack_offset, K, Ho, Wo);
        disco_kernels::check_same_device(plan, inp, "plan");
        const c10::cuda::CUDAGuard guard(inp.device());
        check_device(inp.get_device());
        const cudaDeviceProp &prop = *at::cuda::getDeviceProperties(inp.get_device());
        const int64_t B = inp.size(0), C = inp.size(1), Hi = inp.size(2), Wi = inp.size(3);
        const HkConfig c = make_config(B * C, K, pack_val.size(1), Hi, Wi, Ho, Wo, pack_idx.size(0), prop);
        if (!c.ok || plan.numel() == 0) return kpacked(inp, pack_idx, pack_val, pack_offset, K, Ho, Wo);
        TORCH_CHECK(plan.scalar_type() == at::kByte && plan.dim() == 1 && (size_t)plan.numel() == c.plan_bytes,
                    HK_NAME ": the plan (", plan.numel(), " bytes) was not built for these sizes (", c.plan_bytes,
                    " bytes); build it with " HK_NAME "_plan for B * C = ", B * C);
        // the input rows are read in 16-byte chunks
        TORCH_CHECK(reinterpret_cast<uintptr_t>(inp.data_ptr()) % 16 == 0, HK_NAME ": inp must be 16-byte aligned");

        auto out = at::empty({B, C, K, Ho, Wo}, inp.options());
        auto scratch = at::empty({(int64_t)c.scratch_bytes}, inp.options().dtype(at::kByte));
        cudaStream_t st = at::cuda::getCurrentCUDAStream();

        unsigned char *base = static_cast<unsigned char *>(plan.data_ptr());
        unsigned *stamps = static_cast<unsigned *>(scratch.data_ptr());
        uint4 *xe = c.tx ?
            reinterpret_cast<uint4 *>(static_cast<unsigned char *>(scratch.data_ptr()) + STAMP_WORDS_BYTES) :
            nullptr;
        const unsigned long long *R = c.use_R ? reinterpret_cast<const unsigned long long *>(base + c.o_R) : nullptr;
        const uint4 *x = reinterpret_cast<const uint4 *>(inp.data_ptr());
        const bool bf16 = inp.scalar_type() == at::kBFloat16;

        HkParams prm;
        memset(&prm, 0, sizeof(prm));
        prm.x = reinterpret_cast<const uint16_t *>(inp.data_ptr());
        prm.val = reinterpret_cast<const uint16_t *>(base + c.o_val);
        prm.pieces = reinterpret_cast<const Piece *>(base + c.o_pieces);
        prm.npieces = reinterpret_cast<const int *>(base + c.o_np);
        prm.off = pack_offset.data_ptr<int64_t>();
        prm.T = reinterpret_cast<const uint4 *>(base + c.o_T);
        prm.pxs = reinterpret_cast<const PieceX *>(base + c.o_pxs);
        prm.out = reinterpret_cast<uint16_t *>(out.data_ptr());
        prm.bf16 = bf16 ? 1 : 0;
        prm.BC = c.BC;
        prm.Hi = c.Hi;
        prm.Wi = c.Wi;
        prm.Ho = c.Ho;
        prm.Wo = c.Wo;
        prm.ps = c.ps;
        prm.K = c.K;
        prm.kpad = c.kpad;
        prm.nb = c.nb;
        prm.RPC = c.RPC;
        prm.HB = c.HB;
        prm.G = c.G;
        prm.NCG = c.NCG;
        prm.nitems = c.nitems;
        prm.NG = c.NG;
        prm.ticket = c.ticket;
        prm.ncg_recip = c.ncg_recip;
        prm.nvs = c.nvs;
        prm.nst = c.nst;
        prm.xs_bytes = c.xs_bytes;
        prm.stage_bytes = c.stage_bytes;
        prm.bar_off = c.bar_off;
        prm.z_off = c.z_off;
        prm.ictr = reinterpret_cast<int *>(stamps + ST_ICTR);
        prm.rowsig = reinterpret_cast<const int *>(base + c.o_sig);
        prm.stamps = stamps;
        prm.epoch = 1u;
        prm.co = c.co ? 1 : 0;
        prm.xpre = c.xpre;
        prm.xres = c.xres;
        prm.xe = xe;
        prm.Lc = c.Lc;
        prm.tpl = c.tpl;
        prm.R = R;
        prm.hrot = c.hrot;

        // the call's stamps, claims and item counter
        init_kernel<<<1, INIT_THREADS, 0, st>>>(stamps, STAMP_WORDS, c.xres, c.grid);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // the extended input ahead of the main kernel: every row, or in co-resident mode the first xpre latitudes
        if (c.tx) {
            const long long nxrows = (long long)c.BC * (c.co ? c.xpre : c.Hi);
            const long long nxb = (nxrows + 7) / 8;
            if (nxb > 0) {
                TORCH_CHECK(nxb <= INT32_MAX, HK_NAME ": input too large");
                xpre_kernel<<<(unsigned)nxb, 128, 0, st>>>(x, xe, c.nb, c.Lc, nxrows, c.Hi, c.co ? c.xpre : 0, R, c.hrot);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }

        // the views of the extended copy for the main kernel's TMA loads
        TmapX<NVMAX> tmap;
        memset(&tmap, 0, sizeof(tmap));
        if (c.tx) {
            const int nv = 1 << c.nvs;
            const cuuint64_t gstr[3] = {(cuuint64_t)(16 * nv), (cuuint64_t)c.Lc * 16, (cuuint64_t)c.Hi * c.Lc * 16};
            const cuuint32_t box[4] = {(cuuint32_t)(8 * nv), (cuuint32_t)(c.RPC / nv), 1, (cuuint32_t)c.G};
            const cuuint32_t es[4] = {1, 1, 1, 1};
            const CUtensorMapDataType dt = bf16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
            for (int j = 0; j < nv; ++j) {
                cuuint64_t gdim[4] = {(cuuint64_t)(8 * nv), (cuuint64_t)(c.Lc / nv - (j > 0 ? 1 : 0)), (cuuint64_t)c.Hi,
                                      (cuuint64_t)c.BC};
                const CUresult r = encode_tiled()(&tmap.m[j], dt, 4, xe + j, gdim, gstr, box, es,
                                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
                TORCH_CHECK(r == CUDA_SUCCESS, HK_NAME ": cuTensorMapEncodeTiled failed (", (int)r, ")");
            }
        }

        // the main kernel; a kernel with 8 views reads the first 8 maps (TmapX<8> is a prefix of TmapX<NVMAX>)
        with_family(c, MainLaunch {c, st, &tmap, &prm});

        if (c.co) {
            // the extension blocks must leave each SM configured for the main kernel's shared memory
            C10_CUDA_CHECK(cudaFuncSetAttribute((const void *)xcopy_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                                (int)cudaSharedmemCarveoutMaxShared));
            launch_pdl(true, xcopy_kernel, (unsigned)c.nsm, (unsigned)XC_THREADS, 0, st, x, xe, c.nb, c.Lc, c.Hi, c.BC,
                       c.tpl, c.xpre, stamps, 1u, R, c.hrot, c.xres);
        }
        return out;
    }

} // namespace
