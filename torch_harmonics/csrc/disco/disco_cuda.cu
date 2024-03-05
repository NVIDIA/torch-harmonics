// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2024 The torch-harmonics Authors. All rights reserved.
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

#include "disco.h"
#include "disco_cuda.cuh"

#define MIN_THREADS (64)
#define ELXTH_MAX   (32)

#define DIV_UP(a,b) (((a)+((b)-1))/(b))

template<int BDIM_X,
         int ELXTH>
__device__ void disco_d(const int Hi,
                        const int Wi,
                        const int K,
                        const int Ho,
                        const int Wo,
                        const int pscale,
                        const int64_t *__restrict__ roff,
                        const int64_t *__restrict__ kers, 
                        const int64_t *__restrict__ rows,
                        const int64_t *__restrict__ cols,
                        const float   *__restrict__ vals, 
                        const float   *__restrict__ inp, 
                              float   *__restrict__ out) {

        const int tid = threadIdx.x;

        const int64_t bidx = blockIdx.x; // gloabl row
        const int64_t bidy = blockIdx.y; // bc

        int64_t soff = roff[bidx];
        int64_t eoff = roff[bidx+1];

        const int64_t ker = kers[soff];
        const int64_t row = rows[soff];

        inp += bidy*Hi*Wi;
        out += bidy*K*Ho*Wo + ker*Ho*Wo + row*Wo;

        float __reg[ELXTH] = {0};

        extern __shared__ float __sh[]; // float __sh[2*Wi + ppscale*(BDIM_X*ELXTH - Wo)]

        int h_prev = cols[soff]/Wi;

        // copy current inp row in shmem
        for(int i = tid; i < Wi; i += BDIM_X) {
                const float v = inp[h_prev*Wi + i];
                __sh[     i] = v;
                __sh[Wi + i] = v;
        }
        // locations __sh[2*Wi : ppscale*(BDIM_X*ELXTH-Wo)] are useless
        __syncthreads();

        // loops along the colums of CTA's row
        for(int64_t nz = soff; nz < eoff; nz++) {

                const int   col = cols[nz];
                const float val = vals[nz];

                const int h = col / Wi;        
                const int w = col % Wi;        

                // if we are processing a nz with a col value
                // leading to a new row of inp then copy it
                // to shmem
                if (h_prev != h) {
                        __syncthreads();
                        for(int i = tid; i < Wi; i += BDIM_X) {
                                const float v = inp[h*Wi + i];
                                __sh[     i] = v;
                                __sh[Wi + i] = v;
                        }
                        __syncthreads();

                        h_prev = h;
                }

                #pragma unroll
                for (int i = 0; i < ELXTH; i++) {

                        const int pp = i*BDIM_X + tid;

                        // original lines:
                        //
                        //   if (pp >= Wo) break;
                        //   const int wpp = (w + pscale*pp) % Wi;
                        //
                        // value of (w + pscale*pp) < (Wi + (Wi/Wo)*Wo) = 2*Wi
                        // so we can allocate twice the amount of shmem, 
                        // replicate the current inp row and avoid the costly mod
                        //
                        // also, to avoid the conditional, sh can be extended to 
                        // cover the maximum location accessed during this loop
                        //
                        // float __sh[2*Wi + ppscale*NUM_REM]
                        //
                        //   Wi + (Wi/Wo)*BDIM_X*ELXTH = (since BDIM_X*ELXTH >= Wo) =
                        // = Wi + (Wi/Wo)*(Wo + (BDIM_X*ELXTH - Wo)) =
                        // = 2*Wi + ppscale*NUM_REM
                        //
                        // with NUM_REM = BDIM_X*ELXTH - Wo

                        const int wpp = w + pscale*pp;

                        __reg[i] += val*__sh[wpp];

                }
        }

        #pragma unroll
        for (int i = 0; i < ELXTH; i++) {

                const int pp = i*BDIM_X + tid;
                if (pp >= Wo) break;

                out[pp] = __reg[i];
        }

        return;
}

template<int BDIM_X,
         int ELXTH,
         int WI>
__global__ __launch_bounds__(BDIM_X)
void disco_reg_blk_k(const int Hi,
                     const int Wi,
                     const int K,
                     const int Ho,
                     const int Wo,
                     const int pscale,
                     const int64_t *__restrict__ roff,
                     const int64_t *__restrict__ kers, 
                     const int64_t *__restrict__ rows,
                     const int64_t *__restrict__ cols,
                     const float   *__restrict__ vals, 
                     const float   *__restrict__ inp, 
                           float   *__restrict__ out) {

        if constexpr(WI != 0) { disco_d<BDIM_X, ELXTH>(Hi, WI, K, Ho, Wo, pscale, roff, kers, rows, cols, vals, inp, out); } 
        else                  { disco_d<BDIM_X, ELXTH>(Hi, Wi, K, Ho, Wo, pscale, roff, kers, rows, cols, vals, inp, out); }

        return;
}

template<int NTH,
         int ELXTH>
static void launch_kernel(int BC,
                          int Hi,
                          int Wi,
                          int K,
                          int Ho,
                          int Wo,
                          int64_t nrows,
                          int64_t *roff_d,
                          int64_t *ker_d, 
                          int64_t *row_d,
                          int64_t *col_d,
                          float   *val_d, 
                          float   *inp_d, 
                          float   *out_d,
                          cudaStream_t stream) {

  if constexpr(ELXTH <= ELXTH_MAX) {
      if (NTH*ELXTH >= Wo) {
	dim3 grid(nrows, BC);
        
	const int pscale = Wi/Wo;
	size_t shmem = sizeof(*out_d)*(Wi*2 + pscale*(NTH*ELXTH-Wo));
	
	switch(Wi) {
	case 360:
	  disco_reg_blk_k<NTH, ELXTH, 360><<<grid, NTH, shmem, stream>>>(Hi, Wi,
									 K, Ho, Wo, pscale,
									 roff_d,
									 ker_d, row_d, col_d, val_d,
									 inp_d, out_d);
	  break;
	case 720:
	  disco_reg_blk_k<NTH, ELXTH, 720><<<grid, NTH, shmem, stream>>>(Hi, Wi,
									 K, Ho, Wo, pscale,
									 roff_d,
									 ker_d, row_d, col_d, val_d,
									 inp_d, out_d);
	  break;
	case 1440:
	  disco_reg_blk_k<NTH, ELXTH, 1440><<<grid, NTH, shmem, stream>>>(Hi, Wi,
									  K, Ho, Wo, pscale,
									  roff_d,
									  ker_d, row_d, col_d, val_d,
									  inp_d, out_d);
	  break;
	default:
	  disco_reg_blk_k<NTH, ELXTH, 0><<<grid, NTH, shmem, stream>>>(Hi, Wi,
								       K, Ho, Wo, pscale,
								       roff_d,
								       ker_d, row_d, col_d, val_d,
								       inp_d, out_d);
	}
      } else {
	launch_kernel<NTH, ELXTH+1>(BC,
				    Hi, Wi, 
				    K, Ho, Wo,
				    nrows,
				    roff_d, 
				    ker_d, row_d, col_d, val_d,
				    inp_d, out_d,
				    stream);
      }
    }
  return;
}

torch::Tensor disco_cuda_fwd(torch::Tensor inp,
			     torch::Tensor roff_idx,
			     torch::Tensor ker_idx,
			     torch::Tensor row_idx,
			     torch::Tensor col_idx,
			     torch::Tensor val,
			     int64_t K,
			     int64_t Ho,
			     int64_t Wo) {
  
  // some sanity checks
  CHECK_CUDA_INPUT_TENSOR(inp);
  CHECK_CUDA_INPUT_TENSOR(roff_idx);
  CHECK_CUDA_INPUT_TENSOR(ker_idx);
  CHECK_CUDA_INPUT_TENSOR(row_idx);
  CHECK_CUDA_INPUT_TENSOR(col_idx);
  CHECK_CUDA_INPUT_TENSOR(val);

  // extract some shapes
  int64_t B = inp.size(0);
  int64_t C = inp.size(1);
  int64_t BC = B * C;
  int64_t Hi = inp.size(2);
  int64_t Wi = inp.size(3);
  int64_t nrows = roff_idx.size(0) - 1;

  // allocate output
  int64_t out_dims[] = {B, C, K, Ho, Wo};
  auto options = torch::TensorOptions().device(inp.device()).dtype(inp.dtype());
  torch::Tensor out = torch::zeros(out_dims, options);

  // get handle
  cudnnHandle_t handle_ = torch::native::getCudnnHandle();
  
  // get stream
  cudaStream_t stream;
  cudnnGetStream(handle_, &stream);
  
  // assert
  static_assert(0 == (ELXTH_MAX%2));

  // extract data pointers
  int64_t* roff_d = roff_idx.data_ptr<int64_t>();
  int64_t* ker_d = ker_idx.data_ptr<int64_t>();
  int64_t* row_d = row_idx.data_ptr<int64_t>();
  int64_t* col_d = col_idx.data_ptr<int64_t>();
  float* val_d = val.data_ptr<float>();
  float* inp_d = inp.data_ptr<float>();
  float* out_d = out.data_ptr<float>();
  
  // this if-chain can be moved into launch_kernel template but I think it is clearer like this
  if      (Wo <=   64*ELXTH_MAX) { launch_kernel<  64,               1>(BC, Hi, Wi, K, Ho, Wo, nrows, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d, stream); }
  else if (Wo <=  128*ELXTH_MAX) { launch_kernel< 128, (ELXTH_MAX/2)+1>(BC, Hi, Wi, K, Ho, Wo, nrows, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d, stream); }
  else if (Wo <=  256*ELXTH_MAX) { launch_kernel< 256, (ELXTH_MAX/2)+1>(BC, Hi, Wi, K, Ho, Wo, nrows, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d, stream); }
  else if (Wo <=  512*ELXTH_MAX) { launch_kernel< 512, (ELXTH_MAX/2)+1>(BC, Hi, Wi, K, Ho, Wo, nrows, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d, stream); }
  else if (Wo <= 1024*ELXTH_MAX) { launch_kernel<1024, (ELXTH_MAX/2)+1>(BC, Hi, Wi, K, Ho, Wo, nrows, roff_d, ker_d, row_d, col_d, val_d, inp_d, out_d, stream); }
  else {
    fprintf(stderr,
	    "%s:%d: error, unsupported Wo value (%ld), max supported is %ld\n",
	    __FILE__, __LINE__, Wo, static_cast<int64_t>(1024*ELXTH_MAX));
    exit(EXIT_FAILURE);
  }
  return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &disco_cuda_fwd, "DISCO forward (CUDA)");
  //m.def("backward", &disco_cuda_bwd, "DISCO backward (CUDA)");
}

