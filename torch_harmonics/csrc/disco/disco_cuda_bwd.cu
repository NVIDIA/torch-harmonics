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


template<int BDIM_X,
         int ELXTH,
         typename REAL_T>
__device__ void disco_trans_d(const int Hi,
			      const int Wi,
			      const int K,
			      const int Ho,
			      const int Wo,
			      const int pscale,
			      const int64_t *__restrict__ roff,
			      const int64_t *__restrict__ kers,
			      const int64_t *__restrict__ rows,
			      const int64_t *__restrict__ cols,
			      const REAL_T   *__restrict__ vals,
			      const REAL_T   *__restrict__ inp,
			      REAL_T   *__restrict__ out) {
  
  const int tid = threadIdx.x;
  
  const int64_t bidx = blockIdx.x; // gloabl row
  const int64_t bidy = blockIdx.y; // bc
  
  int64_t soff = roff[bidx];
  int64_t eoff = roff[bidx+1];
  
  const int64_t ker = kers[soff];
  const int64_t row = rows[soff];
  
  inp += bidy*K*Hi*Wi + ker*Hi*Wi + row*Wi;
  out += bidy*Ho*Wo;
  
  // align to larger supported fp type
  extern __shared__ __align__(sizeof(double)) unsigned char __sh_ptr[]; // REAL_T __sh[2*(BDIM_X*ELXTH)*pscale]
  
  REAL_T (*__sh)[BDIM_X*ELXTH*2] = reinterpret_cast<REAL_T (*)[BDIM_X*ELXTH*2]>(__sh_ptr);
  
  // copy current inp row in regs
  REAL_T __reg[ELXTH];
 
  #pragma unroll
  for(int i = 0; i < ELXTH; i++) {
    __reg[i] = (i*BDIM_X+tid < Wi) ? inp[i*BDIM_X+tid] : 0;
  }
  
  // reset shared row up to Wo+2, remaining
  // ppscale*(BDIM_X*ELXTH - Wo) locations
  // will be written to but never copied to
  // global mem
  for(int i = 0; i < pscale; i++) {
    #pragma unroll
    for(int j = 0; j < 2*BDIM_X*ELXTH; j += BDIM_X) {
      __sh[i][j+tid] = 0;
    }
  }
  __syncthreads();
  
  int h_prev = cols[soff]/Wo;
  
  // loops along the colums of CTA's row
  for(int64_t nz = soff; nz < eoff; nz++) {
    
    const int    col = cols[nz];
    const REAL_T val = vals[nz];
    
    const int h = col / Wo;
    const int w = col % Wo;
    
    // if we are processing a nz with a col value
    // leading to a new row of inp then copy it
    // to shmem
    if (h_prev != h) {
      __syncthreads();
      for(int i = 0; i < pscale; i++) {
	for(int j = tid; j < Wi; j += BDIM_X) {
	  
	  const REAL_T v = __sh[i][j] + __sh[i][Wi + j];
	  
	  atomicAdd(&out[h_prev*Wo + j*pscale + i], v);
	  
	  __sh[i][     j] = 0;
	  __sh[i][Wi + j] = 0;
                                }
      }
      __syncthreads();
      
      h_prev = h;
    }
    
    const int w_mod_ps = w % pscale;
    const int w_div_ps = w / pscale;
    
    #pragma unroll
    for (int i = 0; i < ELXTH; i++) {
      
      const int pp = i*BDIM_X + tid;
      __sh[w_mod_ps][w_div_ps + pp] += val*__reg[i];
    }
    
    // to avoid race conditions on __sh[]
    // among consecutive iterations along nz
    __syncthreads();
  }
  __syncthreads();
  
  // write last row
  for(int i = 0; i < pscale; i++) {
    
    for(int j = tid; j < Wi; j += BDIM_X) {
      
      const REAL_T v = __sh[i][j] + __sh[i][Wi + j];
      atomicAdd(&out[h_prev*Wo + j*pscale + i], v);
    }
  }
  return;
}


template<int BDIM_X,
         int ELXTH,
         int WO,
         typename REAL_T>
__global__ __launch_bounds__(BDIM_X)
void disco_trans_blk_k(const int Hi,
		       const int Wi,
		       const int K,
		       const int Ho,
		       const int Wo,
		       const int pscale,
		       const int64_t *__restrict__ roff,
		       const int64_t *__restrict__ kers,
		       const int64_t *__restrict__ rows,
		       const int64_t *__restrict__ cols,
		       const REAL_T   *__restrict__ vals,
		       const REAL_T   *__restrict__ inp,
		       REAL_T   *__restrict__ out) {
  
  if constexpr(WO != 0) {
      switch(pscale) {
      case  1: disco_trans_d<BDIM_X, ELXTH>(Hi, Wi, K, Ho, WO,      1, roff, kers, rows, cols, vals, inp, out); break;
      case  2: disco_trans_d<BDIM_X, ELXTH>(Hi, Wi, K, Ho, WO,      2, roff, kers, rows, cols, vals, inp, out); break;
      case  3: disco_trans_d<BDIM_X, ELXTH>(Hi, Wi, K, Ho, WO,      3, roff, kers, rows, cols, vals, inp, out); break;
      default: disco_trans_d<BDIM_X, ELXTH>(Hi, Wi, K, Ho, WO, pscale, roff, kers, rows, cols, vals, inp, out); break;
      }
    } else {
    switch(pscale) {
    case  1: disco_trans_d<BDIM_X, ELXTH>(Hi, Wi, K, Ho, Wo,      1, roff, kers, rows, cols, vals, inp, out); break;
    case  2: disco_trans_d<BDIM_X, ELXTH>(Hi, Wi, K, Ho, Wo,      2, roff, kers, rows, cols, vals, inp, out); break;
    case  3: disco_trans_d<BDIM_X, ELXTH>(Hi, Wi, K, Ho, Wo,      3, roff, kers, rows, cols, vals, inp, out); break;
    default: disco_trans_d<BDIM_X, ELXTH>(Hi, Wi, K, Ho, Wo, pscale, roff, kers, rows, cols, vals, inp, out); break;
    }
  }
  
  return;
}


template<int NTH,
         int ELXTH,
         typename REAL_T>
static void launch_kernel(int64_t BC,
                          int64_t Hi,
                          int64_t Wi,
                          int64_t K,
                          int64_t Ho,
                          int64_t Wo,
                          int64_t nrows,
                          int64_t *roff_d,
                          int64_t *ker_d,
                          int64_t *row_d,
                          int64_t *col_d,
                          REAL_T  *val_d,
                          REAL_T  *inp_d,
                          REAL_T  *out_d,
                          cudaStream_t stream) {
  
  if constexpr(ELXTH <= ELXTH_MAX) {
      if (NTH*ELXTH >= Wi) {
	dim3 grid(nrows, BC);
	
	const int pscale = static_cast<int>(Wo/Wi);
	size_t shmem = sizeof(*out_d)*(2 * (NTH*ELXTH)*pscale);
	
	switch(Wo) {
	case 360:
	  disco_trans_blk_k<NTH, ELXTH, 360><<<grid, NTH, shmem, stream>>>(static_cast<int>(Hi),
									   static_cast<int>(Wi),
									   static_cast<int>(K),
									   static_cast<int>(Ho),
									   static_cast<int>(Wo),
									   pscale,
									   roff_d,
									   ker_d, row_d, col_d, val_d,
									   inp_d, out_d);
	  break;
	case 720:
	  disco_trans_blk_k<NTH, ELXTH, 720><<<grid, NTH, shmem, stream>>>(static_cast<int>(Hi),
									   static_cast<int>(Wi),
									   static_cast<int>(K),
									   static_cast<int>(Ho),
									   static_cast<int>(Wo),
									   pscale,
									   roff_d,
									   ker_d, row_d, col_d, val_d,
									   inp_d, out_d);
	  break;
	case 1440:
	  disco_trans_blk_k<NTH, ELXTH, 1440><<<grid, NTH, shmem, stream>>>(static_cast<int>(Hi),
									    static_cast<int>(Wi),
									    static_cast<int>(K),
									    static_cast<int>(Ho),
									    static_cast<int>(Wo),
									    pscale,
									    roff_d,
									    ker_d, row_d, col_d, val_d,
									    inp_d, out_d);
	  break;
	default:
	  disco_trans_blk_k<NTH, ELXTH, 0><<<grid, NTH, shmem, stream>>>(static_cast<int>(Hi),
									 static_cast<int>(Wi),
									 static_cast<int>(K),
									 static_cast<int>(Ho),
									 static_cast<int>(Wo),
									 pscale,
									 roff_d,
									 ker_d, row_d, col_d, val_d,
									 inp_d, out_d);
	}
      } else {
	launch_kernel<NTH, ELXTH+1>(BC,
				    Hi, Wi,
				    K,
				    Ho, Wo,
				    nrows,
				    roff_d,
				    ker_d, row_d, col_d, val_d,
				    inp_d, out_d,
				    stream);
      }
    }
  return;
}



torch::Tensor disco_cuda_bwd(torch::Tensor inp,
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
  int64_t Hi = inp.size(3);
  int64_t Wi = inp.size(4);
  int64_t nrows = roff_idx.size(0) - 1;

  // allocate output
  int64_t out_dims[] = {B, C, Ho, Wo};
  auto options = torch::TensorOptions().device(inp.device()).dtype(inp.dtype());
  torch::Tensor out = torch::zeros(out_dims, options);

  // get handle
  cudnnHandle_t handle_ = torch::native::getCudnnHandle();
  
  // get stream
  cudaStream_t stream;
  cudnnGetStream(handle_, &stream);
  
  // assert
  static_assert(0 == (ELXTH_MAX%2));


  if      (Wo <=   64*ELXTH_MAX) {
    AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_cuda", ([&] {
	  launch_kernel<64, 1, scalar_t>(BC, Hi, Wi, K, Ho, Wo, nrows,
					 roff_idx.data_ptr<int64_t>(),
					 ker_idx.data_ptr<int64_t>(),
					 row_idx.data_ptr<int64_t>(),
					 col_idx.data_ptr<int64_t>(),
					 val.data_ptr<scalar_t>(),
					 inp.data_ptr<scalar_t>(),
					 out.data_ptr<scalar_t>(),
					 stream);
	    }));
  }
  else if (Wo <=  128*ELXTH_MAX) {
    AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_cuda", ([&] {
	  launch_kernel<128, (ELXTH_MAX/2)+1, scalar_t>(BC, Hi, Wi, K, Ho, Wo, nrows,
							roff_idx.data_ptr<int64_t>(),
							ker_idx.data_ptr<int64_t>(),
							row_idx.data_ptr<int64_t>(),
							col_idx.data_ptr<int64_t>(),
							val.data_ptr<scalar_t>(),
							inp.data_ptr<scalar_t>(),
							out.data_ptr<scalar_t>(),
							stream);
	    }));
  }
  else if (Wo <=  256*ELXTH_MAX) {
    AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_cuda", ([&] {
          launch_kernel<256, (ELXTH_MAX/2)+1, scalar_t>(BC, Hi, Wi, K, Ho, Wo, nrows,
                                                        roff_idx.data_ptr<int64_t>(),
                                                        ker_idx.data_ptr<int64_t>(),
                                                        row_idx.data_ptr<int64_t>(),
                                                        col_idx.data_ptr<int64_t>(),
                                                        val.data_ptr<scalar_t>(),
                                                        inp.data_ptr<scalar_t>(),
                                                        out.data_ptr<scalar_t>(),
                                                        stream);
	    }));
  }
  else if (Wo <=  512*ELXTH_MAX) {
    AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_cuda", ([&] {
          launch_kernel<512, (ELXTH_MAX/2)+1, scalar_t>(BC, Hi, Wi, K, Ho, Wo, nrows,
                                                        roff_idx.data_ptr<int64_t>(),
                                                        ker_idx.data_ptr<int64_t>(),
                                                        row_idx.data_ptr<int64_t>(),
                                                        col_idx.data_ptr<int64_t>(),
                                                        val.data_ptr<scalar_t>(),
                                                        inp.data_ptr<scalar_t>(),
                                                        out.data_ptr<scalar_t>(),
                                                        stream);
            }));
  }
  else if (Wo <= 1024*ELXTH_MAX) {
    AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "disco_backward_cuda", ([&] {
          launch_kernel<1024, (ELXTH_MAX/2)+1, scalar_t>(BC, Hi, Wi, K, Ho, Wo, nrows,
							 roff_idx.data_ptr<int64_t>(),
							 ker_idx.data_ptr<int64_t>(),
							 row_idx.data_ptr<int64_t>(),
							 col_idx.data_ptr<int64_t>(),
							 val.data_ptr<scalar_t>(),
							 inp.data_ptr<scalar_t>(),
							 out.data_ptr<scalar_t>(),
							 stream);
            }));
  }
  else {
    fprintf(stderr,
            "%s:%d: error, unsupported Wo value (%ld), max supported is %d\n",
            __FILE__, __LINE__, Wo, 1024*ELXTH_MAX);
    exit(EXIT_FAILURE);
  }
  
  
  return out;
}

//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def("backward", &disco_cuda_bwd, "DISCO backward (CUDA)");
//}
