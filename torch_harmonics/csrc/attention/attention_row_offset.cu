#include "ATen/core/TensorAccessor.h"
#include <cmath>
#include <cstdint>
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

__global__ void alpha_count_kernel(int len_alpha_count,
                                   int len_psi_row_idx,
                                   torch::PackedTensorAccessor64<int64_t, 1> psi_row_idx,
                                   int64_t* alpha_start,
                                   torch::PackedTensorAccessor64<int64_t, 1> alpha_count
                                   ) {
  int ho = blockIdx.x * blockDim.x + threadIdx.x;
  if(ho < len_alpha_count) {
    // initialize alpha_count;
    alpha_count[ho] = 0;

    // NOTE: Assumes that psi_row_idx is sorted
    for(int i=alpha_start[ho]; i<len_psi_row_idx; i++) {
      if(psi_row_idx[i] == ho) alpha_count[ho]++;
      else if(psi_row_idx[i] > ho) break;
    }
  }
}

int s2_idx_offset_cuda(const at::Tensor& psi_col_idx,
                       const at::Tensor& psi_row_idx,
                       at::Tensor& row_offset,
                       at::Tensor& row_count) {

  auto stream = at::cuda::getCurrentCUDAStream();

  int64_t* d_alpha_start;
  int64_t* d_sequence;
  int64_t* d_alpha_count = row_count.data_ptr<int64_t>();
  int64_t* d_alpha_offset = row_offset.data_ptr<int64_t>();
  C10_CUDA_CHECK(cudaMalloc(&d_alpha_start, row_offset.size(0)*sizeof(int64_t)));

  // Find the first time each index occurs in psi_row_idx
  // psi_row_idx = [0,0,0,0,1,1,1,1,2,2,2...]
  // 0 starts at idx=0, 1 starts at idx=4, 2 starts at idx=8, etc
  // this assumes that psi_row_idx is sorted!
  C10_CUDA_CHECK(cudaMalloc(&d_sequence, row_offset.size(0)*sizeof(int64_t)));
  thrust::sequence(thrust::device, d_sequence, d_sequence+row_offset.size(0), 0);

  thrust::counting_iterator<int> start(0);
  // thrust::lower_bound(thrust::device,
                      // psi_row_idx.data_ptr<int64_t>(),
                      // psi_row_idx.data_ptr<int64_t>()+psi_row_idx.size(0),
                      // start, start+psi_row_idx.size(0), d_alpha_start);
  thrust::lower_bound(thrust::device,
                      psi_row_idx.data_ptr<int64_t>(),
                      psi_row_idx.data_ptr<int64_t>()+psi_row_idx.size(0),
                      d_sequence, d_sequence+row_offset.size(0), d_alpha_start);

  alpha_count_kernel<<<at::cuda::detail::GET_BLOCKS(row_offset.size(0),512),512,
    0,stream.stream()>>>(row_count.size(0),
                         psi_row_idx.size(0),
                         psi_row_idx.packed_accessor64<int64_t, 1>(),
                         d_alpha_start,
                         row_count.packed_accessor64<int64_t , 1>());

  C10_CUDA_KERNEL_LAUNCH_CHECK();



  int maxAlphaSize = thrust::reduce(thrust::device,
                                    d_alpha_count,
                                    d_alpha_count+row_count.size(0),
                                    0,
                                    thrust::maximum<int>());

  thrust::exclusive_scan(thrust::device,
                         d_alpha_count,
                         d_alpha_count+row_count.size(0),
                         d_alpha_offset);

  C10_CUDA_CHECK(cudaFree(d_alpha_start));
  C10_CUDA_CHECK(cudaFree(d_sequence));

  return maxAlphaSize;

}
