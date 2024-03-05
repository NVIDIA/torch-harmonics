#pragma once

#include "disco.h"

#include <cuda_runtime.h>
#include <ATen/cudnn/Handle.h>  // for getcudnnhandle

#define CHECK_CUDA_TENSOR(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CUDA_INPUT_TENSOR(x) CHECK_CUDA_TENSOR(x); CHECK_CONTIGUOUS_TENSOR(x)
