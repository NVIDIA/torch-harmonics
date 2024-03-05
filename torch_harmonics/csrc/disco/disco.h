#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <cassert>

#define CHECK_CONTIGUOUS_TENSOR(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_TENSOR(x) CHECK_CONTIGUOUS_TENSOR(x)
