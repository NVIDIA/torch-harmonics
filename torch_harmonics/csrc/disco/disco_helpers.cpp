#include "disco.h"

torch::Tensor preprocess_psi(const int64_t K,
			     const int64_t Ho,
			     torch::Tensor ker_idx, 
			     torch::Tensor row_idx,
			     torch::Tensor col_idx,
			     torch::Tensor val) {
  
  CHECK_INPUT_TENSOR(ker_idx);
  CHECK_INPUT_TENSOR(row_idx);
  CHECK_INPUT_TENSOR(col_idx);
  CHECK_INPUT_TENSOR(val);
  
  int64_t nnz = val.size(0);
  int64_t *ker_h = ker_idx.data_ptr<int64_t>();
  int64_t *row_h = row_idx.data_ptr<int64_t>();
  int64_t *col_h = col_idx.data_ptr<int64_t>();
  float *val_h = val.data_ptr<float>();
  int64_t *roff_h = new int64_t[Ho*K+1];
  
  int64_t *Koff = new int64_t[K];
  for(int i = 0; i < K; i++) {
    Koff[i] = 0;
  }
  
  for(int64_t i = 0; i < nnz; i++) {
    Koff[ker_h[i]]++;
  }

  int64_t prev = Koff[0];
  Koff[0] = 0;
  for(int i = 1; i < K; i++) {
    int64_t save = Koff[i];
    Koff[i] = prev + Koff[i-1];
    prev = save;
  }

  int64_t *ker_sort = new int64_t[nnz];
  int64_t *row_sort = new int64_t[nnz];
  int64_t *col_sort = new int64_t[nnz];
  float   *val_sort = new   float[nnz];
  
  for(int64_t i = 0; i < nnz; i++) {
    
    const int64_t ker = ker_h[i];
    const int64_t off = Koff[ker]++;
    
    ker_sort[off] = ker;
    row_sort[off] = row_h[i];
    col_sort[off] = col_h[i];
    val_sort[off] = val_h[i];
  }
  for(int64_t i = 0; i < nnz; i++) {
    ker_h[i] = ker_sort[i];
    row_h[i] = row_sort[i];
    col_h[i] = col_sort[i];
    val_h[i] = val_sort[i];
  }
  delete [] Koff;
  delete [] ker_sort;
  delete [] row_sort;
  delete [] col_sort;
  delete [] val_sort;
  
  // compute rows offsets
  int64_t nrows = 1;
  roff_h[0] = 0;
  for(int64_t i = 1; i < nnz; i++) {
    
    if (row_h[i-1] == row_h[i]) continue;
    roff_h[nrows++] = i;
    
    if (nrows > int64_t(Ho)*K) {
      fprintf(stderr,
	      "%s:%d: error, found more rows in the K COOs than Ho*K (%ld)\n",
	      __FILE__, __LINE__, int64_t(Ho)*K);
      exit(EXIT_FAILURE);
    }
  }
  roff_h[nrows] = nnz;

  // create output tensor
  auto options = torch::TensorOptions().dtype(row_idx.dtype());
  auto roff_idx = torch::empty({nrows+1}, options);
  int64_t *roff_out_h = roff_idx.data_ptr<int64_t>();
  
  for(int64_t i = 0; i < (nrows+1); i++) {
    roff_out_h[i] = roff_h[i];
  }
  delete [] roff_h;
  
  return roff_idx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_psi", &preprocess_psi, "Sort psi matrix, required for using disco_cuda.");
}
