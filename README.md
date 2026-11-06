# Coverage of torch-harmonics

Serial and distributed test suites combined, at 084f5915843eaea345811d8f689dc00bfc452584.
Written by the coverage job of tests.yml; do not edit.

| Name                                                                      |    Stmts |     Miss |   Cover |
|-------------------------------------------------------------------------- | -------: | -------: | ------: |
| torch\_harmonics/\_\_init\_\_.py                                          |       14 |        0 |    100% |
| torch\_harmonics/\_backend.py                                             |       77 |       11 |     86% |
| torch\_harmonics/\_field\_contract.py                                     |       33 |        0 |    100% |
| torch\_harmonics/attention/\_\_init\_\_.py                                |        9 |        2 |     78% |
| torch\_harmonics/attention/\_attention\_utils.py                          |       44 |        1 |     98% |
| torch\_harmonics/attention/\_layout.py                                    |       38 |        4 |     89% |
| torch\_harmonics/attention/attention.py                                   |      301 |       11 |     96% |
| torch\_harmonics/attention/backends.py                                    |       64 |        0 |    100% |
| torch\_harmonics/attention/kernels\_torch/\_\_init\_\_.py                 |        0 |        0 |    100% |
| torch\_harmonics/attention/kernels\_torch/attention\_ragged\_torch.py     |      112 |        3 |     97% |
| torch\_harmonics/attention/kernels\_torch/attention\_regular\_torch.py    |      405 |        4 |     99% |
| torch\_harmonics/attention/optimized/\_\_init\_\_.py                      |        0 |        0 |    100% |
| torch\_harmonics/attention/optimized/attention\_optimized.py              |      235 |       48 |     80% |
| torch\_harmonics/cache.py                                                 |      298 |       16 |     95% |
| torch\_harmonics/disco/\_\_init\_\_.py                                    |        9 |        2 |     78% |
| torch\_harmonics/disco/\_disco\_utils.py                                  |       28 |        0 |    100% |
| torch\_harmonics/disco/\_psi\_layouts.py                                  |      116 |        4 |     97% |
| torch\_harmonics/disco/backends.py                                        |      136 |       17 |     88% |
| torch\_harmonics/disco/convolution.py                                     |      338 |       20 |     94% |
| torch\_harmonics/disco/kernels\_torch/\_\_init\_\_.py                     |        0 |        0 |    100% |
| torch\_harmonics/disco/kernels\_torch/disco\_ragged\_torch.py             |       32 |        0 |    100% |
| torch\_harmonics/disco/kernels\_torch/disco\_regular\_torch.py            |       53 |        0 |    100% |
| torch\_harmonics/disco/optimized/\_\_init\_\_.py                          |        0 |        0 |    100% |
| torch\_harmonics/disco/optimized/disco\_optimized.py                      |      380 |       35 |     91% |
| torch\_harmonics/distributed/\_\_init\_\_.py                              |        8 |        0 |    100% |
| torch\_harmonics/distributed/\_amp\_utils.py                              |       40 |       21 |     48% |
| torch\_harmonics/distributed/distributed\_attention.py                    |      447 |      298 |     33% |
| torch\_harmonics/distributed/distributed\_convolution.py                  |      184 |        4 |     98% |
| torch\_harmonics/distributed/distributed\_quadrature.py                   |       59 |        3 |     95% |
| torch\_harmonics/distributed/distributed\_resample.py                     |      120 |        8 |     93% |
| torch\_harmonics/distributed/distributed\_sht.py                          |      305 |       12 |     96% |
| torch\_harmonics/distributed/distributed\_spectral\_convolution.py        |       86 |        6 |     93% |
| torch\_harmonics/distributed/kernels/\_\_init\_\_.py                      |        1 |        0 |    100% |
| torch\_harmonics/distributed/kernels/distributed\_convolution\_kernels.py |       63 |        0 |    100% |
| torch\_harmonics/distributed/primitives.py                                |      593 |       63 |     89% |
| torch\_harmonics/distributed/utils.py                                     |       58 |        3 |     95% |
| torch\_harmonics/fft.py                                                   |      136 |        2 |     99% |
| torch\_harmonics/filter\_basis.py                                         |      318 |       20 |     94% |
| torch\_harmonics/grid.py                                                  |      578 |       25 |     96% |
| torch\_harmonics/healpix.py                                               |      138 |        4 |     97% |
| torch\_harmonics/integration.py                                           |       46 |        2 |     96% |
| torch\_harmonics/legendre.py                                              |      114 |        2 |     98% |
| torch\_harmonics/neighborhood.py                                          |      143 |        5 |     97% |
| torch\_harmonics/partition.py                                             |        6 |        0 |    100% |
| torch\_harmonics/quadrature.py                                            |      123 |        4 |     97% |
| torch\_harmonics/radial.py                                                |      209 |       20 |     90% |
| torch\_harmonics/random\_fields.py                                        |       36 |        0 |    100% |
| torch\_harmonics/resample.py                                              |       80 |        3 |     96% |
| torch\_harmonics/sht.py                                                   |      361 |       10 |     97% |
| torch\_harmonics/spectral\_convolution.py                                 |       72 |        6 |     92% |
| torch\_harmonics/truncation.py                                            |       64 |        0 |    100% |
| torch\_harmonics/utils.py                                                 |       62 |       13 |     79% |
| **TOTAL**                                                                 | **7172** |  **712** | **90%** |
