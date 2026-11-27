# Coverage of torch-harmonics

Serial and distributed test suites combined, at 284fcd0cffb3038812bc1f41ae8a956e4f30b851.
Written by the coverage job of tests.yml; do not edit.

| Name                                                                      |    Stmts |     Miss |   Cover |
|-------------------------------------------------------------------------- | -------: | -------: | ------: |
| torch\_harmonics/\_\_init\_\_.py                                          |       14 |        0 |    100% |
| torch\_harmonics/\_backend.py                                             |       82 |       11 |     87% |
| torch\_harmonics/\_field\_contract.py                                     |       33 |        0 |    100% |
| torch\_harmonics/attention/\_\_init\_\_.py                                |        9 |        2 |     78% |
| torch\_harmonics/attention/\_attention\_utils.py                          |       44 |        0 |    100% |
| torch\_harmonics/attention/\_layout.py                                    |       38 |        4 |     89% |
| torch\_harmonics/attention/attention.py                                   |      309 |       11 |     96% |
| torch\_harmonics/attention/backends.py                                    |       63 |        0 |    100% |
| torch\_harmonics/attention/kernels\_torch/\_\_init\_\_.py                 |        0 |        0 |    100% |
| torch\_harmonics/attention/kernels\_torch/attention\_ragged\_torch.py     |      112 |        3 |     97% |
| torch\_harmonics/attention/kernels\_torch/attention\_regular\_torch.py    |      405 |        4 |     99% |
| torch\_harmonics/attention/optimized/\_\_init\_\_.py                      |        0 |        0 |    100% |
| torch\_harmonics/attention/optimized/attention\_optimized.py              |      235 |       48 |     80% |
| torch\_harmonics/cache.py                                                 |      298 |       16 |     95% |
| torch\_harmonics/disco/\_\_init\_\_.py                                    |        9 |        2 |     78% |
| torch\_harmonics/disco/\_disco\_utils.py                                  |       58 |        1 |     98% |
| torch\_harmonics/disco/\_psi\_layouts.py                                  |      116 |        4 |     97% |
| torch\_harmonics/disco/backends.py                                        |      180 |       29 |     84% |
| torch\_harmonics/disco/convolution.py                                     |      345 |       20 |     94% |
| torch\_harmonics/disco/kernels\_torch/\_\_init\_\_.py                     |        0 |        0 |    100% |
| torch\_harmonics/disco/kernels\_torch/disco\_ragged\_torch.py             |       32 |        0 |    100% |
| torch\_harmonics/disco/kernels\_torch/disco\_regular\_torch.py            |       53 |        0 |    100% |
| torch\_harmonics/disco/optimized/\_\_init\_\_.py                          |        0 |        0 |    100% |
| torch\_harmonics/disco/optimized/disco\_arch.py                           |       75 |       26 |     65% |
| torch\_harmonics/disco/optimized/disco\_optimized.py                      |      364 |       24 |     93% |
| torch\_harmonics/disco/optimized/disco\_plan.py                           |      142 |       28 |     80% |
| torch\_harmonics/disco/optimized/disco\_plan\_fft.py                      |       82 |       56 |     32% |
| torch\_harmonics/disco/optimized/disco\_plan\_hankel.py                   |       17 |        9 |     47% |
| torch\_harmonics/distributed/\_\_init\_\_.py                              |        8 |        0 |    100% |
| torch\_harmonics/distributed/\_amp\_utils.py                              |       40 |       19 |     52% |
| torch\_harmonics/distributed/distributed\_attention.py                    |      308 |      135 |     56% |
| torch\_harmonics/distributed/distributed\_convolution.py                  |      167 |        4 |     98% |
| torch\_harmonics/distributed/distributed\_quadrature.py                   |       61 |        3 |     95% |
| torch\_harmonics/distributed/distributed\_resample.py                     |      140 |        8 |     94% |
| torch\_harmonics/distributed/distributed\_sht.py                          |      386 |       11 |     97% |
| torch\_harmonics/distributed/distributed\_spectral\_convolution.py        |       82 |        6 |     93% |
| torch\_harmonics/distributed/kernels/\_\_init\_\_.py                      |        1 |        0 |    100% |
| torch\_harmonics/distributed/kernels/distributed\_convolution\_kernels.py |       61 |        0 |    100% |
| torch\_harmonics/distributed/primitives.py                                |      686 |       53 |     92% |
| torch\_harmonics/distributed/utils.py                                     |       86 |        3 |     97% |
| torch\_harmonics/fft.py                                                   |      177 |        3 |     98% |
| torch\_harmonics/filter\_basis.py                                         |      331 |       21 |     94% |
| torch\_harmonics/grid.py                                                  |      578 |       23 |     96% |
| torch\_harmonics/healpix.py                                               |      221 |        7 |     97% |
| torch\_harmonics/integration.py                                           |       46 |        2 |     96% |
| torch\_harmonics/legendre.py                                              |      114 |        2 |     98% |
| torch\_harmonics/neighborhood.py                                          |      143 |        5 |     97% |
| torch\_harmonics/partition.py                                             |        6 |        0 |    100% |
| torch\_harmonics/quadrature.py                                            |      123 |        4 |     97% |
| torch\_harmonics/radial.py                                                |      209 |       20 |     90% |
| torch\_harmonics/random\_fields.py                                        |       36 |        0 |    100% |
| torch\_harmonics/resample.py                                              |      144 |        3 |     98% |
| torch\_harmonics/sht.py                                                   |      460 |       11 |     98% |
| torch\_harmonics/spectral\_convolution.py                                 |       67 |        6 |     91% |
| torch\_harmonics/truncation.py                                            |       75 |        0 |    100% |
| torch\_harmonics/utils.py                                                 |       62 |       13 |     79% |
| **TOTAL**                                                                 | **7933** |  **660** | **92%** |
