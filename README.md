# Coverage of torch-harmonics

Serial and distributed test suites combined, at 97f7d2756d2d2d3b974d8ee31ca04e032fdf1d05.
Written by the coverage job of tests.yml; do not edit.

| Name                                                                      |    Stmts |     Miss |   Cover |
|-------------------------------------------------------------------------- | -------: | -------: | ------: |
| torch\_harmonics/\_\_init\_\_.py                                          |       13 |        0 |    100% |
| torch\_harmonics/\_backend.py                                             |       45 |        4 |     91% |
| torch\_harmonics/attention/\_\_init\_\_.py                                |        9 |        2 |     78% |
| torch\_harmonics/attention/\_attention\_utils.py                          |       40 |        2 |     95% |
| torch\_harmonics/attention/\_layout.py                                    |       38 |        4 |     89% |
| torch\_harmonics/attention/attention.py                                   |      288 |        9 |     97% |
| torch\_harmonics/attention/backends.py                                    |       65 |        0 |    100% |
| torch\_harmonics/attention/kernels\_torch/\_\_init\_\_.py                 |        0 |        0 |    100% |
| torch\_harmonics/attention/kernels\_torch/attention\_ragged\_torch.py     |      112 |        3 |     97% |
| torch\_harmonics/attention/kernels\_torch/attention\_regular\_torch.py    |      405 |        4 |     99% |
| torch\_harmonics/attention/optimized/\_\_init\_\_.py                      |        0 |        0 |    100% |
| torch\_harmonics/attention/optimized/attention\_optimized.py              |      235 |       48 |     80% |
| torch\_harmonics/cache.py                                                 |       16 |        0 |    100% |
| torch\_harmonics/disco/\_\_init\_\_.py                                    |        9 |        2 |     78% |
| torch\_harmonics/disco/\_disco\_utils.py                                  |       19 |        0 |    100% |
| torch\_harmonics/disco/\_psi\_layouts.py                                  |       99 |        3 |     97% |
| torch\_harmonics/disco/backends.py                                        |       91 |       16 |     82% |
| torch\_harmonics/disco/convolution.py                                     |      231 |       15 |     94% |
| torch\_harmonics/disco/kernels\_torch/\_\_init\_\_.py                     |        0 |        0 |    100% |
| torch\_harmonics/disco/kernels\_torch/disco\_torch.py                     |       53 |        0 |    100% |
| torch\_harmonics/disco/optimized/\_\_init\_\_.py                          |        0 |        0 |    100% |
| torch\_harmonics/disco/optimized/disco\_optimized.py                      |      242 |       40 |     83% |
| torch\_harmonics/distributed/\_\_init\_\_.py                              |        8 |        0 |    100% |
| torch\_harmonics/distributed/\_amp\_utils.py                              |       40 |       21 |     48% |
| torch\_harmonics/distributed/distributed\_attention.py                    |      447 |      298 |     33% |
| torch\_harmonics/distributed/distributed\_convolution.py                  |      157 |        2 |     99% |
| torch\_harmonics/distributed/distributed\_quadrature.py                   |       33 |        1 |     97% |
| torch\_harmonics/distributed/distributed\_resample.py                     |      108 |        7 |     94% |
| torch\_harmonics/distributed/distributed\_sht.py                          |      247 |        4 |     98% |
| torch\_harmonics/distributed/distributed\_spectral\_convolution.py        |       77 |        5 |     94% |
| torch\_harmonics/distributed/kernels/\_\_init\_\_.py                      |        1 |        0 |    100% |
| torch\_harmonics/distributed/kernels/distributed\_convolution\_kernels.py |       63 |        0 |    100% |
| torch\_harmonics/distributed/primitives.py                                |      589 |       63 |     89% |
| torch\_harmonics/distributed/utils.py                                     |       58 |        3 |     95% |
| torch\_harmonics/fft.py                                                   |       34 |        2 |     94% |
| torch\_harmonics/filter\_basis.py                                         |      303 |       25 |     92% |
| torch\_harmonics/grid.py                                                  |      530 |       28 |     95% |
| torch\_harmonics/healpix.py                                               |      103 |        3 |     97% |
| torch\_harmonics/integration.py                                           |       21 |        1 |     95% |
| torch\_harmonics/legendre.py                                              |      101 |        4 |     96% |
| torch\_harmonics/neighborhood.py                                          |      143 |        5 |     97% |
| torch\_harmonics/partition.py                                             |        6 |        0 |    100% |
| torch\_harmonics/quadrature.py                                            |      103 |        3 |     97% |
| torch\_harmonics/random\_fields.py                                        |       36 |        0 |    100% |
| torch\_harmonics/resample.py                                              |       75 |        2 |     97% |
| torch\_harmonics/sht.py                                                   |      138 |        4 |     97% |
| torch\_harmonics/spectral\_convolution.py                                 |       63 |        5 |     92% |
| torch\_harmonics/truncation.py                                            |       44 |        0 |    100% |
| torch\_harmonics/utils.py                                                 |       55 |       13 |     76% |
| **TOTAL**                                                                 | **5593** |  **651** | **88%** |
