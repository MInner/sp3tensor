sp3tensor
=========

Sparse Rank-3 Tensors for Python

Here's an implementation of Sparce Rank-3 Tensors for Python originaly written as a part of Numerical Linear Aglebra course project at Skoltech.

== Requirements: ==

- numpy
- scipy

== Usage: ==

import sp3tensor from sp3tensor

To run tests, run sp3tensor.py

See usage in simrank_example.py (SimRank on tensors). 

For more details see *.ipynb file (IPython Notebook)

== Details: ==

The internal storage is: a list of indexes ([0 11 545 32 2325]) and list of values ([34 53 26 23 777]) + function to map from internal index spase into tensor space. To perform any kind of convolution (tensor-tensor or matrix-tensor), convert tensor into one of three *-mode unfoldings and use matrix dot product. The 'unfold-*' functions return csc-matrixes that can be easily multiplied by anything from numpy.

**Note**: in alpha still, bugs and sudden interface changes are possible.
=======
For more details see *.ipynb file (IPython Notebook)
