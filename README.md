torch-svm
=========

SVM packages for Torch7.

So far, there are two packages available. The first one is Leon Bottou's sgd and asgd algorithms.
These algorithms are reimplemented in Torch7. The second package is a wrapper around the LIBLINEAR
library. This package just provides a wrapper around LIBLINEAR code that is already included in
source form.

Requirements
------------

Only Torch7 (http://github.com/andresy/torch)

Building
--------

```
git clone git://github.com/koraykv/torch-svm.git
cd torch-svm
torch-pkg deploy sgd
torch-pkg deploy liblinear
```
