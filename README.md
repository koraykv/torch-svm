torch-svm
=========

SVM packages for Torch7.

torch-svm/sgd
-------------

Reimplementation of Leon Bottou's svmsgd and svmasgd (http://leon.bottou.org/projects/sgd). 
This implementation is 2-10 times slower depending on the sparsity of the input.

torch-svm/liblinear
-------------------

This is wrapper around the well known LIBLINEAR library (http://www.csie.ntu.edu.tw/~cjlin/liblinear/).

Requirements
------------

Only Torch7 (http://github.com/andresy/torch)

Building
--------

```
git clone git://github.com/koraykv/torch-svm.git
cd torch-svm
torch-pkg deploy
torch-pkg deploy
```

Using
----

```
require 'svm'

d = svm.ascread('liblinear/liblinear/heart_scale')
model = liblinear.train(d)
labels,accuracy,dec = liblinear.predict(d,model)
```
