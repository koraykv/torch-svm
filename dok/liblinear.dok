====== LIBLINEAR Interface =======
{{anchor:svm.liblinear.dok}}

This package provides an interface for the well known [[http://www.csie.ntu.edu.tw/~cjlin/liblinear/|LIBLINEAR library]]. The interface follows the interface for matlab very closely.

Two functions are provided, ''liblinear.train'' and ''liblinear.predict''. Calling these functions with no arguments prints the usage information.

<code>
liblinear.train()
Usage: model = train(training_data, 'liblinear_options');
liblinear_options:
-s type : set type of solver (default 1)
	 0 -- L2-regularized logistic regression (primal)
	 1 -- L2-regularized L2-loss support vector classification (dual)
	 2 -- L2-regularized L2-loss support vector classification (primal)
	 3 -- L2-regularized L1-loss support vector classification (dual)
	 4 -- multi-class support vector classification by Crammer and Singer
	 5 -- L1-regularized L2-loss support vector classification
	 6 -- L1-regularized logistic regression
	 7 -- L2-regularized logistic regression (dual)
	11 -- L2-regularized L2-loss epsilon support vector regression (primal)
	12 -- L2-regularized L2-loss epsilon support vector regression (dual)
	13 -- L2-regularized L1-loss epsilon support vector regression (dual)
-c cost : set the parameter C (default 1)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-e epsilon : set tolerance of termination criterion
	-s 0 and 2
		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
		where f is the primal function and pos/neg are # of
		positive/negative data (default 0.01)
	-s 11
		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)
	-s 1, 3, 4 and 7
		Dual maximal violation <= eps; similar to libsvm (default 0.1)
	-s 5 and 6
		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,
		where f is the primal function (default 0.01)
	-s 12 and 13
		|f'(alpha)|_1 <= eps |f'(alpha0)|,
		where f is the dual function (default 0.1)
-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
-wi weight: weights adjust the parameter C of different classes (see README for details)
-v n: n-fold cross validation mode
-q : quiet mode (no outputs)
</code>

===== A Sample Session =====

<code lua>

dtr=svm.ascread('liblinear/liblinear/heart_scale')
dte=svm.ascread('liblinear/liblinear/heart_scale')

print('======================================')
print('LIBLINEAR L2-regularized L2-loss support vector classification (dual)')
model = liblinear.train(dtr)
l,acc,d = liblinear.predict(dte,model)
</code>
And you should see 
<code>
Accuracy=84.4444 % (228/270)
</code>
as the last line.

Note that the dataset used for ''liblinear'' is the raw output of ''svm.ascread''.

==== model = liblinear.train( data [, options]) ====
{{anchor:svm.liblinear.train}}

This function trains a model using the dataset and given options. The options are given as a string and has the same syntax as command line liblinear ''train'' program arguments. An empty call to ''liblinear.train()'' prints the usage information.

==== label,accuracy,decision = liblinear.predict(data, model [,options]) ====
{{anchor:svm.liblinear.predict}}

This functions tests on the ''data'' using the ''model'' trained in previous step. Again, the options are given as a single string and the syntax is the same as ''predict'' program arguments. Any empty call to ''liblinear.predict()'' prints the usage information.

