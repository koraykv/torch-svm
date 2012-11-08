require 'svm'

if #arg == 0 then arg = nil end

-- you can get these files use Leon Bottou's sgd project.
dtr=svm.dataset('../data/rcv1.train.bin')
dte=svm.dataset('../data/rcv1.test.bin')

if arg and (arg[1] == 'dense' or arg[2] == 'dense') then
	dtr:dense()
	dte:dense()
end

-- These tests are comparing with Leon Bottou's experiments.
if not arg or (arg and arg[1] == 'sgd-hinge') then
	print('======================================')
	print('SVM SGD HingeLoss')
	-- sgd (hinge)
	mysvm = svm.SvmSgd(dtr:nfeature(),1e-4)
	print(mysvm)
	mysvm:determineEta0(1001,dtr)
	mysvm:train(dtr,dte,5)
end

if not arg or (arg and arg[1] == 'asgd-hinge') then
	print('======================================')
	print('SVM ASGD HingeLoss')
	-- asgd (hinge)
	mysvm = svm.SvmAsgd(dtr:nfeature(),1e-4)
	print(mysvm)
	mysvm:determineEta0(1001,dtr)
	mysvm:train(dtr,dte,3)
end

if not arg or (arg and arg[1] == 'sgd-log') then
	print('======================================')
	print('SVM SGD LogLoss')
	-- sgd(logloss)
	mysvm = svm.SvmSgd(dtr:nfeature(),5e-7)
	mysvm.svmloss = svm.logloss
	mysvm.regbias = false
	print(mysvm)
	mysvm:determineEta0(1001,dtr)
	mysvm:train(dtr,dte,12)
end

if not arg or (arg and arg[1] == 'asgd-log') then
	print('======================================')
	print('SVM ASGD LogLoss')
	-- asgd(logloss)
	mysvm = svm.SvmAsgd(dtr:nfeature(),5e-7)
	mysvm.svmloss = svm.logloss
	mysvm.regbias = false
	print(mysvm)
	mysvm:determineEta0(1001,dtr)
	mysvm:train(dtr,dte,8)
end
