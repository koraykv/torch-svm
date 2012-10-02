require 'svm'

dtr=svm:dataset('rcv1.train.bin')
dte=svm:dataset('rcv1.test.bin')


-- print('======================================')
-- print('SVM SGD HingeLoss')
-- -- sgd (hinge)
-- mysvm = svm.SvmSgd(dtr:nfeature(),1e-4)
-- print(mysvm)
-- mysvm:determineEta0(1001,dtr)
-- mysvm:train(dtr,dte,5)

print('======================================')
print('SVM ASGD HingeLoss')
-- asgd (hinge)
mysvm = svm.SvmAsgd(dtr:nfeature(),1e-4)
print(mysvm)
mysvm:determineEta0(1001,dtr)
mysvm:train(dtr,dte,3)

print('======================================')
print('SVM ASGD2 HingeLoss')
-- asgd (hinge)
mysvm = svm.SvmAsgd2(dtr:nfeature(),1e-4)
print(mysvm)
mysvm:determineEta0(1001,dtr)
mysvm:train(dtr,dte,3)

-- print('======================================')
-- print('SVM SGD LogLoss')
-- -- sgd(logloss)
-- mysvm = svm.SvmSgd(dtr:nfeature(),5e-7)
-- mysvm.svmloss = svm.logloss
-- mysvm.regbias = false
-- print(mysvm)
-- mysvm:determineEta0(1001,dtr)
-- mysvm:train(dtr,dte,12)

print('======================================')
print('SVM ASGD LogLoss')
-- asgd(logloss)
mysvm = svm.SvmAsgd(dtr:nfeature(),5e-7)
mysvm.svmloss = svm.logloss
mysvm.regbias = false
print(mysvm)
mysvm:determineEta0(1001,dtr)
mysvm:train(dtr,dte,8)

print('======================================')
print('SVM ASGD2 LogLoss')
-- asgd(logloss)
mysvm = svm.SvmAsgd2(dtr:nfeature(),5e-7)
mysvm.svmloss = svm.logloss
mysvm.regbias = false
print(mysvm)
mysvm:determineEta0(1001,dtr)
mysvm:train(dtr,dte,8)
