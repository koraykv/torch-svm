
SET(src
	svm_model_torch.h 
	svm_model_torch.c 
	init.c
	libsvm_train.c 
	libsvm_predict.c 
	libsvm/svm.h
	libsvm/svm.cpp
)

SET(luasrc init.lua 
  )

ADD_TORCH_PACKAGE(libsvm "${src}" "${luasrc}" "LIBSVM Interface")

TARGET_LINK_LIBRARIES(libsvm luaT TH )
