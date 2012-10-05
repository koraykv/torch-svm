CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

SET(src
	linear_model_torch.h 
	linear_model_torch.c 
	init.c
	liblinear_train.c 
	liblinear_predict.c 
	liblinear/linear.h liblinear/linear.cpp 
	liblinear/tron.h liblinear/tron.cpp
)

SET(luasrc init.lua 
  )

SET(blassrc
	liblinear/blas/blas.h
	liblinear/blas/blasp.h
	liblinear/blas/daxpy.c
	liblinear/blas/ddot.c
	liblinear/blas/dnrm2.c
	liblinear/blas/dscal.c
)

ADD_LIBRARY(liblinearblas STATIC ${blassrc})

ADD_TORCH_PACKAGE(liblinear "${src}" "${luasrc}" "LIBLINEAR Interface")

TARGET_LINK_LIBRARIES(liblinear luaT TH liblinearblas)
