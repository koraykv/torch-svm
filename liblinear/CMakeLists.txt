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

ADD_TORCH_PACKAGE(liblinear "${src}" "${luasrc}" "LIBLINEAR Interface")

TARGET_LINK_LIBRARIES(liblinear luaT TH ${CMAKE_CURRENT_SOURCE_DIR}/liblinear/blas/blas.a)
