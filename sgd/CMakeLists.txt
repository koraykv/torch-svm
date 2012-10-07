
SET(src)

SET(luasrc init.lua 
  svmsgd.lua
  svmasgd.lua
  svmloss.lua
  )

INCLUDE_DIRECTORIES(${Torch_SOURCE_INCLUDES})
ADD_TORCH_PACKAGE(svmsgd "${src}"  "${luasrc}" "SGD SVM Interface")

