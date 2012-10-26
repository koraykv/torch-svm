
SET(src)

SET(luasrc init.lua 
  sgd.lua
  asgd.lua
  loss.lua
  )

INCLUDE_DIRECTORIES(${Torch_SOURCE_INCLUDES})
ADD_TORCH_PACKAGE(svmsgd "${src}"  "${luasrc}" "SGD SVM Interface")

