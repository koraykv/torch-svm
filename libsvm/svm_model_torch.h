#include "TH.h"
#include "luaT.h"

int libsvm_model_to_torch_structure( lua_State *L, struct svm_model *model_);
int torch_structure_to_libsvm_model(struct svm_model *model_,  lua_State *L);
