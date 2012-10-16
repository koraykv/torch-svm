#include "TH.h"
#include "luaT.h"

int liblinear_model_to_torch_structure( lua_State *L, struct model *model_);
int torch_structure_to_liblinear_model(struct model *model_,  lua_State *L);
