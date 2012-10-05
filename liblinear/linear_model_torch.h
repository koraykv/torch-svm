#include "TH.h"
#include "luaT.h"

int model_to_torch_structure( lua_State *L, struct model *model_);
int torch_structure_to_model(struct model *model_,  lua_State *L);
