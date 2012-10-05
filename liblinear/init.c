
#include "luaT.h"

extern int libliblinear_predict_init(lua_State *L);
extern int libliblinear_train_init(lua_State *L);

DLL_EXPORT int luaopen_libliblinear(lua_State *L)
{
  	libliblinear_predict_init(L);
  	libliblinear_train_init(L);
  	return 1;
}
