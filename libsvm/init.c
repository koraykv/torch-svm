
#include "luaT.h"

extern int liblibsvm_predict_init(lua_State *L);
extern int liblibsvm_train_init(lua_State *L);

DLL_EXPORT int luaopen_liblibsvm(lua_State *L)
{
  	liblibsvm_predict_init(L);
  	liblibsvm_train_init(L);
  	return 1;
}
