#include "luaT.h"

extern int libsvm_data_init(lua_State *L);
extern int libsvm_util_init(lua_State *L);

DLL_EXPORT int luaopen_libsvmutil(lua_State *L)
{
	libsvm_data_init(L);
	libsvm_util_init(L);
	return 1;
}
