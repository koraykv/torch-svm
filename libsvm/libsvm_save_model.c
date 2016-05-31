#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "TH.h"
#include "luaT.h"

#include "libsvm/svm.h"
#include "svm_model_torch.h"


static void exit_with_help()
{
	printf("Usage: success = save_model(model_file_name, model);\n");
}


// Interface function of torch
static int libsvm_save_model( lua_State *L )
{
	int nrhs = lua_gettop(L);
	if(nrhs != 2)
	{
		exit_with_help();
		return 0;
	}
	if (! lua_isstring(L, -2))
	{
		exit_with_help();
		return 0;
	}
	if (! lua_istable(L, -1))
  	{
  		exit_with_help();
  		return 0;
  	}

	const char *model_file_name = (char *)lua_tostring(L, -2);
	struct svm_model *model = (struct svm_model *)malloc(
		sizeof(struct svm_model));
	torch_structure_to_libsvm_model(model, L);

	const int res = svm_save_model(model_file_name, model);
	lua_pop(L, 1);
	if (res == 0) {
		lua_pushboolean(L, 1);
		return 1;
	}

	lua_pushboolean(L, 0);
	return 1;
}

static const struct luaL_Reg libsvm_save_model_util__ [] = {
  {"save_model", libsvm_save_model},
  {NULL, NULL}
};


int liblibsvm_save_model_init(lua_State *L)
{
  luaL_register(L, "libsvm", libsvm_save_model_util__);
  return 1;
}
