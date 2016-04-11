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
	printf("Usage: model = load_model(model_file_name);\n");
}


// Interface function of torch
static int libsvm_load_model( lua_State *L )
{
	int nrhs = lua_gettop(L);
	if(nrhs != 1)
	{
		exit_with_help();
		return 0;
	}
	if (! lua_isstring(L, -1))
	{
		exit_with_help();
		return 0;
	}

	const char *model_file_name = (char *)lua_tostring(L, -1);
	struct svm_model *model = svm_load_model(model_file_name);
	if (model == NULL)
	{
		printf("Failed to load libsvm model %s\n", model_file_name);
		return 0;
	}

	libsvm_model_to_torch_structure(L, model);

	return 1;
}

static const struct luaL_Reg libsvm_load_model_util__ [] = {
  {"load_model", libsvm_load_model},
  {NULL, NULL}
};


int liblibsvm_load_model_init(lua_State *L)
{
  luaL_register(L, "libsvm", libsvm_load_model_util__);
  return 1;
}
