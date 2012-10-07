
#include "TH.h"
#include "luaT.h"

static int svm_spdot(lua_State *L)
{
	THFloatTensor *tdense = luaT_checkudata(L,1,"torch.FloatTensor");
	THIntTensor *indices;
	if lua_isnil(L,2)
	{
		indices = NULL;
	}
	else
	{
		indices = luaT_checkudata(L,2,"torch.IntTensor");
	}
	THFloatTensor *tsparse = luaT_checkudata(L,3,"torch.FloatTensor");

	luaL_argcheck(L,tdense->nDimension == 1, 1, "Dense Matrix is expected to 1D");
	luaL_argcheck(L,!indices || indices->nDimension == 1, 2, "Index tensor is expected to 1D");
	luaL_argcheck(L,tsparse->nDimension == 1, 3, "Sparse value tensor is expected to 1D");

	if (!indices)
	{
		lua_pushnumber(L,(double)THFloatTensor_dot(tdense,tsparse));
		return 1;
	}

	float *dense_data = THFloatTensor_data(tdense);
	float *sparse_data = THFloatTensor_data(tsparse);
	int *indices_data = THIntTensor_data(indices);

	long i;
	float res = 0;

	for (i=0; i< indices->size[0]; i++)
	{
		res += sparse_data[i]*dense_data[indices_data[i]-1];
	}
	lua_pushnumber(L,(double)res);
	return 1;
}

static int svm_spadd(lua_State *L)
{
	THFloatTensor *tdense = luaT_checkudata(L,1,"torch.FloatTensor");
	float c = (float)lua_tonumber(L,2);
	THIntTensor *indices;
	if (lua_isnil(L,3))
	{
		indices = NULL;
	}
	else
	{
		indices = luaT_checkudata(L,3,"torch.IntTensor");
	}
	THFloatTensor *tsparse = luaT_checkudata(L,4,"torch.FloatTensor");

	luaL_argcheck(L,tdense->nDimension == 1, 1, "Dense Matrix is expected to 1D");
	luaL_argcheck(L,!indices||indices->nDimension == 1, 3, "Index tensor is expected to 1D");
	luaL_argcheck(L,tsparse->nDimension == 1, 4, "Sparse value tensor is expected to 1D");

	if(!indices)
	{
		THFloatTensor_cadd(tdense,tdense,c,tsparse);
		return 0;
	}

	float *dense_data = THFloatTensor_data(tdense);
	float *sparse_data = THFloatTensor_data(tsparse);
	int *indices_data = THIntTensor_data(indices);

	long i;

	for (i=0; i< indices->size[0]; i++)
	{
		dense_data[indices_data[i]-1] += c*sparse_data[i];
	}
	return 0;
}

static const struct luaL_Reg svm_util__ [] = {
  {"spdot", svm_spdot},
  {"spadd", svm_spadd},
  {NULL, NULL}
};


int libsvm_util_init(lua_State *L)
{
  luaL_register(L, "svm", svm_util__);
  return 1;
}
