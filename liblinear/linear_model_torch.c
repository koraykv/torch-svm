#include <stdlib.h>
#include <string.h>
#include "liblinear/linear.h"

#include "TH.h"
#include "luaT.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

int liblinear_model_to_torch_structure(lua_State *L, struct model *model_)
{
	int i;
	int nr_w;
	int n, w_size;

	// model table
	lua_newtable(L);

	// solver type (Parameters, but we only use solver_type)
	lua_pushstring(L,"solver_type");
	lua_pushinteger(L,model_->param.solver_type);
	lua_settable(L,-3);

	// nr_class
	lua_pushstring(L,"nr_class");
	lua_pushinteger(L,model_->nr_class);
	lua_settable(L,-3);

	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	// nr_feature
	lua_pushstring(L,"nr_feature");
	lua_pushinteger(L,model_->nr_feature);
	lua_settable(L,-3);

	// bias
	lua_pushstring(L,"bias");
	lua_pushnumber(L,model_->bias);
	lua_settable(L,-3);

	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;

	w_size = n;
	// Label
	THIntTensor *label;
	if(model_->label)
	{
		label = THIntTensor_newWithSize1d((long)(model_->nr_class));
		int *label_data = THIntTensor_data(label);
		for(i = 0; i < model_->nr_class; i++)
			label_data[i] = model_->label[i];
	}
	else
	{
		label = THIntTensor_new();		
	}
	lua_pushstring(L,"label");
	luaT_pushudata(L,label,"torch.IntTensor");
	lua_settable(L,-3);

	// w
	THDoubleTensor *w = THDoubleTensor_newWithSize2d((long)nr_w,(long)w_size);
	double * w_data = THDoubleTensor_data(w);
	for(i = 0; i < w_size*nr_w; i++)
		w_data[i]=model_->w[i];
	lua_pushstring(L,"weight");
	luaT_pushudata(L,w,"torch.DoubleTensor");
	lua_settable(L,-3);

	return 1;
}

int torch_structure_to_liblinear_model(struct model *model_, lua_State *L)
{
	int i, num_of_fields;
	int nr_w;
	int n, w_size;

	num_of_fields = lua_gettop(L);

	// init
	model_->nr_class=0;
	nr_w=0;
	model_->nr_feature=0;
	model_->w=NULL;
	model_->label=NULL;

	// Parameters
	lua_pushstring(L,"solver_type");
	lua_gettable(L,-2);
	if (!lua_isnumber(L,-1))
		luaL_error(L,"solver_type expected to be integer %s",luaL_typename(L,-1));
	model_->param.solver_type = lua_tointeger(L,-1);
	lua_pop(L,1);

	// nr_class
	lua_pushstring(L,"nr_class");
	lua_gettable(L,-2);
	if (!lua_isnumber(L,-1))
		luaL_error(L,"nr_class expected to be integer");
	model_->nr_class = lua_tointeger(L,-1);
	lua_pop(L,1);

	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	// nr_feature
	lua_pushstring(L,"nr_feature");
	lua_gettable(L,-2);
	if (!lua_isnumber(L,-1))
		luaL_error(L,"nr_feature expected to be integer");
	model_->nr_feature = lua_tointeger(L,-1);
	lua_pop(L,1);

	// bias
	lua_pushstring(L,"bias");
	lua_gettable(L,-2);
	if (!lua_isnumber(L,-1))
		luaL_error(L,"bias expected to be a number");
	model_->bias = lua_tonumber(L,-1);
	lua_pop(L,1);

	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	w_size = n;

	// Label
	lua_pushstring(L,"label");
	lua_gettable(L,-2);
	THIntTensor *label = luaT_checkudata(L,-1,"torch.IntTensor");
	lua_pop(L,1);
	int nlabel = (int)THIntTensor_nElement(label);
	if( nlabel > 0)
	{
		if (nlabel != model_->nr_class)
			luaL_error(L,"Number of elements in label vector is different than nr_class");

		int *label_data = THIntTensor_data(label);
		model_->label = Malloc(int, model_->nr_class);
		for(i=0;i<model_->nr_class;i++)
			model_->label[i] = label_data[i];
	}

	//w
	lua_pushstring(L,"weight");
	lua_gettable(L,-2);
	THDoubleTensor *w = luaT_checkudata(L,-1,"torch.DoubleTensor");
	lua_pop(L,1);
	double *w_data = THDoubleTensor_data(w);
	model_->w=Malloc(double, w_size*nr_w);
	for(i = 0; i < w_size*nr_w; i++)
		model_->w[i] = w_data[i];

	return 1;
}


