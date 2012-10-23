#include <stdlib.h>
#include <string.h>
#include "libsvm/svm.h"

#include "TH.h"
#include "luaT.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

int libsvm_model_to_torch_structure(lua_State *L, struct svm_model *model_)
{
	int i,j,n;

	// model table
	lua_newtable(L);

	// solver type (Parameters, but we only use solver_type)
	lua_pushstring(L,"svm_type");
	lua_pushinteger(L,model_->param.svm_type);
	lua_settable(L,-3);

	// solver type (Parameters, but we only use solver_type)
	lua_pushstring(L,"kernel_type");
	lua_pushinteger(L,model_->param.kernel_type);
	lua_settable(L,-3);

	// solver type (Parameters, but we only use solver_type)
	lua_pushstring(L,"degree");
	lua_pushinteger(L,model_->param.degree);
	lua_settable(L,-3);

	// solver type (Parameters, but we only use solver_type)
	lua_pushstring(L,"gamma");
	lua_pushnumber(L,model_->param.gamma);
	lua_settable(L,-3);

	// solver type (Parameters, but we only use solver_type)
	lua_pushstring(L,"coef0");
	lua_pushnumber(L,model_->param.coef0);
	lua_settable(L,-3);

	// nr_class
	lua_pushstring(L,"nr_class");
	lua_pushinteger(L,model_->nr_class);
	lua_settable(L,-3);

	// total_SV
	lua_pushstring(L,"totalSV");
	lua_pushinteger(L,model_->l);
	lua_settable(L,-3);

	n = model_->nr_class*(model_->nr_class-1)/2;
	THDoubleTensor *rho = THDoubleTensor_newWithSize1d(n);
	double *rho_data = THDoubleTensor_data(rho);
	for (i=0; i<n; i++)
	{
		rho_data[i] = model_->rho[i];
	}
	lua_pushstring(L,"rho");
	luaT_pushudata(L,rho,"torch.DoubleTensor");
	lua_settable(L,-3);

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

	// probA
	THDoubleTensor *probA;
	if(model_->probA != NULL)
	{
		probA = THDoubleTensor_newWithSize1d(n);
		double *probA_data = THDoubleTensor_data(probA);
		for (i=0; i<n; i++)
		{
			probA_data[i] = model_->probA[i];
		}
	}
	else
	{
		probA = THDoubleTensor_new();
	}
	lua_pushstring(L,"probA");
	luaT_pushudata(L,probA,"torch.DoubleTensor");
	lua_settable(L,-3);

	// probB
	THDoubleTensor *probB;
	if(model_->probB != NULL)
	{
		probB = THDoubleTensor_newWithSize1d(n);
		double *probB_data = THDoubleTensor_data(probB);
		for (i=0; i<n; i++)
		{
			probB_data[i] = model_->probB[i];
		}
	}
	else
	{
		probB = THDoubleTensor_new();
	}
	lua_pushstring(L,"probB");
	luaT_pushudata(L,probB,"torch.DoubleTensor");
	lua_settable(L,-3);

	// Label
	THIntTensor *nSV;
	if(model_->nSV)
	{
		nSV = THIntTensor_newWithSize1d((long)(model_->nr_class));
		int *nSV_data = THIntTensor_data(nSV);
		for(i = 0; i < model_->nr_class; i++)
			nSV_data[i] = model_->nSV[i];
	}
	else
	{
		nSV = THIntTensor_new();		
	}
	lua_pushstring(L,"nSV");
	luaT_pushudata(L,nSV,"torch.IntTensor");
	lua_settable(L,-3);

	// sv_coef
	THDoubleTensor *sv_coef = THDoubleTensor_newWithSize2d(model_->l, model_->nr_class-1);
	double *sv_coef_data = THDoubleTensor_data(sv_coef);
	for(i = 0; i < model_->nr_class-1; i++)
		for(j = 0; j < model_->l; j++)
			sv_coef_data[(i*(model_->l))+j] = model_->sv_coef[i][j];
	lua_pushstring(L,"sv_coef");
	luaT_pushudata(L,sv_coef,"torch.DoubleTensor");
	lua_settable(L,-3);

	// SVs
	lua_pushstring(L,"SVs");
	lua_newtable(L);
	for(i = 0;i < model_->l; i++)
	{
		lua_pushnumber(L,i+1);
		if(model_->param.kernel_type == PRECOMPUTED)
		{
			lua_pushnumber(L,model_->SV[i][0].value);
		}
		else
		{
			int x_index = 0;
			while (model_->SV[i][x_index].index != -1)
				x_index++;

			THIntTensor *indices = THIntTensor_newWithSize1d(x_index);
			int *indices_data = THIntTensor_data(indices);
			THDoubleTensor *vals = THDoubleTensor_newWithSize1d(x_index);
			double *vals_data = THDoubleTensor_data(vals);

			x_index = 0;
			while (model_->SV[i][x_index].index != -1)
			{
				indices_data[x_index] = model_->SV[i][x_index].index;
				vals_data[x_index] = model_->SV[i][x_index].value;
				x_index++;
			}
			lua_newtable(L);
			lua_pushnumber(L,1);
			luaT_pushudata(L,indices,"torch.IntTensor");
			lua_settable(L,-3);
			lua_pushnumber(L,2);
			luaT_pushudata(L,vals,"torch.DoubleTensor");
			lua_settable(L,-3);
		}
		lua_settable(L,-3);
	}
	lua_settable(L,-3);
	return 1;
}

int torch_structure_to_libsvm_model(struct svm_model *model_, lua_State *L)
{
	int i, j, n;
	struct svm_node *x_space;

	// init
	model_->rho = NULL;
	model_->probA = NULL;
	model_->probB = NULL;
	model_->label = NULL;
	model_->nSV = NULL;
	model_->free_sv = 1; // XXX

	// Parameters
	lua_pushstring(L,"svm_type");
	lua_gettable(L,-2);
	if (!lua_isnumber(L,-1))
		luaL_error(L,"svm_type expected to be integer %s",luaL_typename(L,-1));
	model_->param.svm_type = lua_tointeger(L,-1);
	lua_pop(L,1);

	lua_pushstring(L,"kernel_type");
	lua_gettable(L,-2);
	if (!lua_isnumber(L,-1))
		luaL_error(L,"kernel_type expected to be integer %s",luaL_typename(L,-1));
	model_->param.kernel_type = lua_tointeger(L,-1);
	lua_pop(L,1);
	
	lua_pushstring(L,"degree");
	lua_gettable(L,-2);
	if (!lua_isnumber(L,-1))
		luaL_error(L,"degree expected to be integer %s",luaL_typename(L,-1));
	model_->param.degree = lua_tointeger(L,-1);
	lua_pop(L,1);

	lua_pushstring(L,"gamma");
	lua_gettable(L,-2);
	if (!lua_isnumber(L,-1))
		luaL_error(L,"gamma expected to be number %s",luaL_typename(L,-1));
	model_->param.gamma = lua_tonumber(L,-1);
	lua_pop(L,1);

	lua_pushstring(L,"coef0");
	lua_gettable(L,-2);
	if (!lua_isnumber(L,-1))
		luaL_error(L,"coef0 expected to be number %s",luaL_typename(L,-1));
	model_->param.coef0 = lua_tonumber(L,-1);
	lua_pop(L,1);

	lua_pushstring(L,"nr_class");
	lua_gettable(L,-2);
	if (!lua_isnumber(L,-1))
		luaL_error(L,"nr_class expected to be integer %s",luaL_typename(L,-1));
	model_->nr_class = lua_tointeger(L,-1);
	lua_pop(L,1);

	lua_pushstring(L,"totalSV");
	lua_gettable(L,-2);
	if (!lua_isnumber(L,-1))
		luaL_error(L,"totalSV expected to be integer %s",luaL_typename(L,-1));
	model_->l = lua_tointeger(L,-1);
	lua_pop(L,1);

	lua_pushstring(L,"rho");
	lua_gettable(L,-2);
	THDoubleTensor *rho = luaT_checkudata(L,-1,"torch.DoubleTensor");
	lua_pop(L,1);
	double *rho_data = THDoubleTensor_data(rho);
	n = model_->nr_class * (model_->nr_class-1)/2;
	model_->rho = (double*) malloc(n*sizeof(double));
	for (i=0; i<n; i++)
		model_->rho[i] = rho_data[i];

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

	// probA
	lua_pushstring(L,"probA");
	lua_gettable(L,-2);
	THDoubleTensor *probA = luaT_checkudata(L,-1,"torch.DoubleTensor");
	lua_pop(L,1);
	int nprobA = (int)THDoubleTensor_nElement(probA);
	if( nprobA > 0)
	{
		if (nprobA != n)
			luaL_error(L,"Number of elements in probA is different than n");
		double *probA_data = THDoubleTensor_data(probA);
		model_->probA = (double*) malloc(n*sizeof(double));
		for(i=0;i<n;i++)
			model_->probA[i] = probA_data[i];
	}

	// probB
	lua_pushstring(L,"probB");
	lua_gettable(L,-2);
	THDoubleTensor *probB = luaT_checkudata(L,-1,"torch.DoubleTensor");
	lua_pop(L,1);
	int nprobB = (int)THDoubleTensor_nElement(probB);
	if( nprobB > 0)
	{
		if (nprobB != n)
			luaL_error(L,"Number of elements in probB is different than n");
		double *probB_data = THDoubleTensor_data(probB);
		model_->probB = (double*) malloc(n*sizeof(double));
		for(i=0;i<n;i++)
			model_->probB[i] = probB_data[i];
	}

	// nSV
	lua_pushstring(L,"nSV");
	lua_gettable(L,-2);
	THIntTensor *nSV = luaT_checkudata(L,-1,"torch.IntTensor");
	lua_pop(L,1);
	int nnSV = (int)THIntTensor_nElement(nSV);
	if( nnSV > 0)
	{
		if (nnSV != model_->nr_class)
			luaL_error(L,"Number of elements in nSV vector is different than nr_class");

		int *nSV_data = THIntTensor_data(nSV);
		model_->nSV = Malloc(int, model_->nr_class);
		for(i=0;i<model_->nr_class;i++)
			model_->nSV[i] = nSV_data[i];
	}

	// sv_coef
	lua_pushstring(L,"sv_coef");
	lua_gettable(L,-2);
	THDoubleTensor *sv_coef = luaT_checkudata(L,-1,"torch.DoubleTensor");
	lua_pop(L,1);
	double *sv_coef_data = THDoubleTensor_data(sv_coef);
	model_->sv_coef = (double**) malloc((model_->nr_class-1)*sizeof(double));
	for( i=0 ; i< model_->nr_class -1 ; i++ )
		model_->sv_coef[i] = (double*) malloc((model_->l)*sizeof(double));
	for(i = 0; i < model_->nr_class - 1; i++)
		for(j = 0; j < model_->l; j++)
			model_->sv_coef[i][j] = sv_coef_data[i*(model_->l)+j];


	// SV
	{
		lua_pushstring(L,"SVs");
		lua_gettable(L,-2);
		int sr, elements, num_samples;

		num_samples = 0;
		sr = lua_objlen(L,-1);
		for (i=0; i<sr; i++)
		{
			lua_pushnumber(L,i+1);
			lua_gettable(L,-2);
			lua_pushnumber(L,1);
			lua_gettable(L,-2);			
			THIntTensor *ind = luaT_checkudata(L,-1,"torch.IntTensor");
			num_samples += ind->size[0];
			lua_pop(L,1);
			lua_pop(L,1);
		}

		elements = num_samples + sr;

		model_->SV = (struct svm_node **) malloc(sr * sizeof(struct svm_node *));
		x_space = (struct svm_node *)malloc(elements * sizeof(struct svm_node));

		int xi = 0;
		for(i=0;i<sr;i++)
		{
			lua_pushnumber(L,i+1);
			lua_gettable(L,-2);
			model_->SV[i] = &x_space[xi];

			lua_pushnumber(L,1);
			lua_gettable(L,-2);
			THIntTensor *inds = luaT_checkudata(L,-1,"torch.IntTensor");
			int *inds_data = THIntTensor_data(inds);
			lua_pop(L,1);
			lua_pushnumber(L,2);
			lua_gettable(L,-2);
			THDoubleTensor *vals = luaT_checkudata(L,-1,"torch.DoubleTensor");
			double *vals_data = THDoubleTensor_data(vals);
			lua_pop(L,1);
			int nf = inds->size[0];
			for(j=0; j<nf; j++)
			{
				model_->SV[i][j].index = inds_data[j];
				model_->SV[i][j].value = vals_data[j];
			}
			model_->SV[i][nf].index = -1;
			xi += nf;
			xi++;
			lua_pop(L,1);
		}
		lua_pop(L,1);
	}
	return 1;
}


