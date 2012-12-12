#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "TH.h"
#include "luaT.h"

#include "liblinear/linear.h"
#include "linear_model_torch.h"


#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void read_sparse_instance(lua_State *L, int index, double *target_label, struct feature_node *x, int feature_number, double bias)
{
	lua_pushnumber(L,index+1);lua_gettable(L,-2);
	luaL_argcheck(L,lua_istable(L,1),1,"Expecting table in read_sparse_instance");
	int j = 0;
	{
		// get label
		lua_pushnumber(L,1);lua_gettable(L,-2);
		*target_label = (double)lua_tonumber(L,-1);
		lua_pop(L,1);
		// get values
		lua_pushnumber(L,2);lua_gettable(L,-2);
		{
			lua_pushnumber(L,1);lua_gettable(L,-2);
			THIntTensor *indices = luaT_checkudata(L,-1,"torch.IntTensor");
			lua_pop(L,1);
			lua_pushnumber(L,2);lua_gettable(L,-2);
			THFloatTensor *vals = luaT_checkudata(L,-1,"torch.FloatTensor");
			lua_pop(L,1);

			int *indices_data = THIntTensor_data(indices);
			float *vals_data = THFloatTensor_data(vals);
			int k;
			for (k=0; k<(int)THIntTensor_nElement(indices); k++)
			{
				x[j].index = indices_data[k];
				x[j].value = vals_data[k];
				j++;
			}
			if (bias >= 0)
			{
				x[j].index = feature_number+1;
				x[j].value = bias;
				j++;
			}
			x[j++].index = -1;
		}
		lua_pop(L,1);
	}
	lua_pop(L,1);
}

int do_predict(lua_State *L, struct model *model_, const int predict_probability_flag)
{
	int label_vector_row_num;
	int feature_number, testing_instance_number;
	int instance_index;
	double *ptr_predict_label;
	double *ptr_prob_estimates, *ptr_dec_values;
	struct feature_node *x;
	THDoubleTensor *label;
	THDoubleTensor *dec;

	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int nr_class=get_nr_class(model_);
	int nr_w;
	double *prob_estimates=NULL;

	if(nr_class==2 && model_->param.solver_type!=MCSVM_CS)
		nr_w=1;
	else
		nr_w=nr_class;


	luaL_argcheck(L,lua_istable(L,1),1,"Expecting table in do_predict");

	// prhs[1] = testing instance matrix
	feature_number = get_nr_feature(model_);
	testing_instance_number = (int) lua_objlen(L,1);
	label_vector_row_num = testing_instance_number;

	prob_estimates = Malloc(double, nr_class);

	label = THDoubleTensor_newWithSize1d(testing_instance_number);

	if (predict_probability_flag)
		dec = THDoubleTensor_newWithSize2d(testing_instance_number,nr_class);
	else
		dec = THDoubleTensor_newWithSize2d(testing_instance_number,nr_w);

	ptr_predict_label = THDoubleTensor_data(label);
	ptr_prob_estimates = THDoubleTensor_data(dec);
	ptr_dec_values = THDoubleTensor_data(dec);

	x = Malloc(struct feature_node, feature_number+2);
	for(instance_index=0;instance_index<testing_instance_number;instance_index++)
	{
		int i;
		double target_label, predict_label;

		// prhs[1] and prhs[1]^T are sparse
		read_sparse_instance(L,instance_index, &target_label, x, feature_number, model_->bias);

		if(predict_probability_flag)
		{
			predict_label = predict_probability(model_, x, prob_estimates);
			ptr_predict_label[instance_index] = predict_label;
			for(i=0;i<nr_class;i++)
				ptr_prob_estimates[instance_index*nr_class + i ] = prob_estimates[i];
		}
		else
		{
			double *dec_values = Malloc(double, nr_class);
			predict_label = predict_values(model_, x, dec_values);
			ptr_predict_label[instance_index] = predict_label;

			for(i=0;i<nr_w;i++)
				ptr_dec_values[instance_index*nr_w + i] = dec_values[i];
			free(dec_values);
		}

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;

		++total;
	}
	
	if(model_->param.solver_type==L2R_L2LOSS_SVR || 
           model_->param.solver_type==L2R_L1LOSS_SVR_DUAL || 
           model_->param.solver_type==L2R_L2LOSS_SVR_DUAL)
        {
                printf("Mean squared error = %g (regression)\n",error/total);
                printf("Squared correlation coefficient = %g (regression)\n",
                       ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
                       ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
                       );
        }
	else
		printf("Accuracy = %g%% (%d/%d)\n", (double) correct/total*100,correct,total);


	// label = res[1]
	luaT_pushudata(L,label,"torch.DoubleTensor");

	// acc = res[2] : {accuracy, mean squared error, squared correlation coefficient}
	lua_newtable(L);
	lua_pushnumber(L,1);
	lua_pushnumber(L,(double)correct/total*100);
	lua_settable(L,-3);
	
	lua_pushnumber(L,2);
	lua_pushnumber(L,(double)error/total);
	lua_settable(L,-3);

	lua_pushnumber(L,3);
	lua_pushnumber(L,(double)
		((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
		((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt)));
	lua_settable(L,-3);

	// prob = res[3]
	luaT_pushudata(L,dec,"torch.DoubleTensor");

	free(x);
	if(prob_estimates != NULL)
		free(prob_estimates);

	return 3;
}

static void exit_with_help()
{
	printf(
			"Usage: [predicted_label, accuracy, decision_values/prob_estimates] = predict(testing_instance_data, model, 'liblinear_options','col')\n"
			"liblinear_options:\n"
			"-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only\n"
			"Returns:\n"
			"  predicted_label: prediction output vector.\n"
			"  accuracy: a table with accuracy, mean squared error, squared correlation coefficient.\n"
			"  prob_estimates: If selected, probability estimate vector.\n"
			);
}

static int liblinear_predict( lua_State *L )
{
	int nrhs = lua_gettop(L);
	int prob_estimate_flag = 0;
	struct model *model_;

	if(nrhs > 3 || nrhs < 2)
	{
		exit_with_help();
		return 0;
	}

	// parse options
	if(nrhs == 3)
	{
		int i, argc = 1;
		char *argv[CMD_LEN/2];

		// put options in argv[]
	        size_t slen;
		const char *tcmd = lua_tolstring(L,3,&slen);
		char cmd[slen];
		strcpy(cmd,tcmd);
		if((argv[argc] = strtok((char*)cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;

		for(i=1;i<argc;i++)
		{
			if(argv[i][0] != '-') break;
			if(++i>=argc)
			{
				exit_with_help();
				return 0;
			}
			switch(argv[i-1][1])
			{
				case 'b':
					prob_estimate_flag = atoi(argv[i]);
					break;
				default:
					printf("unknown option\n");
					exit_with_help();
					return 0;
			}
		}
		lua_pop(L,1);
	}

	model_ = Malloc(struct model, 1);
	torch_structure_to_liblinear_model(model_, L);
	lua_pop(L,1);

	if(prob_estimate_flag)
	{
		if(!check_probability_model(model_))
		{
			printf("probability output is only supported for logistic regression\n");
			prob_estimate_flag=0;
		}
	}
	int nres = do_predict(L, model_, prob_estimate_flag);
	// destroy model_
	free_and_destroy_model(&model_);

	return nres;
}

static const struct luaL_Reg liblinear_predict_util__ [] = {
  {"predict", liblinear_predict},
  {NULL, NULL}
};


int libliblinear_predict_init(lua_State *L)
{
  luaL_register(L, "liblinear", liblinear_predict_util__);
  return 1;
}
