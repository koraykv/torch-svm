#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "TH.h"
#include "luaT.h"

#include "libsvm/svm.h"
#include "svm_model_torch.h"


#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define max_(a,b) (a>=b ? a : b)


void read_sparse_instance(lua_State *L, int index, double *target_label, struct svm_node *x)
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
			x[j++].index = -1;
		}
		lua_pop(L,1);
	}
	lua_pop(L,1);
}


void predict(lua_State *L, struct svm_model *model_, const int predict_probability)
{
	int label_vector_row_num;
	int feature_number, testing_instance_number;
	int instance_index;
	double *ptr_predict_label; 
	double *ptr_prob_estimates, *ptr_dec_values;
	struct svm_node *x;
	THDoubleTensor *label;
	THDoubleTensor *dec;

	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=svm_get_svm_type(model_);
	int nr_class=svm_get_nr_class(model_);
	double *prob_estimates=NULL;

	luaL_argcheck(L,lua_istable(L,1),1,"Expecting table in do_predict");


	// prhs[1] = testing instance matrix
	testing_instance_number = (int) lua_objlen(L,1);
	label_vector_row_num = testing_instance_number;

	int i;
	feature_number = -1;
	for (i=0; i< label_vector_row_num; i++)
	{
		// get the table elem
		lua_pushnumber(L,i+1);
		lua_gettable(L,-2);
		if (!lua_istable(L,-1))
			luaL_error(L,"expected table at index %d while getting max_index\n",i+1);
		{
			// get values
			lua_pushnumber(L,2);lua_gettable(L,-2);
			{
				lua_pushnumber(L,1);lua_gettable(L,-2);
				THIntTensor *indices = luaT_toudata(L,-1,"torch.IntTensor");
				feature_number = max_(feature_number,THIntTensor_get1d(indices,indices->size[0]-1));
				lua_pop(L,1);
			}
			lua_pop(L,1);
		}
		lua_pop(L,1);
	}

	if(predict_probability)
	{
		if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
			printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model_));
		else
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
	}

	label = THDoubleTensor_newWithSize1d(testing_instance_number);
	if(predict_probability)
	{
		// prob estimates are in plhs[2]
		if(svm_type==C_SVC || svm_type==NU_SVC)
			dec = THDoubleTensor_newWithSize2d(testing_instance_number,nr_class);
		else
			dec = THDoubleTensor_new();
	}
	else
	{
		// decision values are in plhs[2]
		if(svm_type == ONE_CLASS ||
		   svm_type == EPSILON_SVR ||
		   svm_type == NU_SVR ||
		   nr_class == 1) // if only one class in training data, decision values are still returned.
		 	dec = THDoubleTensor_newWithSize2d(testing_instance_number,1);
		else
			dec = THDoubleTensor_newWithSize2d(testing_instance_number,nr_class*(nr_class-1)/2);
	}

	ptr_predict_label = THDoubleTensor_data(label);
	ptr_prob_estimates = THDoubleTensor_data(dec);
	ptr_dec_values = THDoubleTensor_data(dec);

	x = (struct svm_node*)malloc((feature_number+1)*sizeof(struct svm_node) );
	for(instance_index=0;instance_index<testing_instance_number;instance_index++)
	{
		int i;
		double target_label, predict_label;

		if(model_->param.kernel_type != PRECOMPUTED) // prhs[1]^T is still sparse
			read_sparse_instance(L, instance_index, &target_label, x);
		else
		{
			printf("only sparse for now.");
			// for(i=0;i<feature_number;i++)
			// {
			// 	x[i].index = i+1;
			// 	x[i].value = ptr_instance[testing_instance_number*i+instance_index];
			// }
			// x[feature_number].index = -1;
		}

		if(predict_probability)
		{
			if(svm_type==C_SVC || svm_type==NU_SVC)
			{
				predict_label = svm_predict_probability(model_, x, prob_estimates);
				ptr_predict_label[instance_index] = predict_label;
				for(i=0;i<nr_class;i++)
					ptr_prob_estimates[instance_index * nr_class + i] = prob_estimates[i];
			} else {
				predict_label = svm_predict(model_,x);
				ptr_predict_label[instance_index] = predict_label;
			}
		}
		else
		{
			if(svm_type == ONE_CLASS ||
			   svm_type == EPSILON_SVR ||
			   svm_type == NU_SVR)
			{
				double res;
				predict_label = svm_predict_values(model_, x, &res);
				ptr_dec_values[instance_index] = res;
			}
			else
			{
				double *dec_values = (double *) malloc(sizeof(double) * nr_class*(nr_class-1)/2);
				predict_label = svm_predict_values(model_, x, dec_values);
				if(nr_class == 1) 
					ptr_dec_values[instance_index] = 1;
				else
					for(i=0;i<(nr_class*(nr_class-1))/2;i++)
						ptr_dec_values[instance_index * (nr_class*(nr_class-1))/2 + i] = dec_values[i];
				free(dec_values);
			}
			ptr_predict_label[instance_index] = predict_label;
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
	if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		printf("Mean squared error = %g (regression)\n",error/total);
		printf("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
		printf("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);

	// label = res[1]
	luaT_pushudata(L,label,"torch.DoubleTensor");

	// return accuracy, mean squared error, squared correlation coefficient
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
}

void svm_exit_with_help()
{
	printf(
		"Usage: [predicted_label, accuracy, decision_values/prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')\n"
		"Parameters:\n"
		"  model: SVM model structure from svmtrain.\n"
		"  libsvm_options:\n"
		"    -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n"
		"Returns:\n"
		"  predicted_label: SVM prediction output vector.\n"
		"  accuracy: a vector with accuracy, mean squared error, squared correlation coefficient.\n"
		"  prob_estimates: If selected, probability estimate vector.\n"
	);
}

static int libsvm_predict(lua_State *L)
{
	int nrhs = lua_gettop(L);
	int prob_estimate_flag = 0;
	struct svm_model *model_;

	if(nrhs > 4 || nrhs < 2)
	{
		svm_exit_with_help();
		return 0;
	}

	{
		const char *error_msg;

		// parse options
		if(nrhs==3)
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
					svm_exit_with_help();
					return 0;
				}
				switch(argv[i-1][1])
				{
					case 'b':
						prob_estimate_flag = atoi(argv[i]);
						break;
					default:
						printf("Unknown option: -%c\n", argv[i-1][1]);
						svm_exit_with_help();
						return 0;
				}
			}
		}

		model_ = Malloc(struct svm_model, 1);
		torch_structure_to_libsvm_model(model_, L);
		lua_pop(L,1);

		if (model_ == NULL)
		{
			printf("Error: can't read model: %s\n", error_msg);
			return 0;
		}

		if(prob_estimate_flag)
		{
			if(svm_check_probability_model(model_)==0)
			{
				printf("Model does not support probabiliy estimates\n");
				svm_free_and_destroy_model(&model_);
				return 0;
			}
		}
		else
		{
			if(svm_check_probability_model(model_)!=0)
				printf("Model supports probability estimates, but disabled in predicton.\n");
		}

		predict(L, model_, prob_estimate_flag);
		// destroy model
		svm_free_and_destroy_model(&model_);
	}

	return 3;
}

static const struct luaL_Reg libsvm_predict_util__ [] = {
  {"predict", libsvm_predict},
  {NULL, NULL}
};


int liblibsvm_predict_init(lua_State *L)
{
  luaL_register(L, "libsvm", libsvm_predict_util__);
  return 1;
}

