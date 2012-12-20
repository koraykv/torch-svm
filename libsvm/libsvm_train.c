#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "TH.h"
#include "luaT.h"

#include "libsvm/svm.h"
#include "svm_model_torch.h"

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#define max_(a,b) (a>=b ? a : b)
#define min_(a,b) (a<=b ? a : b)


// svm arguments
struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;

void print_null(const char *s) {}
void print_string_default(const char *s) {printf("%s",s);}

void exit_with_help()
{
	printf(
	"Usage: model = svmtrain(training_data, 'libsvm_options');\n"
	"libsvm_options:\n"
	"-s svm_type : set type of SVM (default 0)\n"
	"	0 -- C-SVC\n"
	"	1 -- nu-SVC\n"
	"	2 -- one-class SVM\n"
	"	3 -- epsilon-SVR\n"
	"	4 -- nu-SVR\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_instance_matrix)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n : n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
}



double do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);
	double retval = 0.0;

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
		retval = total_error/prob.l;
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
		retval = 100.0*total_correct/prob.l;
	}
	free(target);
	return retval;
}

// nrhs should be 3
int parse_command_line(lua_State *L)
{
	int i, argc = 1;
	char *argv[CMD_LEN/2];
	char cmd[CMD_LEN];
	void (*print_func)(const char *) = print_string_default;

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation = 0;

	int nrhs = lua_gettop(L);

	if(nrhs < 1)
		return 1;

	if(nrhs >= 2)
	{
		// put options in argv[]
	        size_t slen;
		const char *tcmd = lua_tolstring(L,2,&slen);
		strncpy(cmd,tcmd,slen);
		if((argv[argc] = strtok((char*)cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;

		lua_pop(L,1);
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		if(i>=argc && argv[i-1][1] != 'q')	// since option -q has no parameter
			return 1;
		switch(argv[i-1][1])
		{
			case 's':
				param.svm_type = atoi(argv[i]);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					printf("n-fold cross validation: n must >= 2\n");
					return 1;
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			default:
				printf("Unknown option -%c\n", argv[i-1][1]);
				return 1;
		}
	}

	svm_set_print_string_function(print_func);

	return 0;
}

// read in a problem (in svmlight format)
int read_problem_dense(lua_State *L)
{
	int i, j, k;
	int elements, max_index, sc, label_vector_row_num;
	float *samples, *labels;

	prob.x = NULL;
	prob.y = NULL;
	x_space = NULL;

	lua_pushnumber(L,1);
	lua_gettable(L,-2);
	THFloatTensor *tlabels = luaT_checkudata(L,1,"torch.FloatTensor");
	lua_pushnumber(L,2);
	lua_gettable(L,-2);
	THFloatTensor *tsamples = luaT_checkudata(L,2,"torch.FloatTensor");

	labels = THFloatTensor_data(tlabels);
	samples = THFloatTensor_data(tsamples);
	sc = (int)tsamples->size[1];

	elements = 0;
	// the number of instance
	prob.l = (int)tsamples->size[0];
	label_vector_row_num = (int)tlabels->size[0];

	if(label_vector_row_num!=prob.l)
	{
		printf("Length of label vector does not match # of instances.\n");
		return -1;
	}

	if(param.kernel_type == PRECOMPUTED)
		elements = prob.l * (sc + 1);
	else
	{
		for(i = 0; i < prob.l; i++)
		{
			for(k = 0; k < sc; k++)
				if(samples[i * sc + k] != 0)
					elements++;
			// count the '-1' element
			elements++;
		}
	}

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node, elements);

	max_index = sc;
	j = 0;
	for(i = 0; i < prob.l; i++)
	{
		prob.x[i] = &x_space[j];
		prob.y[i] = labels[i];

		for(k = 0; k < sc; k++)
		{
			if(param.kernel_type == PRECOMPUTED || samples[k * prob.l + i] != 0)
			{
				x_space[j].index = k + 1;
				x_space[j].value = samples[i*sc + k];
				j++;
			}
		}
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				printf("Wrong input format: sample_serial_number out of range\n");
				return -1;
			}
		}

	return 0;
}


int read_problem_sparse(lua_State *L)
{

	luaL_argcheck(L,lua_istable(L,1),1,"Expecting table in read_problem_sparse");
	int label_vector_row_num = lua_objlen(L,1);
	int num_samples = 0;
	int max_index = 0;
	int elements;

	prob.l = label_vector_row_num;

	int i;
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
				num_samples += (int)THIntTensor_nElement(indices);
				max_index = max_(max_index,THIntTensor_get1d(indices,indices->size[0]-1));
				// lua_pushnumber(L,2);lua_gettable(L,-2);
				// THFloatTensor *indices = luaT_checkudata(L,-1,"torch.FloatTensor");
				lua_pop(L,1);
			}
			lua_pop(L,1);
		}
		lua_pop(L,1);
	}

	elements = num_samples + prob.l*2;
	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node, elements);

	int j = 0;
	for (i=0; i<prob.l; i++)
	{
		prob.x[i] = &x_space[j];
		// get the table elem
		lua_pushnumber(L,i+1);
		lua_gettable(L,-2);
		if (!lua_istable(L,-1))
			luaL_error(L,"expected table at index %d while reading data\n",i+1);
		{
			// get label
			lua_pushnumber(L,1);lua_gettable(L,-2);
			prob.y[i] = (double)lua_tonumber(L,-1);
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
					x_space[j].index = indices_data[k];
					x_space[j].value = vals_data[k];
					j++;
				}
				x_space[j++].index = -1;
			}
			lua_pop(L,1);
		}
		lua_pop(L,1);
	}
	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	return 0;
}


// Interface function of torch
static int libsvm_train( lua_State *L )
{
	const char *error_msg;

	// fix random seed to have same results for each run
	// (for cross validation and probability estimation)
	srand(1);

	int nrhs = lua_gettop(L);

	// Transform the input Matrix to libsvm format
	if(nrhs >= 1 && nrhs < 3)
	{
		int err;

		if(parse_command_line(L))
		{
			printf("parsing failed\n");
			exit_with_help();
			svm_destroy_param(&param);
			return 0;
		}

		if(param.kernel_type == PRECOMPUTED)
		{
			err = read_problem_dense(L);
		}
		else
			err = read_problem_sparse(L);

		// svmtrain's original code
		error_msg = svm_check_parameter(&prob, &param);

		if(err || error_msg)
		{
			if (error_msg != NULL)
				printf("Error: %s\n", error_msg);
			svm_destroy_param(&param);
			free(prob.y);
			free(prob.x);
			free(x_space);
			return 0;
		}

		if(cross_validation)
		{
			lua_pushnumber(L,do_cross_validation());
		}
		else
		{
			model = svm_train(&prob, &param);
			libsvm_model_to_torch_structure(L, model);
			svm_free_and_destroy_model(&model);
		}
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
		return 1;
	}
	else
	{
		exit_with_help();
		return 0;
	}
}

static const struct luaL_Reg libsvm_util__ [] = {
  {"train", libsvm_train},
  {NULL, NULL}
};


int liblibsvm_train_init(lua_State *L)
{
  luaL_register(L, "libsvm", libsvm_util__);
  return 1;
}
