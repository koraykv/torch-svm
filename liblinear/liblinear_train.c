#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "TH.h"
#include "luaT.h"

#include "liblinear/linear.h"
#include "linear_model_torch.h"

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

#define max_(a,b) (a>=b ? a : b)
#define min_(a,b) (a<=b ? a : b)

// liblinear arguments
struct parameter param;		// set by parse_command_line
struct problem prob;		// set by read_problem
struct model *model_;
struct feature_node *x_space;
int cross_validation_flag;
int nr_fold;
double bias;

void print_string_default(const char *s) {printf("%s",s);}

void print_null(const char *s) {}

static void exit_with_help()
{
	printf(
	"Usage: model = train(training_data, 'liblinear_options');\n"
	"liblinear_options:\n"
	"-s type : set type of solver (default 1)\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"	
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- multi-class support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"	11 -- L2-regularized L2-loss epsilon support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss epsilon support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss epsilon support vector regression (dual)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n" 
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" 
	"		where f is the primal function and pos/neg are # of\n" 
	"		positive/negative data (default 0.01)\n"
	"	-s 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n" 
	"	-s 1, 3, 4 and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
	"		where f is the dual function (default 0.1)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
}

double do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);
	double retval = 0.0;

	cross_validation(&prob,&param,nr_fold,target);
	if(param.solver_type == L2R_L2LOSS_SVR || 
	   param.solver_type == L2R_L1LOSS_SVR_DUAL || 
	   param.solver_type == L2R_L2LOSS_SVR_DUAL)
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
	void (*print_func)(const char *) = print_string_default;	// default printing to matlab display

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation_flag = 0;
	bias = -1;

	int nrhs = lua_gettop(L);

	if(nrhs < 1)
		return 1;

	// put options in argv[]
	if(nrhs > 1)
	{
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
		if(i>=argc && argv[i-1][1] != 'q') // since option -q has no parameter
			return 1;
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'B':
				bias = atof(argv[i]);
				break;
			case 'v':
				cross_validation_flag = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					printf("n-fold cross validation: n must >= 2\n");
					return 1;
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			default:
				printf("unknown option\n");
				return 1;
		}
	}

	set_print_string_function(print_func);

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR: 
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL: 
			case L2R_L1LOSS_SVC_DUAL: 
			case MCSVM_CS: 
			case L2R_LR_DUAL: 
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC: 
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
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
	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct feature_node*, prob.l);
	x_space = Malloc(struct feature_node, elements);
	prob.bias=bias;

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
				if (prob.bias >= 0)
				{
					x_space[j].index = max_index+1;
					x_space[j].value = prob.bias;
					j++;
				}
				x_space[j++].index = -1;
			}
			lua_pop(L,1);
		}
		lua_pop(L,1);
	}
	if (prob.bias >= 0)
		prob.n = max_index+1;
	else
		prob.n = max_index;

	return 0;
}

// Interface function of torch
static int liblinear_train( lua_State *L )
{

	const char *error_msg;
	// fix random seed to have same results for each run
	// (for cross validation)
	srand(1);

	int nrhs = lua_gettop(L);

	// Transform the input Matrix to libsvm format
	if(nrhs >= 1 && nrhs < 3)
	{
		int err=0;

		if(parse_command_line(L))
		{
			printf("parsing failed\n");
			exit_with_help();
			destroy_param(&param);
			return 0;
		}

		err = read_problem_sparse(L);

		// train's original code
		error_msg = check_parameter(&prob, &param);

		if(err || error_msg)
		{
			if (error_msg != NULL)
				printf("Error: %s\n", error_msg);
			destroy_param(&param);
			free(prob.y);
			free(prob.x);
			free(x_space);
			return 0;
		}

		if(cross_validation_flag)
		{
			lua_pushnumber(L,do_cross_validation());
		}
		else
		{
			model_ = train(&prob, &param);
			liblinear_model_to_torch_structure(L, model_);
			free_and_destroy_model(&model_);
		}
		destroy_param(&param);
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
	return 0;
}

static const struct luaL_Reg liblinear_util__ [] = {
  {"train", liblinear_train},
  {NULL, NULL}
};


int libliblinear_train_init(lua_State *L)
{
  luaL_register(L, "liblinear", liblinear_util__);
  return 1;
}
