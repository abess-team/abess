%module cbess

%{
#include "bess.h"
#define SWIG_FILE_WITH_INIT
%}

%include "numpy.i"

%init %{
import_array();
%}

//void pywrap_bess_lm(double* IN_ARRAY2, int DIM1, int DIM2, double* IN_ARRAY1, int DIM1, int T0, int max_steps, double* IN_ARRAY1, int DIM1, double* IN_ARRAY1, int DIM1, double* OUTPUT, double* ARGOUT_ARRAY1, int DIM1, double* OUTPUT, double* OUTPUT, double* OUTPUT, double* OUTPUT, double* OUTPUT, int* ARGOUT_ARRAY1, int DIM1, bool normal);
//void pywrap_bess_lms(double* IN_ARRAY2, int DIM1, int DIM2, double* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int max_steps, double* IN_ARRAY1, int DIM1, double* IN_ARRAY1, int DIM1, double* OUTPUT, double* ARGOUT_ARRAY1, int DIM1, double* OUTPUT, double* OUTPUT, double* OUTPUT, double* OUTPUT, double* OUTPUT, bool warm_start, bool normal);

void pywrap_bess(double* IN_ARRAY2, int DIM1, int DIM2, double* IN_ARRAY1, int DIM1, int data_type, double* IN_ARRAY1, int DIM1,
                 bool is_normal,
                 int algorithm_type, int model_type, int max_iter, int exchange_num,
                 int path_type, bool is_warm_start,
                 int ic_type, bool is_cv, int K,
                 int *IN_ARRAY1, int DIM1,
                 double* IN_ARRAY1, int DIM1,
                 int * IN_ARRAY1, int DIM1,
                 double* IN_ARRAY1, int DIM1,
                 int s_min, int s_max, int K_max, double epsilon,
                 double lambda_min, double lambda_max, int n_lambda,
                 bool is_screening, int screening_size, int powell_path,
                 int * IN_ARRAY1, int DIM1, double tao,
                 double* ARGOUT_ARRAY1, int DIM1, double* ARGOUT_ARRAY1, int DIM1, double* ARGOUT_ARRAY1, int DIM1, double* ARGOUT_ARRAY1, int DIM1, double* OUTPUT, double* ARGOUT_ARRAY1, int DIM1, double* ARGOUT_ARRAY1, int DIM1, double* ARGOUT_ARRAY1, int DIM1, int* ARGOUT_ARRAY1, int DIM1, int*OUTPUT);

// .i文件里面不能加默认变量

