%module cabess

%{
#include "abess.h"
#define SWIG_FILE_WITH_INIT
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double *IN_ARRAY2, int DIM1, int DIM2) {(double *x, int x_row, int x_col),
                                                (double *y, int y_row, int y_col),
                                                (double *sigma, int sigma_row, int sigma_col)}
%apply (double *IN_ARRAY1, int DIM1) {(double *weight, int weight_len),
                                      (double *lambda_sequence, int lambda_sequence_len)}
%apply (int *IN_ARRAY1, int DIM1) {(int *gindex, int gindex_len),
                                   (int *sequence, int sequence_len),
                                   (int *cv_fold_id, int cv_fold_id_len),
                                   (int *always_select, int always_select_len)}
%apply (double *ARGOUT_ARRAY1, int DIM1) {(double *beta_out, int beta_out_len),
                                          (double *coef0_out, int coef0_out_len),
                                          (double *train_loss_out, int train_loss_out_len),
                                          (double *ic_out, int ic_out_len),
                                          (double *aic_out, int aic_out_len),
                                          (double *bic_out, int bic_out_len),
                                          (double *gic_out, int gic_out_len)}
%apply (int *ARGOUT_ARRAY1, int DIM1) {(int *A_out, int A_out_len)}
%apply (double *OUTPUT) {(double *nullloss_out)}
%apply (int *OUTPUT) {(int *l_out)}

void pywrap_abess(double *x, int x_row, int x_col, double *y, int y_row, int y_col, int n, int p, double *weight, int weight_len, double *sigma, int sigma_row, int sigma_col,
                  bool is_normal,
                  int algorithm_type, int model_type, int max_iter, int exchange_num,
                  int path_type, bool is_warm_start,
                  int ic_type, double ic_coef, bool is_cv, int Kfold,
                  int *gindex, int gindex_len,
                  int *sequence, int sequence_len,
                  double *lambda_sequence, int lambda_sequence_len,
                  int *cv_fold_id, int cv_fold_id_len,
                  int s_min, int s_max, int K_max, double epsilon,
                  double lambda_min, double lambda_max, int n_lambda,
                  int screening_size, int powell_path,
                  int *always_select, int always_select_len, double tau,
                  int primary_model_fit_max_iter, double primary_model_fit_epsilon,
                  bool early_stop, bool approximate_Newton,
                  int thread,
                  bool covariance_update,
                  bool sparse_matrix,
                  int splicing_type,
                  int sub_search,
                  int pca_num,
                  double *beta_out, int beta_out_len, double *coef0_out, int coef0_out_len, double *train_loss_out,
                  int train_loss_out_len, double *ic_out, int ic_out_len, double *nullloss_out, double *aic_out,
                  int aic_out_len, double *bic_out, int bic_out_len, double *gic_out, int gic_out_len, int *A_out,
                  int A_out_len, int *l_out);
