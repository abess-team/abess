#ifndef SRC_PYWRAP_H
#define SRC_PYWRAP_H

void pywrap_abess(double *x, int x_row, int x_col, double *y, int y_row, int n, int p, int normalize_type, int y_col, double *weight, int weight_len, double *sigma, int sigma_row, int sigma_col,
                  int algorithm_type, int model_type, int max_iter, int exchange_num,
                  int path_type, bool is_warm_start,
                  int ic_type, double ic_coef, int K,
                  int *gindex, int gindex_len,
                  int *sequence, int sequence_len,
                  double *lambda_sequence, int lambda_sequence_len,
                  int *cv_fold_id, int cv_fold_id_len,
                  int s_min, int s_max, 
                  double lambda_min, double lambda_max, int n_lambda,
                  int screening_size, 
                  int *always_select, int always_select_len, 
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

#endif // SRC_PYWRAP_H