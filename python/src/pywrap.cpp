#include "api.h"
#include "utilities.h"

void pywrap_GLM(double *x, int x_row, int x_col, double *y, int y_row, int y_col, double *weight, int weight_len, 
				int n, int p, int normalize_type, 
				int algorithm_type, int model_type, int max_iter, int exchange_num,
				int path_type, bool is_warm_start,
				int ic_type, double ic_coef, int Kfold,
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
				double *beta_out, int beta_out_len, double *coef0_out, int coef0_out_len, 
				double *train_loss_out, int train_loss_out_len, double *ic_out, int ic_out_len)
{
	Eigen::MatrixXd x_Mat;
	Eigen::MatrixXd y_Mat;
	Eigen::VectorXd weight_Vec;
	Eigen::VectorXi gindex_Vec;
	Eigen::VectorXi sequence_Vec;
	Eigen::VectorXd lambda_sequence_Vec;
	Eigen::VectorXi always_select_Vec;
	Eigen::VectorXi cv_fold_id_Vec;

	x_Mat = Pointer2MatrixXd(x, x_row, x_col);
	y_Mat = Pointer2MatrixXd(y, y_row, y_col);
	weight_Vec = Pointer2VectorXd(weight, weight_len);
	gindex_Vec = Pointer2VectorXi(gindex, gindex_len);
	sequence_Vec = Pointer2VectorXi(sequence, sequence_len);
	lambda_sequence_Vec = Pointer2VectorXd(lambda_sequence, lambda_sequence_len);
	always_select_Vec = Pointer2VectorXi(always_select, always_select_len);
	cv_fold_id_Vec = Pointer2VectorXi(cv_fold_id, cv_fold_id_len);

	List mylist = abessGLM_API(x_Mat, y_Mat, n, p, normalize_type, weight_Vec, 
								algorithm_type, model_type, max_iter, exchange_num,
								path_type, is_warm_start,
								ic_type, ic_coef, Kfold,
								sequence_Vec,
								lambda_sequence_Vec,
								s_min, s_max, 
								lambda_min, lambda_max, n_lambda,
								screening_size, 
								gindex_Vec,
								always_select_Vec, 
								primary_model_fit_max_iter, primary_model_fit_epsilon,
								early_stop, approximate_Newton,
								thread,
								covariance_update,
								sparse_matrix,
								splicing_type,
								sub_search,
								cv_fold_id_Vec);

	if (y_col == 1)
	{
		Eigen::VectorXd beta;
		double coef0 = 0;
		double train_loss = 0;
		double ic = 0;
		mylist.get_value_by_name("beta", beta);
		mylist.get_value_by_name("coef0", coef0);
		mylist.get_value_by_name("train_loss", train_loss);
		mylist.get_value_by_name("ic", ic);

		VectorXd2Pointer(beta, beta_out);
		*coef0_out = coef0;
		*train_loss_out = train_loss;
		*ic_out = ic;
	}
	else
	{
		Eigen::MatrixXd beta;
		Eigen::VectorXd coef0;
		double train_loss = 0;
		double ic = 0;
		mylist.get_value_by_name("beta", beta);
		mylist.get_value_by_name("coef0", coef0);
		mylist.get_value_by_name("train_loss", train_loss);
		mylist.get_value_by_name("ic", ic);

		MatrixXd2Pointer(beta, beta_out);
		VectorXd2Pointer(coef0, coef0_out);
		train_loss_out[0] = train_loss;
		ic_out[0] = ic;
	}
}

void pywrap_PCA(double *x, int x_row, int x_col, double *weight, int weight_len, 
				int n, int p, int normalize_type, 
				double *sigma, int sigma_row, int sigma_col,
				int max_iter, int exchange_num,
				int path_type, bool is_warm_start,
				int ic_type, double ic_coef, int Kfold,
				int *gindex, int gindex_len,
				int *sequence, int sequence_row, int sequence_col,
				int *cv_fold_id, int cv_fold_id_len,
				int s_min, int s_max, 
				int screening_size, 
				int *always_select, int always_select_len, 
				bool early_stop, 
				int thread,
				bool sparse_matrix,
				int splicing_type,
				int sub_search,
				int pca_num,
				double *beta_out, int beta_out_len, double *coef0_out, int coef0_out_len, 
				double *train_loss_out, int train_loss_out_len, double *ic_out, int ic_out_len)
{
	Eigen::MatrixXd x_Mat;
	Eigen::MatrixXd sigma_Mat;
	Eigen::MatrixXi sequence_Mat;
	Eigen::VectorXd weight_Vec;
	Eigen::VectorXi gindex_Vec;
	Eigen::VectorXd lambda_sequence_Vec;
	Eigen::VectorXi always_select_Vec;
	Eigen::VectorXi cv_fold_id_Vec;

	x_Mat = Pointer2MatrixXd(x, x_row, x_col);
	sigma_Mat = Pointer2MatrixXd(sigma, sigma_row, sigma_col);
	sequence_Mat = Pointer2MatrixXi(sequence, sequence_row, sequence_col);
	weight_Vec = Pointer2VectorXd(weight, weight_len);
	gindex_Vec = Pointer2VectorXi(gindex, gindex_len);
	always_select_Vec = Pointer2VectorXi(always_select, always_select_len);
	cv_fold_id_Vec = Pointer2VectorXi(cv_fold_id, cv_fold_id_len);

	List mylist = abessPCA_API(x_Mat, n, p, normalize_type, weight_Vec, sigma_Mat,
								max_iter, exchange_num,
								path_type, is_warm_start,
								ic_type, ic_coef, Kfold,
								sequence_Mat,
								s_min, s_max,
								screening_size, 
								gindex_Vec,
								always_select_Vec, 
								early_stop, 
								thread,
								sparse_matrix,
								splicing_type,
								sub_search,
								cv_fold_id_Vec,
								pca_num);

	Eigen::MatrixXd beta;
	if (pca_num == 1)
	{
		beta.resize(p, 1);
		mylist.get_value_by_name("beta", beta);
		VectorXd2Pointer(beta, beta_out);
	}else{
		beta.resize(p, pca_num);
		mylist.get_value_by_name("beta", beta);
		MatrixXd2Pointer(beta, beta_out);
	}

	double coef0 = 0;
	double train_loss = 0;
	double ic = 0;
	mylist.get_value_by_name("coef0", coef0);
	mylist.get_value_by_name("train_loss", train_loss);
	mylist.get_value_by_name("ic", ic);
	*coef0_out = coef0;
	*train_loss_out = train_loss;
	*ic_out = ic;

	// if (pca_num == 1)
	// {
	// 	Eigen::VectorXd beta;
	// 	double coef0 = 0;
	// 	double train_loss = 0;
	// 	double ic = 0;
	// 	mylist.get_value_by_name("beta", beta);
	// 	mylist.get_value_by_name("coef0", coef0);
	// 	mylist.get_value_by_name("train_loss", train_loss);
	// 	mylist.get_value_by_name("ic", ic);

	// 	VectorXd2Pointer(beta, beta_out);
	// 	*coef0_out = coef0;
	// 	*train_loss_out = train_loss;
	// 	*ic_out = ic;
	// }
	// else
	// {
	// 	Eigen::MatrixXd beta;
	// 	double coef0;
	// 	double train_loss = 0;
	// 	double ic = 0;
	// 	mylist.get_value_by_name("beta", beta);
	// 	mylist.get_value_by_name("coef0", coef0);
	// 	mylist.get_value_by_name("train_loss", train_loss);
	// 	mylist.get_value_by_name("ic", ic);

	// 	MatrixXd2Pointer(beta, beta_out);
	// 	*coef0_out = coef0;
	// 	train_loss_out = train_loss;
	// 	ic_out = ic;
	// }
}

void pywrap_RPCA(double *x, int x_row, int x_col, 
				int n, int p, int normalize_type, 
				int max_iter, int exchange_num,
				int path_type, bool is_warm_start,
				int ic_type, double ic_coef, 
				int *gindex, int gindex_len,
				int *sequence, int sequence_len,
				double *lambda_sequence, int lambda_sequence_len,
				int s_min, int s_max, 
				double lambda_min, double lambda_max, int n_lambda,
				int screening_size, 
				int *always_select, int always_select_len, 
				int primary_model_fit_max_iter, double primary_model_fit_epsilon,
				bool early_stop, 
				int thread,
				bool sparse_matrix,
				int splicing_type,
				int sub_search,
				double *beta_out, int beta_out_len, double *coef0_out, int coef0_out_len, 
				double *train_loss_out, int train_loss_out_len, double *ic_out, int ic_out_len)
{
	Eigen::MatrixXd x_Mat;
	Eigen::VectorXi sequence_Vec;
	Eigen::VectorXi gindex_Vec;
	Eigen::VectorXd lambda_sequence_Vec;
	Eigen::VectorXi always_select_Vec;
	Eigen::VectorXi cv_fold_id_Vec;

	x_Mat = Pointer2MatrixXd(x, x_row, x_col);
	sequence_Vec = Pointer2VectorXi(sequence, sequence_len);
	lambda_sequence_Vec = Pointer2VectorXd(lambda_sequence, lambda_sequence_len);
	gindex_Vec = Pointer2VectorXi(gindex, gindex_len);
	always_select_Vec = Pointer2VectorXi(always_select, always_select_len);

	List mylist = abessRPCA_API(x_Mat, n, p, 
								max_iter, exchange_num,
								path_type, is_warm_start,
								ic_type, ic_coef, 
								sequence_Vec,
								lambda_sequence_Vec,
								s_min, s_max, 
								lambda_min, lambda_max, n_lambda,
								screening_size,
								primary_model_fit_max_iter,
								primary_model_fit_epsilon,
								gindex_Vec,
								always_select_Vec, 
								early_stop, 
								thread,
								sparse_matrix,
								splicing_type,
								sub_search);

	Eigen::VectorXd beta;
	double coef0 = 0;
	double train_loss = 0;
	double ic = 0;
	mylist.get_value_by_name("beta", beta);
	mylist.get_value_by_name("coef0", coef0);
	mylist.get_value_by_name("train_loss", train_loss);
	mylist.get_value_by_name("ic", ic);

	VectorXd2Pointer(beta, beta_out);
	*coef0_out = coef0;
	*train_loss_out = train_loss;
	*ic_out = ic;
}
