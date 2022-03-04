// #define R_BUILD
#ifdef R_BUILD

#include <Rcpp.h>
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;

#else

#include <Eigen/Eigen>

#include "List.h"

#endif

#include <iostream>
#include <vector>

#include "Algorithm.h"
#include "AlgorithmGLM.h"
#include "AlgorithmPCA.h"
#include "utilities.h"
#include "workflow.h"

typedef Eigen::Triplet<double> triplet;

using namespace Eigen;
using namespace std;

// [[Rcpp::export]]
List abessGLM_API(Eigen::MatrixXd x, Eigen::MatrixXd y, int n, int p, int normalize_type, Eigen::VectorXd weight,
                  int algorithm_type, int model_type, int max_iter, int exchange_num, int path_type, bool is_warm_start,
                  int ic_type, double ic_coef, int Kfold, Eigen::VectorXi sequence, Eigen::VectorXd lambda_seq,
                  int s_min, int s_max, double lambda_min, double lambda_max, int nlambda, int screening_size,
                  Eigen::VectorXi g_index, Eigen::VectorXi always_select, int primary_model_fit_max_iter,
                  double primary_model_fit_epsilon, bool early_stop, bool approximate_Newton, int thread,
                  bool covariance_update, bool sparse_matrix, int splicing_type, int sub_search,
                  Eigen::VectorXi cv_fold_id, Eigen::VectorXi A_init) {
#ifdef _OPENMP
    // Eigen::initParallel();
    int max_thread = omp_get_max_threads();
    if (thread == 0 || thread > max_thread) {
        thread = max_thread;
    }

    Eigen::setNbThreads(thread);
    omp_set_num_threads(thread);

#endif
    int algorithm_list_size = max(thread, Kfold);
    vector<Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::MatrixXd> *> algorithm_list_uni_dense(
        algorithm_list_size);
    vector<Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> *> algorithm_list_mul_dense(
        algorithm_list_size);
    vector<Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::SparseMatrix<double>> *>
        algorithm_list_uni_sparse(algorithm_list_size);
    vector<Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::SparseMatrix<double>> *>
        algorithm_list_mul_sparse(algorithm_list_size);

    for (int i = 0; i < algorithm_list_size; i++) {
        if (!sparse_matrix) {
            if (model_type == 1) {
                abessLm<Eigen::MatrixXd> *temp = new abessLm<Eigen::MatrixXd>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->covariance_update = covariance_update;
                algorithm_list_uni_dense[i] = temp;
            } else if (model_type == 2) {
                abessLogistic<Eigen::MatrixXd> *temp = new abessLogistic<Eigen::MatrixXd>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->approximate_Newton = approximate_Newton;
                algorithm_list_uni_dense[i] = temp;
            } else if (model_type == 3) {
                abessPoisson<Eigen::MatrixXd> *temp = new abessPoisson<Eigen::MatrixXd>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->approximate_Newton = approximate_Newton;
                algorithm_list_uni_dense[i] = temp;
            } else if (model_type == 4) {
                abessCox<Eigen::MatrixXd> *temp = new abessCox<Eigen::MatrixXd>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->approximate_Newton = approximate_Newton;
                algorithm_list_uni_dense[i] = temp;
            } else if (model_type == 5) {
                abessMLm<Eigen::MatrixXd> *temp = new abessMLm<Eigen::MatrixXd>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->covariance_update = covariance_update;
                algorithm_list_mul_dense[i] = temp;
            } else if (model_type == 6) {
                abessMultinomial<Eigen::MatrixXd> *temp = new abessMultinomial<Eigen::MatrixXd>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->approximate_Newton = approximate_Newton;
                algorithm_list_mul_dense[i] = temp;
            } else if (model_type == 8) {
                abessGamma<Eigen::MatrixXd> *temp = new abessGamma<Eigen::MatrixXd>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->approximate_Newton = approximate_Newton;
                algorithm_list_uni_dense[i] = temp;
            } else if (model_type == 9) {
                abessOrdinal<Eigen::MatrixXd> *temp = new abessOrdinal<Eigen::MatrixXd>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                algorithm_list_mul_dense[i] = temp;
            }
        } else {
            if (model_type == 1) {
                abessLm<Eigen::SparseMatrix<double>> *temp = new abessLm<Eigen::SparseMatrix<double>>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->covariance_update = covariance_update;
                algorithm_list_uni_sparse[i] = temp;
            } else if (model_type == 2) {
                abessLogistic<Eigen::SparseMatrix<double>> *temp = new abessLogistic<Eigen::SparseMatrix<double>>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->approximate_Newton = approximate_Newton;
                algorithm_list_uni_sparse[i] = temp;
            } else if (model_type == 3) {
                abessPoisson<Eigen::SparseMatrix<double>> *temp = new abessPoisson<Eigen::SparseMatrix<double>>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->approximate_Newton = approximate_Newton;
                algorithm_list_uni_sparse[i] = temp;
            } else if (model_type == 4) {
                abessCox<Eigen::SparseMatrix<double>> *temp = new abessCox<Eigen::SparseMatrix<double>>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->approximate_Newton = approximate_Newton;
                algorithm_list_uni_sparse[i] = temp;
            } else if (model_type == 5) {
                abessMLm<Eigen::SparseMatrix<double>> *temp = new abessMLm<Eigen::SparseMatrix<double>>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->covariance_update = covariance_update;
                algorithm_list_mul_sparse[i] = temp;
            } else if (model_type == 6) {
                abessMultinomial<Eigen::SparseMatrix<double>> *temp = new abessMultinomial<Eigen::SparseMatrix<double>>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->approximate_Newton = approximate_Newton;
                algorithm_list_mul_sparse[i] = temp;
            } else if (model_type == 8) {
                abessGamma<Eigen::SparseMatrix<double>> *temp = new abessGamma<Eigen::SparseMatrix<double>>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                temp->approximate_Newton = approximate_Newton;
                algorithm_list_uni_sparse[i] = temp;
            } else if (model_type == 9) {
                abessOrdinal<Eigen::SparseMatrix<double>> *temp = new abessOrdinal<Eigen::SparseMatrix<double>>(
                    algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                    is_warm_start, exchange_num, always_select, splicing_type, sub_search);
                algorithm_list_mul_sparse[i] = temp;
            }
        }
    }

    // parameter list
    Parameters parameters(sequence, lambda_seq, s_min, s_max);

    List out_result;
    if (!sparse_matrix) {
        if (y.cols() == 1) {
            Eigen::VectorXd y_vec = y.col(0).eval();

            out_result = abessWorkflow<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::MatrixXd>(
                x, y_vec, n, p, normalize_type, weight, algorithm_type, path_type, is_warm_start, ic_type, ic_coef,
                Kfold, parameters, screening_size, g_index, early_stop, thread, sparse_matrix, cv_fold_id, A_init,
                algorithm_list_uni_dense);
        } else {
            out_result = abessWorkflow<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd>(
                x, y, n, p, normalize_type, weight, algorithm_type, path_type, is_warm_start, ic_type, ic_coef, Kfold,
                parameters, screening_size, g_index, early_stop, thread, sparse_matrix, cv_fold_id, A_init,
                algorithm_list_mul_dense);
        }
    } else {
        Eigen::SparseMatrix<double> sparse_x(n, p);

        // std::vector<triplet> tripletList;
        // tripletList.reserve(x.rows());
        // for (int i = 0; i < x.rows(); i++)
        // {
        //   tripletList.push_back(triplet(int(x(i, 1)), int(x(i, 2)), x(i, 0)));
        // }
        // sparse_x.setFromTriplets(tripletList.begin(), tripletList.end());

        sparse_x.reserve(x.rows());
        for (int i = 0; i < x.rows(); i++) {
            sparse_x.insert(int(x(i, 1)), int(x(i, 2))) = x(i, 0);
        }
        sparse_x.makeCompressed();

        if (y.cols() == 1) {
            Eigen::VectorXd y_vec = y.col(0).eval();

            out_result = abessWorkflow<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::SparseMatrix<double>>(
                sparse_x, y_vec, n, p, normalize_type, weight, algorithm_type, path_type, is_warm_start, ic_type,
                ic_coef, Kfold, parameters, screening_size, g_index, early_stop, thread, sparse_matrix, cv_fold_id,
                A_init, algorithm_list_uni_sparse);
        } else {
            out_result = abessWorkflow<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::SparseMatrix<double>>(
                sparse_x, y, n, p, normalize_type, weight, algorithm_type, path_type, is_warm_start, ic_type, ic_coef,
                Kfold, parameters, screening_size, g_index, early_stop, thread, sparse_matrix, cv_fold_id, A_init,
                algorithm_list_mul_sparse);
        }
    }

    for (int i = 0; i < algorithm_list_size; i++) {
        delete algorithm_list_uni_dense[i];
        delete algorithm_list_mul_dense[i];
        delete algorithm_list_uni_sparse[i];
        delete algorithm_list_mul_sparse[i];
    }

    return out_result;
};

// [[Rcpp::export]]
List abessPCA_API(Eigen::MatrixXd x, int n, int p, int normalize_type, Eigen::VectorXd weight, Eigen::MatrixXd sigma,
                  int max_iter, int exchange_num, int path_type, bool is_warm_start, int ic_type, double ic_coef,
                  int Kfold, Eigen::MatrixXi sequence, int s_min, int s_max, int screening_size,
                  Eigen::VectorXi g_index, Eigen::VectorXi always_select, bool early_stop, int thread,
                  bool sparse_matrix, int splicing_type, int sub_search, Eigen::VectorXi cv_fold_id, int pca_num,
                  Eigen::VectorXi A_init) {
    /* this function for abessPCA only (model_type == 7) */

#ifdef _OPENMP
    // Eigen::initParallel();
    int max_thread = omp_get_max_threads();
    if (thread == 0 || thread > max_thread) {
        thread = max_thread;
    }

    Eigen::setNbThreads(thread);
    omp_set_num_threads(thread);
#endif
    int model_type = 7, algorithm_type = 6;
    Eigen::VectorXd lambda_seq = Eigen::VectorXd::Zero(1);
    int lambda_min = 0, lambda_max = 0, nlambda = 100;
    int primary_model_fit_max_iter = 1;
    double primary_model_fit_epsilon = 1e-3;
    int pca_n = -1;
    screening_size = -1;
    sub_search = 0;
    if (!sparse_matrix && n != x.rows()) {
        pca_n = n;
        n = x.rows();
    }
    Eigen::VectorXd y_vec = Eigen::VectorXd::Zero(n);

    //////////////////// function generate_algorithm_pointer() ////////////////////////////
    int algorithm_list_size = max(thread, Kfold);
    vector<Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::MatrixXd> *> algorithm_list_uni_dense(
        algorithm_list_size);
    vector<Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::SparseMatrix<double>> *>
        algorithm_list_uni_sparse(algorithm_list_size);
    for (int i = 0; i < algorithm_list_size; i++) {
        if (!sparse_matrix) {
            abessPCA<Eigen::MatrixXd> *temp = new abessPCA<Eigen::MatrixXd>(
                algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                is_warm_start, exchange_num, always_select, splicing_type, sub_search);
            temp->is_cv = Kfold > 1;
            temp->pca_n = pca_n;
            temp->sigma = sigma;
            algorithm_list_uni_dense[i] = temp;
        } else {
            abessPCA<Eigen::SparseMatrix<double>> *temp = new abessPCA<Eigen::SparseMatrix<double>>(
                algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                is_warm_start, exchange_num, always_select, splicing_type, sub_search);
            temp->is_cv = Kfold > 1;
            temp->pca_n = pca_n;
            temp->sigma = sigma;
            algorithm_list_uni_sparse[i] = temp;
        }
    }

    // call `abessWorkflow` for result
#ifdef R_BUILD
    List out_result(pca_num);
#else
    List out_result;
#endif
    List out_result_next;
    int num = 0;
    
    if (!sparse_matrix) {
        while (num++ < pca_num) {
            int pca_support_size_num = sequence.col(num - 1).sum();
            Eigen::VectorXi pca_support_size(pca_support_size_num);
            // map sequence matrix to support.size
            int non_zero_num = 0;
            for (int i = 0; i < sequence.rows(); i++) {
                if (sequence(i, num - 1) == 1) {
                    pca_support_size(non_zero_num++) = i + 1;
                }
            }

            // parameter list
            Parameters parameters(pca_support_size, lambda_seq, s_min, s_max);

            out_result_next = abessWorkflow<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::MatrixXd>(
                x, y_vec, n, p, normalize_type, weight, algorithm_type, path_type, is_warm_start, ic_type, ic_coef,
                Kfold, parameters, screening_size, g_index, early_stop, thread, sparse_matrix, cv_fold_id, A_init,
                algorithm_list_uni_dense);
            Eigen::VectorXd beta_next;
#ifdef R_BUILD
            beta_next = out_result_next["beta"];
#else
            out_result_next.get_value_by_name("beta", beta_next);
#endif
            if (num == 1) {
#ifdef R_BUILD
                if (pca_num > 1) {
                    out_result(0) = out_result_next;
                } else {
                    out_result = out_result_next;
                }
#else
                out_result = out_result_next;
#endif
            } else {
#ifdef R_BUILD
                // Eigen::MatrixXd beta_new(p, num);
                // Eigen::VectorXd temp = out_result["beta"];
                // Eigen::Map<Eigen::MatrixXd> temp2(temp.data(), p, num - 1);
                // beta_new << temp2, beta_next;
                // out_result["beta"] = beta_new;
                out_result(num - 1) = out_result_next;
#else
                out_result.combine_beta(beta_next);
#endif
            }

            if (num < pca_num) {
                Eigen::MatrixXd temp = beta_next * beta_next.transpose();
                if (Kfold > 1) {
                    x -= x * temp;
                } else {
                    Eigen::MatrixXd temp1 = temp * sigma;
                    sigma += temp1 * temp - temp1 - temp1.transpose();
                    for (int i = 0; i < algorithm_list_size; i++) {
                        abessPCA<Eigen::MatrixXd> *pca_model =
                            dynamic_cast<abessPCA<Eigen::MatrixXd> *>(algorithm_list_uni_dense[i]);
                        if (pca_model) {
                            // cout << "update Sigma"<<endl;
                            pca_model->sigma = sigma;
                        }
                    }
                }
            }
        }
    } else {
        Eigen::SparseMatrix<double> sparse_x(n, p);

        // std::vector<triplet> tripletList;
        // tripletList.reserve(x.rows());
        // for (int i = 0; i < x.rows(); i++)
        // {
        //   tripletList.push_back(triplet(int(x(i, 1)), int(x(i, 2)), x(i, 0)));
        // }
        // sparse_x.setFromTriplets(tripletList.begin(), tripletList.end());

        sparse_x.reserve(x.rows());
        for (int i = 0; i < x.rows(); i++) {
            sparse_x.insert(int(x(i, 1)), int(x(i, 2))) = x(i, 0);
        }
        sparse_x.makeCompressed();

        while (num++ < pca_num) {
            int pca_support_size_num = sequence.col(num - 1).sum();
            Eigen::VectorXi pca_support_size(pca_support_size_num);
            // map sequence matrix to support.size
            int non_zero_num = 0;
            for (int i = 0; i < sequence.rows(); i++) {
                if (sequence(i, num - 1) == 1) {
                    pca_support_size(non_zero_num++) = i + 1;
                }
            }

            // parameter list
            Parameters parameters(pca_support_size, lambda_seq, s_min, s_max);

            out_result_next = abessWorkflow<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::SparseMatrix<double>>(
                sparse_x, y_vec, n, p, normalize_type, weight, algorithm_type, path_type, is_warm_start, ic_type,
                ic_coef, Kfold, parameters, screening_size, g_index, early_stop, thread, sparse_matrix, cv_fold_id,
                A_init, algorithm_list_uni_sparse);
            Eigen::VectorXd beta_next;
#ifdef R_BUILD
            beta_next = out_result_next["beta"];
#else
            out_result_next.get_value_by_name("beta", beta_next);
#endif
            if (num == 1) {
#ifdef R_BUILD
                if (pca_num > 1) {
                    out_result(0) = out_result_next;
                } else {
                    out_result = out_result_next;
                }
#else
                out_result = out_result_next;
#endif
            } else {
#ifdef R_BUILD
                // Eigen::MatrixXd beta_new(p, num);
                // Eigen::VectorXd temp = out_result["beta"];
                // Eigen::Map<Eigen::MatrixXd> temp2(temp.data(), p, num - 1);
                // beta_new << temp2, beta_next;
                // out_result["beta"] = beta_new;
                out_result(num - 1) = out_result_next;
#else
                out_result.combine_beta(beta_next);
#endif
            }

            // update for next PCA
            if (num < pca_num) {
                Eigen::MatrixXd temp = beta_next * beta_next.transpose();
                if (Kfold > 1) {
                    sparse_x = sparse_x - sparse_x * temp;
                } else {
                    Eigen::MatrixXd temp1 = temp * sigma;
                    sigma += temp1 * temp - temp1 - temp1.transpose();
                    for (int i = 0; i < algorithm_list_size; i++) {
                        abessPCA<Eigen::SparseMatrix<double>> *pca_model =
                            dynamic_cast<abessPCA<Eigen::SparseMatrix<double>> *>(algorithm_list_uni_sparse[i]);
                        if (pca_model) {
                            // cout << "update Sigma"<<endl;
                            pca_model->sigma = sigma;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < algorithm_list_size; i++) {
        delete algorithm_list_uni_dense[i];
        delete algorithm_list_uni_sparse[i];
    }
    return out_result;
};

// [[Rcpp::export]]
List abessRPCA_API(Eigen::MatrixXd x, int n, int p, int max_iter, int exchange_num, int path_type, bool is_warm_start,
                   int ic_type, double ic_coef, Eigen::VectorXi sequence,
                   Eigen::VectorXd lambda_seq,  // rank of L
                   int s_min, int s_max, double lambda_min, double lambda_max, int nlambda, int screening_size,
                   int primary_model_fit_max_iter, double primary_model_fit_epsilon, Eigen::VectorXi g_index,
                   Eigen::VectorXi always_select, bool early_stop, int thread, bool sparse_matrix, int splicing_type,
                   int sub_search, Eigen::VectorXi A_init) {
#ifdef _OPENMP
    // Eigen::initParallel();
    int max_thread = omp_get_max_threads();
    if (thread == 0 || thread > max_thread) {
        thread = max_thread;
    }

    Eigen::setNbThreads(thread);
    omp_set_num_threads(thread);

#endif
    int model_type = 10, algorithm_type = 6;
    int Kfold = 1;
    int normalize_type = 0;
    Eigen::VectorXi cv_fold_id = Eigen::VectorXi::Zero(0);
    Eigen::VectorXd weight = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd y_vec = Eigen::VectorXd::Zero(n);

    int algorithm_list_size = max(thread, Kfold);
    vector<Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::MatrixXd> *> algorithm_list_uni_dense(
        algorithm_list_size);
    vector<Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::SparseMatrix<double>> *>
        algorithm_list_uni_sparse(algorithm_list_size);

    for (int i = 0; i < algorithm_list_size; i++) {
        if (!sparse_matrix) {
            algorithm_list_uni_dense[i] = new abessRPCA<Eigen::MatrixXd>(
                algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                is_warm_start, exchange_num, always_select, splicing_type, sub_search);
        } else {
            algorithm_list_uni_sparse[i] = new abessRPCA<Eigen::SparseMatrix<double>>(
                algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon,
                is_warm_start, exchange_num, always_select, splicing_type, sub_search);
        }
    }

    // parameter list
    Parameters parameters(sequence, lambda_seq, s_min, s_max);

    List out_result;
    if (!sparse_matrix) {
        out_result = abessWorkflow<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::MatrixXd>(
            x, y_vec, n, p, normalize_type, weight, algorithm_type, path_type, is_warm_start, ic_type, ic_coef, Kfold,
            parameters, screening_size, g_index, early_stop, thread, sparse_matrix, cv_fold_id, A_init,
            algorithm_list_uni_dense);

    } else {
        Eigen::SparseMatrix<double> sparse_x(n, p);

        // std::vector<triplet> tripletList;
        // tripletList.reserve(x.rows());
        // for (int i = 0; i < x.rows(); i++)
        // {
        //   tripletList.push_back(triplet(int(x(i, 1)), int(x(i, 2)), x(i, 0)));
        // }
        // sparse_x.setFromTriplets(tripletList.begin(), tripletList.end());

        sparse_x.reserve(x.rows());
        for (int i = 0; i < x.rows(); i++) {
            sparse_x.insert(int(x(i, 1)), int(x(i, 2))) = x(i, 0);
        }
        sparse_x.makeCompressed();

        out_result = abessWorkflow<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::SparseMatrix<double>>(
            sparse_x, y_vec, n, p, normalize_type, weight, algorithm_type, path_type, is_warm_start, ic_type, ic_coef,
            Kfold, parameters, screening_size, g_index, early_stop, thread, sparse_matrix, cv_fold_id, A_init,
            algorithm_list_uni_sparse);
    }

    for (int i = 0; i < algorithm_list_size; i++) {
        delete algorithm_list_uni_dense[i];
        delete algorithm_list_uni_sparse[i];
    }

    return out_result;
}
