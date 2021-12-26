/*****************************************************************************
 *  OpenST Basic tool library                                                 *
 *  Copyright (C) 2021 Kangkang Jiang  jiangkk3@mail2.sysu.edu.cn                         *
 *                                                                            *
 *  This file is part of OST.                                                 *
 *                                                                            *
 *  This program is free software; you can redistribute it and/or modify      *
 *  it under the terms of the GNU General Public License version 3 as         *
 *  published by the Free Software Foundation.                                *
 *                                                                            *
 *  You should have received a copy of the GNU General Public License         *
 *  along with OST. If not, see <http://www.gnu.org/licenses/>.               *
 *                                                                            *
 *  Unless required by applicable law or agreed to in writing, software       *
 *  distributed under the License is distributed on an "AS IS" BASIS,         *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 *  See the License for the specific language governing permissions and       *
 *  limitations under the License.                                            *
 *                                                                            *
 *  @file     abess.h                                                         *
 *  @brief    The main function of abess fremework                            *
 *                                                                            *
 *                                                                            *
 *  @author   Kangkang Jiang                                                  *
 *  @email    jiangkk3@mail2.sysu.edu.cn                                      *
 *  @version  0.0.1                                                           *
 *  @date     2021-07-31                                                      *
 *  @license  GNU General Public License (GPL)                                *
 *                                                                            *
 *----------------------------------------------------------------------------*
 *  Remark         : Description                                              *
 *----------------------------------------------------------------------------*
 *  Change History :                                                          *
 *  <Date>     | <Version> | <Author>       | <Description>                   *
 *----------------------------------------------------------------------------*
 *  2021/07/31 | 0.0.1     | Kangkang Jiang | First version                   *
 *----------------------------------------------------------------------------*
 *                                                                            *
 *****************************************************************************/

#ifndef SRC_API_H
#define SRC_API_H

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

/**
 * @brief The main function of abess fremework
 * @param X                             Training data.
 * @param y                             Target values. Will be cast to X's dtype if necessary.
 *                                      For linear regression problem, y should be a $n \time 1$ numpy array with type
 * \code{double}. For classification problem, \code{y} should be a $n \time 1$ numpy array with values \code{0} or
 * \code{1}. For count data, \code{y} should be a $n \time 1$ numpy array of non-negative integer. Note that, for
 * multivariate problem, \code{y} can also be a matrix shaped $n \time M$, where $M > 1$.
 * @param n                             Sample size.
 * @param p                             Variable dimension.
 * @param weight                        Individual weights for each sample.
 * @param sigma                         Sample covariance matrix. For PCA problem under Kfold=1, it should be given as
 * input, instead of X.
 * @param normalize_type                Type of normalization on X before fitting the algorithm. If normalize_type=0,
 * normalization will not be used.
 * @param algorithm_type                Algorithm type.
 * @param model_type                    Model type.
 * @param max_iter                      Maximum number of iterations taken for the splicing algorithm to converge.
 *                                      Due to the limitation of loss reduction, the splicing algorithm must be able to
 * converge. The number of iterations is only to simplify the implementation.
 * @param exchange_num                  Max exchange variable num for the splicing algorithm.
 * @param path_type                     The method to be used to select the optimal support size.
 *                                      For path_type = 1, we solve the best subset selection problem for each size in
 * support_size. For path_type = 2, we solve the best subset selection problem with support size ranged in (s_min,
 * s_max), where the specific support size to be considered is determined by golden section.
 * @param is_warm_start                 When tuning the optimal parameter combination, whether to use the last solution
 * as a warm start to accelerate the iterative convergence of the splicing algorithm.
 * @param ic_type                       The type of criterion for choosing the support size. Available options are
 * "gic", "ebic", "bic", "aic".
 * @param Kfold                         The folds number to use the Cross-validation method. If Kfold=1,
 * Cross-validation will not be used.
 * @param sequence                      An integer vector representing the alternative support sizes. Only used for
 * path_type = 1.
 * @param s_min                         The lower bound of golden-section-search for sparsity searching. Only used for
 * path_type = 2.
 * @param s_max                         The higher bound of golden-section-search for sparsity searching. Only used for
 * path_type = 2.
 * @param thread                        Max number of multithreads. If thread = 0, the program will use the maximum
 * number supported by the device.
 * @param screening_size                Screen the variables first and use the chosen variables in abess process.
 *                                      The number of variables remaining after screening. It should be an integer
 * smaller than p. If screen_size = -1, screening will not be used.
 * @param g_index                       The group index for each variable.
 * @param always_select                 An array contains the indexes of variables we want to consider in the model.
 * @param primary_model_fit_max_iter    The maximal number of iteration in `primary_model_fit()` (in Algorithm.h).
 * @param primary_model_fit_epsilon     The epsilon (threshold) of iteration in `primary_model_fit()` (in Algorithm.h).
 * @param splicing_type                 The type of splicing in `fit()` (in Algorithm.h).
 *                                      "0" for decreasing by half, "1" for decresing by one.
 * @param sub_search                    The number of inactive sets that are split when splicing. It should be a
 * positive integer.
 * @return result list.
 */
List abessGLM_API(Eigen::MatrixXd x, Eigen::MatrixXd y, int n, int p, int normalize_type, Eigen::VectorXd weight,
                  int algorithm_type, int model_type, int max_iter, int exchange_num, int path_type, bool is_warm_start,
                  int ic_type, double ic_coef, int Kfold, Eigen::VectorXi sequence, Eigen::VectorXd lambda_seq,
                  int s_min, int s_max, double lambda_min, double lambda_max, int nlambda, int screening_size,
                  Eigen::VectorXi g_index, Eigen::VectorXi always_select, int primary_model_fit_max_iter,
                  double primary_model_fit_epsilon, bool early_stop, bool approximate_Newton, int thread,
                  bool covariance_update, bool sparse_matrix, int splicing_type, int sub_search,
                  Eigen::VectorXi cv_fold_id, Eigen::VectorXi A_init);

List abessPCA_API(Eigen::MatrixXd x, int n, int p, int normalize_type, Eigen::VectorXd weight, Eigen::MatrixXd sigma,
                  int max_iter, int exchange_num, int path_type, bool is_warm_start, int ic_type, double ic_coef,
                  int Kfold, Eigen::MatrixXi sequence, int s_min, int s_max, int screening_size,
                  Eigen::VectorXi g_index, Eigen::VectorXi always_select, bool early_stop, int thread,
                  bool sparse_matrix, int splicing_type, int sub_search, Eigen::VectorXi cv_fold_id, int pca_num,
                  Eigen::VectorXi A_init);

List abessRPCA_API(Eigen::MatrixXd x, int n, int p, int max_iter, int exchange_num, int path_type, bool is_warm_start,
                   int ic_type, double ic_coef, Eigen::VectorXi sequence,
                   Eigen::VectorXd lambda_seq,  // rank of L
                   int s_min, int s_max, double lambda_min, double lambda_max, int nlambda, int screening_size,
                   int primary_model_fit_max_iter, double primary_model_fit_epsilon, Eigen::VectorXi g_index,
                   Eigen::VectorXi always_select, bool early_stop, int thread, bool sparse_matrix, int splicing_type,
                   int sub_search, Eigen::VectorXi A_init);

#endif  // SRC_API_H
