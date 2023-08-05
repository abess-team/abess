#ifndef SRC_ALGORITHMGLM_H
#define SRC_ALGORITHMGLM_H

#include "Algorithm.h"

using namespace std;

template <class T1, class T2, class T3, class T4>
class _abessGLM : public Algorithm<T1, T2, T3, T4> {
   public:
    _abessGLM(int algorithm_type, int model_type, int max_iter, int primary_model_fit_max_iter,
              double primary_model_fit_epsilon, bool warm_start, int exchange_num, Eigen::VectorXi always_select,
              int splicing_type, int sub_search)
        : Algorithm<T1, T2, T3, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter,
                                               primary_model_fit_epsilon, warm_start, exchange_num, always_select,
                                               splicing_type, sub_search){};
    ~_abessGLM(){};

    bool approximate_Newton;    // use approximate Newton method or not.
    bool fit_intercept = true;  // whether to consider intercept (coef0) or not.
    double newton_step = 1;

    // Eigen::MatrixXd G;  // Gradient
    // Eigen::MatrixXd H;  // Hessian

    double Hessian_range[2] = {1e-7, 1e7};        // the range of acceptable value in Hessian
    double Xbeta_range[2] = {-DBL_MAX, DBL_MAX};  // the range of acceptable value for X * beta

    /* --- TO BE IMPLEMENTED IN CHILD CLASS --- */
    virtual Eigen::MatrixXd gradient_core(T4 &X_full, T1 &y, Eigen::VectorXd &weights, T2 &beta_full) {
        // the gradient matrix can be expressed as G = X^T * A,
        // returns the gradient core A
        return Eigen::MatrixXd::Zero(beta_full.rows(), beta_full.cols());
    };
    virtual Eigen::VectorXd hessian_core(T4 &X_full, T1 &y, Eigen::VectorXd &weights, T2 &beta_full) {
        // the hessian matrix can be expressed as H = X^T * D * X,
        // returns the (diagnal values of) diagnal matrix D.
        return Eigen::VectorXd::Ones(X_full.rows());
    };
    virtual T1 inv_link_function(T4 &X_full, T2 &beta_full) {
        // denote the link function g() as g(y) = X^T * beta,
        // return its inverse g^{-1}(X, beta), i.e. predicted y
        return T1(X_full * beta_full);
    }
    virtual T1 log_probability(T4 &X_full, T2 &beta_full, T1 &y) {
        // returns log P(y | X, beta)
        return T1(Eigen::MatrixXd::Zero(y.rows(), y.cols()));
    }
    virtual bool null_model(T1 &y, Eigen::VectorXd &weights, T3 &coef0) {
        // returns a null model,
        // i.e. given only y, fit an intercept
        return true;
    }
    /* ---------------------------------------- */

    /* --- CAN BE RE-IMPLEMENTED IN CHILD CLASS --- */
    virtual Eigen::MatrixXd gradient(T4 &X_full, T1 &y, Eigen::VectorXd &weights, T2 &beta_full) {
        // returns gradient matrix
        return X_full.transpose() * this->gradient_core(X_full, y, weights, beta_full);
    }
    virtual Eigen::MatrixXd hessian(T4 &X_full, T1 &y, Eigen::VectorXd &weights, T2 &beta_full) {
        // returns hessian matrix
        Eigen::VectorXd H_core = this->hessian_core(X_full, y, weights, beta_full);
        Eigen::VectorXd H_diag(X_full.cols());
        for (int i = 0; i < X_full.cols(); i++) {
            H_diag(i) = X_full.col(i).eval().cwiseProduct(H_core).dot(X_full.col(i).eval());
            trunc(H_diag(i), Hessian_range);
        }
        return Eigen::MatrixXd(H_diag.asDiagonal());
    };
    virtual bool primary_model_fit(T4 &X, T1 &y, Eigen::VectorXd &weights, T2 &beta, T3 &coef0, double loss0,
                                   Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size) {
        if (X.cols() == 0) return null_model(y, weights, coef0);

        if (this->approximate_Newton) {
            // Fitting method 1: Approximate Newton
            return this->_approx_newton_fit(X, y, weights, beta, coef0, loss0, A, g_index, g_size);
        } else {
            // Fitting method 2: Iteratively Reweighted Least Squares
            return this->_IRLS_fit(X, y, weights, beta, coef0, loss0, A, g_index, g_size);
        }
        trunc(beta, this->beta_range);
        return true;
    };
    virtual double loss_function(T4 &X, T1 &y, Eigen::VectorXd &weights, T2 &beta, T3 &coef0, Eigen::VectorXi &A,
                                 Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, double lambda) {
        int n = X.rows();
        int p = X.cols();
        int M = y.cols();

        T4 X_full;
        T2 beta_full;
        add_constant_column(X_full, X, true);
        combine_beta_coef0(beta_full, beta, coef0, true);

        Eigen::VectorXd log_proba = this->log_probability(X_full, beta_full, y);
        return -log_proba.dot(weights) + lambda * beta.cwiseAbs2().sum();
    };
    virtual void sacrifice(T4 &X, T4 &XA, T1 &y, T2 &beta, T2 &beta_A, T3 &coef0, Eigen::VectorXi &A,
                           Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index,
                           Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd,
                           Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num) {
        int p = X.cols();
        int n = X.rows();
        int M = y.cols();
        int A_size = A.size();
        int I_size = I.size();
        T4 X_A_full;
        T2 beta_A_full;
        add_constant_column(X_A_full, XA, true);
        combine_beta_coef0(beta_A_full, beta_A, coef0, true);

        Eigen::MatrixXd G = X.transpose() * this->gradient_core(X_A_full, y, weights, beta_A_full);
        Eigen::VectorXd H_core = this->hessian_core(X_A_full, y, weights, beta_A_full);
        G -= 2 * this->lambda_level * beta;

        Eigen::MatrixXd betabar = Eigen::MatrixXd::Zero(p, M);
        Eigen::MatrixXd dbar = Eigen::MatrixXd::Zero(p, M);

        for (int i = 0; i < N; i++) {
            // focus on X in group i
            T4 XG = X.middleCols(g_index(i), g_size(i));
            T4 XG_new = XG;
            for (int j = 0; j < g_size(i); j++) {
                XG_new.col(j) = XG.col(j).cwiseProduct(H_core);
            }
            Eigen::MatrixXd XGbar =
                XG_new.transpose() * XG + 2 * this->lambda_level * Eigen::MatrixXd::Identity(g_size(i), g_size(i));
            Eigen::MatrixXd phiG;
            XGbar.sqrt().evalTo(phiG);
            Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(g_size(i), g_size(i)));
            // compute sacrifice for each variable
            betabar.block(g_index(i), 0, g_size(i), M) = phiG * beta.block(g_index(i), 0, g_size(i), M);
            dbar.block(g_index(i), 0, g_size(i), M) = invphiG * G.block(g_index(i), 0, g_size(i), M);
        }

        // backward sacrifice (group)
        for (int i = 0; i < A_size; i++) {
            bd(A(i)) = betabar.block(g_index(A[i]), 0, g_size(A[i]), M).squaredNorm() / g_size(A(i));
        }
        // forward sacrifice (group)
        for (int i = 0; i < I_size; i++) {
            bd(I(i)) = dbar.block(g_index(I[i]), 0, g_size(I[i]), M).squaredNorm() / g_size(I(i));
        }
    };
    virtual double effective_number_of_parameter(T4 &X, T4 &XA, T1 &y, Eigen::VectorXd &weights, T2 &beta, T2 &beta_A,
                                                 T3 &coef0) {
        if (XA.cols() == 0) return 0;
        if (this->lambda_level == 0) return XA.cols();

        T4 X_A_full;
        T2 beta_A_full;
        add_constant_column(X_A_full, XA, true);
        combine_beta_coef0(beta_A_full, beta_A, coef0, true);

        Eigen::VectorXd H_core = this->hessian_core(X_A_full, y, weights, beta_A_full);

        T4 XA_new = XA;
        for (int j = 0; j < XA.cols(); j++) {
            XA_new.col(j) = XA.col(j).cwiseProduct(H_core);
        }
        Eigen::MatrixXd XGbar = XA_new.transpose() * XA;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> adjoint_eigen_solver(XGbar);

        double enp = 0.;
        for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++) {
            enp += adjoint_eigen_solver.eigenvalues()(i) / (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
        }

        return enp;
    };
    /* -------------------------------------------- */

    /* --- BUILT-IN FITTING METHOD --- */
    bool _approx_newton_fit(T4 &X, T1 &y, Eigen::VectorXd &weights, T2 &beta, T3 &coef0, double loss0,
                            Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size) {
        // Fitting method 1: Approximate Newton
        int n = X.rows();
        int p = X.cols();
        int M = y.cols();

        T4 X_full;
        T2 beta_full;
        add_constant_column(X_full, X, this->fit_intercept);
        combine_beta_coef0(beta_full, beta, coef0, this->fit_intercept);

        // initialize
        double step = this->newton_step;
        double loss = this->loss_function(X, y, weights, beta, coef0, A, g_index, g_size, this->lambda_level);
        for (int iter = 0; iter < this->primary_model_fit_max_iter; iter++) {
            // get Gradient/Hessian (no penalty)
            Eigen::MatrixXd G = this->gradient(X_full, y, weights, beta_full);
            Eigen::MatrixXd H = this->hessian(X_full, y, weights, beta_full);
            // update direction (add penalty)
            Eigen::MatrixXd direction = G;
            for (int i = 0; i < direction.rows(); i++) {
                double hii = H(i, i);
                if (!this->fit_intercept || i > 0) {
                    direction.row(i) -= 2 * this->lambda_level * beta_full.row(i).eval();
                    hii += 2 * this->lambda_level;
                }
                direction.row(i) /= hii;
            }
            // update beta
            T2 beta_new = beta_full + step * T2(direction);
            extract_beta_coef0(beta_new, beta, coef0, this->fit_intercept);
            double loss_new = this->loss_function(X, y, weights, beta, coef0, A, g_index, g_size, this->lambda_level);
            // step down if loss_new > loss
            while (loss_new > loss && step > this->primary_model_fit_epsilon) {
                step /= 2;
                beta_new = beta_full + step * direction;
                extract_beta_coef0(beta_new, beta, coef0, this->fit_intercept);
                loss_new = this->loss_function(X, y, weights, beta, coef0, A, g_index, g_size, this->lambda_level);
            }

            // Update beta_full if loss decrease
            if (loss_new > loss) {
                break;
            } else {
                beta_full = beta_new;
            }

            // Early stop 1: expected final loss is too large
            double expected_loss = loss - (this->primary_model_fit_max_iter - iter) * (loss - loss_new);
            if (expected_loss >= loss0 + this->tau) break;
            // Early stop 2: step is too small
            if (step < this->primary_model_fit_epsilon) break;
        }

        // extract beta and coef0
        extract_beta_coef0(beta_full, beta, coef0, this->fit_intercept);
        return true;
    };

    bool _IRLS_fit(T4 &X, T1 &y, Eigen::VectorXd &weights, T2 &beta, T3 &coef0, double loss0, Eigen::VectorXi &A,
                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size) {
        // Fitting method 2: Iteratively Reweighted Least Squares
        int n = X.rows();
        int p = X.cols();
        int M = y.cols();

        // X_full: add constant col to X
        T4 X_full;
        T2 beta_full;
        add_constant_column(X_full, X, this->fit_intercept);
        combine_beta_coef0(beta_full, beta, coef0, this->fit_intercept);

        // initialize
        T4 X_new(X_full);
        double loss = this->loss_function(X, y, weights, beta, coef0, A, g_index, g_size, this->lambda_level);
        for (int iter = 0; iter < this->primary_model_fit_max_iter; iter++) {
            Eigen::VectorXd D = this->hessian_core(X_full, y, weights, beta_full);
            // reweight
            T1 y_pred = this->inv_link_function(X_full, beta_full);
            T1 Z = y - y_pred;
            array_quotient(Z, D, 1);
            Z += X_full * beta_full;
            for (int i = 0; i < X_full.cols(); i++) {
                X_new.col(i) = X_full.col(i).cwiseProduct(D);
            }
            // update beta
            Eigen::MatrixXd lambda_one = Eigen::MatrixXd::Identity(X_full.cols(), X_full.cols());
            if (this->fit_intercept) lambda_one(0, 0) = 0;
            Eigen::MatrixXd XTX = 2 * this->lambda_level * lambda_one + X_new.transpose() * X_full;
            beta_full = XTX.ldlt().solve(X_new.transpose() * Z);

            // compute new loss
            extract_beta_coef0(beta_full, beta, coef0, this->fit_intercept);
            double loss_new = this->loss_function(X, y, weights, beta, coef0, A, g_index, g_size, this->lambda_level);

            // Early stop 1: expected final loss is too large
            double expected_loss = loss - (this->primary_model_fit_max_iter - iter) * (loss - loss_new);
            if (expected_loss >= loss0 + this->tau) break;
            // Early stop 2: step is too small
            double step = (loss - loss_new) / (0.1 + loss_new);
            if (step < this->primary_model_fit_epsilon) break;
            // Early stop 3: acceptable precision
            if (loss_new < min(1e-3, this->tau)) break;

            // update loss
            loss = loss_new;
        }

        // extract beta and coef0
        extract_beta_coef0(beta_full, beta, coef0, this->fit_intercept);
        return true;
    };
    /* ------------------------------- */
};

template <class T4>
class abessLogistic : public _abessGLM<Eigen::VectorXd, Eigen::VectorXd, double, T4> {
   public:
    abessLogistic(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10,
                  double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5,
                  Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0)
        : _abessGLM<Eigen::VectorXd, Eigen::VectorXd, double, T4>::_abessGLM(
              algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start,
              exchange_num, always_select, splicing_type, sub_search){};

    ~abessLogistic(){};

    double Xbeta_range[2] = {-30, 30};
    double PiPj_range[2] = {0.001, 1};  // Pi * Pj

    Eigen::MatrixXd gradient_core(T4 &X_full, Eigen::VectorXd &y, Eigen::VectorXd &weights,
                                  Eigen::VectorXd &beta_full) {
        Eigen::VectorXd Pi = this->inv_link_function(X_full, beta_full);
        Eigen::VectorXd G = (y - Pi).cwiseProduct(weights);
        return Eigen::MatrixXd(G);
    };

    Eigen::VectorXd hessian_core(T4 &X_full, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta_full) {
        Eigen::VectorXd Pi = this->inv_link_function(X_full, beta_full);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(X_full.rows());
        Eigen::VectorXd W = Pi.cwiseProduct(one - Pi).cwiseProduct(weights);
        trunc(W, PiPj_range);
        return W;
    };

    Eigen::VectorXd inv_link_function(T4 &X_full, Eigen::VectorXd &beta_full) {
        Eigen::VectorXd eta = X_full * beta_full;
        trunc(eta, Xbeta_range);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(eta.size());
        Eigen::VectorXd expeta = eta.array().exp();
        Eigen::VectorXd Pi = expeta.array() / (one + expeta).array();
        return Pi;
    };

    Eigen::VectorXd log_probability(T4 &X_full, Eigen::VectorXd &beta_full, Eigen::VectorXd &y) {
        Eigen::VectorXd ones = Eigen::VectorXd::Ones(y.size());
        Eigen::VectorXd P_1 = this->inv_link_function(X_full, beta_full);
        Eigen::VectorXd logP_1 = P_1.array().log();
        Eigen::VectorXd logP_0 = (ones - P_1).array().log();
        return y.cwiseProduct(logP_1) + (ones - y).cwiseProduct(logP_0);
    }

    bool null_model(Eigen::VectorXd &y, Eigen::VectorXd &weights, double &coef0) {
        coef0 = -log(1 / y.mean() - 1);
        return true;
    }
};

template <class T4>
class abessLm : public _abessGLM<Eigen::VectorXd, Eigen::VectorXd, double, T4> {
   public:
    bool clear = true;
    Eigen::VectorXd XTy;                            /*X.transpose() * y */
    Eigen::VectorXd XTone;                          /* X.transpose() * Eigen::MatrixXd::one() */
    Eigen::Matrix<Eigen::MatrixXd, -1, -1> PhiG;    /* PhiG for linear model. */
    Eigen::Matrix<Eigen::MatrixXd, -1, -1> invPhiG; /* invPhiG for linear model. */
    Eigen::VectorXd XTy_U;
    Eigen::VectorXd XTone_U;
    Eigen::Matrix<Eigen::MatrixXd, -1, -1> PhiG_U;
    Eigen::Matrix<Eigen::MatrixXd, -1, -1> invPhiG_U;
    Eigen::Matrix<T4, -1, -1> group_XTX;    /* XTX. */
    bool covariance_update;                 /* use covairance update mathod or not. */
    Eigen::VectorXd **covariance = NULL;    /* covairance matrix. */
    Eigen::VectorXi covariance_update_flag; /* each variable have updated in covairance matirx. */

    abessLm(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10,
            double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5,
            Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0)
        : _abessGLM<Eigen::VectorXd, Eigen::VectorXd, double, T4>::_abessGLM(
              algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start,
              exchange_num, always_select, splicing_type, sub_search){};

    ~abessLm(){};

    void inital_setting(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXi &g_index,
                        Eigen::VectorXi &g_size, int &N) {
        int n = X.rows(), p = X.cols();
        if (this->clear) {
            this->group_XTX = compute_group_XTX<T4>(X, g_index, g_size, n, p, N);
            if (this->covariance_update) {
                // cout<<"create pointer"<<endl;
                this->covariance = new Eigen::VectorXd *[p];
                this->covariance_update_flag = Eigen::VectorXi::Zero(p);
                this->XTy = X.transpose() * y;
                this->XTone = X.transpose() * Eigen::MatrixXd::Ones(n, 1);
            }
        }

        if (this->clear || this->lambda_change) {
            this->PhiG = Phi(X, g_index, g_size, n, p, N, this->lambda_level, this->group_XTX);
            this->invPhiG = invPhi(this->PhiG, N);
            this->PhiG_U.resize(N, 1);
            this->invPhiG_U.resize(N, 1);
        }

        this->clear = false;
    };

    void clear_setting() {
        this->clear = true;
        // delete pointer
        if (this->covariance_update) {
            // cout<<"delete pointer"<<endl;
            for (int i = 0; i < (this->covariance_update_flag).size(); i++)
                if (this->covariance_update_flag(i) == 1) delete this->covariance[i];
            delete[] this->covariance;
        }
    };

    bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0,
                           double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size) {
        // int n = x.rows();
        // int p = x.cols();
        if (x.cols() == 0) return true;

        // to ensure
        T4 X_full;

        add_constant_column(X_full, x, this->fit_intercept);

        // beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(),
        // X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);
        Eigen::MatrixXd XTX =
            X_full.adjoint() * X_full + this->lambda_level * Eigen::MatrixXd::Identity(X_full.cols(), X_full.cols());
        // if (check_ill_condition(XTX)) return false;
        Eigen::VectorXd XTy = X_full.adjoint() * y;
        Eigen::VectorXd beta_full = XTX.ldlt().solve(XTy);

        extract_beta_coef0(beta_full, beta, coef0, this->fit_intercept);

        trunc(beta, this->beta_range);
        return true;
    };

    double loss_function(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0,
                         Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, double lambda) {
        int n = X.rows();
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
        return (y - X * beta - coef0 * one).array().square().sum() / n / 2. + lambda * beta.cwiseAbs2().sum();
    }

    void mapping_U(Eigen::VectorXi &U, Eigen::VectorXi &U_ind) {
        int N = U.size(), p = U_ind.size();
        if (this->covariance_update)
            for (int i = 0; i < p; i++) {
                this->XTy_U(i) = this->XTy(U_ind(i), 0);
                this->XTone_U(i) = this->XTone(U_ind(i), 0);
            }

        for (int i = 0; i < N; i++) {
            this->PhiG_U(i, 0) = this->PhiG(U(i), 0);
            this->invPhiG_U(i, 0) = this->invPhiG(U(i), 0);
        }
        return;
    }

    Eigen::MatrixXd covariance_update_f_U(Eigen::VectorXi &U_ind, Eigen::VectorXi &A_ind_U) {
        int k = A_ind_U.size(), p = U_ind.size();
        Eigen::MatrixXd cov_A(p, k);

        for (int i = 0; i < k; i++) {
            int Ai = U_ind(A_ind_U(i));
            if (this->covariance_update_flag(Ai) == 0) {
                this->covariance[Ai] = new Eigen::VectorXd;
                *this->covariance[Ai] = (*this->x).transpose() * (*this->x).col(Ai);
                this->covariance_update_flag(Ai) = 1;
            }
            if (p == this->XTy.rows()) {
                cov_A.col(i) = *this->covariance[Ai];
            } else {
                for (int j = 0; j < p; j++) cov_A(j, i) = (*this->covariance[Ai])(U_ind(j));
            }
        }

        return cov_A;
    }

    void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0,
                   Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index,
                   Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U,
                   Eigen::VectorXi &U_ind, int num) {
        int p = X.cols();
        int n = X.rows();

        if (num == 0) {
            if (p == this->XTy.rows()) {
                this->XTy_U = this->XTy;
                this->XTone_U = this->XTone;
                this->PhiG_U = this->PhiG;
                this->invPhiG_U = this->invPhiG;
            } else {
                this->XTy_U.resize(p, 1);
                this->XTone_U.resize(p, 1);
                this->mapping_U(U, U_ind);
            }
        }

        Eigen::VectorXd d;
        if (!this->covariance_update) {
            Eigen::VectorXd one = Eigen::VectorXd::Ones(n);

            if (beta.size() != 0) {
                d = X.adjoint() * (y - XA * beta_A - coef0 * one) / double(n) - 2 * this->lambda_level * beta;
            } else {
                d = X.adjoint() * (y - coef0 * one) / double(n);
            }
        } else {
            Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
            if (beta.size() != 0) {
                Eigen::VectorXd XTXbeta = this->covariance_update_f_U(U_ind, A_ind) * beta_A;
                d = (this->XTy_U - XTXbeta - this->XTone_U * coef0) / double(n) - 2 * this->lambda_level * beta;
            } else {
                Eigen::VectorXd XTonecoef0 = this->XTone_U * coef0;
                d = (this->XTy_U - XTonecoef0) / double(n);
            }
        }

        int A_size = A.size();
        int I_size = I.size();

        Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
        Eigen::MatrixXd phiG, invphiG;

        for (int i = 0; i < N; i++) {
            phiG = this->PhiG_U(i, 0);
            invphiG = this->invPhiG_U(i, 0);
            betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
            dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
        }
        for (int i = 0; i < A_size; i++) {
            bd(A[i]) = betabar.segment(g_index(A[i]), g_size(A[i])).squaredNorm() / g_size(A[i]);
        }
        for (int i = 0; i < I_size; i++) {
            bd(I[i]) = dbar.segment(g_index(I[i]), g_size(I[i])).squaredNorm() / g_size(I[i]);
        }
    }

    double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &weights,
                                         Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0) {
        if (this->lambda_level == 0.) {
            return XA.cols();
        } else {
            if (XA.cols() == 0) return 0.;

            Eigen::MatrixXd XGbar;
            XGbar = XA.transpose() * XA;

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> adjoint_eigen_solver(XGbar);

            double enp = 0.;
            for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++) {
                enp += adjoint_eigen_solver.eigenvalues()(i) /
                       (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
            }

            return enp;
        }
    }
};

template <class T4>
class abessPoisson : public _abessGLM<Eigen::VectorXd, Eigen::VectorXd, double, T4> {
   public:
    abessPoisson(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10,
                 double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5,
                 Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0)
        : _abessGLM<Eigen::VectorXd, Eigen::VectorXd, double, T4>::_abessGLM(
              algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start,
              exchange_num, always_select, splicing_type, sub_search){};

    ~abessPoisson(){};

    double Xbeta_range[2] = {-30, 30};

    Eigen::MatrixXd gradient_core(T4 &X_full, Eigen::VectorXd &y, Eigen::VectorXd &weights,
                                  Eigen::VectorXd &beta_full) {
        Eigen::VectorXd expeta = this->inv_link_function(X_full, beta_full);
        Eigen::VectorXd G = (y - expeta).cwiseProduct(weights);
        return Eigen::MatrixXd(G);
    };

    Eigen::VectorXd hessian_core(T4 &X_full, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta_full) {
        Eigen::VectorXd expeta = this->inv_link_function(X_full, beta_full);
        return expeta.cwiseProduct(weights);
    };

    Eigen::VectorXd inv_link_function(T4 &X_full, Eigen::VectorXd &beta_full) {
        Eigen::VectorXd eta = X_full * beta_full;
        trunc(eta, Xbeta_range);
        Eigen::VectorXd expeta = eta.array().exp();
        return expeta;
    };

    Eigen::VectorXd log_probability(T4 &X_full, Eigen::VectorXd &beta_full, Eigen::VectorXd &y) {
        Eigen::VectorXd Xbeta = X_full * beta_full;
        Eigen::VectorXd Ey = this->inv_link_function(X_full, beta_full);
        return y.cwiseProduct(Xbeta) - Ey;
    }

    bool null_model(Eigen::VectorXd &y, Eigen::VectorXd &weights, double &coef0) {
        coef0 = log(weights.dot(y) / weights.sum());
        return true;
    }
};

template <class T4>
class abessCox : public _abessGLM<Eigen::VectorXd, Eigen::VectorXd, double, T4> {
   public:
    Eigen::MatrixXd cox_hessian; /* hessian matrix for cox model. */
    Eigen::VectorXd cox_g;       /* score function for cox model. */

    abessCox(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10,
             double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5,
             Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0)
        : _abessGLM<Eigen::VectorXd, Eigen::VectorXd, double, T4>::_abessGLM(
              algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start,
              exchange_num, always_select, splicing_type, sub_search){};

    ~abessCox(){};

    double Xbeta_range[2] = {-30, 30};

    void inital_setting(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXi &g_index,
                        Eigen::VectorXi &g_size, int &N) {
        this->cox_g = Eigen::VectorXd::Zero(0);
        this->cox_hessian = Eigen::MatrixXd::Zero(0, 0);
    }

    bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weight, Eigen::VectorXd &beta, double &coef0,
                           double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size) {
        if (x.cols() == 0) {
            coef0 = 0.;
            return true;
        }

        int n = x.rows();
        int p = x.cols();
        Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p, p);
        Eigen::VectorXd theta(n);
        Eigen::MatrixXd one = (Eigen::MatrixXd::Ones(n, n)).triangularView<Eigen::Upper>();
        Eigen::MatrixXd x_theta(n, p);
        Eigen::VectorXd xij_theta(n);
        Eigen::VectorXd cum_theta(n);
        Eigen::VectorXd g(p);
        Eigen::VectorXd beta0 = beta, beta1;
        Eigen::VectorXd cum_eta(n);
        Eigen::VectorXd cum_eta2(n);
        Eigen::VectorXd cum_eta3(n);
        Eigen::MatrixXd h(p, p);
        Eigen::VectorXd eta;

        Eigen::VectorXd d(p);
        double loglik1,
            loglik0 = -this->loss_function(x, y, weight, beta0, coef0, A, g_index, g_size, this->lambda_level);
        // beta = Eigen::VectorXd::Zero(p);

        double step = 1.0;
        int l;
        for (l = 1; l <= this->primary_model_fit_max_iter; l++) {
            eta = x * beta0;
            trunc(eta, Xbeta_range);
            eta = weight.array() * eta.array().exp();
            cum_eta(n - 1) = eta(n - 1);
            for (int k = n - 2; k >= 0; k--) {
                cum_eta(k) = cum_eta(k + 1) + eta(k);
            }
            cum_eta2(0) = (y(0) * weight(0)) / cum_eta(0);
            for (int k = 1; k <= n - 1; k++) {
                cum_eta2(k) = (y(k) * weight(k)) / cum_eta(k) + cum_eta2(k - 1);
            }
            cum_eta3(0) = (y(0) * weight(0)) / pow(cum_eta(0), 2);
            for (int k = 1; k <= n - 1; k++) {
                cum_eta3(k) = (y(k) * weight(k)) / pow(cum_eta(k), 2) + cum_eta3(k - 1);
            }
            h = -cum_eta3.replicate(1, n);
            h = h.cwiseProduct(eta.replicate(1, n));
            h = h.cwiseProduct(eta.replicate(1, n).transpose());
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    h(j, i) = h(i, j);
                }
            }
            h.diagonal() = cum_eta2.cwiseProduct(eta) + h.diagonal();
            // g = x.transpose() * (weight.cwiseProduct(y) - cum_eta2.cwiseProduct(eta));
            // g = g - 2 * this->lambda_level * beta0;

            g = weight.cwiseProduct(y) - cum_eta2.cwiseProduct(eta);

            Eigen::MatrixXd temp = x.transpose() * h * x;
            if (this->approximate_Newton) {
                // d = g.cwiseQuotient((x.transpose() * h * x + 2 * this->lambda_level * lambdamat).diagonal());
                d = (x.transpose() * g - 2 * this->lambda_level * beta0).cwiseQuotient(temp.diagonal());
            } else {
                // d = (x.transpose() * h * x + 2 * this->lambda_level * lambdamat).ldlt().solve(g);
                // if (check_ill_condition(temp)) return false;
                d = temp.ldlt().solve(x.transpose() * g - 2 * this->lambda_level * beta0);
            }

            // theta = x * beta0;
            // for (int i = 0; i < n; i++)
            // {
            //   if (theta(i) > 30)
            //     theta(i) = 30;
            //   else if (theta(i) < -30)
            //     theta(i) = -30;
            // }
            // theta = theta.array().exp();
            // cum_theta = one * theta;
            // x_theta = x.array().colwise() * theta.array();
            // x_theta = one * x_theta;
            // x_theta = x_theta.array().colwise() / cum_theta.array();
            // g = (x - x_theta).transpose() * (weights.cwiseProduct(y));

            // if (this->approximate_Newton)
            // {
            //   Eigen::VectorXd h(p);
            //   for (int k1 = 0; k1 < p; k1++)
            //   {
            //     xij_theta = (theta.cwiseProduct(x.col(k1))).cwiseProduct(x.col(k1));
            //     for (int j = n - 2; j >= 0; j--)
            //     {
            //       xij_theta(j) = xij_theta(j + 1) + xij_theta(j);
            //     }
            //     h(k1) = -(xij_theta.cwiseQuotient(cum_theta) -
            //     x_theta.col(k1).cwiseProduct(x_theta.col(k1))).dot(weights.cwiseProduct(y));
            //   }
            //   d = g.cwiseQuotient(h);
            // }
            // else
            // {
            //   Eigen::MatrixXd h(p, p);
            //   for (int k1 = 0; k1 < p; k1++)
            //   {
            //     for (int k2 = k1; k2 < p; k2++)
            //     {
            //       xij_theta = (theta.cwiseProduct(x.col(k1))).cwiseProduct(x.col(k2));
            //       for (int j = n - 2; j >= 0; j--)
            //       {
            //         xij_theta(j) = xij_theta(j + 1) + xij_theta(j);
            //       }
            //       h(k1, k2) = -(xij_theta.cwiseQuotient(cum_theta) -
            //       x_theta.col(k1).cwiseProduct(x_theta.col(k2))).dot(weights.cwiseProduct(y)); h(k2, k1) = h(k1, k2);
            //     }
            //   }
            //   d = h.ldlt().solve(g);
            // }

            beta1 = beta0 + step * d;

            loglik1 = -this->loss_function(x, y, weight, beta1, coef0, A, g_index, g_size, this->lambda_level);

            while (loglik1 < loglik0 && step > this->primary_model_fit_epsilon) {
                step = step / 2;
                beta1 = beta0 + step * d;
                loglik1 = -this->loss_function(x, y, weight, beta1, coef0, A, g_index, g_size, this->lambda_level);
            }

            bool condition1 =
                -(loglik1 + (this->primary_model_fit_max_iter - l - 1) * (loglik1 - loglik0)) + this->tau > loss0;
            if (condition1) {
                loss0 = -loglik0;
                beta = beta0;
                this->cox_hessian = h;
                this->cox_g = g;

                return true;
            }

            if (loglik1 > loglik0) {
                beta0 = beta1;
                loglik0 = loglik1;
                this->cox_hessian = h;
                this->cox_g = g;
            }

            if (step < this->primary_model_fit_epsilon) {
                loss0 = -loglik0;
                beta = beta0;
                this->cox_hessian = h;
                this->cox_g = g;

                return true;
            }
        }

        beta = beta0;
        trunc(beta, this->beta_range);
        return true;
    };

    double loss_function(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0,
                         Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, double lambda) {
        int n = X.rows();
        Eigen::VectorXd eta = X * beta;
        trunc(eta, Xbeta_range);
        Eigen::VectorXd expeta = eta.array().exp();
        Eigen::VectorXd cum_expeta(n);
        cum_expeta(n - 1) = expeta(n - 1);
        for (int i = n - 2; i >= 0; i--) {
            cum_expeta(i) = cum_expeta(i + 1) + expeta(i);
        }
        Eigen::VectorXd ratio = (expeta.cwiseQuotient(cum_expeta)).array().log();
        double loglik_cox = (ratio.cwiseProduct(y)).dot(weights);

        return -loglik_cox + lambda * beta.cwiseAbs2().sum();
    }

    void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0,
                   Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index,
                   Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U,
                   Eigen::VectorXi &U_ind, int num) {
        int p = X.cols();
        int n = X.rows();

        Eigen::VectorXd d;
        Eigen::MatrixXd h;
        Eigen::VectorXd g;
        if (this->cox_g.size() != 0) {
            h = this->cox_hessian;
            g = this->cox_g;
        } else {
            Eigen::VectorXd cum_eta(n);
            Eigen::VectorXd cum_eta2(n);
            Eigen::VectorXd cum_eta3(n);
            Eigen::VectorXd eta = XA * beta_A;
            for (int i = 0; i <= n - 1; i++) {
                if (eta(i) < -30.0) eta(i) = -30.0;
                if (eta(i) > 30.0) eta(i) = 30.0;
            }
            eta = weights.array() * eta.array().exp();
            cum_eta(n - 1) = eta(n - 1);
            for (int k = n - 2; k >= 0; k--) {
                cum_eta(k) = cum_eta(k + 1) + eta(k);
            }
            cum_eta2(0) = (y(0) * weights(0)) / cum_eta(0);
            for (int k = 1; k <= n - 1; k++) {
                cum_eta2(k) = (y(k) * weights(k)) / cum_eta(k) + cum_eta2(k - 1);
            }
            cum_eta3(0) = (y(0) * weights(0)) / pow(cum_eta(0), 2);
            for (int k = 1; k <= n - 1; k++) {
                cum_eta3(k) = (y(k) * weights(k)) / pow(cum_eta(k), 2) + cum_eta3(k - 1);
            }
            h = -cum_eta3.replicate(1, n);
            h = h.cwiseProduct(eta.replicate(1, n));
            h = h.cwiseProduct(eta.replicate(1, n).transpose());
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    h(j, i) = h(i, j);
                }
            }
            h.diagonal() = cum_eta2.cwiseProduct(eta) + h.diagonal();
            g = weights.cwiseProduct(y) - cum_eta2.cwiseProduct(eta);
        }

        d = X.transpose() * g - 2 * this->lambda_level * beta;

        // Eigen::VectorXd d = X.transpose() * res;
        // Eigen::VectorXd h = weights.array() * pr.array() * (one - pr).array();

        int A_size = A.size();
        int I_size = I.size();

        Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);

        for (int i = 0; i < N; i++) {
            T4 XG = X.middleCols(g_index(i), g_size(i));
            Eigen::MatrixXd XGbar =
                XG.transpose() * h * XG + 2 * this->lambda_level * Eigen::MatrixXd::Identity(g_size(i), g_size(i));

            Eigen::MatrixXd phiG;
            XGbar.sqrt().evalTo(phiG);
            Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(g_size(i), g_size(i)));
            betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
            dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
        }
        for (int i = 0; i < A_size; i++) {
            bd(A[i]) = betabar.segment(g_index(A[i]), g_size(A[i])).squaredNorm() / g_size(A[i]);
        }
        for (int i = 0; i < I_size; i++) {
            bd(I[i]) = dbar.segment(g_index(I[i]), g_size(I[i])).squaredNorm() / g_size(I[i]);
        }
    }

    double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &weights,
                                         Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0) {
        if (this->lambda_level == 0.) {
            return XA.cols();
        } else {
            if (XA.cols() == 0) return 0.;

            // int p = X.cols();
            // int n = X.rows();

            Eigen::VectorXd d;
            Eigen::MatrixXd h;
            Eigen::VectorXd g;
            if (this->cox_g.size() != 0) {
                h = this->cox_hessian;
                // g = this->cox_g;
            }

            // Eigen::VectorXd d = X.transpose() * res - 2 * this->lambda_level * beta;
            // Eigen::VectorXd h = weights.array() * pr.array() * (one - pr).array();

            Eigen::MatrixXd XGbar = XA.transpose() * h * XA;

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> adjoint_eigen_solver(XGbar);

            double enp = 0.;
            for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++) {
                enp += adjoint_eigen_solver.eigenvalues()(i) /
                       (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
            }

            return enp;
        }
    }
};

template <class T4>
class abessMLm : public _abessGLM<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4> {
   public:
    bool clear = true;
    Eigen::MatrixXd XTy;                            /*X.transpose() * y */
    Eigen::MatrixXd XTone;                          /* X.transpose() * Eigen::MatrixXd::one() */
    Eigen::Matrix<Eigen::MatrixXd, -1, -1> PhiG;    /* PhiG for linear model. */
    Eigen::Matrix<Eigen::MatrixXd, -1, -1> invPhiG; /* invPhiG for linear model. */
    Eigen::MatrixXd XTy_U;
    Eigen::MatrixXd XTone_U;
    Eigen::Matrix<Eigen::MatrixXd, -1, -1> PhiG_U;
    Eigen::Matrix<Eigen::MatrixXd, -1, -1> invPhiG_U;
    Eigen::Matrix<T4, -1, -1> group_XTX;    /* XTX. */
    bool covariance_update;                 /* use covairance update mathod or not. */
    Eigen::VectorXd **covariance = NULL;    /* covairance matrix. */
    Eigen::VectorXi covariance_update_flag; /* each variable have updated in covairance matirx. */

    abessMLm(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10,
             double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5,
             Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0)
        : _abessGLM<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4>::_abessGLM(
              algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start,
              exchange_num, always_select, splicing_type, sub_search){};

    ~abessMLm(){};

    void inital_setting(T4 &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::VectorXi &g_index,
                        Eigen::VectorXi &g_size, int &N) {
        int n = X.rows(), p = X.cols();
        if (this->clear) {
            this->group_XTX = compute_group_XTX<T4>(X, g_index, g_size, n, p, N);
            if (this->covariance_update) {
                this->covariance = new Eigen::VectorXd *[p];
                this->covariance_update_flag = Eigen::VectorXi::Zero(p);
                this->XTy = X.transpose() * y;
                this->XTone = X.transpose() * Eigen::MatrixXd::Ones(n, y.cols());
            }
        }

        if (this->clear || this->lambda_change) {
            this->PhiG = Phi(X, g_index, g_size, n, p, N, this->lambda_level, this->group_XTX);
            this->invPhiG = invPhi(this->PhiG, N);
            this->PhiG_U.resize(N, 1);
            this->invPhiG_U.resize(N, 1);
        }

        this->clear = false;
    };

    void clear_setting() {
        this->clear = true;
        // delete pointer
        if (this->covariance_update) {
            for (int i = 0; i < (this->covariance_update_flag).size(); i++)
                if (this->covariance_update_flag(i) == 1) delete this->covariance[i];
            delete[] this->covariance;
        }
    };

    bool primary_model_fit(T4 &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta,
                           Eigen::VectorXd &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index,
                           Eigen::VectorXi &g_size) {
        // beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(),
        // X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);

        int n = x.rows();
        int p = x.cols();
        int M = y.cols();
        if (p == 0) return true;

        // to ensure
        T4 X;
        add_constant_column(X, x, this->fit_intercept);
        // beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(),
        // X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);
        Eigen::MatrixXd XTX = X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols());
        // if (check_ill_condition(XTX)) return false;
        Eigen::MatrixXd beta0 = XTX.ldlt().solve(X.adjoint() * y);

        extract_beta_coef0(beta0, beta, coef0, this->fit_intercept);
        trunc(beta, this->beta_range);
        return true;
        // if (X.cols() == 0)
        // {
        //   // coef0 = y.colwise().sum();
        //   return;
        // }
        //
        // // overload_ldlt(X, X, y, beta);
        // Eigen::MatrixXd XTX = X.transpose() * X;
        // beta = (XTX + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).ldlt().solve(X.transpose()
        // * y);

        // CG
        // ConjugateGradient<T4, Lower | Upper> cg;
        // cg.compute(X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols()));
        // beta = cg.solveWithGuess(X.adjoint() * y, beta);
    };

    double loss_function(T4 &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta,
                         Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size,
                         double lambda) {
        int n = X.rows();
        Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, y.cols());
        return (y - X * beta - array_product(one, coef0)).array().square().sum() / n / 2.0 +
               lambda * beta.cwiseAbs2().sum();
    }

    void mapping_U(Eigen::VectorXi &U, Eigen::VectorXi &U_ind) {
        int N = U.size(), p = U_ind.size(), M = this->XTy.cols();
        if (this->covariance_update)
            for (int i = 0; i < p; i++)
                for (int j = 0; j < M; j++) {
                    this->XTy_U(i, j) = this->XTy(U_ind(i), j);
                    this->XTone_U(i, j) = this->XTone(U_ind(i), j);
                }

        for (int i = 0; i < N; i++) {
            this->PhiG_U(i, 0) = this->PhiG(U(i), 0);
            this->invPhiG_U(i, 0) = this->invPhiG(U(i), 0);
        }
        return;
    }

    Eigen::MatrixXd covariance_update_f_U(Eigen::VectorXi &U_ind, Eigen::VectorXi &A_ind_U) {
        int k = A_ind_U.size(), p = U_ind.size();
        Eigen::MatrixXd cov_A(p, k);

        for (int i = 0; i < k; i++) {
            int Ai = U_ind(A_ind_U(i));
            if (this->covariance_update_flag(Ai) == 0) {
                this->covariance[Ai] = new Eigen::VectorXd;
                *this->covariance[Ai] = (*this->x).transpose() * (*this->x).col(Ai);
                this->covariance_update_flag(Ai) = 1;
            }
            if (p == this->XTy.rows()) {
                cov_A.col(i) = *this->covariance[Ai];
            } else {
                for (int j = 0; j < p; j++) cov_A(j, i) = (*this->covariance[Ai])(U_ind(j));
            }
        }

        return cov_A;
    }

    void sacrifice(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A,
                   Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights,
                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind,
                   Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num) {
        int p = X.cols();
        int n = X.rows();
        int M = y.cols();

        if (num == 0) {
            if (p == this->XTy.rows()) {
                this->XTy_U = this->XTy;
                this->XTone_U = this->XTone;
                this->PhiG_U = this->PhiG;
                this->invPhiG_U = this->invPhiG;
            } else {
                this->XTy_U.resize(p, M);
                this->XTone_U.resize(p, M);
                this->mapping_U(U, U_ind);
            }
        }

        Eigen::MatrixXd d;
        if (!this->covariance_update) {
            Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, y.cols());
            if (beta.size() != 0) {
                d = X.adjoint() * (y - XA * beta_A - array_product(one, coef0)) / double(n) -
                    2 * this->lambda_level * beta;
            } else {
                d = X.adjoint() * (y - array_product(one, coef0)) / double(n);
            }
        } else {
            if (beta.size() != 0) {
                Eigen::MatrixXd XTXbeta = this->covariance_update_f_U(U_ind, A_ind) * beta_A;

                d = (this->XTy_U - XTXbeta - array_product(this->XTone_U, coef0)) / double(n) -
                    2 * this->lambda_level * beta;
            } else {
                Eigen::MatrixXd XTonecoef0 = array_product(this->XTone_U, coef0);
                d = (this->XTy_U - XTonecoef0) / double(n);
            }
        }

        int A_size = A.size();
        int I_size = I.size();

        Eigen::MatrixXd betabar = Eigen::MatrixXd::Zero(p, M);
        Eigen::MatrixXd dbar = Eigen::MatrixXd::Zero(p, M);
        Eigen::MatrixXd phiG, invphiG;

        for (int i = 0; i < N; i++) {
            phiG = this->PhiG_U(i, 0);
            invphiG = this->invPhiG_U(i, 0);
            betabar.block(g_index(i), 0, g_size(i), M) = phiG * beta.block(g_index(i), 0, g_size(i), M);
            dbar.block(g_index(i), 0, g_size(i), M) = invphiG * d.block(g_index(i), 0, g_size(i), M);
        }
        for (int i = 0; i < A_size; i++) {
            bd(A[i]) = betabar.block(g_index(A[i]), 0, g_size(A[i]), M).squaredNorm() / g_size(A[i]);
        }
        for (int i = 0; i < I_size; i++) {
            bd(I[i]) = dbar.block(g_index(I[i]), 0, g_size(I[i]), M).squaredNorm() / g_size(I[i]);
        }
    }

    double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::VectorXd &weights,
                                         Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A, Eigen::VectorXd &coef0) {
        if (this->lambda_level == 0.) {
            return XA.cols();
        } else {
            if (XA.cols() == 0) return 0.;

            Eigen::MatrixXd XGbar;
            XGbar = XA.transpose() * XA;

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> adjoint_eigen_solver(XGbar);

            double enp = 0.;
            for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++) {
                enp += adjoint_eigen_solver.eigenvalues()(i) /
                       (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
            }

            return enp;
        }
    }
};

template <class T4>
class abessMultinomial : public _abessGLM<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4> {
   public:
    abessMultinomial(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10,
                     double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5,
                     Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0,
                     int sub_search = 0)
        : _abessGLM<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4>::_abessGLM(
              algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start,
              exchange_num, always_select, splicing_type, sub_search){};

    ~abessMultinomial(){};

    double PiPj_range[2] = {0.001, 1};  // Pi * Pj

    bool primary_model_fit(T4 &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta,
                           Eigen::VectorXd &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index,
                           Eigen::VectorXi &g_size) {
        // if (X.cols() == 0)
        // {
        //   coef0 = -log(y.colwise().sum().eval() - 1.0);
        //   return;
        // }
        int n = x.rows();
        int p = x.cols();
        int M = y.cols();
        if (p == 0) return true;

        T4 X;
        add_constant_column(X, x, this->fit_intercept);
        Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(X.cols(), X.cols());
        Eigen::MatrixXd beta0;
        combine_beta_coef0(beta0, beta, coef0, this->fit_intercept);

        Eigen::MatrixXd one_vec = Eigen::VectorXd::Ones(n);
        Eigen::MatrixXd Pi;
        pi(X, y, beta0, Pi);
        Eigen::MatrixXd log_Pi = Pi.array().log();
        array_product(log_Pi, weights, 1);
        double loglik1 = DBL_MAX,
               loglik0 = (log_Pi.array() * y.array()).sum() - this->lambda_level * beta.cwiseAbs2().sum();

        int j;
        if (this->approximate_Newton) {
            Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, M);
            double t = 2 * (Pi.array() * (one - Pi).array()).maxCoeff();
            Eigen::MatrixXd res = y - Pi;
            array_product(res, weights, 1);
            Eigen::MatrixXd XTres = X.transpose() * res / t;
            // ConjugateGradient<MatrixXd, Lower | Upper> cg;
            // cg.compute(X.adjoint() * X);
            Eigen::MatrixXd XTX =
                X.transpose() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols());
            // if (check_ill_condition(XTX)) return false;
            Eigen::MatrixXd invXTX = XTX.ldlt().solve(Eigen::MatrixXd::Identity(X.cols(), X.cols()));

            Eigen::MatrixXd beta1;
            for (j = 0; j < this->primary_model_fit_max_iter; j++) {
                // beta1 = beta0 + cg.solve(res);
                beta1 = beta0 + invXTX * XTres;

                // double app_loss0, app_loss1, app_loss2;
                // app_loss0 = ((y - Pi) / t).squaredNorm();
                // app_loss1 = (-X * beta0 - (y - Pi) / t).squaredNorm();
                // app_loss2 = (X * (beta1 - beta0) - (y - Pi) / t).squaredNorm();

                pi(X, y, beta1, Pi);
                log_Pi = Pi.array().log();
                array_product(log_Pi, weights, 1);
                loglik1 = (log_Pi.array() * y.array()).sum() - this->lambda_level * beta.cwiseAbs2().sum();

                bool condition1 =
                    -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + this->tau > loss0;
                // bool condition1 = false;
                bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < this->primary_model_fit_epsilon;
                bool condition3 = abs(loglik1) < min(1e-3, this->tau);
                bool condition4 = loglik1 < loglik0;
                // bool condition4 = false;
                if (condition1 || condition2 || condition3 || condition4) {
                    break;
                }
                loglik0 = loglik1;

                for (int m1 = 0; m1 < M; m1++) {
                    beta0.col(m1) = beta1.col(m1) - beta1.col(M - 1);
                }

                // beta0 = beta1;
                t = 2 * (Pi.array() * (one - Pi).array()).maxCoeff();
                res = y - Pi;
                array_product(res, weights, 1);
                XTres = X.transpose() * res / t;
            }
        } else {
            Eigen::MatrixXd W(M * n, M * n);
            Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
            for (int m1 = 0; m1 < M; m1++) {
                for (int m2 = m1; m2 < M; m2++) {
                    if (m1 == m2) {
                        W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);
                        Eigen::VectorXd PiPj = Pi.col(m1).array() * (one - Pi.col(m1).eval()).array() * weights.array();
                        trunc(PiPj, PiPj_range);
                        W.block(m1 * n, m2 * n, n, n).diagonal() = PiPj;
                    } else {
                        W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);
                        Eigen::VectorXd PiPj = Pi.col(m1).array() * Pi.col(m2).array() * weights.array();
                        trunc(PiPj, PiPj_range);
                        W.block(m1 * n, m2 * n, n, n).diagonal() = -PiPj;
                        W.block(m2 * n, m1 * n, n, n) = W.block(m1 * n, m2 * n, n, n);
                    }
                }
            }

            Eigen::MatrixXd XTWX(M * X.cols(), M * X.cols());
            Eigen::MatrixXd XTW(M * X.cols(), M * n);
            for (int m1 = 0; m1 < M; m1++) {
                for (int m2 = m1; m2 < M; m2++) {
                    XTW.block(m1 * X.cols(), m2 * n, X.cols(), n) = X.transpose() * W.block(m1 * n, m2 * n, n, n);
                    XTWX.block(m1 * X.cols(), m2 * X.cols(), X.cols(), X.cols()) =
                        XTW.block(m1 * X.cols(), m2 * n, X.cols(), n) * X + 2 * this->lambda_level * lambdamat;
                    XTW.block(m2 * X.cols(), m1 * n, X.cols(), n) = XTW.block(m1 * X.cols(), m2 * n, X.cols(), n);
                    XTWX.block(m2 * X.cols(), m1 * X.cols(), X.cols(), X.cols()) =
                        XTWX.block(m1 * X.cols(), m2 * X.cols(), X.cols(), X.cols());
                }
            }

            // Eigen::Matrix<Eigen::MatrixXd, -1, -1> res(M, 1);
            Eigen::VectorXd res(M * n);
            for (int m1 = 0; m1 < M; m1++) {
                res.segment(m1 * n, n) = y.col(m1).eval() - Pi.col(m1).eval();
            }

            Eigen::VectorXd Xbeta(M * n);
            for (int m1 = 0; m1 < M; m1++) {
                Xbeta.segment(m1 * n, n) = X * beta0.col(m1).eval();
            }

            Eigen::VectorXd Z = Xbeta + W.ldlt().solve(res);

            Eigen::MatrixXd beta1;
            Eigen::VectorXd beta0_tmp;
            for (j = 0; j < this->primary_model_fit_max_iter; j++) {
                beta0_tmp = XTWX.ldlt().solve(XTW * Z);
                for (int m1 = 0; m1 < M; m1++) {
                    beta0.col(m1) =
                        beta0_tmp.segment(m1 * X.cols(), X.cols()) - beta0_tmp.segment((M - 1) * X.cols(), X.cols());
                }
                for (int m1 = 0; m1 < M; m1++) {
                    beta0.col(m1) = beta0_tmp.segment(m1 * X.cols(), X.cols());
                }

                pi(X, y, beta0, Pi);
                log_Pi = Pi.array().log();
                array_product(log_Pi, weights, 1);
                loglik1 = (log_Pi.array() * y.array()).sum() - this->lambda_level * beta.cwiseAbs2().sum();

                bool condition1 =
                    -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + this->tau > loss0;
                // bool condition1 = false;
                bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < this->primary_model_fit_epsilon;
                bool condition3 = abs(loglik1) < min(1e-3, this->tau);
                bool condition4 = loglik1 < loglik0;
                if (condition1 || condition2 || condition3 || condition4) {
                    break;
                }
                loglik0 = loglik1;

                for (int m1 = 0; m1 < M; m1++) {
                    for (int m2 = m1; m2 < M; m2++) {
                        if (m1 == m2) {
                            // W(m1, m2) = Eigen::MatrixXd::Zero(n, n);
                            // W(m1, m2).diagonal() = Pi.col(m1).array() * (one - Pi.col(m1).eval()).array();

                            W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);
                            Eigen::VectorXd PiPj = Pi.col(m1).array() * (one - Pi.col(m1).eval()).array();
                            trunc(PiPj, PiPj_range);
                            W.block(m1 * n, m2 * n, n, n).diagonal() = PiPj;
                        } else {
                            W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);
                            Eigen::VectorXd PiPj = Pi.col(m1).array() * Pi.col(m2).array();
                            trunc(PiPj, PiPj_range);
                            W.block(m1 * n, m2 * n, n, n).diagonal() = -PiPj;
                            W.block(m2 * n, m1 * n, n, n) = W.block(m1 * n, m2 * n, n, n);
                        }
                    }
                }

                for (int m1 = 0; m1 < M; m1++) {
                    for (int m2 = m1; m2 < M; m2++) {
                        XTW.block(m1 * X.cols(), m2 * n, X.cols(), n) = X.transpose() * W.block(m1 * n, m2 * n, n, n);
                        XTWX.block(m1 * X.cols(), m2 * X.cols(), X.cols(), X.cols()) =
                            XTW.block(m1 * X.cols(), m2 * n, X.cols(), n) * X + 2 * this->lambda_level * lambdamat;
                        XTW.block(m2 * X.cols(), m1 * n, X.cols(), n) = XTW.block(m1 * X.cols(), m2 * n, X.cols(), n);
                        XTWX.block(m2 * X.cols(), m1 * X.cols(), X.cols(), X.cols()) =
                            XTWX.block(m1 * X.cols(), m2 * X.cols(), X.cols(), X.cols());
                    }
                }

                for (int m1 = 0; m1 < M; m1++) {
                    res.segment(m1 * n, n) = y.col(m1).eval() - Pi.col(m1).eval();
                }

                for (int m1 = 0; m1 < M; m1++) {
                    Xbeta.segment(m1 * n, n) = X * beta0.col(m1).eval();
                }

                Z = Xbeta + W.ldlt().solve(res);
            }
        }

        extract_beta_coef0(beta0, beta, coef0, this->fit_intercept);
        trunc(beta, this->beta_range);
        return true;
    };

    double loss_function(T4 &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta,
                         Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size,
                         double lambda) {
        // weight
        Eigen::MatrixXd pr;
        pi(X, y, beta, coef0, pr);
        Eigen::MatrixXd log_pr = pr.array().log();
        // Eigen::VectorXd one_vec = Eigen::VectorXd::Ones(X.rows());

        array_product(log_pr, weights, 1);

        return -((log_pr.array() * y.array()).sum()) + lambda * beta.cwiseAbs2().sum();
    }

    void sacrifice(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A,
                   Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights,
                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind,
                   Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num) {
        int p = X.cols();
        int n = X.rows();
        int M = y.cols();

        Eigen::MatrixXd d;
        Eigen::MatrixXd h;
        Eigen::MatrixXd pr;
        pi(XA, y, beta_A, coef0, pr);
        Eigen::MatrixXd Pi = pr.leftCols(M - 1);
        Eigen::MatrixXd res = (y.leftCols(M - 1) - Pi);
        for (int i = 0; i < n; i++) {
            res.row(i) = res.row(i) * weights(i);
        }
        d = X.transpose() * res - 2 * this->lambda_level * beta;
        h = Pi;

        // int A_size = A.size();
        // int I_size = I.size();

        Eigen::MatrixXd betabar = Eigen::MatrixXd::Zero(p, M);
        Eigen::MatrixXd dbar = Eigen::MatrixXd::Zero(p, M);
        Eigen::MatrixXd phiG, invphiG;

        for (int i = 0; i < N; i++) {
            T4 XG = X.middleCols(g_index(i), g_size(i));
            T4 XG_new(h.rows(), h.cols());
            for (int m = 0; m < M - 1; m++) {
                XG_new.col(m) = h.col(m).cwiseProduct(XG).cwiseProduct(weights);
            }
            Eigen::MatrixXd XGbar = -XG_new.transpose() * XG_new;

            XGbar.diagonal() = Eigen::VectorXd(XG_new.transpose() * XG) + XGbar.diagonal();

            XGbar = XGbar + 2 * this->lambda_level * Eigen::MatrixXd::Identity(M - 1, M - 1);

            // Eigen::MatrixXd phiG;
            // XGbar.sqrt().evalTo(phiG);
            // Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(M, M));
            // betabar.block(g_index(i), 0, g_size(i), M) = phiG * beta.block(g_index(i), 0, g_size(i), M);
            // dbar.block(g_index(i), 0, g_size(i), M) = invphiG * d.block(g_index(i), 0, g_size(i), M);

            Eigen::MatrixXd invXGbar = XGbar.ldlt().solve(Eigen::MatrixXd::Identity(M - 1, M - 1));
            Eigen::MatrixXd temp =
                d.block(g_index(i), 0, g_size(i), M - 1) * invXGbar + beta.block(g_index(i), 0, g_size(i), M - 1);
            bd(i) = (temp * XGbar * temp.transpose()).squaredNorm() / g_size(i);
        }
        // for (int i = 0; i < A_size; i++)
        // {
        //   bd(A[i]) = betabar.block(g_index(A[i]), 0, g_size(A[i]), M).squaredNorm() / g_size(A[i]);
        // }
        // for (int i = 0; i < I_size; i++)
        // {
        //   bd(I[i]) = dbar.block(g_index(I[i]), 0, g_size(I[i]), M).squaredNorm() / g_size(I[i]);
        // }
    }

    double effective_number_of_parameter(T4 &x, T4 &XA, Eigen::MatrixXd &y, Eigen::VectorXd &weights,
                                         Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A, Eigen::VectorXd &coef0) {
        if (this->lambda_level == 0.) {
            return XA.cols();
        } else {
            if (XA.cols() == 0) return 0.;

            int n = XA.rows();
            int p = XA.cols();
            int M = y.cols();
            T4 X(n, p + 1);
            X.rightCols(p) = XA;
            add_constant_column(X);
            Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p + 1, p + 1);
            Eigen::MatrixXd beta0 = Eigen::MatrixXd::Zero(p + 1, M);

            Eigen::MatrixXd one_vec = Eigen::VectorXd::Ones(n);
            beta0.row(0) = coef0;
            beta0.block(1, 0, p, M) = beta;
            Eigen::MatrixXd Pi;
            pi(X, y, beta0, Pi);

            Eigen::MatrixXd W(M * n, M * n);
            Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
            for (int m1 = 0; m1 < M; m1++) {
                for (int m2 = m1; m2 < M; m2++) {
                    if (m1 == m2) {
                        W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);
                        Eigen::VectorXd PiPj = Pi.col(m1).array() * (one - Pi.col(m1).eval()).array() * weights.array();
                        trunc(PiPj, PiPj_range);
                        W.block(m1 * n, m2 * n, n, n).diagonal() = PiPj;
                    } else {
                        W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);

                        Eigen::VectorXd PiPj = Pi.col(m1).array() * Pi.col(m2).array() * weights.array();
                        trunc(PiPj, PiPj_range);
                        W.block(m1 * n, m2 * n, n, n).diagonal() = -PiPj;
                        W.block(m2 * n, m1 * n, n, n) = W.block(m1 * n, m2 * n, n, n);
                    }
                }
            }

            Eigen::MatrixXd XTWX(M * (p + 1), M * (p + 1));
            Eigen::MatrixXd XTW(M * (p + 1), M * n);
            for (int m1 = 0; m1 < M; m1++) {
                for (int m2 = m1; m2 < M; m2++) {
                    XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) = X.transpose() * W.block(m1 * n, m2 * n, n, n);
                    XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1)) =
                        XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) * X + 2 * this->lambda_level * lambdamat;
                    XTW.block(m2 * (p + 1), m1 * n, (p + 1), n) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n);
                    XTWX.block(m2 * (p + 1), m1 * (p + 1), (p + 1), (p + 1)) =
                        XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1));
                }
            }

            // Eigen::VectorXd d = X.transpose() * res - 2 * this->lambda_level * beta;
            // Eigen::VectorXd h = weights.array() * pr.array() * (one - pr).array();

            // T4 XA_new = XA;
            // for (int j = 0; j < XA.cols(); j++)
            // {
            //   XA_new.col(j) = XA.col(j).cwiseProduct(h);
            // }
            // Eigen::MatrixXd XGbar;
            // XGbar = XA_new.transpose() * XA;

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> adjoint_eigen_solver(XTWX);

            double enp = 0.;
            for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++) {
                enp += adjoint_eigen_solver.eigenvalues()(i) /
                       (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
            }

            return enp;
        }
    }
};

template <class T4>
class abessGamma : public _abessGLM<Eigen::VectorXd, Eigen::VectorXd, double, T4> {
   public:
    abessGamma(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10,
               double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5,
               Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0)
        : _abessGLM<Eigen::VectorXd, Eigen::VectorXd, double, T4>::_abessGLM(
              algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start,
              exchange_num, always_select, splicing_type, sub_search){};
    ~abessGamma(){};

    double Xbeta_range[2] = {-DBL_MAX, -1e-20};  // the range of acceptable value for X * beta

    Eigen::MatrixXd gradient_core(T4 &X_full, Eigen::VectorXd &y, Eigen::VectorXd &weights,
                                  Eigen::VectorXd &beta_full) {
        Eigen::VectorXd EY = this->inv_link_function(X_full, beta_full);
        Eigen::VectorXd G = (y - EY).cwiseProduct(weights);
        return Eigen::MatrixXd(G);
    };

    Eigen::VectorXd hessian_core(T4 &X_full, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta_full) {
        Eigen::VectorXd EY = this->inv_link_function(X_full, beta_full);
        Eigen::VectorXd W = EY.array().square();
        return W.cwiseProduct(weights);
    };

    Eigen::VectorXd inv_link_function(T4 &X_full, Eigen::VectorXd &beta_full) {
        Eigen::VectorXd eta = X_full * beta_full;
        trunc(eta, Xbeta_range);
        return -eta.cwiseInverse();
    }

    Eigen::VectorXd log_probability(T4 &X_full, Eigen::VectorXd &beta_full, Eigen::VectorXd &y) {
        Eigen::VectorXd eta = X_full * beta_full;
        trunc(eta, Xbeta_range);
        return (-eta).array().log().matrix() + eta.cwiseProduct(y);
    }

    bool null_model(Eigen::VectorXd &y, Eigen::VectorXd &weights, double &coef0) {
        coef0 = -weights.sum() / weights.dot(y);
        return true;
    }

    bool primary_model_fit(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0,
                           double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size) {
        if (X.cols() == 0) return null_model(y, weights, coef0);

        Eigen::VectorXd temp = X * beta + coef0 * Eigen::VectorXd::Ones(X.rows());
        if (temp.maxCoeff() > this->Xbeta_range[1]) {
            coef0 -= abs(temp.maxCoeff()) + 0.1;
        }

        if (this->approximate_Newton) {
            // Fitting method 1: Approximate Newton
            return this->_approx_newton_fit(X, y, weights, beta, coef0, loss0, A, g_index, g_size);
        } else {
            // Fitting method 2: Iteratively Reweighted Least Squares
            return this->_IRLS_fit(X, y, weights, beta, coef0, loss0, A, g_index, g_size);
        }
        trunc(beta, this->beta_range);
        return true;
    };
};

template <class T4>
class abessOrdinal : public _abessGLM<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4> {
   public:
    abessOrdinal(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10,
                 double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5,
                 Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0)
        : _abessGLM<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4>::_abessGLM(
              algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start,
              exchange_num, always_select, splicing_type, sub_search){};
    ~abessOrdinal(){};
    // We only use beta.col(0) and coef0.head(k) which k = coef0.size() - 1 rather than the whole beta and coef0.
    bool gradient(T4 &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0,
                  Eigen::VectorXd &g) {
        const int n = X.rows();
        const int p = X.cols();
        const int k = coef0.size() - 1;
        if (g.size() != p && g.size() != p + k) {
            return false;
        }

        Eigen::VectorXd coef(p + k);
        coef.head(k) = coef0.head(k);
        coef.tail(p) = beta.col(0);

        Eigen::MatrixXd logit(n, k);
        Eigen::MatrixXd P(n, k + 1);
        Eigen::MatrixXd grad_L(n, k);
        Eigen::VectorXd xbeta = X * coef.tail(p);

        // compute logit
        for (int i1 = 0; i1 < n; i1++) {
            for (int i2 = 0; i2 < k; i2++) {
                logit(i1, i2) = 1.0 / (1 + exp(-xbeta(i1) - coef(i2)));
            }
        }
        // compute P
        for (int i1 = 0; i1 < n; i1++) {
            for (int i2 = 0; i2 < k + 1; i2++) {
                if (i2 == 0) {
                    P(i1, 0) = logit(i1, 0);
                } else if (i2 == k) {
                    P(i1, k) = 1 - logit(i1, k - 1);
                } else {
                    P(i1, i2) = logit(i1, i2) - logit(i1, i2 - 1);
                }
                if (P(i1, i2) < 1e-10) P(i1, i2) = 1e-10;
            }
        }
        // compute gradient

        for (int i1 = 0; i1 < n; i1++) {
            for (int i2 = 0; i2 < k; i2++) {
                grad_L(i1, i2) = (y(i1, i2) / P(i1, i2) - y(i1, i2 + 1) / P(i1, i2 + 1)) * logit(i1, i2) *
                                 (1.0 - logit(i1, i2)) * weights(i1);
            }
        }

        if (g.size() == p + k) {
            g.head(k) = grad_L.colwise().sum();
        }
        g.tail(p) = grad_L.rowwise().sum().transpose() * X - 2 * this->lambda_level * coef.tail(p).eval();

        return true;
    }

    bool hessianCore(T4 &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0,
                     Eigen::VectorXd &h_intercept, Eigen::VectorXd &W) {
        const int n = X.rows();
        const int p = X.cols();
        const int k = coef0.size() - 1;
        if (h_intercept.size() != k || W.size() != n) {
            return false;
        }

        Eigen::VectorXd coef(p + k);
        coef.head(k) = coef0.head(k);
        coef.tail(p) = beta.col(0);

        Eigen::MatrixXd logit(n, k);
        Eigen::MatrixXd P(n, k + 1);
        Eigen::VectorXd xbeta = X * coef.tail(p);

        // compute logit
        for (int i1 = 0; i1 < n; i1++) {
            for (int i2 = 0; i2 < k; i2++) {
                logit(i1, i2) = 1.0 / (1 + exp(-xbeta(i1) - coef(i2)));
            }
        }
        // compute P
        for (int i1 = 0; i1 < n; i1++) {
            for (int i2 = 0; i2 < k + 1; i2++) {
                if (i2 == 0) {
                    P(i1, 0) = logit(i1, 0);
                } else if (i2 == k) {
                    P(i1, k) = 1 - logit(i1, k - 1);
                } else {
                    P(i1, i2) = logit(i1, i2) - logit(i1, i2 - 1);
                }
                if (P(i1, i2) < 1e-10) P(i1, i2) = 1e-10;
            }
        }

        for (int i2 = 0; i2 < k; i2++) {
            for (int i1 = 0; i1 < n; i1++) {
                h_intercept(i2) += (1.0 / P(i1, i2) + 1.0 / P(i1, i2 + 1)) * logit(i1, i2) * logit(i1, i2) *
                                   (1.0 - logit(i1, i2)) * (1.0 - logit(i1, i2)) * weights(i1);
            }
        }

        W = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < n; i++) {
            for (int i1 = 0; i1 < k; i1++) {
                W(i) += (1.0 / P(i, i1) + 1.0 / P(i, i1 + 1)) * logit(i, i1) * logit(i, i1) * (1.0 - logit(i, i1)) *
                        (1.0 - logit(i, i1));
            }
            for (int i1 = 0; i1 < k - 1; i1++) {
                W(i) -= 1.0 / P(i, i1 + 1) * logit(i, i1) * logit(i, i1 + 1) * (1.0 - logit(i, i1)) *
                        (1.0 - logit(i, i1 + 1));
            }
            W(i) *= weights(i);
        }

        return true;
    }

    bool primary_model_fit(T4 &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta,
                           Eigen::VectorXd &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_indeX,
                           Eigen::VectorXi &g_size) {
        int i;
        int n = X.rows();
        int p = X.cols();
        int k = coef0.size() - 1;
        // make sure that coef0 is increasing
        for (int i = 1; i < k; i++) {
            if (coef0(i) <= coef0(i - 1)) {
                coef0(i) = coef0(i - 1) + 1;
            }
        }

        double step = 1;
        double loglik_new, loglik;
        Eigen::VectorXd g(p + k);
        Eigen::VectorXd coef_new;
        Eigen::VectorXd h_intercept = Eigen::VectorXd::Zero(k);
        Eigen::VectorXd h_diag = Eigen::VectorXd::Zero(k + p);
        Eigen::VectorXd W(n);
        Eigen::VectorXd desend_direction;  // coef_new = coef + step * desend_direction
        Eigen::VectorXd coef = Eigen::VectorXd::Zero(p + k);
        coef.head(k) = coef0.head(k);
        coef.tail(p) = beta.col(0);
        loglik = -loss_function(X, y, weights, beta, coef0, A, g_indeX, g_size, this->lambda_level);

        for (int j = 0; j < this->primary_model_fit_max_iter; j++) {
            if (!gradient(X, y, weights, beta, coef0, g) || !hessianCore(X, y, weights, beta, coef0, h_intercept, W)) {
                return false;
            }

            for (int i = 0; i < k; i++) {
                if (h_intercept(i) < 1e-7 && h_intercept(i) >= 0)
                    h_diag(i) = 1e7;
                else if (h_intercept(i) > -1e-7 && h_intercept(i) < 0)
                    h_diag(i) = -1e7;
                else
                    h_diag(i) = 1.0 / h_intercept(i);
            }

            for (int i = 0; i < p; i++) {
                h_diag(i + k) = X.col(i).cwiseProduct(W).dot(X.col(i)) + 2 * this->lambda_level;
                if (h_diag(i + k) < 1e-7 && h_diag(i + k) >= 0)
                    h_diag(i + k) = 1e7;
                else if (h_diag(i + k) > -1e-7 && h_diag(i + k) < 0)
                    h_diag(i + k) = -1e7;
                else
                    h_diag(i + k) = 1.0 / h_diag(i + k);
            }

            desend_direction = g.cwiseProduct(h_diag);
            coef_new = coef + step * desend_direction;  // ApproXimate Newton method

            i = 1;
            while (i < k) {
                for (i = 1; i < k; i++) {
                    if (coef_new(i) <= coef_new(i - 1)) {
                        step = step / 2;
                        coef_new = coef + step * desend_direction;
                        break;
                    }
                }
            }

            beta.col(0) = coef_new.tail(p);
            coef0.head(k) = coef_new.head(k);
            loglik_new = -loss_function(X, y, weights, beta, coef0, A, g_indeX, g_size, this->lambda_level);
            while (loglik_new < loglik && step > this->primary_model_fit_epsilon) {
                step = step / 2;
                coef_new = coef + step * desend_direction;
                i = 1;
                while (i < k) {
                    for (i = 1; i < k; i++) {
                        if (coef_new(i) <= coef_new(i - 1)) {
                            step = step / 2;
                            coef_new = coef + step * desend_direction;
                            break;
                        }
                    }
                }
                beta.col(0) = coef_new.tail(p);
                coef0.head(k) = coef_new.head(k);
                loglik_new = -loss_function(X, y, weights, beta, coef0, A, g_indeX, g_size, this->lambda_level);
            }

            bool condition1 = step < this->primary_model_fit_epsilon;
            bool condition2 =
                -(loglik_new + (this->primary_model_fit_max_iter - j - 1) * (loglik_new - loglik)) + this->tau > loss0;

            if (condition1 || condition2) {
                break;
            }

            beta.col(0) = coef_new.tail(p);
            coef0.head(k) = coef_new.head(k);
            coef = coef_new;
            loglik = loglik_new;
        }

        for (int i = 0; i < beta.cols(); i++) {
            beta.col(i) = coef.tail(p).eval();
        }
        coef0.head(k) = coef.head(k);
        trunc(beta, this->beta_range);
        return true;
    }

    double loss_function(T4 &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta,
                         Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_indeX, Eigen::VectorXi &g_size,
                         double lambda) {
        int n = X.rows();
        // int p = X.cols();
        int k = coef0.size() - 1;

        Eigen::VectorXd xbeta = X * beta.col(0);
        double loss = lambda * beta.col(0).cwiseAbs2().sum();

        double pro = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k + 1; j++) {
                if (y(i, j) == 1) {
                    if (j == 0) {
                        loss += log(1 + exp(-xbeta(i) - coef0(0))) * weights(i);
                    } else if (j == k) {
                        loss -= log(1 - 1.0 / (1 + exp(-xbeta(i) - coef0(k - 1)))) * weights(i);
                    } else {
                        pro = 1.0 / (1 + exp(-xbeta(i) - coef0(j))) - 1.0 / (1 + exp(-xbeta(i) - coef0(j - 1)));
                        if (pro < 1e-20) pro = 1e-20;
                        loss -= log(pro) * weights(i);
                    }
                    break;
                }
            }
        }

        return loss;
    }

    void sacrifice(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A,
                   Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights,
                   Eigen::VectorXi &g_indeX, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind,
                   Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num) {
        int n = X.rows();
        int p = X.cols();
        int k = coef0.size() - 1;

        Eigen::VectorXd W = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd d(p);
        Eigen::VectorXd h_intercept = Eigen::VectorXd::Zero(k);

        gradient(X, y, weights, beta, coef0, d);
        hessianCore(X, y, weights, beta, coef0, h_intercept, W);

        Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
        // we only need N diagonal sub-matriX of hessian of Loss, X^T %*% diag(EY^2) %*% X is OK, but waste.
        for (int i = 0; i < N; i++) {
            T4 XG = X.middleCols(g_indeX(i), g_size(i));
            T4 XG_new = XG;
            for (int j = 0; j < g_size(i); j++) {
                XG_new.col(j) = XG.col(j).cwiseProduct(W);
            }
            // hessianG is the ith group diagonal sub-matriX of hessian matriX of Loss.
            Eigen::MatrixXd hessianG =
                XG_new.transpose() * XG + 2 * this->lambda_level * Eigen::MatrixXd::Identity(g_size(i), g_size(i));
            Eigen::MatrixXd phiG;
            hessianG.sqrt().evalTo(phiG);
            Eigen::MatrixXd invphiG = phiG.ldlt().solve(
                Eigen::MatrixXd::Identity(g_size(i), g_size(i)));  // this is a way to inverse a matriX.
            betabar.segment(g_indeX(i), g_size(i)) = phiG * beta.col(0).segment(g_indeX(i), g_size(i));
            dbar.segment(g_indeX(i), g_size(i)) = invphiG * d.segment(g_indeX(i), g_size(i));
        }
        int A_size = A.size();
        int I_size = I.size();
        for (int i = 0; i < A_size; i++) {
            bd(A[i]) = betabar.segment(g_indeX(A[i]), g_size(A[i])).squaredNorm() / g_size(A[i]);
        }
        for (int i = 0; i < I_size; i++) {
            bd(I[i]) = dbar.segment(g_indeX(I[i]), g_size(I[i])).squaredNorm() / g_size(I[i]);
        }
    }

    double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::VectorXd &weights,
                                         Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A, Eigen::VectorXd &coef0) {
        if (this->lambda_level == 0.) return XA.cols();

        if (XA.cols() == 0) return 0.;

        int n = X.rows();
        // int p = X.cols();
        int k = coef0.size() - 1;

        Eigen::VectorXd h_intercept = Eigen::VectorXd::Zero(k);
        Eigen::VectorXd W = Eigen::VectorXd::Zero(n);

        hessianCore(X, y, weights, beta, coef0, h_intercept, W);

        T4 XA_new = XA;
        for (int j = 0; j < XA.cols(); j++) {
            XA_new.col(j) = XA.col(j).cwiseProduct(W);
        }
        Eigen::MatrixXd XGbar = XA_new.transpose() * XA;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> adjoint_eigen_solver(XGbar);
        double enp = 0.;
        for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++) {
            enp += adjoint_eigen_solver.eigenvalues()(i) / (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
        }
        return enp;
    }
};

#endif  // SRC_ALGORITHMGLM_H
