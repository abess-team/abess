#ifndef SRC_ALGORITHMPCA_H
#define SRC_ALGORITHMPCA_H

#include <Spectra/SymEigsSolver.h>

#include "Algorithm.h"

using namespace Spectra;

template <class T4>
class abessPCA : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4> {
   public:
    int pca_n = -1;
    bool is_cv = false;
    MatrixXd sigma;

    abessPCA(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10,
             double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5,
             Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 1, int sub_search = 0)
        : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(
              algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start,
              exchange_num, always_select, splicing_type, sub_search){};

    ~abessPCA(){};

    void inital_setting(T4 &X, VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size,
                        int &N) {
        if (this->is_cv) {
            this->sigma = compute_Sigma(X);
        }
    }

    void updata_tau(int train_n, int N) {
        if (this->pca_n > 0) train_n = this->pca_n;
        if (train_n == 1) {
            this->tau = 0.0;
        } else {
            this->tau =
                0.01 * (double)this->sparsity_level * log((double)N) * log(log((double)train_n)) / (double)train_n;
        }
    }

    bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0,
                           double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size) {
        if (beta.size() == 0) return true;
        if (beta.size() == 1) {
            beta(0) = 1;
            return true;
        }

        MatrixXd Y = SigmaA(this->sigma, A, g_index, g_size);

        DenseSymMatProd<double> op(Y);
        SymEigsSolver<DenseSymMatProd<double>> eig(op, 1, 2);
        eig.init();
        eig.compute();
        MatrixXd temp;
        if (eig.info() == CompInfo::Successful) {
            temp = eig.eigenvectors(1);
        } else {
            return false;
        }

        beta = temp.col(0);

        return true;
    };

    double loss_function(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0,
                         Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, double lambda) {
        MatrixXd Y;
        if (this->is_cv) {
            MatrixXd sigma_test = compute_Sigma(X);
            Y = SigmaA(sigma_test, A, g_index, g_size);
        } else {
            Y = SigmaA(this->sigma, A, g_index, g_size);
        }

        return -beta.transpose() * Y * beta;
    };

    void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0,
                   Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index,
                   Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U,
                   Eigen::VectorXi &U_ind, int num) {
        VectorXd D = -this->sigma * beta + beta.transpose() * this->sigma * beta * beta;

        for (int i = 0; i < A.size(); i++) {
            VectorXd temp = beta.segment(g_index(A(i)), g_size(A(i)));
            bd(A(i)) = temp.squaredNorm();
        }
        for (int i = 0; i < I.size(); i++) {
            VectorXd temp = D.segment(g_index(I(i)), g_size(I(i)));
            bd(I(i)) = temp.squaredNorm();
        }
    };

    MatrixXd SigmaA(Eigen::MatrixXd &Sigma, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size) {
        int len = 0;
        for (int i = 0; i < A.size(); i++) {
            len += g_size(A(i));
        }
        int k = 0;
        VectorXd ind(len);
        for (int i = 0; i < A.size(); i++)
            for (int j = 0; j < g_size(A(i)); j++) ind(k++) = g_index(A(i)) + j;

        MatrixXd SA(len, len);
        for (int i = 0; i < len; i++)
            for (int j = 0; j < i + 1; j++) {
                int di = ind(i), dj = ind(j);
                SA(i, j) = Sigma(di, dj);
                SA(j, i) = Sigma(dj, di);
            }

        return SA;
    }

    MatrixXd compute_Sigma(T4 &X) {
        MatrixXd X1 = MatrixXd(X);
        MatrixXd centered = X1.rowwise() - X1.colwise().mean();
        return centered.adjoint() * centered / (X1.rows() - 1);
    }
};

template <class T4>
class abessRPCA : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4> {
   private:
    MatrixXd L;
    int r = 10;

   public:
    abessRPCA(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10,
              double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5,
              Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 1, int sub_search = 0)
        : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(
              algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start,
              exchange_num, always_select, splicing_type, sub_search){};

    ~abessRPCA(){};

    int get_beta_size(int n, int p) { return n * p; }

    void update_tau(int train_n, int N) { this->tau = 0.0; }

    Eigen::VectorXi inital_screening(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0,
                                     Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &bd,
                                     Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size,
                                     int &N) {
        MatrixXd S;
        if (bd.size() == 0) {
            // variable initialization
            bd = VectorXd::Zero(N);

            this->L = this->trun_svd(X);
            S = X - this->L;
            S.resize(N, 1);

            for (int i = 0; i < N; i++) bd(i) = abs(S(i, 0));

            for (int i = 0; i < (this->always_select).size(); i++) {
                bd(this->always_select(i)) = DBL_MAX;
            }

            // A_init
            for (int i = 0; i < A.size(); i++) {
                bd(A(i)) = DBL_MAX - 1;
            }

            this->r = (int)this->lambda_level;
        }

        // get Active-set A according to max_k bd
        VectorXi A_new = max_k(bd, this->sparsity_level);

        return A_new;
    }

    bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0,
                           double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size) {
        int n = x.rows();

        MatrixXd L_old = this->L;
        this->L = this->HardImpute(x, A, 1000, 1e-5);
        for (int i = 0; i < A.size(); i++) {
            int mi = A(i) % n;
            int mj = int(A(i) / n);
            beta(i) = x.coeff(mi, mj) - this->L(mi, mj);
        }

        double loss1 = this->loss_function(x, y, weights, beta, coef0, A, g_index, g_size, 0);
        if (loss0 - loss1 <= this->tau) {
            this->L = L_old;
        }

        return true;
    };

    double loss_function(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0,
                         Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, double lambda) {
        int n = X.rows();
        int p = X.cols();
        // MatrixXd L = this->HardImpute(X, A, 1000, 1e-5);
        MatrixXd S = compute_S(beta, A, n, p);
        MatrixXd W = X - this->L - S;
        return W.squaredNorm() / n / p;
    };

    void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0,
                   Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index,
                   Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U,
                   Eigen::VectorXi &U_ind, int num) {
        int n = X.rows();
        int p = X.cols();
        // MatrixXd L = this->HardImpute(X, A, 1000, 1e-5);
        MatrixXd S = compute_S(beta_A, A, n, p);
        MatrixXd W = X - this->L - S;

        for (int i = 0; i < A.size(); i++) {
            int mi = A(i) % n;
            int mj = int(A(i) / n);
            bd(A(i)) = S(mi, mj) * S(mi, mj) + 2 * S(mi, mj) * W(mi, mj);
        }
        for (int i = 0; i < I.size(); i++) {
            int mi = I(i) % n;
            int mj = int(I(i) / n);
            bd(I(i)) = W(mi, mj) * W(mi, mj);
        }
        return;
    };

    MatrixXd trun_svd(MatrixXd X) {
        int m = X.rows(), n = X.cols(), K = this->r;
        MatrixXd Y(m, n);
        if (m > n) {
            MatrixXd R = X.transpose() * X;
            DenseSymMatProd<double> op_r(R);
            SymEigsSolver<DenseSymMatProd<double>> eig_r(op_r, K, 2 * K > n ? n : 2 * K);
            eig_r.init();
            eig_r.compute(SortRule::LargestAlge);
            VectorXd evalues;
            if (eig_r.info() == CompInfo::Successful) {
                evalues = eig_r.eigenvalues();
                int num = 0;
                for (int s = 0; s < K; s++) {
                    if (evalues(s) > 0) {
                        num++;
                    }
                }
                if (num < K) {
                    K = num;
                }
                MatrixXd vec_r = eig_r.eigenvectors(K);
                Y = X * vec_r * vec_r.transpose();
            }
        } else {
            MatrixXd L = X * X.transpose();
            DenseSymMatProd<double> op_l(L);
            SymEigsSolver<DenseSymMatProd<double>> eig_l(op_l, K, 2 * K > m ? m : 2 * K);
            eig_l.init();
            eig_l.compute(SortRule::LargestAlge);
            VectorXd evalues;
            if (eig_l.info() == CompInfo::Successful) {
                evalues = eig_l.eigenvalues();
                int num = 0;
                for (int s = 0; s < K; s++) {
                    if (evalues(s) > 0) {
                        num++;
                    }
                }
                if (num < K) {
                    K = num;
                }
                MatrixXd vec_l = eig_l.eigenvectors(K);
                Y = vec_l * vec_l.transpose() * X;
            }
        }
        return Y;
    };

    MatrixXd HardImpute(T4 &X, VectorXi &A, int max_it, double tol) {
        int m = X.rows(), n = X.cols();
        MatrixXd Z_old = MatrixXd::Zero(m, n);
        MatrixXd Z_new(m, n);
        MatrixXd lambda = MatrixXd::Zero(m, n);
        double eps = 1;
        int count = 0;
        while (eps > tol && count < max_it) {
            lambda = X - Z_old;
            for (int i = 0; i < A.size(); i++) {
                int r = A(i) % m;
                int c = int(A(i) / m);
                lambda(r, c) = 0;
            }
            Z_new = trun_svd(Z_old + lambda);
            eps = (Z_new - Z_old).squaredNorm() / Z_old.squaredNorm();
            Z_old = Z_new;
            count++;
        }
        return Z_new;
    }

    MatrixXd compute_S(VectorXd &beta, VectorXi &A, int n, int p) {
        MatrixXd S = MatrixXd::Zero(n, p);
        for (int i = 0; i < A.size(); i++) {
            int mi = A(i) % n;
            int mj = int(A(i) / n);
            S(mi, mj) = beta(i);
        }
        return S;
    };
};
#endif  // SRC_ALGORITHMPCA_H
