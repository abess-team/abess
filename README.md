abess: R & Python Softwares for Best-Subset Selection in Polynomial Time
---

![Github action](https://github.com/abess-team/abess/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/abess-team/abess/branch/master/graph/badge.svg?token=LK56LHXV00)](https://codecov.io/gh/abess-team/abess)
[![docs](https://readthedocs.org/projects/abess/badge/?version=latest)](https://abess.readthedocs.io/en/latest/?badge=latest)
[![cran](https://img.shields.io/cran/v/abess?logo=R)](https://cran.r-project.org/package=abess)
[![pypi](https://badge.fury.io/py/abess.svg)](https://badge.fury.io/py/abess)
[![pyversions](https://img.shields.io/pypi/pyversions/abess)](https://img.shields.io/pypi/pyversions/abess)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3f6e60a3a3e44699a033159633981b76)](https://www.codacy.com/gh/abess-team/abess/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=abess-team/abess&amp;utm_campaign=Badge_Grade)
<!-- [![Build Status](https://travis-ci.com/abess-team/abess.svg?branch=master)](https://travis-ci.com/abess-team/abess) -->

**abess** (Adaptive BEst Subset Selection) aims to find a small subset of predictors such
that the resulting linear model is expected to have the most desirable
prediction accuracy. This project implements a polynomial algorithm proposed to solve these problems. It supports:

-  linear regression
-  classification (binary or multi-class)
-  counting-response modeling
-  censored-response modeling
-  multi-response modeling (multi-tasks learning)
-  group best subset selection
-  nuisance penalized regression
-  sure independence screening


## Installation
The abess softwares both Python and R's interfaces. 

### Python package
Install the stable version of Python-package from [Pypi](https://pypi.org/project/abess/) with:
```shell
pip install abess
```

### R package
Install the stable version of R-package from [CRAN](https://cran.r-project.org/web/packages/abess) with:
```shell
install.packages("abess")
```

## Performance

To show the computational efficiency of abess, 
we compare abess R package with popular R libraries: glmnet, ncvreg, picasso for linear, logistic and poisson regressions; 
<!-- Timings of the CPU execution are recorded in seconds and averaged over 100 replications on a sequence
of 100 regularization parameters. -->

We consider three aspects. The first one is the prediction performance on a validation data set of size 1000. For linear and poisson regression, this is measured by $\|X\hat{\beta}-X\beta^*\|_2$ where $\hat{\beta}$ is the fitted coefficients and $\beta^*$ is the true coefficients. For the logistic regression, we use the AUC. The second is the selection performance in terms of true positive rate (TPR) and false positive rate (FPR). The third is the running time.

The designed matrix is formed by i.i.d sample generated from a multivariate normal distribution with mean 0 and covariance matrix $\Sigma = (\sigma_{ij})$. We consider two settings—low correlation and high correlation. For the low correlation scenario, we set $\sigma_{ij} = 0.1^{|i-j|}$ and for the high correlation $\sigma_{ij} = 0.7$. The number of predictors is 1000. The true coefficient $\beta^*$ is a vector with 10 nonzero entries uniformly distributed in $[b,B]$. We set $b=5\sqrt{2\log(p)/n}$, $B = 100b$ for linear regression $b = 10\sqrt{2\log(p)/n}$, $B = 5*b$ for logistic regression and $b = -10 \sqrt{2  \log(p) / n}$, $B=10 \sqrt{2 \log(p) / n}$ for poisson regression. A random noise generated from a standard Gaussian distribution is added to the linear predictor $x^\prime\beta$ for linear regression. The size of training data is 500.
All experiments are
evaluated on an Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz and under R version 3.6.1. for 100 replicas.


```r
source("R-package/example/timing.R")
```

Results are presented in the following table and figure. For all the scenarios, the L0-based estimators in L0Learn and abess show the best prediction performance with the abess performs better in terms of variable selection and efficiency. For linear regression, we see that the Lasso estimator in both glmnet and ncvreg has the largest prediction error compared with other estimators. With the increase in correlation, difficulties of identifying the sparsity structure increase for MCP (ncvreg) and SCAD (ncvreg). For logistic regression, abess shows a great advantage in efficiency compared with L0Learn. Here we see that it is difficult for L0-based method to identify all the true significant predictors in high correlation setting, but abess is generally the least likely to make a mistake. 
For poisson regression, our abess package continues to exhibit the dominance of over prediction performance and variable selection. 
Notably, as a package aiming at best subset selection, abess shows a competitive short run time, never been eclipsed by comparison with glmnet and ncvreg which are famous for high efficiency.

<!-- Results are presented in the following picture. As a package solving best subset selection, abess reaches a high efficient performance especially in linear regression where it gives the fastest solution. -->

![avatar](R-package/vignettes/readmeTiming.png)

## Reference
A polynomial algorithm for best-subset selection problem. Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, Xueqin Wang. Proceedings of the National Academy of Sciences Dec 2020, 117 (52) 33117-33123; DOI: 10.1073/pnas.2014241117    
Fan, J. and Lv, J. (2008), Sure independence screening for ultrahigh dimensional feature space. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 70: 849-911. https://doi.org/10.1111/j.1467-9868.2008.00674.x
Qiang Sun & Heping Zhang (2020) Targeted Inference Involving High-Dimensional Data Using Nuisance Penalized Regression, Journal of the American Statistical Association, DOI: 10.1080/01621459.2020.1737079
Zhang, Y., Zhu, J., Zhu, J. and Wang, X., 2021. Certifiably Polynomial Algorithm for Best Group Subset Selection. arXiv preprint arXiv:2104.12576.
