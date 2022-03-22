<img src='https://raw.githubusercontent.com/abess-team/abess/master/docs/image/icon_long.png' align="center"/></a>     

# abess: Fast Best-Subset Selection in R

<!-- badges: start -->
[![Python Build](https://github.com/abess-team/abess/actions/workflows/python_test.yml/badge.svg)](https://github.com/abess-team/abess/actions/workflows/python_test.yml)
[![R Build](https://github.com/abess-team/abess/actions/workflows/r_test.yml/badge.svg)](https://github.com/abess-team/abess/actions/workflows/r_test.yml)
[![codecov](https://codecov.io/gh/abess-team/abess/branch/master/graph/badge.svg?token=LK56LHXV00)](https://app.codecov.io/gh/abess-team/abess)
[![docs](https://readthedocs.org/projects/abess/badge/?version=latest)](https://abess.readthedocs.io/en/latest/?badge=latest)
[![R docs](https://github.com/abess-team/abess/actions/workflows/r_website.yml/badge.svg)](https://abess-team.github.io/abess/)
[![cran](https://img.shields.io/cran/v/abess?logo=R)](https://cran.r-project.org/package=abess)
[![pypi](https://badge.fury.io/py/abess.svg)](https://badge.fury.io/py/abess)
[![pyversions](https://img.shields.io/pypi/pyversions/abess)](https://img.shields.io/pypi/pyversions/abess)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3f6e60a3a3e44699a033159633981b76)](https://app.codacy.com/gh/abess-team/abess/dashboard?utm_source=github.com&utm_medium=referral&utm_content=abess-team/abess&utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/abess-team/abess/badge)](https://www.codefactor.io/repository/github/abess-team/abess)
<!-- badges: end -->

`abess` (Adaptive BEst Subset Selection) library aims to solve the general best subset selection problem, i.e., 
find a small subset of predictors such that the resulting model is expected to have the highest accuracy. 
The selection for best subset shows great value in scientific researches and practical application. 
For example, clinicians wants to know whether a patient is health or not based on the expression level of a few of important genes.

This library implements a generic algorithm framework to find the optimal solution in an extremely fast way.
This framework now supports the detection of best subset under: 
[linear regression](https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_1_LinearRegression.html),
[classification (binary or multi-class)](https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_2_LogisticRegression.html),
[counting-response modeling](https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_5_PossionGammaRegression.html),
[censored-response modeling](https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_4_CoxRegression.html#sphx-glr-auto-gallery-1-glm-plot-4-coxregression-py),
[multi-response modeling (multi-tasks learning)](https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_3_MultiTaskLearning.html), etc.
It also supports the variants of best subset selection like 
[group best subset selection](https://abess.readthedocs.io/en/latest/auto_gallery/3-advanced-features/plot_best_group.html),
[nuisance penalized regression](https://abess.readthedocs.io/en/latest/auto_gallery/3-advanced-features/plot_best_nuisance.html),
especially, the time complexity of the best (group) subset selection for linear regression is certifiably polynomial.

## Installation

To install the `abess` R package from CRAN, just run:

``` r
install.packages("abess")
```

Alternative, you can install the newest version of `abess` by following [this instruction](https://abess.readthedocs.io/en/latest/Installation.html#r-1).

## Runtime Performance

To show the power of `abess` in computation, we assess its timings of the CPU execution (seconds) on synthetic datasets, and compare to state-of-the-art variable selection methods. The variable selection and estimation results are deferred to [performance](https://abess-team.github.io/abess/articles/v11-power-of-abess.html). All computations are conducted on a Ubuntu platform with Intel(R) Core(TM) i9-9940X CPU @ 3.30GHz and 48 RAM. We compare `abess` R package with three widely used R packages: `glmnet`, `ncvreg`, and `L0Learn`. We get the runtime comparison results:

<img src='https://raw.githubusercontent.com/abess-team/abess/master/docs/image/r_runtime.png'/></a>

Compared with the other packages, 
`abess` shows competitive computational efficiency, 
and achieves the best computational power when variables have a large correlation.

Conducting the following command in shell can reproduce the above results in R: 

```shell
$ Rscript abess/docs/simulation/R/timings.R
```

## What's news?

New features supported by the latest version (0.4.5):

* Support generalized linear model for ordinal response (setting `family = "ordinal"`), 
also named as rank learning in machine learning community. 

* Support robust principal analysis

* Modify R package structure to make many internal components are reusable.

## Citation         

If you use `abess` or refer to our tutorials in a presentation or publication, we would appreciate citations of our library.

Jin Zhu, Xueqin Wang, Liyuan Hu, Junhao Huang, Kangkang Jiang, Yanhang Zhang, Shiyun Lin, Junxian Zhu (2021). “abess: A Fast Best Subset Selection Library in Python and R.” arXiv:2110.09697.

The corresponding BibteX entry:

```
@article{zhu-abess-arxiv,
  author    = {Jin Zhu and Xueqin Wang and Liyuan Hu and Junhao Huang and Kangkang Jiang and Yanhang Zhang and Shiyun Lin and Junxian Zhu},
  title     = {abess: A Fast Best Subset Selection Library in Python and R},
  journal   = {arXiv:2110.09697},
  year      = {2021},
}
```

## References

- Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang (2020). A polynomial algorithm for best-subset selection problem. Proceedings of the National Academy of Sciences, 117(52):33117-33123.

- Jin Zhu, Liyuan Hu, Junhao Huang, Kangkang Jiang, Yanhang Zhang, Shiyun Lin, Junxian Zhu, Xueqin Wang (2021). abess: A Fast Best Subset Selection Library in Python and R. arXiv preprint arXiv:2110.09697.

- Pölsterl, S (2020). scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. J. Mach. Learn. Res., 21(212), 1-6.

- Yanhang Zhang, Junxian Zhu, Jin Zhu, and Xueqin Wang (2021). Certifiably Polynomial Algorithm for Best Group Subset Selection. arXiv preprint arXiv:2104.12576.

- Qiang Sun and Heping Zhang (2020). Targeted Inference Involving High-Dimensional Data Using Nuisance Penalized Regression, Journal of the American Statistical Association, DOI: 10.1080/01621459.2020.1737079.
