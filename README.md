abess: R & Python Softwares for Best-Subset Selection in Polynomial Time
---

![Github action](https://github.com/abess-team/abess/actions/workflows/ABESS-CI.yaml/badge.svg)
[![Build Status](https://travis-ci.com/abess-team/abess.svg?branch=master)](https://travis-ci.com/abess-team/abess)
[![codecov](https://codecov.io/gh/abess-team/abess/branch/master/graph/badge.svg?token=LK56LHXV00)](https://codecov.io/gh/abess-team/abess)
[![docs](https://readthedocs.org/projects/abess/badge/?version=latest)](https://abess.readthedocs.io/en/latest/?badge=latest)
[![cran](https://img.shields.io/cran/v/abess?logo=R)](https://cran.r-project.org/package=abess)
[![pypi](https://badge.fury.io/py/abess.svg)](https://badge.fury.io/py/abess)
[![pyversions](https://img.shields.io/pypi/pyversions/abess)](https://img.shields.io/pypi/pyversions/abess)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3f6e60a3a3e44699a033159633981b76)](https://www.codacy.com/gh/abess-team/abess/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=abess-team/abess&amp;utm_campaign=Badge_Grade)

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

## Reference
A polynomial algorithm for best-subset selection problem. Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, Xueqin Wang. Proceedings of the National Academy of Sciences Dec 2020, 117 (52) 33117-33123; DOI: 10.1073/pnas.2014241117    
Fan, J. and Lv, J. (2008), Sure independence screening for ultrahigh dimensional feature space. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 70: 849-911. https://doi.org/10.1111/j.1467-9868.2008.00674.x
Qiang Sun & Heping Zhang (2020) Targeted Inference Involving High-Dimensional Data Using Nuisance Penalized Regression, Journal of the American Statistical Association, DOI: 10.1080/01621459.2020.1737079
Zhang, Y., Zhu, J., Zhu, J. and Wang, X., 2021. Certifiably Polynomial Algorithm for Best Group Subset Selection. arXiv preprint arXiv:2104.12576.
