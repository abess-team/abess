# abess: An R package for Best-Subset Selection in Polynomial Time

<!-- badges: start -->
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3f6e60a3a3e44699a033159633981b76)](https://www.codacy.com/gh/abess-team/abess/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=abess-team/abess&amp;utm_campaign=Badge_Grade)
[![Travis build status](https://travis-ci.com/abess-team/abess.svg?branch=master)](https://travis-ci.com/abess-team/abess)
[![Codecov test coverage](https://codecov.io/gh/Mamba413/abess/branch/master/graph/badge.svg)](https://codecov.io/gh/Mamba413/abess?branch=master)
<!-- badges: end -->

Best-subset selection aims to find a small subset of predictors such that the resulting linear model is expected to have the most desirable prediction accuracy. This project implements a polynomial algorithm proposed by Zhu et al (2020) to solve the problem. More over, the softwares includes helpful features for high-dimensional data analysis:

- binary-classification, censored-response modeling
- sure independence screening
- nuisance penalized regression

## Installation

You can install the newest version of abess from [github](https://github.com/) with:

``` r
install.packages("abess")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(abess)
## basic example code
```

## Reference
A polynomial algorithm for best-subset selection problem. Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, Xueqin Wang. Proceedings of the National Academy of Sciences Dec 2020, 117 (52) 33117-33123; DOI: 10.1073/pnas.2014241117    
Fan, J. and Lv, J. (2008), Sure independence screening for ultrahigh dimensional feature space. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 70: 849-911. https://doi.org/10.1111/j.1467-9868.2008.00674.x
Qiang Sun & Heping Zhang (2020) Targeted Inference Involving High-Dimensional Data Using Nuisance Penalized Regression, Journal of the American Statistical Association, DOI: 10.1080/01621459.2020.1737079
