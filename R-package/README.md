# abess: An R package for Best-Subset Selection in Polynomial Time

<!-- badges: start -->
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/49a2e64460124d55a6986f0bf28b738e)](https://www.codacy.com/gh/Mamba413/abess/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Mamba413/abess&amp;utm_campaign=Badge_Grade)
<!-- badges: end -->

Best-subset selection aims to find a small subset of predictors such that the resulting linear model is expected to have the most desirable prediction accuracy. This project implements a polynomial algorithm proposed by Zhu et al (2020) to solve the problem. More over, the softwares includes helpful features for high-dimensional data analysis:

- binary-classification, censored-response modeling
- sure independence screening
- nuisance penalized regression

## Installation

You can install the released version of abess from [CRAN](https://CRAN.R-project.org) with:

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
