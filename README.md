<img src='https://raw.githubusercontent.com/abess-team/abess/master/docs/image/icon_long.png' align="center"/></a>

# abess: Fast Best-Subset Selection in Python and R

<p align="center">
<a href="https://github.com/abess-team/abess/actions/workflows/python_test.yml"><img src="https://github.com/abess-team/abess/actions/workflows/python_test.yml/badge.svg" alt="Python Build"></a>
<a href="https://github.com/abess-team/abess/actions/workflows/r_test.yml"><img src="https://github.com/abess-team/abess/actions/workflows/r_test.yml/badge.svg" alt="R Build"></a>
<a href="https://codecov.io/gh/abess-team/abess"><img src="https://codecov.io/gh/abess-team/abess/branch/master/graph/badge.svg?token=LK56LHXV00" alt="codecov"></a>
<a href="https://www.codacy.com/gh/abess-team/abess/dashboard?utm_source=github.com&utm_medium=referral&utm_content=abess-team/abess&utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/3f6e60a3a3e44699a033159633981b76" alt="Codacy Badge"></a>
<a href="https://www.codefactor.io/repository/github/abess-team/abess"><img src="https://www.codefactor.io/repository/github/abess-team/abess/badge" alt="CodeFactor"></a>
<br>
<a href="https://abess.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/abess/badge/?version=latest" alt="docs"></a>
<a href="https://abess-team.github.io/abess/"><img src="https://github.com/abess-team/abess/actions/workflows/r_website.yml/badge.svg" alt="R docs"></a>
<a href="https://cran.r-project.org/package=abess"><img src="https://img.shields.io/cran/v/abess?logo=R" alt="cran"></a>
<a href="https://pypi.org/project/abess"><img src="https://img.shields.io/pypi/v/abess?logo=Pypi" alt="pypi"></a>
<a href="https://anaconda.org/conda-forge/abess"><img src="https://img.shields.io/conda/vn/conda-forge/abess.svg?logo=condaforge" alt="Conda version"></a>
<a href="https://img.shields.io/pypi/pyversions/abess"><img src="https://img.shields.io/pypi/pyversions/abess" alt="pyversions"></a>
<a href="https://anaconda.org/conda-forge/abess"><img src="https://anaconda.org/conda-forge/abess/badges/platforms.svg" alt="Platform"></a>
<a href="https://pepy.tech/project/abess"><img src="https://pepy.tech/badge/abess" alt="Downloads"></a>
<a href="http://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPL%20v3-blue.svg" alt="License"></a>
</p>

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Runtime Performance](#runtime-performance)
- [What's New](#whats-new)
- [Citation](#citation)
- [Open Source Software](#open-source-software)
- [References](#references)
- [Contributing](#contributions)

## Overview

`abess` (Adaptive BEst Subset Selection) library aims to solve general best subset selection, i.e.,
find a small subset of predictors such that the resulting model is expected to have the highest accuracy.
The selection for best subset shows great value in scientific researches and practical applications.
For example, clinicians want to know whether a patient is healthy or not based on the expression levels of a few of important genes.

This library implements a generic algorithm framework to find the optimal solution in an extremely fast way.
This framework now supports the detection of best subset under:
[linear regression](https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_1_LinearRegression.html),
[classification (binary or multi-class)](https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_2_LogisticRegression.html),
[counting-response modeling](https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_5_PossionGammaRegression.html),
[censored-response modeling](https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_4_CoxRegression.html#sphx-glr-auto-gallery-1-glm-plot-4-coxregression-py),
[multi-response modeling (multi-tasks learning)](https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_3_MultiTaskLearning.html),
[Ising model estimation](https://abess.readthedocs.io/en/latest/auto_gallery/index.html), etc.
It also supports the variants of best subset selection like
[group best subset selection](https://abess.readthedocs.io/en/latest/auto_gallery/3-advanced-features/plot_best_group.html),
[nuisance penalized regression](https://abess.readthedocs.io/en/latest/auto_gallery/3-advanced-features/plot_best_nuisance.html),
Especially, the time complexity of (group) best subset selection for linear regression is certifiably polynomial.

## Quick start

The `abess` software has both Python and R's interfaces. Here a quick start will be given and for more details, please view: [Installation](https://abess.readthedocs.io/en/latest/Installation.html).

### Python package

Install the stable version of Python-package from [Pypi](https://pypi.org/project/abess/):

```shell
$ pip install abess
```

or [conda-forge](https://anaconda.org/conda-forge/abess):

```shell
$ conda install abess
```

Best subset selection for linear regression on a simulated dataset in Python:

```python
from abess.linear import LinearRegression
from abess.datasets import make_glm_data
sim_dat = make_glm_data(n = 300, p = 1000, k = 10, family = "gaussian")
model = LinearRegression()
model.fit(sim_dat.x, sim_dat.y)
```

See more examples analyzed with Python in the [Python tutorials](https://abess.readthedocs.io/en/latest/auto_gallery/index.html).

### R package

Install the stable version of R-package from [CRAN](https://cran.r-project.org/web/packages/abess) with:

```r
install.packages("abess")
```

Best subset selection for linear regression on a simulated dataset in R:

```r
library(abess)
sim_dat <- generate.data(n = 300, p = 1000)
abess(x = sim_dat[["x"]], y = sim_dat[["y"]])
```

See more examples analyzed with R in the [R tutorials](https://abess-team.github.io/abess/articles/).

## Runtime Performance

To show the power of abess in computation, we assess its timings of the CPU execution (seconds) on synthetic datasets, and compare to state-of-the-art variable selection methods. The variable selection and estimation results are deferred to [Python performance](https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_a1_power_of_abess.html) and [R performance](https://abess-team.github.io/abess/articles/v11-power-of-abess.html). All computations are conducted on a Ubuntu platform with Intel(R) Core(TM) i9-9940X CPU @ 3.30GHz and 48 RAM.

### Python package

We compare `abess` Python package with `scikit-learn` on linear regression and logistic regression. Results are presented in the below figure:

![](./docs/image/timings.png)

It can be see that `abess` uses the least runtime to find the solution. This results can be reproduced by running the command in shell:

```shell
$ python abess/docs/simulation/Python/timings.py
```

### R package

We compare `abess` R package with three widely used R packages: `glmnet`, `ncvreg`, and `L0Learn`.
We get the runtime comparison results:

![](docs/image/r_runtime.png)

Compared with other packages,
`abess` shows competitive computational efficiency,
and achieves the best computational power when variables have a large correlation.

Conducting the following command in shell can reproduce the above results in R:

```shell
$ Rscript abess/docs/simulation/R/timings.R
```

## Open source software

`abess` is a free software and its source code is publicly available on [Github](https://github.com/abess-team/abess). The core framework is programmed in C++, and user-friendly R and Python interfaces are offered. You can redistribute it and/or modify it under the terms of the [GPL-v3 License](https://www.gnu.org/licenses/gpl-3.0.html). We welcome contributions for `abess`, especially stretching `abess` to the other best subset selection problems.

## What's New

New features version `0.4.11`:

- New `slide()` function (R) and `IsingModel` class (Python) for sparse Ising model estimation via the SLIDE algorithm. Reference: Chen et al. (2025), *JASA*.
- sklearn 1.6/1.7 and NumPy 2.x compatibility.
- Free-threaded Python support on Windows.
- `sample_weight` equivalence fixes for weighted regression.

New features version `0.4.8`:

- Support no-intercept GLM (`fit_intercept=False` / `fit.intercept`).
- Beta range restriction (`beta_high`/`beta_low`) for non-negative fitting.
- Support AUC criterion for logistic/multinomial regression.
- Linux ARM64 (aarch64) wheel builds.

New features version `0.4.7`:

- Support limiting beta into a range by clipping method. One application is to perform non-negative fitting.
- Support no-intercept model for most regressors in ``abess.linear`` with argument ``fit_intercept=False``. We assume that the data has been centered for these models.
- Support AUC criterion for Logistic and Multinomial Regression.

New features version `0.4.6`:

- Support no-intercept model for most regressors in `abess.linear` with argument `fit_intercept=False`. We assume that the data has been centered for these models. (Python)
- `abess` can be used via `mlr3extralearners` as learners `regr.abess` and `classif.abess`. (R)
- Use [CMake](https://cmake.org/) on compiling to increase scalability.
- Support score functions for all GLM models. (Python)
- Rearrange some arguments in Python package to improve legibility. Please check the latest [API document](https://abess.readthedocs.io/en/latest/Python-package/index.html). (Python)

## Citation

If you use `abess` or reference our tutorials in a presentation or publication, we would appreciate citations of our library.

> Zhu Jin, Xueqin Wang, Liyuan Hu, Junhao Huang, Kangkang Jiang, Yanhang Zhang, Shiyun Lin, and Junxian Zhu. "abess: A Fast Best-Subset Selection Library in Python and R." Journal of Machine Learning Research 23, no. 202 (2022): 1-7.

The corresponding BibteX entry:

```
@article{JMLR:v23:21-1060,
  author  = {Jin Zhu and Xueqin Wang and Liyuan Hu and Junhao Huang and Kangkang Jiang and Yanhang Zhang and Shiyun Lin and Junxian Zhu},
  title   = {abess: A Fast Best-Subset Selection Library in Python and R},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {202},
  pages   = {1--7},
  url     = {http://jmlr.org/papers/v23/21-1060.html}
}
```

If you use the SLIDE algorithm for Ising model reconstruction, please also cite:

> Chen, Xuanyu, Jin Zhu, Junxian Zhu, Xueqin Wang, and Heping Zhang. "Reconstruct Ising Model with Global Optimality via SLIDE." Journal of the American Statistical Association (2025). doi:10.1080/01621459.2025.2571245.

The corresponding BibTeX entry:

```
@article{chen2025slide,
  author  = {Xuanyu Chen and Jin Zhu and Junxian Zhu and Xueqin Wang and Heping Zhang},
  title   = {Reconstruct Ising Model with Global Optimality via SLIDE},
  journal = {Journal of the American Statistical Association},
  year    = {2025},
  doi     = {10.1080/01621459.2025.2571245}
}
```

## References

- Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang (2020). A polynomial algorithm for best-subset selection problem. Proceedings of the National Academy of Sciences, 117(52):33117-33123.
- Pölsterl, S (2020). scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. J. Mach. Learn. Res., 21(212), 1-6.
- Yanhang Zhang, Junxian Zhu, Jin Zhu, and Xueqin Wang. A splicing approach to best
subset of groups selection. INFORMS Journal on Computing, 35(1):104–119, 2023. doi:
10.1287/ijoc.2022.1241.
- Qiang Sun and Heping Zhang (2020). Targeted Inference Involving High-Dimensional Data Using Nuisance Penalized Regression, Journal of the American Statistical Association, DOI: 10.1080/01621459.2020.1737079.
- Zhu Jin, Xueqin Wang, Liyuan Hu, Junhao Huang, Kangkang Jiang, Yanhang Zhang, Shiyun Lin, and Junxian Zhu. "abess: A Fast Best-Subset Selection Library in Python and R." Journal of Machine Learning Research 23, no. 202 (2022): 1-7.
- Chen, Xuanyu, Jin Zhu, Junxian Zhu, Xueqin Wang, and Heping Zhang (2025). Reconstruct Ising Model with Global Optimality via SLIDE. Journal of the American Statistical Association. doi:10.1080/01621459.2025.2571245.


## Contributions

<div align="center">
<!-- <details> -->

<summary>Thanks for the following support</summary>

### Stargazers


<div align="center">

[![Stargazers repo roster for @abess-team/abess](http://reporoster.com/stars/abess-team/abess)](https://github.com/abess-team/abess/stargazers)



</div>

### Forkers

<div align="center" >

[![Forkers repo roster for @abess-team/abess](http://reporoster.com/forks/abess-team/abess)](https://github.com/abess-team/abess/network/members)

</div>
<br/></details><br/>

</div>

Any kind of contribution to `abess` would be highly appreciated! Please check the [contributor's guide](https://abess.readthedocs.io/en/latest/Contributing/).

- Bug report via [github issues](https://github.com/abess-team/abess/issues)
