Welcome to abess's documentation!
==========================================================================

.. raw:: html

   <!-- badges: start -->

|GithubAction build status| |codecov| |docs| |cran| |pypi| |pyversions| |License| |Codacy|

.. |Codacy| image:: https://app.codacy.com/project/badge/Grade/3f6e60a3a3e44699a033159633981b76 
   :target: https://www.codacy.com/gh/abess-team/abess/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=abess-team/abess&amp;utm_campaign=Badge_Grade
.. |Travis build status| image:: https://travis-ci.com/abess-team/abess.svg?branch=master
   :target: https://travis-ci.com/abess-team/abess
.. |GithubAction build status| image:: https://github.com/abess-team/abess/actions/workflows/main.yml/badge.svg?branch=master
   :target: https://github.com/abess-team/abess/actions
.. |codecov| image:: https://codecov.io/gh/abess-team/abess/branch/master/graph/badge.svg?token=LK56LHXV00
   :target: https://codecov.io/gh/abess-team/abess
.. |docs| image:: https://readthedocs.org/projects/abess/badge/?version=latest
   :target: https://abess.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. |cran| image:: https://img.shields.io/cran/v/abess?logo=R
   :target: https://cran.r-project.org/package=abess
.. |pypi| image:: https://badge.fury.io/py/abess.svg
   :target: https://badge.fury.io/py/abess
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/abess
.. |License| image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg 
   :target: http://www.gnu.org/licenses/gpl-3.0


Overview
============

**abess** (Adaptive BEst Subset Selection) library aims to solve general best subset selection, i.e., 
find a small subset of predictors such that the resulting model is expected to have the highest accuracy. 
The selection for best subset shows great value in scientific researches and practical application. 
For example, clinicians wants to know whether a patient is health or not  
based on the expression level of a few of important genes.

This library implements a generic algorithm framework to find the optimal solution in polynomial time [#1abess]_. 
This framework now supports the detection of best subset under: 
`linear regression`_, `(multi-class) classification`_, `censored-response modeling`_ [#4sksurv]_, 
`multi-response modeling (a.k.a. multi-tasks learning)`_, etc. 
It also supports the variants of best subset selection like 
`group best subset selection`_ [#2gbes]_ and `nuisance best subset selection`_ [#3nbes]_. 

.. _linear regression: https://abess.readthedocs.io/en/latest/Tutorial/LinearRegression.html
.. _(multi-class) classification: https://abess.readthedocs.io/en/latest/Tutorial/logi_and_multiclass.html
.. _counting-response modeling: https://abess.readthedocs.io/en/latest/Tutorial/PoissonRegression.html
.. _censored-response modeling: https://abess.readthedocs.io/en/latest/Tutorial/CoxRegression.html
.. _multi-response modeling (a.k.a. multi-tasks learning): https://abess.readthedocs.io/en/latest/Tutorial/MultiTaskLearning.html
.. _group best subset selection: https://abess.readthedocs.io/en/latest/Tutorial/advanced_features.html#Best-group-subset-selection
.. _nuisance best subset selection: https://abess.readthedocs.io/en/latest/Tutorial/advanced_features.html#Nuisance-Regression

Quick start
============

R package
-----------

Install abess from R CRAN by running the following command in R: 

.. code-block:: r

   install.packages("abess")


Best subset selection for linear regression on a simulated dataset in R:

.. code-block:: r

   library(abess)
   sim_dat <- generate.data(n = 300, p = 1000)
   abess(x = sim_dat[["x"]], y = sim_dat[["y"]])

See more examples analyzed with R in the tutorials available `here <https://abess-team.github.io/abess/articles/>`_.


Python package
-----------

To install abess, please view chapter: `Installation`_.

.. code-block:: shell

   $ pip install abess

.. _Installation: https://abess.readthedocs.io/en/latest/Installation.html

Import best subset selection solver for linear regression in a Python project:    

.. code-block:: python

   from abess.linear import abessLm

See more examples in the tutorials; the notebooks are available `here <https://abess.readthedocs.io/en/latest/Tutorial/index.html>`_.

Performance
===========

To show the power of abess in computation, 
we assess its timings of the CPU execution (seconds) on synthetic datasets, and compare to 
state-of-the-art variable selection methods. 
The variable selection and estimation result are deferred to Tutorial.

R package    
-----------
We compare abess R package with three widely used R packages: glmnet, ncvreg, L0Learn. 
Conducting the following commands in shell: 

.. code-block:: shell

   $ Rscript ./R-package/example/timings.R

we obtain the runtime comparison picture:

|Rpic1|

.. |Rpic1| image:: ./perform/Rtimings.png
   :width: 100%

Python package   
-----------

We compare abess Python package with scikit-learn on linear and logistic regression.
Results are presented in the below figure, and can be reproduce by running the commands in shell:

.. code-block:: shell

   $ python ./docs/perform/timings.py


|pic1| 

.. |pic1| image:: ./perform/timings.png
   :width: 100%

In both R and Python environments, 
abess reaches a high efficient performance especially in linear regression where it gives the fastest solution.

Open source software       
==========

abess is a free software and its source code are publicly available in `Github`_.  
The core framework is programmed in C++, and user-friendly R and Python interfaces are offered.
You can redistribute it and/or modify it under the terms of the `GPL-v3 License`_. 
We welcome contributions for abess, especially stretching abess to 
the other best subset selection problems. 

.. _github: https://github.com/abess-team/abess
.. _GPL-v3 License: https://www.gnu.org/licenses/gpl-3.0.html

.. Citation         
.. ==========

.. If you use abess or reference our tutorials in a presentation or publication, we would appreciate citations of our library.
.. | Zhu J, Pan W, Zheng W, Wang X (2021). “Ball: An R Package for Detecting Distribution Difference and Association in Metric Spaces.” arXiv, 97(6), 1–31. doi: 10.18637/jss.v097.i06.

.. The corresponding BibteX entry:

References
=========

.. [#1abess] Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang (2020). A polynomial algorithm for best-subset selection problem. Proceedings of the National Academy of Sciences, 117(52):33117-33123.

.. [#4sksurv] Pölsterl, S (2020). scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. J. Mach. Learn. Res., 21(212), 1-6.

.. [#2gbes] Yanhang Zhang, Junxian Zhu, Jin Zhu, and Xueqin Wang (2021). Certifiably Polynomial Algorithm for Best Group Subset Selection. arXiv preprint arXiv:2104.12576.

.. [#3nbes] Qiang Sun and Heping Zhang (2020). Targeted Inference Involving High-Dimensional Data Using Nuisance Penalized Regression, Journal of the American Statistical Association, DOI: 10.1080/01621459.2020.1737079.
    
