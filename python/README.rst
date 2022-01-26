   
|logopic|      

.. |logopic| image:: https://github.com/abess-team/abess/raw/master/docs/image/icon_long.png    


|Python build status| |R build status| |codecov| |docs| |cran| |pypi| |pyversions| |License| |Codacy| |CodeFactor|

.. |Codacy| image:: https://app.codacy.com/project/badge/Grade/3f6e60a3a3e44699a033159633981b76 
   :target: https://www.codacy.com/gh/abess-team/abess/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=abess-team/abess&amp;utm_campaign=Badge_Grade
.. |Travis build status| image:: https://travis-ci.com/abess-team/abess.svg?branch=master
   :target: https://travis-ci.com/abess-team/abess
.. |Python build status| image:: https://github.com/abess-team/abess/actions/workflows/python_test.yml/badge.svg?branch=master
   :target: https://github.com/abess-team/abess/actions/workflows/python_test.yml
.. |R build status| image:: https://github.com/abess-team/abess/actions/workflows/r_test.yml/badge.svg?branch=master
   :target: https://github.com/abess-team/abess/actions/workflows/r_test.yml
.. |codecov| image:: https://codecov.io/gh/abess-team/abess/branch/master/graph/badge.svg?token=LK56LHXV00
   :target: https://codecov.io/gh/abess-team/abess
.. |docs| image:: https://readthedocs.org/projects/abess/badge/?version=latest
   :target: https://abess.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. |R website| image:: https://github.com/abess-team/abess/actions/workflows/r_website.yml
   :target: https://abess-team.github.io/abess/
.. |cran| image:: https://img.shields.io/cran/v/abess?logo=R
   :target: https://cran.r-project.org/package=abess
.. |pypi| image:: https://badge.fury.io/py/abess.svg
   :target: https://badge.fury.io/py/abess
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/abess
.. |License| image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg 
   :target: http://www.gnu.org/licenses/gpl-3.0
.. |CodeFactor| image:: https://www.codefactor.io/repository/github/abess-team/abess/badge 
   :target: https://www.codefactor.io/repository/github/abess-team/abess

Overview
============

**abess** (Adaptive BEst Subset Selection) library aims to solve general best subset selection, i.e., 
find a small subset of predictors such that the resulting model is expected to have the highest accuracy. 
The selection for best subset shows great value in scientific researches and practical application. 
For example, clinicians wants to know whether a patient is health or not  
based on the expression level of a few of important genes.

This library implements a generic algorithm framework to find the optimal solution in an extremely fast way [#1abess]_. 
This framework now supports the detection of best subset under: 
`linear regression`_, `(multi-class) classification`_, `censored-response modeling`_ [#4sksurv]_, 
`multi-response modeling (a.k.a. multi-tasks learning)`_, etc. 
It also supports the variants of best subset selection like 
`group best subset selection`_ [#2gbes]_ and `nuisance best subset selection`_ [#3nbes]_. 
Especially, the time complexity of (group) best subset selection for linear regression is certifiably polynomial [#1abess]_ [#2gbes]_.

.. _linear regression: https://abess.readthedocs.io/en/latest/Tutorial/LinearRegression.html
.. _(multi-class) classification: https://abess.readthedocs.io/en/latest/Tutorial/logi_and_multiclass.html
.. _counting-response modeling: https://abess.readthedocs.io/en/latest/Tutorial/PoissonRegression.html
.. _censored-response modeling: https://abess.readthedocs.io/en/latest/Tutorial/CoxRegression.html
.. _multi-response modeling (a.k.a. multi-tasks learning): https://abess.readthedocs.io/en/latest/Tutorial/MultiTaskLearning.html
.. _group best subset selection: https://abess.readthedocs.io/en/latest/Tutorial/advanced_features.html#Best-group-subset-selection
.. _nuisance best subset selection: https://abess.readthedocs.io/en/latest/Tutorial/advanced_features.html#Nuisance-Regression

Quick start
============

Install the stable abess Python package from Pypi: 

.. code-block:: shell

   $ pip install abess

Best subset selection for linear regression on a simulated dataset in Python:    

.. code-block:: python

   from abess.linear import LinearRegression
   from abess.datasets import make_glm_data
   sim_dat = make_glm_data(n = 300, p = 1000, k = 10, family = "gaussian")
   model = LinearRegression()
   model.fit(sim_dat.x, sim_dat.y)

See more examples analyzed with Python in the tutorials; the notebooks are available `here <https://abess.readthedocs.io/en/latest/Tutorial/index.html>`_.

Runtime Performance
===================

To show the power of abess in computation, 
we assess its timings of the CPU execution (seconds) on synthetic datasets, and compare to 
state-of-the-art variable selection methods. 
The variable selection and estimation results are deferred to `performance`_.

.. _performance: https://abess.readthedocs.io/en/latest/Tutorial/power_of_abess.html

We compare abess Python package with scikit-learn on linear and logistic regression.
Results are presented in the below figure, and can be reproduce by running the commands in shell:

.. code-block:: shell

   $ python ./simulation/Python/timings.py

we obtain the runtime comparison picture:

|pic1| 

.. |pic1| image:: https://github.com/abess-team/abess/raw/master/docs/image/timings.png
   :width: 100%

abess reaches a high efficient performance especially in linear regression where it gives the fastest solution.

Open source software     
====================

abess is a free software and its source code are publicly available in `Github`_.  
The core framework is programmed in C++, and user-friendly R and Python interfaces are offered.
You can redistribute it and/or modify it under the terms of the `GPL-v3 License`_. 
We welcome contributions for abess, especially stretching abess to 
the other best subset selection problems. 

.. _github: https://github.com/abess-team/abess
.. _GPL-v3 License: https://www.gnu.org/licenses/gpl-3.0.html

Citation         
==========

If you use `abess` or reference our tutorials in a presentation or publication, we would appreciate citations of our library [#5abesslib]_.

| Jin Zhu, Liyuan Hu, Junhao Huang, Kangkang Jiang, Yanhang Zhang, Shiyun Lin, Junxian Zhu, Xueqin Wang (2021). “abess: A Fast Best Subset Selection Library in Python and R.” arXiv:2110.09697.

The corresponding BibteX entry:

.. code-block:: shell

   @article{zhu-abess-arxiv,
      author  = {Jin Zhu and Liyuan Hu and Junhao Huang and Kangkang Jiang and Yanhang Zhang and Shiyun Lin and Junxian Zhu and Xueqin Wang},
      title   = {abess: A Fast Best Subset Selection Library in Python and R},
      journal = {arXiv:2110.09697},
      year    = {2021},
   }

References
==========

.. [#1abess] Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang (2020). A polynomial algorithm for best-subset selection problem. Proceedings of the National Academy of Sciences, 117(52):33117-33123.

.. [#4sksurv] Pölsterl, S (2020). scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. J. Mach. Learn. Res., 21(212), 1-6.

.. [#2gbes] Yanhang Zhang, Junxian Zhu, Jin Zhu, and Xueqin Wang (2021). Certifiably Polynomial Algorithm for Best Group Subset Selection. arXiv preprint arXiv:2104.12576.

.. [#3nbes] Qiang Sun and Heping Zhang (2020). Targeted Inference Involving High-Dimensional Data Using Nuisance Penalized Regression, Journal of the American Statistical Association, DOI: 10.1080/01621459.2020.1737079.
    
.. [#5abesslib] Jin Zhu, Liyuan Hu, Junhao Huang, Kangkang Jiang, Yanhang Zhang, Shiyun Lin, Junxian Zhu, and Xueqin Wang (2021). abess: A Fast Best Subset Selection Library in Python and R. arXiv preprint arXiv:2110.09697.
