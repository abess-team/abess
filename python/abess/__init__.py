#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Site    :
# @File    : __init__.py

from abess.linear import (
    LinearRegression,
    LogisticRegression,
    CoxPHSurvivalAnalysis,
    PoissonRegression,
    MultiTaskRegression,
    MultinomialRegression,
    GammaRegression)
from abess.decomposition import (SparsePCA, RobustPCA)
from abess.datasets import (make_glm_data, make_multivariate_glm_data)

# To be deprecated in version 0.5.0
from abess.linear import (
    abessLogistic,
    abessLm,
    abessCox,
    abessPoisson,
    abessMultigaussian,
    abessMultinomial,
    abessGamma)
from abess.pca import (abessPCA, abessRPCA)
