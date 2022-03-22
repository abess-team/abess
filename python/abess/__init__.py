#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Site    :
# @File    : __init__.py

__version__ = "0.4.5"
__author__ = ("Jin Zhu, Kangkang Jiang, "
              "Junhao Huang, Yanhang Zhang, "
              "Yanhang Zhang, Shiyun Lin, "
              "Junxian Zhu, Xueqin Wang")

from .linear import (
    LinearRegression,
    LogisticRegression,
    CoxPHSurvivalAnalysis,
    PoissonRegression,
    MultiTaskRegression,
    MultinomialRegression,
    GammaRegression,
    OrdinalRegression
)
from .decomposition import (SparsePCA, RobustPCA)
from .datasets import (make_glm_data, make_multivariate_glm_data)

# To be deprecated in version 0.6.0
from .linear import (  # noqa
    abessLm, abessLogistic, abessCox, abessPoisson,
    abessMultigaussian, abessMultinomial, abessGamma
)
from .pca import (abessPCA, abessRPCA)  # noqa

__all__ = [
    # linear
    "LinearRegression",
    "LogisticRegression",
    "CoxPHSurvivalAnalysis",
    "PoissonRegression",
    "MultiTaskRegression",
    "MultinomialRegression",
    "GammaRegression",
    "OrdinalRegression",
    # decomposition
    "SparsePCA",
    "RobustPCA",
    # datasets
    "make_glm_data",
    "make_multivariate_glm_data"
]
