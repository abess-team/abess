"""
Data download example 2
=======================

We can use the same ``iris`` dataset in this example, without downloading it
twice as we know ``data_download`` will check if the data has already been
downloaded.
"""

import numpy as np
# from abess.datasets import make_glm_data
np.random.seed(0)

n = 300
p = 1000
k = 3
real_coef = np.zeros(p)
real_coef[[0, 1, 4]] = 3, 1.5, 2