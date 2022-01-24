"""
Data download example
=====================

This example shows one way of dealing with large data files required for your
examples.

The ``download_data`` function first checks if the data has already been
downloaded, looking in either the data directory saved the configuration file
(by default ``~/.sg_template``) or the default data directory. If the data has
not already been downloaded, it downloads the data from the url and saves the
data directory to the configuration file. This allows you to use the data
again in a different example without downloading it again.

Note that examples in the gallery are ordered according to their filenames, thus
the number after 'plot\_' dictates the order the example appears in the gallery.
"""

import numpy as np
# from abess.datasets import make_glm_data
np.random.seed(0)

n = 300
p = 1000
k = 3
real_coef = np.zeros(p)
real_coef[[0, 1, 4]] = 3, 1.5, 2





