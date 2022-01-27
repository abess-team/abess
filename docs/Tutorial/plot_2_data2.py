"""
Data download example 2
=======================

We can use the same ``iris`` dataset in this example, without downloading it
twice as we know ``data_download`` will check if the data has already been
downloaded.
"""

# #%%
# # pandas dataframes have a html representation, and this is captured:

# import pandas as pd

# df = pd.DataFrame({'col1': [1,2,3],
#                    'col2': [4,5,6]})
# df

# s = pd.Series([1,2,3])

# #%%
# # test numpy

# import numpy as np 
# x = np.empty([3,2], dtype = int) 
# print (x)

# #%%
# # test abess
import numpy as np 
from abess.datasets import make_glm_data
np.random.seed(0)

n = 300
p = 1000
k = 3
real_coef = np.zeros(p)
real_coef[[0, 1, 4]] = 3, 1.5, 2
data1 = make_glm_data(n = n, p = p, k = k, family = "gaussian", coef_ = real_coef)


print(data1.x.shape)
print(data1.y.shape)

#%% 
# plot 

import matplotlib.pyplot as plt

_ = plt.plot([1,2,3])
