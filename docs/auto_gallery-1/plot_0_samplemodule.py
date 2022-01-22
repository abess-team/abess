"""
SampleModule example
====================

This example will demonstrate the ``power`` function and ``class_power`` from
our package 'SampleModule'.
"""


import matplotlib.pyplot as plt

_ = plt.plot([1,2,3])

#%%
# pandas dataframes have a html representation, and this is captured:

import pandas as pd

df = pd.DataFrame({'col1': [1,2,3],
                   'col2': [4,5,6]})
df

s = pd.Series([1,2,3])