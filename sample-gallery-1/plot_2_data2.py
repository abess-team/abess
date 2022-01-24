"""
Data download example 2
=======================

We can use the same ``iris`` dataset in this example, without downloading it
twice as we know ``data_download`` will check if the data has already been
downloaded.
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