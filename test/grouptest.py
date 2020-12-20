import pandas as pd
from bess.linear import GroupPdasCox
import numpy as np



x = pd.read_csv("x.csv")
y = pd.read_csv("y.csv")

# print(x.head())
# print(y.head())

x = x.values
y = y.values
# print(y)

group = [int(i/3)+1 for i in range(300)]

print(1)
model = GroupPdasCox(path_type="seq", sequence=range(1,15), ic_type="bic")
model.fit(x,y,group=group)
print(2)

print(np.nonzero(model.beta))
print(model.beta[np.nonzero(model.beta)])
