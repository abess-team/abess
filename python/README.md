# bess: An python Package for Best Subset Selection


## Introduction

One of the main tasks of statistical modeling is to exploit the association between
a response variable and multiple predictors. Linear model (LM), as a simple parametric
regression model, is often used to capture linear dependence between response and
predictors. Generalized linear model (GLM) can be considered as
the extensions of linear model, depending on the types of responses. Parameter estimation in these models
can be computationally intensive when the number of predictors is large. Meanwhile,
Occam's razor is widely accepted as a heuristic rule for statistical modeling,
which balances goodness of fit and model complexity. This rule leads to a relative 
small subset of important predictors. 

**bess** package provides solutions for best subset selection problem for sparse LM,
and GLM models.

We consider a primal-dual active set (PDAS) approach to exactly solve the best subset
selection problem for sparse LM and GLM models. The PDAS algorithm for linear 
least squares problems was first introduced by [Ito and Kunisch (2013)](https://iopscience.iop.org/article/10.1088/0266-5611/30/1/015001)
and later discussed by [Jiao, Jin, and Lu (2015)](https://arxiv.org/abs/1403.0515) and [Huang, Jiao, Liu, and Lu (2017)](https://arxiv.org/abs/1701.05128). 
It utilizes an active set updating strategy and fits the sub-models through use of
complementary primal and dual variables. We generalize the PDAS algorithm for 
general convex loss functions with the best subset constraint, and further 
extend it to support both sequential and golden section search strategies
for optimal k determination. 


## Install

Python Version
- python >= 3.5

Modules needed
- numpy 

The package has been publish in PyPI. You can easy install by:
```sh
$ pip install bess
```

## Example
```python
### PdasLm sample
from bess.linear import *
import numpy as np

np.random.seed(12345)   # fix seed to get the same result
x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
noise = np.random.normal(0, 1, 100)
y = np.matmul(x, beta) + noise

### Sparsity known
model = PdasLm(path_type="seq", sequence=[5])
model.fit(X=x, y=y)
model.predict(x)

### Sparsity unknown
# path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
model = PdasLm(path_type="seq", sequence=range(1,10))
model.fit(X=x, y=y)
model.predict(x)

# path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
model = PdasLm(path_type="pgs", s_max=20)
model.fit(X=x, y=y)
model.predict(x)


### PdasLogistic sample
np.random.seed(12345)
x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
xbeta = np.matmul(x, beta)
p = np.exp(xbeta)/(1+np.exp(xbeta))
y = np.random.binomial(1, p)

### Sparsity known
model = PdasLogistic(path_type="seq", sequence=[5])
model.fit(X=x, y=y)
model.predict(x)

### Sparsity unknown
# path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
model = PdasLogistic(path_type="seq", sequence=range(1,10))
model.fit(X=x, y=y)
model.predict(x)

# path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
model = PdasLogistic(path_type="pgs")
model.fit(X=x, y=y)
model.predict(x)


### PdasPoisson sample
np.random.seed(12345)
x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
lam = np.exp(np.matmul(x, beta))
y = np.random.poisson(lam=lam)

### Sparsity known
model = PdasPoisson(path_type="seq", sequence=[5])
model.fit(X=x, y=y)
model.predict(x)

### Sparsity unknown
# path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
model = PdasPoisson(path_type="seq", sequence=range(1,10))
model.fit(X=x, y=y)
model.predict(x)

# path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
model = PdasPoisson(path_type="pgs")
model.fit(X=x, y=y)
model.predict(x)


### PdasCox sample
from bess.gen_data import gen_data
np.random.seed(12345)
data = gen_data(100, 200, family="cox", k=5, rho=0, sigma=1, c=10)

### Sparsity known
model = PdasCox(path_type="seq", sequence=[5])
model.fit(data.x, data.y, is_normal=True)
model.predict(data.x)

### Sparsity unknown
# path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
model = PdasCox(path_type="seq", sequence=range(1,10))
model.fit(data.x, data.y)
model.predict(data.x)

# path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
model = PdasCox(path_type="pgs")
model.fit(data.x, data.y)
model.predict(data.x)

```

## Reference

- Wen, C. , Zhang, A. , Quan, S. , & Wang, X. . (2017). [Bess: an r package for best subset selection in linear, logistic and coxph models](https://arxiv.org/pdf/1709.06254.pdf)


## Bug report

Connect to [@Jiang-Kangkang](https://github.com/Jiang-Kangkang), or send an email to Jiang Kangkang(jiangkk3@mail2.sysu.edu.cn)

