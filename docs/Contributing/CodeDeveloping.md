# Code Developing

Before developing the code, please make sure:
- following the [Installation](../Installation.md), the code in github works on your device;
- read the [Architecture](Before.md) of abess library.

## Core C++

The main files related to the core are in `abess/python/src`. Among them, some important files:

- `Algorithm.h` records the implement of each concrete algorithm; 
- `abess.cpp` contain the calling procedure.

If you want to add a new algorithm, both of them should be noticed.



In `Algorithm.h`, we give a base class *Algorithm*, and the new method should inheritate it. The concrete algorithms are programmed in the subclass of Algorithm by rewriting the virtual function interfaces of class *Algorithm*. Besides, the implementation is modularized such that you can easily extend the package. 

We have implemented some GLM algorithms that you can check them on [`abess/python/src/AlgorithmGLM.h`](https://github.com/abess-team/abess/blob/master/python/src/AlgorithmGLM.h). 

>  The format of a new algorithm's name is "**abess+your_algorithm**", which means that using abess to solve the problem.

Take PCA as an example, the name should be `abessPCA`. Now we can create a new file, named `AlgorithmPCA.h`, and define a concrete algorithm like: [[code link]](https://github.com/abess-team/abess/blob/master/python/src/AlgorithmPCA.h)

```Cpp
template <class T4>
class abessPCA : public Algorithm<...>
{
public:
    abessPCA(...) : Algorithm<...>::Algorithm(...){};
    ~abessPCA(){};
 
    void primary_model_fit(...){...};
        // solve the subproblem under given active set
        // record the sparse answer in variable "beta"

    double neg_loglik_loss(...){...};
        // define and compute loss under given active set
        // return the current loss

    void sacrifice(...){...};
        // define and compute sacrifice for all variables (both forward and backward)
        // record sacrifice in variable "bd"

    double effective_number_of_parameter(...){...};
		// return effective number of parameter
}
```

Note that `sacrifice` function here would compute "forward/backward sacrifices" and record them in `bd`.

- For active variable, the lower (backward) sacrifice is, the more likely it will be dropped;
- For inactive variable, the higher (forward) sacrifice is, the more likely it will come into use.

Since it can be quite different to compute sacrifices for different problem, you may need to derivate by yourself.



After that, turn to `abess.cpp` and you will find some `new` command like: [[code link]](https://github.com/abess-team/abess/blob/master/python/src/abess.cpp#:~:text=algorithm_uni_dense%20%3D%20new%20abessLm)

```Cpp
if (model_type == 1)
{
    algorithm_uni_dense = new abessLm<...>(...);
}
else if ...
```

They are used to request memory and call the algorithms in `Algorithm.h`. Here you need to add your own algorithm and give it a unused `model_type` number, e.g. 7.

```Cpp
...
else if (model_type == 7) // indicates PCA
{
    algorithm_uni_dense = new abessPCA<...>(...);
}
```

Note that there is a small difference between single response variable and multiple response variables question:

```Cpp
...
else if (model_type == 123) // indicates a multiple response algorithm
{
    // name it "mul"
    algorithm_mul_dense = new abessXXX<...>(...);
}
```

What is more, the variable named `algorithm_list_uni_dense[i]` or `algorithm_list_mul_dense[i]` is similar to what we said above. They are for parallel computing. [[code link]](https://github.com/abess-team/abess/blob/master/python/src/abess.cpp#:~:text=algorithm_list_uni_dense%5Bi%5D%20%3D%20new%20abessLm)

Hence, you should also add:

```Cpp
...
else if (model_type == 7)
{
    // for single response variable question, name it "uni"
    algorithm_list_uni_dense[i] = new abessPCA<...>(...);
}
```

or,

```Cpp
...
else if (model_type == 123)
{
    // for multiple response variables question, name it "mul"
    algorithm_list_mul_dense[i] = new abessXXX<...>(...);
}
```



After that, remember to include your new `.h` file in `abess.cpp`, like [this](https://github.com/abess-team/abess/blob/master/python/src/abess.cpp#:~:text=%23include%20%22AlgorithmPCA.h%22). Now your new method has been connected to the whole frame. In the next section, we focus on how to build R or Python package based on the core code.

## R & Python Package

### R Package

To make sure your code available for R, run 
```powershell
R CMD INSTALL R-package
```
Then, this package would be installed into R session if the R package dependence (`Rcpp` and `Matrix`) have been installed. 

After that, the object in R can be passed to Cpp via the 
unified API `abessCpp`. We strongly suggest the R function is named as `abessXXX` and use `roxygen2` to write R documentation and `devtools` to configure your package. 

### Python Package

To make your code available for Python, `cd` into directory `abess/python` and run `$ python setup.py install`. (Same steps in [Installation](https://abess.readthedocs.io/en/latest/Installation.html#latest-release).)

It may take a few minutes to install:

- if the installation throw some errors, it means that the C++ code may be wrong;
- if the installation runs without errors, it will finish with message like "*Finished processing dependencies for abess*". 

Now a file named `cabess.py` will be appeared in the directory `abess/python/src`, which help to link Python and C++. You need to move it into directory `abess/python/abess` and replace the duplicated file there.

Then create a new python file in `abess/python/abess` or open the existed file, such as `abess/python/abess/linear.py`, to add a python api for your new method. 

Here we create `abess/python/abess/pca.py`. A simple new method can be added like: [[code link]](https://github.com/abess-team/abess/blob/master/python/abess/pca.py).

```Python
# all methods are based on the temple class `bess_base`
from .bess_base import bess_base

class abessPCA(bess_base): 
    """
    Here is some introduction.
    """
    def __init__(self, ...):
        super(abessXXX, self).__init__(
            algorithm_type="abess", 
            model_type="PCA", 
            # ...
        )
    def custom_function(self, ...):
        # ...
```

As an example, we define two new functions (`ratio` and `transform`) and override the `fit` function for `abessPCA`. [[code link]](https://github.com/abess-team/abess/blob/master/python/abess/pca.py).

Then, the final step is to link this Python class with the model type number (it has been defined in Section **Core C++**). In `bess_base.py`, you can find somewhere like (in the `fit` function): 

```Python
if self.model_type == "Lm":
    model_type_int = 1
elif # ...
```

Note that the new PCA method has been related to number "7" above, so we need to denote `model_type_int = 7` in our `fit` function. 

After finished all changes before, run `$ python setup.py install` again and this time the installation would be finished quickly. 

Congratulation! Your work can now be used by:

```Python
from abess.pca import abessPCA
```

#### bess_base

As we show above, any new methods are based on `bess_base`, which can be found in `bess_base.py`: [[code link]](https://github.com/abess-team/abess/blob/master/python/abess/bess_base.py)

```python
from sklearn.base import BaseEstimator
class bess_base(BaseEstimator):
     def __init__(...):
        #...
     def fit(...):	# abess process, warp with cpp
        #...
     #...
```

Actually, it is based on `sklearn.base.BaseEstimator` [[link]](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html). Two methods, `get_params` and `set_params` are offered in this base class. 

In our package, we write an method called `fit` to realize the abess process. Of cause, you can also override it like `abessPCA`.

## Miscellaneous

### Code style
New R code should follow the tidyverse [style guide](https://style.tidyverse.org/). You can use the styler package to apply these styles. 
New Python code...
Please don’t restyle code that has nothing to do with your code.

### Test cases

It is always a good habit to do some test for the changed package.
Contributions with test cases included are easier to accept.

We use [testthat](https://cran.r-project.org/web/packages/testthat) for unit tests in R and
[pytest](https://docs.pytest.org/) in Python. You may need to install first.

You can find some examples here and please feel free to add your test code into it (or create a new test file) under the test folder: 

- [R test folder](https://github.com/abess-team/abess/tree/master/R-package/tests/testthat): `abess/R-package/tests/testthat`.
- [Python test folder](https://github.com/abess-team/abess/tree/master/python/pytest): `abess/python/pytest`.

A good test code should contain:

- possible input modes as well as some wrong input;
- check whether the output is expected; 
- possible extreme cases;

All test under pytest folder should be checked after coding.

