Code Developing
===============

In this tutorial, we will show you how to develop a new algorithm with
abess procedure.

Before developing the code, please make sure: - following the
`Installation <../Installation.md>`__, the code in github works on your
device; - read the `Architecture <Before.md>`__ of abess library.

Core C++
--------

The main files related to the core are in ``abess/python/src``, which
are written in C++. Among them, some important files:

-  ``api.cpp/api.h`` contain the API’s, which are the entrance for both
   R & Python.
-  ``AlgorithmXXX.h`` records the implement of each concrete algorithm;

If you want to add a new algorithm, all of them should be noticed.

Besides, we have implemented some GLM algorithms on
``abess/python/src/AlgorithmGLM.h``\ `[code
temp] <https://github.com/abess-team/abess/blob/master/python/src/AlgorithmGLM.h>`__
and PCA algorithm on ``abess/python/src/AlgorithmPCA.h``\ `[code
temp] <https://github.com/abess-team/abess/blob/master/python/src/AlgorithmPCA.h>`__.
You can check them to help your own developing.

Write an API
~~~~~~~~~~~~

   API’s are all defined in the ``abess/python/src/api.cpp``\ `[code
   temp] <https://github.com/abess-team/abess/blob/master/python/src/api.cpp>`__
   and the related header file ``abess/python/src/api.h``\ `[code
   temp] <https://github.com/abess-team/abess/blob/master/python/src/api.h>`__.
   We have written some API functions (e.g. ``abessGLM_API()``), so you
   can either add a new function for the new algorithm or simply add
   into existing one.

First of all, the algorithm name and its number should be determined.

The format of a new algorithm’s name is “**abess+your_algorithm**”,
which means that using abess to solve the problem, and its number should
be an integer unused by others.

   In the following part, we suppose to create an algorthm named
   ``abess_new_algorithm`` with number 123.

Next, four important data type should be determined:

-  T1 : type of Y
-  T2 : type of coefficients
-  T3 : type of intercept
-  T4 : type of X

The algorithm variable are based on them: `[code
temp] <https://github.com/abess-team/abess/blob/master/python/src/api.cpp#:~:text=vector%3CAlgorithm%3C>`__

.. code:: cpp

   vector<Algorithm<{T1}, {T2}, {T3}, {T4}>*> algorithm_list(algorithm_list_size);

..

   Take ``abessLm`` (the linear regression on abess) as an example,

   -  Y: dense vector
   -  Coefficients: dense vector
   -  Intercept: numeric
   -  X: dense/sparse matrix

   so that we define:

   .. code:: cpp

      vector<Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::MatrixXd> *> algorithm_list_uni_dense(algorithm_list_size);
      vector<Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, Eigen::SparseMatrix<double>> *> algorithm_list_uni_sparse(algorithm_list_size);

After that, request memory to initial the algorithm: `[code
temp] <https://github.com/abess-team/abess/blob/master/python/src/api.cpp#:~:text=%7B-,if%20(model_type%20%3D%3D%201),%7B,-abessLm%3CEigen%3A%3AMatrixXd>`__

.. code:: cpp

   for (int i = 0; i < algorithm_list_size; i++)
   {
       if (model_type == 123)    // number of algorithm
       {
           algorithm_list[i] = new abessLm<{T4}>(...);
       }
   }

Finally, call ``abessWorkflow()``, which would compute the result:
`[code
temp] <https://github.com/abess-team/abess/blob/master/python/src/api.cpp#:~:text=Eigen%3A%3AVectorXd%20y_vec%20%3D%20y.col(0).eval()%3B-,out_result%20%3D%20abessWorkflow,-%3CEigen%3A%3AVectorXd%2C%20Eigen%3A%3AVectorXd%2C%20double%2C%20Eigen%3A%3AMatrixXd>`__

.. code:: cpp

   // "List" is a preset structure to store results
   List out_result = abessWorkflow<{T1}, {T2}, {T3}, {T4}>(..., algorithm_list);

Implement your Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~

   The implemented algorithms are stored in
   ``abess/python/src/AlgorithmXXX.h``. We have implemented some
   algorithms (e.g. ``AlgorithmGLM.h``), so you can either create a new
   file containing new algorithm or simply add into existing one.

The new algorithm should inheritate a base class, called *Algorithm*,
which defined in ``Algorithm.h``. And then rewrite some virtual function
interfaces to fit specify problem. The implementation is modularized
such that you can easily extend the package.

.. raw:: html

   <!-- [NOT SUPPORTED]
   A simplest concrete algorithm looks like:  

   ```cpp
   // [NOT SUPPORTED]
   #include "Algorithm.h"

   template <class T4>
   class abess_new_algorithm : public Algorithm<{T1}, {T2}, {T3}, T4>  // T1, T2, T3 are the same as above, which are fixed.
   {
   public:
       // constructor and destructor
       abess_new_algorithm(...) : Algorithm<...>::Algorithm(...){};
       ~abess_new_algorithm(){};

       double loss_function(...){
           // define and compute loss under given active set
           // return the current loss
       };

       void g(...){
           // define the first order derivative of loss
       };

       void h(...){
           // define the second order derivative of loss
       };

   }
   ```

   The abess process can automatically use the loss and its derivatives to complete algorithm. However, it should be noted that if you want to achieve higher efficiency, a FULL concrete algorithm can be: [[code temp]](https://github.com/abess-team/abess/blob/master/python/src/AlgorithmGLM.h#:~:text=template%20%3Cclass%20T4%3E-,class%20abessLogistic,-%3A%20public%20Algorithm%3CEigen)

   -->

A concrete algorithm is like: `[code
temp] <https://github.com/abess-team/abess/blob/master/python/src/AlgorithmGLM.h#:~:text=template%20%3Cclass%20T4%3E-,class%20abessLogistic,-%3A%20public%20Algorithm%3CEigen>`__

.. code:: cpp

   #include "Algorithm.h"

   template <class T4>
   class abess_new_algorithm : public Algorithm<{T1}, {T2}, {T3}, T4>  // T1, T2, T3 are the same as above, which are fixed.
   {
   public:
       // constructor and destructor
       abess_new_algorithm(...) : Algorithm<...>::Algorithm(...){};
       ~abess_new_algorithm(){};

       void primary_model_fit(...){
           // solve the subproblem under given active set
           // record the sparse answer in variable "beta"
       };

       double loss_function(...){
           // define and compute loss under given active set
           // return the current loss      
       };

       void sacrifice(...){
           // define and compute sacrifice for all variables (both forward and backward)
           // record sacrifice in variable "bd"        
       };

       double effective_number_of_parameter(...){
           // return effective number of parameter        
       };
   }

Note that ``sacrifice`` function here would compute “forward/backward
sacrifices” and record them in ``bd``.

-  For active variable, the lower (backward) sacrifice is, the more
   likely it will be dropped;
-  For inactive variable, the higher (forward) sacrifice is, the more
   likely it will come into use.

..

   If you create a new file to store the algorithm, remember to include
   it inside ``abess/python/src/api.cpp``. `[code
   temp] <https://github.com/abess-team/abess/blob/master/python/src/api.cpp#:~:text=%23include%20%22AlgorithmGLM.h%22>`__

Now your new method has been connected to the whole frame. In the next
section, we focus on how to build R or Python package based on the core
code.

R & Python Package
------------------

R Package
~~~~~~~~~

To make sure your code available for R, run

.. code:: powershell

   R CMD INSTALL R-package

Then, this package would be installed into R session if the R package
dependence (``Rcpp`` and ``Matrix``) have been installed.

After that, the object in R can be passed to Cpp via the unified API
``abessCpp``. We strongly suggest the R function is named as
``abessXXX`` and use ``roxygen2`` to write R documentation and
``devtools`` to configure your package.

Python Package
~~~~~~~~~~~~~~

First of all, you should ensure the C++ code available for Python,
``cd`` into directory ``abess/python`` and run
``$ python setup.py install``. (Same steps in
`Installation <https://abess.readthedocs.io/en/latest/Installation.html#latest-release>`__)

It may take a few minutes to install:

-  if the installation throw some errors, it means that the C++ code may
   be wrong;
-  if the installation runs without errors, it will finish with message
   like “*Finished processing dependencies for abess*”.

Now a file named ``cabess.py`` will be appeared in the directory
``abess/python/src``, which help to link Python and C++. You need to
move it into directory ``abess/python/abess`` and replace the duplicated
file there.

Then create a new python file in ``abess/python/abess`` or open an
existed file, such as ``abess/python/abess/linear.py``, to add a python
API for your new method.

A simple new method can be added like: `[code
temp] <https://github.com/abess-team/abess/blob/master/python/abess/pca.py#:~:text=class%20abessPCA(bess_base)%3A>`__.

.. code:: python

   # all algorithms should inheritate the base class `bess_base`
   from .bess_base import bess_base

   class new_algorithm(bess_base): 
       """
       Here is some introduction.
       """
       def __init__(self, ...):
           super(abess_new_algorithm, self).__init__(
               algorithm_type="abess", 
               model_type="new_algorithm", 
               # other init
           )
       def fit(self, ...):
           # override `bess_base.fit()`, if necessary

       def custom_function(self, ...):
           # some custom functions, e.g. predict

The base class implements a ``fit`` function, which plays a role on
checking input and calling C++ API to compute results. You may want to
override it for custom features. `[code
temp] <https://github.com/abess-team/abess/blob/master/python/abess/pca.py#:~:text=def%20fit(self%2C%20X%3DNone%2C%20is_normal%3DFalse%2C%20group%3DNone%2C%20Sigma%3DNone%2C%20number%3D1%2C%20n%3DNone)%3A>`__.

Then, the final step is to link this Python class with the model type
number (it has been defined in Section **Core C++**). In the ``fit``
function, you would find somewhere like:

.. code:: python

   if self.model_type == "new_algorithm":
       model_type_int = 123    # same number in C++

Finally, don’t forget to import the new algorithm in
``abess/python/abess/__init__.py``.

Now run ``$ python setup.py install`` again and this time the
installation would be finished quickly. Congratulation! Your work can
now be used by:

.. code:: python

   from abess import new_algorithm

bess_base
^^^^^^^^^

As we show above, any new methods are based on ``bess_base``, which can
be found in ``bess_base.py``: `[code
link] <https://github.com/abess-team/abess/blob/master/python/abess/bess_base.py>`__

.. code:: python

   from sklearn.base import BaseEstimator
   class bess_base(BaseEstimator):
        def __init__(...):
           # some init
        def fit(...):  
           # check input, warp with cpp

Actually, it is based on ``sklearn.base.BaseEstimator`` `[code
link] <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`__.
Two methods, ``get_params`` and ``set_params`` are offered in this base
class.

In our package, we write an method called ``fit`` to realize the abess
process. Of cause, you can also override it like ``SparsePCA``.

Verify you result
-----------------

After programming the code, it is necessary to verify the contributed
function can return a reasonable result. Here, we share our experience
for it. Notice that the core our algorithm are forward and backward
sacrifices, as long as they are properly programming, the contributed
function would work well.

-  Check ``primary_model_fit`` and ``loss_function``

Secondly, we recommend you consider ``primary_model_fit`` for the
computation of backward sacrifices. To check whether it works well, you
can leverage the parameter ``always.include`` in R. Actually, when the
number of elements pass to ``always.include`` is equal to
``support.size`` (``always_include`` and ``support_size`` in Python),
our algorithm is no need to do variable selection since all element must
be selected, and thus, our implementation framework would just simply
solving a convex problem by conducting ``primary_model_fit`` and the
solution should match to (or close to) the function implemented in
R/Python. Take the PCA task as an example, we should expect that, the
results returned by ``abess``:

.. code:: r

   data(USArrests)
   abess_fit <- abesspca(USArrests, always.include = c(1:3), support.size = 3)
   as.vector(spca_fit[["coef"]])[1:3]

should match with that returned by the ``princomp`` function:

.. code:: r

   princomp_fit <- loadings(princomp(USArrests[, 1:3]))[, 1]
   princomp_fit

Actually, in our implementation, the results returned in two code blocks
is match in magnitude. If the results are match, you can congratulate
for your correct coding. We also recommend you write a automatic test
case for this following the content below.

At the same time, you can see whether the ``loss_function`` is right by
comparing ``spca_fit[["loss"]]`` and the variance of the first principal
component.

-  Check ``sacrifice``

Thirdly, we recommend you consider ``sacrifice``. Checking the function
``sacrifice`` needs more efforts. Monte Carlo studies should be conduct
to check whether ``sacrifice`` is properly programmed such that the
effective/relevant variables can be detected when sample size is large.
We strongly recommend to check the result by setting: - sample size at
least 1000 - dimension is less than 50 - the true support size is less
than 5 - variables are independence - the support size from 0 to the
ground true - the :math:`l_2` regularization is zero.

In most of the cases, this setting is very helpful for checking code.
Generally, the output of ``abess`` would match to the correct under this
setting. Take linear regression in R as our example, the code for
checking is demonstrated below:

.. code:: r

   n <- 1000
   p <- 50
   support_size <- 3
   dataset <- generate.data(n, p, support_size, seed = 1)
   abess_fit <- abess(dataset[["x"]], dataset[["y"]], support.size = 0:support_size)
   ## estimated support:
   extract(abess_fit, support.size = support_size)[["support.vars"]]
   ## true support:
   which(dataset[["beta"]] != 0)

In this example, the estimated support set is the same as the true.

-  Check ``effective_number_of_parameter``

Finally,

Miscellaneous
-------------

Code style
~~~~~~~~~~

New R code should follow the tidyverse `style
guide <https://style.tidyverse.org/>`__. You can use the
```styler`` <https://styler.r-lib.org>`__ R package to apply this style
by conducting R command: ``style_file("path-to-newfile.R")`` New Python
code should follow the PEP8 `style
guide <https://www.python.org/dev/peps/pep-0008/>`__ Please don’t
restyle code that has nothing to do with your code.

Test cases
~~~~~~~~~~

It is always a good habit to do some test for the changed package.
Contributions with test cases included are easier to accept.

We use `testthat <https://cran.r-project.org/web/packages/testthat>`__
for unit tests in R and `pytest <https://docs.pytest.org/>`__ in Python.
You may need to install first.

You can find some examples here and please feel free to add your test
code into it (or create a new test file) under the test folder:

-  `R test
   folder <https://github.com/abess-team/abess/tree/master/R-package/tests/testthat>`__:
   ``abess/R-package/tests/testthat``.
-  `Python test
   folder <https://github.com/abess-team/abess/tree/master/python/pytest>`__:
   ``abess/python/pytest``.

A good test code should contain:

-  possible input modes as well as some wrong input;
-  check whether the output is expected;
-  possible extreme cases;

All test under pytest folder should be checked after coding.
