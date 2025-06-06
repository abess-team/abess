# abess 0.4.10
* Fix note in NOTE about possible bashisms.

# abess 0.4.9
* Fix bug in Cpp level
* Fix error in: https://www.stats.ox.ac.uk/pub/bdr/clang19/abess.log
* Fix notes in https://cran.r-project.org/web/checks/check_results_abess.html

# abess 0.4.8
* Support no-intercept GLM model by param 'fit.intercept'.
* Allow to restrict the range of estimation for beta by param 'beta.high' and 'beta.low'.
* Add cite message when load 'abess'.
* Fix a bug when support.size is 0.

# abess 0.4.7

* Allow the other criterion for model selection: AUC for (multinomial) logistic regression such as the area under the curve (AUC). 
* Simplify the C++ code structure. 
* Fix note "Specified C++11: please update to current default of C++17" in CRAN.


# abess 0.4.6

* Adapt to the API change of the `Matrix` package.
* Change the package structure such that the API functions can reuse the 
utility function. It facilitates the testing for package.
* Update citation information.

# abess 0.4.5

* Support generalized linear model for ordinal response, also named as rank learning in machine learning community. 
* Support robust principal analysis
* Modify R package structure to make many internal components are reusable.
* Update README.md

# abess 0.4.0

* Support generalized linear model when the link function is Gamma distribution. 
By setting `family = "gamma"` in `abess` function, users can analyze the dataset with a positive valued and skewed response. 
* Support flexible support size for sequential principal component analysis (PCA), accompanied with several helpful generic function like `plot`. 
* Support user-specified cross validation division for `abess` and `abesspca` function by additional argument `foldid`. 
* Support robust principal component analysis now. A new R function `abessrpca` can access it.
* Improve the R package document by: adding more details and giving more links related to core functions.  

# abess 0.3.0

* Add docs2search for R's website
* Support important searching to improve computational efficiency when dimension is 10,000.

# abess 0.2.0

* Support sparse matrix as input
* Support golden section search for optimal support size
* Support ridge-regularized penalty as a generic component
* Support group subset selection as a generic component
* Best subset selection for principal component analysis via *abesspca*
* Bug fixed

# abess 0.1.0

* Initial stable version abess package
* Support best subset selection for linear regression, logistic regression, poisson regression, cox proportional hazard regression, multi-gaussian regression, 
multi-nominal regression. 
* Support nuisance selection as a generic component
* Unified API for cross validation and information criterion to select the optimal support size.
* A documentation website is support for *abess* package
