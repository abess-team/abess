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
