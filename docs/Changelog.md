# Changelog


## Version 0.3.0    

It is the third stable release for `abess`. This version improve the runtime performance, the clarity of project's documentation, and add helpful continuous integration.

* Cpp
  * New features:
    * Support important searching to significantly improve computational efficiency when dimensionality is large.
  * Performance improvement:
    * Update the version of dependencies: from Spectra 0.9.0 to 1.0.0
    * Bug fixed

* R package
  * Support important searching for generalized linear model `abess`
  * A new release in CRAN.

* Python package
  * Remove useless parameter to improve clarity. 
  * Support important searching for generalized linear model `abessLm`, `abessLogistic`, `abessPoisson`, `abessCox`, `abessMlm`, `abessMultinomial`
  * A new release in Pypi.

* Project development
  * Code coverage for line covering rate for both Python and R. And the coverage rates are summarized and report. 
  * Documentation
    * Add docs2search for the R package's website
    * Add algorithm details and simulation results into Tutorial. 
    * Add a logo for our project
  * Improve code coverage.
  * Continuous integration
    * Automatically generate the .whl files and publish the Python package into Pypi when tagging the project.
    * Check the installation in

## Version 0.2.0

It is the second stable release for `abess`. This version includes multiple several generic features, and optimize memory usage when input data is a sparse matrix. 
We also significantly enhancements to the project' documentation. 

* Cpp
  * New generic best subset features:
    * The selection of group-structured best subset selection;
    * Ridge-regularized penalty for parameter as a generic component. 
  * New best subset selection tasks: 
    * principal component analysis 
  * Performance improvement:
    * Support sparse matrix as input
    * Support golden section search for optimal support size. It is much faster than sequentially searching strategy. 
    * The logic behind cross validation is optimized to gain speed improvement
    * Covariance update
    * Bug fixed

* R package
  * New best subset selection features and tasks implemented in Cpp are wrapped in R functions.
  * `abesspca` supports best subset selection for the first loading vector in principal component analysis. A iterative algorithm supports multiple loading vectors. 
  * Generic S3 function for `abesspca`.
  * Both `abess` and `abesspca` supports sparse matrix input (inherit from class "sparseMatrix" as in package Matrix).
  * Upload to CRAN.

* Python package
  * New best subset selection features and tasks implemented in Cpp are wrapped in Python functions.
  * *abessPCA* supports best subset selection for the first loading vector in principal component analysis. A iterative algorithm supports multiple loading vectors. 
  * Support integration with `scikit-learn`. It is compatible with model evaluation and selection module with `scikit-learn`. 
  * Initial Upload to Pypi.

* Project development
  * Documentation
    * A more clear project website layout.
    * Add an instruction for 
    * Add tutorials to show simple use-cases and non-trival examples of typical use-cases of the software.
    * Link to R-package website.
    * Add an instruction to help package development. 
  * Code coverage for line covering rate for Python.
  * Continuous integration: 
    * Change toolbox from Travis CI to Github-Action. 
    * Auto deploy code coverage result to codecov. 

## Version 0.1.0

We’re happy to announce the first major stable version of `abess`. This version includes multiple new algorithms and features. Here are some highlights of the big updates.

* Cpp
  * New generic best subset features:
    * generic splicing technique
    * nuisance selection
  * New best subset selection tasks: 
    * linear regression
    * logistic regression
    * poisson regression
    * cox proportional hazard regression
    * multi-gaussian regression
    * multi-nomial regression. 
  * Cross validation and information criterion to select the optimal support size
  * Performance improvement:
    * Support OPENMP for the parallelism when performing cross validation
    * Warm start initialization
  * Create a List object to: 1. facilitate transfer the data object from Cpp to Python; 2. use the maximum compatible code for python and R

* R package
  * All best subset selection features and tasks implemented in Cpp are wrapped in a R function `abess`.
  * Unified API for cross validation and information criterion to select the optimal support size.
  * Support generic S3 functions like `coef` and `plot` in R.
  * A short vignettes for demonstrating the usage of package.
  * Support formula interface. 
  * Support convenient function for generating synthetic dataset.
  * Initial upload to CRAN.

* Python 
  * All best subset selection features implemented in Cpp are wrapped in a Python according to tasks. For instance, *abessLm* supports best subset selection for the linear model.
  * Write the Python class on the basis of `scikit-learn` package. The usage of the python package is the same as the common module in `scikit-learn`.
  * Support convenient function for generating synthetic dataset in Python.

* Project developing
  * Build R package website via the `pkgdown` package. 
  * Build a documentation website on based the Python package via the `sphnix` package.
  * The website is continuous integrated via Travis CI. The content will automatically change whether a Travis CI is triggered.
  * Complete testing for R functions in package.