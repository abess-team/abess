# Installation

## Stable release

### Python
To install abess on Python, you can simply get the stable version with:

```bash
$ pip install abess
```

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.

### R
To install stable version into R environment, run the command:

```R
install.packages("abess")
```

## Latest release

This page gives instructions on how to build and install abess from the source code. 
If the instructions do not help for you, please feel free to ask questions by opening an [issue](https://github.com/abess-team/abess/issues).

### Python 
Clone our [github project](https://github.com/abess-team/abess) to your device:

```bash
$ git clone https://github.com/abess-team/abess.git
$ cd abess
```

Before installing abess itself, some dependencies should be installed first: `swig`, `bash`, `mingw`, which may be a little different in different platforms:

- **Linux**: `$ sudo apt install swig bash mingw-w64` (for Ubuntu, but other Linux systems are similar);
- **Windows**: `$ choco install swig mingw git ` (using [Chocolatey](https://community.chocolatey.org/packages)), or manually install the software and add them into PATH;
- **MacOS**: `$ brew install swig mingw-w64 bash` (using [Homebrew](https://brew.sh/)).

After that, you can manually install abess by conducting command:

```bash
$ cd ./python
$ python setup.py install --user
```

If it finishes with "*Finished processing dependencies for abess*", the installation is successful.

### R
To install the development version from GitHub, run the following code in R console:

```r
library(devtools)
install_github("abess-team/abess", subdir = "R-package")
```

Windows user will need to install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) first.

## Dependencies

### C++

Our core C++ code is based on some dependencies, which can be found in [abess/python/include](https://github.com/abess-team/abess/tree/master/python/include):

- [Eigen](https://gitlab.com/libeigen/eigen/-/releases/3.3.4) (version 3.3.4):
    a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.
- [Spectra](https://github.com/yixuan/spectra/releases/tag/v1.0.0) (version 1.0.0):
    a header-only C++ library for large scale eigenvalue problems.

They would be automatically included while installing the abess packages.

#### OpenMP

To support OpenMP parallelism in Cpp, the dependence for OpenMP should be install. Actually, many compliers and tools have supported and you can check [here](https://www.openmp.org/resources/openmp-compilers-tools/#compilers). 

> What is more, if you receive a warning like "*Unknown option '-fopenmp'*" while installing abess, it means that OpenMP has not been enabled. Without OpenMP, abess only use a single CPU core, leading to suboptimal learning speed.

To enable OpenMP:

- In Windows, [Visual C++](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or many other C++ compliers can support OpenMP API, but you may need to enable it manually in additional features (based on the complier you use).
- In Linux, the dependence would be supported if GCC is installed (version 4.2+).
- In MacOS, the dependence can be installed by:       

    ```bash
    $ brew install llvm
    $ brew install libomp
    ```

### Python

Some [basic Python packages](https://github.com/abess-team/abess/blob/master/python/setup.py#:~:text=install_requires%3D%5B,%5D%2C) are required for abess. Actually, they can be found on 
`abess/python/setup.py` and automatically installed during the installation.

- [numpy](https://pypi.org/project/numpy/): the fundamental package for array computing with Python.
- [scipy](https://pypi.org/project/scipy/): work with NumPy arrays, and provides many user-friendly and efficient numerical routines.
- [scikit-learn](https://pypi.org/project/scikit-learn/): a Python module for machine learning built on top of SciPy. 

Furthermore, if you want to develop the Python packages, some additional packages should be installed:

- [pytest](https://pypi.org/project/pytest/): simple powerful testing with Python.
- [Sphinx](https://pypi.org/project/Sphinx/): develop the Python documentation.
    - [nbsphinx](https://pypi.org/project/nbsphinx/): support jupyter notebook for Sphinx.
    - [myst-parser](https://pypi.org/project/myst-parser/): support markdown for Sphinx.
    - [sphinx-rtd-theme](https://pypi.org/project/sphinx-rtd-theme/): "Read the Docs" theme for Sphinx.

### R

The R version should be 3.1.0 and newer in order to support C++11. 
abess R package relies on limited R packages dependencies:

- [Rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html): convert R Matrix/Vector object into C++.
- [RcppEigen](https://cran.r-project.org/web/packages/RcppEigen/index.html): linear algebra in C++.

Furthermore, if you would to develop the R package, it would be better to additionally install:

- [testthat](https://cran.r-project.org/web/packages/testthat/index.html): conduct unit tests.
- [knitr](https://cran.r-project.org/web/packages/knitr/index.html) and [rmarkdown](https://cran.r-project.org/web/packages/rmarkdown/index.html): write tutorials for R users.

