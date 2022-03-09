Installation
============

Stable release
--------------

Python
~~~~~~

To install ``abess`` on Python, you can simply get the stable version with:

.. code:: bash

   $ pip install abess

If you don’t have `pip <https://pip.pypa.io>`__ installed, this `Python
installation
guide <http://docs.python-guide.org/en/latest/starting/installation/>`__
can guide you through the process.

R
~

To install stable version into R environment, run the command:

.. code:: r

   install.packages("abess")

Latest version
--------------

This page gives instructions on how to build and install ``abess`` from the
source code. If the instructions do not help for you, please feel free
to ask questions by opening an
`issue <https://github.com/abess-team/abess/issues>`__.

First of all, clone our the latest `github
project <https://github.com/abess-team/abess>`__ to your device:

.. code:: bash

   $ git clone https://github.com/abess-team/abess.git
   $ cd abess

Next, there have different processing depend on the programming
langulage you prefer.

.. _python-1:

Python
~~~~~~

Before installing abess itself, some dependent libraries should be installed
first, which may be a little different in
different platforms:

-  **Linux**: ``$ sudo apt install bash`` (for Ubuntu,
   but other Linux systems are similar);
-  **Windows**: ``$ choco install git`` (using
   `Chocolatey <https://community.chocolatey.org/packages>`__), or
   manually install the software and add them into PATH;
   Windows user will need to install Microsoft Visual Studio.
-  **MacOS**: ``$ brew install bash`` (using
   `Homebrew <https://brew.sh/>`__).

Then, you need to install ``pybind11`` via ``pip install pybind11`` (or `other methods <https://pybind11.readthedocs.io/en/stable/installing.html#>`__). 
After that, we can manually install ``abess`` by conducting command:

.. code:: bash

   $ cd ./python
   $ python setup.py install --user

If it finishes with ``Finished processing dependencies for abess``, the
installation is successful.

.. _r-1:

R
~

To install the development version, run the following code in R console:

.. code:: bash

   R CMD INSTALL R-package

If it finishes with ``* DONE (abess)``, the installation is successful.

Windows user will need to install
`Rtools <https://cran.r-project.org/bin/windows/Rtools/>`__ first.

Dependencies
------------

C++
~~~

Our core C++ code is based on some dependencies, which can be found in
`abess/python/include <https://github.com/abess-team/abess/tree/master/python/include>`__:

-  `Eigen <https://gitlab.com/libeigen/eigen/-/releases/3.3.4>`__
   (version 3.3.4): a C++ template library for linear algebra: matrices,
   vectors, numerical solvers, and related algorithms.
-  `Spectra <https://github.com/yixuan/spectra/releases/tag/v1.0.0>`__
   (version 1.0.0): a header-only C++ library for large scale eigenvalue
   problems.

They would be automatically included while installing the abess
packages.

OpenMP
^^^^^^

To support OpenMP parallelism in Cpp, the dependence for OpenMP should
be install. Actually, many compliers and tools have supported and you
can check
`here <https://www.openmp.org/resources/openmp-compilers-tools/#compilers>`__.

   What is more, if you receive a warning like “*Unknown option
   ‘-fopenmp’*” while installing abess, it means that OpenMP has not
   been enabled. Without OpenMP, abess only use a single CPU core,
   leading to suboptimal learning speed.

To enable OpenMP:

-  In Windows, `Visual
   C++ <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`__
   or many other C++ compliers can support OpenMP API, but you may need
   to enable it manually in additional features (based on the complier
   you use).

-  In Linux, the dependence would be supported if GCC is installed
   (version 4.2+).

-  In MacOS, the dependence can be installed by:

   .. code:: bash

      $ brew install llvm
      $ brew install libomp

.. _python-2:

Python
~~~~~~

Some `basic Python
packages <https://github.com/abess-team/abess/blob/master/python/setup.py#:~:text=install_requires%3D%5B,%5D%2C>`__
are required for abess. Actually, they can be found on
``abess/python/setup.py`` and automatically installed during the
installation.

-  `pybind11 <https://pybind11.readthedocs.io/en/stable/>`__: seamless operability between C++11 and Python
-  `numpy <https://pypi.org/project/numpy/>`__: the fundamental package
   for array computing with Python.
-  `scipy <https://pypi.org/project/scipy/>`__: work with NumPy arrays,
   and provides many user-friendly and efficient numerical routines.
-  `scikit-learn <https://pypi.org/project/scikit-learn/>`__: a Python
   module for machine learning built on top of SciPy.

Furthermore, if you want to develop the Python packages, some additional
packages should be installed:

-  `pytest <https://pypi.org/project/pytest/>`__: simple powerful
   testing with Python.
   
   - `lifelines <https://pypi.org/project/lifelines/>`__: support testing 
     for survival analysis.

-  `Sphinx <https://pypi.org/project/Sphinx/>`__: develop the Python
   documentation.

   -  `sphinx-rtd-theme <https://pypi.org/project/sphinx-rtd-theme/>`__:
      “Read the Docs” theme for Sphinx.
   -  `sphinix-gallery <https://pypi.org/project/sphinx-gallery/>`__: develop the gallery of Python examples.

-  `pandas <https://pypi.org/project/pandas/>`__: 
   support data reading and preprocessing in examples.
      

.. _r-2:

R
~

The R version should be 3.1.0 and newer in order to support C++11. abess
R package relies on limited R packages dependencies:

-  `Rcpp <https://cran.r-project.org/web/packages/Rcpp/index.html>`__:
   convert R Matrix/Vector object into C++.
-  `RcppEigen <https://cran.r-project.org/web/packages/RcppEigen/index.html>`__:
   linear algebra in C++.

Furthermore, if you would to develop the R package, it would be better
to additionally install:

-  `testthat <https://cran.r-project.org/web/packages/testthat/index.html>`__:
   conduct unit tests.
-  `roxygen2 <https://cran.r-project.org/web/packages/roxygen2/index.html>`__:
   write R documentations.
-  `knitr <https://cran.r-project.org/web/packages/knitr/index.html>`__
   and
   `rmarkdown <https://cran.r-project.org/web/packages/rmarkdown/index.html>`__:
   write tutorials.
-  `pkgdown <https://cran.r-project.org/web/packages/pkgdown/index.html>`__:
   build website for the ``abess`` R package.
