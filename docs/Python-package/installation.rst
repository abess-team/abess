.. highlight:: shell

============
Installation
============


Stable release
--------------

- Python       

To install abess, run this command in your terminal:

.. code-block:: console

    $ pip install abess

This is the preferred method to install abess, as it will always install the most recent stable release. 

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

- R 
Run the command in R session:
.. code-block:: console

    install.packages("abess")

From sources
------------

The sources for abess can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/abess-team/abess

Or download the `tarball`_.

Once you have a copy of the source, if you have an installation of `pip`_:

.. code-block:: console

    $ pip install <path to the repo/path to the tarball>

Or, after having unpacked the tarball if relevant, you can install it with:

.. code-block:: console

    $ cd <scikit-network folder>
    $ python setup.py develop

OpenMP Support
---------------
To support OpenMP parallelism in Cpp, the dependence for OpenMP should be install. 
In MacOS:       

.. code-block:: console

    $ brew install llvm
    $ brew install libomp





.. _Github repo: https://github.com/abess-team/abess
.. _tarball: https://github.com/sknetwork-team/scikit-network/tarball/master