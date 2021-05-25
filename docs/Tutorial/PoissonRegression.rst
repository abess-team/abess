=================
PoissonRegression
=================

First, we generate some simulated data.

.. code-block:: r

    dataset <- generate.data(n, p, support.size, family = "poisson")


Then, we call the `abess` function.

.. code-block:: r

    abess_fit <- abess(dataset[["x"]], dataset[["y"]], 
                   family = "poisson", tune.type = "cv")
    abess_fit

.. code-block:: console

    $ pip install <path to the repo/path to the tarball>

