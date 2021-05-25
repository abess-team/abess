=================
PoissonRegression
=================

First, we generate some simulated data.
.. code-block:: console
    dataset <- generate.data(n, p, support.size, family = "poisson")

Then, we call the `abess` function.
.. code-block:: console
    abess_fit <- abess(dataset[["x"]], dataset[["y"]], 
                   family = "poisson", tune.type = "cv")
    abess_fit