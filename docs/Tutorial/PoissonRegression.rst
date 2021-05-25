=================
PoissonRegression
=================

First, we generate some simulated data.

.. code-block::R
    dataset <- generate.data(n, p, support.size, family = "poisson")


Then, we call the `abess` function.

.. code-block::R
    abess_fit <- abess(dataset[["x"]], dataset[["y"]], 
                   family = "poisson", tune.type = "cv")
    abess_fit
