==================
LogisticRegression
==================

First, we generate some simulated data.
.. code-block:: console
    n <- 100
    p <- 20
    support.size <- 3
    dataset <- generate.data(n, p, support.size, family = "binomial")

Then, we call the `abess` function.
.. code-block:: console
    abess_fit <- abess(dataset[["x"]], dataset[["y"]], 
                   family = "binomial", tune.type = "cv")
    abess_fit


