================
CoxRegression
================
First, we generate some simulated data.
.. code-block:: r

    n <- 100
    p <- 20
    support.size <- 3
    dataset <- generate.data(n, p, support.size, family = "cox")

Then, we call the `abess` function.
.. code-block:: r
    abess_fit <- abess(dataset[["x"]], dataset[["y"]], 
                   family = "cox", tune.type = "cv")