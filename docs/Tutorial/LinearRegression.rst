================
LinearRegression
================

First, we generate some simulated data.

.. code-block:: console
    n <- 100
    p <- 20
    support.size <- 3
    dataset <- generate.data(n, p, support.size)

Then, we call the `abess` function. By default, the support size is tuned from 0 to `min(n, round(n/(log(log(n))log(p))))` sequentially according to general information criterion (GIC).

.. code-block:: console
    abess_fit <- abess(dataset[["x"]], dataset[["y"]])


