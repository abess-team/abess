================
LinearRegression
================

First, we generate some simulated data.

.. code-block:: r

    n <- 100
    p <- 20
    support.size <- 3
    dataset <- generate.data(n, p, support.size)

Then, we call the `abess` function. By default, the support size is tuned from 0 to `min(n, round(n/(log(log(n))log(p))))` sequentially according to general information criterion (GIC).

.. code-block:: r

    abess_fit <- abess(dataset[["x"]], dataset[["y"]])

Followings are some helpful generic functions to draw information from the `abess` estimator.
'' code-block:: r

    print(abess_fit)
    coef(abess_fit, support.size = 3)
    predict(abess_fit, newx = dataset[["x"]][1:10, ], 
        support.size = c(3, 4))
    str(extract(abess_fit, 3))
    deviance(abess_fit)
    plot(abess_fit)
