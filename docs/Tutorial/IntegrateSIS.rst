================
IntegrateSIS
================

.. code-block:: r

    n <- 100
    support.size <- 3
    p <- 1000
    dataset <- generate.data(n, p, support.size)
    abess_fit <- abess(dataset[["x"]], dataset[["y"]], 
                   screening.num = 100)
    str(extract(abess_fit))