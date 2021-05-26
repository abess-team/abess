===================
MultiClassification
===================

.. code-block:: r

    n <- 100
    p <- 20
    support.size <- 3
    dataset <- generate.data(n, p, support.size, family = "multinomial")

.. code-block:: r

    abess_fit <- abess(dataset[["x"]], dataset[["y"]], 
                   family = "multinomial", tune.type = "cv")

.. code-block:: r
    predict(abess_fit, newx = dataset[["x"]][1:10, ], 
        support.size = c(3, 4), type = "response")