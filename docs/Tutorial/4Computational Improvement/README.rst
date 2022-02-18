Computational Improvement
--------------------------
 The generic splicing technique certifiably guarantees the best subset can be selected in a polynomial time. In practice, the computational efficiency can be improved to handle large scale datasets. The tips for computational improvement include:
 
 - exploit sparse strucute of input matrix;
 - use golden-section to search best support size;
 - focus on important variables when splicing;
 - early-stop scheme;
 - sure independence screening;
 - warm-start initialization;
 - parallel computing when performing cross validation;
 - covariance update for `LinearRegression` or `MultiTaskRegression`;
 - approximate Newton iteration for `LogisticRegression`, `PoissonRegression`, `CoxRegression`.
 
 This vignette illustrate the first two tips. For the other tips, they have been efficiently implemented and set as the default in abess package.
