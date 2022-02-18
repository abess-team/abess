Advanced Generic Features
--------------------------

When analyzing the real world datasets, we may have the following targets:
 
1. certain variables must be selected when some prior information is given;
2. selecting the weak signal variables when the prediction performance is mainly interested;
3. identifying predictors when group structure are provided;
4. pre-excluding a part of predictors when datasets have ultra high-dimensional predictors;
5. specify the division of sample in cross validation;
6. specify the initial active set before splicing.

In the following content, we will illustrate the statistic methods to reach these targets in a one-by-one manner, and give quick examples to show how to perform the statistic methods in `LinearRegression` and the same steps can be implemented in all methods. Actually, in our methods, the targets can be properly handled by simply change some default arguments in the functions. 
