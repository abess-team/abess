:mod:`abess.linear`
===================

.. py:module:: abess.linear


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   abess.linear.bess_base
   abess.linear.abessLogistic
   abess.linear.abessLm
   abess.linear.abessCox
   abess.linear.abessPoisson
   abess.linear.abessMLm
   abess.linear.abessMultinomial



Functions
~~~~~~~~~

.. autoapisummary::

   abess.linear.fix_docs



.. function:: fix_docs(cls)


.. class:: bess_base(algorithm_type, model_type, path_type, max_iter=20, exchange_num=5, is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None, K_max=None, epsilon=0.0001, lambda_min=0, lambda_max=0, ic_type='ebic', ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1, always_select=[], tau=0.0, primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-08, early_stop=False, approximate_Newton=False, thread=1, covariance_update=False, sparse_matrix=False)


   :param max_iter: Max iteration time in PDAS.
                    Default: max_iter = 20.
   :type max_iter: int, optional
   :param is_warm_start: When search the best sparsity,whether use the last parameter as the initial parameter for the next search.
                         Default:is_warm_start = False.
   :type is_warm_start: bool, optional
   :param path_type: The method we use to search the sparsity。
   :type path_type: {"seq", "pgs"}
   :param sequence: The  sparsity list for searching. If choose path_type = "seq", we prefer you to give the sequence.If not
                    given, we will search all the sparsity([1,2,...,p],p=min(X.shape[0], X.shape[1])).
                    Default: sequence = None.
   :type sequence: array_like, optional
   :param s_min: The lower bound of golden-section-search for sparsity searching.If not given, we will set s_min = 1.
                 Default: s_min = None.
   :type s_min: int, optional
   :param s_max: The higher bound of golden-section-search for sparsity searching.If not given, we will set s_max = p(p = X.shape[1]).
                 Default: s_max = None.
   :type s_max: int, optional
   :param K_max: The search times of golden-section-search for sparsity searching.If not given, we will set K_max = int(log(p, 2/(math.sqrt(5) - 1))).
                 Default: K_max = None.
   :type K_max: int, optional
   :param epsilon: The stop condition of golden-section-search for sparsity searching.
                   Default: epsilon = 0.0001.
   :type epsilon: double, optional
   :param ic_type: The metric when choose the best sparsity.
                   Input must be one of the set above. Default: ic_type = 'ebic'.
   :type ic_type: {'aic', 'bic', 'gic', 'ebic'}, optional
   :param is_cv: Use the Cross-validation method to caculate the loss.
                 Default: is_cv = False.
   :type is_cv: bool, optional
   :param K: The folds number when Use the Cross-validation method to caculate the loss.
             Default: K = 5.
   :type K: int optional
   :param Atrributes:
   :param ----------:
   :param beta: Estimated coefficients for the best subset selection problem.
   :type beta: array of shape (n_features, ) or (n_targets, n_features)

   .. rubric:: References

   - Wen, C. , Zhang, A. , Quan, S. , & Wang, X. . (2017). [Bess: an r package for best subset selection in linear,
       logistic and coxph models]

   .. method:: _arg_check(self)

      Arguments check.


   .. method:: fit(self, X, y, is_weight=False, is_normal=True, weight=None, state=None, group=None)

      The fit function is used to transfer the information of data and return the fit result.

      :param X: Training data
      :type X: array-like of shape (n_samples, n_features)
      :param y: Target values. Will be cast to X's dtype if necessary. For linear regression problem, y should be a n time 1 numpy array with type \code{double}. For classification problem, \code{y} should be a $n       ime 1$ numpy array with values \code{0} or \code{1}. For count data, \code{y} should be a $n    ime 1$ numpy array of non-negative integer.
      :type y: array-like of shape (n_samples,) or (n_samples, n_targets)
      :param is_weight: whether to weight sample yourself.
                        Default: is$\_$weight = False.
      :type is_weight: bool
      :param is_normal: whether normalize the variables array before fitting the algorithm.
                        Default: is$\_$normal=True.
      :type is_normal: bool, optional
      :param weight: Individual weights for each sample. If set is$\_$weight = True, weight should be given.
                     Default: \code{weight} = \code{numpy.ones(n)}.
      :type weight: array-like of shape (n_samples,), default=None
      :param group: The group index for each variable.
                    Default: \code{group} = \code{numpy.ones(p)}.
      :type group: int, optional


   .. method:: predict(self, X)

      The predict function is used to give prediction for new data.

      We will return the prediction of response variable.
      For linear and poisson regression problem, we return a numpy array of the prediction of the mean.
      For classification problem, we return a \code{dict} of \code{pr} and \code{y}, where \code{pr} is the probability of response variable is 1 and \code{y} is predicted to be 1 if \code{pr} > 0.5 else \code{y} is 0.

      :param X: Test data.
      :type X: array-like of shape (n_samples, n_features)



.. class:: abessLogistic(max_iter=20, exchange_num=5, path_type='seq', is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None, K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type='ebic', ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1, always_select=[], tau=0.0, primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-08, early_stop=False, approximate_Newton=False, thread=1, sparse_matrix=False)


   Bases: :py:obj:`bess_base`

   .. rubric:: Examples

   ### Sparsity known
   >>> from bess.linear import *
   >>> import numpy as np
   >>> np.random.seed(12345)
   >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
   >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
   >>> xbeta = np.matmul(x, beta)
   >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
   >>> y = np.random.binomial(1, p)
   >>> model = GroupPdasLogistic(path_type="seq", sequence=[5])
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)

   ### Sparsity unknown
   >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
   >>> model = GroupPdasLogistic(path_type="seq")
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)

   >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
   >>> model = GroupPdasLogistic(path_type="pgs")
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)


.. class:: abessLm(max_iter=20, exchange_num=5, path_type='seq', is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None, K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type='ebic', ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1, always_select=[], tau=0.0, primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-08, early_stop=False, approximate_Newton=False, thread=1, covariance_update=False, sparse_matrix=False)


   Bases: :py:obj:`bess_base`

   .. rubric:: Examples

   ### Sparsity known
   >>> from bess.linear import *
   >>> import numpy as np
   >>> np.random.seed(12345)
   >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
   >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
   >>> xbeta = np.matmul(x, beta)
   >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
   >>> y = np.random.binomial(1, p)
   >>> model = GroupPdasLogistic(path_type="seq", sequence=[5])
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)

   ### Sparsity unknown
   >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
   >>> model = GroupPdasLogistic(path_type="seq")
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)

   >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
   >>> model = GroupPdasLogistic(path_type="pgs")
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)


.. class:: abessCox(max_iter=20, exchange_num=5, path_type='seq', is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None, K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type='ebic', ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1, always_select=[], tau=0.0, primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-08, early_stop=False, approximate_Newton=False, thread=1, sparse_matrix=False)


   Bases: :py:obj:`bess_base`

   .. rubric:: Examples

   ### Sparsity known
   >>> from bess.linear import *
   >>> import numpy as np
   >>> np.random.seed(12345)
   >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
   >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
   >>> xbeta = np.matmul(x, beta)
   >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
   >>> y = np.random.binomial(1, p)
   >>> model = GroupPdasLogistic(path_type="seq", sequence=[5])
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)

   ### Sparsity unknown
   >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
   >>> model = GroupPdasLogistic(path_type="seq")
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)

   >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
   >>> model = GroupPdasLogistic(path_type="pgs")
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)


.. class:: abessPoisson(max_iter=20, exchange_num=5, path_type='seq', is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None, K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type='ebic', ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1, always_select=[], tau=0.0, primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-08, early_stop=False, approximate_Newton=False, thread=1, sparse_matrix=False)


   Bases: :py:obj:`bess_base`

   .. rubric:: Examples

   ### Sparsity known
   >>> from bess.linear import *
   >>> import numpy as np
   >>> np.random.seed(12345)
   >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
   >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
   >>> xbeta = np.matmul(x, beta)
   >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
   >>> y = np.random.binomial(1, p)
   >>> model = GroupPdasLogistic(path_type="seq", sequence=[5])
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)

   ### Sparsity unknown
   >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
   >>> model = GroupPdasLogistic(path_type="seq")
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)

   >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
   >>> model = GroupPdasLogistic(path_type="pgs")
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)


.. class:: abessMLm(max_iter=20, exchange_num=5, path_type='seq', is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None, K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type='ebic', ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1, always_select=[], tau=0.0, primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-08, early_stop=False, approximate_Newton=False, thread=1, covariance_update=False, sparse_matrix=False)


   Bases: :py:obj:`bess_base`

   .. rubric:: Examples

   ### Sparsity known
   >>> from bess.linear import *
   >>> import numpy as np
   >>> np.random.seed(12345)
   >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
   >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
   >>> xbeta = np.matmul(x, beta)
   >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
   >>> y = np.random.binomial(1, p)
   >>> model = GroupPdasLogistic(path_type="seq", sequence=[5])
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)

   ### Sparsity unknown
   >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
   >>> model = GroupPdasLogistic(path_type="seq")
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)

   >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
   >>> model = GroupPdasLogistic(path_type="pgs")
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)


.. class:: abessMultinomial(max_iter=20, exchange_num=5, path_type='seq', is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None, K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type='ebic', ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1, always_select=[], tau=0.0, primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-08, early_stop=False, approximate_Newton=False, thread=1, sparse_matrix=False)


   Bases: :py:obj:`bess_base`

   .. rubric:: Examples

   ### Sparsity known
   >>> from bess.linear import *
   >>> import numpy as np
   >>> np.random.seed(12345)
   >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
   >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
   >>> xbeta = np.matmul(x, beta)
   >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
   >>> y = np.random.binomial(1, p)
   >>> model = GroupPdasLogistic(path_type="seq", sequence=[5])
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)

   ### Sparsity unknown
   >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
   >>> model = GroupPdasLogistic(path_type="seq")
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)

   >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
   >>> model = GroupPdasLogistic(path_type="pgs")
   >>> model.fit(X=x, y=y)
   >>> model.predict(x)


