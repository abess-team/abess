:mod:`abess`
============

.. py:module:: abess


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1
   
   gen_data/index.rst
   linear/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   abess.abessLogistic
   abess.abessLm
   abess.abessCox
   abess.abessPoisson
   abess.abessMLm
   abess.abessMultinomial



Functions
~~~~~~~~~

.. autoapisummary::

   abess.gen_data



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


.. function:: gen_data(n, p, family, k, rho=0, sigma=1, beta=None, censoring=True, c=1, scal=10)


