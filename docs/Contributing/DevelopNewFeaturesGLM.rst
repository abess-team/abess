Develop New Features: GLM
=========================

Specifically, it could be pretty easy that you don't even need to understand the whole `ABESS algorithm <https://www.pnas.org/doi/10.1073/pnas.2014241117#sec-21>`__
before developing a best-subset problem on Generalized Linear Models (GLM).
In this tutorial, we will show you how to implement it.

For more general models, please check `Develop New Features <DevelopNewFeatures.html>`__ instead.

Preliminaries
-------------

We have endeavor to make developing new features easily. Before developing the code, please make sure you have:    

- installed ``abess`` via the code in github by following `Installation <../Installation.html>`__ instruction;
- some experience on writing R or Python code.

Generalized linear model
------------------------

In mathematics, we often denote the variables as :math:`\mathbf{X}` and
the outcome as :math:`\mathbf{y}`, which is assumed to be 
generated from a distribution in an exponential family, e.g.

- Normal Distribution;
- Binomial Distribution;
- Possion Distribution;
- ...

And the generalized linear model would be like:

.. math::
   \mathbb{E}(\mathbf{y}|\mathbf{X}) = \mathbf{\mu} = g^{-1}(\mathbf{X\beta}),

where :math:`\mathbb{E}(\mathbf{y}|\mathbf{X})` is the expected value of :math:`\mathbf{y}`
conditional on :math:`\mathbf{X}`; :math:`g` is the link function; 
:math:`\beta` is the model parameters.
Let's take the logistic regression as an example, where:

.. math::
   \mathbb{E}(y) = \mathbb{P}(y=1) = p,\quad
   g^{-1}(z) = \frac{1}{1+\exp(-z)}.

Core C++ for GLM
----------------

The first step is to write an API, which is the same as `Develop New Features: Write an API <DevelopNewFeatures.html#write-an-api>`__.

Then, to implemented algorithms on GLM, you need to focus on
``src/AlgorithmGLM.h``, where we have implemented a base model
called ``_abessGLM`` and the new algorithm should inherit it.

.. code:: cpp

   template <class T4>
   class abess_new_GLM_algorithm : public _abessGLM<{T1}, {T2}, {T3}, T4>  // T1, T2, T3 are the same as above, which are fixed.
   {
   public:
       // constructor and destructor
       abess_new_GLM_algorithm(...) : _abessGLM<...>::_abessGLM(...){};
       ~abess_new_GLM_algorithm(){};

       Eigen::MatrixXd gradian_core(...) {
           // the gradian matrix can be expressed as G = X^T * A,   
           // returns the gradian core A
       };
       Eigen::VectorXd hessian_core(...) {
           // the hessian matrix can be expressed as H = X^T * D * X,
           // returns the (diagnal values of) diagnal matrix D.
       };
       {T1} inv_link_function(...) {
           // returns inverse link function g^{-1}(X, beta),
           // i.e. the predicted y
       };
       {T1} log_probability(...) {
           // returns log P(y | X, beta)
       };
       bool null_model(...) {
           // returns a null model,
           // i.e. given only y, fit an intercept
       };
   }

Compared with the `general implement <DevelopNewFeaturesGLM.html#implement-your-algorithm>`__,
it do not require touching the core of abess, but only use the knowledge of model itself.

Let's still discuss the logistic model and consider we hope to maximize log-likelihood:
`[code link] <https://github.com/oooo26/abess/blob/master/src/AlgorithmGLM.h#:~:text=class%20T4%3E-,class%20abessLogistic,-%3A%20public%20_abessGLM>`__

.. math::
   l &= \frac{1}{2}\log\prod_i \mathbb{P}(y_i|X_i, \beta)\\
   &= \frac{1}{2}\sum_i \left[y_i\log(\hat{y_i}) + (1-y_i)\log(1-\hat{y_i})\right]\\
   &= -\frac{1}{2}\sum_i \left[y_i\log(1-e^{-X_i^T\beta}) + (1-y_i)\log(1-e^{X_i^T\beta})\right],

From this formula, we can get the ``inv_link_function`` and ``log_probability``.
And we continue on its derivatives on :math:`\beta`:

.. math::
   \frac{\partial l}{\partial \beta} 
   &= \sum_i X_i y_i(1-y_i),\\
   \frac{\partial^2 l}{\partial \beta^2}
   &= \sum_i X_iX_i^T y_i(1-y_i)

From this formula, we can get the ``gradian_core`` and ``hessian_core``. Finally,
the ``null_model`` should be:

.. math::
   \mathbb{E}(y) = g^{-1}(C),\quad
   i.e.\quad 
   C = g(\overline{y}),

where :math:`\overline{y}` is the mean of :math:`y`.

Now your new method has been connected to the whole frame. 
You can continue on the following steps like `Develop New Features: R & Python Package <DevelopNewFeatures.html#r-python-package>`__.
