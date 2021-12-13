ABESS algorithm: details
========================

Introduction 
------------

The ABESS algorithm employing "splicing" technique can exactly solve
general best subset problem in a polynomial time. The aim of this page
to provide a complete and coherent documentation for ABESS algorithm
such that users can easily understand the ABESS algorithm and its
variants, thereby facilitating the usage of ``abess`` software.

linear regression 
-----------------

.. _sacrifices-1:

Sacrifices
~~~~~~~~~~

Consider the :math:`\ell_{0}` constraint minimization problem,

.. math:: \min _{\boldsymbol{\beta}} \mathcal{L}_{n}(\beta), \quad \text { s.t }\|\boldsymbol{\beta}\|_{0} \leq \mathrm{s},

where
:math:`\mathcal{L}_{n}(\boldsymbol \beta)=\frac{1}{2 n}\|y-X \beta\|_{2}^{2} .`
Without loss of generality, we consider
:math:`\|\boldsymbol{\beta}\|_{0}=\mathrm{s}`. Given any initial set
:math:`\mathcal{A} \subset \mathcal{S}=\{1,2, \ldots, p\}` with
cardinality :math:`|\mathcal{A}|=s`, denote
:math:`\mathcal{I}=\mathcal{A}^{\mathrm{c}}` and compute:

.. math:: \hat{\boldsymbol{\beta}}=\arg \min _{\boldsymbol{\beta}_{\mathcal{I}}=0} \mathcal{L}_{n}(\boldsymbol{\beta}).

We call :math:`\mathcal{A}` and :math:`\mathcal{I}` as the active set
and the inactive set, respectively.

Given the active set :math:`\mathcal{A}` and
:math:`\hat{\boldsymbol{\beta}}`, we can define the following two types
of sacrifices:

1. Backward sacrifice: For any :math:`j \in \mathcal{A}`, the magnitude
of discarding variable :math:`j` is,

.. math:: \xi_{j}=\mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}}^{\mathcal{A} \backslash\{j\}}\right)-\mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}}^{\mathcal{A}}\right)=\frac{X_{j}^{\top} X_{j}}{2 n}\left(\hat{\boldsymbol\beta}_{j}\right)^{2},

2. Forward sacrifice: For any :math:`j \in \mathcal{I}`, the magnitude
of adding variable :math:`j` is,

.. math:: \zeta_{j}=\mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}^{\mathcal{A}}}\right)-\mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}}^{\mathcal{A}}+\hat{t}^{\{j\}}\right)=\frac{X_{j}^{\top} X_{j}}{2 n}\left(\frac{\hat{\boldsymbol d}_{j}}{X_{j}^{\top} X_{j} / n}\right)^{2}.

| where
  :math:`\hat{t}=\arg \min _{t} \mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}}^{\mathcal{A}}+t^{\{j\}}\right), \hat{\boldsymbol d}_{j}=X_{j}^{\top}(y-X \hat{\boldsymbol{\beta}}) / n`.
  Intuitively, for :math:`j \in \mathcal{A}` (or
  :math:`j \in \mathcal{I}` ), a large :math:`\xi_{j}` (or
  :math:`\zeta_{j}`) implies the :math:`j` th variable is potentially
  important.

.. _algorithm-1:

Algorithm
~~~~~~~~~

.. _best-subset-selection-with-a-given-support-size-1:

Best-Subset Selection with a Given Support Size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unfortunately, it is noteworthy that these two sacrifices are
incomparable because they have different sizes of support set. However,
if we exchange some "irrelevant" variables in :math:`\mathcal{A}` and
some "important" variables in :math:`\mathcal{I}`, it may result in a
higher-quality solution. This intuition motivates our splicing method.
Specifically, given any splicing size :math:`k \leq s`, define

.. math:: \mathcal{A}_{k}=\left\{j \in \mathcal{A}: \sum_{i \in \mathcal{A}} \mathrm{I}\left(\xi_{j} \geq \xi_{i}\right) \leq k\right\},

to represent :math:`k` least relevant variables in :math:`\mathcal{A}`
and,

.. math:: \mathcal{I}_{k}=\left\{j \in \mathcal{I}: \sum_{i \in \mathcal{I}} \mid\left(\zeta_{j} \leq \zeta_{i}\right) \leq k\right\},

to represent :math:`k` most relevant variables in :math:`\mathcal{I} .`

| Then, we splice :math:`\mathcal{A}` and :math:`\mathcal{I}` by
  exchanging :math:`\mathcal{A}_{k}` and :math:`\mathcal{I}_{k}` and
  obtain a new active
  set::math:`\tilde{\mathcal{A}}=\left(\mathcal{A} \backslash \mathcal{A}_{k}\right) \cup \mathcal{I}_{k}.`
  Let
  :math:`\tilde{\mathcal{I}}=\tilde{\mathcal{A}}^{c}, \tilde{\boldsymbol{\beta}}=\arg \min _{\boldsymbol{\beta}_{\overline{\mathcal{I}}=0}} \mathcal{L}_{n}(\boldsymbol{\beta})`,
  and :math:`\tau_{s}>0` be a threshold. If :math:`\tau_{s}<\mathcal{L}_{n}(\hat{\boldsymbol\beta})-\mathcal{L}_{n}(\tilde{\boldsymbol\beta})`,
  then :math:`\tilde{A}` is preferable to :math:`\mathcal{A} .` 
| The
  active set can be updated
  iteratively until the loss function cannot be improved by splicing.
  Once the algorithm recovers the true active set, we may splice some
  irrelevant variables, and then the loss function may decrease
  slightly. The threshold :math:`\tau_{s}` can reduce this unnecessary
  calculation. Typically, :math:`\tau_{s}` is relatively small, e.g.
  :math:`\tau_{s}=0.01 s \log (p) \log (\log n) / n.`

.. _algorithm-1-bessfixs-best-subset-selection-with-a-given-support-size-:

Algorithm 1: BESS.Fix(s): Best-Subset Selection with a given support size :math:`s`.
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

1. Input: :math:`X, y`, a positive integer :math:`k_{\max }`, and a
   threshold :math:`\tau_{s}`.

2. Initialize: 
   
   .. math::

      \begin{align*}
         \mathcal{A}^{0}=\left\{j: \sum_{i=1}^{p} \mathrm{I}\left(\left|\frac{X_{j}^{\top} y}{\sqrt{X_{j}^{\top} X_{j}}}\right| \leq \left| \frac{X_{i}^{\top} y}{\sqrt{X_{i}^{\top} X_{i}}}\right| \leq \mathrm{s}\right\}, \mathcal{I}^{0}=\left(\mathcal{A}^{0}\right)^{c}\right.
      \end{align*}
   
   and :math:`\left(\boldsymbol\beta^{0}, d^{0}\right):`

   .. math::

      \begin{align*}
         &\boldsymbol{\beta}_{\mathcal{I}^{0}}^{0}=0,\\
         &d_{\mathcal{A}^{0}}^{0}=0,\\
      &\boldsymbol{\beta}_{\mathcal{A}^{0}}^{0}=\left(\boldsymbol{X}_{\mathcal{A}^{0}}^{\top} \boldsymbol{X}_{\mathcal{A}^{0}}\right)^{-1} \boldsymbol{X}_{\mathcal{A}^{0}}^{\top} \boldsymbol{y},\\
      &d_{\mathcal{I}^{0}}^{0}=X_{\mathcal{I}^{0}}^{\top}\left(\boldsymbol{y}-\boldsymbol{X} \boldsymbol{\beta}^{0}\right).
      \end{align*}

3. For :math:`m=0,1, \ldots`, do

      .. math:: \left(\boldsymbol{\beta}^{m+1}, \boldsymbol{d}^{m+1}, \mathcal{A}^{m+1}, \mathcal{I}^{m+1}\right)= \text{Splicing} \left(\boldsymbol{\beta}^{m}, \boldsymbol{d}^{m}, \mathcal{A}^{m}, \mathcal{I}^{m}, k_{\max }, \tau_{s}\right).

      If :math:`\left(\mathcal{A}^{m+1}, \mathcal{I}^{m+1}\right)=\left(\mathcal{A}^{m},\mathcal{I}^{m}\right)`,
      then stop.

   End For

4. Output
   :math:`(\hat{\boldsymbol{\beta}}, \hat{\boldsymbol{d}}, \hat{\mathcal{A}}, \hat{\mathcal{I}})=\left(\boldsymbol{\beta}^{m+1}, \boldsymbol{d}^{m+1} \mathcal{A}^{m+1}, \mathcal{I}^{m+1}\right).`

.. _algorithm-2-splicing-1:

Algorithm 2: Splicing :math:`\left(\boldsymbol\beta, d, \mathcal{A}, \mathcal{I}, k_{\max }, \tau_{s}\right)`
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

1. Input:
   :math:`\boldsymbol{\beta}, \boldsymbol{d}, \mathcal{A}, \mathcal{I}, k_{\max }`,
   and :math:`\tau_{\mathrm{s}} .`

2. Initialize: 
   :math:`L_{0}=L=\frac{1}{2 n}\|y-X \beta\|_{2}^{2}`, and set

   .. math:: \xi_{j}=\frac{X_{j}^{\top} X_{j}}{2 n}\left(\beta_{j}\right)^{2}, \zeta_{j}=\frac{X_{j}^{\top} X_{j}}{2 n}\left(\frac{d_{j}}{X_{j}^{\top} X_{j} / n}\right)^{2}, j=1, \ldots, p.

3. For :math:`k=1,2, \ldots, k_{\max }`, do

      .. math::

         \mathcal{A}_{k}=\left\{j \in \mathcal{A}: \sum_{i \in \mathcal{A}} \mathrm{I}\left(\xi_{j} \geq \xi_{i}\right) \leq k\right\},\\
         \mathcal{I}_{k}=\left\{j \in \mathcal{I}: \sum_{i \in \mathcal{I}} \mathrm{I}\left(\zeta_{j} \leq \zeta_{i}\right) \leq k\right\}.

      Let
      :math:`\tilde{\mathcal{A}}_{k}=\left(\mathcal{A} \backslash \mathcal{A}_{k}\right) \cup \mathcal{I}_{k}, \tilde{\mathcal{I}}_{k}=\left(\mathcal{I} \backslash \mathcal{I}_{k}\right) \cup \mathcal{A}_{k}`
      and solve:

      .. math::

         \tilde{\boldsymbol{\beta}}_{{\mathcal{A}}_{k}}=\left(\boldsymbol{X}_{\mathcal{A}_{k}}^{\top} \boldsymbol{X}_{{\mathcal{A}}_{k}}\right)^{-1} \boldsymbol{X}_{{\mathcal{A}_{k}}}^{\top} y, \quad \tilde{\boldsymbol{\beta}}_{{\mathcal{I}}_{k}}=0\\
         \tilde{\boldsymbol d}_{\mathcal{I}^k}=X_{\mathcal{I}^k}^{\top}(y-X \tilde{\beta}) / n,\quad \tilde{\boldsymbol d}_{\mathcal{A}^k} = 0.

      Compute:
      :math:`\mathcal{L}_{n}(\tilde{\boldsymbol\beta})=\frac{1}{2 n}\|y-X \tilde{\boldsymbol\beta}\|_{2}^{2}.`
      If :math:`L>\mathcal{L}_{n}(\tilde{\boldsymbol\beta})`, then

      .. math::

         (\hat{\boldsymbol{\beta}}, \hat{\boldsymbol{d}}, \hat{\mathcal{A}}, \hat{\mathcal{I}})=\left(\tilde{\boldsymbol{\beta}}, \tilde{\boldsymbol{d}}, \tilde{\mathcal{A}}_{k}, \tilde{\mathcal{I}}_{k}\right)\\
         L=\mathcal{L}_{n}(\tilde{\boldsymbol\beta}).

   End for

3. If :math:`L_{0}-L<\tau_{s}`, then
   :math:`(\hat{\boldsymbol\beta}, \hat{d}, \hat{A}, \hat{I})=(\boldsymbol\beta, d, \mathcal{A}, \mathcal{I}).`

2. Output
   :math:`(\hat{\boldsymbol{\beta}}, \hat{\boldsymbol{d}}, \hat{\mathcal{A}}, \hat{\mathcal{I}})`.

Determining the Best Support Size with SIC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In practice, the support size is usually unknown. We use a datadriven
procedure to determine s. For any active set :math:`\mathcal{A}`, define
an :math:`\mathrm{SIC}` as follows:

.. math:: \operatorname{SIC}(\mathcal{A})=n \log \mathcal{L}_{\mathcal{A}}+|\mathcal{A}| \log (p) \log \log n,

where
:math:`\mathcal{L}_{\mathcal{A}}=\min _{\beta_{\mathcal{I}}=0} \mathcal{L}_{n}(\beta), \mathcal{I}=(\mathcal{A})^{c}`.
To identify the true model, the model complexity penalty is
:math:`\log p` and the slow diverging rate :math:`\log \log n` is set to
prevent underfitting. Theorem 4 states that the following ABESS
algorithm selects the true support size via SIC.

Let :math:`s_{\max }` be the maximum support size. We suggest
:math:`s_{\max }=o\left(\frac{n}{\log p}\right)` as the maximum possible
recovery size. Typically, we set
:math:`s_{\max }=\left[\frac{n}{\log p \log \log n}\right]` where
:math:`[x]` denotes the integer part of :math:`x`.

.. _algorithm-3-abess:

Algorithm 3: ABESS.
'''''''''''''''''''

1. Input: :math:`X, y`, and the maximum support size :math:`s_{\max } .`

2. For :math:`s=1,2, \ldots, s_{\max }`, do

   .. math:: \left(\hat{\boldsymbol{\beta}}_{s}, \hat{\boldsymbol{d}}_{s}, \hat{\mathcal{A}}_{s}, \hat{\mathcal{I}}_{s}\right)= \text{BESS.Fixed}(s).

   End for

3. Compute the minimum of SIC:

   .. math:: s_{\min }=\arg \min _{s} \operatorname{SIC}\left(\hat{\mathcal{A}}_{s}\right).

4. Output
   :math:`\left(\hat{\boldsymbol{\beta}}_{s_{\min}}, \hat{\boldsymbol{d}}_{s_{\min }}, \hat{A}_{s_{\min }}, \hat{\mathcal{I}}_{s_{\min }}\right) .`

Group linear model
------------------

.. _sacrifices-2:

Sacrifices
~~~~~~~~~~

Consider the :math:`\ell_{0,2}` constraint minimization problem with
:math:`n` samples and :math:`J` non-overlapping groups,

.. math:: \min _{\boldsymbol{{\boldsymbol\beta}}} \mathcal{L}({\boldsymbol\beta}), \quad \text { s.t }\|{{\boldsymbol\beta}}\|_{0,2} \leq \mathrm{T}.

where :math:`\mathcal{L}({\boldsymbol\beta})` is the negative
log-likelihood function and support size :math:`\mathrm{T}` is a
positive number. Without loss of generality, we consider
:math:`\|\boldsymbol{{\boldsymbol\beta}}\|_{0,2}=\mathrm{T}`. Given any
group subset :math:`\mathcal{A} \subset \mathcal{S}=\{1,2, \ldots, J\}`
with cardinality :math:`|\mathcal{A}|=\mathrm{T}`, denote
:math:`\mathcal{I}=\mathcal{A}^{\mathrm{c}}` and compute:

.. math:: \hat{{{\boldsymbol\beta}}}=\arg \min _{{{\boldsymbol\beta}}_{\mathcal{I}}=0} \mathcal{L}({{\boldsymbol\beta}}).

| We call :math:`\mathcal{A}` and :math:`\mathcal{I}` as the selected
  group subset and the unselected group subset, respectively.
| Denote
  :math:`g_{G_j} = [{\nabla} \mathcal{L}({\boldsymbol\beta})]_{G_j} ` as
  the :math:`j`\ th group gradient of :math:`({\boldsymbol\beta})` and
  :math:`h_{G_j} = [{\nabla}^2 \mathcal{L}({\boldsymbol\beta})]_{G_j} `
  as the :math:`j`\ th group diagonal sub-matrix of hessian matrix of
  :math:`\mathcal{L}({\boldsymbol\beta})`. Let dual variable
  :math:`d_{G_j} = -g_{G_j}` and
  :math:`\Psi_{G_j} =  (h_{G_j})^{\frac{1}{2}}`.

Given the selected group subset :math:`\mathcal{A}` and
:math:`\hat{\boldsymbol{{\boldsymbol\beta}}}`, we can define the
following two types of sacrifices:

1. Backward sacrifice: For any :math:`j \in \mathcal{A}`, the magnitude
   of discarding group :math:`j` is,

   .. math:: \xi_j = \mathcal{L}({\boldsymbol\beta}^{\mathcal{A}^k\backslash j})-\mathcal{L}({\boldsymbol\beta}^k)=\frac{1}{2}({\boldsymbol\beta}^k_{G_j})^k h^k_{G_j}{\boldsymbol\beta}^k_{G_j} = \frac{1}{2}\|\bar{{\boldsymbol\beta}}_{G_j}^k\|_2^2,

   where :math:`{\boldsymbol\beta}^{\mathcal{A}^k\backslash j}` is the
   estimator assigning the :math:`j`\ th group of
   :math:`{\boldsymbol\beta}^k` to be zero and
   :math:`\bar {\boldsymbol\beta}_{G_j}^k=\Psi^k_{G_j} {\boldsymbol\beta}_{G_j}^k`.

2. Forward sacrifice: For any :math:`j \in \mathcal{I}`, the magnitude
   of adding variable :math:`j` is,

   .. math:: \zeta_{j}=\mathcal{L}({\boldsymbol\beta}^k)-\mathcal{L}({\boldsymbol\beta}^k+t_j^k)=\frac{1}{2}(d_{G_j}^k)^\top (h^k_{G_j})^{-1} d^k_{G_j}= \frac{1}{2}\|\bar{d}^k_{G_j}\|_2^2,

   where
   :math:`t^k_j = \arg\min\limits_{t_{G_j} \neq 0}L({\boldsymbol\beta}^k+t)`
   and :math:`\bar d_{G_j}^k = (\Psi^k_{G_j})^{-1} d^k_{G_j}`.

Intuitively, for :math:`j \in \mathcal{A}` (or :math:`j \in \mathcal{I}`
), a large :math:`\xi_{j}` (or :math:`\zeta_{j}`) implies the :math:`j`
th group is potentially important.

We show four useful examples in the following.

.. _case-1--group-linear-model:

Case 1 : Group linear model.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In group linear model, the loss function is

.. math::

   \begin{equation*}
   \mathcal{L}({\boldsymbol\beta}) = \frac{1}{2}\|y-X{\boldsymbol\beta}\|_2^2.
   \end{equation*}

We have

.. math::

   \begin{equation*}
   d_{G_j} = X_{G_j}^\top(y-X{\boldsymbol\beta})/n,\ \Psi_{G_j} = (X_{G_j}^\top X_{G_j}/n)^{\frac{1}{2}}, \ j=1,\ldots,J.
   \end{equation*}

Under the assumption of orthonormalization, that is
:math:`X_{G_j}^\top X_{G_j}/n = I_{p_j}, j=1,\ldots, J`. we have
:math:`\Psi_{G_j}=I_{p_j}`. Thus for linear regression model, we do not
need to update :math:`\Psi` during iteration procedures.

.. _case-2--group-logistic-model:

Case 2 : Group logistic model.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the data :math:`\{(X_i, y_i)\}_{i=1}^{n}` with
:math:`y_i \in \{0, 1\}, X_i \in \mathbb{R}^p`, and denote
:math:`X_i = (X_{i, G_1}^\top,\ldots, X_{i, G_J}^\top)^\top`.

Consider the logistic model
:math:`\log\{\pi/(1-\pi)\} = {\boldsymbol\beta}_0 +  x^\top{\boldsymbol\beta}`
with :math:`x \in \mathbb{R}^p` and :math:`\pi = P(y=1|x)`.

Thus the negative log-likelihood function is:

.. math::

   \begin{equation*}
   \mathcal{L}({\boldsymbol\beta}_0, {\boldsymbol\beta}) =  \sum_{i=1}^n  \{\log(1+\exp({\boldsymbol\beta}_0+X_i^\top {\boldsymbol\beta}))-y_i ({\boldsymbol\beta}_0+X_i^\top {\boldsymbol\beta})\}.
   \end{equation*}

We have

.. math::

   \begin{equation*}
   d_{G_j} = X_{G_j}^\top(y-\pi),\ \Psi_{G_j} = (X_{G_j}^\top W X_{G_j})^{\frac{1}{2}}, \ j=1,\ldots,J,
   \end{equation*}

where :math:`\pi = (\pi_1,\ldots,\pi_n)` with
:math:`\pi_i = \exp(X_i^\top {\boldsymbol\beta})/(1+\exp(X_i^\top {\boldsymbol\beta}))`,
and :math:`W` is a diagonal matrix with :math:`i`\ th diagonal entry
equal to :math:`\pi_i(1-\pi_i)`.

.. _case-3--group-poisson-model:

Case 3 : Group poisson model.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the data :math:`\{(X_i, y_i)\}_{i=1}^{n}` with
:math:`y_i \in \mathbb{N}, X_i \in \mathbb{R}^p`, and denote
:math:`X_i = (X_{i, G_1}^\top,\ldots, X_{i, G_J}^\top)^\top`.

Consider the poisson model
:math:`\log(\mathbb{E}(y|x)) = {\boldsymbol\beta}_0 + x^\top {\boldsymbol\beta}`
with :math:`x \in \mathbb{R}^p`.

Thus the negative log-likelihood function is:

.. math::

   \begin{equation*}
     \mathcal{L}({\boldsymbol\beta}_0, {\boldsymbol\beta}) =  \sum_{i=1}^n  \{\exp({\boldsymbol\beta}_0+X_i^\top {\boldsymbol\beta})+\log(y_i !)-y_i ({\boldsymbol\beta}_0+X_i^\top {\boldsymbol\beta})\}.
   \end{equation*}

We have:

.. math::

   \begin{equation*}
   d_{G_j} = X_{G_j}^\top(y-\eta),\ \Psi_{G_j} = (X_{G_j}^\top W X_{G_j})^{\frac{1}{2}}, \ j=1,\ldots,J,
   \end{equation*}

where :math:`\eta = (\eta_1,\ldots,\eta_n)` with
:math:`\eta_i = \exp({\boldsymbol\beta}_0+X_i^\top{\boldsymbol\beta})`,
and :math:`W` is a diagonal matrix with :math:`i`\ th diagonal entry
equal to :math:`\eta_i`.

.. _case-4--group-cox-proportional-hazard-model:

Case 4 : Group Cox proportional hazard model.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the survival data :math:`\{(T_i, \delta_i, x_i)\}_{i=1}^n` with
observation of survival time :math:`T_i` an censoring indicator
:math:`\delta_i`.

Consider the Cox proportional hazard model
:math:`\lambda(x|t) = \lambda_0(t) \exp(x^\top {\boldsymbol\beta})`
with a baseline hazard :math:`\lambda_0(t)` and
:math:`x \in \mathbb{R}^p`. By the method of partial likelihood,
we can write the negative log-likelihood function as:

.. math::

   \begin{equation*}
     \mathcal{L}({\boldsymbol\beta}) =  \log\{\sum_{i':T_{i'} \geqslant T_i} \exp(X_i^\top{\boldsymbol\beta})\}-\sum_{i:\delta_i = 1} X_i^\top {\boldsymbol\beta}.
   \end{equation*}

We have:

.. math::

   \begin{align*}
     &d_{G_j} = \sum_{i:\delta_i=1} (X_{i, G_j} - \sum_{i':T_{i'} > T_i} X_{i', G_j} \omega_{i, i'}),\\
     &\Psi_{G_j}=\{\sum_{i:\delta_i=1} (\{\sum_{i':T_{i'} > T_i} \omega_{i, i'} X_{i',G_j}\}\{\sum_{i':T_{i'} > T_i} \omega_{i, i'} X_{i',G_j}\}^\top-\sum_{i':T_{i'} > T_i} \omega_{i, i'} X_{i',G_j} X_{i', G_j}^\top)\}^{\frac{1}{2}},
   \end{align*}

where
:math:`\omega_{i, i'} = \exp(X_{i'}^\top{\boldsymbol\beta})/\sum_{i':T_{i'} > T_i} \exp(X_{i'}^\top {\boldsymbol\beta})`.

.. _algorithm-2:

Algorithm
~~~~~~~~~

Best Group Subset Selection with a determined support size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Motivated by the definition of sacrifices, we can extract the
"irrelevant" groups in :math:`\mathcal{A}` and the "important" groups in
:math:`\mathcal{I}`, respectively, and then exchange them to get a
high-quality solution.

Given any exchange subset size :math:`C \leq C_{max}`, define the
exchanged group subset as:

.. math:: \mathcal{S}_{C,1}=\left\{j \in \mathcal{A}: \sum_{i \in \mathcal{A}} \mathrm{I}\left(\frac{1}{p_j}\xi_{j} \geq \frac{1}{p_i}\xi_{i}\right) \leq C\right\},

and

.. math:: \mathcal{S}_{C,2}=\left\{j \in \mathcal{I}: \sum_{i \in \mathcal{I}} I\left(\frac{1}{p_j}\zeta_{j} \leq \frac{1}{p_i}\zeta_{i}\right) \leq C\right\},

where :math:`p_j` is the number of variables in :math:`j`\ th group.

From the definition of sacrifices,
:math:`\mathcal{S}_{C,1}\ (\mathcal{S}_{C,2})` can be interpreted as the
groups in :math:`\mathcal{A}\ (\mathcal{I})` with :math:`C` smallest
(largest) contributions to the loss function. Then, we splice
:math:`\mathcal{A}` and :math:`\mathcal{I}` by exchanging
:math:`\mathcal{S}_{C,1}` and :math:`\mathcal{S}_{C,2}` and obtain a
novel selected group subset

.. math:: \tilde{\mathcal{A}}=\left(\mathcal{A} \backslash \mathcal{S}_{C,1}\right) \cup \mathcal{S}_{C,2}.

Let
:math:`\tilde{\mathcal{I}}=\tilde{\mathcal{A}}^{c}, \tilde{\boldsymbol{{\boldsymbol\beta}}}=\arg \min _{\boldsymbol{{\boldsymbol\beta}}_{\overline{\mathcal{I}}}=0} \mathcal{L}(\boldsymbol{{\boldsymbol\beta}})`,
and :math:`\pi_{T}>0` be a threshold to eliminate unnecessary
iterations.

We summarize the group-splicing algorithm as follows:

.. _algorithm-1-group-splicing:

Algorithm 1: Group-Splicing.
''''''''''''''''''''''''''''

1. Input:
   :math:`X,\ y,\ \{G_j\}_{j=1}^J,\ T, \ \mathcal{A}^0,\ \pi_T, \ C_{\max}`.

2. Initialize :math:`k=0` and solve primal variable :math:`{\boldsymbol\beta}^{k}` and dual variable :math:`d^{k}:`

   .. math::

      \begin{align*}
         &{{\boldsymbol\beta}}_{\mathcal{A}^{k}}^{k}=[\arg \min _{{{\boldsymbol\beta}}_{\mathcal{I}^{k}}=0} \mathcal{L}({{\boldsymbol\beta}})]_{\mathcal{A}^{k}},\ {{\boldsymbol\beta}}_{\mathcal{I}^{k}}^{k}=0,\\
         &d_{\mathcal{I}^{k}}^{k}=[\nabla \mathcal{L}({\boldsymbol\beta}^k)]_{\mathcal{I}^k},\ d_{\mathcal{A}^{k}}^{k}=0.\\
         \end{align*}

3. While :math:`\mathcal{A}^{k+1} \neq \mathcal{A}^{k}`, do

      Compute :math:`L=\mathcal{L}({\boldsymbol\beta}^k)` and :math:`( {\bar{\boldsymbol\beta}}, {\bar{d}} )`.
      
      Update :math:`\mathcal{S}_1^k, \mathcal{S}_2^k`

      .. math::

         \begin{align*}
         &\mathcal{S}_1^k = \{j \in \mathcal{A}^k: \sum\limits_{i\in \mathcal{A}^k} I(\frac{1}{p_j}\|{\bar {\boldsymbol\beta}_{G_j}^k}\|_2^2 \geq \frac{1}{p_i}\|{\bar {\boldsymbol\beta}_{G_i}^k}\|_2^2) \leq C_{\max}\},\\
         &\mathcal{S}_2^k = \{j \in \mathcal{I}^k: \sum\limits_{i\in \mathcal{I}^k} I(\frac{1}{p_j}\|{\bar d_{G_j}^k}\|_2^2 \leq \frac{1}{p_i}\|{\bar d_{G_i}^k}\|_2^2) \leq C_{\max}\}.
         \end{align*}

4. For :math:`C=C_{\max}, \ldots, 1`, do

      Let
      :math:`\tilde{\mathcal{A}}^k_C=(\mathcal{A}^k\backslash \mathcal{S}_1^k)\cup \mathcal{S}_2^k\ \text{and}\ \tilde{\mathcal{I}}^k_C = (\mathcal{I}^k\backslash \mathcal{S}_2^k)\cup \mathcal{S}_1^k`.

      Update primal variable :math:`\tilde{{\boldsymbol\beta}}` and dual
      variable :math:`\tilde{d}`

      .. math::

         \begin{align*}
         \tilde{\boldsymbol\beta}=\arg \min _{{{\boldsymbol\beta}}_{\tilde{\mathcal{I}}^k_C}=0} \mathcal{L}({{\boldsymbol\beta}}),\ \tilde d = \nabla \mathcal{L}(\tilde{\boldsymbol\beta}).
         \end{align*}

      Compute :math:`\tilde L = \mathcal{L}(\tilde {\boldsymbol\beta})`.

      If :math:`L-\tilde L < \pi_T`, denote
      :math:`(\tilde{\mathcal{A}}^k_C, \tilde{\mathcal{I}}^k_C, \tilde {\boldsymbol\beta} , \tilde d )`
      as
      :math:`(\mathcal{A}^{k+1}, \mathcal{I}^{k+1}, {\boldsymbol\beta}^{k+1}, d^{k+1})`
      and break.

      Else, Update :math:`\mathcal{S}_1^k \text{ and } \mathcal{S}_2^k`:

      .. math::

         \begin{align*}
         &\mathcal{S}_1^k = \mathcal{S}_1^k\backslash \arg\max\limits_{i \in \mathcal{S}_1^k} \{\frac{1}{p_i}\|{\bar {\boldsymbol\beta}_{G_i}^k}\|_2^2\},\\
         &\mathcal{S}_2^k = \mathcal{S}_2^k\backslash \arg\min\limits_{i \in \mathcal{S}_2^k} \{\frac{1}{p_i}\|{\bar d_{G_i}^k}\|_2^2\}.
         \end{align*}

   End For

      If
      :math:`\left(\mathcal{A}^{k+1}, \mathcal{I}^{k+1}\right)=\left(\mathcal{A}^{k}, \mathcal{I}^{k}\right)`,
      then stop.

   End While

5. Output
   :math:`(\hat{\boldsymbol{{\boldsymbol\beta}}}, \hat{\boldsymbol{d}}, \hat{\mathcal{A}}, \hat{\mathcal{I}})=\left(\boldsymbol{{\boldsymbol\beta}}^{m+1}, \boldsymbol{d}^{m+1} \mathcal{A}^{m+1}, \mathcal{I}^{m+1}\right).`

Determining the best support size with information criterion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Practically, the optimal support size is usually unknown. Thus, we use
  a data-driven procedure to determine :math:`\mathrm{T}`. Due to the
  computational burden of cross validation, we prefer information
  criterion to conduct the selection procedure.
| For any selected group subset :math:`\mathcal{A}`, define an group
  information criterion(GIC) as follows:

.. math:: \operatorname{GIC}(\mathcal{A})=n \log \mathcal{L}_{\mathcal{A}}+ \log J \log \log n \#\{\mathcal{A}\},

| where
  :math:`\mathcal{L}_{\mathcal{A}}=\min _{{\boldsymbol\beta}_{\mathcal{I}}=0} \mathcal{L}_{n}({\boldsymbol\beta}), \mathcal{I}=(\mathcal{A})^{c}` and
  :math:`\#\{\mathcal{A}\}` is the number of variables contained in :math:`\cup_{j\in \mathcal{A}}G_j`.
  To identify the true model, the
  model complexity penalty is :math:`\log J` and the slow diverging rate
  :math:`\log \log n` is set to prevent underfitting. Besides, we define
  the Bayesian group information criterion (BGIC) as follows:

.. math:: \operatorname{BGIC}(\mathcal{A})=n \log \mathcal{L}_{\mathcal{A}}+ (\gamma \log J +\log n)\#\{\mathcal{A}\},

where :math:`\gamma` is a pre-determined positive constant, controlling
the diverging rate of group numbers :math:`J`.

| A natural idea to determine the optimal support size is regarding
  :math:`\mathrm{T}` as a tuning parameter, and running GSplicing
  algorithm over a sequence about :math:`\mathrm{T}`. Next, combined
  with aforementioned information criterion, we can obtain an optimal
  support size.
| Let :math:`T_{\max }` be the maximum support size. We suggest
  :math:`T_{\max }=o\left(\frac{n}{p_{\max}\log J}\right)` where
  :math:`p_{\max} = \max_{j\in \mathcal{S}} p_j`.

We summarize the sequential group-splicing algorithm with GIC as
follows:

.. _algorithm-2-sequential-group-splicing-sgsplicing:

Algorithm 2: Sequential Group-Splicing (SGSplicing).
''''''''''''''''''''''''''''''''''''''''''''''''''''

1. Input:
   :math:`X,\ y,\ \{G_j\}_{j=1}^J,\ T_{\max}, \ \pi_T, \ C_{\max}.`

2. For :math:`T=1,2, \ldots, T_{\max }`, do

   .. math:: \left(\hat{\boldsymbol{{\boldsymbol\beta}}}_{T}, \hat{\boldsymbol{d}}_{T}, \hat{\mathcal{A}}_{T}, \hat{\mathcal{I}}_{T}\right)=\text{GSplicing}(X, y, \{G_j\}_{j=1}^J, T,  \mathcal{A}^0_T, \pi_T, C_{\max}).

   End for

3. Compute the minimum of GIC:

   .. math:: T_{\min }=\arg \min _{T} \operatorname{GIC}\left(\hat{\mathcal{A}}_{T}\right).

4. Output
   :math:`\left(\hat{\boldsymbol{{\boldsymbol\beta}}}_{T_{\operatorname{min}}}, \hat{\boldsymbol{d}}_{T_{\min }}, \hat{\mathcal{A}}_{T_{\min }}, \hat{\mathcal{I}}_{T_{\min }}\right) .`

Nuisance selection 
------------------

Principal Component Analysis
----------------------------

.. _sacrifices-3:

Sacrifices 
~~~~~~~~~~

Consider the :math:`\ell_{0}` constraint minimization problem,

.. math::

   \min_v\ -v^T\Sigma v,\\
   s.t.\quad v^Tv = 1,\ ||v||_0 = s,

where :math:`\Sigma` is the given covariance matrix and :math:`s` is the
chosen sparsity level.

Denote the active set and inactive set as:

.. math::

   \mathcal{A} = \{i|v_i\neq 0\},\quad
   \mathcal{I} = \{i|v_i = 0\},

and :math:`\alpha = -2\Sigma v + 2\beta v`. Since there are only
:math:`s` elements in :math:`\mathcal{A}`, the definition can actually
be proved as:

.. math::

   \mathcal{A} = \{i|\sum_j 
   	I(|v_i - \frac{\alpha_i}{\rho}|\leq|v_j - \frac{\alpha_j}{\rho}|)\leq s\},\\
   \mathcal{I} = \{i|\sum_j
   	I(|v_i - \frac{\alpha_i}{\rho}|\leq|v_j - \frac{\alpha_j}{\rho}|)> s\},\\

where :math:`\rho` is a constant and it decides the distribution in
:math:`\mathcal{A}, \mathcal{I}`. Now the choice of active and inactive
set is based on :math:`\frac{\alpha_i}{\rho}`. When we change
:math:`\rho`, we are actually exchanging the elements between
:math:`\mathcal{A}` and :math:`\mathcal{I}`. This exchanging is regular:
smaller :math:`|v_i-\frac{\alpha_i}{\rho}|` is tend to be inactive and
larger is tend to be active.

Note that we can define forward and backward sacrifice here,

1. Forward sacrifice: for each :math:`i\in \mathcal{I}`, the larger
   :math:`|v_i - \frac{\alpha_i}{\rho}|`, the more possible to be
   exchanged to :math:`\mathcal{A}`. Since :math:`v_i = 0`, we can focus
   on :math:`|\alpha_i|`,

   .. math:: \zeta_{i} = |\alpha_i|.

2. Backward sacrifice: for each :math:`i\in \mathcal{A}`, the smaller
   :math:`|v_i - \frac{\alpha_i}{\rho}|`, the more possible to be
   exchanged to :math:`\mathcal{I}`. Since
   :math:`v_i = H_{\frac{2\mu}{\rho}}(v_i-\frac{\alpha_i}{\rho})` and so
   that :math:`\alpha_i=0`, we can only focus on :math:`|v_i|`,

   .. math:: \xi_i = |v_i|.

.. _algorithm-3:

Algorithm
~~~~~~~~~

.. _best-subset-selection-with-a-given-support-size-2:

Best-Subset Selection with a Given Support Size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As we discuss above, we can iteratively solve :math:`v`, and in each
iteration, we compute:

.. math:: \alpha = -2\Sigma v + 2\beta v,

and the active/inactive set. Then the sacrifices are:

.. math::

   \begin{cases}
       \zeta_{i} = |\alpha_i|, & i\in \mathcal{I}\\
       \xi_i = |v_i|, & i\in \mathcal{A}
   \end{cases},

We try all number of the exchanging from 0 to :math:`\min(s, p-s)` and
choose the best one with higher :math:`v^T\Sigma v`. If no element need
to be exchanged, the program will return :math:`v` as the result.

Algorithm 1: SPCA
'''''''''''''''''

1. Input :math:`s, \Sigma` (or :math:`X`). If :math:`X` is given, set
   :math:`\Sigma = cov(X)`;

2. Initialize :math:`v` with :math:`s` non-zero positions;

3. For :math:`m = 0, 1, \cdots` do:

      Compute :math:`\mathcal{A}`, :math:`\mathcal{I}` and :math:`\alpha`;

      Set :math:`v = \text{Splicing}(s,\Sigma, \mathcal{A}, \mathcal{I}, \alpha)`;

      If :math:`v` is not changed, break.
   
   End For

4. Return :math:`v`.

.. _algorithm-2-splicing-2:

Algorithm 2: Splicing
'''''''''''''''''''''

1. Input :math:`s,\Sigma, \mathcal{A}, \mathcal{I}, \alpha`;

2. Compute forward sacrifices:
   :math:`\zeta_{i} = |\alpha_i|, i\in \mathcal{I}` and backward
   sacrifices: :math:`\xi_i = |v_i|, i\in \mathcal{A}`;

3. For :math:`k = 0, 1, \cdots, \min(s, p-s)` do:

      Exchange :math:`k` elements in :math:`\mathcal{I}` with :math:`k`
      largest :math:`\zeta` and in :math:`\mathcal{A}` with :math:`k`
      smallest :math:`\xi`;

      Form a normal PCA on active set to get :math:`v`;

      Re-compute :math:`v^T\Sigma v`;

      Record the :math:`v_0 = \arg\max_v v^T\Sigma v`;
   
   End For

4. Return :math:`v_0`.

Multiple SPCA
^^^^^^^^^^^^^

Sometimes we require more than one principle components. Actually, we
can iteratively solve the largest principal component and then mapping
the covariance matrix to its orthogonal space:

.. math:: \Sigma' = (1-vv^T)\Sigma(1-vv^T),

where :math:`\Sigma` is the currect covariance matrix and :math:`v` is
its (sparse) principal component solved above. We map it into
:math:`Σ^′`, which indicates the orthogonal space of :math:`v`, and then
solve again.

Algorithm 3: Multi-SPCA 
'''''''''''''''''''''''

1. Input :math:`s, \Sigma` (or :math:`X`), and :math:`number`. If
   :math:`X` is given, set :math:`\Sigma = cov(X)`;

2. For :math:`num = 1, 2, \cdots, number`:

      Compute :math:`v = \text{SPCA}(s,\Sigma);`

      Set :math:`\Sigma = (1-vv^T)\Sigma(1-vv^T);`

      Record :math:`v;`

   End For

3. Print all :math:`v`'s.

Group Principal Component Analysis
----------------------------------

.. _sacrifices-4:

Sacrifices
~~~~~~~~~~

With group information, consider the :math:`\ell_{0}` constraint
minimization problem,

.. math::

   \min_v\ -v^T\Sigma v,\\
   s.t.\quad v^Tv = 1,\ ||v||_{0,g} = s,

where :math:`\Sigma ` is the given covariance matrix and :math:`s` is
the chosen sparsity level. :math:`||v||_{0,g}` indicates the number of
non-zero groups in :math:`v`, i.e.

.. math:: ||v||_{0,g} = \sum_g I(||v_{(g)}||\neq 0),

where :math:`v_{(g)}` is the :math:`g`-th group of predictors and
:math:`v^T = (v_{(1)}^T, v_{(2)}^T, \cdots, v_{(G)}^T)`.

Similar to the `Principal Component
Analysis <#principal-component-analysis>`__, the problem can be
rewritten as:

.. math::

   \mathcal{A} = \{i|\sum_j 
   	I(||v_i - \frac{\alpha_i}{\rho}||_2\leq||v_j - \frac{\alpha_j}{\rho}||_2)\leq s\},\\
   \mathcal{I} = \{i|\sum_j
   	I(||v_i - \frac{\alpha_i}{\rho}||_2\leq||v_j - \frac{\alpha_j}{\rho}||_2)> s\},\\

We can define forward and backward sacrifice by

1. Forward sacrifice: for each :math:`i\in \mathcal{I}`, the larger
   :math:`||v_{(i)} - \frac{\alpha_{(i)}}{\rho}||_2`, the more possible
   to be exchanged to :math:`\mathcal{A}`. Since :math:`v_i = 0`, we can
   focus on :math:`||\alpha_{(i)}||_2`,

   .. math:: \zeta_{i} = ||\alpha_{(i)}||_2.

2. Backward sacrifice: for each :math:`i\in \mathcal{A}`, the smaller
   :math:`||v_{(i)} - \frac{\alpha_{(i)}}{\rho}||_2`, the more possible
   to be exchanged to :math:`\mathcal{I}`. Since
   :math:`v_i = H_{\frac{2\mu}{\rho}}(v_{(i)}-\frac{\alpha_{(i)}}{\rho})`
   and so that :math:`\alpha_i=0`, we can focus on
   :math:`||v_{(i)}||_2`,

   .. math:: \xi_i = ||v_{(i)}||_2.

Note that if each group contains only one predictor, the sacrifices
become the non-group ones.

.. _algorithm-4:

Algorithm
~~~~~~~~~

Actually, the workflow is almost the same as non-group situation. We
just change the sacrifices in **Algorithm 2** to:

Algorithm 4: Group-splicing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Input :math:`s,\Sigma, \mathcal{A}, \mathcal{I}, \alpha`;

2. Compute forward sacrifices:
   :math:`\zeta_{i} = ||\alpha_{(i)}||_2, i\in \mathcal{I}` and backward
   sacrifices: :math:`\xi_i = ||v_{(i)}||_2, i\in \mathcal{A}`;

3. For :math:`k = 0, 1, \cdots, \min(s, p-s)` do:

      Exchange :math:`k` elements in :math:`\mathcal{I}` with :math:`k`
      largest :math:`\zeta` and in :math:`\mathcal{A}` with :math:`k`
      smallest :math:`\xi`;

      Form a normal PCA on active set to get :math:`v`;

      Re-compute :math:`v^T\Sigma v`;

      Record the :math:`v_0 = \arg\max_v v^T\Sigma v`;

4. return :math:`v_0`.

Important Search
----------------

Suppose that there are only a few variables are important (i.e. too many noise variables), 
it may be a vise choice to focus on some important variables during splicing process. 
This can save a lot of time, especially under a large $p$.

Algorithm
~~~~~~~~~

Suppose we are focus on the sparsity level :math:`s` and we have the sacrifice :math:`\zeta, \xi`
from the last sparsity level's searching. Now we focus on an variables' subset :math:`U` with size `U\_size`, 
which is not larger than :math:`p`:

Algorithm : Important Search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Input :math:`s, X, y, group\_index, group\_size, \zeta, \xi, U\_size, max\_iter`;

2. Sort all sacrifices and choose the largest :math:`U\_size` variables as :math:`U`, initially;

3. For :math:`iter = 0, 1, \cdots, max\_iter` do:

      Mapping :math:`X, y, group\_index, group\_size` to `U`;

      Form splicing on this subset, until the active set is stable;

      Inverse mapping to full set;

      Re-compute the sacrifices with the new active set;

      Sort and update :math:`U` (similar to Step 2);

      If :math:`U` is unchanged (not in order), break;

4. Return :math:`\mathcal{A},  \mathcal{I}`.
