"""
========================
ABESS Algorithm: Details
========================
"""

###########################################################
# Introduction
# ^^^^^^^^^^^^
# With the ``abess`` library, users can use the ABESS algorithm to efficiently solve many best subset selection problems.
# The aim of this page is providing a complete and coherent documentation for ABESS algorithm under linear model
# such that users can easily understand the ABESS algorithm,
# thereby facilitating the usage of ``abess`` software.
#
# linear regression
# ^^^^^^^^^^^^^^^^^
#
# .. _sacrifices-1:
#
# Sacrifices
# ~~~~~~~~~~
#
# Consider the :math:`\ell_{0}` constraint minimization problem,
#
# .. math:: \min _{\boldsymbol{\beta}} \mathcal{L}_{n}(\beta), \quad \text { s.t }\|\boldsymbol{\beta}\|_{0} \leq \mathrm{s},
#
# where
# :math:`\mathcal{L}_{n}(\boldsymbol \beta)=\frac{1}{2 n}\|y-X \beta\|_{2}^{2} .`
# Without loss of generality, we consider
# :math:`\|\boldsymbol{\beta}\|_{0}=\mathrm{s}`. Given any initial set
# :math:`\mathcal{A} \subset \mathcal{S}=\{1,2, \ldots, p\}` with
# cardinality :math:`|\mathcal{A}|=s`, denote
# :math:`\mathcal{I}=\mathcal{A}^{\mathrm{c}}` and compute:
#
# .. math:: \hat{\boldsymbol{\beta}}=\arg \min _{\boldsymbol{\beta}_{\mathcal{I}}=0} \mathcal{L}_{n}(\boldsymbol{\beta}).
#
# We call :math:`\mathcal{A}` and :math:`\mathcal{I}` as the active set
# and the inactive set, respectively.
#
# Given the active set :math:`\mathcal{A}` and
# :math:`\hat{\boldsymbol{\beta}}`, we can define the following two types
# of sacrifices:
#
# 1. Backward sacrifice: For any :math:`j \in \mathcal{A}`, the magnitude
# of discarding variable :math:`j` is,
#
# .. math:: \xi_{j}=\mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}}^{\mathcal{A} \backslash\{j\}}\right)-\mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}}^{\mathcal{A}}\right)=\frac{X_{j}^{\top} X_{j}}{2 n}\left(\hat{\boldsymbol\beta}_{j}\right)^{2},
#
# 2. Forward sacrifice: For any :math:`j \in \mathcal{I}`, the magnitude
# of adding variable :math:`j` is,
#
# .. math:: \zeta_{j}=\mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}^{\mathcal{A}}}\right)-\mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}}^{\mathcal{A}}+\hat{t}^{\{j\}}\right)=\frac{X_{j}^{\top} X_{j}}{2 n}\left(\frac{\hat{\boldsymbol d}_{j}}{X_{j}^{\top} X_{j} / n}\right)^{2}.
#
# | where
#   :math:`\hat{t}=\arg \min _{t} \mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}}^{\mathcal{A}}+t^{\{j\}}\right), \hat{\boldsymbol d}_{j}=X_{j}^{\top}(y-X \hat{\boldsymbol{\beta}}) / n`.
#   Intuitively, for :math:`j \in \mathcal{A}` (or
#   :math:`j \in \mathcal{I}` ), a large :math:`\xi_{j}` (or
#   :math:`\zeta_{j}`) implies the :math:`j` th variable is potentially
#   important.
#
# .. _algorithm-1:
#
# Algorithm
# ~~~~~~~~~
#
# .. _best-subset-selection-with-a-given-support-size-1:
#
# Best-Subset Selection with a Given Support Size
# """""""""""""""""""""""""""""""""""""""""""""""
#
# Unfortunately, it is noteworthy that these two sacrifices are
# incomparable because they have different sizes of support set. However,
# if we exchange some "irrelevant" variables in :math:`\mathcal{A}` and
# some "important" variables in :math:`\mathcal{I}`, it may result in a
# higher-quality solution. This intuition motivates our splicing method.
# Specifically, given any splicing size :math:`k \leq s`, define
#
# .. math:: \mathcal{A}_{k}=\left\{j \in \mathcal{A}: \sum_{i \in \mathcal{A}} \mathrm{I}\left(\xi_{j} \geq \xi_{i}\right) \leq k\right\},
#
# to represent :math:`k` least relevant variables in :math:`\mathcal{A}`
# and,
#
# .. math:: \mathcal{I}_{k}=\left\{j \in \mathcal{I}: \sum_{i \in \mathcal{I}} \mid\left(\zeta_{j} \leq \zeta_{i}\right) \leq k\right\},
#
# to represent :math:`k` most relevant variables in :math:`\mathcal{I} .`
#
# | Then, we splice :math:`\mathcal{A}` and :math:`\mathcal{I}` by
#   exchanging :math:`\mathcal{A}_{k}` and :math:`\mathcal{I}_{k}` and
#   obtain a new active
#   set::math:`\tilde{\mathcal{A}}=\left(\mathcal{A} \backslash \mathcal{A}_{k}\right) \cup \mathcal{I}_{k}.`
#   Let
#   :math:`\tilde{\mathcal{I}}=\tilde{\mathcal{A}}^{c}, \tilde{\boldsymbol{\beta}}=\arg \min _{\boldsymbol{\beta}_{\overline{\mathcal{I}}=0}} \mathcal{L}_{n}(\boldsymbol{\beta})`,
#   and :math:`\tau_{s}>0` be a threshold. If :math:`\tau_{s}<\mathcal{L}_{n}(\hat{\boldsymbol\beta})-\mathcal{L}_{n}(\tilde{\boldsymbol\beta})`,
#   then :math:`\tilde{A}` is preferable to :math:`\mathcal{A} .`
# | The
#   active set can be updated
#   iteratively until the loss function cannot be improved by splicing.
#   Once the algorithm recovers the true active set, we may splice some
#   irrelevant variables, and then the loss function may decrease
#   slightly. The threshold :math:`\tau_{s}` can reduce this unnecessary
#   calculation. Typically, :math:`\tau_{s}` is relatively small, e.g.
#   :math:`\tau_{s}=0.01 s \log (p) \log (\log n) / n.`
#
# .. _algorithm-1-bessfixs-best-subset-selection-with-a-given-support-size-:
#
# Algorithm 1: BESS.Fix(s): Best-Subset Selection with a given support size :math:`s`.
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#
# 1. Input: :math:`X, y`, a positive integer :math:`k_{\max }`, and a
#    threshold :math:`\tau_{s}`.
#
# 2. Initialize:
#
# .. math::
#          \mathcal{A}^{0}=\left\{j: \sum_{i=1}^{p} \mathrm{I}\left(\left|\frac{X_{j}^{\top} y}{\sqrt{X_{j}^{\top} X_{j}}}\right| \leq \left| \frac{X_{i}^{\top} y}{\sqrt{X_{i}^{\top} X_{i}}}\right| \leq \mathrm{s}\right\}, \mathcal{I}^{0}=\left(\mathcal{A}^{0}\right)^{c}\right.
#
# and :math:`\left(\boldsymbol\beta^{0}, d^{0}\right):`
#
# .. math::
#          &\boldsymbol{\beta}_{\mathcal{I}^{0}}^{0}=0,\\
#          &d_{\mathcal{A}^{0}}^{0}=0,\\
#       &\boldsymbol{\beta}_{\mathcal{A}^{0}}^{0}=\left(\boldsymbol{X}_{\mathcal{A}^{0}}^{\top} \boldsymbol{X}_{\mathcal{A}^{0}}\right)^{-1} \boldsymbol{X}_{\mathcal{A}^{0}}^{\top} \boldsymbol{y},\\
#       &d_{\mathcal{I}^{0}}^{0}=X_{\mathcal{I}^{0}}^{\top}\left(\boldsymbol{y}-\boldsymbol{X} \boldsymbol{\beta}^{0}\right).
#
# 3. For :math:`m=0,1, \ldots`, do
#
#       .. math:: \left(\boldsymbol{\beta}^{m+1}, \boldsymbol{d}^{m+1}, \mathcal{A}^{m+1}, \mathcal{I}^{m+1}\right)= \text{Splicing} \left(\boldsymbol{\beta}^{m}, \boldsymbol{d}^{m}, \mathcal{A}^{m}, \mathcal{I}^{m}, k_{\max }, \tau_{s}\right).
#
#       If :math:`\left(\mathcal{A}^{m+1}, \mathcal{I}^{m+1}\right)=\left(\mathcal{A}^{m},\mathcal{I}^{m}\right)`,
#       then stop.
#
#    End For
#
# 4. Output
#    :math:`(\hat{\boldsymbol{\beta}}, \hat{\boldsymbol{d}}, \hat{\mathcal{A}}, \hat{\mathcal{I}})=\left(\boldsymbol{\beta}^{m+1}, \boldsymbol{d}^{m+1} \mathcal{A}^{m+1}, \mathcal{I}^{m+1}\right).`
#
# .. _algorithm-2-splicing-1:
#
# Algorithm 2: Splicing :math:`\left(\boldsymbol\beta, d, \mathcal{A}, \mathcal{I}, k_{\max }, \tau_{s}\right)`
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#
# 1. Input:
#    :math:`\boldsymbol{\beta}, \boldsymbol{d}, \mathcal{A}, \mathcal{I}, k_{\max }`,
#    and :math:`\tau_{\mathrm{s}} .`
#
# 2. Initialize:
#    :math:`L_{0}=L=\frac{1}{2 n}\|y-X \beta\|_{2}^{2}`, and set
#
#    .. math:: \xi_{j}=\frac{X_{j}^{\top} X_{j}}{2 n}\left(\beta_{j}\right)^{2}, \zeta_{j}=\frac{X_{j}^{\top} X_{j}}{2 n}\left(\frac{d_{j}}{X_{j}^{\top} X_{j} / n}\right)^{2}, j=1, \ldots, p.
#
# 3. For :math:`k=1,2, \ldots, k_{\max }`, do
#
#       .. math::
#
#          \mathcal{A}_{k}=\left\{j \in \mathcal{A}: \sum_{i \in \mathcal{A}} \mathrm{I}\left(\xi_{j} \geq \xi_{i}\right) \leq k\right\},\\
#          \mathcal{I}_{k}=\left\{j \in \mathcal{I}: \sum_{i \in \mathcal{I}} \mathrm{I}\left(\zeta_{j} \leq \zeta_{i}\right) \leq k\right\}.
#
#       Let
#       :math:`\tilde{\mathcal{A}}_{k}=\left(\mathcal{A} \backslash \mathcal{A}_{k}\right) \cup \mathcal{I}_{k}, \tilde{\mathcal{I}}_{k}=\left(\mathcal{I} \backslash \mathcal{I}_{k}\right) \cup \mathcal{A}_{k}`
#       and solve:
#
#       .. math::
#
#          \tilde{\boldsymbol{\beta}}_{{\mathcal{A}}_{k}}=\left(\boldsymbol{X}_{\mathcal{A}_{k}}^{\top} \boldsymbol{X}_{{\mathcal{A}}_{k}}\right)^{-1} \boldsymbol{X}_{{\mathcal{A}_{k}}}^{\top} y, \quad \tilde{\boldsymbol{\beta}}_{{\mathcal{I}}_{k}}=0\\
#          \tilde{\boldsymbol d}_{\mathcal{I}^k}=X_{\mathcal{I}^k}^{\top}(y-X \tilde{\beta}) / n,\quad \tilde{\boldsymbol d}_{\mathcal{A}^k} = 0.
#
#       Compute:
#       :math:`\mathcal{L}_{n}(\tilde{\boldsymbol\beta})=\frac{1}{2 n}\|y-X \tilde{\boldsymbol\beta}\|_{2}^{2}.`
#       If :math:`L>\mathcal{L}_{n}(\tilde{\boldsymbol\beta})`, then
#
#       .. math::
#
#          (\hat{\boldsymbol{\beta}}, \hat{\boldsymbol{d}}, \hat{\mathcal{A}}, \hat{\mathcal{I}})=\left(\tilde{\boldsymbol{\beta}}, \tilde{\boldsymbol{d}}, \tilde{\mathcal{A}}_{k}, \tilde{\mathcal{I}}_{k}\right)\\
#          L=\mathcal{L}_{n}(\tilde{\boldsymbol\beta}).
#
#    End for
#
# 4. If :math:`L_{0}-L<\tau_{s}`, then
#    :math:`(\hat{\boldsymbol\beta}, \hat{d}, \hat{A}, \hat{I})=(\boldsymbol\beta, d, \mathcal{A}, \mathcal{I}).`
#
# 5. Output
#    :math:`(\hat{\boldsymbol{\beta}}, \hat{\boldsymbol{d}}, \hat{\mathcal{A}}, \hat{\mathcal{I}})`.
#
# Determining the Best Support Size with SIC
# """"""""""""""""""""""""""""""""""""""""""
#
# In practice, the support size is usually unknown. We use a datadriven
# procedure to determine s. For any active set :math:`\mathcal{A}`, define
# an :math:`\mathrm{SIC}` as follows:
#
# .. math:: \operatorname{SIC}(\mathcal{A})=n \log \mathcal{L}_{\mathcal{A}}+|\mathcal{A}| \log (p) \log \log n,
#
# where
# :math:`\mathcal{L}_{\mathcal{A}}=\min _{\beta_{\mathcal{I}}=0} \mathcal{L}_{n}(\beta), \mathcal{I}=(\mathcal{A})^{c}`.
# To identify the true model, the model complexity penalty is
# :math:`\log p` and the slow diverging rate :math:`\log \log n` is set to
# prevent underfitting. Theorem 4 states that the following ABESS
# algorithm selects the true support size via SIC.
#
# Let :math:`s_{\max }` be the maximum support size. We suggest
# :math:`s_{\max }=o\left(\frac{n}{\log p}\right)` as the maximum possible
# recovery size. Typically, we set
# :math:`s_{\max }=\left[\frac{n}{\log p \log \log n}\right]` where
# :math:`[x]` denotes the integer part of :math:`x`.
#
# .. _algorithm-3-abess:
#
# Algorithm 3: ABESS.
# '''''''''''''''''''
#
# 1. Input: :math:`X, y`, and the maximum support size :math:`s_{\max } .`
#
# 2. For :math:`s=1,2, \ldots, s_{\max }`, do
#
#    .. math:: \left(\hat{\boldsymbol{\beta}}_{s}, \hat{\boldsymbol{d}}_{s}, \hat{\mathcal{A}}_{s}, \hat{\mathcal{I}}_{s}\right)= \text{BESS.Fixed}(s).
#
#    End for
#
# 3. Compute the minimum of SIC:
#
#    .. math:: s_{\min }=\arg \min _{s} \operatorname{SIC}\left(\hat{\mathcal{A}}_{s}\right).
#
# 4. Output
#    :math:`\left(\hat{\boldsymbol{\beta}}_{s_{\min}}, \hat{\boldsymbol{d}}_{s_{\min }}, \hat{A}_{s_{\min }}, \hat{\mathcal{I}}_{s_{\min }}\right) .`
#
# Now, enjoy the data analysis with ``abess`` library:
import abess

# sphinx_gallery_thumbnail_path = '_static/icon_noborder.png'
