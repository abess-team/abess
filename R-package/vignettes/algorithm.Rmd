---
title: 'ABESS algorithm: details'
date: "2021/7/22"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, message = F, warning = F)
```

## Sacrifices

Consider the $\ell_{0}$ constraint minimization problem,
$$
\min _{\boldsymbol{\beta}} \mathcal{L}_{n}(\beta), \quad \text { s.t }\|\boldsymbol{\beta}\|_{0} \leq \mathrm{s}
$$
where $\mathcal{L}_{n}(\beta)=\frac{1}{2 n}\|y-X \beta\|_{2}^{2} .$ Without loss of generality, we consider $\|\boldsymbol{\beta}\|_{0}=\mathrm{s}$. Given any initial set $\mathcal{A} \subset \mathcal{S}=\{1,2, \ldots, p\}$ with cardinality $|\mathcal{A}|=s$,
denote $\mathcal{I}=\mathcal{A}^{\mathrm{c}}$ and compute
$$
\hat{\boldsymbol{\beta}}=\arg \min _{\boldsymbol{\beta}_{\mathcal{I}}=0} \mathcal{L}_{n}(\boldsymbol{\beta})
$$
We call $\mathcal{A}$ and $\mathcal{I}$ as the active set and the inactive set, respectively.

Given the active set $\mathcal{A}$ and $\hat{\boldsymbol{\beta}}$, we can define the following two types of sacrifices:
1) Backward sacrifice: For any $j \in \mathcal{A}$, the magnitude of discarding variable $j$ is,
$$
\xi_{j}=\mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}}^{\mathcal{A} \backslash\{j\}}\right)-\mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}}^{\mathcal{A}}\right)=\frac{X_{j}^{\top} X_{j}}{2 n}\left(\hat{\beta}_{j}\right)^{2}
$$
2) Forward sacrifice: For any $j \in \mathcal{I}$, the magnitude of adding variable $j$ is,
$$
\zeta_{j}=\mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}^{\mathcal{A}}}\right)-\mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}}^{\mathcal{A}}+\hat{t}^{\{j\}}\right)=\frac{\boldsymbol{X}_{j}^{\top} X_{j}}{2 n}\left(\frac{\hat{d}_{j}}{X_{j}^{\top} X_{j} / n}\right)^{2}
$$
where $\hat{t}=\arg \min _{t} \mathcal{L}_{n}\left(\hat{\boldsymbol{\beta}}^{\mathcal{A}}+t^{\{j\}}\right), \hat{d}_{j}=X_{j}^{\top}(y-X \hat{\boldsymbol{\beta}}) / n$
Intuitively, for $j \in \mathcal{A}$ (or $j \in \mathcal{I}$ ), a large $\xi_{j}$ (or $\zeta_{j}$) implies the $j$ th variable is potentially important.

## Algorithm

### Best-Subset Selection with a Given Support Size
 Unfortunately, it is noteworthy that these two sacrifices are incomparable because they have different sizes of support set. However, if we exchange some "irrelevant" variables in $\mathcal{A}$ and some "important" variables in $\mathcal{I}$, it may result in a higher-quality solution. This intuition motivates our splicing method. Specifically, given any splicing size $k \leq s$, define

$$
\mathcal{A}_{k}=\left\{j \in \mathcal{A}: \sum_{i \in \mathcal{A}} \mathrm{I}\left(\xi_{j} \geq \xi_{i}\right) \leq k\right\}
$$
to represent $k$ least relevant variables in $\mathcal{A}$ and
$$
\mathcal{I}_{k}=\left\{j \in \mathcal{I}: \sum_{i \in \mathcal{I}} \mid\left(\zeta_{j} \leq \zeta_{i}\right) \leq k\right\}
$$
to represent $k$ most relevant variables in $\mathcal{I} .$ Then, we splice $\mathcal{A}$ and $\mathcal{I}$ by exchanging $\mathcal{A}_{k}$ and $\mathcal{I}_{k}$ and obtain a new active set
$$
\tilde{\mathcal{A}}=\left(\mathcal{A} \backslash \mathcal{A}_{k}\right) \cup \mathcal{I}_{k}
$$
Let $\tilde{\mathcal{I}}=\tilde{\mathcal{A}}^{c}, \tilde{\boldsymbol{\beta}}=\arg \min _{\boldsymbol{\beta}_{\overline{\mathcal{I}}}=0} \mathcal{L}_{n}(\boldsymbol{\beta})$, and $\tau_{s}>0$ be a threshold. If $\tau_{s}<$
$\mathcal{L}_{n}(\hat{\beta})-\mathcal{L}_{n}(\tilde{\beta})$, then $\tilde{A}$ is preferable to $\mathcal{A} .$ The active set can be updated
iteratively until the loss function cannot be improved by splicing. Once the algorithm recovers the true active set, we may splice some irrelevant variables, and then the loss function may decrease slightly. The threshold $\tau_{s}$ can reduce this unnecessary calculation. Typically, $\tau_{s}$ is relatively small, e.g. $\tau_{s}=0.01 s \log (p) \log (\log n) / n$


#### Algorithm 1: BESS.Fix(s): Best-Subset Selection with a given support size s.

1) Input: $X, y$, a positive integer $k_{\max }$, and a threshold $\tau_{s}$.
2) Initialize $\mathcal{A}^{0}=\left\{j: \sum_{i=1}^{p} \mathrm{I}\left(\left|\frac{\boldsymbol{X}_{I}^{\top} \boldsymbol{y}}{\sqrt{\boldsymbol{x}_{j}^{\top} \boldsymbol{x}_{j}}}\right| \leq \mid \frac{\boldsymbol{X}_{i}^{\top} \boldsymbol{y}}{\sqrt{\boldsymbol{x}_{I}^{\top} \boldsymbol{x}_{l}}} \leq \mathrm{s}\right\}, \mathcal{I}^{0}=\left(\mathcal{A}^{0}\right)^{c}\right.$,
and $\left(\beta^{0}, d^{0}\right):$

    $\boldsymbol{\beta}_{\mathcal{I}^{0}}^{0}=0$

    $d_{\mathcal{A}^{0}}^{0}=0$

    $\boldsymbol{\beta}_{\mathcal{A}^{0}}^{0}=\left(\boldsymbol{X}_{\mathcal{A}^{0}}^{\top} \boldsymbol{X}_{\mathcal{A}^{0}}\right)^{-1} \boldsymbol{X}_{\mathcal{A}^{0}}^{\top} \boldsymbol{y}$

    $d_{\mathcal{I}^{0}}^{0}=X_{\mathcal{I}^{0}}^{\top}\left(\boldsymbol{y}-\boldsymbol{X} \boldsymbol{\beta}^{0}\right) / n$

3) For $m=0,1, \ldots$, do 

   $\left(\boldsymbol{\beta}^{m+1}, \boldsymbol{d}^{m+1}, \mathcal{A}^{m+1}, \mathcal{I}^{m+1}\right)=$ Splicing $\left(\boldsymbol{\beta}^{m}, \boldsymbol{d}^{m}, \mathcal{A}^{m}, \mathcal{I}^{m}, k_{\max }, \tau_{s}\right)$

    If $\left(\mathcal{A}^{m+1}, \mathcal{I}^{m+1}\right)=\left(\mathcal{A}^{m}, \mathcal{I}^{m}\right)$, 
    
    then stop

    end for

4) Output $(\hat{\boldsymbol{\beta}}, \hat{\boldsymbol{d}}, \hat{\mathcal{A}}, \hat{\mathcal{I}})=\left(\boldsymbol{\beta}^{m+1}, \boldsymbol{d}^{m+1} \mathcal{A}^{m+1}, \mathcal{I}^{m+1}\right)$


#### Algorithm 2: Splicing $\left(\beta, d, \mathcal{A}, \mathcal{I}, k_{\max }, \tau_{s}\right)$

1) Input: $\boldsymbol{\beta}, \boldsymbol{d}, \mathcal{A}, \mathcal{I}, k_{\max }$, and $\tau_{\mathrm{s}} .$
2) Initialize $L_{0}=L=\frac{1}{2 n}\|y-X \beta\|_{2}^{2}$, and set $\xi_{j}=\frac{x_{l}^{\top} X_{J}}{2 n}\left(\beta_{j}\right)^{2}, \zeta_{j}=\frac{x_{J}^{\top} x_{j}}{2 n}\left(\frac{d_{j}}{x_{J}^{\top} x_{j} / n}\right)^{2}, j=1, \ldots, p$
3) For $k=1,2, \ldots, k_{\max }$, do

     $\mathcal{A}_{k}=\left\{j \in \mathcal{A}: \sum_{i \in \mathcal{A}} \mathrm{I}\left(\xi_{j} \geq \xi_{i}\right) \leq k\right\}$

    $\mathcal{I}_{k}=\left\{j \in \mathcal{I}: \sum_{i \in \mathcal{I}}^{i \in \mathcal{A}} \mathrm{I}\left(\zeta_{j} \leq \zeta_{i}\right) \leq k\right\}$

    Let $\tilde{\mathcal{A}}_{k}=\left(\mathcal{A} \backslash \mathcal{A}_{k}\right) \cup \mathcal{I}_{k}, \tilde{\mathcal{I}}_{k}=\left(\mathcal{I} \backslash \mathcal{I}_{k}\right) \cup \mathcal{A}_{k}$ and solve

    $\tilde{\boldsymbol{\beta}}_{\overline{\mathcal{A}}_{k}}=\left(\boldsymbol{X}_{\mathcal{A}_{k}}^{\top} \boldsymbol{X}_{\overline{\mathcal{A}}_{k}}\right)^{-1} \boldsymbol{X}_{\overline{\mathcal{A}_{k}}}^{\top} y, \quad \tilde{\boldsymbol{\beta}}_{\overline{\mathcal{I}}_{k}}=0$

    $\tilde{d}=X^{\top}(y-X \tilde{\beta}) / n, \quad \mathcal{L}_{n}(\tilde{\beta})=\frac{1}{2 n}\|y-X \tilde{\beta}\|_{2}^{2}$

    If $L>\mathcal{L}_{n}(\tilde{\beta})$, then

    $(\hat{\boldsymbol{\beta}}, \hat{\boldsymbol{d}}, \hat{\mathcal{A}}, \hat{\mathcal{I}})=\left(\tilde{\boldsymbol{\beta}}, \tilde{\boldsymbol{d}}, \tilde{\mathcal{A}}_{k}, \tilde{\mathcal{I}}_{k}\right)$

    $L=\mathcal{L}_{n}(\tilde{\beta})$

    End for

4) If $L_{0}-L<\tau_{s}$, then $(\hat{\beta}, \hat{d}, \hat{A}, \hat{I})=(\beta, d, \mathcal{A}, \mathcal{I})$
5) Output $(\hat{\boldsymbol{\beta}}, \hat{\boldsymbol{d}}, \hat{\mathcal{A}}, \hat{\mathcal{I}})$.


ABESS. In practice, the support size is usually unknown. We use a datadriven procedure to determine s. Information criteria such as highdimensional BIC (HBIC) (13) and extended BIC (EBIC) (14) are commonly used for this purpose. Specifically, HBIC (13) can be applied to select the tuning parameter in penalized likelihood estimation. To recover the support size $s$ for the best-subset selection, we introduce a criterion that is a special case of HBIC (13). While HBIC aims to tune the parameter for a nonconvex penalized regression, our proposal is used to determine the size of best subset. For any active set $\mathcal{A}$, define an $\mathrm{SIC}$ as follows:
$$
\operatorname{SIC}(\mathcal{A})=n \log \mathcal{L}_{\mathcal{A}}+|\mathcal{A}| \log (p) \log \log n
$$
where $\mathcal{L}_{\mathcal{A}}=\min _{\beta_{\mathcal{I}}=0} \mathcal{L}_{n}(\beta), \mathcal{I}=(\mathcal{A})^{c}$. To identify the true model, the
model complexity penalty is $\log p$ and the slow diverging rate $\log \log n$ is set to prevent underfitting. Theorem 4 states that the following ABESS algorithm selects the true support size via SIC.

Let $s_{\max }$ be the maximum support size. Theorem 4 suggests $s_{\max }=o\left(\frac{n}{\log p}\right)$ as the maximum possible recovery size. Typically, we set $s_{\max }=\left[\frac{n}{\log p \log \log n}\right]$
where $[x]$ denotes the integer part of $x$.

### Determining the Best Support Size with SIC

In practice, the support size is usually unknown. We use a datadriven procedure to determine s. For any active set $\mathcal{A}$, define an $\mathrm{SIC}$ as follows:
$$
\operatorname{SIC}(\mathcal{A})=n \log \mathcal{L}_{\mathcal{A}}+|\mathcal{A}| \log (p) \log \log n
$$
where $\mathcal{L}_{\mathcal{A}}=\min _{\beta_{\mathcal{I}}=0} \mathcal{L}_{n}(\beta), \mathcal{I}=(\mathcal{A})^{c}$. To identify the true model, the
model complexity penalty is $\log p$ and the slow diverging rate $\log \log n$ is set to prevent underfitting. Theorem 4 states that the following ABESS algorithm selects the true support size via SIC.

Let $s_{\max }$ be the maximum support size. We suggest $s_{\max }=o\left(\frac{n}{\log p}\right)$ as the maximum possible recovery size. Typically, we set $s_{\max }=\left[\frac{n}{\log p \log \log n}\right]$
where $[x]$ denotes the integer part of $x$.


#### Algorithm 3: ABESS.

1) Input: $X, y$, and the maximum support size $s_{\max } .$

2) For $s=1,2, \ldots, s_{\max }$, do

    $\left(\hat{\boldsymbol{\beta}}_{s}, \hat{\boldsymbol{d}}_{s}, \hat{\mathcal{A}}_{s}, \hat{\mathcal{I}}_{s}\right)=$ BESS.Fixed(s)

    End for

3) Compute the minimum of SIC:

    $s_{\min }=\arg \min _{s} \operatorname{SIC}\left(\hat{\mathcal{A}}_{s}\right)$

4) Output $\left(\hat{\boldsymbol{\beta}}_{s_{\operatorname{mln}}}, \hat{\boldsymbol{d}}_{s_{\min }}, \hat{A}_{s_{m \ln }}, \hat{\mathcal{I}}_{s_{\min }}\right) .$