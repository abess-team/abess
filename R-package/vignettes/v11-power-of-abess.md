---
title: "Power of abess"
author: "Liyuan Hu"
date: "2021/8/2"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE,warning = F, message = F)
```

## Simulation

To show the computational efficiency of abess, 
we compare abess R package with popular R libraries: glmnet, ncvreg, picasso for linear, logistic and poisson regressions; 
Timings of the CPU execution are recorded in seconds and averaged over 100 replications on a sequence
of 100 regularization parameters.


All experiments are
evaluated on an  Intel(R) Core(TM) i9-9940X CPU @ 3.30GHz 3.31 GHz and under R version 3.6.1. 

```r
source("R-package/example/perform.R")
```

The results are presented in the following picture. First, among all of the methods implemented in different packages,
the estimator obtained by abess package shows the best prediction performance and can reasonably control the false-positive rate 
at a low level like SCAD and MCP. Furthermore, our abess package is highly efficient compared with 
other packages.


<center> Figure 1. Performance for different packages </center>
![avatar]('https://raw.githubusercontent.com/abess-team/abess/master/docs/perform/performance.png')
<img src='https://raw.githubusercontent.com/abess-team/abess/master/docs/perform/performance.png' align="right" height="120"  align=center/>


<center> Figure 2. Runing Time for different packages </center>
<img src='https://raw.githubusercontent.com/abess-team/abess/master/docs/perform/time_perform.png' align="right" height="120"  align=center/>
