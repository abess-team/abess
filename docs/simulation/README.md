## Introduction

We compare `abess` and other well-known algorithms under linear regression and logistic regression model. 
The comparison is conducted on both Python and R environments. 

## Prerequisite

### Python (version 3.9.1):

- abess (0.4.5)
- scikit-learn (1.0.2)
- numpy

### R (version 3.6.3)
- abess (0.4.5)
- glmnet (4.1-1)
- ncvreg (3.13.0)
- L0Learn (2.0.3)
- ggpubr
- mccr
- pROC
- ggplot2
- tidyr

## Python directory

### Files

- `run_benchmark_linear.py`: conducts simulation on sparse linear model
- `run_benchmark_logistic.py`: conducts simulation on sparse logistic regression
- `plot_results_figure.py`: visualizes the simulation results outputted by `run_benchmark_linear.py` and `run_benchmark_logistic.py`
- `python plot_important_search.py`: conducts simulation on the with/without important-searching technique

### Instructions

- To reproduce the simulation results in demonstrated in: https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_a1_power_of_abess.html, conduct:
```
python plot_results_figure.py
```


- Run `python plot_important_search.py` to reproduce the lastest figure in: https://abess.readthedocs.io/en/latest/auto_gallery/4-computation-tips/plot_large_dimension.html#experimental-evidences-important-searching.

## R directory

### Files

- `linear_source.R` and `logistic_source.R`: include core code for simulation
- `run_benchmark.R`: conducts simulation on sparse linear model and sparse logistic regression
- `plot_results_figure.R`: visualizes the simulation results outputted by `run_benchmark.R`

### Instruction
To reproduce the Figures in [this article](https://abess-team.github.io/abess/articles/v11-power-of-abess.html#results), run:
```bash
Rscript run_benchmark.R
Rscript plot_results_figure.R
```
