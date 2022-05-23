## Introduction

We compare `abess` and other well-known algorithms under linear regression and logistic regression models on several real-world datasets. The comparison is conducted on both Python and R environments. In what follow, we depict step-by-step instructions for properly using scripts in subdirectories. 
Note that, at presented, users should manually download the datasets from corresponding websites. 

## Prerequisite

### Python (version 3.9.1):

- abess (0.4.5)
- celer (0.6.1)
- scikit-learn (1.0.2)
- pandas
- numpy

### R (version 3.6.3)
- abess (0.4.5)
- elasiticnet (1.3.0)

## Step-by-step instruction
### superconductivity dataset
1. From https://archive.ics.uci.edu/ml/machine-learning-databases/00464/, download the `superconduct.zip` file into the `superconductivity` directory. Extract the `train.csv` from `superconduct.zip`, and put the `csv` file into the `superconductivity` directory. 
2. Run the `superconductivity.py` script:

```bash
python superconductivity.py
```

### cancer dataset

1. Download `chin.RData` (https://github.com/ramhiser/datamicroarray/blob/master/data/chin.RData) into the `cancer` directory. 
2. Run the `preprocess.R` script to produce files `chin_x.txt` and `chin_y.txt`:

```bash
Rscript preprocess.R
```

3. Run the `chin.py` script:
```bash
python chin.py
```

### musk dataset

1. Download `clean1.data` and `clean2.data` from https://archive.ics.uci.edu/ml/machine-learning-databases/musk. Put them into the `musk` directory. 

2. Run the `musk.py` script:
```bash
python musk.py
```

### genetic dataset

1. Download `christensen.RData` from https://github.com/ramhiser/datamicroarray/blob/master/data/christensen.RData, and put it into the `genetic` directory.
2. Run the `christensen.R` script:
```bash
Rscript christensen.R
```

