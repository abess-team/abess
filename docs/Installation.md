# Installation

## Stable release

### Python
To install abess on Python, you can simply get the stable version with:

```python
pip install abess
```

### R
To install stable version into R environment, run the command:
```R
install.packages("Ball")
```

### OpenMP Support

To support OpenMP parallelism in Cpp, the dependence for OpenMP should be install. 
In MacOS, the dependence can be installed by:       

```powershell
brew install llvm
brew install libomp
```

Without OpenMP, abess only use a single CPU core, leading to suboptimal learning speed.

## Latest release

This page gives instructions on how to build and install abess from the source code. 
If the instructions do not help for you, please feel free to ask questions by opening an [issue](https://github.com/abess-team/abess/issues).

### Python 
Clone our github project to your device:

```powershell
git clone https://github.com/abess-team/abess.git
```

After cloning, and manually install python by conducting command:

```powershell
cd abess/python
python setup.py install
```

If it finishes with "*Finished processing dependencies for abess*", the installation is successful.

### R
To install the development version from GitHub, run the following code in R console:

```r
library(devtools)
install_github("abess-team/abess", build_vignettes = TRUE)
```
Windows user will need to install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) first.




