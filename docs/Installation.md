# Installation

## Stable release

### Python
To install abess on Python, you can simply get the stable version with:

```python
pip install abess
```

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.

### R
To install stable version into R environment, run the command:
```R
install.packages("abess")
```

## Latest release

This page gives instructions on how to build and install abess from the source code. 
If the instructions do not help for you, please feel free to ask questions by opening an [issue](https://github.com/abess-team/abess/issues).

### Python 
Clone our [github project](https://github.com/abess-team/abess) to your device:

```powershell
git clone https://github.com/abess-team/abess.git
cd abess
```

Before installing abess itself, some dependencies should be installed first: `swig`, `bash`, `mingw`, which may be a little different in different platforms:

- **Linux**: `$ sudo apt install swig bash mingw-w64` (for Ubuntu, but other Linux systems are similar);
- **Windows**: `choco install swig mingw git ` (using [Chocolatey](https://community.chocolatey.org/packages)), or manually download the software and add them into PATH;
- **MacOS**: `brew install swig mingw-w64 bash` (using [brew](https://brew.sh/)).

What's more, some basic python packages are required. We have written in `abess/python/requirements.txt` and just install them with `pip`:

```powershell
pip install -r ./python/requirements.txt
```

After that, `cd` into `python` and manually install abess by conducting command:

```powershell
cd ./python
python setup.py install
```

If it finishes with "*Finished processing dependencies for abess*", the installation is successful.

> If you receive an error said "*Can't create or remove files in install directory*", this may be caused by permission denied. The step below may help with it.
>
> - For Linux/MacOS: run `python setup.py install --user` or `sudo python setup.py install` instead.
> - For Windows: run the command as administrator.

### R
To install the development version from GitHub, run the following code in R console:

```r
library(devtools)
install_github("abess-team/abess", build_vignettes = TRUE)
```

Windows user will need to install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) first.

## OpenMP Support

To support OpenMP parallelism in Cpp, the dependence for OpenMP should be install. 
In MacOS, the dependence can be installed by:       

```powershell
brew install llvm
brew install libomp
```

Without OpenMP, abess only use a single CPU core, leading to suboptimal learning speed.

