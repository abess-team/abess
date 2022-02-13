# FAQ

Some frequently asked questions would be shown here. 
If the error you met is not contained here, please open an issue
on out project [https://github.com/abess-team/abess/issues](https://github.com/abess-team/abess/issues).

## Python package

### Installation failed

#### Compliers

First of all, please check the version of Python and GCC.
To make sure that `abess` package runs correctly, 

- Python 2.7 or later is required
- GCC 4.7 or later is required (support c++11)

What's more, the newer version is recommended. So if you meet some 
errors, please try to update the complier first.

In Windows, you may receive an error said `Microsoft Visual C++ *version* is required`. That is because MinGW is not used to complie. Please check if MinGW is in the PATH ( or try to reinstall it) and run the installation again. 

However, if it still doesn't work, please install by:

```powershell
# in the `abess/python` directory
$ pip install --global-option build --global-option --complier=mingw32 .
```

It will force the installation to use MinGW.

#### Permission

If you receive an error said "*Can't create or remove files in install directory*" during the installation, this may be caused by permission denied. The step below would help with it.

- For Linux/MacOS: run `$ python setup.py install --user` or `$ sudo python setup.py install` instead.
- For Windows: run the command as an administrator.


### Import failed

#### Folder name

Make sure your working folder path is not named "abess". If not, Python would not import the `abess` packages and give some errors.

## R package       

- Update Rcpp package if you encounter the following errors:
```r
function 'Rcpp_precious_remove' not provided by package 'Rcpp'
```