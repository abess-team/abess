## abess configure.ac
##
## Copyright (C) Jin Zhu <zhuj37@mail2.sysu.edu.cn>

# require at least autoconf 2.61
AC_PREREQ(2.61)

# Process this file with autoconf to produce a configure script.
# Here, we haven't used command "AC_INIT([RProtoBuf], m4_esyscmd_s([awk -e '/^Version:/ {print $2}' DESCRIPTION]))" because it doesn't work sometimes.
AC_INIT([abess], [0.4.5])

# Find the compiler and compiler flags used by R.
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  AC_MSG_ERROR([Could not determine R_HOME.])
fi
CXX=`"${R_HOME}/bin/R" CMD config CXX`
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`
AC_LANG(C++)
AC_REQUIRE_CPP
AC_PROG_CPP

# copy necessary dependence code into R-package
echo '*****************************************************************************************'
## Spectra library
if test -d inst/include/Spectra/
then
  echo "Spectra library exists. "
else
  echo "Spectra library does not exist. Copy from ../include/ directory"
  cp -rf ../include inst
fi
echo '*****************************************************************************************'

## copy core code into R-package/src
cd src
echo '*****************************************************************************************'
if test -f workflow.h 
then
  echo "Complier code exists in src/ directory. Install R package according to the complier code."
else 
  echo "Complier code does not exist. Copy from the ../src/ directory."
  cp -rf ../../src/*.h ../../src/*.cpp .
fi
echo '*****************************************************************************************'
cd ..

"${R_HOME}/bin/Rscript" -e "Rcpp::compileAttributes()"

OPENMP_CXXFLAGS=""

if test `uname -s` = "Linux"
then
  OPENMP_CXXFLAGS="\$(SHLIB_OPENMP_CXXFLAGS)"
fi

if test `uname -s` = "Darwin"
then
  OPENMP_CXXFLAGS='-Xclang -fopenmp'
  OPENMP_LIB='-lomp'
  ac_pkg_openmp=no
  AC_MSG_CHECKING([whether OpenMP will work in a package])
  AC_LANG_CONFTEST([AC_LANG_PROGRAM([[#include <omp.h>]], [[ return (omp_get_max_threads() <= 1); ]])])
  ${CC} -o conftest conftest.c ${OPENMP_LIB} ${OPENMP_CXXFLAGS} 2>/dev/null && ./conftest && ac_pkg_openmp=yes
  AC_MSG_RESULT([${ac_pkg_openmp}])
  if test "${ac_pkg_openmp}" = no; then
    OPENMP_CXXFLAGS=''
    OPENMP_LIB=''
    echo '*****************************************************************************************'
    echo '         OpenMP is unavailable on this Mac OSX system. Training speed may be suboptimal.'
    echo '         To use all CPU cores for training jobs, you should install OpenMP by running\n'
    echo '             brew install libomp'
    echo '*****************************************************************************************'
  fi
fi

# Write the flags into the src/Makevars file.
AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST(OPENMP_LIB)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
