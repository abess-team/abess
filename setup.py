import os
import sys
import numpy
from setuptools import setup, find_packages, Extension
from os import path

os_type = 'MS_WIN64'
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if sys.platform.startswith('win32'):
    python_path = sys.base_prefix
    temp = python_path.split("\\")
    version = str(sys.version_info.major) + str(sys.version_info.minor)
    path1 = "-I" + python_path + "\\include"
    path2 = "-L" + python_path + "\\libs"
    os.system('bash pre.sh ' + python_path + ' ' + version)

    cbess_module = Extension(name='bess._cbess',
                             sources=['src/bess.cpp', 'src/List.cpp', 'src/utilities.cpp', 'src/normalize.cpp', 'src/bess.i',
                                      'src/Algorithm.cpp', 'src/Data.cpp', 'src/Metric.cpp', 'src/path.cpp',
                                      'src/logistic.cpp', 'src/coxph.cpp', 'src/poisson.cpp', 'src/screening.cpp'],
                             language='c++',
                             extra_compile_args=["-DNDEBUG", "-fopenmp", "-O2", "-Wall", "-std=c++11", "-mtune=generic", "-D%s" % os_type, path1, path2],
                             extra_link_args=['-lgomp'],
                             libraries=["vcruntime140"],
                             include_dirs=[numpy.get_include(), 'include'],
                             swig_opts=["-c++"]
                             )
else:
      eigen_path = CURRENT_DIR + "/include"
      print(eigen_path)
      # eigen_path = "/usr/local/include/eigen3/Eigen"
      cbess_module = Extension(name='bess._cbess',
                              sources=['src/bess.cpp', 'src/List.cpp', 'src/utilities.cpp', 'src/normalize.cpp', 'src/bess.i',
                                          'src/Algorithm.cpp', 'src/Data.cpp', 'src/Metric.cpp', 'src/path.cpp',
                                          'src/logistic.cpp', 'src/coxph.cpp', 'src/poisson.cpp', 'src/screening.cpp'],
                              language='c++',
                              extra_compile_args=["-DNDEBUG", "-fopenmp", "-O2", "-Wall", "-std=c++11"],
                              extra_link_args=['-lgomp'],
                              include_dirs=[numpy.get_include(), eigen_path],
                              swig_opts=["-c++"]
                              )
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
      long_description = f.read()

setup(name='bess',
      version='0.0.12',
      author="Kangkang Jiang, Jin Zhu, Yanhang Zhang, Shijie Quan, Xueqin Wang",
      author_email="jiangkk3@mail2.sysu.edu.cn",
      maintainer="Kangkang Jiang",
      maintainer_email="jiangkk3@mail2.sysu.edu.cn",
      packages=find_packages(),
      description="bess Python Package",
      long_description=long_description,
      long_description_content_type="text/markdown",
      install_requires=[
          'numpy'
      ],
      license="GPL-3",
      url="https://github.com/Mamba413/bess",
      classifiers=[
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
      ],
      python_requires='>=3.5',
      ext_modules=[cbess_module]
      )


