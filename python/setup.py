import os
import sys
from setuptools import setup, find_packages, Extension, dist
from os import path

dist.Distribution().fetch_build_eggs(['numpy'])
import numpy

os_type = 'MS_WIN64'
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# copy src
os.system(f'bash "{CURRENT_DIR}/copy_src.sh" "{CURRENT_DIR}"')

if sys.platform.startswith('win32'):
    python_path = sys.base_prefix
    temp = python_path.split("\\")
    version = str(sys.version_info.major) + str(sys.version_info.minor)
    path1 = "-I" + python_path + "\\include"
    path2 = "-L" + python_path + "\\libs"
    os.system(f'bash "{CURRENT_DIR}/pre.sh" ' + python_path + ' ' + version)

    cabess_module = Extension(name='abess._cabess',
                              sources=[CURRENT_DIR + '/src/api.cpp', CURRENT_DIR + '/src/List.cpp', CURRENT_DIR + '/src/utilities.cpp',
                                       CURRENT_DIR + '/src/normalize.cpp', CURRENT_DIR + '/src/pywrap.cpp',
                                       CURRENT_DIR + '/src/pywrap.i'],
                              language='c++',
                              extra_compile_args=["-DNDEBUG", "-fopenmp", "-O2", "-Wall", "-mavx", "-mfma", "-march=native",
                                                  "-std=c++11", "-mtune=generic", "-D%s" % os_type, path1, path2],
                              extra_link_args=['-lgomp'],
                              libraries=["vcruntime140"],
                              include_dirs=[
                                  numpy.get_include(), CURRENT_DIR + '/include'],
                              swig_opts=["-c++"]
                              )
elif sys.platform.startswith('darwin'):
    eigen_path = CURRENT_DIR + "/include"
    # print(eigen_path)
    # eigen_path = "/usr/local/include/eigen3/Eigen"
    cabess_module = Extension(name='abess._cabess',
                              sources=[CURRENT_DIR + '/src/api.cpp', CURRENT_DIR + '/src/List.cpp', CURRENT_DIR + '/src/utilities.cpp',
                                       CURRENT_DIR + '/src/normalize.cpp', CURRENT_DIR + '/src/pywrap.cpp',
                                       CURRENT_DIR + '/src/pywrap.i'],
                              language='c++',
                              extra_compile_args=[
                                  "-DNDEBUG", "-O2", "-Wall", "-std=c++11", "-mavx", "-mfma", "-march=native"],
                              include_dirs=[numpy.get_include(), eigen_path],
                              swig_opts=["-c++"]
                              )
else:
    eigen_path = CURRENT_DIR + "/include"
    # print(eigen_path)
    # eigen_path = "/usr/local/include/eigen3/Eigen"
    cabess_module = Extension(name='abess._cabess',
                              sources=[CURRENT_DIR + '/src/api.cpp', CURRENT_DIR + '/src/List.cpp', CURRENT_DIR + '/src/utilities.cpp',
                                       CURRENT_DIR + '/src/normalize.cpp', CURRENT_DIR + '/src/pywrap.cpp',
                                       CURRENT_DIR + '/src/pywrap.i'],
                              language='c++',
                              extra_compile_args=[
                                  "-DNDEBUG", "-fopenmp", "-O2", "-Wall", "-std=c++11", "-mavx", "-mfma", "-march=native"],
                              extra_link_args=['-lgomp'],
                              include_dirs=[numpy.get_include(), eigen_path],
                              swig_opts=["-c++"]
                              )
    pass

with open(path.join(CURRENT_DIR, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='abess',
      version='0.4.0',
      author="Jin Zhu, Kangkang Jiang, Junhao Huang, Yanhang Zhang, Junxian Zhu, Xueqin Wang",
      author_email="zhuj37@mail2.sysu.edu.cn",
      maintainer="Kangkang Jiang",
      maintainer_email="jiangkk3@mail2.sysu.edu.cn",
      package_dir={'': CURRENT_DIR},
      packages=find_packages(CURRENT_DIR),
      description="abess Python Package",
      long_description=long_description,
      long_description_content_type="text/x-rst",
      install_requires=[
          "numpy",
          "scipy",
          "scikit-learn>=0.24"
      ],
      license="GPL-3",
      url="https://abess.readthedocs.io",
      classifiers=[
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
      ],
      python_requires='>=3.5',
      ext_modules=[cabess_module]
      )
