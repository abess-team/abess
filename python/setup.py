import os
import sys
import platform
from setuptools import setup, find_packages, Extension, dist

dist.Distribution().fetch_build_eggs(['numpy'])
import numpy

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def get_info():
    # get information from `__init__.py`
    labels = ["__version__", "__author__"]
    values = ["" for label in labels]
    with open(os.path.join(CURRENT_DIR, "abess/__init__.py")) as f:
        for line in f.read().splitlines():
            for i, label in enumerate(labels):
                if line.startswith(label):
                    values[i] = line.split('"')[1]
                    break
            if "" not in values:
                break
    return dict(zip(labels, values))


package_info = get_info()

# copy src
os.system('bash "{}/copy_src.sh" "{}"'.format(CURRENT_DIR, CURRENT_DIR))

if sys.platform.startswith('win32'):
    os_type = 'MS_WIN64'
    python_path = sys.base_prefix
    temp = python_path.split("\\")
    version = str(sys.version_info.major) + str(sys.version_info.minor)
    path1 = "-I" + python_path + "\\include"
    path2 = "-L" + python_path + "\\libs"
    os.system('bash "{}/pre.sh" '.format(CURRENT_DIR) +
              python_path + ' ' + version)

    ## compiler options:
    extra_compile_args=[
        "-DNDEBUG", "-fopenmp",
        "-O2", "-Wall",
        "-Wno-int-in-bool-context"
    ]
    ## uncomment mingw64_unable_extra_compile_args if the error:
    ## "Error: invalid register for .seh_savexmm"
    mingw64_unable_extra_compile_args=[
        "-mavx", "-mfma",
        "-march=native"
    ]
    extra_compile_args2=[
        "-std=c++11",
        "-mtune=generic",
        "-D%s" % os_type,
        path1, path2
    ]
    extra_compile_args.extend(mingw64_unable_extra_compile_args)
    extra_compile_args.extend(extra_compile_args2)

    ## C extension:
    cabess_module = Extension(
        name='abess._cabess',
        sources=[
            CURRENT_DIR + '/src/api.cpp',
            CURRENT_DIR + '/src/List.cpp',
            CURRENT_DIR + '/src/utilities.cpp',
            CURRENT_DIR + '/src/normalize.cpp',
            CURRENT_DIR + '/src/pywrap.cpp',
            CURRENT_DIR + '/src/pywrap.i'],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=['-lgomp'],
        libraries=["vcruntime140"],
        include_dirs=[
            numpy.get_include(),
            CURRENT_DIR + '/include'
        ],
        swig_opts=["-c++"]
    )
elif sys.platform.startswith('darwin'):
    eigen_path = CURRENT_DIR + "/include"

    # compatible compile args with M1 chip:
    extra_compile_args = [
        "-DNDEBUG", "-O2",
        "-Wall", "-std=c++11",
        "-Wno-int-in-bool-context"
    ]
    m1chip_unable_extra_compile_args = [
        "-mavx", "-mfma",
        "-march=native"
    ]
    if platform.processor() != 'arm':
        extra_compile_args.extend(m1chip_unable_extra_compile_args)
        pass

    cabess_module = Extension(
        name='abess._cabess',
        sources=[CURRENT_DIR + '/src/api.cpp',
                 CURRENT_DIR + '/src/List.cpp',
                 CURRENT_DIR + '/src/utilities.cpp',
                 CURRENT_DIR + '/src/normalize.cpp',
                 CURRENT_DIR + '/src/pywrap.cpp',
                 CURRENT_DIR + '/src/pywrap.i'],
        language='c++',
        extra_compile_args=extra_compile_args,
        include_dirs=[
            numpy.get_include(),
            eigen_path
        ],
        swig_opts=["-c++"]
    )
else:
    eigen_path = CURRENT_DIR + "/include"
    # print(eigen_path)
    # eigen_path = "/usr/local/include/eigen3/Eigen"
    cabess_module = Extension(
        name='abess._cabess',
        sources=[CURRENT_DIR + '/src/api.cpp',
                 CURRENT_DIR + '/src/List.cpp',
                 CURRENT_DIR + '/src/utilities.cpp',
                 CURRENT_DIR + '/src/normalize.cpp',
                 CURRENT_DIR + '/src/pywrap.cpp',
                 CURRENT_DIR + '/src/pywrap.i'],
        language='c++',
        extra_compile_args=[
            "-DNDEBUG", "-fopenmp",
            "-O2", "-Wall",
            "-std=c++11", "-mavx",
            "-mfma", "-march=native",
            # "-Wno-unused-variable",
            # "-Wno-unused-but-set-variable",
            "-Wno-int-in-bool-context"  # avoid warnings from Eigen
        ],
        extra_link_args=['-lgomp'],
        include_dirs=[
            numpy.get_include(),
            eigen_path
        ],
        swig_opts=["-c++"]
    )
    pass

with open(os.path.join(CURRENT_DIR, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='abess',
    version=package_info['__version__'],
    author=package_info['__author__'],
    author_email="zhuj37@mail2.sysu.edu.cn",
    maintainer="Kangkang Jiang",
    maintainer_email="jiangkk3@mail2.sysu.edu.cn",
    packages=find_packages(),
    description="abess: Fast Best Subset Selection",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn>=0.24"
    ],
    license="GPL-3",
    url="https://abess.readthedocs.io",
    download_url="https://pypi.python.org/pypi/abess",
    project_urls={
        "Bug Tracker": "https://github.com/abess-team/abess/issues",
        "Documentation": "https://abess.readthedocs.io",
        "Source Code": "https://github.com/abess-team/abess",
    },
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
