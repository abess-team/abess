import os
import sys
import platform
import distutils
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension

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

# copy files from parent dir
NEED_CLEAN_TREE = set()
try:
    target_dir = CURRENT_DIR
    src_dir = os.path.join(CURRENT_DIR, os.path.pardir)

    dst = os.path.join(target_dir, 'src')
    src = os.path.join(src_dir, 'src')
    distutils.dir_util.copy_tree(src, dst)
    NEED_CLEAN_TREE.add(os.path.abspath(dst))

    dst = os.path.join(target_dir, 'include')
    src = os.path.join(src_dir, 'include')
    distutils.dir_util.copy_tree(src, dst)
    NEED_CLEAN_TREE.add(os.path.abspath(dst))
except BaseException:
    pass

# print("sys.platform output: {}".format(sys.platform))
# print("platform.processor() output: {}".format(platform.processor()))

if sys.platform.startswith('win32'):
    # os_type = 'MS_WIN64'

    pybind_cabess_module = Pybind11Extension(
        name='pybind_cabess',
        sources=[
            'src/api.cpp',
            'src/List.cpp',
            'src/utilities.cpp',
            'src/normalize.cpp',
            'src/pywrap.cpp'],
        extra_compile_args=[
            "/openmp",
            "/O2", "/W4",
            "/arch:AVX2"
        ],
        include_dirs=[
            'include'
        ]
    )
elif sys.platform.startswith('darwin'):
    # compatible compile args with M1 chip:
    extra_compile_args = [
        "-DNDEBUG", "-O2",
        "-Wall", "-std=c++11",
        "-Wno-int-in-bool-context"
    ]
    m1chip_unable_extra_compile_args = [
        # "-mavx",
        # "-mfma"
        # "-march=native"

        # Enable the "-mavx", "-mfma", "-march=native"
        # would improve the computational efficiency.
        # "-mavx" and "-mfma" do not supported
        # by github-action environment when arch = x86_64
        # "-mavx" and "-mfma" do not supported
        # by github-action when building arm64
        # (because it the default is not arm64)
    ]
    if platform.processor() not in ('arm', 'arm64'):
        extra_compile_args.extend(m1chip_unable_extra_compile_args)
        pass

    pybind_cabess_module = Pybind11Extension(
        name='pybind_cabess',
        sources=['src/api.cpp',
                 'src/List.cpp',
                 'src/utilities.cpp',
                 'src/normalize.cpp',
                 'src/pywrap.cpp'],
        extra_compile_args=extra_compile_args,
        include_dirs=[
            'include'
        ]
    )
else:
    pybind_cabess_module = Pybind11Extension(
        name='pybind_cabess',
        sources=['src/api.cpp',
                 'src/List.cpp',
                 'src/utilities.cpp',
                 'src/normalize.cpp',
                 'src/pywrap.cpp'],
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
            'include'
        ]
    )
    pass

with open(os.path.join(CURRENT_DIR, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='abess',
    version=package_info['__version__'],
    author=package_info['__author__'],
    author_email="zhuj37@mail2.sysu.edu.cn",
    maintainer="Junhao Huang",
    maintainer_email="huangjh256@mail2.sysu.edu.cn",
    # package_dir={'': CURRENT_DIR},
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
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.5',
    ext_modules=[pybind_cabess_module]
)
