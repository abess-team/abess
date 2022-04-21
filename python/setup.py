import os
import re
import sys
import distutils
import subprocess

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

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

try:
    src_dir = os.path.join(CURRENT_DIR, os.path.pardir)

    dst = os.path.join(CURRENT_DIR, 'src')
    src = os.path.join(src_dir, 'src')
    distutils.dir_util.copy_tree(src, dst)

    dst = os.path.join(CURRENT_DIR, 'include')
    src = os.path.join(src_dir, 'include')
    distutils.dir_util.copy_tree(src, dst)
except BaseException:
    pass

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # extension dir
        extdir = os.path.abspath(
            os.path.dirname(
                self.get_ext_fullpath(
                    ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # cmake build dir
        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        # config: debug or release
        cfg = "DEBUG" if self.debug else "RELEASE"

        # arguments for "cmake" and "cmake --build"
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}"
        ]
        build_args = [
            "-j4",
            f"--config {cfg}"
        ]

        # Windows (MSVC)
        if self.compiler.compiler_type == "msvc":
            cmake_args += [
                f"-DMSVC=ON",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg}={extdir}",
                f"-A{PLAT_TO_CMAKE[self.plat_name]}"
            ]

        # MacOS
        if sys.platform.startswith("darwin"):
            # cmake_args += [
            #     f"-DDARWIN=ON"
            # ]

            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [
                    "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        subprocess.check_call(["cmake", ext.sourcedir]
                              + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."]
                              + build_args, cwd=build_temp)


with open(os.path.join(CURRENT_DIR, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

package_info = get_info()

setup(
    name='abess',
    version=package_info['__version__'],
    author=package_info['__author__'],
    author_email="zhuj37@mail2.sysu.edu.cn",
    maintainer="Junhao Huang",
    maintainer_email="huangjh256@mail2.sysu.edu.cn",
    packages=find_packages(),
    description="abess: Fast Best Subset Selection",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    install_requires=[
        "numpy",
        "pandas",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.6',
    ext_modules=[CMakeExtension("abess.pybind_cabess")],
    cmdclass={"build_ext": CMakeBuild}
)
