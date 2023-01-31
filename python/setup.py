import os
import re
import sys
import distutils
import subprocess
# import platform

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.parallel = 4


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(
                self.get_ext_fullpath(
                    ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(
            os.environ.get(
                "DEBUG",
                0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}"
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:
            cmake_args += ["-DMSVC=ON"]

            # Single config generators are handled "normally"
            single_config = any(
                x in cmake_generator for x in {
                    "NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward
            # compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            cmake_args += ["-DDARWIN=ON"]
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [
                    "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or
            # PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] +
                              cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] +
                              build_args, cwd=build_temp)


def get_info():
    # get information from `__init__.py`
    labels = ["__version__", "__author__"]
    values = ["" for _ in labels]
    with open(os.path.join(CURRENT_DIR, "abess/__init__.py")) as f:
        for line in f.read().splitlines():
            for i, label in enumerate(labels):
                if line.startswith(label):
                    values[i] = line.split('"')[1]
                    break
            if "" not in values:
                break
    with open(os.path.join(CURRENT_DIR, 'README.rst'), encoding='utf-8') as f:
        s = f.read()
        labels.append("long_description")
        values.append(s)
    return dict(zip(labels, values))


def copy_src():
    # copy files from parent dir
    need_clean_tree = set()
    try:
        src_dir = os.path.join(CURRENT_DIR, os.path.pardir)

        dst = os.path.join(CURRENT_DIR, 'src')
        src = os.path.join(src_dir, 'src')
        distutils.dir_util.copy_tree(src, dst)
        need_clean_tree.add(os.path.abspath(dst))

        dst = os.path.join(CURRENT_DIR, 'include')
        src = os.path.join(src_dir, 'include')
        distutils.dir_util.copy_tree(src, dst)
        need_clean_tree.add(os.path.abspath(dst))
    except BaseException:
        pass
    return need_clean_tree

# print("sys.platform output: {}".format(sys.platform))
# print("platform.processor() output: {}".format(platform.processor()))


need_clean_tree = copy_src()
package_info = get_info()

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
    long_description=package_info['long_description'],
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.6',
    ext_modules=[CMakeExtension("abess.pybind_cabess")],
    cmdclass={"build_ext": CMakeBuild}
)
