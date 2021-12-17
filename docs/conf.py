# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
# libpath = os.path.join(curr_path, '../python/')
# sys.path.insert(0, libpath)
# sys.path.insert(0, curr_path)
# sys.path.insert(0, sys.path.insert(0, os.path.abspath("../python/")))

# -- Project information -----------------------------------------------------
project = 'abess'
copyright = '2021, abess team'
author = 'abess team'

# The full version, including alpha/beta/rc tags
release = '0.3.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.githubpages',
              'nbsphinx',
              'myst_parser']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# autoapi_type = 'python'
# autoapi_dirs = ['../python/abess']
# autoapi_keep_files = False
autodoc_typehints = 'description'
# autoapi_add_toctree_entry = False
# autoapi_generate_api_docs = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

master_doc = 'index'

suppress_warnings=['myst.mathjax']

autodoc_inherit_docstrings=True
