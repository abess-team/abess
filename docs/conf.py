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
import sphinx_gallery
import sphinx_rtd_theme
import sphinx_gallery.sorting
# import os
# import sys
import sys
# import abess
import matplotlib
# sys.path.insert(0, os.path.join(os.path.abspath('..'), "python"))
# import SampleModule
# from sphinx_gallery.sorting import FileNameSortKey

# Use RTD Theme

# -- Project information -----------------------------------------------------

project = 'ABESS'
copyright = '2020, abess-team'
author = 'abess-team'

# The full version, including alpha/beta/rc tags
release = '0.4.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    'sphinx.ext.napoleon',
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    'sphinx.ext.intersphinx',
    'sphinx_gallery.gen_gallery'
    # ,
    # 'sphinx_toggleprompt'
]

matplotlib.use('agg')

# -- numpydoc
# Below is needed to prevent errors
numpydoc_show_class_members = False

# -- sphinx.ext.autosummary
autosummary_generate = True

# -- sphinx.ext.autodoc
autoclass_content = "both"
autodoc_default_flags = ["members", "inherited-members"]
autodoc_member_order = "bysource"  # default is alphabetical

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
source_suffix = ".rst"
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
master_doc = "index"
source_encoding = "utf-8"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

pygments_style = "sphinx"
smartquotes = False

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    # 'includehidden': False,
    "collapse_navigation": False,
    "navigation_depth": 3,
    "logo_only": True,
}
html_logo = "./image/apple-touch-icon.png"
html_favicon = "./image/favicon-32x32.png"
html_context = {
    # Enable the "Edit in GitHub link within the header of each page.
    "display_github": True,
    # Set the following variables to generate the resulting github URL for each page.
    # Format Template: https://{{ github_host|default("github.com") }}/{{
    # github_user }}/{{ github_repo }}/blob/{{ github_version }}{{
    # conf_py_path }}{{ pagename }}{{ suffix }}
    "github_user": "abess-team",
    "github_repo": "abess",
    "github_version": "main/docs/",
}
htmlhelp_basename = "abessdoc"


def setup(app):
    # to hide/show the prompt in code examples:
    app.add_js_file("js/copybutton.js")


# sphinx-gallery configuration
sphinx_gallery_conf = {
    'doc_module': 'abess',
    # path to your example scripts
    'examples_dirs': ['./Tutorial'],
    # path to where to save gallery generated output
    'gallery_dirs': ['auto_gallery'],
    # specify that examples should be ordered according to filename
    'within_subsection_order': sphinx_gallery.sorting.FileNameSortKey,
    # directory where function granular galleries are stored
    # 'backreferences_dir': 'gen_modules/backreferences',
    # Modules for which function level galleries are created.  In
    # this case sphinx_gallery and numpy in a tuple of strings.
    # 'doc_module': ('SampleModule'),
    'reference_url': {
        'abess': None,
    }
    # ,
    # 'filename_pattern': '/plot_',
    # 'ignore_pattern': r'__init__\.py',
    # 'ignore_pattern': r'noinclude\.py'

}

# configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/', None),
    "sklearn": ("https://scikit-learn.org/dev/", None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None)
}
