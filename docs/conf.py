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
from sphinx.application import Sphinx
from typing import Any, Dict
import sphinx_gallery
import pydata_sphinx_theme
import sphinx_gallery.sorting
import os
# import sys
import sys
# import abess
import matplotlib
# sys.path.append("../python/")
# sys.path.insert(0, os.path.join(os.path.abspath('..'), "python"))
# import SampleModule
# from sphinx_gallery.sorting import FileNameSortKey

# Use RTD Theme

# -- Project information -----------------------------------------------------

project = 'ABESS'
copyright = '2020, abess-team'
author = 'abess-team'
json_url = "https://abess.readthedocs.io/en/latest/_static/switcher.json"

# The full version, including alpha/beta/rc tags
version = os.environ.get("READTHEDOCS_VERSION")
rd = os.environ.get("READTHEDOCS")

if not rd:
    version = "dev"
    json_url = "_static/switcher.json"


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
    'sphinx_gallery.gen_gallery',
    "sphinx_design",
    "sphinx_copybutton",
    "numpydoc",
    "sphinx_favicon",
    "sphinx_togglebutton",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "matplotlib.sphinxext.plot_directive",
    # ,
    # 'sphinx_toggleprompt'
]

if not os.environ.get("READTHEDOCS"):
    extensions += ["sphinx_sitemap"]

    html_baseurl = os.environ.get("SITEMAP_URL_BASE", "http://127.0.0.1:8000/")
    sitemap_locales = [None]
    sitemap_url_scheme = "{link}"

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
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
source_suffix = ".rst"
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/README.rst']
master_doc = "index"
source_encoding = "utf-8"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

pygments_style = "sphinx"
smartquotes = False

html_theme_options = {
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/abess-team/abess",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/abess/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "CRAN",
            "url": "https://cran.rstudio.com/web/packages/abess/index.html",
            "icon": "fa-solid fa-r",
        },
    ],
    # alternative way to set twitter and github header icons
    # "github_url": "https://github.com/pydata/pydata-sphinx-theme",
    # "twitter_url": "https://twitter.com/PyData",
    "logo": {
        "text": "abess",
        "image_dark": ".image/apple-touch-icon.png",
        "alt_text": "abess",
    },
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navbar_align": "left",  # [left, content, right] For testing that the navbar items align properly
    "navbar_center": ["version-switcher", "navbar-nav"],
    # "announcement": """<div class="sidebar-message">
    # skscope is a optimization tool for Python. If you'd like to contribute, <a href="https://github.com/abess-team/skscope">check out our GitHub repository</a>
    # Your contributions are welcome!
    # </div>""",
    # "show_nav_level": 2,
    # "navbar_start": ["navbar-logo"],
    # "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    # "primary_sidebar_end": ["custom-template.html", "sidebar-ethical-ads.html"],
    # "article_footer_items": ["test.html", "test.html"],
    # "content_footer_items": ["test.html", "test.html"],
    # "footer_start": ["test.html", "test.html"],
    # "secondary_sidebar_items": ["page-toc.html"],  # Remove the source buttons
    "switcher": {
        "json_url": json_url,
        "version_match": version,
    },
    "navbar_end": ["search-field.html","theme-switcher","navbar-icon-links.html"],
    # "search_bar_position": "navbar",  # TODO: Deprecated - remove in future version
}

html_sidebars = {
    "auto_gallery/**": [
    #    "search-field",
        "sidebar-nav-bs",
    ],
    "Python-package/**": [
    #    "search-field",
        "sidebar-nav-bs",
    ],  # This ensures we test for custom sidebars
    "Contributing/**": [
    #    "search-field",
        "sidebar-nav-bs",
    ],
    # "examples/no-sidebar": [],  # Test what page looks like with no sidebar items
    # "examples/persistent-search-field": ["search-field"],
    # Blog sidebars
    # ref: https://ablog.readthedocs.io/manual/ablog-configuration-options/#blog-sidebars
    # "examples/blog/*": [
    #    "ablog/postcard.html",
    #    "ablog/recentposts.html",
    #    "ablog/tagcloud.html",
    #    "ablog/categories.html",
    #    "ablog/authors.html",
    #    "ablog/languages.html",
    #    "ablog/locations.html",
    #    "ablog/archives.html",
    # ],
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

html_css_files = ["custom.css"]
html_js_files = ["custom-icon.js"]

# configuration for intersphinx: refer to the Python standard library.
""" intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    "sklearn": ("https://scikit-learn.org/dev/", None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None)
} """

def setup_to_main(
    app: Sphinx, pagename: str, templatename: str, context, doctree
) -> None:
    """Add a function that jinja can access for returning an "edit this page" link pointing to `main`."""

    def to_main(link: str) -> str:
        """Transform "edit on github" links and make sure they always point to the main branch.

        Args:
            link: the link to the github edit interface

        Returns:
            the link to the tip of the main branch for the same file
        """
        links = link.split("/")
        idx = links.index("docs")
        return "/".join(links[: idx + 1]) + "/source/" + "/".join(links[idx + 1 :])

    context["to_main"] = to_main


def setup(app: Sphinx) -> Dict[str, Any]:
    """Add custom configuration to sphinx app.

    Args:
        app: the Sphinx application
    Returns:
        the 2 parralel parameters set to ``True``.
    """
    app.connect("html-page-context", setup_to_main)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

