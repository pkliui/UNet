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
import os
import sys
sys.path.insert(0, os.path.abspath('../UNet'))


# -- Project information -----------------------------------------------------

project = 'unet-segmentation'
copyright = 'Pavel Kliuiev'
author = 'Pavel Kliuiev'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------


# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
import furo

# simply add the extension to your list of extensions
extensions = ['myst_parser', "sphinx_togglebutton"]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


# The master toctree document.
master_doc = 'index'

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'WordCountdoc'
