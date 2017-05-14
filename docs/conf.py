"""Sphinx configuration."""

import os
import sphinx_rtd_theme
import sys


needs_sphinx = "1.5.4"


# Metadata
project = "Controlboros"
copyright = "2017 Mikhail Pak"
author = "Mikhail Pak"

version = "0.1"
release = "0.1.0"


language = None

sys.path.insert(0, os.path.abspath("../controlboros"))


# Configure appearance
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

pygments_style = "sphinx"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    ]

templates_path = []

source_suffix = [".rst"]

master_doc = "index"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

todo_include_todos = False

autoclass_content = "both"

html_static_path = ["_static"]

htmlhelp_basename = "controlborosdoc"
