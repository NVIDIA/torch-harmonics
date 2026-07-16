# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as _pkg_version

# NOTE: we intentionally do NOT add the repository root to sys.path. This
# package ships compiled extension modules (torch_harmonics/*/_C) with no
# pure-Python fallback, so autodoc must import the *installed* build
# (``pip install ".[docs]"``) rather than the un-built source checkout, which
# would otherwise shadow it and fail to import.

# -- Project information -----------------------------------------------------

project = "torch-harmonics"
# %Y is substituted by Sphinx with the current year (honoring SOURCE_DATE_EPOCH
# for reproducible builds), so the range always ends at the build year.
copyright = "2022-%Y, The torch-harmonics Authors"
author = "The torch-harmonics Authors"

try:
    release = _pkg_version("torch_harmonics")
except Exception:
    release = "0.9.2a"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # parse the existing NumPy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_nb",  # render Markdown pages and .ipynb notebooks
    "sphinx_copybutton",  # copy button on code blocks
    "sphinxcontrib.bibtex",  # BibTeX bibliography support
]

# -- bibliography (sphinxcontrib-bibtex) -------------------------------------

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "label"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST (Markdown) options
myst_enable_extensions = [
    "amsmath",  # LaTeX-style math environments
    "dollarmath",  # $...$ and $$...$$ inline/display math
    "colon_fence",  # ::: fences
    "deflist",
]

# -- autosummary / autodoc ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# NOTE: torch_harmonics registers custom ops at import time and has no
# pure-Python fallback for its compiled helper modules, so the package must be
# *installed* (``pip install -e ".[docs]"``) before building the docs. A
# CPU-only build is sufficient — no CUDA toolkit required. We therefore import
# the real package (better signatures) rather than mocking it.


def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    if name == "extra_repr":
        return True
    return skip


def autodoc_process_docstring_handler(app, what, name, obj, options, lines):
    """Unwrap lru_cache wrappers so autodoc can see the original docstring."""
    import functools

    if hasattr(obj, "__wrapped__") and isinstance(obj, functools._lru_cache_wrapper):
        wrapped = obj.__wrapped__
        if wrapped.__doc__ and not lines:
            lines.extend(wrapped.__doc__.splitlines())


napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- myst-nb (notebooks) -----------------------------------------------------

# Do NOT execute notebooks at build time: they require CUDA / datasets and
# would fail on ReadTheDocs. Render the outputs stored in the .ipynb files.
nb_execution_mode = "off"

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "nvidia_sphinx_theme"  # official NVIDIA theme (extends pydata-sphinx-theme)
html_static_path = ["_static"]
html_css_files = ["custom.css"]  # project-specific tweaks on top of the theme
html_title = f"torch-harmonics {version}"

html_theme_options = {
    "github_url": "https://github.com/NVIDIA/torch-harmonics",
    "show_prev_next": False,
    "navigation_with_keys": True,
}


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member_handler)
    app.connect("autodoc-process-docstring", autodoc_process_docstring_handler)
