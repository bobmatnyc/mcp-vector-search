"""Templates for visualization HTML generation.

This package contains modular template components for generating
the D3.js visualization HTML page.
"""

from .base import generate_html_template, inject_data
from .scripts import get_all_scripts
from .styles import get_all_styles

__all__ = [
    "generate_html_template",
    "inject_data",
    "get_all_scripts",
    "get_all_styles",
]
