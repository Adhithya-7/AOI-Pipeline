"""
infer_tab.py — Compatibility wrapper for the inference tab.

The tab implementation still lives in ois.widgets after the modular split.
Re-export it here so the rest of the package can keep importing
ois.tabs.infer_tab.InferTab.
"""

import os
import sys

if __package__ in (None, ""):
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    __package__ = "ois.tabs"

from ..widgets import InferTab, _InferThread

__all__ = ["InferTab", "_InferThread"]
