"""
MELSS Optical Inspection System — modular package.

This package is the modular split of the monolithic optical_inspection_system.py.
All functionality is preserved; only the file structure has changed.
"""
import sys, os

# ── PyInstaller / Windows multiprocessing guard ───────────────────────────────
# torch/ultralytics spawn worker processes; without this the exe re-runs the
# entire module on each spawn, causing infinite process storms on Windows.
if __name__ == "__main__" or getattr(sys, "frozen", False):
    import multiprocessing
    multiprocessing.freeze_support()
