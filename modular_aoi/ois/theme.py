"""
theme.py — Visual constants and UI helper functions.

Zero business logic, zero imports from own code — pure Qt styling.
"""
import os
import sys

if __package__ in (None, ""):
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    __package__ = "ois"

from PySide6.QtWidgets import QFrame, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QPen, QPalette

# ── Colour constants ─────────────────────────────────────────────────────────
BG_DEEP="#DDE3EC"; BG_PANEL="#EAEDF2"; BG_CARD="#EAEDF2"; BG_CARD2="#E0E6EF"
BG_BORDER="#CAD2E6"; BG_BORDER2="#B6C0D4"
CYAN="#1A5FC8"; CYAN_DIM="#E4EDFC"; CYAN_GLOW="#1A5FC814"
GREEN="#15803D"; GREEN_DIM="#DCFCE7"; GREEN_GLOW="#15803D14"
AMBER="#B45309"; AMBER_DIM="#FEF3C7"; AMBER_GLOW="#B4530914"
RED="#C8192A"; RED_DIM="#FEE2E2"; RED_GLOW="#C8192A14"
PURPLE="#6D28D9"; TEXT_PRI="#1A2440"; TEXT_SEC="#4B5880"; TEXT_DIM="#9BAAC8"
SIDEBAR_W = 96

# ── Font constants ───────────────────────────────────────────────────────────
_F_MONO_9=QFont("Consolas",9,QFont.Weight.Bold); _F_MONO_8=QFont("Consolas",8,QFont.Weight.Bold)
_F_MONO_10=QFont("Consolas",10,QFont.Weight.Bold); _F_MONO_11=QFont("Consolas",11)
_F_MONO_12=QFont("Consolas",12,QFont.Weight.Bold); _PEN_DASH=None

# ── Helper functions ─────────────────────────────────────────────────────────
def _pen_dash():
    global _PEN_DASH
    if _PEN_DASH is None: _PEN_DASH=QPen(QColor(CYAN),1,Qt.PenStyle.DashLine)
    return _PEN_DASH

def _card_style(oid):
    return (f"#{oid}{{background:{BG_CARD};border:1px solid {BG_BORDER};border-radius:8px;}}")
def make_card(oid):
    f=QFrame(); f.setObjectName(oid); f.setStyleSheet(_card_style(oid)); return f
def make_sep():
    f=QFrame(); f.setFrameShape(QFrame.Shape.HLine); f.setObjectName("sep")
    f.setStyleSheet(f"#sep{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 transparent,stop:0.15 {BG_BORDER2},stop:0.85 {BG_BORDER2},stop:1 transparent);max-height:1px;border:none;margin:6px 0;}}"); return f
def sec_lbl(text):
    l=QLabel(text.upper()); l.setObjectName("seclbl")
    l.setStyleSheet(
        f"#seclbl{{color:{CYAN};font-size:10px;font-weight:bold;font-family:'Consolas';"
        f"letter-spacing:2px;border:none;border-left:2px solid {CYAN};"
        f"padding:3px 0 3px 8px;margin:4px 0;}}"
    ); return l
def set_fg(w,c): p=w.palette(); p.setColor(QPalette.ColorRole.WindowText,QColor(c)); w.setPalette(p)

# ── Write tiny SVG arrow files so QSS ::up-arrow / ::down-arrow have real images ─
# We write into DATA_ROOT/_arrows/ instead of next to __file__ so the path is
# writable both during development AND inside a frozen PyInstaller executable.
def _make_arrow_svgs(data_root):
    _dir = os.path.join(data_root, "_arrows")
    os.makedirs(_dir, exist_ok=True)
    _svgs = {
        "up":    f'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="5"><polygon points="4,0 8,5 0,5" fill="{TEXT_SEC}"/></svg>',
        "up_h":  f'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="5"><polygon points="4,0 8,5 0,5" fill="{CYAN}"/></svg>',
        "dn":    f'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="5"><polygon points="0,0 8,0 4,5" fill="{TEXT_SEC}"/></svg>',
        "dn_h":  f'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="5"><polygon points="0,0 8,0 4,5" fill="{CYAN}"/></svg>',
    }
    _paths = {}
    for _k, _s in _svgs.items():
        _p = os.path.join(_dir, f"arrow_{_k}.svg")
        with open(_p, "w") as _f: _f.write(_s)
        _paths[_k] = _p.replace("\\", "/")
    return _paths


def _build_global_qss(spin_arrows):
    """Build the global application stylesheet. Must be called after _make_arrow_svgs()."""
    return f"""
QWidget {{
    color: {TEXT_PRI};
    font-family: 'Segoe UI', 'SF Pro Display', 'SF Pro Text', 'Ubuntu', 'Noto Sans',
                 'Segoe UI Symbol', 'Segoe UI Emoji', 'Arial Unicode MS', sans-serif;
    font-size: 12px;
    background-color: {BG_DEEP};
}}
/* ── Labels: transparent background so they never bleed white on cards ── */
QLabel {{
    background: transparent;
}}
QToolTip {{
    background: {BG_PANEL};
    color: {TEXT_PRI};
    border: 1px solid {BG_BORDER2};
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 11px;
}}
/* ── Buttons ── */
QPushButton {{
    background: {BG_PANEL};
    color: {TEXT_PRI};
    border: 1px solid {BG_BORDER2};
    border-radius: 6px;
    padding: 7px 16px;
    font-family: 'Segoe UI Symbol', 'Segoe UI', 'Segoe UI Emoji', 'Apple Symbols', 'Arial Unicode MS', sans-serif;
    font-size: 11px;
    font-weight: 600;
}}
QPushButton:hover {{
    background: {CYAN_DIM};
    border-color: {CYAN};
    color: {CYAN};
}}
QPushButton:pressed {{
    background: #D0E2FA;
    border-color: {CYAN};
    color: {CYAN};
}}
QPushButton:disabled {{
    background: {BG_DEEP};
    color: {TEXT_DIM};
    border-color: {BG_BORDER};
}}
/* ── Inputs ── */
QLineEdit, QPlainTextEdit, QTextEdit {{
    background: {BG_PANEL};
    color: {TEXT_PRI};
    border: 1px solid {BG_BORDER2};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {CYAN_DIM};
}}
QLineEdit:focus, QPlainTextEdit:focus {{
    border-color: {CYAN};
    background: {BG_PANEL};
}}
/* ── Combo / Spin ── */
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {BG_PANEL};
    color: {TEXT_PRI};
    border: 1px solid {BG_BORDER2};
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 28px;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {CYAN};
    background: {BG_PANEL};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {CYAN};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    width: 0px;
    height: 0px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_SEC};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background: {BG_PANEL};
    color: {TEXT_PRI};
    border: 1px solid {BG_BORDER2};
    selection-background-color: {CYAN_DIM};
    outline: none;
    padding: 4px;
}}



QSpinBox::up-button, QDoubleSpinBox::up-button {{
    background: {BG_DEEP};
    border: none;
    border-left: 1px solid {BG_BORDER};
    border-bottom: 1px solid {BG_BORDER};
    width: 18px;
    border-top-right-radius: 5px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {BG_DEEP};
    border: none;
    border-left: 1px solid {BG_BORDER};
    width: 18px;
    border-bottom-right-radius: 5px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({spin_arrows['up']});
    width: 8px;
    height: 5px;
}}
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{
    image: url({spin_arrows['up_h']});
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({spin_arrows['dn']});
    width: 8px;
    height: 5px;
}}
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
    image: url({spin_arrows['dn_h']});
}}
/* ── Slider ── */
QSlider::groove:horizontal {{
    background: {BG_DEEP};
    height: 4px;
    border-radius: 2px;
    border: 1px solid {BG_BORDER2};
}}
QSlider::handle:horizontal {{
    background: {CYAN};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
    border: 2px solid {BG_PANEL};
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6EB0F0,stop:1 {CYAN});
    border-radius: 2px;
}}
/* ── Progress bars ── */
QProgressBar {{
    background: {BG_DEEP};
    border: 1px solid {BG_BORDER};
    border-radius: 5px;
    text-align: center;
    font-size: 10px;
    color: {TEXT_SEC};
    padding: 1px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6EB0F0,stop:1 {CYAN});
    border-radius: 4px;
}}
/* ── Checkboxes ── */
QCheckBox {{
    color: {TEXT_SEC};
    spacing: 8px;
    padding: 2px 0;
}}
QCheckBox:hover {{ color: {TEXT_PRI}; }}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BG_BORDER2};
    border-radius: 4px;
    background: {BG_PANEL};
}}
QCheckBox::indicator:checked {{
    background: {CYAN_DIM};
    border-color: {CYAN};
}}
QCheckBox::indicator:hover {{
    border-color: {CYAN};
}}
/* ── List widgets ── */
QListWidget {{
    background: {BG_PANEL};
    border: 1px solid {BG_BORDER};
    border-radius: 6px;
    outline: none;
    padding: 4px;
}}
QListWidget::item {{
    padding: 8px 12px;
    border-radius: 5px;
    color: {TEXT_SEC};
    font-size: 11px;
    margin: 1px 0;
}}
QListWidget::item:selected {{
    background: {CYAN_DIM};
    color: {CYAN};
}}
QListWidget::item:hover:!selected {{
    background: {BG_CARD2};
    color: {TEXT_PRI};
}}
/* ── Scrollbars ── */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    border-radius: 4px;
    margin: 2px 0;
}}
QScrollBar::handle:vertical {{
    background: {BG_BORDER2};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_DIM};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    border-radius: 4px;
    margin: 0 2px;
}}
QScrollBar::handle:horizontal {{
    background: {BG_BORDER2};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {TEXT_DIM}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
/* ── Dialogs ── */
QDialog {{
    background: {BG_PANEL};
    border: 1px solid {BG_BORDER2};
}}
QDialog QLabel {{ color: {TEXT_SEC}; }}
QMessageBox {{ background: {BG_PANEL}; }}
QDialogButtonBox QPushButton {{
    min-width: 84px;
    min-height: 32px;
}}
/* ── Status bar ── */
QStatusBar {{
    background: {BG_PANEL};
    color: {TEXT_SEC};
    border-top: 1px solid {BG_BORDER};
    font-family: 'Consolas';
    font-size: 10px;
    padding: 0 10px;
}}
QStatusBar::item {{ border: none; }}
/* ── Scroll areas ── */
QScrollArea {{ border: none; background: transparent; }}
/* ── Table ── */
QTableWidget {{
    background: {BG_PANEL};
    border: 1px solid {BG_BORDER};
    border-radius: 6px;
    gridline-color: {BG_BORDER};
    color: {TEXT_PRI};
}}
QHeaderView::section {{
    background: {BG_CARD2};
    color: {TEXT_SEC};
    border: none;
    border-right: 1px solid {BG_BORDER};
    border-bottom: 1px solid {BG_BORDER};
    padding: 5px 10px;
    font-weight: 600;
    font-size: 11px;
}}
QTableWidget::item:selected {{
    background: {CYAN_DIM};
    color: {CYAN};
}}
"""


# ── Initialise arrow SVGs and build the stylesheet ───────────────────────────
# DATA_ROOT is resolved lazily the first time it's needed. We import it here
# to avoid a circular dependency (utils imports nothing from theme).
def _init_theme():
    """Call once at startup after DATA_ROOT is known. Returns (SPIN_ARROWS, GLOBAL_QSS)."""
    from .utils import DATA_ROOT
    arrows = _make_arrow_svgs(DATA_ROOT)
    qss = _build_global_qss(arrows)
    return arrows, qss


# Module-level singletons — populated by _init_theme() called from main_window.main()
_SPIN_ARROWS: dict = {}
GLOBAL_QSS: str = ""


def init():
    """Initialise theme globals. Must be called once before any widget creation."""
    global _SPIN_ARROWS, GLOBAL_QSS
    _SPIN_ARROWS, GLOBAL_QSS = _init_theme()
