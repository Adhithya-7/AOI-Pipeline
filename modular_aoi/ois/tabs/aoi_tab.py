"""
aoi_tab.py — Offline AOI inspection tab (OfflineAOITab).
"""
import glob
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime

if __package__ in (None, ""):
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    __package__ = "ois.tabs"

from PySide6.QtWidgets import (
    QWidget, QLabel, QFrame, QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QLineEdit, QFileDialog, QScrollArea, QSizePolicy, QListWidget,
    QListWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMenu, QProgressBar, QSlider, QApplication,
    QDialog, QDialogButtonBox, QPlainTextEdit, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, Property, QSize
from PySide6.QtGui import (
    QIcon, QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor,
    QTextCursor, QPalette, QKeySequence, QShortcut
)

from ..theme import (
    BG_DEEP, BG_PANEL, BG_CARD, BG_CARD2, BG_BORDER, BG_BORDER2,
    CYAN, CYAN_DIM, CYAN_GLOW, GREEN, GREEN_DIM, GREEN_GLOW,
    AMBER, AMBER_DIM, AMBER_GLOW, RED, RED_DIM, RED_GLOW,
    PURPLE, TEXT_PRI, TEXT_SEC, TEXT_DIM, SIDEBAR_W,
    _F_MONO_8, _F_MONO_9, _F_MONO_10, _F_MONO_11, _F_MONO_12,
    _pen_dash, _card_style, make_card, make_sep, sec_lbl, set_fg
)
from ..utils import (
    HAS_CV2, HAS_NP, HAS_YOLO, DATA_ROOT,
    ProjectConfig, MASTER_CLASS_LIST, DEFAULT_AUG,
    build_golden_dataset, OG_MODEL, calc_iou, match_detections
)
from ..aoi_engine import OfflineAOIThread, _DEFECT_COLOURS
from ..filters import apply_filters, FILTER_REGISTRY, FilterNode, run_roi
from ..widgets import (
    FastLog, ResultStrip, ROICanvas, ROIZone, StatCard, ToastManager,
    ZoomableImageView
)

if HAS_CV2:
    import cv2
if HAS_NP:
    import numpy as np
if HAS_YOLO:
    from ultralytics import YOLO as _YOLO


class OfflineAOITab(QWidget):
    """
    Offline inspection mode:
      Upload a golden board image → upload test images → run AOI → view results.
    Uses diff-based detection + SAHI-aware YOLO inference.
    """
    def __init__(self, cfg):
        super().__init__()
        self._cfg         = cfg
        self._golden_path = ""
        self._test_paths  = []
        self._results     = []
        self._cur_result  = 0
        self._worker      = None
        self._golden_rendered = False  # render annotated golden once per run
        # Pipeline state (received from LogicTab)
        self._pipe_filters  = []
        self._pipe_rois     = []
        self._pipe_model    = None
        self._pipe_active   = False
        self._db = None   # set by MainWindow after creation
        self._build()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12,12,12,12); root.setSpacing(12)

        # ── LEFT PANEL: fixed-width controls card ────────────────────────
        lf = make_card("aoi_left"); ll = QVBoxLayout(lf)
        ll.setContentsMargins(12,12,12,12); ll.setSpacing(8)
        lf.setFixedWidth(360)
        lf.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        ll.addWidget(sec_lbl("GOLDEN BOARD"))
        gr = QHBoxLayout()
        self._golden_lbl = QLabel("No golden loaded")
        self._golden_lbl.setObjectName("aoi_glbl")
        self._golden_lbl.setStyleSheet(f"#aoi_glbl{{color:{TEXT_SEC};font-size:10px;border:none;}}")
        self._golden_lbl.setWordWrap(True)
        b_load_g = QPushButton("Load Golden"); b_load_g.setFixedHeight(28)
        b_load_g.clicked.connect(self._load_golden)
        gr.addWidget(self._golden_lbl,1); gr.addWidget(b_load_g)
        ll.addLayout(gr)

        ll.addWidget(make_sep()); ll.addWidget(sec_lbl("TEST BOARDS"))
        tr = QHBoxLayout()
        b_add_f = QPushButton("Add Files"); b_add_f.setFixedHeight(28)
        b_add_f.clicked.connect(self._add_files)
        b_add_d = QPushButton("Add Folder"); b_add_d.setFixedHeight(28)
        b_add_d.clicked.connect(self._add_folder)
        b_clr_t = QPushButton("Clear"); b_clr_t.setFixedHeight(28)
        b_clr_t.clicked.connect(self._clear_tests)
        b_rem = QPushButton("Remove"); b_rem.setFixedHeight(28)
        b_rem.setToolTip("Remove selected board from list (Del key)")
        b_rem.clicked.connect(self._remove_current_board)
        tr.addWidget(b_add_f); tr.addWidget(b_add_d); tr.addWidget(b_rem); tr.addWidget(b_clr_t)
        ll.addLayout(tr)

        self._test_list = QListWidget(); self._test_list.setObjectName("aoi_tlist")
        self._test_list.setStyleSheet(f"#aoi_tlist{{background:{BG_PANEL};border:1px solid {BG_BORDER};border-radius:6px;}}")
        self._test_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._test_list.currentRowChanged.connect(self._on_list_sel)
        self._test_list.setMinimumHeight(80)
        self._test_list.setMaximumHeight(200)
        ll.addWidget(self._test_list)

        self._test_count_lbl = QLabel("0 boards")
        self._test_count_lbl.setStyleSheet(f"color:{TEXT_SEC};font-size:10px;border:none;")
        ll.addWidget(self._test_count_lbl)

        # ── SETTINGS section removed — model, confidence, calibration, and SAHI
        # are now managed entirely in the dedicated Settings tab (SettingsDialog).
        # Hidden widget instances are kept below so that _run(), on_project_changed(),
        # and the MainWindow sync callbacks can continue to read/write them without
        # needing extra guards throughout the code.
        _mp = self._cfg.get("model_path") or ""
        self._model_le = QLineEdit(); self._model_le.setText(_mp); self._model_le.setReadOnly(True)
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.99)
        self._conf_spin.setValue(float(self._cfg.get("aoi_conf", 0.25)))
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.valueChanged.connect(lambda v: self._cfg.set('aoi_conf', v))
        self._cal_spin = QSpinBox()
        self._cal_spin.setRange(4, 20)
        self._cal_spin.setValue(int(self._cfg.get("aoi_cal_runs", 8)))
        self._cal_spin.valueChanged.connect(lambda v: self._cfg.set('aoi_cal_runs', v))
        self._sahi_cb = QCheckBox()
        self._sahi_cb.setChecked(False)
        self._sahi_cb.toggled.connect(lambda v: self._cfg.set('aoi_use_sahi', v))

        ll.addWidget(make_sep()); ll.addWidget(sec_lbl("FILTER"))
        pipe_row = QHBoxLayout()
        self._pipe_status_lbl = QLabel("No pipeline loaded")
        self._pipe_status_lbl.setObjectName("aoi_pipe_lbl")
        self._pipe_status_lbl.setStyleSheet(f"#aoi_pipe_lbl{{color:{TEXT_SEC};font-size:10px;font-family:Consolas;border:none;}}")
        b_use_pipe = QPushButton("Use Filter"); b_use_pipe.setFixedHeight(26)
        b_use_pipe.setToolTip("Applies the filter from the FILTERS tab to each test board before inspection")
        b_use_pipe.clicked.connect(self._use_logic_pipeline)
        b_clr_pipe = QPushButton("Clear"); b_clr_pipe.setFixedHeight(26)
        b_clr_pipe.clicked.connect(self._clear_pipeline)
        pipe_row.addWidget(self._pipe_status_lbl,1); pipe_row.addWidget(b_use_pipe); pipe_row.addWidget(b_clr_pipe)
        ll.addLayout(pipe_row)

        ll.addWidget(make_sep())

        # Run button
        self._btn_run = QPushButton("▶  RUN OFFLINE INSPECTION")
        self._btn_run.setObjectName("aoi_run"); self._btn_run.setFixedHeight(46)
        self._btn_run.setStyleSheet(
            f"#aoi_run{{background:{CYAN};"
            f"color:#FFFFFF;border:none;border-radius:8px;"
            f"font-size:13px;font-weight:bold;font-family:Consolas;letter-spacing:2px;}}"
            f"#aoi_run:hover{{background:#1347A0;color:#FFFFFF;}}"
            f"#aoi_run:disabled{{background:{BG_CARD};color:{TEXT_DIM};border-color:{BG_BORDER};}}"
        )
        self._btn_run.clicked.connect(self._run)
        ll.addWidget(self._btn_run)

        self._prog = QProgressBar(); self._prog.setFixedHeight(14); self._prog.setValue(0)
        self._prog.setTextVisible(True)
        self._prog.setStyleSheet(
            f"QProgressBar{{background:{BG_PANEL};border:1px solid {BG_BORDER};border-radius:6px;"
            f"text-align:center;font-family:Consolas;font-size:9px;color:{TEXT_DIM};}}"
            f"QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {CYAN_DIM},stop:1 {CYAN});border-radius:5px;}}"
        )
        ll.addWidget(self._prog)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setObjectName("aoi_stat")
        self._status_lbl.setStyleSheet(f"#aoi_stat{{color:{TEXT_SEC};font-size:10px;font-family:Consolas;border:none;}}")
        self._status_lbl.setWordWrap(True)
        ll.addWidget(self._status_lbl)

        # Mini summary — compact inline cards that don't clip
        ll.addWidget(make_sep()); ll.addWidget(sec_lbl("SUMMARY"))
        sg = QHBoxLayout(); sg.setSpacing(6)
        for attr, label, col in [("_sc_total","TOTAL",CYAN),("_sc_pass","PASS",GREEN),("_sc_fail","FAIL",RED)]:
            cf = QFrame(); cf.setObjectName(f"sc_{attr}")
            cf.setStyleSheet(
                f"QFrame#sc_{attr}{{background:{BG_CARD2};border:1px solid {BG_BORDER2};"
                f"border-top:2px solid {col};border-radius:6px;}}"
            )
            cl = QVBoxLayout(cf); cl.setContentsMargins(8,6,8,6); cl.setSpacing(1)
            val_lbl = QLabel("0"); val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_lbl.setStyleSheet(f"color:{col};font-size:20px;font-weight:bold;font-family:Consolas;border:none;")
            lbl_lbl = QLabel(label); lbl_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_lbl.setStyleSheet(f"color:{TEXT_SEC};font-size:8px;letter-spacing:1px;font-family:Consolas;border:none;")
            cl.addWidget(val_lbl); cl.addWidget(lbl_lbl)
            setattr(self, attr, val_lbl)   # store label directly
            sg.addWidget(cf)
        ll.addLayout(sg)

        exp_row = QHBoxLayout()
        b_export = QPushButton("Export JSON"); b_export.setFixedHeight(28)
        b_export.clicked.connect(self._export_report)
        b_save_all = QPushButton("Save All Images"); b_save_all.setFixedHeight(28)
        b_save_all.setToolTip("Save all annotated overlay images to a folder")
        b_save_all.clicked.connect(self._save_all_images)
        exp_row.addWidget(b_export); exp_row.addWidget(b_save_all)
        ll.addLayout(exp_row)
        ll.addStretch()

        root.addWidget(lf)

        # ── RIGHT PANEL: result viewer ─────────────────────────────────────
        rf = make_card("aoi_right"); rl = QVBoxLayout(rf)
        rl.setContentsMargins(10,10,10,10); rl.setSpacing(8)

        # Nav bar
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("\u2190  Previous"); self._prev_btn.setFixedHeight(28)
        self._prev_btn.clicked.connect(self._show_prev)
        self._next_btn = QPushButton("Next  \u2192"); self._next_btn.setFixedHeight(28)
        self._next_btn.clicked.connect(self._show_next)
        self._board_lbl = QLabel("\u2014"); self._board_lbl.setObjectName("aoi_blbl")
        self._board_lbl.setStyleSheet(f"#aoi_blbl{{color:{CYAN};font-family:Consolas;font-size:12px;font-weight:bold;border:none;}}")
        self._verdict_lbl = QLabel(""); self._verdict_lbl.setObjectName("aoi_verd")
        self._verdict_lbl.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        self._verdict_lbl.setFixedHeight(26)
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._board_lbl,1)
        nav.addWidget(self._verdict_lbl)
        nav.addWidget(self._next_btn)
        rl.addLayout(nav)

        # ── Dual image panel: golden (fixed, annotated) | test (result overlay) ──
        dual_layout = QHBoxLayout(); dual_layout.setSpacing(6); dual_layout.setContentsMargins(0,0,0,0)

        # Golden side
        gv_wrap = QVBoxLayout(); gv_wrap.setSpacing(2); gv_wrap.setContentsMargins(0,0,0,0)
        self._golden_hdr = QLabel("◈  GOLDEN  (annotated)")
        self._golden_hdr.setStyleSheet(
            f"color:{GREEN};font-size:9px;font-family:Consolas;font-weight:bold;"
            f"letter-spacing:2px;border:none;border-bottom:1px solid {GREEN_DIM};"
            f"padding:0 2px 2px 2px;"
        )
        gv_wrap.addWidget(self._golden_hdr)
        self._golden_img_view = ZoomableImageView()
        gv_wrap.addWidget(self._golden_img_view, 1)
        dual_layout.addLayout(gv_wrap, 1)

        # Test side
        tv_wrap = QVBoxLayout(); tv_wrap.setSpacing(2); tv_wrap.setContentsMargins(0,0,0,0)
        self._test_hdr = QLabel("◈  TEST BOARD  (result overlay)")
        self._test_hdr.setStyleSheet(
            f"color:{CYAN};font-size:9px;font-family:Consolas;font-weight:bold;"
            f"letter-spacing:2px;border:none;border-bottom:1px solid {CYAN_DIM};"
            f"padding:0 2px 2px 2px;"
        )
        tv_wrap.addWidget(self._test_hdr)
        self._img_view = ZoomableImageView()
        tv_wrap.addWidget(self._img_view, 1)
        dual_layout.addLayout(tv_wrap, 1)

        # Wire zoom/pan sync between the two panels
        self._golden_img_view._peer = self._img_view
        self._img_view._peer = self._golden_img_view
        self._golden_img_view._idle_text = "GOLDEN BOARD\n\nLoad a golden image\nand run inspection"
        self._img_view._idle_text        = "TEST BOARD RESULT\n\nRun inspection\nto see overlay here"

        rl.addLayout(dual_layout, 6)   # stretch=6 → maximise board image area

        # Zoom controls bar
        zbar = QHBoxLayout(); zbar.setContentsMargins(0,0,0,0); zbar.setSpacing(4)
        b_fit = QPushButton("Fit"); b_fit.setFixedHeight(24)
        b_fit.setToolTip("Fit both golden and test views to their windows  (double-click image)")
        def _fit_both():
            self._img_view._fit(); self._golden_img_view._fit()
        b_fit.clicked.connect(_fit_both)
        b_save_img = QPushButton("Save Image"); b_save_img.setFixedHeight(24)
        b_save_img.clicked.connect(self._save_current_image)
        hint = QLabel("scroll=zoom  drag=pan  dbl-click=fit  \u2194 synced")
        hint.setStyleSheet(f"color:{TEXT_DIM};font-size:9px;font-family:Consolas;border:none;")
        zbar.addWidget(b_fit)
        zbar.addWidget(b_save_img); zbar.addStretch(); zbar.addWidget(hint)
        rl.addLayout(zbar)

        # ── Defect list with filter ────────────────────────────────────────
        dl_hdr = QHBoxLayout()
        dl_hdr.addWidget(sec_lbl("DEFECTS ON THIS BOARD"))
        self._defect_filter = QComboBox(); self._defect_filter.setFixedHeight(22)
        for _disp,_key in [('All defects','All'),('Missing','missing'),
                           ('Wrong Part','wrong_component'),
                           ('Misaligned','misaligned'),
                           ('Wrong Polarity','wrong_polarity')]:
            self._defect_filter.addItem(_disp, _key)
        self._defect_filter.currentIndexChanged.connect(lambda _: self._apply_defect_filter())
        dl_hdr.addWidget(self._defect_filter)
        rl.addLayout(dl_hdr)
        self._defect_list = QListWidget(); self._defect_list.setObjectName("aoi_dlist")
        self._defect_list.setStyleSheet(
            f"#aoi_dlist{{background:{BG_PANEL};border:1px solid {BG_BORDER};border-radius:6px;}}"
            f"QListWidget::item{{padding:6px 10px;border-radius:4px;font-family:Consolas;font-size:12px;margin:1px 0;}}"
            f"QListWidget::item:hover:!selected{{background:{BG_CARD2};}}"
            f"QListWidget::item:selected{{background:{CYAN_DIM};}}"
        )
        self._defect_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._defect_list.setMinimumHeight(60)
        self._defect_list.setMaximumHeight(160)
        self._defect_list.itemClicked.connect(self._on_defect_clicked)
        rl.addWidget(self._defect_list)

        # ── Log (collapsible height) ────────────────────────────────────────
        log_hdr = QHBoxLayout()
        log_hdr.addWidget(sec_lbl("INSPECTION LOG"))
        b_log_tog = QPushButton("▾"); b_log_tog.setFixedSize(20,18)
        b_log_tog.setCheckable(True); b_log_tog.setChecked(True)
        log_hdr.addWidget(b_log_tog); log_hdr.addStretch()
        rl.addLayout(log_hdr)
        self._log = FastLog(); self._log.setObjectName("aoi_log")
        self._log.setStyleSheet(
            f"#aoi_log{{background:{BG_PANEL};border:1px solid {BG_BORDER};border-radius:6px;"
            f"padding:6px;color:{GREEN};font-family:Consolas;font-size:10px;}}"
        )
        self._log.setMaximumHeight(60)
        b_log_tog.toggled.connect(lambda on: self._log.setVisible(on))
        rl.addWidget(self._log)

        root.addWidget(rf, 2)

        # ── Window-level shortcuts so arrow keys work regardless of focus ─────
        # Qt.ShortcutContext.WindowShortcut fires even when a child widget has focus
        sc_left  = QShortcut(QKeySequence(Qt.Key.Key_Left),  self)
        sc_right = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        sc_del   = QShortcut(QKeySequence(Qt.Key.Key_Delete),self)
        sc_save  = QShortcut(QKeySequence("Ctrl+S"),         self)
        sc_run   = QShortcut(QKeySequence(Qt.Key.Key_F5),    self)
        sc_left.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_right.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_del.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_save.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_run.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_left.activated.connect(self._show_prev)
        sc_right.activated.connect(self._show_next)
        sc_del.activated.connect(self._remove_current_board)
        sc_save.activated.connect(self._save_current_image)
        sc_run.activated.connect(self._run)

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _render_golden_annotated(self, golden_comps: list):
        """Draw component boxes on the golden image and load into the golden view (once).
        The same filter pipeline used during inspection is applied here so the displayed
        golden panel is visually identical to what the AOI engine actually compared against."""
        if not self._golden_path or not HAS_CV2 or not HAS_NP: return
        img = cv2.imread(self._golden_path)
        if img is None: return
        # Apply pipeline so the displayed golden matches what the engine actually saw
        if self._pipe_active and self._pipe_filters:
            img = apply_filters(img, self._pipe_filters)
        h, w = img.shape[:2]
        for gc in golden_comps:
            cx_px = int(gc['cx'] * w); cy_px = int(gc['cy'] * h)
            bw_px = max(4, int(gc['w'] * w)); bh_px = max(4, int(gc['h'] * h))
            x1, y1 = cx_px - bw_px // 2, cy_px - bh_px // 2
            x2, y2 = cx_px + bw_px // 2, cy_px + bh_px // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 120), 2)
            tag = gc.get('label', '')[:14]
            if tag:
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
                cv2.rectangle(img, (x1, max(0, y1-th-4)), (x1+tw+2, y1), (0,220,120), -1)
                cv2.putText(img, tag, (x1+1, y1-2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, (0, 0, 0), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self._golden_img_view.load_bytes(bytes(buf.tobytes()))
            n = len(golden_comps)
            self._golden_hdr.setText(f"◈  GOLDEN  ({n} component{'s' if n!=1 else ''}, annotated)")

    def _show_img_idle(self):
        self._img_view.clear()

    def _refresh_test_list(self):
        self._test_list.clear()
        for p in self._test_paths:
            self._test_list.addItem(os.path.basename(p))
        self._test_count_lbl.setText(f"{len(self._test_paths)} boards")

    def _update_summary_cards(self):
        total = len(self._results)
        pass_n = sum(1 for r in self._results if r.get("pass"))
        fail_n = total - pass_n
        self._sc_total.setText(str(total))
        self._sc_pass.setText(str(pass_n))
        self._sc_fail.setText(str(fail_n))

    def _show_result(self, idx: int):
        if not self._results or idx < 0 or idx >= len(self._results): return
        self._cur_result = idx
        r = self._results[idx]
        name_short = r["name"][:40]
        self._board_lbl.setText(f"{idx+1}/{len(self._results)}  {name_short}")

        # Verdict banner
        nd = r.get("n_defects", 0)
        if r.get("pass"):
            self._verdict_lbl.setText("  ✔  PASS  ")
            self._verdict_lbl.setStyleSheet(
                f"#aoi_verd{{color:#000;font-weight:bold;font-family:Consolas;font-size:12px;"
                f"background:{GREEN};border-radius:4px;padding:2px 8px;border:none;}}")
        else:
            dtype_counts = {}
            for _d in r.get("defects",[]):
                _t = _d["defect_type"]; dtype_counts[_t] = dtype_counts.get(_t,0)+1
            _DTS = {"missing":"MISS","wrong_component":"WRONG PART",
                    "misaligned":"MISALIGN","wrong_polarity":"POLARITY"}
            _parts = [f"{n}×{_DTS.get(t,t)}" for t,n in dtype_counts.items()]
            self._verdict_lbl.setText(f"  ✘  FAIL  {nd}  {'  '.join(_parts)}  ")
            self._verdict_lbl.setStyleSheet(
                f"#aoi_verd{{color:#fff;font-weight:bold;font-family:Consolas;font-size:12px;"
                f"background:{RED_DIM};border:1px solid {RED};border-radius:4px;padding:2px 8px;}}")

        # Load full-res image into zoomable viewer and update test header
        img_bytes = r.get("img_bytes", b"")
        if img_bytes:
            self._img_view.load_bytes(img_bytes)
        nd = r.get("n_defects", 0)
        if r.get("pass"):
            self._test_hdr.setText(f"◈  TEST BOARD  —  PASS")
            self._test_hdr.setStyleSheet(
                f"color:{GREEN};font-size:9px;font-family:Consolas;font-weight:bold;"
                f"letter-spacing:2px;border:none;border-bottom:1px solid {GREEN_DIM};"
                f"padding:0 2px 2px 2px;"
            )
        else:
            self._test_hdr.setText(f"◈  TEST BOARD  —  {nd} DEFECT{'S' if nd!=1 else ''}")
            self._test_hdr.setStyleSheet(
                f"color:{RED};font-size:9px;font-family:Consolas;font-weight:bold;"
                f"letter-spacing:2px;border:none;border-bottom:1px solid {RED_DIM};"
                f"padding:0 2px 2px 2px;"
            )

        # Defect list — populated via filter (preserves current filter selection)
        self._apply_defect_filter()


    def _apply_defect_filter(self, ftype: str = ""):
        """Show only defects matching the selected type filter, with thumbnail crops."""
        if not hasattr(self, "_defect_filter"): ftype = "All"
        elif not ftype: ftype = self._defect_filter.currentData() or "All"
        if not self._results or self._cur_result >= len(self._results): return
        r = self._results[self._cur_result]
        colours = {"missing":RED,"wrong_component":PURPLE,
                   "misaligned":AMBER,"wrong_polarity":CYAN,"roi_fail":CYAN}
        self._defect_list.clear()
        self._defect_list.setIconSize(QSize(48, 48))

        # Pre-decode result image for cropping thumbnails
        result_qimg = None
        if HAS_CV2:
            img_bytes = r.get("img_bytes", b"")
            if img_bytes:
                result_qimg = QImage.fromData(img_bytes)

        for d in r.get("defects",[]):
            dt = d["defect_type"]
            if ftype != "All" and dt != ftype: continue
            lbl = d.get("expected_label","?"); found = d.get("found_label","")
            det = d.get("details","")[:50]
            _DL = {"missing":("✘","MISSING"),"wrong_component":("⊗","WRONG PART"),
                   "misaligned":("↔","MISALIGNED"),"wrong_polarity":("↻","WRONG POLARITY"),
                   "roi_fail":("!","ROI FAIL")}
            icon, label_str = _DL.get(dt, ("!", dt.upper()))
            text = f" {icon}  {label_str:<16}  {lbl}"
            if found and found not in ("none",""): text += f"  →  {found}"
            if det: text += f"  ({det})"
            item = QListWidgetItem(text)
            item.setForeground(QColor(colours.get(dt, TEXT_SEC)))
            item.setData(Qt.ItemDataRole.UserRole, d)

            # ── Defect thumbnail icon ──────────────────────────────────────
            if result_qimg and not result_qimg.isNull():
                iw, ih = result_qimg.width(), result_qimg.height()
                cx_px = int(d.get("cx", 0.5) * iw)
                cy_px = int(d.get("cy", 0.5) * ih)
                # Use det bbox if available, otherwise estimate from golden comps
                det_w = det_h = 0
                for gc in r.get("golden_comps",[]):
                    if gc.get("id") == d.get("component_id",-1):
                        det_w = int(gc.get("w",0.06) * iw)
                        det_h = int(gc.get("h",0.06) * ih)
                        break
                pad = max(18, max(det_w, det_h) // 2 + 12)
                x1 = max(0, cx_px - pad); y1 = max(0, cy_px - pad)
                x2 = min(iw, cx_px + pad); y2 = min(ih, cy_px + pad)
                if x2 > x1 and y2 > y1:
                    crop = result_qimg.copy(x1, y1, x2-x1, y2-y1)
                    px = QPixmap.fromImage(crop).scaled(
                        48, 48,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    # Draw a coloured border on the thumbnail matching defect type
                    painter = QPainter(px)
                    painter.setPen(QPen(QColor(colours.get(dt, TEXT_SEC)), 2))
                    painter.drawRect(1, 1, px.width()-2, px.height()-2)
                    painter.end()
                    item.setIcon(QIcon(px))

            # Full tooltip
            tip = f"{label_str}: {lbl}"
            if found and found not in ("none",""): tip += f"\nFound instead: {found}"
            if det: tip += f"\nDetail: {det}"
            for gc in r.get("golden_comps",[]):
                if gc.get("id") == d.get("component_id",-1):
                    tip += f"\nLocation: cx={gc['cx']:.3f}  cy={gc['cy']:.3f}"
                    break
            item.setToolTip(tip)
            self._defect_list.addItem(item)

    def _on_defect_clicked(self, item):
        """Click a defect → zoom both viewers to that component region."""
        d = item.data(Qt.ItemDataRole.UserRole)
        if not d or not self._results: return
        r = self._results[self._cur_result]
        cid = d.get("component_id", -1)
        golden = r.get("golden_comps", [])
        for gc in golden:
            if gc.get("id") == cid:
                cx, cy, w_norm, h_norm = gc["cx"], gc["cy"], gc["w"], gc["h"]
                # Zoom both panels to the component bbox (sized to fill ~35% of view)
                self._img_view.zoom_to_component(cx, cy, w_norm, h_norm)
                self._golden_img_view.zoom_to_component(cx, cy, w_norm, h_norm)
                return
        # Fallback: just zoom in to centre
        self._img_view.zoom_in(); self._img_view.zoom_in()

    def _save_current_image(self):
        """Save the currently displayed annotated overlay image."""
        if not self._results: return
        r = self._results[self._cur_result]
        img_bytes = r.get("img_bytes", b"")
        if not img_bytes: return
        name = os.path.splitext(r["name"])[0] + "_aoi.jpg"
        p, _ = QFileDialog.getSaveFileName(self, "Save Image", name,
                                            "JPEG (*.jpg);;PNG (*.png)")
        if not p: return
        with open(p, "wb") as f: f.write(img_bytes)
        self._log.append(f"[Save] {p}")

    def on_project_changed(self, path):
        # ── Full reset first ─────────────────────────────────────────────
        self._results = []
        self._cur_result = 0
        self._test_paths = []
        self._golden_path = ""
        self._log.clear()
        if hasattr(self, "_defect_list"):   self._defect_list.clear()
        if hasattr(self, "_test_list"):     self._test_list.clear()
        if hasattr(self, "_img_view"):      self._img_view.clear()
        if hasattr(self, "_golden_img_view"): self._golden_img_view.clear()
        if hasattr(self, "_golden_lbl"):
            self._golden_lbl.setText("No golden image")
            self._golden_lbl.setStyleSheet(f"#aoi_glbl{{color:{TEXT_DIM};font-size:10px;border:none;}}")
        if hasattr(self, "_prog"):          self._prog.setValue(0)
        # ── Restore model path ────────────────────────────────────────────
        mp = self._cfg.get("model_path") or ""
        if mp and os.path.exists(mp):
            self._model_le.setText(mp)
            self._model_le.setStyleSheet(
                f"color:{GREEN};font-family:Consolas;font-size:10px;"
            )
        elif mp:
            self._model_le.setText(mp)
            self._model_le.setStyleSheet(
                f"color:{RED};font-family:Consolas;font-size:10px;"
            )
        # Restore AOI settings
        conf = self._cfg.get('aoi_conf')
        if conf: self._conf_spin.setValue(float(conf))
        cal = self._cfg.get('aoi_cal_runs')
        if cal: self._cal_spin.setValue(int(cal))
        # aoi_use_sahi is intentionally NOT restored from config.
        # SAHI inflates golden component counts (~146 vs ~80 from standard YOLO)
        # which corrupts the entire AOI comparison. User must opt in each run.
        self._cfg.set('aoi_use_sahi', False)
        self._sahi_cb.setChecked(False)
        # Restore last golden
        lg = self._cfg.get('aoi_last_golden') or ''
        if lg and os.path.exists(lg) and not self._golden_path:
            self._golden_path = lg
            self._golden_lbl.setText(os.path.basename(lg))
            self._golden_lbl.setStyleSheet(f'#aoi_glbl{{color:{GREEN};font-size:10px;border:none;}}')
            self._log.append(f'[Restored] Golden: {lg}')
        # Restore last test folder
        ltd = self._cfg.get('aoi_last_test_dir') or ''
        if ltd and os.path.isdir(ltd) and not self._test_paths:
            exts = {'.jpg','.jpeg','.png','.bmp'}
            new = [os.path.join(ltd,f) for f in sorted(os.listdir(ltd))
                   if os.path.splitext(f)[1].lower() in exts]
            if new:
                self._test_paths = new; self._refresh_test_list()
                self._log.append(f'[Restored] {len(new)} boards from {os.path.basename(ltd)}')

    # ── Slots ────────────────────────────────────────────────────────────────

    def deploy_pipeline(self, filters, rois, model_path=""):
        """Receive pipeline from LogicTab (same signal as RunTab)."""
        self._pipe_filters = list(filters)
        self._pipe_rois    = list(rois)
        self._pipe_model   = None
        self._pipe_active  = bool(filters or rois)
        if model_path and os.path.exists(model_path) and HAS_YOLO:
            try: self._pipe_model = _YOLO(model_path)
            except: pass
        self._refresh_pipe_status()

    def _use_logic_pipeline(self):
        """Copy the deployed pipeline from MainWindow into this tab."""
        # Find MainWindow and grab its logic tab state
        mw = self.window()
        if hasattr(mw, '_logic'):
            lt = mw._logic
            self._pipe_filters = list(lt._filters)
            self._pipe_rois    = list(lt._rois)
            self._pipe_active  = bool(self._pipe_filters or self._pipe_rois)
            self._pipe_model   = None
            mp = mw._cfg.get("model_path") or ""
            if mp and os.path.exists(mp) and HAS_YOLO:
                try: self._pipe_model = _YOLO(mp)
                except: pass
            self._refresh_pipe_status()
        else:
            self._pipe_status_lbl.setText("Logic tab not found")

    def _clear_pipeline(self):
        self._pipe_filters=[]; self._pipe_rois=[]; self._pipe_model=None; self._pipe_active=False
        self._refresh_pipe_status()

    def _refresh_pipe_status(self):
        if not self._pipe_active:
            self._pipe_status_lbl.setText("No pipeline")
            self._pipe_status_lbl.setStyleSheet(f"#aoi_pipe_lbl{{color:{TEXT_SEC};font-size:10px;font-family:Consolas;border:none;}}")
        else:
            nf=len(self._pipe_filters); nr=len(self._pipe_rois)
            self._pipe_status_lbl.setText(f"Active: {nf} filters | {nr} ROI zones")
            self._pipe_status_lbl.setStyleSheet(f"#aoi_pipe_lbl{{color:{CYAN};font-size:10px;font-family:Consolas;border:none;}}")


    def _remove_current_board(self):
        """Remove current board from test list (e.g. bad scan)."""
        if not self._test_paths or self._cur_result >= len(self._test_paths): return
        idx = self._cur_result
        self._test_paths.pop(idx)
        self._refresh_test_list()
        if self._results and idx < len(self._results):
            self._results.pop(idx)
        self._cur_result = max(0, idx-1)
        self._update_summary_cards()
        if self._results: self._show_result(self._cur_result)

    def _load_golden(self):
        start = self._cfg.get('aoi_last_golden') or self._cfg.get('aoi_last_test_dir') or ''
        p,_ = QFileDialog.getOpenFileName(self,"Load Golden Board",os.path.dirname(start),"Images (*.jpg *.png *.jpeg *.bmp)")
        if not p: return
        self._golden_path = p
        self._cfg.set('aoi_last_golden', p)
        self._golden_lbl.setText(os.path.basename(p))
        self._golden_lbl.setStyleSheet(f"#aoi_glbl{{color:{GREEN};font-size:10px;border:none;}}")
        self._log.append(f"Golden: {p}")

    def _add_files(self):
        files,_ = QFileDialog.getOpenFileNames(self,"Add Test Boards","","Images (*.jpg *.png *.jpeg *.bmp)")
        self._test_paths += [f for f in files if f not in self._test_paths]
        self._refresh_test_list()

    def _add_folder(self):
        start = self._cfg.get('aoi_last_test_dir') or os.path.dirname(self._golden_path) or ''
        d = QFileDialog.getExistingDirectory(self,"Add Test Folder", start)
        if not d: return
        self._cfg.set('aoi_last_test_dir', d)
        exts = {".jpg",".jpeg",".png",".bmp"}
        new = [os.path.join(d,f) for f in sorted(os.listdir(d))
               if os.path.splitext(f)[1].lower() in exts]
        self._test_paths += [f for f in new if f not in self._test_paths]
        self._refresh_test_list()
        self._log.append(f"Added {len(new)} images from {os.path.basename(d)}")

    def _clear_tests(self):
        self._test_paths=[]; self._refresh_test_list()
        self._results=[]; self._update_summary_cards()
        self._defect_list.clear(); self._img_view.clear()
        self._golden_img_view.clear()
        self._golden_rendered = False

    def _on_list_sel(self, idx):
        if 0<=idx<len(self._results):
            self._show_result(idx)

    def _show_prev(self):
        if self._results and self._cur_result > 0:
            self._show_result(self._cur_result-1)
            self._test_list.setCurrentRow(self._cur_result)

    def _show_next(self):
        if self._results and self._cur_result < len(self._results)-1:
            self._show_result(self._cur_result+1)
            self._test_list.setCurrentRow(self._cur_result)

    def _run(self):
        if not self._golden_path:
            QMessageBox.warning(self,"No Golden","Load a golden board image first."); return
        if not self._test_paths:
            QMessageBox.warning(self,"No Tests","Add test board images first."); return
        mp = self._model_le.text().strip()
        if not mp or not os.path.exists(mp):
            QMessageBox.warning(self,"No Model","Set a valid model path in Settings."); return
        if self._worker and self._worker.isRunning():
            self._worker.abort(); return

        # Reset
        self._results=[]; self._update_summary_cards()
        self._defect_list.clear(); self._img_view.clear()
        self._golden_img_view.clear()
        self._golden_rendered = False
        self._log.clear(); self._prog.setValue(0)
        self._btn_run.setText("■  STOP")
        self._status_lbl.setText("Running…")
        self._status_lbl.setStyleSheet(f"#aoi_stat{{color:{CYAN};font-size:10px;font-family:Consolas;border:none;}}")

        self._worker = OfflineAOIThread(
            golden_path = self._golden_path,
            test_paths  = self._test_paths,
            model_path  = mp,
            conf        = self._conf_spin.value(),
            use_sahi    = self._sahi_cb.isChecked(),
            cal_runs    = self._cal_spin.value(),
            filters     = self._pipe_filters if self._pipe_active else [],
            rois        = self._pipe_rois    if self._pipe_active else [],
            pipe_model  = self._pipe_model,
        )
        self._worker.progress.connect(self._on_progress, Qt.ConnectionType.QueuedConnection)
        self._worker.board_done.connect(self._on_board_done, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.ConnectionType.QueuedConnection)
        self._worker.start()

    @Slot(int, str)
    def _on_progress(self, pct: int, msg: str):
        self._log.append(msg)
        if pct >= 0:   # -1 = log-only (debug detail, don't update bar/status)
            self._prog.setValue(pct)
            self._status_lbl.setText(msg)

    @Slot(str, bytes, int, int, list, list)
    def _on_board_done(self, name: str, img_bytes: bytes, iw: int, ih: int,
                        defects: list, golden_comps: list):
        r = {
            "name":         name,
            "img_bytes":    img_bytes,
            "iw":           iw,
            "ih":           ih,
            "defects":      defects,
            "n_defects":    len(defects),
            "pass":         len(defects)==0,
            "golden_comps": golden_comps,
        }
        self._results.append(r)
        # Render annotated golden board once per run (uses first board's component list)
        if not self._golden_rendered and golden_comps:
            self._render_golden_annotated(golden_comps)
            self._golden_rendered = True
        # Update list item colour and add defect count badge
        idx = len(self._results)-1
        item = self._test_list.item(idx)
        if item:
            nd = r["n_defects"]
            base = os.path.splitext(r["name"])[0]
            if r["pass"]:
                item.setText(f"✔  {base}")
                item.setForeground(QColor(GREEN))
            else:
                item.setText(f"✘  {base}  [{nd}]")
                item.setForeground(QColor(RED))
        self._update_summary_cards()
        # Log to history DB
        if self._db and self._cfg.get("log_history", True):
            proj = os.path.basename(self._cfg._path or "")
            mdl  = os.path.basename(self._cfg.get("model_path",""))
            nd   = r["n_defects"]
            dtypes = ",".join(sorted({d["defect_type"] for d in r.get("defects",[])}))
            _db_row_id = self._db.log_result(project=proj, mode="AOI", passed=r["pass"],
                                n_defects=nd, defect_types=dtypes[:120], model=mdl)
        else:
            _db_row_id = None
        # Save annotated side-by-side history image (respects setting)
        if self._cfg.get("save_annot_history", True) and img_bytes and HAS_CV2 and HAS_NP:
            self._save_aoi_history_image(r, db_row_id=_db_row_id)
        # Auto-show latest
        self._show_result(idx)
        self._test_list.setCurrentRow(idx)

    @Slot(dict)
    def _on_finished(self, summary: dict):
        self._btn_run.setText("▶  RUN OFFLINE INSPECTION")
        if not summary:
            self._status_lbl.setText("Failed — check model/image paths")
            self._status_lbl.setStyleSheet(f"#aoi_stat{{color:{RED};font-size:10px;font-family:Consolas;border:none;}}")
            return
        p, f, t = summary.get("pass",0), summary.get("fail",0), summary.get("total",0)
        rate = f"{p/t*100:.1f}%" if t else "N/A"
        msg = f"Done: {p}/{t} PASS ({rate})  |  {f} boards need attention"
        self._status_lbl.setText(msg)
        col = GREEN if f==0 else (AMBER if p/max(t,1)>=0.8 else RED)
        self._status_lbl.setStyleSheet(f"#aoi_stat{{color:{col};font-size:10px;font-family:Consolas;border:none;}}")
        self._log.append(f"=== {msg} ===")
        dc = summary.get("defect_counts",{})
        for dt,n in sorted(dc.items()):
            self._log.append(f"  {dt:<22} × {n}")
        ToastManager.show(self.window(), msg, "success" if f==0 else "warning")


    def _save_aoi_history_image(self, r: dict, db_row_id=None):
        """Build + save a side-by-side annotated history image for an AOI result (background thread)."""
        cfg_path    = self._cfg._path or ""
        golden_path = self._golden_path or ""
        # Snapshot pipeline state now (before the background thread runs)
        snap_filters = list(self._pipe_filters) if self._pipe_active and self._pipe_filters else []
        _thr = threading   # module-level — no frozen-module lookup

        def _work():
            try:
                img_bytes = r.get("img_bytes", b"")
                if not img_bytes: return
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                right_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if right_bgr is None: return
                h, w = right_bgr.shape[:2]

                # LEFT panel — golden board with component outlines
                # Apply the same pipeline so history thumbnails are consistent
                if golden_path and os.path.exists(golden_path):
                    left_bgr = cv2.imread(golden_path)
                    if left_bgr is None: left_bgr = right_bgr.copy()
                    else:
                        if snap_filters:
                            left_bgr = apply_filters(left_bgr, snap_filters)
                        left_bgr = cv2.resize(left_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    left_bgr = right_bgr.copy()

                for gc in r.get("golden_comps", []):
                    cx_px = int(gc["cx"]*w); cy_px = int(gc["cy"]*h)
                    bw_px = max(4, int(gc["w"]*w)); bh_px = max(4, int(gc["h"]*h))
                    x1,y1 = cx_px-bw_px//2, cy_px-bh_px//2
                    x2,y2 = cx_px+bw_px//2, cy_px+bh_px//2
                    cv2.rectangle(left_bgr,(x1,y1),(x2,y2),(50,200,80),1)
                    tag = gc.get("label","")[:12]
                    if tag:
                        cv2.putText(left_bgr,tag,(x1,min(y2+12,h-2)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.27,(120,200,120),1,cv2.LINE_AA)

                _DEFECT_COL = {
                    "missing":         (40, 40,230),
                    "misaligned":      (0,255,255),
                    "wrong_component": (200,40,200),
                    "wrong_polarity":  (40,165,255),
                }
                for d in r.get("defects", []):
                    dt  = d.get("defect_type",""); cid = d.get("component_id",-1)
                    col = _DEFECT_COL.get(dt,(80,80,80))
                    for gc in r.get("golden_comps",[]):
                        if gc.get("id") == cid:
                            cx_px = int(gc["cx"]*w); cy_px = int(gc["cy"]*h)
                            bw_px = max(6,int(gc["w"]*w)); bh_px = max(6,int(gc["h"]*h))
                            x1,y1=cx_px-bw_px//2,cy_px-bh_px//2
                            x2,y2=cx_px+bw_px//2,cy_px+bh_px//2
                            overlay = left_bgr.copy()
                            cv2.rectangle(overlay,(x1,y1),(x2,y2),col,-1)
                            cv2.addWeighted(overlay,0.35,left_bgr,0.65,0,left_bgr)
                            cv2.rectangle(left_bgr,(x1,y1),(x2,y2),col,2)
                            badge = dt[:4].upper(); ly=max(y1-4,14)
                            cv2.rectangle(left_bgr,(x1,ly-12),(x1+30,ly+2),col,-1)
                            cv2.putText(left_bgr,badge,(x1+1,ly),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1,cv2.LINE_AA)
                            break

                PAD=8; HEADER=30; FOOTER=24
                total_w=w*2+PAD; total_h=h+HEADER+FOOTER
                canvas = np.zeros((total_h,total_w,3),dtype=np.uint8)
                canvas[:,:] = (16,18,22)
                canvas[HEADER:HEADER+h, 0:w]           = left_bgr
                canvas[HEADER:HEADER+h, w+PAD:w*2+PAD] = right_bgr

                cv2.rectangle(canvas,(0,0),(total_w,HEADER-1),(22,26,34),-1)
                lbl_l = "GOLDEN REF" if (golden_path and os.path.exists(golden_path)) else "GOLDEN (fallback)"
                cv2.putText(canvas,lbl_l,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.46,(100,180,220),1,cv2.LINE_AA)
                cv2.putText(canvas,f"AOI RESULT  {r['name'][:30]}",(w+PAD+8,20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.46,(100,180,220),1,cv2.LINE_AA)
                is_pass = r.get("pass",True)
                res_col = (40,200,60) if is_pass else (50,50,220)
                bx = total_w-76
                cv2.rectangle(canvas,(bx-2,3),(total_w-4,HEADER-4),res_col,-1)
                cv2.putText(canvas," PASS" if is_pass else " FAIL",(bx,20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)
                cv2.line(canvas,(w+PAD//2,HEADER),(w+PAD//2,HEADER+h),(50,55,70),1)

                fy=HEADER+h+17
                cv2.rectangle(canvas,(0,HEADER+h),(total_w,total_h),(22,26,34),-1)
                nd=r.get("n_defects",0)
                if nd:
                    counts={}
                    for d in r.get("defects",[]):
                        k=d.get("defect_type","?"); counts[k]=counts.get(k,0)+1
                    summary="  ·  ".join(f"{k}x{v}" for k,v in counts.items())
                    cv2.putText(canvas,summary[:95],(8,fy),cv2.FONT_HERSHEY_SIMPLEX,0.35,(100,100,240),1,cv2.LINE_AA)
                    cv2.putText(canvas,f"{nd} defect(s)",(total_w-95,fy),cv2.FONT_HERSHEY_SIMPLEX,0.35,(100,100,240),1,cv2.LINE_AA)
                else:
                    cv2.putText(canvas,"All components present",(8,fy),cv2.FONT_HERSHEY_SIMPLEX,0.35,(50,185,50),1,cv2.LINE_AA)
                cv2.rectangle(canvas,(0,0),(total_w-1,total_h-1),res_col,2)

                hist_dir = os.path.join(cfg_path,"history") if cfg_path else os.path.join(os.getcwd(),"history")
                os.makedirs(hist_dir,exist_ok=True)
                ts_str   = datetime.now().strftime("%H%M%S")
                prefix   = "PASS" if is_pass else "FAIL"
                base_name= os.path.splitext(r["name"])[0][:20]
                fname    = f"AOI_{prefix}_{ts_str}_{base_name}.jpg"
                saved_path = os.path.join(hist_dir, fname)
                cv2.imwrite(saved_path, canvas)
                if db_row_id and self._db:
                    self._db.update_image_path(db_row_id, saved_path)
            except Exception as e:
                print(f"[AOI History] save error: {e}")

        _thr.Thread(target=_work,daemon=True).start()

    def _save_all_images(self):
        """Save all annotated overlay images to a user-chosen folder."""
        if not self._results:
            QMessageBox.information(self,"No Results","Run an inspection first."); return
        d = QFileDialog.getExistingDirectory(self,"Save All Overlay Images")
        if not d: return
        saved = 0
        for r in self._results:
            ib = r.get("img_bytes",b"")
            if not ib: continue
            name = os.path.splitext(r["name"])[0] + "_aoi.jpg"
            try:
                with open(os.path.join(d,name),"wb") as f: f.write(ib)
                saved += 1
            except Exception as e:
                self._log.append(f"Save failed {name}: {e}")
        QMessageBox.information(self,"Saved",f"Saved {saved} images to:\n{d}")

    def _export_report(self):
        if not self._results:
            QMessageBox.information(self,"No Results","Run an inspection first."); return
        p,_ = QFileDialog.getSaveFileName(self,"Export Report","aoi_report.json","JSON (*.json)")
        if not p: return
        report = []
        for r in self._results:
            # Build a position map from golden_comps stored in result
            gc_pos = {g["id"]:g for g in r.get("golden_comps",[])}
            # Enrich each defect with the golden component cx/cy
            defects_out = []
            for d in r.get("defects",[]):
                de = dict(d)
                gc = gc_pos.get(d.get("component_id",-1),{})
                de["cx"] = round(gc.get("cx",0),5)
                de["cy"] = round(gc.get("cy",0),5)
                de["label"] = gc.get("label",de.get("expected_label",""))
                defects_out.append(de)
            report.append({"name":r["name"],"pass":r["pass"],
                           "n_defects":r["n_defects"],"defects":defects_out})
        try:
            with open(p,"w") as f: json.dump(report, f, indent=2)
            QMessageBox.information(self,"Saved",f"Report saved:\n{p}")
        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# INFER TAB  (F6) — standalone YOLO/SAHI inference on images or camera frames
# ─────────────────────────────────────────────────────────────────────────────
