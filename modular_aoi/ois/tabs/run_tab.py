"""
run_tab.py — Live inspection tab (RunTab).
"""
import glob
import json
import os
import platform
import shutil
import sys
import threading
import time
from collections import deque
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
from PySide6.QtCore import Qt, Signal, Slot, QTimer, Property
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor,
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
    build_golden_dataset, OG_MODEL, calc_iou, match_detections, safe_predict,
    load_optimized_yolo
)
from ..threads import CameraThread, InferenceThread
from ..aoi_engine import _AOIComp, _MIN_POS_TOL, _aoi_calibrate, _aoi_check_board
from ..filters import apply_filters, run_roi
from ..widgets import (
    AnnunciatorBanner, FastLog, LabelCanvas, ProjectDialog,
    ResultStrip, StatCard, ToastManager, VideoWidget
)

if HAS_YOLO:
    from ultralytics import YOLO as _YOLO

if HAS_CV2:
    import cv2
if HAS_NP:
    import numpy as np


class RunTab(QWidget):
    project_selected=Signal(str)
    flag_saved=Signal(str)   # emits golden_boards path so GoldenTab can reload
    def __init__(self,cfg):
        super().__init__(); self._cfg=cfg
        self._cam=self._ai=None; self._running=False
        self._golden=[]; self._persist={}
        self._pass=self._fail=self._total=0
        self._latest_dets=[]; self._last_missing=[]
        self._pipe_filters=[]; self._pipe_rois=[]; self._pipe_model=None; self._pipe_active=False
        self._last_yield_col=None
        # Direct-inference bypass state (for OpenVINO / InferenceThread-incompatible models)
        self._direct_infer_busy = False
        self._last_frame_bgr   = None
        # Golden reference image cache — avoid re-reading from disk on every inspection
        self._golden_bgr_cache: "np.ndarray|None" = None
        self._golden_img_cached_path: str = ""
        self._golden_filter_sig: tuple = ()   # (filter_name, …) — invalidate on pipeline change
        self._history_entries=[]       # list of (ts, save_path, result, missing_names)
        # FIX: batch counter updates at most 2x/sec
        self._counter_dirty=False; self._counter_timer=QTimer(self)
        self._counter_timer.setInterval(500); self._counter_timer.timeout.connect(self._flush_counters)
        self._counter_timer.start()
        self._result_times: deque = deque(maxlen=60)   # timestamps for throughput
        self._db: "HistoryDB|None" = None              # set by MainWindow after creation
        self._build()

    def _build(self):
        root=QHBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(12)
        L=QVBoxLayout(); L.setSpacing(10)
        # ── Project bar: single button opens ProjectDialog ─────────────
        pf=make_card("project_bar"); pf.setFixedHeight(48); pl=QHBoxLayout(pf); pl.setContentsMargins(14,6,14,6)
        self._proj_lbl=QLabel("No project"); self._proj_lbl.setObjectName("proj_lbl")
        self._proj_lbl.setStyleSheet(
            f"#proj_lbl{{color:{CYAN};font-family:Consolas;font-size:11px;font-weight:bold;border:none;}}")
        b_proj=QPushButton("⊞  SWITCH PROJECT"); b_proj.setFixedHeight(32)
        b_proj.setStyleSheet(
            f"QPushButton{{background:{AMBER_DIM};color:{AMBER};border:1px solid {AMBER}66;"
            f"border-radius:6px;font-size:11px;font-weight:bold;padding:0 12px;}}"
            f"QPushButton:hover{{background:{AMBER}22;border-color:{AMBER};color:{AMBER};}}"
        )
        b_proj.clicked.connect(self._open_project_manager)
        pl.addWidget(self._proj_lbl,1); pl.addWidget(b_proj)
        L.addWidget(pf)
        self.banner=AnnunciatorBanner(); L.addWidget(self.banner)
        self.video=VideoWidget(); L.addWidget(self.video,1)
        conf_f=make_card("conf_bar"); conf_f.setFixedHeight(62); cfl=QVBoxLayout(conf_f); cfl.setContentsMargins(14,6,14,8); cfl.setSpacing(4)
        cr=QHBoxLayout(); cr.addWidget(sec_lbl("AI CONFIDENCE")); cr.addStretch()
        self._conf_lbl=QLabel("50%"); self._conf_lbl.setObjectName("clbl")
        self._conf_lbl.setStyleSheet(f"#clbl{{color:{CYAN};font-weight:bold;border:none;}}"); cr.addWidget(self._conf_lbl); cfl.addLayout(cr)
        sr=QHBoxLayout(); self._conf=QSlider(Qt.Orientation.Horizontal); self._conf.setRange(0,100); self._conf.setValue(50)
        self._conf.valueChanged.connect(lambda v:self._conf_lbl.setText(f"{v}%"))
        self._conf.sliderReleased.connect(lambda:self._cfg.set("confidence",self._conf.value()/100))
        sr.addWidget(QLabel("0")); sr.addWidget(self._conf,1); sr.addWidget(QLabel("100")); cfl.addLayout(sr)
        L.addWidget(conf_f)
        br=QHBoxLayout(); br.setSpacing(8)
        self.btn_start=QPushButton("START INSPECTION"); self.btn_start.setObjectName("btn_start")
        self.btn_start.setFixedHeight(48); self.btn_start.setFont(_F_MONO_12)
        self.btn_start.setStyleSheet(
            f"#btn_start{{background:{CYAN};"
            f"color:#FFFFFF;border:none;border-radius:8px;padding:6px 14px;font-size:13px;font-weight:bold;letter-spacing:2px;}}"
            f"#btn_start:hover{{background:#1347A0;color:#FFFFFF;}}"
        )
        self.btn_start.clicked.connect(self._toggle)
        self.btn_master=QPushButton("⬆  UPLOAD MASTER"); self.btn_master.setObjectName("btn_master"); self.btn_master.setFixedHeight(48)
        self.btn_master.setStyleSheet(
            f"#btn_master{{background:{AMBER_DIM};"
            f"color:{AMBER};border:1px solid {AMBER_DIM};border-radius:8px;padding:6px 12px;font-size:11px;letter-spacing:1px;}}"
            f"#btn_master:hover{{background:{AMBER_DIM};border-color:{AMBER};}}"
        )
        self.btn_master.clicked.connect(self._upload_master)
        self.btn_golden_img=QPushButton("📷  SAVE REF"); self.btn_golden_img.setObjectName("btn_gimg")
        self.btn_golden_img.setFixedHeight(48)
        self.btn_golden_img.setToolTip("Save current camera frame as golden reference image for diff-based AOI\n"
                                        "Saved as golden_board.jpg in the project folder")
        self.btn_golden_img.setStyleSheet(
            f"#btn_gimg{{background:{BG_CARD2};color:{PURPLE};border:1px solid {PURPLE}44;"
            f"border-radius:8px;font-size:10px;letter-spacing:1px;}}"
            f"#btn_gimg:hover{{border-color:{PURPLE}99;color:{PURPLE};}}"
        )
        self.btn_golden_img.clicked.connect(self._save_golden_img)
        self.btn_flag=QPushButton("⚑  FLAG"); self.btn_flag.setFixedHeight(48)
        self.btn_flag.setStyleSheet(
            f"QPushButton{{background:{BG_CARD2};color:{TEXT_SEC};border:1px solid {BG_BORDER2};border-radius:8px;font-size:11px;}}"
            f"QPushButton:hover{{color:{RED};border-color:{RED_DIM};}}"
        )
        self.btn_flag.clicked.connect(self._flag)
        br.addWidget(self.btn_start,2); br.addWidget(self.btn_master,1); br.addWidget(self.btn_golden_img); br.addWidget(self.btn_flag)
        L.addLayout(br)
        pf=make_card("pipe_bar"); pf.setFixedHeight(48); pl=QHBoxLayout(pf); pl.setContentsMargins(14,6,14,6)
        self._pipe_lbl=QLabel("No pipeline deployed"); self._pipe_lbl.setObjectName("plbl")
        self._pipe_lbl.setStyleSheet(f"#plbl{{color:{TEXT_SEC};font-size:11px;font-family:Consolas;border:none;}}")
        pl.addWidget(self._pipe_lbl,1)
        self.btn_inspect=QPushButton("INSPECT NOW"); self.btn_inspect.setObjectName("btn_insp")
        self.btn_inspect.setFixedHeight(34); self.btn_inspect.setEnabled(False)
        self.btn_inspect.setStyleSheet(f"#btn_insp{{background:{CYAN_DIM};color:{CYAN};border:1px solid {CYAN}44;border-radius:5px;padding:4px 12px;}} #btn_insp:disabled{{background:{BG_BORDER};color:{TEXT_DIM};}}")
        self.btn_inspect.clicked.connect(self._inspect); pl.addWidget(self.btn_inspect)
        L.addWidget(pf); root.addLayout(L,3)
        R=QVBoxLayout(); R.setSpacing(10)
        # Wrap R in a widget with minimum width so it never clips
        R_widget=QWidget(); R_widget.setLayout(R); R_widget.setMinimumWidth(260)
        root.addWidget(R_widget,1)
        R.addWidget(sec_lbl("YIELD"))
        sg=QGridLayout(); sg.setSpacing(6)
        self._c_total=StatCard("TOTAL","0",CYAN); self._c_pass=StatCard("PASS","0",GREEN); self._c_fail=StatCard("FAIL","0",RED)
        self._c_total.setMinimumHeight(72); self._c_pass.setMinimumHeight(72); self._c_fail.setMinimumHeight(72)
        sg.addWidget(self._c_total,0,0); sg.addWidget(self._c_pass,0,1); sg.addWidget(self._c_fail,1,0,1,2)
        R.addLayout(sg)
        self._ybar=QProgressBar(); self._ybar.setFixedHeight(14); self._ybar.setValue(0)
        self._ybar.setTextVisible(False)
        self._ybar.setStyleSheet(
            f"QProgressBar{{background:{BG_PANEL};border:1px solid {BG_BORDER};border-radius:6px;}}"
            f"QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {CYAN_DIM},stop:1 {CYAN});border-radius:5px;}}"
        )
        self._ypct=QLabel("YIELD  —"); self._ypct.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ypct.setObjectName("ypct")
        self._ypct.setStyleSheet(f"#ypct{{color:{TEXT_SEC};font-size:13px;font-weight:bold;font-family:'Consolas';letter-spacing:2px;border:none;}}")
        R.addWidget(self._ybar); R.addWidget(self._ypct)
        b_rst=QPushButton("RESET"); b_rst.setObjectName("b_rst"); b_rst.setFixedHeight(32)
        b_rst.setStyleSheet(
            f"#b_rst{{background:{BG_CARD2};color:{TEXT_DIM};border:1px solid {BG_BORDER2};border-radius:6px;"
            f"font-size:10px;font-family:'Consolas';letter-spacing:2px;}}"
            f"#b_rst:hover{{color:{RED};border-color:{RED_DIM};}}"
        )
        b_rst.clicked.connect(self._reset); R.addWidget(b_rst)

        # ── Last-10 result strip + throughput ─────────────────────────────
        strip_card = make_card("strip_card"); strip_card.setFixedHeight(54)
        strip_lay = QVBoxLayout(strip_card); strip_lay.setContentsMargins(12,4,12,4); strip_lay.setSpacing(2)
        strip_hdr = QHBoxLayout()
        strip_hdr.addWidget(sec_lbl("LAST 10 BOARDS"))
        strip_hdr.addStretch()
        self._throughput_lbl = QLabel("— boards/min")
        self._throughput_lbl.setStyleSheet(f"color:{CYAN};font-size:10px;font-family:Consolas;border:none;")
        strip_hdr.addWidget(self._throughput_lbl)
        strip_lay.addLayout(strip_hdr)
        self._result_strip = ResultStrip()
        strip_lay.addWidget(self._result_strip)
        R.addWidget(strip_card)
        lf_row=QHBoxLayout()
        lf_row.addWidget(sec_lbl("INSPECTION LOG"))
        lf_row.addStretch()
        self._det_count_lbl=QLabel(""); self._det_count_lbl.setObjectName("det_cnt")
        self._det_count_lbl.setStyleSheet(f"#det_cnt{{color:{CYAN};font-size:10px;font-family:Consolas;border:none;padding:0 6px;}}")
        self._lat_lbl=QLabel("— ms | — fps"); self._lat_lbl.setObjectName("lat_lbl")
        self._lat_lbl.setStyleSheet(f"#lat_lbl{{color:{TEXT_SEC};font-size:10px;font-family:Consolas;border:none;}}")
        lf_row.addWidget(self._det_count_lbl); lf_row.addWidget(self._lat_lbl)
        R.addWidget(make_sep()); R.addLayout(lf_row)
        # FIX: FastLog replaces QTextEdit - plain text, batched, 500-line cap
        self._log=FastLog(); self._log.setObjectName("runlog")
        self._log.setStyleSheet(
            f"#runlog{{background:{BG_DEEP};border:1px solid {BG_BORDER2};border-radius:8px;"
            f"padding:10px;color:{GREEN};font-family:'Consolas';font-size:11px;line-height:1.5;}}"
        )
        R.addWidget(self._log,1)

        # ── DEFECT HISTORY strip ──────────────────────────────────────────────
        R.addWidget(make_sep())
        hist_hdr = QHBoxLayout()
        hist_hdr.addWidget(sec_lbl("DEFECT HISTORY"))
        hist_hdr.addStretch()
        self._hist_count = QLabel("0 defects")
        self._hist_count.setObjectName("hcnt")
        self._hist_count.setStyleSheet(
            f"#hcnt{{color:{RED};font-size:10px;font-family:Consolas;border:none;padding:0 4px;}}")
        b_hist_clear = QPushButton("Clear")
        b_hist_clear.setFixedHeight(20); b_hist_clear.setFixedWidth(42)
        b_hist_clear.setStyleSheet(
            f"QPushButton{{background:{BG_CARD};color:{TEXT_DIM};border:1px solid {BG_BORDER};"
            f"border-radius:4px;font-size:10px;padding:0;}}")
        b_hist_clear.clicked.connect(self._clear_history)
        hist_hdr.addWidget(self._hist_count); hist_hdr.addWidget(b_hist_clear)
        R.addLayout(hist_hdr)
        # Horizontal scrolling strip of annotated defect thumbnails
        self._hist_scroll = QScrollArea()
        self._hist_scroll.setFixedHeight(158)
        self._hist_scroll.setWidgetResizable(False)    # we control size manually
        self._hist_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._hist_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._hist_scroll.setStyleSheet(
            f"QScrollArea{{border:1px solid {BG_BORDER};border-radius:6px;background:{BG_DEEP};}}")
        self._hist_inner = QWidget()
        # Fixed height = card height + 8px top/bottom breathing room.
        # Width will grow as cards are added; scroll area provides horizontal scroll.
        self._hist_inner.setFixedHeight(146)
        self._hist_inner.setStyleSheet(f"background:{BG_CARD2};")
        self._hist_layout = QHBoxLayout(self._hist_inner)
        self._hist_layout.setContentsMargins(6, 4, 6, 4)
        self._hist_layout.setSpacing(6)
        self._hist_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._hist_inner.setMinimumWidth(300)   # guaranteed visible before first card
        self._hist_scroll.setWidget(self._hist_inner)
        R.addWidget(self._hist_scroll)

        R.addWidget(make_sep()); R.addWidget(sec_lbl("GOLDEN MASTER"))
        gr=QHBoxLayout(); self._gm_lbl=QLabel("No master loaded"); self._gm_lbl.setObjectName("gmlbl")
        self._gm_lbl.setStyleSheet(f"#gmlbl{{color:{TEXT_SEC};font-size:11px;border:none;}}")
        b_lgm=QPushButton("Load .json"); b_lgm.setFixedHeight(26); b_lgm.clicked.connect(self._load_golden_dlg)
        gr.addWidget(self._gm_lbl,1); gr.addWidget(b_lgm); R.addLayout(gr)
        root.addLayout(R,1) if False else None  # R_widget added above

    def deploy_pipeline(self,filters,rois,model_path=None):
        self._pipe_filters=list(filters); self._pipe_rois=list(rois); self._pipe_active=bool(rois or filters); self._pipe_model=None
        # Invalidate golden cache — filters changed so the pre-filtered golden is stale
        self._golden_bgr_cache = None; self._golden_filter_sig = ()
        # Use plain _YOLO (same as aoi_tab) — load_optimized_yolo can silently return a
        # broken model that is non-None, causing _inspect to skip the fallback _YOLO load
        # path entirely and then produce zero detections from safe_predict.
        if model_path and os.path.exists(model_path) and HAS_YOLO:
            try:
                # Pass task='detect' explicitly — OpenVINO models can't auto-infer task
                # (suppresses "Unable to automatically guess model task" warning)
                self._pipe_model = _YOLO(model_path, task="detect")
                self._log.append(f"[Model] Loaded: {os.path.basename(model_path)}")
            except Exception as e:
                self._log.append(f"[Model] Load error: {e}")
        self._pipe_lbl.setText(f"Pipeline: {len(filters)} filters | {len(rois)} zones")
        self._pipe_lbl.setStyleSheet(f"#plbl{{color:{CYAN};font-size:11px;font-family:Consolas;border:none;}}")
        self._update_inspect_btn()

    def _update_inspect_btn(self):
        """Enable INSPECT NOW whenever camera is live, regardless of pipeline state."""
        enabled = self._running
        self.btn_inspect.setEnabled(enabled)
        if enabled:
            self.btn_inspect.setToolTip("Run a single AOI check on the current frame"
                                        + (" (pipeline active)" if self._pipe_active else " (golden diff mode)"))
        else:
            self.btn_inspect.setToolTip("Start camera first")

    def _inspect(self):
        frame = self._cam.get_small_bgr() if self._cam else None
        if frame is None:
            self._log.append("[!] Start camera first."); return
        frame = frame.copy()
        ts = datetime.now().strftime("%H:%M:%S")

        # Apply filter pipeline
        filtered = apply_filters(frame, self._pipe_filters)

        # ── ROI zone checks ───────────────────────────────────────────────
        all_ok = True; roi_disp = []
        for roi in self._pipe_rois:
            res = run_roi(filtered, roi, self._pipe_model)
            passed = res.get("passed", False); info = res.get("info", "")
            if not passed: all_ok = False
            self._log.append(f"[{ts}] {'OK' if passed else 'FAIL'} [{roi.zone_type.upper()}] {roi.name}: {info}")
            roi_disp.append({"name":roi.name,"rect":roi.rect,"type":roi.zone_type,
                             "passed":passed,"info":info[:20]})

        thresh = self._conf.value() / 100

        # ── YOLO detections on filtered frame ─────────────────────────────
        mp = self._cfg.get("model_path","")
        active_model = self._pipe_model
        if active_model is None and mp and os.path.exists(mp) and HAS_YOLO:
            if not hasattr(self, "_inspect_model") or self._inspect_model_path != mp:
                try:
                    self._inspect_model = _YOLO(mp)
                    self._inspect_model_path = mp
                except Exception as e:
                    self._log.append(f"[!] Model load error: {e}")
                    self._inspect_model = None
                    self._inspect_model_path = ""
            active_model = getattr(self, "_inspect_model", None)

        live = []
        if active_model is not None and HAS_YOLO:
            try:
                res = safe_predict(active_model, filtered, conf=thresh, iou=0.35, imgsz=640, verbose=False, device='cpu')[0]

                for box in res.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    c = float(box.conf[0]); cls = int(box.cls[0])
                    name = active_model.names.get(cls, "?")
                    live.append({"class":cls,"name":name,"xyxy":[x1,y1,x2,y2],"conf":c})
            except Exception as e:
                self._log.append(f"[{ts}] [Detect] {e}")
                live = [d for d in self._latest_dets if d.get("conf",1) >= thresh]
        else:
            live = [d for d in self._latest_dets if d.get("conf",1) >= thresh]

        # ── AOI diff engine (same as offline) when golden image is present ─
        missing_names = []; missing_idx = []; defects_aoi = []
        golden_img_path = os.path.join(self._cfg._path or ".", "golden_board.jpg") \
                          if self._cfg._path else ""
        # Also accept .png
        if not os.path.exists(golden_img_path):
            golden_img_path = os.path.join(self._cfg._path or ".", "golden_board.png") \
                              if self._cfg._path else ""

        use_aoi_diff = (HAS_CV2 and HAS_NP and
                        self._golden and
                        os.path.exists(golden_img_path))

        if use_aoi_diff:
            try:
                # ── Golden-image cache ─────────────────────────────────────────
                # Build a signature for the current filter pipeline so we can
                # detect when it changes even if deploy_pipeline wasn't called.
                cur_sig = tuple(
                    (f.name, f.enabled, tuple(sorted(f.params.items())))
                    for f in self._pipe_filters
                )
                if (self._golden_bgr_cache is None
                        or self._golden_img_cached_path != golden_img_path
                        or self._golden_filter_sig != cur_sig):
                    raw_golden = cv2.imread(golden_img_path)
                    if raw_golden is not None:
                        if self._pipe_filters:
                            self._golden_bgr_cache = apply_filters(raw_golden, self._pipe_filters)
                        else:
                            self._golden_bgr_cache = raw_golden
                        self._golden_img_cached_path = golden_img_path
                        self._golden_filter_sig = cur_sig
                golden_bgr = self._golden_bgr_cache
                if golden_bgr is None:
                    use_aoi_diff = False
                else:
                    gH, gW = golden_bgr.shape[:2]
                    fH, fW = filtered.shape[:2]
                    test_for_diff = cv2.resize(filtered, (gW, gH)) \
                                    if (fW, fH) != (gW, gH) else filtered

                    # Build _AOIComp list from golden JSON
                    aoi_golden = []
                    for gi, g in enumerate(self._golden):
                        bx = g["xyxy"]  # [x1,y1,x2,y2] in pixels of the cam frame
                        cx = ((bx[0]+bx[2])/2) / fW
                        cy = ((bx[1]+bx[3])/2) / fH
                        w  = (bx[2]-bx[0]) / fW
                        h  = (bx[3]-bx[1]) / fH
                        aoi_golden.append(_AOIComp(
                            id=gi, label=g.get("name","?"),
                            conf=1.0, cx=cx, cy=cy, w=max(w,0.01), h=max(h,0.01)))

                    # Convert live YOLO dets to _AOIComp
                    aoi_dets = []
                    for di, d in enumerate(live):
                        bx = d["xyxy"]
                        cx = ((bx[0]+bx[2])/2) / fW
                        cy = ((bx[1]+bx[3])/2) / fH
                        w  = (bx[2]-bx[0]) / fW
                        h  = (bx[3]-bx[1]) / fH
                        aoi_dets.append(_AOIComp(
                            id=di, label=d["name"],
                            conf=d.get("conf",1.0), cx=cx, cy=cy,
                            w=max(w,0.01), h=max(h,0.01)))

                    # Use cached calibration if golden unchanged, else use defaults
                    pos_tol  = getattr(self, "_live_pos_tol",  _MIN_POS_TOL)
                    per_thr  = getattr(self, "_live_per_thr",  {})

                    defects_aoi, _ = _aoi_check_board(
                        aoi_golden, golden_bgr, test_for_diff,
                        aoi_dets, pos_tol=pos_tol, per_thr=per_thr)

                    for d in defects_aoi:
                        dt = d["defect_type"]
                        lbl = d.get("expected_label","?")
                        found = d.get("found_label","")
                        if dt == "missing":
                            i = d.get("component_id", -1)
                            if 0 <= i < len(self._golden):
                                missing_idx.append(i)
                                missing_names.append(lbl)
                        self._log.append(
                            f"[{ts}] [{dt.upper():<16}] {lbl}"
                            + (f"  → {found}" if found and found!="none" else ""))
                    if defects_aoi:
                        all_ok = False
            except Exception as e:
                self._log.append(f"[{ts}] [AOI-diff] {e}")
                use_aoi_diff = False

        # Fallback: IoU/persistence matching when no golden image
        if not use_aoi_diff and self._golden:
            missing_idx = match_detections(self._golden, live,
                                           self._cfg.get("match_iou"),
                                           self._persist,
                                           self._cfg.get("persistence"))
            missing_names = [self._golden[i]["name"]
                             for i in missing_idx if i < len(self._golden)]
            if missing_names:
                all_ok = False

        self.video.set_detections(live, self._golden, missing_idx)
        self.video.set_roi_results(roi_disp)

        self._total += 1
        n_defects = len(defects_aoi) + sum(1 for r in roi_disp if not r.get("passed"))
        if all_ok:
            self._pass += 1; self.banner.set_pass()
            self._log.append(f"[{ts}] ✔ PASS  ({len(live)} det)")
        else:
            self._fail += 1
            reasons = [d["defect_type"].upper() + " " + d.get("expected_label","")
                       for d in defects_aoi[:2]] + \
                      [r.get("info","") for r in roi_disp if not r.get("passed")]
            self.banner.set_fail(reasons[0] if reasons else "FAIL")
            if defects_aoi:
                types = {}
                for d in defects_aoi: types[d["defect_type"]] = types.get(d["defect_type"],0)+1
                self._log.append(f"[{ts}] ✘ FAIL  " +
                    "  ".join(f"{k}×{v}" for k,v in types.items()))
            elif missing_names:
                miss_str = f"  MISSING: {', '.join(missing_names[:3])}{'…' if len(missing_names)>3 else ''}"
                self._log.append(f"[{ts}] ✘ FAIL{miss_str}")
            else:
                self._log.append(f"[{ts}] ✘ FAIL  ROI check")

        self._counter_dirty = True
        # Result strip + throughput
        self._result_strip.push(all_ok)
        self._result_times.append(time.time())
        now2 = time.time()
        recent = [t for t in self._result_times if now2-t <= 60]
        rate = len(recent)/max(1,(now2-recent[0])/60) if len(recent) > 1 else 0
        self._throughput_lbl.setText(f"{rate:.1f} boards/min")
        # History DB log
        if self._db and self._cfg.get("log_history", True):
            proj  = os.path.basename(self._cfg._path or "")
            mdl   = os.path.basename(self._cfg.get("model_path",""))
            dtypes = ",".join(sorted({d["defect_type"] for d in defects_aoi} |
                                     {r.get("info","") for r in roi_disp if not r.get("passed")} |
                                     set(missing_names)))
            _db_row_id = self._db.log_result(project=proj, mode="RUN", passed=all_ok,
                                n_defects=n_defects,
                                defect_types=dtypes[:120], model=mdl)
        else:
            _db_row_id = None
        # Annotated side-by-side history image
        _golden_img = golden_img_path if os.path.exists(golden_img_path) else None
        self._add_history_entry(
            ts, frame, live, self._golden, missing_idx,
            "PASS" if all_ok else "FAIL", missing_names, defects_aoi, _golden_img,
            db_row_id=_db_row_id)

    def _save_golden_img(self):
        """Save the current camera frame as the golden reference image for diff-based AOI."""
        frame = self._cam.get_small_bgr() if self._cam else None
        if frame is None:
            QMessageBox.warning(self,"No Frame","Start the camera first, then save the reference frame.")
            return
        if not self._cfg._path:
            QMessageBox.warning(self,"No Project","Load a project first.")
            return
        # Apply the active filter pipeline so the saved golden already matches
        # what every future test frame will look like after pre-processing.
        if self._pipe_filters:
            frame = apply_filters(frame.copy(), self._pipe_filters)
        path = os.path.join(self._cfg._path, "golden_board.jpg")
        if HAS_CV2:
            cv2.imwrite(path, frame)
            self._log.append(f"[REF] Golden reference image saved → {os.path.basename(path)}")
            self._log.append(f"[REF] Resolution: {frame.shape[1]}×{frame.shape[0]}")
            # Invalidate cached calibration AND cached golden BGR
            self._live_pos_tol = _MIN_POS_TOL
            self._live_per_thr = {}
            self._golden_bgr_cache = None; self._golden_img_cached_path = ""
            ToastManager.show(self.window(), "Golden reference image saved — diff AOI active", "success")
        else:
            self._log.append("[REF] OpenCV not available")

    def _calibrate_live(self, golden_bgr, golden_dets, model):
        """Background calibration for live diff AOI — runs once after golden image saved."""
        if not HAS_CV2 or not HAS_NP or not golden_dets: return
        try:
            pos_tol, per_thr = _aoi_calibrate(
                model, golden_bgr, golden_dets,
                self._conf.value()/100, cal_runs=4, use_sahi=False,
                model_path=self._cfg.get("model_path",""))
            self._live_pos_tol = pos_tol
            self._live_per_thr = per_thr
            self._log.append(f"[CAL] Live calibration done — pos_tol={pos_tol:.4f} ({len(per_thr)} slots)")
        except Exception as e:
            self._log.append(f"[CAL] Calibration error: {e}")

    def _load_golden_dlg(self):
        p,_=QFileDialog.getOpenFileName(self,"Load Golden Master","","JSON (*.json)")
        if p: self.load_golden(p)

    def load_golden(self, path):
        """Load a golden master JSON file into self._golden and update the UI label."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._golden = data if isinstance(data, list) else []
            self._persist = {i: 0 for i in range(len(self._golden))}
            lbl = f"{os.path.basename(path)}  ({len(self._golden)} components)"
            self._gm_lbl.setText(lbl)
            self._gm_lbl.setStyleSheet(f"#gmlbl{{color:{GREEN};font-size:11px;border:none;}}")
            self._log.append(f"[Master] Loaded {len(self._golden)} components from {os.path.basename(path)}")
        except Exception as e:
            self._log.append(f"[Master] Failed to load {path}: {e}")

    def _save_golden(self):
        # BUG FIX 2: filter by current confidence threshold — _latest_dets is unfiltered
        thresh = self._conf.value() / 100
        filtered_dets = [d for d in self._latest_dets if d.get("conf", 1) >= thresh]
        if not filtered_dets: self._log.append("[!] No detections above threshold."); return
        path=os.path.join(self._cfg._path or ".",self._cfg.get("golden_file"))
        with open(path,"w") as f: json.dump([{"class":d["class"],"name":d["name"],"xyxy":d["xyxy"]} for d in filtered_dets],f,indent=2)
        self.load_golden(path); self.banner.set_pass()


    # ─────────────────────────────────────────────────────────────────────
    # DEFECT HISTORY helpers
    # ─────────────────────────────────────────────────────────────────────
    def _add_history_entry(self, ts, frame_bgr, live, golden, missing_idx,
                           result, missing_names, defects_aoi=None, golden_img_path=None,
                           db_row_id=None):
        """Spawn a background thread to build + save the annotated side-by-side image."""
        defects_aoi = list(defects_aoi or [])

        # Always show a card in the strip; only save the image file if setting enabled
        save_image = bool(self._cfg.get("save_annot_history", True))

        # Without OpenCV show a text-only card
        if not HAS_CV2 or not HAS_NP:
            QTimer.singleShot(0, lambda: self._show_history_thumb(
                None, ts, result, list(missing_names), list(defects_aoi), ""))
            return

        # Capture everything by VALUE before handing off to the worker thread
        frame_copy     = frame_bgr.copy()
        golden_copy    = list(golden)
        missing_copy   = list(missing_idx)
        missing_n_copy = list(missing_names)
        defects_copy   = list(defects_aoi)
        cfg_path       = self._cfg._path or ""

        _thr = threading   # module-level — no frozen-module lookup
        def _work():
            try:
                ann = self._build_annotated_image(
                    frame_copy, live, golden_copy, missing_copy,
                    result, missing_n_copy, defects_copy, ts, golden_img_path)

                save_path = ""
                if ann is not None and save_image:
                    hist_dir  = os.path.join(cfg_path, "history") if cfg_path \
                                else os.path.join(os.getcwd(), "history")
                    os.makedirs(hist_dir, exist_ok=True)
                    fname     = f"{result}_{ts.replace(':','')}.jpg"
                    save_path = os.path.join(hist_dir, fname)
                    if not cv2.imwrite(save_path, ann):
                        save_path = ""   # imwrite failed — still show card
                    elif db_row_id and self._db:
                        self._db.update_image_path(db_row_id, save_path)

                qimg = None
                if ann is not None:
                    h2, w2 = ann.shape[:2]
                    rgb    = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                    # .copy() makes QImage own its buffer so it survives the thread
                    qimg   = QImage(bytes(rgb.tobytes()), w2, h2,
                                    w2*3, QImage.Format.Format_RGB888).copy()

                # All widget creation must be on the main thread.
                # Capture loop variables explicitly in default args.
                QTimer.singleShot(0, lambda qi=qimg, sp=save_path: self._show_history_thumb(
                    qi, ts, result, missing_n_copy, defects_copy, sp))

            except Exception as exc:
                err_msg = str(exc)
                QTimer.singleShot(0, lambda m=err_msg: self._log.append(
                    f"[History] Error building annotated image: {m}"))
                QTimer.singleShot(0, lambda: self._show_history_thumb(
                    None, ts, result, missing_n_copy, defects_copy, ""))

        _thr.Thread(target=_work, daemon=True).start()

    def _build_annotated_image(self, frame_bgr, live, golden, missing_idx,
                               result, missing_names, defects_aoi, ts,
                               golden_img_path=None):
        """
        Build and return a side-by-side BGR numpy image.
          LEFT  — golden reference board (golden_board.jpg if saved, else live fallback)
          RIGHT — current scan with live detections, ghost outlines and defect markers
        Raises on error (caller logs it).
        """
        h, w   = frame_bgr.shape[:2]
        PAD    = 10
        HEADER = 34
        FOOTER = 26

        # ── LEFT panel — golden reference ────────────────────────────────
        if golden_img_path and os.path.exists(golden_img_path):
            ref = cv2.imread(golden_img_path)
            if ref is None:
                ref = frame_bgr.copy()
            else:
                ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            ref = frame_bgr.copy()

        left     = ref.copy()
        miss_set = set(missing_idx)

        for i, g in enumerate(golden):
            x1, y1, x2, y2 = [int(v) for v in g["xyxy"]]
            name = g.get("name", "?")
            if i in miss_set:
                overlay = left.copy()
                cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,180), -1)
                cv2.addWeighted(overlay, 0.40, left, 0.60, 0, left)
                cv2.rectangle(left, (x1,y1), (x2,y2), (50,50,240), 2)
                ly = max(y1 - 4, 14)
                cv2.rectangle(left, (x1,ly-13), (x1+38, ly+2), (0,0,150), -1)
                cv2.putText(left, "MISS", (x1+2, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(left, name, (x1, min(y2+13, h-2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200,140,140), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(left, (x1,y1), (x2,y2), (50,200,80), 1)
                cv2.putText(left, name, (x1, min(y2+13, h-2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130,210,130), 1, cv2.LINE_AA)

        # ── RIGHT panel — live scan ───────────────────────────────────────
        right = frame_bgr.copy()

        # Faint ghost outlines of all golden positions for spatial context
        for i, g in enumerate(golden):
            x1, y1, x2, y2 = [int(v) for v in g["xyxy"]]
            ghost = (35,35,90) if i in miss_set else (30,65,30)
            cv2.rectangle(right, (x1,y1), (x2,y2), ghost, 1)

        # Live YOLO detections coloured by confidence
        for d in live:
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            cf  = d.get("conf", 1.0)
            col = (40,220,40) if cf >= 0.75 else (40,165,255) if cf >= 0.50 else (80,80,220)
            cv2.rectangle(right, (x1,y1), (x2,y2), col, 2)
            tag = f"{d['name']} {cf:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            ty = max(y1 - 4, 12)
            cv2.rectangle(right, (x1, ty-th-2), (x1+tw+4, ty+2), (0,0,0), -1)
            cv2.putText(right, tag, (x1+2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

        # Red × cross on each missing component position
        for i in miss_set:
            if i >= len(golden): continue
            x1, y1, x2, y2 = [int(v) for v in golden[i]["xyxy"]]
            cx, cy = (x1+x2)//2, (y1+y2)//2
            r = max(7, min(x2-x1, y2-y1) // 3)
            cv2.line(right, (cx-r,cy-r), (cx+r,cy+r), (60,60,230), 2)
            cv2.line(right, (cx+r,cy-r), (cx-r,cy+r), (60,60,230), 2)

        # AOI non-missing defects (misaligned / wrong_component / wrong_polarity)
        for d in defects_aoi:
            if d.get("defect_type") == "missing":
                continue
            bx = d.get("xyxy")
            if bx and len(bx) == 4:
                x1, y1, x2, y2 = [int(v) for v in bx]
                dt = d.get("defect_type", "?").upper()[:10]
                cv2.rectangle(right, (x1,y1), (x2,y2), (50,50,210), 2)
                cv2.putText(right, dt, (x1, max(y1-4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100,100,255), 1, cv2.LINE_AA)

        # ── Compose final canvas ──────────────────────────────────────────
        total_w = w*2 + PAD
        total_h = h + HEADER + FOOTER
        canvas  = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:,:] = (16, 18, 22)
        canvas[HEADER:HEADER+h,  0:w]            = left
        canvas[HEADER:HEADER+h,  w+PAD:w*2+PAD]  = right

        # Header bar
        cv2.rectangle(canvas, (0,0), (total_w, HEADER-1), (22,26,34), -1)
        lbl_left = "GOLDEN REF (photo)"                    if (golden_img_path and os.path.exists(golden_img_path or ""))                    else "GOLDEN (live fallback)"
        cv2.putText(canvas, lbl_left,           (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100,180,220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"LIVE SCAN  {ts}", (w+PAD+8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100,180,220), 1, cv2.LINE_AA)

        # PASS / FAIL badge top-right
        res_col = (40,200,60) if result == "PASS" else (50,50,220)
        badge_x = total_w - 76
        cv2.rectangle(canvas, (badge_x-2,3), (total_w-4, HEADER-4), res_col, -1)
        cv2.putText(canvas, f" {result}", (badge_x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2, cv2.LINE_AA)

        # Vertical divider between panels
        cv2.line(canvas, (w+PAD//2, HEADER), (w+PAD//2, HEADER+h), (50,55,70), 1)

        # Footer bar
        fy = HEADER + h + 17
        cv2.rectangle(canvas, (0,HEADER+h), (total_w,total_h), (22,26,34), -1)
        if defects_aoi:
            counts  = {}
            for d in defects_aoi:
                k = d.get("defect_type","?")
                counts[k] = counts.get(k,0)+1
            summary = "  ·  ".join(f"{k}×{v}" for k,v in counts.items())
            cv2.putText(canvas, summary[:95], (8,fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,100,240), 1, cv2.LINE_AA)
        elif missing_names:
            txt = "MISSING: " + "  ·  ".join(missing_names[:7])                   + ("  …" if len(missing_names)>7 else "")
            cv2.putText(canvas, txt[:100], (8,fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,100,240), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "All components present", (8,fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (50,185,50), 1, cv2.LINE_AA)

        n_def = len(defects_aoi) + len(missing_names)
        if n_def:
            cv2.putText(canvas, f"{n_def} defect(s)", (total_w-95, fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,100,240), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (0,0), (total_w-1,total_h-1), res_col, 2)
        return canvas

    def _show_history_thumb(self, qimg, ts, result, missing_names, defects_aoi, save_path):
        """
        Add a clickable thumbnail card to the history strip.
        Must run on the main (UI) thread.
        qimg may be None — a text-only fallback card is shown in that case.
        """
        # Card must fit inside the 158px scroll area.
        # 158 - 16px (scrollbar) = 142px usable.  Card = 138px + 2px margin each side = 142. ✓
        CARD_W = 192
        CARD_H = 138

        col_hex = RED if result == "FAIL" else GREEN

        card = QFrame()
        card.setFixedSize(CARD_W, CARD_H)
        card.setStyleSheet(
            f"QFrame{{background:{BG_CARD2};border:2px solid {col_hex}66;"
            f"border-radius:6px;}}"
            f"QFrame:hover{{border-color:{col_hex};}}")

        cl = QVBoxLayout(card)
        cl.setContentsMargins(3, 3, 3, 3)
        cl.setSpacing(2)

        THUMB_H = CARD_H - 26   # reserve 26px for info label + padding

        if qimg is not None:
            pm = QPixmap.fromImage(qimg).scaled(
                CARD_W - 6, THUMB_H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            img_lbl = QLabel()
            img_lbl.setPixmap(pm)
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setFixedHeight(THUMB_H)
            if save_path and os.path.exists(save_path):
                img_lbl.setToolTip(f"Click to open full-size annotated image\n{save_path}")
                img_lbl.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                def _open_img(ev, p=save_path):
                    if os.path.exists(p):
                        if platform.system() == "Windows":
                            os.startfile(p)
                        elif platform.system() == "Darwin":
                            __import__("subprocess").Popen(["open", p])
                        else:
                            __import__("subprocess").Popen(["xdg-open", p])
                img_lbl.mousePressEvent = _open_img
            else:
                img_lbl.setToolTip("Image saved to project/history/")
            cl.addWidget(img_lbl)
        else:
            # No cv2 / failed render — show coloured placeholder
            ph = QLabel("No image\n(see log)")
            ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ph.setFixedHeight(THUMB_H)
            ph.setStyleSheet(
                f"color:{col_hex};font-size:9px;font-family:Consolas;"
                f"border:none;background:transparent;")
            cl.addWidget(ph)

        # Info label: timestamp + defect summary
        n_def = len(defects_aoi) + len(missing_names)
        if result == "FAIL" and defects_aoi:
            types = sorted({d.get("defect_type","?")[:5] for d in defects_aoi})
            info_txt = f"{ts}  {'+'.join(types)}×{n_def}"
        elif result == "FAIL" and missing_names:
            info_txt = f"{ts}  MISS×{len(missing_names)}"
        else:
            info_txt = f"{ts}  ✔ PASS"

        info_lbl = QLabel(info_txt)
        info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_lbl.setFixedHeight(18)
        info_lbl.setStyleSheet(
            f"color:{col_hex};font-size:9px;font-family:Consolas;"
            f"border:none;background:transparent;")
        cl.addWidget(info_lbl)

        # Append card to the layout (no stretch — left-aligned)
        self._hist_layout.addWidget(card)

        # Expand inner widget width to fit all cards
        n_cards = self._hist_layout.count()
        total_w = n_cards * (CARD_W + 6) + 12   # cards * (width+spacing) + margins
        self._hist_inner.setMinimumWidth(total_w)

        # Update badge counter
        self._history_entries.append((ts, save_path, result, missing_names))
        fail_cnt = sum(1 for e in self._history_entries if e[2] == "FAIL")
        total    = len(self._history_entries)
        self._hist_count.setText(
            f"{fail_cnt} fail{'s' if fail_cnt!=1 else ''} / {total} total")

        # Auto-scroll to show newest card
        QTimer.singleShot(80, lambda: self._hist_scroll.horizontalScrollBar().setValue(
            self._hist_scroll.horizontalScrollBar().maximum()))

    def _clear_history(self):
        """Remove all cards and reset the counter label."""
        while self._hist_layout.count() > 0:
            item = self._hist_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._hist_inner.setMinimumWidth(self._hist_scroll.width() or 260)
        self._history_entries.clear()
        self._hist_count.setText("0 defects")



    def _upload_master(self):
        """Upload a golden master: image auto-detected via YOLO, or load existing JSON."""
        if not self._running:
            # Not live — offer to load existing JSON or auto-detect from image
            choice=QMessageBox.question(self,"Upload Master",
                "Load from image (auto-detect components)?\n\nYes = pick image  |  No = pick JSON",
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No|QMessageBox.StandardButton.Cancel)
            if choice==QMessageBox.StandardButton.Yes:
                p,_=QFileDialog.getOpenFileName(self,"Golden Board Image","","Images (*.jpg *.png *.jpeg *.bmp)")
                if not p: return
                self._log.append(f"[Master] Detecting components in {os.path.basename(p)}…")
                if not HAS_YOLO or not HAS_CV2:
                    self._log.append("[Master] Need OpenCV+ultralytics to auto-detect"); return
                mp=self._cfg.get("model_path")
                if not mp or not os.path.exists(mp):
                    self._log.append("[Master] No model loaded – check FILTERS tab"); return
                try:
                    model=_YOLO(mp, task="detect"); img=cv2.imread(p); h,w=img.shape[:2]
                    res=safe_predict(model, img,conf=self._conf.value()/100,imgsz=640,verbose=False,device='cpu')[0]
                    dets=[{"class":int(b.cls[0]),"name":model.names[int(b.cls[0])],
                           "xyxy":list(map(int,b.xyxy[0]))} for b in res.boxes]
                    if not dets: self._log.append("[Master] No components detected"); return
                    path=os.path.join(self._cfg._path or ".","golden_master.json")
                    with open(path,"w") as f: json.dump(dets,f,indent=2)
                    self.load_golden(path); self.banner.set_pass()
                    self._log.append(f"[Master] Saved {len(dets)} components from image")
                except Exception as e:
                    self._log.append(f"[Master] Error: {e}")
            elif choice==QMessageBox.StandardButton.No:
                self._load_golden_dlg()
        else:
            # Live — snapshot current detections
            # BUG FIX 3: filter by confidence threshold — _latest_dets is unfiltered
            thresh = self._conf.value() / 100
            filtered_live = [d for d in self._latest_dets if d.get("conf", 1) >= thresh]
            if not filtered_live: self._log.append("[!] No detections to save."); return
            path=os.path.join(self._cfg._path or ".","golden_master.json")
            with open(path,"w") as f:
                json.dump([{"class":d["class"],"name":d["name"],"xyxy":d["xyxy"]} for d in filtered_live],f,indent=2)
            self.load_golden(path); self.banner.set_pass()
            self._log.append(f"[Master] Saved {len(filtered_live)} components from live camera")

    def _toggle(self):
        if not self._running: self._start()
        else: self._stop()

    def _start(self):
        # Guard: validate model path before starting (mirrors aoi_tab._run())
        mp = self._cfg.get("model_path","")
        if not mp or not os.path.exists(mp):
            self._log.append("[!] No valid model path — set model in Settings. Camera will start but detections are disabled.")
        else:
            self._log.append(f"[Start] Model: {os.path.basename(mp)}")
        self._running=True; self._log.append("[Start] Camera starting…")
        self.btn_start.setText("■  STOP")
        self.btn_start.setStyleSheet(
            f"#btn_start{{background:{RED};"
            f"color:#FFFFFF;border:none;border-radius:8px;padding:6px 14px;font-size:13px;font-weight:bold;letter-spacing:2px;}}"
            f"#btn_start:hover{{background:#A01020;color:#FFFFFF;}}"
        )
        self._cam=CameraThread(self._cfg.get("camera_index"))
        self._cam.frame_ready.connect(self._on_frame,Qt.ConnectionType.QueuedConnection)
        self._cam.error.connect(lambda e:(self._log.append(e),self._stop()))
        self._cam.start()
        self._ai=InferenceThread(self._cfg,self._cam,self._pipe_filters,self._pipe_rois)

        self._ai.result_ready.connect(self._on_result,Qt.ConnectionType.QueuedConnection)
        self._ai.log.connect(self._log.append)  # FastLog buffers these automatically
        self._ai.start()
        self.banner.set_setup() if not self._golden else self.banner.reset()
        self._update_inspect_btn()

    def _stop(self):
        self._running=False
        if self._cam: self._cam.stop(); self._cam=None
        if self._ai: self._ai.stop(); self._ai=None
        self.btn_start.setText("START INSPECTION")
        self.btn_start.setStyleSheet(
            f"#btn_start{{background:{CYAN};"
            f"color:#FFFFFF;border:none;border-radius:8px;padding:6px 14px;font-size:13px;font-weight:bold;letter-spacing:2px;}}"
            f"#btn_start:hover{{background:#1347A0;color:#FFFFFF;}}"
        )
        self.video.clear(); self.banner.reset(); self._update_inspect_btn()

    @Slot(QImage)
    def _on_frame(self,qimg):
        if not self._running: return
        if not self.isVisible(): return
        # Cache as BGR numpy for direct inference + OpenCV annotation
        if HAS_CV2 and HAS_NP:
            try:
                fmt = qimg.convertToFormat(QImage.Format.Format_RGB888)
                w, h = fmt.width(), fmt.height()
                ptr = fmt.bits()
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3))
                self._last_frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception:
                self._last_frame_bgr = None
        self.video.set_frame(qimg)

    @Slot(list,float)

    def _on_result(self,dets,lat):
        """
        Display-only per frame. Counters NOT incremented here.
        Called by InferenceThread with background AI results.
        """
        self._latest_dets = dets

        # FPS tracking — deque auto-evicts; no per-frame list rebuild
        now=time.time()
        if not hasattr(self,'_fps_buf'): self._fps_buf=deque()
        self._fps_buf.append(now)
        while self._fps_buf and now-self._fps_buf[0]>2.0: self._fps_buf.popleft()
        fps=len(self._fps_buf)/2.0
        if hasattr(self,'_lat_lbl'):
            col=GREEN if lat<80 else (AMBER if lat<200 else RED)
            self._lat_lbl.setText(f"{lat:.0f}ms | {fps:.1f}fps")
            # Only call setStyleSheet when colour changes — avoids Qt style recalc at 8fps
            if not hasattr(self,'_lat_col') or self._lat_col!=col:
                self._lat_col=col
                self._lat_lbl.setStyleSheet(f"#lat_lbl{{color:{col};font-size:10px;font-family:Consolas;border:none;}}")
        if not self.isVisible(): return
        thresh=self._conf.value()/100
        live=[d for d in dets if d.get("conf",1)>=thresh]
        if hasattr(self,'_det_count_lbl'):
            self._det_count_lbl.setText(f"{len(live)} det")
        # ── Annotate frame directly with OpenCV (mirrors aoi_tab / OfflineAOIThread) ──
        if HAS_CV2 and HAS_NP and hasattr(self, '_last_frame_bgr') and self._last_frame_bgr is not None:
            try:
                ann = self._last_frame_bgr.copy()
                fH, fW = ann.shape[:2]
                for i, g in enumerate(self._golden):
                    x1,y1,x2,y2 = [int(v) for v in g["xyxy"]]
                    cv2.rectangle(ann, (x1,y1), (x2,y2), (30,65,30), 1)
                for d in live:
                    x1,y1,x2,y2 = [int(v) for v in d["xyxy"]]
                    cf = d.get("conf",1.0)
                    col = (40,220,40) if cf>=0.75 else (40,165,255) if cf>=0.50 else (80,80,220)
                    cv2.rectangle(ann, (x1,y1), (x2,y2), col, 2)
                    tag = f"{d['name']} {cf:.2f}"
                    (tw,th),_ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
                    ty = max(y1-4, 12)
                    cv2.rectangle(ann, (x1,ty-th-2), (x1+tw+4,ty+2), (0,0,0), -1)
                    cv2.putText(ann, tag, (x1+2,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
                if not self._pipe_active and self._golden:
                    missing_idx=match_detections(self._golden,live,self._cfg.get("match_iou"),self._persist,self._cfg.get("persistence"))
                    missing_names=[self._golden[i]["name"] for i in missing_idx if i<len(self._golden)]
                    for i in missing_idx:
                        if i>=len(self._golden): continue
                        x1,y1,x2,y2=[int(v) for v in self._golden[i]["xyxy"]]
                        cx,cy=(x1+x2)//2,(y1+y2)//2
                        r=max(7,min(x2-x1,y2-y1)//3)
                        cv2.line(ann,(cx-r,cy-r),(cx+r,cy+r),(60,60,230),2)
                        cv2.line(ann,(cx+r,cy-r),(cx-r,cy+r),(60,60,230),2)
                else:
                    missing_idx=[]; missing_names=[]
                rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                annotated_qimg = QImage(rgb.data, fW, fH, fW*3, QImage.Format.Format_RGB888).copy()
                self.video.set_frame(annotated_qimg)
                self.video.set_detections(live, self._golden, missing_idx)
                if missing_names != self._last_missing:
                    self._last_missing = missing_names
                    if missing_names:
                        cnt = len(missing_names)
                        self.banner.set_fail(f"MISSING ×{cnt}: {missing_names[0]}" + ("…" if cnt>1 else ""))
                    else:
                        self.banner.set_pass()
                return
            except Exception:
                pass  # fall through to original path on any error
        if not self._pipe_active and self._golden:
            missing_idx=match_detections(self._golden,live,self._cfg.get("match_iou"),self._persist,self._cfg.get("persistence"))
            missing_names=[self._golden[i]["name"] for i in missing_idx if i<len(self._golden)]
            self.video.set_detections(live,self._golden,missing_idx)
            if missing_names!=self._last_missing:
                self._last_missing=missing_names
                if missing_names:
                    cnt=len(missing_names)
                    self.banner.set_fail(f"MISSING ×{cnt}: {missing_names[0]}" + ("…" if cnt>1 else ""))
                else: self.banner.set_pass()
        else:
            self.video.set_detections(live,self._golden,[])

    def _flush_counters(self):
        """Push counter values to UI at most 2x/sec."""
        if not self._counter_dirty or not self.isVisible(): return
        self._counter_dirty=False
        self._c_total.set_value(self._total)
        self._c_pass.set_value(self._pass)
        self._c_fail.set_value(self._fail)
        if self._total>0:
            pct=int(self._pass/self._total*100); self._ybar.setValue(pct)
            col=GREEN if pct>=90 else (AMBER if pct>=70 else RED)
            if col!=self._last_yield_col:
                self._last_yield_col=col
                self._ybar.setStyleSheet(
                    f"QProgressBar{{background:{BG_CARD};border:1px solid {BG_BORDER2};border-radius:4px;}}"
                    f"QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {col}88,stop:1 {col});border-radius:4px;}}"
                )
                self._ypct.setStyleSheet(
                    f"#ypct{{color:{col};font-size:13px;font-weight:bold;font-family:'Consolas';letter-spacing:2px;border:none;}}"
                )
            self._ypct.setText(f"YIELD  {pct}%  ({self._pass}/{self._total})")
            self._ypct.setStyleSheet(
                f"#ypct{{color:{col};font-size:15px;font-weight:bold;"
                f"font-family:'Consolas';letter-spacing:2px;border:none;}}"
            )

    def _update_counters(self): self._counter_dirty=True  # legacy shim

    def _flag(self):
        frame=self._cam.get_small_bgr() if self._cam else None
        if frame is None: self._log.append("[!] No frame to flag \u2014 start camera first."); return
        if not self._cfg._path: self._log.append("[!] No project loaded \u2014 cannot flag."); return
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save directly to golden_boards so it appears immediately in the Train tab
        d=os.path.join(self._cfg._path,"golden_boards")
        os.makedirs(d,exist_ok=True)
        out=os.path.join(d,f"flag_{ts}.jpg")
        if HAS_CV2:
            cv2.imwrite(out,frame)
            self._log.append(f"[{ts}] Flagged \u2192 golden_boards/flag_{ts}.jpg")
            self.flag_saved.emit(d)   # tell GoldenTab to reload
        else:
            self._log.append("[!] OpenCV not available \u2014 cannot save flag image.")

    def _reset(self):
        self._pass=self._fail=self._total=0; self._last_missing=[]
        for k in self._persist: self._persist[k]=0
        self._c_total.set_value(0); self._c_pass.set_value(0); self._c_fail.set_value(0)
        self._ybar.setValue(0); self._ypct.setText("YIELD: -"); self._log.clear()
        self.banner.reset(); self.video.set_roi_results([]); self._counter_dirty=False
        self._result_strip.clear_results(); self._result_times.clear()
        self._throughput_lbl.setText("\u2014 boards/min")

    def _on_cam_idx(self,_): pass  # camera bar removed; index set via Settings

    def _test_cam(self): pass  # camera bar removed

    def get_latest_frame(self): return self._cam.get_small_bgr() if self._cam else None
    def on_project_changed(self, path):
        # Stop camera if running
        if self._running: self._stop()
        # Full UI reset \u2014 counters, log, video, banner, history strip, golden master
        self._reset()
        self._log.clear()
        self._golden=[]; self._persist={}
        self._latest_dets=[]; self._last_missing=[]
        self._gm_lbl.setText("No master loaded")
        self._gm_lbl.setStyleSheet(f"#gmlbl{{color:{TEXT_SEC};font-size:11px;border:none;}}")
        self.video.clear()
        self.banner.reset()
        self._clear_history()
        self._pipe_lbl.setText("No pipeline deployed")
        self._pipe_lbl.setStyleSheet(f"#plbl{{color:{TEXT_SEC};font-size:11px;font-family:Consolas;border:none;}}")
        self._update_inspect_btn()
        # Update project label
        self.set_active_project(path)
        # Load golden master if it exists for this project
        gf = os.path.join(path, self._cfg.get("golden_file"))
        if os.path.exists(gf): self.load_golden(gf)

        # ── Auto-load pipeline and model ──────────────────────────────────
        # This ensures the pipe and model are ready without visiting FILTERS tab.
        saved_pipe = os.path.join(path, "pipeline.json")
        if os.path.exists(saved_pipe):
            try:
                import json
                from ois.filters import FILTER_REGISTRY
                from ois.widgets import ROIZone
                with open(saved_pipe) as f:
                    cfg = json.load(f)

                filters = []
                for fc in cfg.get("filters", []):
                    cls = FILTER_REGISTRY.get(fc.get("name"))
                    if cls:
                        n = cls()
                        n.params = fc.get("params", n.params)
                        n.enabled = fc.get("enabled", True)
                        filters.append(n)

                rois = []
                for rc in cfg.get("rois", []):
                    rz = ROIZone(rc.get("name","Zone"), rc.get("type","yolo"), rc.get("rect",[0,0,60,60]))
                    rz.enabled = rc.get("enabled", True)
                    rois.append(rz)

                mp = self._cfg.get("model_path")
                self.deploy_pipeline(filters, rois, mp)
                self._log.append(f"[Project] Pipeline auto-loaded")
            except Exception as e:
                self._log.append(f"[Project] Pipeline load failed: {e}")
        else:
            # Default fallback: load model alone if no pipeline exists
            mp = self._cfg.get("model_path")
            if mp and os.path.exists(mp):
                self.deploy_pipeline([], [], mp)

    def keyPressEvent(self,e):
        if e.key()==Qt.Key.Key_S: self._upload_master()

    def set_active_project(self,path):
        """Update the project name label in the Run tab header."""
        name = os.path.basename(path) if path else "No project"
        if hasattr(self,"_proj_lbl"):
            self._proj_lbl.setText(name)

    def _open_project_manager(self,*_):
        dlg=ProjectDialog(self._cfg,self)
        dlg.project_opened.connect(lambda p:(self.set_active_project(p),self.project_selected.emit(p)))
        dlg.exec()