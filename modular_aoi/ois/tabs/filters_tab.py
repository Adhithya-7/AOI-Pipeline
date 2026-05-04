"""
filters_tab.py — Filter pipeline management tab (LogicTab).
"""
import os, sys, json, time, glob, shutil
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
    build_golden_dataset, OG_MODEL, calc_iou, match_detections
)
from ..threads import CameraThread
from ..filters import (
    FilterNode, FILTER_REGISTRY, apply_filters, AutoCalibrateWorker, run_roi
)
from ..widgets import (
    FastLog, ROICanvas, ROIZone, StatCard, ToastManager, VideoWidget,
    ZoomableImageView
)

if HAS_CV2:
    import cv2
if HAS_NP:
    import numpy as np


class LogicTab(QWidget):
    pipeline_deployed=Signal(list,list,str)
    frame_requested=Signal()   # grab-frame only — does NOT deploy or switch tabs
    def __init__(self,cfg):
        super().__init__(); self._cfg=cfg; self._filters=[]; self._rois=[]; self._src_frame=None
        self._dirty=False
        self._preview_timer=QTimer(self); self._preview_timer.setSingleShot(True); self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._preview)
        self._save_timer=QTimer(self); self._save_timer.setSingleShot(True); self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._autosave_pipeline)
        self._build()

    def _build(self):
        root=QHBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(12)
        # ── Left: filter pipeline ──────────────────────────────────────────
        lf=make_card("lf_logic"); ll=QVBoxLayout(lf); ll.setContentsMargins(10,10,10,10); ll.setSpacing(8)
        ll.addWidget(sec_lbl("FILTER PIPELINE"))
        # ── Auto Calibrate button ──────────────────────────────────────────
        b_acal=QPushButton("⚡  AUTO CALIBRATE"); b_acal.setObjectName("b_acal"); b_acal.setFixedHeight(36)
        b_acal.setToolTip("Analyse the loaded image and automatically choose the best filter pipeline")
        b_acal.setStyleSheet(
            f"#b_acal{{background:#1A1A2E;color:#A78BFA;border:1px solid #6D28D9;"
            f"border-radius:7px;font-weight:bold;font-size:11px;font-family:'Consolas';letter-spacing:1px;}}"
            f"#b_acal:hover{{background:#2D1B69;border-color:#A78BFA;color:#C4B5FD;}}"
            f"#b_acal:disabled{{color:{TEXT_DIM};border-color:{BG_BORDER};background:{BG_DEEP};}}"
        )
        b_acal.clicked.connect(self._auto_calibrate); ll.addWidget(b_acal)
        ll.addWidget(make_sep())
        ar=QHBoxLayout(); ar.addWidget(QLabel("Add:"))
        self._filter_add=QComboBox()
        for n in FILTER_REGISTRY: self._filter_add.addItem(n)
        ar.addWidget(self._filter_add,1)
        b_add=QPushButton("+"); b_add.setFixedSize(32,28); b_add.clicked.connect(self._add_filter)
        ar.addWidget(b_add); ll.addLayout(ar)
        self._filter_list=QListWidget(); self._filter_list.setObjectName("flist")
        self._filter_list.setStyleSheet(f"#flist{{background:{BG_PANEL};border:1px solid {BG_BORDER};border-radius:6px;}}")
        self._filter_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # FIX 6: NoFocus
        self._filter_list.currentRowChanged.connect(self._on_filter_selected)
        self._filter_list.setMaximumHeight(180); ll.addWidget(self._filter_list)
        fr=QHBoxLayout()
        b_up=QPushButton("↑"); b_up.setFixedWidth(36); b_up.clicked.connect(self._filter_up)
        b_dn=QPushButton("↓"); b_dn.setFixedWidth(36); b_dn.clicked.connect(self._filter_dn)
        b_rm=QPushButton("Remove"); b_rm.clicked.connect(self._filter_rm)
        b_tog=QPushButton("Toggle"); b_tog.clicked.connect(self._filter_toggle)
        fr.addWidget(b_up); fr.addWidget(b_dn); fr.addWidget(b_rm); fr.addWidget(b_tog)
        ll.addLayout(fr)
        self._param_card=QFrame(); self._param_card.setObjectName("param_card")
        self._param_card.setStyleSheet(f"#param_card{{background:{BG_PANEL};border:1px solid {BG_BORDER};border-radius:6px;}}")
        self._param_layout=QVBoxLayout(self._param_card); self._param_layout.setContentsMargins(8,8,8,8); self._param_layout.setSpacing(6)
        self._param_layout.addWidget(QLabel("Select a filter to edit parameters."))
        ll.addWidget(self._param_card)
        # ROI zone widgets kept as non-layout attributes so internal logic still works
        self._roi_list=QListWidget(); self._roi_list.setObjectName("rlist")
        self._roi_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._roi_list.currentRowChanged.connect(self._on_roi_sel)
        self._roi_type=QComboBox()
        for t in ["yolo","ocr","barcode"]: self._roi_type.addItem(t)
        self._roi_type.currentTextChanged.connect(self._on_roi_type_change)
        b_frame=QPushButton("Grab Frame from Camera"); b_frame.setFixedHeight(28); b_frame.clicked.connect(self._grab_frame)
        ll.addWidget(b_frame)
        b_dep=QPushButton("▶  DEPLOY PIPELINE"); b_dep.setObjectName("b_dep_logic"); b_dep.setFixedHeight(44)
        b_dep.setStyleSheet(
            f"#b_dep_logic{{background:{GREEN_DIM};"
            f"color:{GREEN};border:1px solid {GREEN_DIM};border-radius:8px;"
            f"font-weight:bold;font-size:13px;letter-spacing:2px;font-family:'Consolas';}}"
            f"#b_dep_logic:hover{{background:{GREEN_DIM};border-color:{GREEN};}}"
        )
        b_dep.clicked.connect(self._deploy); ll.addWidget(b_dep)
        sv=QHBoxLayout()
        b_save=QPushButton("Save Pipeline"); b_save.setFixedHeight(30); b_save.clicked.connect(self._save_pipeline_now)
        self._save_lbl=QLabel("No project loaded")
        self._save_lbl.setStyleSheet(f"color:{TEXT_SEC};font-size:10px;border:none;")
        sv.addWidget(b_save); sv.addWidget(self._save_lbl,1); ll.addLayout(sv)
        self._dep_lbl=QLabel("Not deployed"); self._dep_lbl.setObjectName("dlog_lbl")
        self._dep_lbl.setStyleSheet(f"#dlog_lbl{{color:{TEXT_SEC};font-size:11px;border:none;}}"); ll.addWidget(self._dep_lbl)
        ll.addStretch(); root.addWidget(lf,1)
        # ── Right: preview ─────────────────────────────────────────────────
        rf=make_card("rf_logic"); rl=QVBoxLayout(rf); rl.setContentsMargins(10,10,10,10); rl.setSpacing(8)
        rl.addWidget(sec_lbl("LIVE PREVIEW"))
        # Dual side-by-side view (before / after) ──────────────────────────
        dual=QHBoxLayout(); dual.setSpacing(6); dual.setContentsMargins(0,0,0,0)
        # Left: ROI canvas (original + zone drawing)
        bv_wrap=QVBoxLayout(); bv_wrap.setSpacing(2); bv_wrap.setContentsMargins(0,0,0,0)
        bv_hdr=QLabel("◈  ORIGINAL  (source + ROI zones)")
        bv_hdr.setStyleSheet(
            f"color:{CYAN};font-size:9px;font-family:Consolas;font-weight:bold;"
            f"letter-spacing:2px;border:none;border-bottom:1px solid {CYAN_DIM};"
            f"padding:0 2px 3px 2px;background:transparent;")
        bv_wrap.addWidget(bv_hdr)
        self._roi_canvas=ROICanvas()
        self._roi_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        self._roi_canvas.set_default_type(self._roi_type.currentText())
        self._roi_canvas.roi_added.connect(self._on_roi_added)
        self._roi_canvas.roi_modified.connect(self._on_roi_modified)
        self._roi_canvas.roi_removed.connect(self._on_roi_removed)
        bv_wrap.addWidget(self._roi_canvas,1)
        dual.addLayout(bv_wrap,1)
        # Right: ZoomableImageView (filtered output)
        av_wrap=QVBoxLayout(); av_wrap.setSpacing(2); av_wrap.setContentsMargins(0,0,0,0)
        av_hdr=QLabel("◈  FILTERED  (pipeline output)")
        av_hdr.setStyleSheet(
            f"color:{PURPLE};font-size:9px;font-family:Consolas;font-weight:bold;"
            f"letter-spacing:2px;border:none;border-bottom:1px solid {PURPLE}44;"
            f"padding:0 2px 3px 2px;")
        av_wrap.addWidget(av_hdr)
        self._prev_view=ZoomableImageView()
        self._prev_view._idle_text="FILTERED OUTPUT\n\nAdd filters and click\nPreview Filters"
        av_wrap.addWidget(self._prev_view,1)
        dual.addLayout(av_wrap,1)
        rl.addLayout(dual,1)
        # Link zoom peers so wheel/pan on either view syncs the other
        self._roi_canvas._peer_ziv = self._prev_view
        self._prev_view._canvas_peer = self._roi_canvas
        # Controls bar
        br=QHBoxLayout(); br.setContentsMargins(0,0,0,0); br.setSpacing(4)
        b_img=QPushButton("Load Image"); b_img.setFixedHeight(28); b_img.clicked.connect(self._load_img)
        b_prev_btn=QPushButton("Preview Filters"); b_prev_btn.setFixedHeight(28); b_prev_btn.clicked.connect(self._preview)
        b_fit=QPushButton("Fit"); b_fit.setFixedHeight(28); b_fit.clicked.connect(self._fit_both)
        br.addWidget(b_img); br.addWidget(b_prev_btn); br.addWidget(b_fit); br.addStretch()
        rl.addLayout(br)
        root.addWidget(rf,2)

    def _auto_calibrate(self):
        if self._src_frame is None:
            ToastManager.show(self.window(), "Load an image first (Load Image or Grab Frame)", "warning")
            return
        # Disconnect & discard any previous worker to prevent double-callback
        old = getattr(self, "_cal_worker", None)
        if old is not None:
            try:
                old.done.disconnect(self._on_calibrate_done)
            except RuntimeError:
                pass   # already disconnected
        # Flip button into loading state
        btn = self.findChild(QPushButton, "b_acal")
        if btn:
            btn.setEnabled(False)
            btn.setText("⏳  CALIBRATING…")
        model_path = self._cfg.get("model_path") or ""
        self._cal_worker = AutoCalibrateWorker(self._src_frame, model_path=model_path)
        self._cal_worker.done.connect(self._on_calibrate_done)
        self._cal_worker.start()

    def _on_calibrate_done(self, pipeline, desc):
        btn = self.findChild(QPushButton, "b_acal")
        if btn:
            btn.setEnabled(True)
            btn.setText("⚡  AUTO CALIBRATE")
        if pipeline is None:
            # True failure — model error, no frame, or exception
            ToastManager.show(self.window(), f"Auto-calibrate failed: {desc}", "error")
            return
        # pipeline may legitimately be [] (Raw / no-filter winner)
        self._filters = list(pipeline)
        self._refresh_filter_list()
        self._filter_list.setCurrentRow(0)
        self._mark_pipeline_dirty()
        self._queue_live_preview()
        if pipeline:
            names = " → ".join(f.name for f in pipeline)
            label = f"✔ Pipeline set: {names}"
        else:
            label = "✔ Pipeline set: Raw (no filters — image already optimal)"
        ToastManager.show(self.window(), label, "success", 5000)

    def _add_filter(self):
        name=self._filter_add.currentText(); cls=FILTER_REGISTRY.get(name)
        if cls:
            f = cls()
            self._filters.append(f); self._refresh_filter_list()
            self._filter_list.setCurrentRow(len(self._filters)-1)
            self._mark_pipeline_dirty(); self._queue_live_preview()

    def _refresh_filter_list(self):
        cur=self._filter_list.currentRow()
        self._filter_list.clear()
        for f in self._filters:
            col=CYAN if f.enabled else TEXT_DIM
            item=QListWidgetItem(f"{'ON' if f.enabled else 'OFF'}  {f.name}")
            item.setForeground(QColor(col)); self._filter_list.addItem(item)
        if self._filters:
            self._filter_list.setCurrentRow(min(max(cur,0),len(self._filters)-1))
        else:
            self._on_filter_selected(-1)

    def _filter_up(self):
        i=self._filter_list.currentRow()
        if i>0:
            self._filters[i-1],self._filters[i]=self._filters[i],self._filters[i-1]
            self._refresh_filter_list(); self._filter_list.setCurrentRow(i-1)
            self._mark_pipeline_dirty(); self._queue_live_preview()

    def _filter_dn(self):
        i=self._filter_list.currentRow()
        if 0<=i<len(self._filters)-1:
            self._filters[i],self._filters[i+1]=self._filters[i+1],self._filters[i]
            self._refresh_filter_list(); self._filter_list.setCurrentRow(i+1)
            self._mark_pipeline_dirty(); self._queue_live_preview()

    def _filter_rm(self):
        i=self._filter_list.currentRow()
        if 0<=i<len(self._filters):
            self._filters.pop(i); self._refresh_filter_list()
            self._mark_pipeline_dirty(); self._queue_live_preview()

    def _filter_toggle(self):
        i=self._filter_list.currentRow()
        if 0<=i<len(self._filters):
            self._filters[i].enabled=not self._filters[i].enabled; self._refresh_filter_list()
            self._mark_pipeline_dirty(); self._queue_live_preview()

    def _on_roi_added(self,roi):
        item=QListWidgetItem(f"[{roi.zone_type.upper()}] {roi.name}")
        item.setForeground(QColor(CYAN)); self._roi_list.addItem(item); self._rois.append(roi)
        self._roi_list.setCurrentRow(len(self._rois)-1)
        self._mark_pipeline_dirty()

    def _on_roi_modified(self,roi):
        self._refresh_roi_list(); self._mark_pipeline_dirty()

    def _on_roi_removed(self,roi):
        """Called when user right-clicks to delete a zone directly on the canvas."""
        if roi in self._rois:
            self._rois.remove(roi)
        self._refresh_roi_list()
        self._mark_pipeline_dirty()

    def _refresh_roi_list(self):
        self._roi_list.clear()
        for roi in self._rois:
            item=QListWidgetItem(f"[{roi.zone_type.upper()}] {roi.name}")
            col={"ocr":QColor(GREEN),"barcode":QColor(PURPLE)}.get(roi.zone_type,QColor(CYAN))
            item.setForeground(col); self._roi_list.addItem(item)

    def _on_roi_sel(self,i):
        if 0<=i<len(self._rois):
            roi=self._rois[i]
            idx=self._roi_type.findText(roi.zone_type)
            if idx>=0:
                self._roi_type.blockSignals(True); self._roi_type.setCurrentIndex(idx); self._roi_type.blockSignals(False)

    def _on_roi_type_change(self,t):
        self._roi_canvas.set_default_type(t)
        i=self._roi_list.currentRow()
        if 0<=i<len(self._rois):
            self._rois[i].zone_type=t; self._roi_canvas.set_selected_type(t); self._refresh_roi_list()
            self._mark_pipeline_dirty()

    def _roi_rm(self):
        i=self._roi_list.currentRow()
        if 0<=i<len(self._rois):
            roi=self._rois.pop(i)
            if roi in self._roi_canvas.rois: self._roi_canvas.rois.remove(roi)
            self._roi_canvas.update(); self._refresh_roi_list()
            self._mark_pipeline_dirty()

    def _grab_frame(self):
        # Emit a lightweight signal — MainWindow handles the frame fetch.
        # Does NOT emit pipeline_deployed, so no deploy and no tab switch.
        self.frame_requested.emit()

    def set_src_frame(self,bgr):
        if bgr is None: return
        self._src_frame=bgr; self._roi_canvas.load_frame(bgr); self._queue_live_preview()

    def _load_img(self):
        p,_=QFileDialog.getOpenFileName(self,"Reference image","","Images (*.jpg *.png *.jpeg)")
        if p: self._roi_canvas.load_image(p)
        if p and HAS_CV2: self._src_frame=cv2.imread(p)
        self._queue_live_preview()

    def _preview(self):
        if self._src_frame is None:
            self._prev_view._idle_text="No source frame. Load an image first."
            self._prev_view.clear(); return
        out=apply_filters(self._src_frame,self._filters)
        if HAS_CV2:
            rgb=cv2.cvtColor(out,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]
            qimg=QImage(rgb.tobytes(),w,h,w*3,QImage.Format.Format_RGB888).copy()
            self._prev_view.load_pixmap(QPixmap.fromImage(qimg))

    def _fit_both(self):
        """Fit both the ROI canvas and the filtered preview to their respective containers."""
        self._roi_canvas._fit()
        self._prev_view._fit()

    def _deploy(self):
        mp=self._cfg.get("model_path") or ""
        self._dep_lbl.setText(f"Deployed: {len(self._filters)} filters | {len(self._rois)} ROI zones")
        self._dep_lbl.setStyleSheet(f"#dlog_lbl{{color:{GREEN};font-size:11px;border:none;}}")
        self.pipeline_deployed.emit(list(self._filters),list(self._rois),mp)

    def on_project_changed(self,path):
        saved=os.path.join(path,"pipeline.json")
        self._save_timer.stop(); self._dirty=False
        if not os.path.exists(saved):
            self._filters=[]; self._rois=[]; self._refresh_filter_list(); self._refresh_roi_list()
            self._roi_canvas.rois=[]; self._roi_canvas.update()
            self._update_save_status("No saved pipeline",TEXT_DIM)
            return
        try:
            with open(saved) as f: cfg=json.load(f)
            self._filters=[]
            for fc in cfg.get("filters",[]):
                cls=FILTER_REGISTRY.get(fc.get("name"))
                if cls:
                    n=cls(); n.params=fc.get("params",n.params); n.enabled=fc.get("enabled",True); self._filters.append(n)
            self._refresh_filter_list()
            self._rois=[]
            for rc in cfg.get("rois",[]):
                name=rc.get("name","Zone"); zt=rc.get("type","yolo"); rect=rc.get("rect",[0,0,60,60])
                rz=ROIZone(name,zt,rect); rz.enabled=rc.get("enabled",True); self._rois.append(rz)
            self._refresh_roi_list(); self._roi_canvas.rois=list(self._rois); self._roi_canvas.update()
            self._update_save_status("Loaded pipeline",GREEN)
        except Exception as e:
            self._update_save_status(f"Load failed: {e}",RED)
        self._queue_live_preview()

    def save_pipeline(self,path):
        cfg={"filters":[f.get_config() for f in self._filters],"rois":[r.get_config() for r in self._rois]}
        try:
            with open(os.path.join(path,"pipeline.json"),"w") as f: json.dump(cfg,f,indent=2)
            return True,""
        except Exception as e:
            return False,str(e)

    def _clear_layout_recursive(self,lay):
        while lay.count():
            it=lay.takeAt(0)
            w=it.widget()
            if w is not None:
                w.deleteLater(); continue
            child=it.layout()
            if child is not None: self._clear_layout_recursive(child)

    def _clear_param_layout(self):
        self._clear_layout_recursive(self._param_layout)

    def _add_slider_int(self,title,minv,maxv,cur,on_change):
        row=QHBoxLayout(); row.addWidget(QLabel(title))
        s=QSlider(Qt.Orientation.Horizontal); s.setRange(minv,maxv); s.setValue(int(cur))
        v=QLabel(str(int(cur))); v.setFixedWidth(42); v.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        def _changed(x):
            v.setText(str(int(x))); on_change(int(x)); self._mark_pipeline_dirty(); self._queue_live_preview()
        s.valueChanged.connect(_changed)
        row.addWidget(s,1); row.addWidget(v); self._param_layout.addLayout(row)

    def _add_slider_odd(self,title,minv,maxv,cur,on_change):
        vals=list(range(minv,maxv+1,2))
        cur=int(cur); cur=cur if cur%2==1 else cur+1
        cur=max(minv,min(maxv,cur)); idx=max(0,min(len(vals)-1,(cur-minv)//2))
        row=QHBoxLayout(); row.addWidget(QLabel(title))
        s=QSlider(Qt.Orientation.Horizontal); s.setRange(0,len(vals)-1); s.setValue(idx)
        v=QLabel(str(vals[idx])); v.setFixedWidth(42); v.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        def _changed(i):
            val=vals[int(i)]; v.setText(str(val)); on_change(val); self._mark_pipeline_dirty(); self._queue_live_preview()
        s.valueChanged.connect(_changed)
        row.addWidget(s,1); row.addWidget(v); self._param_layout.addLayout(row)

    def _add_slider_tenths(self,title,minv,maxv,cur,on_change):
        iv=int(round(float(cur)*10))
        row=QHBoxLayout(); row.addWidget(QLabel(title))
        s=QSlider(Qt.Orientation.Horizontal); s.setRange(int(minv*10),int(maxv*10)); s.setValue(iv)
        v=QLabel(f"{iv/10:.1f}"); v.setFixedWidth(42); v.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        def _changed(x):
            val=float(x)/10.0; v.setText(f"{val:.1f}"); on_change(val); self._mark_pipeline_dirty(); self._queue_live_preview()
        s.valueChanged.connect(_changed)
        row.addWidget(s,1); row.addWidget(v); self._param_layout.addLayout(row)

    def _on_filter_selected(self,idx):
        self._clear_param_layout()
        if idx<0 or idx>=len(self._filters):
            self._param_layout.addWidget(QLabel("Select a filter to edit parameters.")); return
        f=self._filters[idx]
        head=QLabel(f"Editing: {f.name}"); head.setStyleSheet(f"color:{CYAN};font-weight:bold;border:none;")
        self._param_layout.addWidget(head)
        cb=QCheckBox("Enabled"); cb.setChecked(f.enabled)
        def _toggle(st):
            f.enabled=bool(st); self._refresh_filter_list(); self._mark_pipeline_dirty(); self._queue_live_preview()
        cb.stateChanged.connect(_toggle); self._param_layout.addWidget(cb)
        if f.name=="CLAHE":
            self._add_slider_tenths("Clip Limit",1.0,10.0,f.params.get("clip_limit",2.0),lambda v: f.params.update({"clip_limit":v}))
            self._add_slider_int("Tile Size",2,32,f.params.get("tile_size",8),lambda v: f.params.update({"tile_size":v}))
        elif f.name=="Gaussian Blur":
            self._add_slider_odd("Kernel",1,31,f.params.get("kernel",5),lambda v: f.params.update({"kernel":v}))
        elif f.name=="Canny Edge":
            self._add_slider_int("Threshold 1",0,255,f.params.get("thresh1",50),lambda v: f.params.update({"thresh1":v}))
            self._add_slider_int("Threshold 2",0,255,f.params.get("thresh2",150),lambda v: f.params.update({"thresh2":v}))
        elif f.name=="Adaptive Thresh":
            self._add_slider_odd("Block Size",3,51,f.params.get("block_size",11),lambda v: f.params.update({"block_size":v}))
            self._add_slider_int("C",-20,20,f.params.get("c",2),lambda v: f.params.update({"c":v}))
        elif f.name=="Median Blur":
            self._add_slider_odd("Kernel",1,31,f.params.get("kernel",5),lambda v: f.params.update({"kernel":v}))
        elif f.name=="Bilateral Denoise":
            self._add_slider_int("Diameter",1,21,f.params.get("d",9),lambda v: f.params.update({"d":v}))
            self._add_slider_int("Sigma Color",10,200,f.params.get("sigma_color",60),lambda v: f.params.update({"sigma_color":v}))
            self._add_slider_int("Sigma Space",10,200,f.params.get("sigma_space",60),lambda v: f.params.update({"sigma_space":v}))
        elif f.name=="Brightness/Contrast":
            self._add_slider_tenths("Alpha",0.5,2.0,f.params.get("alpha",1.2),lambda v: f.params.update({"alpha":v}))
            self._add_slider_int("Beta",-100,100,f.params.get("beta",0),lambda v: f.params.update({"beta":v}))
        elif f.name=="Gamma":
            self._add_slider_tenths("Gamma",0.3,2.5,f.params.get("gamma",1.2),lambda v: f.params.update({"gamma":v}))
        elif f.name=="Sharpen":
            self._add_slider_tenths("Amount",0.1,2.0,f.params.get("amount",1.0),lambda v: f.params.update({"amount":v}))
        elif f.name=="Unsharp Mask":
            self._add_slider_tenths("Sigma",0.5,5.0,f.params.get("sigma",2.0),lambda v: f.params.update({"sigma":v}))
            self._add_slider_tenths("Amount",0.5,3.0,f.params.get("amount",1.4),lambda v: f.params.update({"amount":v}))
        elif f.name=="Morph Close":
            self._add_slider_odd("Kernel",3,31,f.params.get("kernel",5),lambda v: f.params.update({"kernel":v}))
            self._add_slider_int("Iterations",1,5,f.params.get("iters",1),lambda v: f.params.update({"iters":v}))
        self._param_layout.addStretch()

    def _queue_live_preview(self):
        if self._src_frame is not None: self._preview_timer.start()

    def _update_save_status(self,text,color):
        self._save_lbl.setText(text)
        self._save_lbl.setStyleSheet(f"color:{color};font-size:10px;border:none;")

    def _mark_pipeline_dirty(self):
        self._dirty=True
        self._update_save_status("Unsaved changes",AMBER)
        if self._cfg._path: self._save_timer.start()

    def _autosave_pipeline(self):
        if self._dirty and self._cfg._path: self._save_pipeline_now()

    def _save_pipeline_now(self,*_):
        path=self._cfg._path
        if not path:
            self._update_save_status("No project loaded",TEXT_DIM); return
        ok,err=self.save_pipeline(path)
        if ok:
            self._dirty=False
            self._update_save_status(f"Saved {datetime.now().strftime('%H:%M:%S')}",GREEN)
        else:
            self._update_save_status(f"Save failed: {err}",RED)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — SYSTEM LOG
# ─────────────────────────────────────────────────────────────────────────────
class SysLogTab(QWidget):
    def __init__(self):
        super().__init__(); self._build()
    def _build(self):
        lay=QVBoxLayout(self); lay.setContentsMargins(10,10,10,10); lay.setSpacing(8)
        hr=QHBoxLayout(); hr.addWidget(sec_lbl("SYSTEM LOG")); hr.addStretch()
        b_clr=QPushButton("Clear"); b_clr.setFixedHeight(28); b_clr.setFixedWidth(70); b_clr.clicked.connect(self._clear)
        b_exp=QPushButton("Export"); b_exp.setFixedHeight(28); b_exp.setFixedWidth(70); b_exp.clicked.connect(self._export)
        hr.addWidget(b_clr); hr.addWidget(b_exp); lay.addLayout(hr)
        self._log=FastLog(); self._log.setObjectName("syslog")
        self._log.setStyleSheet(
            f"#syslog{{background:{BG_DEEP};border:1px solid {BG_BORDER2};border-radius:8px;"
            f"padding:8px;color:{TEXT_SEC};font-family:'Consolas';font-size:11px;line-height:1.5;}}"
        )
        lay.addWidget(self._log,1)
        sf=make_card("sf_sys"); sf.setFixedHeight(48); sl=QHBoxLayout(sf); sl.setContentsMargins(10,6,10,6)
        sl.addWidget(sec_lbl("SYSTEM")); sl.addStretch()
        self._lbl_cpu=QLabel("CPU: -"); self._lbl_mem=QLabel("MEM: -"); self._lbl_gpu=QLabel("GPU: -")
        for l in [self._lbl_cpu,self._lbl_mem,self._lbl_gpu]:
            l.setObjectName("syslbl"); l.setStyleSheet(f"#syslbl{{color:{CYAN};font-family:Consolas;font-size:11px;border:none;}}")
            sl.addWidget(l); sl.addSpacing(16)
        lay.addWidget(sf)
        self._timer=QTimer(self); self._timer.setInterval(3000); self._timer.timeout.connect(self._update_sys); self._timer.start()
        self._update_sys()
    def _update_sys(self):
        # Lazy-import and cache once — bare `import` + `nvmlInit` every 3 s is expensive
        if not hasattr(self, "_ps"):
            try: import psutil as _m; self._ps = _m
            except: self._ps = None
        if not hasattr(self, "_nv"):
            try:
                import pynvml as _m; _m.nvmlInit()
                self._nv = _m
                self._nv_handle = _m.nvmlDeviceGetHandleByIndex(0)
            except:
                self._nv = None; self._nv_handle = None
        try:
            if self._ps:
                self._lbl_cpu.setText(f"CPU: {self._ps.cpu_percent():.1f}%")
                vm = self._ps.virtual_memory()
                self._lbl_mem.setText(f"MEM: {vm.percent:.1f}% ({vm.used//1024//1024}MB)")
            else:
                self._lbl_cpu.setText("CPU: n/a"); self._lbl_mem.setText("MEM: n/a")
        except: self._lbl_cpu.setText("CPU: n/a"); self._lbl_mem.setText("MEM: n/a")
        try:
            if self._nv and self._nv_handle:
                u = self._nv.nvmlDeviceGetUtilizationRates(self._nv_handle)
                m = self._nv.nvmlDeviceGetMemoryInfo(self._nv_handle)
                self._lbl_gpu.setText(f"GPU: {u.gpu}% VRAM: {m.used//1024//1024}/{m.total//1024//1024}MB")
            else:
                self._lbl_gpu.setText("GPU: n/a")
        except: self._lbl_gpu.setText("GPU: n/a")
    def append(self,msg): self._log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    def _clear(self): self._log.clear()
    def _export(self):
        p,_=QFileDialog.getSaveFileName(self,"Export Log","ois_log.txt","Text (*.txt)")
        if p:
            try:
                with open(p,"w",encoding="utf-8") as f: f.write(self._log.toPlainText())
            except Exception as e: QMessageBox.critical(self,"Error",str(e))



