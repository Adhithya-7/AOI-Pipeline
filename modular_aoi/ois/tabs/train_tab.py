"""
train_tab.py — Golden board management & training tab (GoldenTab).
"""
import glob
import json
import os
import platform
import shutil
import sys
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
    build_golden_dataset, OG_MODEL, calc_iou, find_best_pt, match_detections,
    load_optimized_yolo
)
from ..threads import ThumbThread, TrainingThread, AugThread, AutoLabelThread
from ..filters import apply_filters
from ..widgets import (
    AdvancedTrainDialog, FastLog, LabelCanvas, StatCard, ToastManager
)

if HAS_CV2:
    import cv2
if HAS_NP:
    import numpy as np
if HAS_YOLO:
    from ultralytics import YOLO as _YOLO


class GoldenTab(QWidget):
    def __init__(self,cfg):
        super().__init__(); self._cfg=cfg; self._project=None
        self._images=[]; self._cur=0; self._label_cache={}
        self._aug_thread=self._train_thread=self._auto_thread=self._thumb_thread=None
        self._trained_model=None; self._cached_model=None; self._cached_model_path=None
        self._aug_params=dict(DEFAULT_AUG); self._thumb_buf=[]; self._thumb_flush_pending=False
        self._thumb_generation=0; self._thumb_loaded=set()
        self._px_cache={}
        # FIX 5: throttle training label text — 250ms debounce prevents setText storm
        self._tr_msg_pending=None; self._tr_timer=QTimer(self)
        self._tr_timer.setSingleShot(True); self._tr_timer.setInterval(250)
        self._tr_timer.timeout.connect(self._flush_tr_msg)
        self._build()

    def _build(self):
        root=QHBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(12)
        lf=make_card("lf_panel"); ll=QVBoxLayout(lf); ll.setContentsMargins(10,10,10,10); ll.setSpacing(8)
        br=QHBoxLayout()
        b_up=QPushButton("Upload"); b_up.setObjectName("b_up"); b_up.setFixedHeight(32)
        b_up.setStyleSheet(f"#b_up{{background:{CYAN_DIM};color:{CYAN};border:1px solid {CYAN}44;border-radius:5px;}} #b_up:hover{{background:{CYAN};color:#FFFFFF;}}")
        b_up.clicked.connect(self._upload)
        b_fo=QPushButton("Folder"); b_fo.setFixedHeight(32); b_fo.clicked.connect(self._open_folder)
        b_del=QPushButton("Delete"); b_del.setObjectName("b_del"); b_del.setFixedHeight(32)
        b_del.setStyleSheet(
            f"#b_del{{background:{BG_CARD};color:{RED};border:1px solid {RED_DIM};"
            f"border-radius:5px;font-weight:600;}}"
            f"#b_del:hover{{background:{RED};color:#FFFFFF;border-color:{RED};}}"
            f"#b_del:disabled{{color:{TEXT_DIM};border-color:{BG_BORDER};}}"
        )
        b_del.setToolTip("Delete current image and its label from disk (Del)")
        b_del.clicked.connect(self._delete_current)
        self._b_del = b_del   # keep ref to enable/disable
        br.addWidget(b_up); br.addWidget(b_fo); br.addWidget(b_del); ll.addLayout(br)
        self._thumbs=QListWidget(); self._thumbs.setObjectName("thumbs")
        self._thumbs.setStyleSheet(
            f"#thumbs{{background:{BG_PANEL};border:1px solid {BG_BORDER};border-radius:6px;}}"
            f"QListWidget::item{{padding:2px;border-radius:4px;}}"
            f"QListWidget::item:selected{{background:{CYAN_DIM};border:1px solid {CYAN};}}"
            f"QListWidget::item:hover:!selected{{background:{BG_CARD2};}}"
        )
        self._thumbs.setViewMode(QListWidget.ViewMode.IconMode)
        self._thumbs.setIconSize(QSize(100,100))
        self._thumbs.setGridSize(QSize(108,108))   # 3 cols × 108 + spacing/padding fits lf=378
        self._thumbs.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._thumbs.setUniformItemSizes(True)
        self._thumbs.setWordWrap(False)
        self._thumbs.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._thumbs.setSpacing(2)
        self._thumbs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._thumbs.currentRowChanged.connect(self._on_select)
        # Delete key removes the current image without needing to grab focus
        _del_sc = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        _del_sc.activated.connect(self._delete_current)
        ll.addWidget(self._thumbs,1)
        self._stats=QLabel("0 images"); self._stats.setObjectName("stats_lbl")
        self._stats.setStyleSheet(f"#stats_lbl{{color:{TEXT_SEC};font-size:11px;border:none;}}")
        ll.addWidget(self._stats); lf.setFixedWidth(378); root.addWidget(lf)
        cf=make_card("cf_panel"); cl=QVBoxLayout(cf); cl.setContentsMargins(8,8,8,8); cl.setSpacing(6)
        self._canvas=LabelCanvas(); self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        cl.addWidget(self._canvas,1)
        ctrl=QHBoxLayout(); ctrl.setSpacing(4); ctrl.addWidget(QLabel("Class:"))
        self._cls_cb=QComboBox()
        for c in MASTER_CLASS_LIST: self._cls_cb.addItem(c)
        self._cls_cb.currentIndexChanged.connect(self._canvas.set_class); ctrl.addWidget(self._cls_cb,1)
        QCOLS={0:"#FFA500",1:"#FF3333",4:"#FFFF00",8:"#3366FF"}; QLBLS={0:"P",1:"R",4:"C",8:"IC"}
        for i,lbl in QLBLS.items():
            b=QPushButton(lbl); b.setFixedSize(28,28); oid=f"qc_{i}"
            b.setObjectName(oid); b.setStyleSheet(f"#{oid}{{background:{QCOLS[i]};color:black;border-radius:4px;font-size:9px;font-weight:bold;border:none;}}")
            b.clicked.connect(lambda _,idx=i:(self._canvas.set_class(idx),self._cls_cb.setCurrentIndex(idx))); ctrl.addWidget(b)
        # Pan toggle button
        self._btn_pan=QPushButton("✋  Pan"); self._btn_pan.setFixedHeight(30); self._btn_pan.setCheckable(True)
        self._btn_pan.setObjectName("b_pan")
        self._btn_pan.setStyleSheet(
            f"#b_pan{{background:{BG_CARD};color:{TEXT_SEC};border:1px solid {BG_BORDER2};border-radius:5px;padding:0 8px;font-size:10px;}}"
            f"#b_pan:checked{{background:{CYAN_DIM};color:{CYAN};border-color:{CYAN};}}"
            f"#b_pan:hover{{border-color:{CYAN};}}"
        )
        self._btn_pan.setToolTip("Toggle pan mode (hold Space or use middle-mouse to pan without switching)")
        self._btn_pan.toggled.connect(self._canvas.set_pan_mode); ctrl.addWidget(self._btn_pan)
        b_fit=QPushButton("⊡"); b_fit.setFixedSize(30,30); b_fit.setToolTip("Reset zoom/fit")
        b_fit.clicked.connect(self._canvas._fit); ctrl.addWidget(b_fit)
        b_prev=QPushButton("\u2190 Prev"); b_prev.setFixedHeight(30); b_prev.clicked.connect(self._prev)
        self._nav=QLabel("0/0"); self._nav.setObjectName("navlbl"); self._nav.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._nav.setStyleSheet(f"#navlbl{{color:{CYAN};font-family:Consolas;min-width:60px;border:none;}}")
        b_next=QPushButton("Save & Next \u2192"); b_next.setObjectName("b_next"); b_next.setFixedHeight(30)
        b_next.setStyleSheet(f"#b_next{{background:{GREEN_DIM};color:{GREEN};border:1px solid {GREEN_DIM};border-radius:5px;}} #b_next:hover{{background:{GREEN_DIM};}}")
        b_next.clicked.connect(self._next)
        b_clr=QPushButton("Clear"); b_clr.setObjectName("b_clr"); b_clr.setFixedHeight(30)
        b_clr.setStyleSheet(f"#b_clr{{background:{BG_CARD};color:{RED};border:1px solid {RED_DIM};border-radius:5px;}}")
        b_clr.clicked.connect(self._clear_lbl)
        ctrl.addWidget(b_prev); ctrl.addWidget(self._nav); ctrl.addWidget(b_next); ctrl.addWidget(b_clr)
        cl.addLayout(ctrl); root.addWidget(cf,2)
        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300); scroll.setMaximumWidth(360)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus); scroll.setObjectName("rscroll")
        scroll.setStyleSheet("#rscroll{border:none;}")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rw=QWidget(); rw.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        rl=QVBoxLayout(rw); rl.setSpacing(4); rl.setContentsMargins(2,2,2,2)
        rl.addWidget(sec_lbl("AUTO-LABELER"))
        b_al=QPushButton("Auto-Label All"); b_al.setObjectName("b_al"); b_al.setFixedHeight(30)
        b_al.setStyleSheet(f"#b_al{{background:{AMBER_DIM};color:{AMBER};border:1px solid {AMBER_DIM};border-radius:5px;}} #b_al:hover{{background:#9a6600;}}")
        b_al.clicked.connect(self._auto_label); rl.addWidget(b_al)
        self._al_prog=QProgressBar(); self._al_prog.setFixedHeight(4); self._al_prog.setValue(0); rl.addWidget(self._al_prog)
        # FIX: FastLog replaces QTextEdit for auto-label log
        self._al_log=FastLog(); self._al_log.setMaximumHeight(44); self._al_log.setObjectName("allog")
        self._al_log.setStyleSheet(f"#allog{{background:{BG_CARD};border:1px solid {BG_BORDER2};border-radius:4px;padding:3px;color:{AMBER};font-family:Consolas;font-size:10px;}}"); rl.addWidget(self._al_log)
        rl.addWidget(make_sep()); rl.addWidget(sec_lbl("AUGMENTATION"))
        self._aug_cbs={}
        aug_grid=QGridLayout(); aug_grid.setHorizontalSpacing(10); aug_grid.setVerticalSpacing(2)
        for i,(lbl,key) in enumerate([("Brightness x1.4/x0.55","brightness"),("Rotation 90/180/270","rot90"),
                                      ("Flip Horizontal","flip_h"),("Flip Vertical","flip_v"),
                                      ("Gaussian Noise","noise"),("Gaussian Blur","blur")]):
            cb=QCheckBox(lbl); cb.setChecked(True); self._aug_cbs[key]=cb; aug_grid.addWidget(cb,i//2,i%2)
        rl.addLayout(aug_grid)
        b_aug=QPushButton("Run Augmentation"); b_aug.setObjectName("b_aug"); b_aug.setFixedHeight(28)
        b_aug.setStyleSheet(f"#b_aug{{background:{CYAN_DIM};color:{CYAN};border:1px solid {CYAN}44;border-radius:5px;}} #b_aug:hover{{background:{CYAN};color:#FFFFFF;}}")
        b_aug.clicked.connect(self._run_aug); rl.addWidget(b_aug)
        self._aug_lbl=QLabel("Not run"); self._aug_lbl.setObjectName("auglbl")
        self._aug_lbl.setStyleSheet(f"#auglbl{{color:{TEXT_SEC};font-size:10px;border:none;}}"); rl.addWidget(self._aug_lbl)
        rl.addWidget(make_sep()); rl.addWidget(sec_lbl("FINE-TUNE"))
        self._chain_lbl=QLabel("Base: checking..."); self._chain_lbl.setWordWrap(True); self._chain_lbl.setObjectName("chainlbl")
        self._chain_lbl.setStyleSheet(f"#chainlbl{{color:{TEXT_SEC};font-size:10px;font-family:Consolas;background:{BG_CARD};border:1px solid {BG_BORDER2};border-radius:4px;padding:4px;}}"); rl.addWidget(self._chain_lbl)
        self._refresh_chain()
        ep_r=QHBoxLayout(); ep_r.addWidget(QLabel("Epochs:"))
        self._epochs=QSpinBox(); self._epochs.setRange(5,2000); self._epochs.setValue(50); self._epochs.setFixedWidth(70)
        ep_r.addWidget(self._epochs); ep_r.addWidget(QLabel("Batch:"))
        self._batch=QSpinBox(); self._batch.setRange(1,64); self._batch.setValue(4); self._batch.setFixedWidth(55)
        ep_r.addWidget(self._batch); ep_r.addStretch(); rl.addLayout(ep_r)
        ar=QHBoxLayout(); b_adv=QPushButton("Advanced..."); b_adv.setFixedHeight(26); b_adv.clicked.connect(self._open_adv)
        ar.addWidget(b_adv); ar.addStretch(); rl.addLayout(ar)
        b_tr=QPushButton("▶  TRAIN ON GOLDEN BOARDS"); b_tr.setObjectName("b_tr"); b_tr.setFixedHeight(38)
        b_tr.setStyleSheet(
            f"#b_tr{{background:{GREEN_DIM};"
            f"color:{GREEN};border:1px solid {GREEN_DIM};border-radius:8px;font-weight:bold;font-size:12px;letter-spacing:1px;}}"
            f"#b_tr:hover{{background:{GREEN_DIM};border-color:{GREEN};}}"
        )
        b_tr.clicked.connect(self._train); rl.addWidget(b_tr)
        self._tr_prog=QProgressBar(); self._tr_prog.setFixedHeight(8); self._tr_prog.setValue(0); rl.addWidget(self._tr_prog)
        self._tr_lbl=QLabel(""); self._tr_lbl.setObjectName("trlbl"); self._tr_lbl.setWordWrap(True)
        self._tr_lbl.setStyleSheet(f"#trlbl{{color:{CYAN};font-size:10px;font-family:Consolas;border:none;}}"); rl.addWidget(self._tr_lbl)
        rl.addWidget(make_sep()); rl.addWidget(sec_lbl("DEPLOY"))
        b_dep=QPushButton("⬆  DEPLOY TRAINED MODEL"); b_dep.setObjectName("b_dep"); b_dep.setFixedHeight(34)
        b_dep.setStyleSheet(
            f"#b_dep{{background:{AMBER_DIM};"
            f"color:{AMBER};border:1px solid {AMBER_DIM};border-radius:8px;font-weight:bold;font-size:11px;letter-spacing:1px;}}"
            f"#b_dep:hover{{background:{AMBER_DIM};border-color:{AMBER};}}"
        )
        b_dep.clicked.connect(self._deploy); rl.addWidget(b_dep)
        self._dep_lbl=QLabel("No model deployed"); self._dep_lbl.setObjectName("deplbl")
        self._dep_lbl.setStyleSheet(f"#deplbl{{color:{TEXT_SEC};font-size:10px;border:none;}}"); rl.addWidget(self._dep_lbl)
        rl.addStretch(); scroll.setWidget(rw); root.addWidget(scroll)

    def _refresh_chain(self):
        base=self._cfg.get_next_base_model(); og=self._cfg.get("og_model") or ""
        if not base: txt="No base model found"; col=AMBER
        elif og and os.path.abspath(base)==os.path.abspath(og): txt=f"Base: OG\n{os.path.basename(base)}"; col=CYAN
        else: txt=f"Base: Latest\n{os.path.basename(base)}"; col=GREEN
        self._chain_lbl.setText(txt)
        self._chain_lbl.setStyleSheet(f"#chainlbl{{color:{col};font-size:10px;font-family:Consolas;background:{BG_CARD};border:1px solid {BG_BORDER2};border-radius:4px;padding:4px;}}")

    def _open_adv(self):
        self._aug_params["epochs"]=self._epochs.value(); self._aug_params["batch"]=self._batch.value()
        dlg=AdvancedTrainDialog(self._aug_params,self)
        if dlg.exec()==QDialog.DialogCode.Accepted:
            self._aug_params=dlg.get_params()
            self._epochs.setValue(int(self._aug_params.get("epochs",50))); self._batch.setValue(int(self._aug_params.get("batch",4)))
            self._cfg.set("aug_params",self._aug_params)

    def _upload(self):
        if not self._project: QMessageBox.warning(self,"No Project","Load a project first"); return
        files,_=QFileDialog.getOpenFileNames(self,"Upload","","Images (*.jpg *.png *.jpeg)")
        if not files: return
        d=os.path.join(self._project,"golden_boards")
        for f in files: shutil.copy(f,os.path.join(d,os.path.basename(f)))
        self._load_images()

    def _open_folder(self):
        d=os.path.join(self._project,"golden_boards") if self._project else ""
        if not os.path.exists(d): return
        if platform.system()=="Windows": os.startfile(d)
        elif platform.system()=="Darwin": __import__("subprocess").Popen(["open",d])
        else: __import__("subprocess").Popen(["xdg-open",d])

    def _stop_thumb_loader(self, clear_buffer=True):
        self._thumb_generation += 1
        if self._thumb_thread:
            try: self._thumb_thread.thumb_done.disconnect()
            except (RuntimeError, TypeError): pass
            try: self._thumb_thread.all_done.disconnect()
            except (RuntimeError, TypeError): pass
            if self._thumb_thread.isRunning():
                self._thumb_thread.abort()
        self._thumb_thread=None
        if clear_buffer:
            self._thumb_buf=[]; self._thumb_flush_pending=False

    def _start_thumb_loader(self, reset_loaded=False):
        self._stop_thumb_loader()
        if reset_loaded:
            self._thumb_loaded=set()
        pending=list(range(len(self._images))) if reset_loaded else [i for i in range(len(self._images)) if i not in self._thumb_loaded]
        if not pending:
            return
        generation=self._thumb_generation
        self._thumb_thread=ThumbThread([self._images[i] for i in pending],96)
        self._thumb_thread.thumb_done.connect(
            lambda idx,raw,w,h,g=generation,m=pending: self._on_thumb(g,m[idx],raw,w,h)
        )
        self._thumb_thread.all_done.connect(
            lambda g=generation: self._on_thumb_batch_done(g)
        )
        self._thumb_thread.start()

    def _needs_thumb_reload(self):
        return bool(self._images) and len(self._thumb_loaded)<len(self._images)

    def _on_thumb_batch_done(self,generation):
        if generation!=self._thumb_generation:
            return
        self._thumb_thread=None
        self._flush_thumbs()

    def _delete_current(self):
        """Delete the currently selected image (and its label) from disk."""
        if not self._images or self._cur >= len(self._images):
            return
        ip = self._images[self._cur]
        lp = self._lbl_path(ip)

        reply = QMessageBox.question(
            self, "Delete Image",
            f"Permanently delete:\n  {os.path.basename(ip)}"
            + (f"\n  {os.path.basename(lp)}  (label)" if os.path.exists(lp) else "")
            + "\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._stop_thumb_loader()

        try:
            os.remove(ip)
        except OSError as e:
            QMessageBox.critical(self, "Delete Failed", f"Could not delete image:\n{e}")
            return
        if os.path.exists(lp):
            try:
                os.remove(lp)
            except OSError:
                pass

        self._images.pop(self._cur)
        self._label_cache.pop(ip, None)
        self._px_cache.pop(ip, None)
        self._thumbs.takeItem(self._cur)

        if self._images:
            self._cur = min(self._cur, len(self._images) - 1)
            self._load_cur()
        else:
            self._cur = 0
            self._canvas.clear_labels()
            self._nav.setText("0/0")

        self._update_stats()

        self._start_thumb_loader(reset_loaded=True)

    def _load_images(self):
        if not self._project: return
        d=os.path.join(self._project,"golden_boards"); lbl_dir=os.path.join(self._project,"labels")
        self._images=sorted([os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        self._label_cache={}; self._px_cache={}
        self._thumbs.setUpdatesEnabled(False); self._thumbs.clear()
        def _make_placeholder(col_hex):
            pm = QPixmap(100, 100); pm.fill(QColor(col_hex))
            p2 = QPainter(pm)
            p2.setPen(QPen(QColor(BG_BORDER2), 1)); p2.drawRect(0,0,99,99)
            p2.end(); return QIcon(pm)
        _ico_ok   = _make_placeholder(GREEN_DIM)
        _ico_pend = _make_placeholder(AMBER_DIM)
        for ip in self._images:
            bn=os.path.splitext(os.path.basename(ip))[0]; has=os.path.exists(os.path.join(lbl_dir,bn+".txt"))
            self._label_cache[ip]=has
            item=QListWidgetItem()
            item.setIcon(_ico_ok if has else _ico_pend)
            item.setToolTip(os.path.basename(ip))
            item.setData(Qt.ItemDataRole.UserRole, ip)
            self._thumbs.addItem(item)
        self._thumbs.setUpdatesEnabled(True)
        if self._cur>=len(self._images): self._cur=max(0,len(self._images)-1)
        if self._images: self._load_cur()
        self._update_stats()
        self._start_thumb_loader(reset_loaded=True)

    def _on_thumb(self,generation,idx,raw,w,h):
        if generation!=self._thumb_generation:
            return
        self._thumb_buf.append((idx,raw,w,h))
        if not self._thumb_flush_pending:
            self._thumb_flush_pending=True; QTimer.singleShot(150,self._flush_thumbs)

    def _flush_thumbs(self):
        self._thumb_flush_pending=False
        if not self._thumb_buf: return
        if not self.isVisible():
            # Tab is not currently shown — reschedule so the buffer is not lost.
            # This covers the case where images are loaded while the user is on
            # a different tab; without this the whole batch was silently dropped.
            self._thumb_flush_pending=True
            QTimer.singleShot(300, self._flush_thumbs)
            return
        self._thumbs.setUpdatesEnabled(False)
        for idx,raw,w,h in self._thumb_buf:
            item=self._thumbs.item(idx)
            if item:
                qimg=QImage(raw,w,h,w*3,QImage.Format.Format_RGB888).copy()
                pm=QPixmap.fromImage(qimg)
                # Draw 2-px label-status border directly on the pixmap
                ip = self._images[idx] if idx < len(self._images) else None
                has = self._label_cache.get(ip, False) if ip else False
                p2 = QPainter(pm)
                p2.setPen(QPen(QColor(GREEN if has else AMBER), 3))
                p2.drawRect(1, 1, pm.width()-2, pm.height()-2)
                p2.end()
                item.setIcon(QIcon(pm))
                self._thumb_loaded.add(idx)
                item.setText("")   # keep IconMode grid clean
        self._thumb_buf=[]; self._thumbs.setUpdatesEnabled(True); self._thumbs.viewport().update()

    def _lbl_path(self,ip):
        return os.path.join(self._project,"labels",os.path.splitext(os.path.basename(ip))[0]+".txt")

    def _load_cur(self):
        if not self._images or self._cur>=len(self._images): return
        ip=self._images[self._cur]
        if ip in self._px_cache:
            self._canvas._px=self._px_cache[ip]; self._canvas._scaled=None
            self._canvas._ow=self._canvas._px.width(); self._canvas._oh=self._canvas._px.height()
            self._canvas._boxes=[]; self._canvas._fit()
        else: self._canvas.load_image(ip)
        self._canvas.load_labels(self._lbl_path(ip))
        self._nav.setText(f"{self._cur+1}/{len(self._images)}")
        self._thumbs.setCurrentRow(self._cur); QTimer.singleShot(50,self._preload_adj)

    def _preload_adj(self):
        if not self._images: return
        idxs=[i for i in range(max(0,self._cur-1),min(len(self._images),self._cur+3))]
        needed={self._images[i] for i in idxs}
        for k in list(self._px_cache):
            if k not in needed: del self._px_cache[k]
        for i in idxs:
            path=self._images[i]
            if path not in self._px_cache:
                px=QPixmap(); px.load(path)
                if not px.isNull(): self._px_cache[path]=px

    def _save_cur(self):
        if not self._images or self._cur>=len(self._images): return
        ip=self._images[self._cur]; lp=self._lbl_path(ip); before=self._label_cache.get(ip,False)
        self._canvas.save_labels(lp)
        has=bool(self._canvas.get_boxes()) or (os.path.exists(lp) and os.path.getsize(lp)>0)
        if has!=before:
            self._label_cache[ip]=has; item=self._thumbs.item(self._cur)
            if item:
                # Re-tint the border on the existing icon thumbnail
                ico = item.icon()
                if not ico.isNull():
                    pm = ico.pixmap(100,100)
                    p2 = QPainter(pm)
                    p2.setPen(QPen(QColor(GREEN if has else AMBER), 3))
                    p2.drawRect(1,1,pm.width()-2,pm.height()-2)
                    p2.end(); item.setIcon(QIcon(pm))

    def _clear_lbl(self):
        if not self._images: return
        if QMessageBox.question(self,"Clear","Clear all boxes?",QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)==QMessageBox.StandardButton.No: return
        ip=self._images[self._cur]; lp=self._lbl_path(ip)
        if os.path.exists(lp): os.remove(lp)
        self._canvas.clear_labels(); self._label_cache[ip]=False
        item=self._thumbs.item(self._cur)
        if item: item.setForeground(QColor(AMBER))
        self._update_stats()

    def _next(self):
        self._save_cur()
        if self._images: self._cur=min(self._cur+1,len(self._images)-1)
        self._load_cur(); self._update_stats()

    def _prev(self):
        self._save_cur()
        if self._images: self._cur=max(0,self._cur-1)
        self._load_cur()

    def _on_select(self,idx):
        if 0<=idx<len(self._images) and idx!=self._cur:
            self._save_cur(); self._cur=idx; self._load_cur()

    def _update_stats(self):
        total=len(self._images); labeled=sum(1 for v in self._label_cache.values() if v)
        self._stats.setText(f"{total} images  {labeled} labeled  {total-labeled} remaining")
        if hasattr(self, "_b_del"):
            self._b_del.setEnabled(total > 0)

    def _auto_label(self):
        if not self._project: QMessageBox.warning(self,"No Project","Load a project first"); return
        # Model: use the chain (latest trained → OG)
        tp = self._cfg.get_next_base_model() or ""
        if not tp or not os.path.exists(tp):
            QMessageBox.warning(self,"No Model",
                "No model found in the chain.\nTrain a model first or set one in Settings."); return
        # Confidence: dedicated autolabel confidence from settings
        conf = float(self._cfg.get("autolabel_conf") or 0.50)
        self._al_log.clear(); self._al_prog.setValue(0)
        self._al_log.append(f"[Auto] Model: {os.path.basename(tp)}  conf={conf:.2f}")
        if self._cached_model_path != tp:
            if not HAS_YOLO: QMessageBox.warning(self,"Missing","ultralytics not installed"); return
            try:
                self._cached_model = load_optimized_yolo(tp, task="detect")
                self._cached_model_path = tp
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e)); return
        self._auto_thread=AutoLabelThread(tp,
            os.path.join(self._project,"golden_boards"),
            os.path.join(self._project,"labels"), conf)
        self._auto_thread.progress.connect(lambda p,m:(self._al_prog.setValue(p),self._al_log.append(m)))
        self._auto_thread.finished.connect(lambda c:(
            self._al_log.append(f"Done: {c} labeled"),self._al_prog.setValue(100),self._load_images()))
        self._auto_thread.start()

    def _run_aug(self):
        if not self._project: QMessageBox.warning(self,"No Project","Load a project first"); return
        if not any(self._label_cache.values()): QMessageBox.warning(self,"No Labels","Label at least one image first"); return
        opts={k:cb.isChecked() for k,cb in self._aug_cbs.items()}
        self._aug_thread=AugThread(os.path.join(self._project,"golden_boards"),os.path.join(self._project,"labels"),
            os.path.join(self._project,"augmented_images"),os.path.join(self._project,"augmented_labels"),opts)
        self._aug_thread.progress.connect(lambda p,m:self._aug_lbl.setText(m))
        self._aug_thread.finished.connect(self._on_aug_done); self._aug_thread.start()
        self._aug_lbl.setText("Generating..."); self._aug_lbl.setStyleSheet(f"#auglbl{{color:{CYAN};font-size:10px;border:none;}}")

    def _on_aug_done(self,n):
        self._aug_lbl.setText(f"Done: {n} images"); self._aug_lbl.setStyleSheet(f"#auglbl{{color:{GREEN};font-size:10px;border:none;}}")

    def _train(self):
        if not self._project: QMessageBox.warning(self,"No Project","Load a project first"); return
        if not any(self._label_cache.values()): QMessageBox.warning(self,"No Labels","Label images first"); return
        try: yaml,nt,nv=build_golden_dataset(self._project,MASTER_CLASS_LIST)
        except Exception as e: QMessageBox.critical(self,"Build Error",str(e)); return
        base=self._cfg.get_next_base_model()
        if not base or not os.path.exists(base):
            if QMessageBox.question(self,"No base","Train from scratch with yolo11s.pt?",
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)==QMessageBox.StandardButton.No: return
            base="yolo11s.pt"
        self._aug_params["epochs"]=self._epochs.value(); self._aug_params["batch"]=self._batch.value()
        mdir=os.path.join(self._project,"models"); os.makedirs(mdir,exist_ok=True)
        self._train_thread=TrainingThread(base,yaml,self._aug_params,mdir,"golden_ft")
        self._train_thread.progress.connect(self._on_train_progress)
        self._train_thread.finished.connect(self._on_train_done); self._train_thread.start()
        self._tr_prog.setValue(0); self._tr_lbl.setText(f"Training {nt}tr/{nv}val from {os.path.basename(base)}")

    def _on_train_progress(self,pct,msg):
        """FIX 5: Buffer rapid epoch callbacks, flush only every 250ms."""
        self._tr_prog.setValue(pct); self._tr_msg_pending=msg[:120]
        if not self._tr_timer.isActive(): self._tr_timer.start()

    def _flush_tr_msg(self):
        if self._tr_msg_pending is not None:
            self._tr_lbl.setText(self._tr_msg_pending); self._tr_msg_pending=None

    def _on_train_done(self,path):
        self._tr_timer.stop(); self._tr_msg_pending=None
        if path and os.path.exists(path):
            self._trained_model=path; self._cfg.record_trained_model(path)
            self._cached_model_path=None; self._cached_model=None
            self._tr_lbl.setText(f"Done: {os.path.basename(path)}")
            self._tr_lbl.setStyleSheet(f"#trlbl{{color:{GREEN};font-size:10px;font-family:Consolas;border:none;}}")
            self._tr_prog.setValue(100); self._refresh_chain()
            ToastManager.show(self.window(), f"Training complete — {os.path.basename(path)}", "success", 6000)
        else:
            if self._project:
                found=find_best_pt(os.path.join(self._project,"models"),"golden_ft")
                if found:
                    self._trained_model=found; self._cfg.record_trained_model(found)
                    self._cached_model_path=None; self._cached_model=None
                    self._tr_lbl.setText(f"Recovered: {os.path.basename(found)}")
                    self._tr_lbl.setStyleSheet(f"#trlbl{{color:{AMBER};font-size:10px;font-family:Consolas;border:none;}}"); self._refresh_chain(); return
            self._tr_lbl.setText("Failed - check console")
            self._tr_lbl.setStyleSheet(f"#trlbl{{color:{RED};font-size:10px;font-family:Consolas;border:none;}}")
            ToastManager.show(self.window(), "Training failed — check log", "error", 6000)

    def _deploy(self):
        if not self._trained_model or not os.path.exists(self._trained_model):
            p,_=QFileDialog.getOpenFileName(self,"Select trained model","","*.pt")
            if not p: return
            self._trained_model=p
        self._cfg.set("model_path",self._trained_model)
        self._dep_lbl.setText(f"Active: {os.path.basename(self._trained_model)}")
        self._dep_lbl.setStyleSheet(f"#deplbl{{color:{GREEN};font-size:10px;border:none;}}")
        ToastManager.show(self.window(), f"Model deployed: {os.path.basename(self._trained_model)}", "success")

    def showEvent(self,e):
        super().showEvent(e)
        if self._thumb_buf:
            QTimer.singleShot(0,self._flush_thumbs)
        elif self._needs_thumb_reload() and not (self._thumb_thread and self._thumb_thread.isRunning()):
            self._start_thumb_loader()

    def hideEvent(self,e):
        super().hideEvent(e)
        if self._thumb_thread and self._thumb_thread.isRunning():
            self._stop_thumb_loader()
        if getattr(self, "_auto_thread", None) and self._auto_thread.isRunning():
            self._auto_thread.terminate(); self._auto_thread.wait(500)
        if getattr(self, "_train_thread", None) and self._train_thread.isRunning():
            self._train_thread.terminate(); self._train_thread.wait(500)
        if getattr(self, "_aug_thread", None) and self._aug_thread.isRunning():
            self._aug_thread.terminate(); self._aug_thread.wait(500)

    def on_project_changed(self,path):
        self._project=path
        # Abort any running threads
        self._stop_thumb_loader()
        if self._aug_thread and self._aug_thread.isRunning(): self._aug_thread.terminate()
        # Reset all state
        self._images=[]; self._cur=0; self._label_cache={}; self._px_cache={}; self._thumb_loaded=set()
        self._thumbs.clear()
        self._canvas.clear_labels()
        self._canvas.load_image("")   # clears canvas
        self._nav.setText("0/0")
        self._stats.setText("0 images")
        self._al_prog.setValue(0); self._al_log.clear()
        self._aug_lbl.setText("Not run")
        self._aug_lbl.setStyleSheet(f"#auglbl{{color:{TEXT_SEC};font-size:10px;border:none;}}")
        self._tr_prog.setValue(0); self._tr_lbl.setText("")
        self._dep_lbl.setText("No model deployed")
        self._dep_lbl.setStyleSheet(f"#deplbl{{color:{TEXT_SEC};font-size:10px;border:none;}}")
        # Ensure project subdirs exist
        for sub in ["golden_boards","labels","models","augmented_images","augmented_labels","inbox"]:
            os.makedirs(os.path.join(path,sub),exist_ok=True)
        saved=self._cfg.get("aug_params")
        if saved: self._aug_params=dict(DEFAULT_AUG); self._aug_params.update(saved)
        self._epochs.setValue(int(self._aug_params.get("epochs",50)))
        self._batch.setValue(int(self._aug_params.get("batch",4)))
        self._load_images(); self._refresh_chain()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — LOGIC (Filter Pipeline + ROI)
# ─────────────────────────────────────────────────────────────────────────────
