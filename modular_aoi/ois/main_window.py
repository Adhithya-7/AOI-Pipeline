import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

"""
main_window.py — Sidebar, MainWindow, and entry point.
"""
import os
import platform
import sys
import time
from datetime import datetime

from PySide6.QtWidgets import (
    QInputDialog, QMainWindow, QWidget, QLabel, QFrame, QHBoxLayout, QVBoxLayout,
    QPushButton, QApplication, QStatusBar, QStackedWidget,
    QMessageBox, QFileDialog
)
from PySide6.QtCore import (
    Qt, Signal, QTimer
)
from PySide6.QtGui import (
    QFont, QIcon, QKeySequence, QPalette, QPixmap, QColor, QShortcut
)

if __package__ in (None, ""):
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    from ois.theme import (
        BG_DEEP, BG_PANEL, BG_CARD, BG_CARD2, BG_BORDER, BG_BORDER2,
        CYAN, CYAN_DIM, CYAN_GLOW, GREEN, GREEN_DIM, GREEN_GLOW,
        AMBER, AMBER_DIM, AMBER_GLOW, RED, RED_DIM, RED_GLOW,
        PURPLE, TEXT_PRI, TEXT_SEC, TEXT_DIM, SIDEBAR_W,
        _F_MONO_8, _F_MONO_9, _F_MONO_10, _F_MONO_11, _F_MONO_12,
        _card_style, make_card, make_sep, sec_lbl, set_fg
    )
    import ois.theme as _theme_mod
    from ois.utils import (
        HAS_CV2, HAS_NP, HAS_YOLO, DATA_ROOT,
        ProjectConfig, SettingsDialog, HistoryDB,
        match_detections, MASTER_CLASS_LIST
    )
    from ois.tabs.run_tab import RunTab
    from ois.tabs.train_tab import GoldenTab
    from ois.tabs.filters_tab import LogicTab
    from ois.tabs.aoi_tab import OfflineAOITab
    from ois.tabs.infer_tab import InferTab
    from ois.tabs.history_tab import HistoryTab
    from ois.widgets import SysLogTab, ProjectDialog, ToastManager
else:
    from .theme import (
        BG_DEEP, BG_PANEL, BG_CARD, BG_CARD2, BG_BORDER, BG_BORDER2,
        CYAN, CYAN_DIM, CYAN_GLOW, GREEN, GREEN_DIM, GREEN_GLOW,
        AMBER, AMBER_DIM, AMBER_GLOW, RED, RED_DIM, RED_GLOW,
        PURPLE, TEXT_PRI, TEXT_SEC, TEXT_DIM, SIDEBAR_W,
        _F_MONO_8, _F_MONO_9, _F_MONO_10, _F_MONO_11, _F_MONO_12,
        _card_style, make_card, make_sep, sec_lbl, set_fg
    )
    from . import theme as _theme_mod
    from .utils import (
        HAS_CV2, HAS_NP, HAS_YOLO, DATA_ROOT,
        ProjectConfig, SettingsDialog, HistoryDB,
        match_detections, MASTER_CLASS_LIST
    )
    from .tabs.run_tab import RunTab
    from .tabs.train_tab import GoldenTab
    from .tabs.filters_tab import LogicTab
    from .tabs.aoi_tab import OfflineAOITab
    from .tabs.infer_tab import InferTab
    from .tabs.history_tab import HistoryTab
    from .widgets import SysLogTab, ProjectDialog, ToastManager

try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
except ImportError:
    pynvml = None


class Sidebar(QWidget):
    tab_clicked=Signal(int)
    _SS_ON=(
        f"QPushButton{{background:{CYAN_DIM};"
        f"color:{CYAN};border:none;border-left:3px solid {CYAN};border-radius:0;padding:0;}}"
    )
    _SS_OFF=(
        f"QPushButton{{background:transparent;color:{TEXT_SEC};border:none;border-radius:0;padding:0;}}"
        f"QPushButton:hover{{background:{BG_CARD2};color:{TEXT_PRI};}}"
    )
    _ICONS=["⚡","◉","⚙","⬡","🔍","◷","⚙"]
    _LABELS=["RUN","TRAIN","FILTERS","AOI","INFER","HISTORY","SETTINGS"]

    def __init__(self):
        super().__init__(); self.setFixedWidth(SIDEBAR_W); self.setObjectName("sidebar")
        self.setStyleSheet(
            f"#sidebar{{background:{BG_PANEL};"
            f"border-right:1px solid {BG_BORDER2};}}"
        )
        self._active_idx=-1
        lay=QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        # Logo block
        _LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logo.png")
        _LOGO_IMG_W, _LOGO_IMG_H = 525, 250          # actual image dimensions
        logo_frame=QFrame(); logo_frame.setObjectName("logo_frame"); logo_frame.setFixedHeight(72)
        logo_frame.setStyleSheet(
            f"#logo_frame{{background:#FFFFFF;"
            f"border-bottom:1px solid {BG_BORDER2};}}"
        )
        ll=QVBoxLayout(logo_frame); ll.setContentsMargins(4,4,4,4); ll.setSpacing(2)
        logo_top=QLabel(); logo_top.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_top.setStyleSheet("border:none;background:transparent;")
        _logo_px = QPixmap(_LOGO_PATH)
        if _logo_px.isNull():
            # Fallback to text if image not found
            logo_top.setText("OIS")
            logo_top.setStyleSheet(
                f"color:{CYAN};font-size:22px;font-weight:bold;font-family:'Consolas';"
                f"border:none;letter-spacing:5px;"
            )
        else:
            # Scale to fit within sidebar (96px wide, ~56px tall keeping aspect ratio)
            _max_w = SIDEBAR_W - 8    # 88 px
            _max_h = 56
            _scale = min(_max_w / _LOGO_IMG_W, _max_h / _LOGO_IMG_H)
            _disp_w = int(_LOGO_IMG_W * _scale)
            _disp_h = int(_LOGO_IMG_H * _scale)
            logo_top.setPixmap(
                _logo_px.scaled(_disp_w, _disp_h,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            )
        logo_sub=QLabel("Optical \nInspection \nSystem"); logo_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_sub.setStyleSheet(
            f"color:{TEXT_SEC};font-size:8px;font-family:'Consolas';"
            f"letter-spacing:2px;font-weight:bold;border:none;background:transparent;"
        )
        ll.addStretch(); ll.addWidget(logo_top); ll.addWidget(logo_sub); ll.addStretch()
        lay.addWidget(logo_frame)

        # Spacer
        lay.addSpacing(8); self._btns=[]
        _FKEYS=["F1","F2","F3","F4","F5","F6",""]
        for i,(icon,lbl,fk) in enumerate(zip(self._ICONS,self._LABELS,_FKEYS)):
            if i==5:  # separator before utility buttons
                sep=QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet(f"background:{BG_BORDER2};max-height:1px;border:none;margin:4px 8px;")
                lay.addWidget(sep)
            b=QPushButton(); b.setFixedHeight(62 if i>=5 else 74); b.setObjectName(f"sb_{i}")
            b.setStyleSheet(self._SS_OFF)
            b.setToolTip(f"{lbl}{'  ('+fk+')' if fk else ''}")
            b.clicked.connect(lambda _,idx=i:self.tab_clicked.emit(idx))
            bl=QVBoxLayout(b); bl.setSpacing(3); bl.setContentsMargins(0,6,0,6)
            ic=QLabel(icon); ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ic.setStyleSheet("font-size:18px;border:none;")
            tx=QLabel(lbl); tx.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tx.setStyleSheet(
                f"font-size:11px;letter-spacing:1px;font-weight:bold;"
                f"font-family:'Consolas';border:none;"
            )
            bl.addWidget(ic); bl.addWidget(tx)
            if fk:
                fk_lbl=QLabel(fk); fk_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                fk_lbl.setStyleSheet(f"font-size:7px;color:{TEXT_DIM};font-family:Consolas;border:none;letter-spacing:1px;")
                bl.addWidget(fk_lbl)
            lay.addWidget(b); self._btns.append(b)

        lay.addStretch()

        # Version tag
        ver_frame=QFrame(); ver_frame.setObjectName("ver_frame"); ver_frame.setFixedHeight(36)
        ver_frame.setStyleSheet(f"#ver_frame{{border-top:1px solid {BG_BORDER2};background:{BG_PANEL};}}")
        vl=QHBoxLayout(ver_frame); vl.setContentsMargins(0,0,0,0)
        ver=QLabel("v7.1"); ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver.setStyleSheet(f"color:{TEXT_DIM};font-size:8px;font-family:'Consolas';letter-spacing:1px;border:none;")
        vl.addWidget(ver); lay.addWidget(ver_frame)

    def _set_active(self, idx):
        """Update only the two buttons that actually change: old active→OFF, new active→ON.
        Avoids looping all N buttons, calling styleSheet() (string alloc) on each,
        doing N CSS string comparisons, and firing up to N Qt stylesheet reparses."""
        if idx == self._active_idx:
            return
        if 0 <= self._active_idx < len(self._btns):
            self._btns[self._active_idx].setStyleSheet(self._SS_OFF)
        if 0 <= idx < len(self._btns):
            self._btns[idx].setStyleSheet(self._SS_ON)
        self._active_idx = idx


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._cfg=ProjectConfig()
        self._cur_tab=-1  # FIX 8: tab switch idempotency
        self._build()
        self._apply_palette()
        self.showMaximized()
        self._try_load_last()

    def _build(self):
        self.setWindowTitle("MELSS Optical Inspection System  ·  v7.1")
        self.setMinimumSize(1200,720)
        central=QWidget(); self.setCentralWidget(central)
        main_layout=QHBoxLayout(central); main_layout.setContentsMargins(0,0,0,0); main_layout.setSpacing(0)
        self._sidebar=Sidebar(); self._sidebar.tab_clicked.connect(self._on_sidebar_click)
        main_layout.addWidget(self._sidebar)
        self._stack=QStackedWidget(); main_layout.addWidget(self._stack,1)

        # ── Shared database ───────────────────────────────────────────────
        self._db = HistoryDB()

        # ── Tabs (indices 0-5 in stack) ───────────────────────────────────
        self._run=RunTab(self._cfg); self._run._db=self._db; self._stack.addWidget(self._run)
        self._run.project_selected.connect(self._load_project_path)
        self._golden=GoldenTab(self._cfg); self._stack.addWidget(self._golden)
        self._run.flag_saved.connect(lambda _: self._golden._load_images())
        self._logic=LogicTab(self._cfg); self._stack.addWidget(self._logic)
        self._aoi_offline=OfflineAOITab(self._cfg); self._aoi_offline._db=self._db
        self._stack.addWidget(self._aoi_offline)
        self._infer=InferTab(self._cfg); self._stack.addWidget(self._infer)
        self._history=HistoryTab(self._db); self._stack.addWidget(self._history)  # idx 5

        # SysLogTab kept as invisible sink
        self._syslog=SysLogTab()
        self._logic.pipeline_deployed.connect(self._on_pipeline)
        self._logic.pipeline_deployed.connect(self._aoi_offline.deploy_pipeline)
        self._logic.frame_requested.connect(self._on_frame_requested)

        # ── Status bar ────────────────────────────────────────────────────
        self._statusbar=QStatusBar(); self.setStatusBar(self._statusbar)
        self._statusbar.setStyleSheet(
            f"QStatusBar{{background:{BG_PANEL};color:{TEXT_SEC};"
            f"border-top:1px solid {BG_BORDER2};font-family:Consolas;font-size:10px;}}"
            f"QStatusBar::item{{border:none;}}"
        )

        # Mode pill (left-permanent)
        self._mode_pill=QLabel("  IDLE  ")
        self._mode_pill.setStyleSheet(
            f"color:{TEXT_DIM};background:{BG_CARD2};border:1px solid {BG_BORDER2};"
            f"border-radius:3px;font-family:Consolas;font-size:10px;font-weight:bold;"
            f"padding:1px 6px;letter-spacing:1px;"
        )
        self._statusbar.addPermanentWidget(self._mode_pill)

        # Camera live dot
        self._cam_dot=QLabel("●")
        self._cam_dot.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;padding:0 4px;")
        self._statusbar.addPermanentWidget(self._cam_dot)

        # System stats
        self._sb_stats=QLabel("")
        self._sb_stats.setStyleSheet(f"color:{TEXT_SEC};font-family:Consolas;font-size:10px;padding:0 10px;")
        self._statusbar.addPermanentWidget(self._sb_stats)

        # Clock (far right)
        self._clock_lbl=QLabel()
        self._clock_lbl.setStyleSheet(f"color:{TEXT_SEC};font-family:Consolas;font-size:10px;padding:0 8px;")
        self._statusbar.addPermanentWidget(self._clock_lbl)
        self._clock_timer=QTimer(self); self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(self._tick_clock); self._clock_timer.start()
        self._tick_clock()

        self._sys_timer=QTimer(self); self._sys_timer.setInterval(4000)
        self._sys_timer.timeout.connect(self._update_sb_stats); self._sys_timer.start()
        self._update_sb_stats()
        self._statusbar.showMessage("Ready")
        self._switch(0)

    def _update_sb_stats(self):
        parts = []
        # Lazily import and cache heavy monitor modules (avoids repeated import overhead)
        if not hasattr(self, "_psutil"):
            try: import psutil as _ps; self._psutil = _ps
            except: self._psutil = None
        if not hasattr(self, "_pynvml"):
            try:
                import pynvml as _nv; _nv.nvmlInit()
                self._pynvml = _nv
                # Cache the device handle once — nvmlDeviceGetHandleByIndex is slow
                self._pynvml_handle = _nv.nvmlDeviceGetHandleByIndex(0)
            except:
                self._pynvml = None
                self._pynvml_handle = None
        if self._psutil:
            try:
                parts.append(f"CPU {self._psutil.cpu_percent():.0f}%")
                vm = self._psutil.virtual_memory(); parts.append(f"MEM {vm.percent:.0f}%")
            except: pass
        if self._pynvml and self._pynvml_handle:
            try:
                u = self._pynvml.nvmlDeviceGetUtilizationRates(self._pynvml_handle)
                m = self._pynvml.nvmlDeviceGetMemoryInfo(self._pynvml_handle)
                parts.append(f"GPU {u.gpu}%  VRAM {m.used//1024//1024}/{m.total//1024//1024}MB")
            except: pass
        if parts: self._sb_stats.setText("  ·  ".join(parts))

    def _setup_project_tab(self):
        lay=QVBoxLayout(self._proj_placeholder); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        # Header band
        header=QFrame(); header.setObjectName("proj_header"); header.setFixedHeight(120)
        header.setStyleSheet(
            f"#proj_header{{background:{CYAN};border-bottom:1px solid #1347A0;}}"
        )
        hl=QVBoxLayout(header); hl.setContentsMargins(24,0,24,0); hl.setSpacing(4)
        hl.addStretch()
        t=QLabel("OPTICAL INSPECTION SYSTEM"); t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet(
            f"color:#FFFFFF;font-size:24px;font-weight:bold;font-family:'Consolas';"
            f"letter-spacing:8px;border:none;"
        )
        sub=QLabel("COMPONENT DEFECT DETECTION  ·  MELSS"); sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(f"color:#B8D4FF;font-size:9px;font-family:'Consolas';letter-spacing:4px;border:none;")
        hl.addWidget(t); hl.addWidget(sub); hl.addStretch(); lay.addWidget(header)

        # Body
        body=QWidget(); body.setObjectName("proj_body")
        body.setStyleSheet(f"#proj_body{{background:{BG_DEEP};}}")
        bl=QVBoxLayout(body); bl.setContentsMargins(24,50,24,24); bl.setSpacing(14); bl.addStretch()

        for txt,slot,col,sub_txt in [
            ("＋  NEW PROJECT",  self._new_project,  CYAN,  "Create a new inspection project"),
            ("▶  OPEN PROJECT",  self._open_project, AMBER, "Browse for existing project folder"),
            ("◷  RECENT",        self._recent_projects,PURPLE,"Manage and reopen recent projects"),
        ]:
            grp=QFrame(); grp.setObjectName(f"pg_{txt[0]}"); gl=QVBoxLayout(grp); gl.setContentsMargins(0,0,0,0); gl.setSpacing(3)
            b=QPushButton(txt); b.setFixedHeight(52); b.setFixedWidth(340)
            b.setStyleSheet(
                f"QPushButton{{background:{BG_PANEL};"
                f"color:{col};border:2px solid {col}66;border-radius:8px;"
                f"font-size:13px;font-weight:bold;font-family:'Consolas';letter-spacing:2px;padding:0 20px;}}"
                f"QPushButton:hover{{background:{col}18;border-color:{col}BB;}}"
                f"QPushButton:pressed{{background:{col}28;}}"
            )
            b.clicked.connect(slot)
            sl=QLabel(sub_txt); sl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sl.setStyleSheet(f"color:{TEXT_SEC};font-size:10px;border:none;")
            gl.addWidget(b,alignment=Qt.AlignmentFlag.AlignCenter); gl.addWidget(sl,alignment=Qt.AlignmentFlag.AlignCenter)
            bl.addWidget(grp,alignment=Qt.AlignmentFlag.AlignCenter)

        bl.addSpacing(20)
        self._proj_lbl=QLabel("No project loaded"); self._proj_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._proj_lbl.setStyleSheet(
            f"color:{TEXT_SEC};font-family:'Consolas';font-size:11px;"
            f"background:{BG_CARD};border:1px solid {BG_BORDER2};border-radius:6px;padding:8px 20px;"
        )
        self._proj_lbl.setFixedWidth(340)
        bl.addWidget(self._proj_lbl,alignment=Qt.AlignmentFlag.AlignCenter)
        bl.addStretch(); lay.addWidget(body,1)

    def _tick_clock(self):
        self._clock_lbl.setText(datetime.now().strftime("%H:%M:%S"))

    def _on_sidebar_click(self, idx: int):
        if idx == 6:   # SETTINGS — opens dialog, no tab switch
            dlg = SettingsDialog(self._cfg, self)
            def _on_saved():
                new_conf = int(self._cfg.get("confidence", 0.5) * 100)
                if hasattr(self._run, "_conf"):
                    self._run._conf.setValue(new_conf)
                if hasattr(self._aoi_offline, "_conf_spin"):
                    self._aoi_offline._conf_spin.setValue(float(self._cfg.get("aoi_conf", 0.25)))
                if hasattr(self._aoi_offline, "_cal_spin"):
                    self._aoi_offline._cal_spin.setValue(int(self._cfg.get("aoi_cal_runs", 8)))
            dlg.settings_changed.connect(_on_saved)
            dlg.exec()
        else:
            self._switch(idx)

    def _set_mode_pill(self, text: str, color: str):
        new_text = f"  {text}  "
        if getattr(self, "_pill_text", None) == new_text and getattr(self, "_pill_color", None) == color:
            return
        self._pill_text = new_text
        self._pill_color = color
        self._mode_pill.setText(new_text)
        self._mode_pill.setStyleSheet(
            f"color:{color};background:{color}22;border:1px solid {color}55;"
            f"border-radius:3px;font-family:Consolas;font-size:10px;font-weight:bold;"
            f"padding:1px 6px;letter-spacing:1px;"
        )

    def _set_cam_dot(self, live: bool):
        if getattr(self, "_cam_dot_live", None) == live:
            return
        self._cam_dot_live = live
        self._cam_dot.setStyleSheet(
            f"color:{GREEN if live else TEXT_DIM};font-size:12px;padding:0 4px;"
        )

    def _switch(self, idx: int):
        """Skip redundant tab switch."""
        if idx == self._cur_tab: return
        self._cur_tab = idx
        self._stack.setCurrentIndex(idx)
        self._sidebar._set_active(idx)
        names = ["RUN","TRAIN","FILTERS","AOI","INFER","HISTORY"]
        name = names[idx] if idx < len(names) else str(idx)
        self._statusbar.showMessage(f"{name} — {os.path.basename(self._cfg._path or 'No project')}")
        if idx == 5:   # History tab — refresh on switch
            self._history.refresh()
        # Update mode pill colour per tab
        run_running = getattr(getattr(self, "_run", None), "_running", False)
        pill_map = {0:("RUNNING" if run_running else "STOPPED", GREEN if run_running else TEXT_DIM),
                    1:("TRAIN", AMBER), 2:("FILTERS", PURPLE),
                    3:("AOI", GREEN),  4:("INFER", CYAN), 5:("HISTORY", TEXT_SEC)}
        if idx in pill_map:
            self._set_mode_pill(*pill_map[idx])

    def _update_mode_pill(self):
        """Called by RunTab start/stop to update the pill live."""
        if self._cur_tab == 0:
            running = self._run._running
            self._set_mode_pill("RUNNING" if running else "STOPPED",
                                GREEN if running else TEXT_DIM)
            self._set_cam_dot(running)

    def _load_project_path(self,path):
        if not path or not os.path.isdir(path): return
        self._cfg.load(path); self._on_project_opened(path)

    def _on_pipeline(self,filters,model_path):
        self._run.deploy_pipeline(filters,model_path)
        frame=self._run.get_latest_frame()
        if frame is not None: self._logic.set_src_frame(frame)
        self._syslog.append(f"Pipeline deployed: {len(filters)} filters")
        self._switch(0)
        ToastManager.show(self, f"Pipeline deployed — {len(filters)} filters", "success")

    def _on_frame_requested(self):
        """Grab Frame from Camera in LogicTab — no deploy, no tab switch.
        If camera is running: grab the latest frame and hand it to LogicTab.
        If camera is off: navigate to Run tab so the user can start it."""
        if self._run._running:
            frame = self._run.get_latest_frame()
            if frame is not None:
                self._logic.set_src_frame(frame)
                self._syslog.append("[Logic] Frame grabbed from live camera")
                ToastManager.show(self, "Frame grabbed from camera", "info")
            else:
                ToastManager.show(self, "Camera is running but no frame yet — try again", "warning")
        else:
            # Camera is off — go to Run tab so user can start it
            self._switch(0)
            ToastManager.show(self, "Start the camera in RUN tab, then Grab Frame", "info")

    def _on_project_opened(self,path):
        name=os.path.basename(path)
        self._run.set_active_project(path)
        self._run.on_project_changed(path)
        self._golden.on_project_changed(path)
        self._logic.on_project_changed(path)
        self._syslog.append(f"Project: {name}")
        self._aoi_offline.on_project_changed(path)
        self._history.on_project_changed(path)
        self.setWindowTitle(f"MELSS Optical Inspection System — {name}")
        self._statusbar.showMessage(f"Project: {name}")
        ToastManager.show(self, f"Project loaded: {name}", "info")

    def _new_project(self):
        name,ok=QInputDialog.getText(self,"New Project","Project name:")
        if ok and name.strip():
            path=os.path.join(DATA_ROOT,"Projects",name.strip())
            for sub in ["golden_boards","labels","models","augmented_images","augmented_labels","inbox"]:
                os.makedirs(os.path.join(path,sub),exist_ok=True)
            self._cfg.load(path); self._on_project_opened(path)

    def _open_project(self):
        p=QFileDialog.getExistingDirectory(self,"Open Project",os.path.join(DATA_ROOT,"Projects"))
        if p: self._cfg.load(p); self._on_project_opened(p)

    def _recent_projects(self):
        dlg=ProjectDialog(self._cfg,self); dlg.project_opened.connect(self._on_project_opened); dlg.exec()

    def _try_load_last(self):
        last=self._cfg.get_last_project()
        if last and os.path.isdir(last): self._cfg.load(last); self._on_project_opened(last)
        else:
            self._switch(0)
            self._statusbar.showMessage("RUN - No project loaded (select one from the project switcher)")

    def _apply_palette(self):
        p=QPalette()
        p.setColor(QPalette.ColorRole.Window,QColor(BG_DEEP))
        p.setColor(QPalette.ColorRole.WindowText,QColor(TEXT_PRI))
        p.setColor(QPalette.ColorRole.Base,QColor(BG_PANEL))
        p.setColor(QPalette.ColorRole.AlternateBase,QColor(BG_CARD2))
        p.setColor(QPalette.ColorRole.Text,QColor(TEXT_PRI))
        p.setColor(QPalette.ColorRole.Button,QColor(BG_CARD2))
        p.setColor(QPalette.ColorRole.ButtonText,QColor(TEXT_PRI))
        p.setColor(QPalette.ColorRole.Highlight,QColor(CYAN_DIM))
        p.setColor(QPalette.ColorRole.HighlightedText,QColor(CYAN))
        p.setColor(QPalette.ColorRole.ToolTipBase,QColor(BG_PANEL))
        p.setColor(QPalette.ColorRole.ToolTipText,QColor(TEXT_PRI))
        p.setColor(QPalette.ColorRole.Mid,QColor(BG_BORDER))
        p.setColor(QPalette.ColorRole.Dark,QColor(BG_BORDER2))
        p.setColor(QPalette.ColorRole.Light,QColor(BG_PANEL))
        p.setColor(QPalette.ColorRole.PlaceholderText,QColor(TEXT_DIM))
        app=QApplication.instance(); app.setPalette(p); self.setPalette(p)
        app.setStyleSheet(_theme_mod.GLOBAL_QSS)

    def keyPressEvent(self,e):
        k=e.key()
        if k==Qt.Key.Key_F1: self._switch(0)
        elif k==Qt.Key.Key_F2: self._switch(1)
        elif k==Qt.Key.Key_F3: self._switch(2)
        elif k==Qt.Key.Key_F4: self._switch(3)
        elif k==Qt.Key.Key_F5: self._switch(4)
        elif k==Qt.Key.Key_F6: self._switch(5)
        else: super().keyPressEvent(e)

    def closeEvent(self,e):
        if self._run._running: self._run._stop()
        if self._golden._train_thread and self._golden._train_thread.isRunning():
            if QMessageBox.question(self,"Training","Training is running. Quit anyway?",
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)==QMessageBox.StandardButton.No:
                e.ignore(); return
            self._golden._train_thread.terminate()
        if self._cfg._path: self._cfg.save(); self._logic.save_pipeline(self._cfg._path)
        self._db.close()
        e.accept()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _theme_mod.init()
    if HAS_CV2:
        import cv2
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)
    app=QApplication(sys.argv)
    from PySide6.QtGui import QPixmapCache
    QPixmapCache.setCacheLimit(256 * 1024)
    app.setApplicationName("Optical Inspection System")
    app.setOrganizationName("MELSS")
    app.setStyle("Fusion")
    # Per-OS native font stack
    _os_fonts = {
        "Windows": ["Segoe UI", "Arial"],
        "Darwin":  ["SF Pro Display", "SF Pro Text", "Helvetica Neue", "Arial"],
        "Linux":   ["Ubuntu", "Noto Sans", "Liberation Sans", "DejaVu Sans", "Arial"],
    }
    f = QFont()
    f.setPointSize(10)
    for family in _os_fonts.get(platform.system(), ["Arial"]):
        f.setFamily(family)
        if f.exactMatch():
            break
    app.setFont(f)
    win=MainWindow()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
