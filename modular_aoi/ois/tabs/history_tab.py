"""
history_tab.py — Inspection results history tab (HistoryTab).
"""
import json
import os
import platform
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
from PySide6.QtCore import Qt, Signal, Slot, QTimer, Property, QRect
from PySide6.QtGui import (
    QBrush, QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor,
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
from ..widgets import (
    FastLog, StatCard, ToastManager, ZoomableImageView
)
from ..utils import HistoryDB

if HAS_CV2:
    import cv2
if HAS_NP:
    import numpy as np


class HistoryTab(QWidget):
    def __init__(self, db: "HistoryDB"):
        super().__init__(); self._db = db; self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(10)

        # ── Filter bar ────────────────────────────────────────────────────
        fb = make_card("hist_filter"); fl = QHBoxLayout(fb)
        fl.setContentsMargins(14,8,14,8); fl.setSpacing(10)
        fl.addWidget(sec_lbl("HISTORY"))

        fl.addWidget(QLabel("Project:"))
        self._proj_cb = QComboBox(); self._proj_cb.setFixedHeight(28)
        self._proj_cb.setMinimumWidth(140)
        self._proj_cb.addItem("All projects", None)
        fl.addWidget(self._proj_cb)

        fl.addWidget(QLabel("Mode:"))
        self._mode_cb = QComboBox(); self._mode_cb.setFixedHeight(28)
        for m in ["All","RUN","AOI"]: self._mode_cb.addItem(m)
        fl.addWidget(self._mode_cb)

        fl.addWidget(QLabel("Result:"))
        self._res_cb = QComboBox(); self._res_cb.setFixedHeight(28)
        for r in ["All","Pass","Fail"]: self._res_cb.addItem(r)
        fl.addWidget(self._res_cb)

        fl.addWidget(QLabel("Days:"))
        self._days_sp = QSpinBox(); self._days_sp.setRange(1,365)
        self._days_sp.setValue(30); self._days_sp.setFixedWidth(60)
        fl.addWidget(self._days_sp)

        b_ref = QPushButton("Refresh"); b_ref.setFixedHeight(28)
        b_ref.clicked.connect(self.refresh)
        b_exp = QPushButton("Export CSV"); b_exp.setFixedHeight(28)
        b_exp.clicked.connect(self._export)
        b_clr = QPushButton("Clear All"); b_clr.setFixedHeight(28)
        b_clr.setStyleSheet(f"QPushButton{{color:{RED};border-color:{RED_DIM};}}")
        b_clr.clicked.connect(self._clear_all)
        fl.addStretch()
        fl.addWidget(b_ref); fl.addWidget(b_exp); fl.addWidget(b_clr)
        root.addWidget(fb)

        # ── Summary cards row ─────────────────────────────────────────────
        sc_row = QHBoxLayout(); sc_row.setSpacing(8)
        self._sc_total = StatCard("TOTAL",  "0", CYAN)
        self._sc_pass  = StatCard("PASS",   "0", GREEN)
        self._sc_fail  = StatCard("FAIL",   "0", RED)
        self._sc_yield = StatCard("YIELD",  "—", AMBER)
        for c in [self._sc_total,self._sc_pass,self._sc_fail,self._sc_yield]:
            sc_row.addWidget(c)
        root.addLayout(sc_row)

        # ── Main area: table + trend chart side by side ───────────────────
        mid = QHBoxLayout(); mid.setSpacing(10)

        # Table
        tf = make_card("hist_table"); tl = QVBoxLayout(tf)
        tl.setContentsMargins(8,8,8,8); tl.setSpacing(6)
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            ["Timestamp","Project","Mode","Result","Defects","Defect Types","Model"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background:{BG_PANEL};border:1px solid {BG_BORDER};
                border-radius:6px;gridline-color:{BG_BORDER};
                font-size:11px;font-family:Consolas;
            }}
            QTableWidget::item{{padding:4px 8px;}}
            QTableWidget::item:selected{{background:{CYAN_DIM};color:{CYAN};}}
            QHeaderView::section{{
                background:{BG_CARD2};color:{TEXT_SEC};
                border:none;border-bottom:1px solid {BG_BORDER};
                padding:6px 8px;font-size:10px;font-weight:bold;letter-spacing:1px;
            }}
            QTableWidget::item:alternate{{background:{BG_CARD};}}
        """)
        for w in [140,120,60,64,64,0,120]:
            pass  # set column widths after
        tl.addWidget(self._table)
        mid.addWidget(tf, 3)

        # Trend chart
        cf = make_card("hist_chart"); cl = QVBoxLayout(cf)
        cl.setContentsMargins(12,12,12,12); cl.setSpacing(6)
        cl.addWidget(sec_lbl("YIELD TREND (14 DAYS)"))
        self._chart = _YieldChart()
        cl.addWidget(self._chart, 1)
        mid.addWidget(cf, 1)

        root.addLayout(mid, 1)

        # ── Defect thumbnail strip ────────────────────────────────────────
        thumb_hdr = QHBoxLayout()
        thumb_hdr.addWidget(sec_lbl("DEFECT THUMBNAILS"))
        thumb_hdr.addStretch()
        self._thumb_count_lbl = QLabel("")
        self._thumb_count_lbl.setObjectName("htcnt")
        self._thumb_count_lbl.setStyleSheet(
            f"#htcnt{{color:{TEXT_SEC};font-size:10px;font-family:Consolas;border:none;padding:0 4px;}}")
        thumb_hdr.addWidget(self._thumb_count_lbl)
        root.addLayout(thumb_hdr)

        self._thumb_scroll = QScrollArea()
        self._thumb_scroll.setFixedHeight(162)
        self._thumb_scroll.setWidgetResizable(False)
        self._thumb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._thumb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._thumb_scroll.setStyleSheet(
            f"QScrollArea{{border:1px solid {BG_BORDER};border-radius:6px;background:{BG_DEEP};}}")
        self._thumb_inner = QWidget()
        self._thumb_inner.setFixedHeight(150)
        self._thumb_inner.setStyleSheet(f"background:{BG_CARD2};")
        self._thumb_hlayout = QHBoxLayout(self._thumb_inner)
        self._thumb_hlayout.setContentsMargins(6, 4, 6, 4)
        self._thumb_hlayout.setSpacing(6)
        self._thumb_hlayout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._thumb_inner.setMinimumWidth(300)
        self._thumb_scroll.setWidget(self._thumb_inner)
        root.addWidget(self._thumb_scroll)

    def refresh(self):
        proj = self._proj_cb.currentData()
        mode = None if self._mode_cb.currentText()=="All" else self._mode_cb.currentText()
        res_text = self._res_cb.currentText()
        passed = None if res_text=="All" else (res_text=="Pass")
        days = self._days_sp.value()

        rows = self._db.fetch(project=proj, days=days, mode=mode, passed=passed)

        # Update summary
        total = len(rows)
        pass_n = sum(1 for r in rows if r[4])
        fail_n = total - pass_n
        yield_pct = f"{pass_n/total*100:.1f}%" if total else "—"
        self._sc_total.set_value(total)
        self._sc_pass.set_value(pass_n)
        self._sc_fail.set_value(fail_n)
        self._sc_yield.set_value(yield_pct)

        # Populate table
        self._table.setRowCount(len(rows))
        for r_idx, row in enumerate(rows):
            # id,ts,project,mode,passed,n_defects,defect_types,model,image_path
            _id,ts,project,m,p,nd,dt,mdl,img_path = row
            items = [ts, project, m,
                     "PASS" if p else "FAIL",
                     str(nd), dt or "—", os.path.basename(mdl) if mdl else "—"]
            for c_idx, val in enumerate(items):
                item = QTableWidgetItem(str(val))
                if c_idx == 3:  # Result column
                    item.setForeground(QColor(GREEN if p else RED))
                self._table.setItem(r_idx, c_idx, item)

        # Column widths
        self._table.setColumnWidth(0,140); self._table.setColumnWidth(1,110)
        self._table.setColumnWidth(2,54);  self._table.setColumnWidth(3,56)
        self._table.setColumnWidth(4,56);  self._table.setColumnWidth(5,130)

        # Update trend chart
        trend = self._db.yield_trend(project=proj, days=14)
        self._chart.set_data(trend)

        # Refresh project list in combo
        current_proj = self._proj_cb.currentData()
        self._proj_cb.blockSignals(True)
        self._proj_cb.clear()
        self._proj_cb.addItem("All projects", None)
        for pr in self._db.projects():
            self._proj_cb.addItem(pr, pr)
        # Restore selection
        for i in range(self._proj_cb.count()):
            if self._proj_cb.itemData(i) == current_proj:
                self._proj_cb.setCurrentIndex(i); break
        self._proj_cb.blockSignals(False)

        # ── Rebuild thumbnail strip ───────────────────────────────────────
        # Cancel any in-progress deferred load first
        if hasattr(self, "_thumb_timer") and self._thumb_timer.isActive():
            self._thumb_timer.stop()

        # Clear existing cards
        while self._thumb_hlayout.count():
            item = self._thumb_hlayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        CARD_W, CARD_H = 196, 142
        THUMB_H = CARD_H - 28
        thumb_rows = [r for r in rows if r[8] and os.path.exists(r[8])][:40]

        if not thumb_rows:
            ph = QLabel("No saved images yet — run an inspection to see thumbnails here")
            ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ph.setStyleSheet(
                f"color:{TEXT_DIM};font-size:10px;font-family:Consolas;border:none;")
            self._thumb_hlayout.addWidget(ph)
            self._thumb_count_lbl.setText("")
            self._thumb_inner.setMinimumWidth(self._thumb_scroll.width() or 400)
        else:
            # Pre-compute widths and labels so the layout is stable from the start
            total_w = len(thumb_rows) * (CARD_W + 6) + 12
            self._thumb_inner.setMinimumWidth(total_w)
            fail_count = sum(1 for r in thumb_rows if not r[4])
            self._thumb_count_lbl.setText(
                f"{len(thumb_rows)} image{'s' if len(thumb_rows)!=1 else ''}  ·  {fail_count} fail")

            # ── Deferred card loading via QTimer — one card per tick ──────────
            # This prevents the main-thread IO burst that caused the tab to freeze
            # or briefly white-out when switching to History with many saved images.
            self._thumb_pending = list(thumb_rows)
            self._thumb_card_dims = (CARD_W, CARD_H, THUMB_H)

            if not hasattr(self, "_thumb_timer"):
                self._thumb_timer = QTimer(self)
                self._thumb_timer.setInterval(0)   # idle-time processing
                self._thumb_timer.timeout.connect(self._load_next_thumb)
            self._thumb_timer.start()

    def _load_next_thumb(self):
        """Load one thumbnail card per timer tick to keep the UI responsive."""
        if not getattr(self, "_thumb_pending", None):
            if hasattr(self, "_thumb_timer"):
                self._thumb_timer.stop()
            return

        from PySide6.QtGui import QImageReader
        row = self._thumb_pending.pop(0)
        CARD_W, CARD_H, THUMB_H = self._thumb_card_dims
        _id,ts,project,m,p,nd,dt,mdl,img_path = row
        col_hex = GREEN if p else RED

        card = QFrame()
        card.setFixedSize(CARD_W, CARD_H)
        card.setStyleSheet(
            f"QFrame{{background:{BG_CARD2};border:2px solid {col_hex}55;"
            f"border-radius:6px;}}"
            f"QFrame:hover{{border-color:{col_hex};}}")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(3, 3, 3, 3); cl.setSpacing(2)

        # Thumbnail image — scaled read avoids 256 MB Qt allocation limit
        reader = QImageReader(img_path)
        orig = reader.size()
        if orig.isValid() and orig.width() > 0:
            scale = min((CARD_W - 6) / orig.width(), THUMB_H / orig.height())
            reader.setScaledSize(orig.scaled(
                int(orig.width() * scale), int(orig.height() * scale),
                Qt.AspectRatioMode.KeepAspectRatio))
        qimg = reader.read()
        px = QPixmap.fromImage(qimg) if not qimg.isNull() else QPixmap()
        if not px.isNull():
            img_lbl = QLabel()
            img_lbl.setPixmap(px)
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setFixedHeight(THUMB_H)
            img_lbl.setToolTip(
                f"{'PASS' if p else 'FAIL'} — {ts}\n{nd} defect(s)\nClick to open full image")
            img_lbl.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            def _open(ev, path=img_path):
                if os.path.exists(path):
                    if platform.system()=="Windows":   os.startfile(path)
                    elif platform.system()=="Darwin":  __import__("subprocess").Popen(["open",path])
                    else:                              __import__("subprocess").Popen(["xdg-open",path])
            img_lbl.mousePressEvent = _open
            cl.addWidget(img_lbl)
        else:
            ph2 = QLabel("Image\nnot found")
            ph2.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ph2.setFixedHeight(THUMB_H)
            ph2.setStyleSheet(
                f"color:{TEXT_DIM};font-size:9px;font-family:Consolas;border:none;")
            cl.addWidget(ph2)

        # Info label
        ts_short = ts[11:16] if len(ts) > 10 else ts
        mode_tag = m[:3].upper()
        info_txt = f"{ts_short}  {mode_tag}  {'✔' if p else '✘'}{nd}"
        info_lbl = QLabel(info_txt)
        info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_lbl.setFixedHeight(18)
        info_lbl.setStyleSheet(
            f"color:{col_hex};font-size:9px;font-family:Consolas;"
            f"border:none;background:transparent;")
        cl.addWidget(info_lbl)

        self._thumb_hlayout.addWidget(card)

        if not self._thumb_pending and hasattr(self, "_thumb_timer"):
            self._thumb_timer.stop()

    def on_project_changed(self, path):
        self.refresh()

    def _export(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export History CSV",
                                            "ois_history.csv", "CSV (*.csv)")
        if p:
            try:
                self._db.export_csv(p)
                ToastManager.show(self.window(), f"Exported to {os.path.basename(p)}", "success")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _clear_all(self):
        if QMessageBox.question(self,"Clear History",
            "Delete all inspection history? This cannot be undone.",
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            with self._db._lock:
                self._db._con.execute("DELETE FROM results")
                self._db._con.commit()
            self.refresh()


class _YieldChart(QWidget):
    """Simple yield-over-time line chart drawn with QPainter."""
    def __init__(self):
        super().__init__()
        self._data: list = []   # [(date_str, total, pass_n)]
        self.setMinimumHeight(140)

    def set_data(self, data):
        self._data = data; self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0,0,W,H, QColor(BG_DEEP))

        if not self._data:
            p.setPen(QPen(QColor(TEXT_DIM)))
            p.setFont(_F_MONO_9)
            p.drawText(QRect(0,0,W,H), Qt.AlignmentFlag.AlignCenter, "No data yet")
            p.end(); return

        PAD_L,PAD_R,PAD_T,PAD_B = 36,10,14,28
        cw = W - PAD_L - PAD_R
        ch = H - PAD_T - PAD_B

        # Draw grid lines
        p.setPen(QPen(QColor(BG_BORDER), 1, Qt.PenStyle.DotLine))
        for pct in [0,25,50,75,100]:
            y = PAD_T + ch - int(pct/100*ch)
            p.drawLine(PAD_L, y, PAD_L+cw, y)
            p.setPen(QPen(QColor(TEXT_DIM))); p.setFont(_F_MONO_8)
            p.drawText(QRect(0,y-8,PAD_L-4,16), Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter, f"{pct}%")
            p.setPen(QPen(QColor(BG_BORDER), 1, Qt.PenStyle.DotLine))

        # Draw yield line
        n = len(self._data)
        if n < 1: p.end(); return
        pts = []
        for i,(date,total,pass_n) in enumerate(self._data):
            if not total: continue
            yld = pass_n / total
            x = PAD_L + int(i/(max(n-1,1))*cw)
            y = PAD_T + ch - int(yld*ch)
            pts.append((x,y,yld))

        if len(pts) >= 2:
            pen = QPen(QColor(CYAN), 2); p.setPen(pen)
            for i in range(len(pts)-1):
                p.drawLine(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1])

        # Draw dots
        for x,y,yld in pts:
            col = QColor(GREEN if yld>=0.9 else (AMBER if yld>=0.7 else RED))
            p.setBrush(QBrush(col)); p.setPen(QPen(QColor(BG_DEEP),1))
            p.drawEllipse(x-4, y-4, 8, 8)

        # X-axis labels (first and last)
        p.setPen(QPen(QColor(TEXT_DIM))); p.setFont(_F_MONO_8)
        if self._data:
            p.drawText(QRect(PAD_L,H-PAD_B+4,80,14), Qt.AlignmentFlag.AlignLeft, self._data[0][0][-5:])
            p.drawText(QRect(PAD_L+cw-80,H-PAD_B+4,80,14), Qt.AlignmentFlag.AlignRight, self._data[-1][0][-5:])
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
