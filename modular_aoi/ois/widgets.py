"""
widgets.py — Reusable UI components.

FastLog, VideoWidget, AnnunciatorBanner, StatCard, LabelCanvas,
ROIZone, ROICanvas, ZoomableImageView, AdvancedTrainDialog,
ProjectDialog, ToastManager, ResultStrip, SysLogTab, _YieldChart.
"""
import json
import math
import os
import platform
import shutil
import sqlite3
import sys
import threading
import time
from collections import deque
from datetime import datetime

if __package__ in (None, ""):
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    __package__ = "ois"

from PySide6.QtWidgets import (
    QWidget, QLabel, QFrame, QHBoxLayout, QVBoxLayout, QGridLayout,
    QFormLayout, QInputDialog, QMessageBox, QPlainTextEdit, QPushButton, QComboBox, QDialog, QSpinBox,
    QDoubleSpinBox, QCheckBox, QLineEdit, QFileDialog, QScrollArea,
    QSizePolicy, QDialogButtonBox, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QMenu, QProgressBar, QSlider, QApplication
)
from PySide6.QtCore import (
    Qt, QThread, Signal, Slot, QTimer, QPropertyAnimation, QEasingCurve,
    QRect, Property, QPoint, QSize
)
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor,
    QIcon, QKeySequence, QTextCursor, QPalette, QBrush, QShortcut
)

from .theme import (
    BG_DEEP, BG_PANEL, BG_CARD, BG_CARD2, BG_BORDER, BG_BORDER2,
    CYAN, CYAN_DIM, CYAN_GLOW, GREEN, GREEN_DIM, GREEN_GLOW,
    AMBER, AMBER_DIM, AMBER_GLOW, RED, RED_DIM, RED_GLOW,
    PURPLE, TEXT_PRI, TEXT_SEC, TEXT_DIM, SIDEBAR_W,
    _F_MONO_8, _F_MONO_9, _F_MONO_10, _F_MONO_11, _F_MONO_12,
    _pen_dash, _card_style, make_card, make_sep, sec_lbl, set_fg
)
from .utils import (
    HAS_CV2, HAS_NP, HAS_SAHI, HAS_YOLO, DATA_ROOT,
    ProjectConfig, MASTER_CLASS_LIST, DEFAULT_AUG,
    build_golden_dataset, OG_MODEL, find_best_pt, match_detections, safe_predict,
    load_optimized_yolo,
    _AOIComp, _AOI_SAHI_TRIGGER, _AOI_SLICE_HW, _AOI_OVERLAP, _AOI_MATCH_R,
    _aoi_get_sahi_model, _aoi_bbox_overlap, _aoi_nms_by_centre, _aoi_infer_arr
)
from .threads import (
    AugThread, AutoLabelThread, CameraThread, InferenceThread,
    ThumbThread, TrainingThread
)
from .filters import FILTER_REGISTRY, FilterNode, AutoCalibrateWorker, apply_filters, run_roi

if HAS_CV2:
    import cv2
if HAS_NP:
    import numpy as np
if HAS_YOLO:
    from ultralytics import YOLO as _YOLO


class FastLog(QPlainTextEdit):
    """QPlainTextEdit log with 100ms batch flush. Replaces QTextEdit everywhere.
    QPlainTextEdit uses a simple block model vs QTextEdit rich-HTML model.
    Batching means one insertText + one reflow per 100ms instead of per line."""
    MAX_LINES=500
    def __init__(self,parent=None):
        super().__init__(parent); self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)   # Qt trims top natively
        self._buf=[]; self._timer=QTimer(self)
        self._timer.setSingleShot(True); self._timer.setInterval(100)
        self._timer.timeout.connect(self._flush)
    def append(self,text:str):
        self._buf.append(str(text))
        if not self._timer.isActive(): self._timer.start()
    def _flush(self):
        if not self._buf: return
        sb=self.verticalScrollBar(); at_bottom=sb.value()>=sb.maximum()-4
        block="\n".join(self._buf); self._buf.clear()
        cursor=self.textCursor(); cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(("\n" if self.document().blockCount()>1 else "")+block)
        if at_bottom: sb.setValue(sb.maximum())
    def clear(self):
        self._buf.clear(); self._timer.stop(); super().clear()

class VideoWidget(QLabel):
    def __init__(self):
        super().__init__(); self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400,300); self.setObjectName("videoWidget")
        self.setStyleSheet(f"#videoWidget{{background:#1A1A2E;border-radius:6px;}}")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        self._raw_px=None; self._dets=[]; self._golden=[]; self._missing=set(); self._roi_res=[]
        self._show_idle()
    def _show_idle(self):
        W,H=max(400,self.width()),max(300,self.height())
        px=QPixmap(W,H); px.fill(QColor("#1A1F2E")); p=QPainter(px)
        # Subtle dot grid
        dot_col=QColor(60,70,100,80)
        p.setPen(QPen(dot_col,1))
        for gx in range(0,W,32):
            for gy in range(0,H,32):
                p.drawPoint(gx,gy)
        # Faint centre crosshair
        cx,cy=W//2,H//2
        cross_col=QColor(CYAN); cross_col.setAlpha(30)
        p.setPen(QPen(cross_col,1,Qt.PenStyle.DashLine))
        p.drawLine(0,cy,W,cy); p.drawLine(cx,0,cx,H)
        # Corner scan brackets
        cs=24; bdr_col=QColor(CYAN); bdr_col.setAlpha(70)
        p.setPen(QPen(bdr_col,2))
        for bx,by,dx,dy in [(12,12,1,1),(W-12-cs,12,-1,1),(12,H-12-cs,1,-1),(W-12-cs,H-12-cs,-1,-1)]:
            p.drawLine(bx,by,bx+cs*dx,by); p.drawLine(bx,by,bx,by+cs*dy)
        # Centre circle target
        tgt=QColor(CYAN); tgt.setAlpha(25)
        p.setPen(QPen(tgt,1)); p.setBrush(Qt.BrushStyle.NoBrush)
        for r in (20,40): p.drawEllipse(cx-r,cy-r,r*2,r*2)
        # Text
        p.setPen(QPen(QColor(TEXT_DIM))); p.setFont(_F_MONO_11)
        p.drawText(QRect(0,0,W,H),Qt.AlignmentFlag.AlignCenter,"No camera feed\nStart inspection to begin")
        p.end(); self.setPixmap(px)
    def resizeEvent(self,e):
        super().resizeEvent(e)
        if self._raw_px is None: self._show_idle()
        else: self._compose()
    def set_frame(self,qimg):
        new_px=QPixmap.fromImage(qimg)
        if self._raw_px and new_px.size()==self._raw_px.size():
            self._raw_px=new_px  # same size — reuse cached scale
        else:
            self._raw_px=new_px; self._sc_cache=None  # size changed, invalidate
        self._compose()
    def set_detections(self,dets,golden,missing_indices):
        self._dets=dets; self._golden=golden; self._missing=set(missing_indices)  # set of int golden-slot indices
        if self._raw_px: self._compose()
    def set_roi_results(self, rr):
        pass
    def clear(self):
        self._raw_px=None; self._dets=[]; self._golden=[]; self._missing=set(); self._show_idle()
    def _compose(self):
        W,H=self.width(),self.height()
        if W<10 or H<10 or self._raw_px is None: return
        sc=self._raw_px.scaled(W,H,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.FastTransformation)
        x0=(W-sc.width())//2; y0=(H-sc.height())//2
        canvas=QPixmap(W,H); canvas.fill(QColor(BG_CARD)); p=QPainter(canvas); p.drawPixmap(x0,y0,sc)
        pw,ph=sc.width(),sc.height(); rw,rh=self._raw_px.width(),self._raw_px.height()
        sx=pw/max(1,rw); sy=ph/max(1,rh)
        for idx,g in enumerate(self._golden):
            x1,y1,x2,y2=g["xyxy"]
            col=QColor(RED) if idx in self._missing else QColor(50,50,70)
            p.setPen(QPen(col,1,Qt.PenStyle.DashLine)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(int(x1*sx)+x0,int(y1*sy)+y0,int((x2-x1)*sx),int((y2-y1)*sy))
        p.setFont(_F_MONO_9)
        for d in self._dets:
            x1,y1,x2,y2=d["xyxy"]; c=d.get("conf",1.0)
            col=QColor(GREEN) if c>=0.75 else (QColor(AMBER) if c>=0.5 else QColor(RED))
            s1x=int(x1*sx)+x0; s1y=int(y1*sy)+y0; s2x=int(x2*sx)+x0; s2y=int(y2*sy)+y0
            p.setPen(QPen(col,2)); p.setBrush(QBrush(QColor(col.red(),col.green(),col.blue(),18)))
            p.drawRect(s1x,s1y,s2x-s1x,s2y-s1y); p.setPen(QPen(col))
            p.drawText(s1x,s1y-4,f"{d['name']} {c:.2f}")
        # Corner brackets (scan frame effect)
        cm=QColor(CYAN); cm.setAlpha(140); p.setPen(QPen(cm,2)); cs=18
        for cx2,cy2 in [(10,10),(W-10-cs,10),(10,H-10-cs),(W-10-cs,H-10-cs)]:
            p.drawLine(cx2,cy2,cx2+cs,cy2); p.drawLine(cx2,cy2,cx2,cy2+cs)
        # Live indicator dot
        lc=QColor(GREEN) if self._raw_px else QColor(TEXT_DIM)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(lc)); p.drawEllipse(W-18,8,10,10)
        # Detection count badge (top-left)
        if self._dets:
            p.setFont(QFont('Consolas',9,QFont.Weight.Bold))
            badge=f'{len(self._dets)} det'
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(0,0,0,140))); p.drawRoundedRect(8,H-26,72,18,4,4)
            p.setPen(QPen(QColor(CYAN))); p.drawText(12,H-12,badge)
        # Arrow pointers for missing component slots — draw after the boxes so arrows are on top
        if self._missing:
            arr_pen=QPen(QColor(RED),2); arr_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(arr_pen)
            for idx in sorted(self._missing):
                if idx>=len(self._golden): continue
                x1,y1,x2,y2=self._golden[idx]["xyxy"]
                cx=int((x1+x2)/2*sx)+x0
                top=int(y1*sy)+y0
                tip=max(top-4,16)        # arrowhead tip (just above box top)
                stem=max(tip-14,4)       # stem top
                aw=8
                p.drawLine(cx,stem,cx,tip)              # vertical stem ↓
                p.drawLine(cx-aw,tip-aw,cx,tip)         # left wing
                p.drawLine(cx+aw,tip-aw,cx,tip)         # right wing
        # Missing components warning banner (self._missing is a set of golden slot indices)
        if self._missing:
            _names=[self._golden[i]["name"] for i in sorted(self._missing) if i<len(self._golden)]
            msg='MISSING: '+', '.join(_names[:3])
            if len(_names)>3: msg+=f' +{len(_names)-3}'
            # FIX: set the banner font BEFORE measuring text width so the
            # background rect is sized for the font actually used to draw it.
            # Previously fontMetrics() was called before setFont(), measuring
            # with the prior (smaller) font and clipping the text on long names.
            p.setFont(QFont('Consolas',10,QFont.Weight.Bold))
            tw2=p.fontMetrics().boundingRect(msg).width()
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(QColor(180,20,20,200)))
            p.drawRoundedRect(W//2-tw2//2-8,H-30,tw2+16,22,4,4)
            p.setPen(QPen(QColor('#ffffff')))
            p.drawText(QRect(0,H-30,W,22),Qt.AlignmentFlag.AlignCenter,msg)
        p.end(); self.setPixmap(canvas)

class AnnunciatorBanner(QLabel):
    _SS={
        "ready":(f"background:{BG_CARD2};"
                 f"color:{TEXT_SEC};font-size:12px;font-weight:bold;font-family:'Consolas';"
                 f"letter-spacing:4px;border-radius:8px;border:1px solid {BG_BORDER};padding:12px 16px;"),
        "setup":(f"background:{PURPLE}18;"
                 f"color:{PURPLE};font-size:12px;font-weight:bold;font-family:'Consolas';"
                 f"letter-spacing:4px;border-radius:8px;border:1px solid {PURPLE}44;padding:12px 16px;"),
        "pass":(f"background:{GREEN_DIM};"
                f"color:{GREEN};font-size:15px;font-weight:bold;font-family:'Consolas';"
                f"letter-spacing:8px;border-radius:8px;border:2px solid {GREEN}66;padding:12px 16px;"),
        "fail":(f"background:{RED_DIM};"
                f"color:{RED};font-size:15px;font-weight:bold;font-family:'Consolas';"
                f"letter-spacing:6px;border-radius:8px;border:2px solid {RED}66;padding:12px 16px;"),
    }
    _TX={"ready":"●  SYSTEM READY","setup":"◆  SETUP MODE  —  PRESS  [S]  TO SAVE MASTER","pass":"✔  PASS"}
    def __init__(self):
        super().__init__(); self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(62); self._state=None; self._text=None; self._set("ready",self._TX["ready"])
    def _set(self,state,text):
        if state!=self._state: self.setStyleSheet(self._SS[state]); self._state=state
        if text!=self._text: self.setText(text); self._text=text
    def set_ready(self): self._set("ready",self._TX["ready"])
    def set_setup(self): self._set("setup",self._TX["setup"])
    def set_pass(self): self._set("pass",self._TX["pass"])
    def set_fail(self,r=""): self._set("fail",f"✘  FAIL  —  {r.upper()[:44]}" if r else "✘  FAIL")
    def reset(self): self.set_ready()

class StatCard(QFrame):
    # FIX 2: equality gate on set_value() - no setText if value unchanged
    def __init__(self,label,value="0",color=CYAN):
        super().__init__(); oid=f"sc_{label.replace(' ','_')}"
        self.setObjectName(oid); self._color=color; self._oid=oid
        self.setStyleSheet(
            f"#{oid}{{background:{BG_PANEL};"
            f"border:1px solid {BG_BORDER};border-top:3px solid {color};border-radius:8px;}}"
        )
        lay=QVBoxLayout(self); lay.setContentsMargins(16,12,16,14); lay.setSpacing(4)
        self._val=QLabel(value); self._val.setObjectName(f"{oid}_v")
        self._val.setStyleSheet(
            f"#{oid}_v{{color:{color};font-size:32px;font-weight:bold;"
            f"font-family:'Consolas';border:none;letter-spacing:-1px;}}"
        )
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl=QLabel(label); lbl.setObjectName(f"{oid}_l")
        lbl.setStyleSheet(
            f"#{oid}_l{{color:{TEXT_SEC};font-size:9px;letter-spacing:2px;"
            f"font-weight:bold;border:none;font-family:'Consolas';}}"
        )
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._val); lay.addWidget(lbl)
        self._last=value; self._flash_timer=QTimer(self); self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._unflash)

    def set_value(self,v):
        s=str(v)
        if s==self._last: return
        self._val.setText(s); self._last=s
        # Brief border-glow on change
        self.setStyleSheet(
            f"#{self._oid}{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {BG_CARD2},stop:1 {BG_CARD});"
            f"border:1px solid {self._color}55;border-top:3px solid {self._color};border-radius:8px;}}"
        )
        self._flash_timer.start(300)

    def _unflash(self):
        self.setStyleSheet(
            f"#{self._oid}{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {BG_CARD2},stop:1 {BG_CARD});"
            f"border:1px solid {BG_BORDER2};border-top:3px solid {self._color};border-radius:8px;}}"
        )


class LabelCanvas(QWidget):
    box_added=Signal(list)
    COLORS={0:"#FFA500",1:"#FF3333",2:"#AA00FF",3:"#00FF00",4:"#FFFF00",5:"#00FFFF",8:"#3366FF",11:"#FF00FF"}
    DEF_COL="#CCCCCC"
    def __init__(self):
        super().__init__(); self.setMouseTracking(True); self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self._px=QPixmap(); self._ow=self._oh=1; self._scale=1.0; self._ox=self._oy=0.0
        self._boxes=[]; self._sel_cls=0; self._drawing=False; self._sx=self._sy=self._mx=self._my=0
        self._scaled=None; self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent,True)
        self._rtimer=QTimer(); self._rtimer.setSingleShot(True); self._rtimer.timeout.connect(self._fit)
        # Pan mode
        self._pan_mode=False; self._panning=False; self._pan_sx=self._pan_sy=0; self._pan_ox=self._pan_oy=0.0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def set_pan_mode(self,on):
        self._pan_mode=on
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor if on else Qt.CursorShape.CrossCursor))
    def keyPressEvent(self,e):
        if e.key()==Qt.Key.Key_Space and not e.isAutoRepeat() and not self._pan_mode:
            self._pan_mode=True; self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else: super().keyPressEvent(e)
    def keyReleaseEvent(self,e):
        if e.key()==Qt.Key.Key_Space and not e.isAutoRepeat() and not getattr(self,'_btn_pan_checked',False):
            self._pan_mode=False; self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else: super().keyReleaseEvent(e)
    def load_image(self,path):
        self._px=QPixmap(); self._px.load(path); self._scaled=None
        self._ow=self._px.width() if not self._px.isNull() else 1
        self._oh=self._px.height() if not self._px.isNull() else 1
        self._boxes=[]; self._fit()
    def load_labels(self,txt):
        self._boxes=[]
        if os.path.exists(txt):
            for l in open(txt):
                try: self._boxes.append(list(map(float,l.strip().split())))
                except: pass
        self.update()
    def save_labels(self,txt):
        if not self._boxes:
            if not os.path.exists(txt): return
        os.makedirs(os.path.dirname(os.path.abspath(txt)),exist_ok=True)
        with open(txt,"w") as f:
            for b in self._boxes: f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
    def clear_labels(self): self._boxes=[]; self.update()
    def set_class(self,idx): self._sel_cls=idx
    def get_boxes(self): return self._boxes
    def _fit(self):
        if self._px.isNull(): self.update(); return
        cw,ch=max(1,self.width()),max(1,self.height())
        self._scale=min(cw/self._ow,ch/self._oh)
        self._ox=(cw-self._ow*self._scale)/2; self._oy=(ch-self._oh*self._scale)/2
        self._scaled=None; self.update()
    def resizeEvent(self,e): self._rtimer.start(60)
    def paintEvent(self,e):
        p=QPainter(self); W,H=self.width(),self.height()
        p.fillRect(0,0,W,H,QColor(BG_CARD))
        if self._px.isNull(): return
        dw=int(self._ow*self._scale); dh=int(self._oh*self._scale)
        ox=int(self._ox); oy=int(self._oy)
        if self._scaled is None or self._scaled.width()!=dw or self._scaled.height()!=dh:
            self._scaled=self._px.scaled(dw,dh,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.FastTransformation)
        p.drawPixmap(ox,oy,self._scaled)
        sc=self._scale; p.setFont(_F_MONO_8)
        for b in self._boxes:
            cls=int(b[0]); cx,cy,bw,bh=b[1],b[2],b[3],b[4]
            px2=int((cx-bw/2)*self._ow*sc+ox); py2=int((cy-bh/2)*self._oh*sc+oy)
            pw2=int(bw*self._ow*sc); ph2=int(bh*self._oh*sc)
            col=QColor(self.COLORS.get(cls,self.DEF_COL))
            p.setPen(QPen(col,2)); p.setBrush(QBrush(QColor(col.red(),col.green(),col.blue(),30)))
            p.drawRect(px2,py2,pw2,ph2); p.setPen(QPen(col)); p.drawText(px2+2,py2-3,str(cls))
        if self._drawing:
            x1,y1=min(self._sx,self._mx),min(self._sy,self._my)
            p.setPen(_pen_dash()); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(x1,y1,abs(self._mx-self._sx),abs(self._my-self._sy))
    def mousePressEvent(self,e):
        mx,my=e.position().x(),e.position().y()
        # Middle mouse always pans regardless of mode
        if e.button()==Qt.MouseButton.MiddleButton:
            self._panning=True; self._pan_sx=mx; self._pan_sy=my
            self._pan_ox=self._ox; self._pan_oy=self._oy
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor)); return
        if e.button()==Qt.MouseButton.RightButton:
            if self._pan_mode: return
            for i,b in enumerate(self._boxes):
                cls,cx,cy,bw,bh=b; sc=self._scale; ox=self._ox; oy=self._oy
                px2=int((cx-bw/2)*self._ow*sc+ox); py2=int((cy-bh/2)*self._oh*sc+oy)
                pw2=int(bw*self._ow*sc); ph2=int(bh*self._oh*sc)
                if px2<=mx<=px2+pw2 and py2<=my<=py2+ph2: self._boxes.pop(i); self.update(); return
        elif e.button()==Qt.MouseButton.LeftButton:
            if self._pan_mode:
                self._panning=True; self._pan_sx=mx; self._pan_sy=my
                self._pan_ox=self._ox; self._pan_oy=self._oy
                self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            else:
                self._drawing=True; self._sx=self._mx=int(mx); self._sy=self._my=int(my)
    def mouseMoveEvent(self,e):
        mx,my=e.position().x(),e.position().y()
        self._mx=int(mx); self._my=int(my)
        if self._panning:
            self._ox=self._pan_ox+(mx-self._pan_sx)
            self._oy=self._pan_oy+(my-self._pan_sy)
            self._scaled=None; self.update(); return
        if self._drawing: self.update()
    def mouseReleaseEvent(self,e):
        if e.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton) and self._panning:
            self._panning=False
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor if self._pan_mode else Qt.CursorShape.CrossCursor))
        elif e.button()==Qt.MouseButton.LeftButton and self._drawing:
            self._drawing=False
            x1,y1=min(self._sx,self._mx),min(self._sy,self._my)
            x2,y2=max(self._sx,self._mx),max(self._sy,self._my)
            if (x2-x1)>8 and (y2-y1)>8:
                ox,oy,sc=self._ox,self._oy,self._scale
                cx=((x1+x2)/2-ox)/(self._ow*sc); cy=((y1+y2)/2-oy)/(self._oh*sc)
                bw=(x2-x1)/(self._ow*sc); bh=(y2-y1)/(self._oh*sc)
                box=[self._sel_cls,max(0,min(1,cx)),max(0,min(1,cy)),max(0,min(1,bw)),max(0,min(1,bh))]
                self._boxes.append(box); self.box_added.emit(box)
            self.update()
    def wheelEvent(self,e):
        factor=1.12 if e.angleDelta().y()>0 else (1/1.12)
        mx,my=e.position().x(),e.position().y()
        old=self._scale; new_scale=max(0.1,min(16,old*factor))
        # Zoom toward cursor position
        self._ox=mx-(mx-self._ox)*(new_scale/old)
        self._oy=my-(my-self._oy)*(new_scale/old)
        self._scale=new_scale; self._scaled=None; self.update()


# ROIZone and ROICanvas classes scraped


# ─────────────────────────────────────────────────────────────────────────────
# DIALOGS
# ─────────────────────────────────────────────────────────────────────────────
class AdvancedTrainDialog(QDialog):
    def __init__(self,params,parent=None):
        super().__init__(parent); self.setWindowTitle("Advanced Training Settings")
        self.setMinimumSize(500,540); self._p=dict(params)
        lay=QVBoxLayout(self); lay.setContentsMargins(16,12,16,12); lay.setSpacing(8)
        lay.addWidget(sec_lbl("ADVANCED TRAINING SETTINGS"))
        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus); scroll.setObjectName("adv_scroll")
        scroll.setStyleSheet("#adv_scroll{border:none;}")
        inner=QWidget(); inner.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        form=QFormLayout(inner); form.setSpacing(8); form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._w={}
        def row(key,label,lo,hi,dec=None,tip=""):
            if dec is not None: w=QDoubleSpinBox(); w.setRange(lo,hi); w.setDecimals(dec); w.setValue(float(self._p.get(key,lo))); w.setSingleStep(10**(-dec))
            else: w=QSpinBox(); w.setRange(int(lo),int(hi)); w.setValue(int(self._p.get(key,lo)))
            w.setFixedWidth(100)
            if tip: w.setToolTip(tip)
            self._w[key]=w; form.addRow(label,w)
        form.addRow(sec_lbl("TRAINING"),QLabel(""))
        row("epochs","Epochs",1,2000,None,"Training passes"); row("batch","Batch",1,64,None,"GPU batch")
        row("imgsz","Image Size",320,1280,None,"Mult of 32"); row("patience","Patience",5,200,None,"Early stop")
        row("workers","Workers",0,16,None,"0=safe on Windows")
        form.addRow(sec_lbl("AUGMENTATION"),QLabel(""))
        row("mosaic","Mosaic",0.0,1.0,3); row("mixup","MixUp",0.0,1.0,3); row("copy_paste","Copy-Paste",0.0,1.0,3)
        row("hsv_h","HSV Hue",0.0,0.5,4); row("hsv_s","HSV Sat",0.0,1.0,3); row("hsv_v","HSV Val",0.0,1.0,3)
        row("degrees","Rotation",0.0,90.0,1); row("translate","Translate",0.0,0.5,3); row("scale","Scale",0.1,2.0,2)
        row("shear","Shear",0.0,10.0,2); row("perspective","Perspective",0.0,0.001,6); row("erasing","Erasing",0.0,1.0,2)
        scroll.setWidget(inner); lay.addWidget(scroll,1)
        rr=QHBoxLayout(); b_rst=QPushButton("Reset Defaults"); b_rst.clicked.connect(self._reset)
        rr.addWidget(b_rst); rr.addStretch(); lay.addLayout(rr)
        bbox=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        bbox.accepted.connect(self.accept); bbox.rejected.connect(self.reject); lay.addWidget(bbox)
    def _reset(self):
        for k,w in self._w.items():
            v=DEFAULT_AUG.get(k)
            if v is not None: w.setValue(v)
    def get_params(self): return {k:w.value() for k,w in self._w.items()}

class ProjectDialog(QDialog):
    project_opened=Signal(str)
    def __init__(self,cfg,parent=None):
        super().__init__(parent); self._cfg=cfg; self.setWindowTitle("Project Manager"); self.setMinimumSize(460,320)
        lay=QVBoxLayout(self); lay.setContentsMargins(14,14,14,14); lay.setSpacing(8)
        lay.addWidget(sec_lbl("PROJECT MANAGER"))
        br=QHBoxLayout()
        for txt,slot,col in [("+ New",self._new,CYAN),("Open",self._load,AMBER),("Delete",self._delete,RED)]:
            b=QPushButton(txt); b.setFixedHeight(32); oid=f"pdlg_{txt[0]}"
            b.setObjectName(oid); b.setStyleSheet(f"#{oid}{{background:{BG_CARD};color:{col};border:1px solid {col};border-radius:5px;padding:0 12px;}}")
            b.clicked.connect(slot); br.addWidget(b)
        lay.addLayout(br); lay.addWidget(sec_lbl("RECENT PROJECTS"))
        self._lst=QListWidget(); self._lst.setObjectName("plist")
        self._lst.setStyleSheet(f"#plist{{background:{BG_PANEL};border:1px solid {BG_BORDER};border-radius:6px;}}")
        self._lst.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._lst.itemDoubleClicked.connect(lambda it:self._open(os.path.join(DATA_ROOT,"Projects",it.text())))
        lay.addWidget(self._lst,1)
        self._al=QLabel("Active: None"); self._al.setObjectName("pal"); self._al.setStyleSheet(f"#pal{{color:{TEXT_DIM};font-family:Consolas;font-size:11px;border:none;}}")
        lay.addWidget(self._al)
        bbox=QDialogButtonBox(QDialogButtonBox.StandardButton.Close); bbox.rejected.connect(self.accept); lay.addWidget(bbox)
        self._refresh()
    def _refresh(self):
        self._lst.clear(); p=os.path.join(DATA_ROOT,"Projects")
        if os.path.exists(p):
            for d in sorted(os.listdir(p)):
                if os.path.isdir(os.path.join(p,d)): self._lst.addItem(d)
    def _new(self):
        name,ok=QInputDialog.getText(self,"New Project","Project name:")
        if ok and name.strip():
            path=os.path.join(DATA_ROOT,"Projects",name.strip())
            for sub in ["golden_boards","labels","models","augmented_images","augmented_labels","inbox"]:
                os.makedirs(os.path.join(path,sub),exist_ok=True)
            self._open(path)
    def _load(self):
        p=QFileDialog.getExistingDirectory(self,"Select Project",os.path.join(DATA_ROOT,"Projects"))
        if p: self._open(p)
    def _delete(self):
        item=self._lst.currentItem()
        if not item: return
        if QMessageBox.question(self,"Delete",f"Delete '{item.text()}'? Cannot undo!",
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)==QMessageBox.StandardButton.Yes:
            try: shutil.rmtree(os.path.join(DATA_ROOT,"Projects",item.text())); self._refresh()
            except Exception as e: QMessageBox.critical(self,"Error",str(e))
    def _open(self,path):
        if not os.path.exists(path): return
        self._cfg.load(path); name=os.path.basename(path)
        self._al.setText(f"Active: {name}"); self._al.setStyleSheet("#pal{color:{GREEN};font-family:Consolas;font-size:11px;border:none;}")
        self.project_opened.emit(path); self._refresh()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RUN
# ─────────────────────────────────────────────────────────────────────────────

class ZoomableImageView(QWidget):
    """
    Proper interactive image viewer:
      • Scroll wheel  → zoom in/out (centred on cursor)
      • Left drag     → pan
      • Double-click  → fit to window
      • +/- buttons   → zoom
      Displays a QPixmap at arbitrary zoom with smooth rendering.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ziv")
        self.setStyleSheet(f"#ziv{{background:{BG_CARD};border-radius:6px;border:1px solid {BG_BORDER2};}}")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(300)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMouseTracking(True)

        self._px: QPixmap = QPixmap()          # full-resolution source
        self._zoom: float = 1.0
        self._offset      = (0.0, 0.0)         # (ox, oy) pan offset in widget coords
        self._drag        = False
        self._drag_start  = (0, 0)
        self._drag_offset = (0.0, 0.0)

        # Peer-sync support: set both views' _peer to each other to sync zoom/pan
        self._peer         = None
        self._sync_blocked = False
        self._fit_zoom     = 1.0    # zoom level at last _fit() — used for sync ratio
        self._idle_text    = "No image loaded"   # override per instance if needed

        # Zoom controls overlay
        self._zoom_lbl = QLabel("100%", self)
        self._zoom_lbl.setStyleSheet(
            f"color:{CYAN};font-family:Consolas;font-size:10px;font-weight:bold;"
            f"background:{BG_CARD2}cc;border-radius:4px;padding:2px 6px;border:none;"
        )
        self._zoom_lbl.move(8, 8)
        self._zoom_lbl.adjustSize()

    def load_bytes(self, img_bytes: bytes):
        """Load image from JPEG/PNG bytes."""
        if not img_bytes:
            self._px = QPixmap(); self.update(); return
        qimg = QImage.fromData(img_bytes)
        if qimg.isNull(): return
        self._px = QPixmap.fromImage(qimg)
        self._fit()

    def load_pixmap(self, px: QPixmap):
        self._px = px; self._fit()

    def load_frame(self, bgr):
        if not HAS_CV2 or not HAS_NP: return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self._px = QPixmap.fromImage(qimg)
        self._fit()

    def load_image(self, path):
        self._px = QPixmap(path)
        self._fit()

    def clear(self):
        self._px = QPixmap(); self.update()

    def _fit(self):
        """Fit image to widget, centred."""
        if self._px.isNull(): return
        W, H = self.width(), self.height()
        if W < 2 or H < 2: return
        sx = W / self._px.width(); sy = H / self._px.height()
        self._zoom = min(sx, sy) * 0.97
        self._fit_zoom = self._zoom   # record for sync ratio conversion
        self._center()
        self.update()
        self._zoom_lbl.setText(f"{self._zoom*100:.0f}%")

    def _center(self):
        W, H = self.width(), self.height()
        pw = self._px.width() * self._zoom
        ph = self._px.height() * self._zoom
        self._offset = ((W - pw) / 2, (H - ph) / 2)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._px.isNull(): self._fit()
        self._zoom_lbl.move(8, 8)

    def paintEvent(self, e):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#1A1F2E"))
        if self._px.isNull():
            # Subtle dot-grid
            dot_col = QColor(60, 70, 100, 70)
            p.setPen(QPen(dot_col, 1))
            for gx in range(0, W, 28):
                for gy in range(0, H, 28):
                    p.drawPoint(gx, gy)
            # Faint crosshair
            cx, cy = W//2, H//2
            cross = QColor(CYAN); cross.setAlpha(28)
            p.setPen(QPen(cross, 1, Qt.PenStyle.DashLine))
            p.drawLine(0, cy, W, cy); p.drawLine(cx, 0, cx, H)
            # Corner brackets
            cs = 18; bdr = QColor(CYAN); bdr.setAlpha(60)
            p.setPen(QPen(bdr, 2))
            for bx,by,dx,dy in [(6,6,1,1),(W-6-cs,6,-1,1),(6,H-6-cs,1,-1),(W-6-cs,H-6-cs,-1,-1)]:
                p.drawLine(bx,by,bx+cs*dx,by); p.drawLine(bx,by,bx,by+cs*dy)
            # Centre target ring
            tgt = QColor(CYAN); tgt.setAlpha(20)
            p.setPen(QPen(tgt, 1)); p.setBrush(Qt.BrushStyle.NoBrush)
            r = 16; p.drawEllipse(cx-r, cy-r, r*2, r*2)
            # Text
            p.setPen(QPen(QColor(TEXT_DIM)))
            p.setFont(_F_MONO_11)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._idle_text)
            p.end(); return
        ox, oy = self._offset
        dw = int(self._px.width()  * self._zoom)
        dh = int(self._px.height() * self._zoom)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                        self._zoom < 2.0)
        p.drawPixmap(int(ox), int(oy), dw, dh, self._px)
        p.end()

    def _sync_from(self, zoom: float, offset: tuple):
        """Called by peer to set zoom/pan without triggering infinite loop."""
        if self._sync_blocked: return
        self._sync_blocked = True
        self._zoom = zoom
        self._offset = offset
        self._zoom_lbl.setText(f"{zoom*100:.0f}%")
        self.update()
        self._sync_blocked = False

    def _push_sync(self):
        """Push current zoom/pan state to peer (if any)."""
        if self._peer and not self._sync_blocked:
            self._peer._sync_from(self._zoom, self._offset)

    def zoom_to_component(self, cx_norm: float, cy_norm: float,
                           w_norm: float, h_norm: float):
        """Zoom so the component + generous context fills the view.
        We inflate the effective bbox by 4× so there is plenty of surrounding
        board visible — avoids the "face pressed against IC" problem."""
        if self._px.isNull(): return
        iw, ih = self._px.width(), self._px.height()
        # Expand the region of interest 4× around the component centre
        roi_w = min(1.0, w_norm * 4.0)
        roi_h = min(1.0, h_norm * 4.0)
        comp_w_px = max(1, roi_w * iw)
        comp_h_px = max(1, roi_h * ih)
        vW, vH = self.width(), self.height()
        zoom_x = vW / comp_w_px
        zoom_y = vH / comp_h_px
        self._zoom = min(zoom_x, zoom_y)
        self._zoom = max(0.1, min(12.0, self._zoom))
        cx_px = cx_norm * iw
        cy_px = cy_norm * ih
        self._offset = (vW / 2 - cx_px * self._zoom,
                        vH / 2 - cy_px * self._zoom)
        self._zoom_lbl.setText(f"{self._zoom*100:.0f}%")
        self.update()
        self._push_sync()

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        factor = 1.15 if delta > 0 else (1/1.15)
        mx, my = e.position().x(), e.position().y()
        ox, oy = self._offset
        # Zoom centred on cursor
        new_zoom = max(0.05, min(32.0, self._zoom * factor))
        ratio = new_zoom / self._zoom
        self._offset = (mx - (mx - ox) * ratio,
                        my - (my - oy) * ratio)
        self._zoom = new_zoom
        self._zoom_lbl.setText(f"{self._zoom*100:.0f}%")
        self.update()
        self._push_sync()

    def mousePressEvent(self, e):
        if e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag = True
            self._drag_start  = (e.position().x(), e.position().y())
            self._drag_offset = self._offset
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def mouseMoveEvent(self, e):
        if self._drag:
            dx = e.position().x() - self._drag_start[0]
            dy = e.position().y() - self._drag_start[1]
            self._offset = (self._drag_offset[0] + dx,
                            self._drag_offset[1] + dy)
            self.update()
            self._push_sync()

    def mouseReleaseEvent(self, e):
        if e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag = False
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def mouseDoubleClickEvent(self, e):
        self._fit()   # double-click resets to fit
        self._push_sync()

    def keyPressEvent(self, e):
        # Don't swallow navigation keys — let the parent tab handle them.
        e.ignore()

    def zoom_in(self):
        mx, my = self.width()/2, self.height()/2
        ox, oy = self._offset
        new_zoom = min(32.0, self._zoom * 1.3)
        ratio = new_zoom / self._zoom
        self._offset = (mx - (mx-ox)*ratio, my - (my-oy)*ratio)
        self._zoom = new_zoom
        self._zoom_lbl.setText(f"{self._zoom*100:.0f}%"); self.update()
        self._push_sync()

    def zoom_out(self):
        mx, my = self.width()/2, self.height()/2
        ox, oy = self._offset
        new_zoom = max(0.05, self._zoom / 1.3)
        ratio = new_zoom / self._zoom
        self._offset = (mx - (mx-ox)*ratio, my - (my-oy)*ratio)
        self._zoom = new_zoom
        self._zoom_lbl.setText(f"{self._zoom*100:.0f}%"); self.update()
        self._push_sync()

# ─────────────────────────────────────────────────────────────────────────────

class AdvancedTrainDialog(QDialog):
    def __init__(self,params,parent=None):
        super().__init__(parent); self.setWindowTitle("Advanced Training Settings")
        self.setMinimumSize(500,540); self._p=dict(params)
        lay=QVBoxLayout(self); lay.setContentsMargins(16,12,16,12); lay.setSpacing(8)
        lay.addWidget(sec_lbl("ADVANCED TRAINING SETTINGS"))
        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus); scroll.setObjectName("adv_scroll")
        scroll.setStyleSheet("#adv_scroll{border:none;}")
        inner=QWidget(); inner.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        form=QFormLayout(inner); form.setSpacing(8); form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._w={}
        def row(key,label,lo,hi,dec=None,tip=""):
            if dec is not None: w=QDoubleSpinBox(); w.setRange(lo,hi); w.setDecimals(dec); w.setValue(float(self._p.get(key,lo))); w.setSingleStep(10**(-dec))
            else: w=QSpinBox(); w.setRange(int(lo),int(hi)); w.setValue(int(self._p.get(key,lo)))
            w.setFixedWidth(100)
            if tip: w.setToolTip(tip)
            self._w[key]=w; form.addRow(label,w)
        form.addRow(sec_lbl("TRAINING"),QLabel(""))
        row("epochs","Epochs",1,2000,None,"Training passes"); row("batch","Batch",1,64,None,"GPU batch")
        row("imgsz","Image Size",320,1280,None,"Mult of 32"); row("patience","Patience",5,200,None,"Early stop")
        row("workers","Workers",0,16,None,"0=safe on Windows")
        form.addRow(sec_lbl("AUGMENTATION"),QLabel(""))
        row("mosaic","Mosaic",0.0,1.0,3); row("mixup","MixUp",0.0,1.0,3); row("copy_paste","Copy-Paste",0.0,1.0,3)
        row("hsv_h","HSV Hue",0.0,0.5,4); row("hsv_s","HSV Sat",0.0,1.0,3); row("hsv_v","HSV Val",0.0,1.0,3)
        row("degrees","Rotation",0.0,90.0,1); row("translate","Translate",0.0,0.5,3); row("scale","Scale",0.1,2.0,2)
        row("shear","Shear",0.0,10.0,2); row("perspective","Perspective",0.0,0.001,6); row("erasing","Erasing",0.0,1.0,2)
        scroll.setWidget(inner); lay.addWidget(scroll,1)
        rr=QHBoxLayout(); b_rst=QPushButton("Reset Defaults"); b_rst.clicked.connect(self._reset)
        rr.addWidget(b_rst); rr.addStretch(); lay.addLayout(rr)
        bbox=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        bbox.accepted.connect(self.accept); bbox.rejected.connect(self.reject); lay.addWidget(bbox)
    def _reset(self):
        for k,w in self._w.items():
            v=DEFAULT_AUG.get(k)
            if v is not None: w.setValue(v)
    def get_params(self): return {k:w.value() for k,w in self._w.items()}

class ProjectDialog(QDialog):
    project_opened=Signal(str)
    def __init__(self,cfg,parent=None):
        super().__init__(parent); self._cfg=cfg; self.setWindowTitle("Project Manager"); self.setMinimumSize(460,320)
        lay=QVBoxLayout(self); lay.setContentsMargins(14,14,14,14); lay.setSpacing(8)
        lay.addWidget(sec_lbl("PROJECT MANAGER"))
        br=QHBoxLayout()
        for txt,slot,col in [("+ New",self._new,CYAN),("Open",self._load,AMBER),("Delete",self._delete,RED)]:
            b=QPushButton(txt); b.setFixedHeight(32); oid=f"pdlg_{txt[0]}"
            b.setObjectName(oid); b.setStyleSheet(f"#{oid}{{background:{BG_CARD};color:{col};border:1px solid {col};border-radius:5px;padding:0 12px;}}")
            b.clicked.connect(slot); br.addWidget(b)
        lay.addLayout(br); lay.addWidget(sec_lbl("RECENT PROJECTS"))
        self._lst=QListWidget(); self._lst.setObjectName("plist")
        self._lst.setStyleSheet(f"#plist{{background:{BG_PANEL};border:1px solid {BG_BORDER};border-radius:6px;}}")
        self._lst.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._lst.itemDoubleClicked.connect(lambda it:self._open(os.path.join(DATA_ROOT,"Projects",it.text())))
        lay.addWidget(self._lst,1)
        self._al=QLabel("Active: None"); self._al.setObjectName("pal"); self._al.setStyleSheet(f"#pal{{color:{TEXT_DIM};font-family:Consolas;font-size:11px;border:none;}}")
        lay.addWidget(self._al)
        bbox=QDialogButtonBox(QDialogButtonBox.StandardButton.Close); bbox.rejected.connect(self.accept); lay.addWidget(bbox)
        self._refresh()
    def _refresh(self):
        self._lst.clear(); p=os.path.join(DATA_ROOT,"Projects")
        if os.path.exists(p):
            for d in sorted(os.listdir(p)):
                if os.path.isdir(os.path.join(p,d)): self._lst.addItem(d)
    def _new(self):
        name,ok=QInputDialog.getText(self,"New Project","Project name:")
        if ok and name.strip():
            path=os.path.join(DATA_ROOT,"Projects",name.strip())
            for sub in ["golden_boards","labels","models","augmented_images","augmented_labels","inbox"]:
                os.makedirs(os.path.join(path,sub),exist_ok=True)
            self._open(path)
    def _load(self):
        p=QFileDialog.getExistingDirectory(self,"Select Project",os.path.join(DATA_ROOT,"Projects"))
        if p: self._open(p)
    def _delete(self):
        item=self._lst.currentItem()
        if not item: return
        if QMessageBox.question(self,"Delete",f"Delete '{item.text()}'? Cannot undo!",
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)==QMessageBox.StandardButton.Yes:
            try: shutil.rmtree(os.path.join(DATA_ROOT,"Projects",item.text())); self._refresh()
            except Exception as e: QMessageBox.critical(self,"Error",str(e))
    def _open(self,path):
        if not os.path.exists(path): return
        self._cfg.load(path); name=os.path.basename(path)
        self._al.setText(f"Active: {name}"); self._al.setStyleSheet("#pal{color:{GREEN};font-family:Consolas;font-size:11px;border:none;}")
        self.project_opened.emit(path); self._refresh()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RUN
# ─────────────────────────────────────────────────────────────────────────────
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
        self._pipe_filters=filters; self._pipe_rois=rois; self._pipe_active=bool(rois or filters); self._pipe_model=None
        # Invalidate golden cache — filters changed so the pre-filtered golden is stale
        self._golden_bgr_cache = None; self._golden_filter_sig = ()
        if model_path and os.path.exists(model_path) and HAS_YOLO:
            try: self._pipe_model=_YOLO(model_path, task="detect")
            except: pass
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
                    self._inspect_model = _YOLO(mp, task="detect")
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
        if not self._latest_dets: self._log.append("[!] No detections."); return
        path=os.path.join(self._cfg._path or ".",self._cfg.get("golden_file"))
        with open(path,"w") as f: json.dump([{"class":d["class"],"name":d["name"],"xyxy":d["xyxy"]} for d in self._latest_dets],f,indent=2)
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
                    res=safe_predict(model, img, conf=self._conf.value()/100, iou=0.35, imgsz=640, verbose=False, device='cpu')[0]

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
            if not self._latest_dets: self._log.append("[!] No detections to save."); return
            path=os.path.join(self._cfg._path or ".","golden_master.json")
            with open(path,"w") as f:
                json.dump([{"class":d["class"],"name":d["name"],"xyxy":d["xyxy"]} for d in self._latest_dets],f,indent=2)
            self.load_golden(path); self.banner.set_pass()
            self._log.append(f"[Master] Saved {len(self._latest_dets)} components from live camera")

    def _toggle(self):
        if not self._running: self._start()
        else: self._stop()

    def _start(self):
        self._running=True; self._log.clear()
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
        self._ai=InferenceThread(self._cfg,self._cam)
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
        self.video.set_frame(qimg)

    @Slot(list,float)
    def _on_result(self,dets,lat):
        """
        Display-only per frame. Counters NOT incremented here.
        Updates overlay, banner, and latency display.
        """
        self._latest_dets=dets
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
        if not self._pipe_active and self._golden:
            missing_idx=match_detections(self._golden,live,self._cfg.get("match_iou"),self._persist,self._cfg.get("persistence"))
            missing_names=[self._golden[i]["name"] for i in missing_idx if i<len(self._golden)]
            self.video.set_detections(live,self._golden,missing_idx)
            if missing_names!=self._last_missing:
                self._last_missing=missing_names
                if missing_names:
                    cnt=len(missing_names)
                    self.banner.set_fail(f"MISSING ×{cnt}: {missing_names[0]}{'…' if cnt>1 else ''}")
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
        if frame is None: self._log.append("[!] No frame to flag — start camera first."); return
        if not self._cfg._path: self._log.append("[!] No project loaded — cannot flag."); return
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save directly to golden_boards so it appears immediately in the Train tab
        d=os.path.join(self._cfg._path,"golden_boards")
        os.makedirs(d,exist_ok=True)
        out=os.path.join(d,f"flag_{ts}.jpg")
        if HAS_CV2:
            cv2.imwrite(out,frame)
            self._log.append(f"[{ts}] Flagged → golden_boards/flag_{ts}.jpg")
            self.flag_saved.emit(d)   # tell GoldenTab to reload
        else:
            self._log.append("[!] OpenCV not available — cannot save flag image.")

    def _reset(self):
        self._pass=self._fail=self._total=0; self._last_missing=[]
        for k in self._persist: self._persist[k]=0
        self._c_total.set_value(0); self._c_pass.set_value(0); self._c_fail.set_value(0)
        self._ybar.setValue(0); self._ypct.setText("YIELD: -"); self._log.clear()
        self.banner.reset(); self.video.set_roi_results([]); self._counter_dirty=False
        self._result_strip.clear_results(); self._result_times.clear()
        self._throughput_lbl.setText("— boards/min")

    def _on_cam_idx(self,_): pass  # camera bar removed; index set via Settings

    def _test_cam(self): pass  # camera bar removed

    def get_latest_frame(self): return self._cam.get_small_bgr() if self._cam else None
    def on_project_changed(self, path):
        # Stop camera if running
        if self._running: self._stop()
        # Full UI reset — counters, log, video, banner, history strip, golden master
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


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — TRAIN
# ─────────────────────────────────────────────────────────────────────────────
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
            try: self._cached_model=_YOLO(tp, task="detect"); self._cached_model_path=tp
            except Exception as e: QMessageBox.critical(self,"Error",str(e)); return
        self._auto_thread=AutoLabelThread(self._cached_model,
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




# ═════════════════════════════════════════════════════════════════════════════
# AOI ENGINE  — proven implementation (ported from pcb_aoi.py v10.1)
# Uses Component dataclass + scipy Hungarian matching for accuracy parity
# ═════════════════════════════════════════════════════════════════════════════
from dataclasses import dataclass as _dataclass
# math and copy are already imported at module top — no re-import needed here.
_math = math   # local alias keeps all AOI engine call-sites unchanged

# ── Constants ────────────────────────────────────────────────────────────────
_AOI_SAHI_TRIGGER = 1280
_AOI_SLICE_HW     = 640
_AOI_OVERLAP      = 0.20
_MIN_DIFF_THR     = 8.0
_MIN_POS_TOL      = 0.010
_AOI_MATCH_R      = 0.06
_AOI_MATCH_R_SAHI = 0.14   # larger: SAHI tile-merge jitter 3-5× standard YOLO
_AOI_SAHI_JITTER_MULT = 3.5  # pos_tol multiplier for SAHI boards
_AOI_DETECT_BAND  = 2.0

# ── Per-slot pixel disambiguation signals (ported from evaluator v3.6) ────────
# These thresholds were validated on 100-board runs; see evaluator changelog.
_AOI_SSIM_MISS_MAX      = 0.32  # SSIM below this → missing (fill has no structure)
_AOI_SSIM_WC_NODET      = 0.55  # SSIM this high even without same-label det → wrong_component
_AOI_VAR_FLAT           = 5.0   # patch variance below → empty Gaussian fill
_AOI_VAR_WC_CONFIRM     = 80.0  # patch variance above → a real component is present
_AOI_NCC_POL_B          = 0.05  # Zone A/B polarity NCC margin
_AOI_NCC_POL_IC         = 0.28  # Zone C IC margin  (near-symmetric ICs score ~0.36)
_AOI_NCC_POL_C          = 0.38  # Zone C cap/diode margin
_AOI_NCC_POL_STRONG     = 0.95  # Unconditional WPOL: true WPOL always ≥ 1.058; WCOM FPs ≤ 0.910
_AOI_NCC_POL_SSIM_NEG   =-0.33  # Rotated-cap branch: WCOM FP cap had ssim=-0.306 (blocked); true ≤ -0.336
_AOI_SSIM_POL_STRUCT    = 0.28  # Structural polarity fallback ssim gate  (catches near-sym ICs ssim≈0.30)
_AOI_TMPL_POL_STRUCT    = 0.25  # Structural polarity fallback tmpl gate  (C0 IC has tmpl=0.265)
_AOI_NCC_POL_STRUCT_MIN = 0.25  # Min NCC delta for structural fallback:  FP max=0.186, TP min=0.393
_AOI_CROSS_RESCUE_TMPL  = 0.20  # cross_polar_rescue tmpl floor:          WCOM donors score < 0.20
_AOI_CROSS_RESCUE_NCC   = 0.50  # cross_polar_rescue NCC delta floor:     FP had delta=0.401, TP ≥ 0.708
_AOI_SSIM_CROSS_MAX     = 0.45  # cross_polar_rescue SSIM upper bound:    WCOM caps score ssim > 0.45

# v3.9: Two-pass misalignment sweep (ported from evaluator v3.9 Fix P)
_MISALIGN_RECOVER_NCC_MIN    = 0.60
_MISALIGN_RECOVER_GAIN_MIN   = 0.06
_MISALIGN_RECOVER_ANGLE_MIN  = 10.0
_MISALIGN_FINE_WINDOW        = 8
_MISALIGN_FINE_STEP          = 1

# v3.9 Fix N: Gradient Orientation Histogram polarity
_GOH_POLAR_BINS          = 12
_GOH_POLAR_MIN_DELTA     = 0.08
_GOH_ASYM_VOTE_BAND_LO  = 0.05
_GOH_ASYM_VOTE_BAND_HI  = 0.10
_ASYM_POLAR_MIN          = 0.025

# v3.9 Fix O: Edge-density rescue
_EDGE_DENSITY_WC_MIN     = 0.045

_AOI_COMPATIBLE = {
    ("misaligned","missing"),    ("missing","misaligned"),
    ("wrong_component","missing"),("missing","wrong_component"),
    ("wrong_polarity","missing"), ("missing","wrong_polarity"),
    ("wrong_polarity","misaligned"),("misaligned","wrong_polarity"),
    ("wrong_component","wrong_polarity"),("wrong_polarity","wrong_component"),
    ("wrong_component","misaligned"),("misaligned","wrong_component"),
}

_DEFECT_COLOURS = {  # BGR
    "ok":              (  0,200,  0),
    "missing":         (  0,  0,255),
    "misaligned":      (  0,255,255),
    "wrong_component": (255,  0,255),
    "wrong_polarity":  (255,128,  0),
    "ghost":           ( 50, 50, 50),
    "roi_fail":        (  0,200,200),
}

# Inference and helper functions now imported from .utils

# ── Hungarian matching — exact port ──────────────────────────────────────────
def _aoi_hungarian_match(golden: list, dets: list,
                          max_dist: float = _AOI_MATCH_R) -> dict:
    """Returns {golden_id → det_id | None}. Uses scipy when available.

    SAME-LABEL ONLY: cross-label detections are excluded from the cost matrix
    entirely (cost=1e9). This prevents Hungarian from assigning a neighbouring
    Resistor to an empty Capacitor slot, which causes MISSING → WRONG COMPONENT.
    Cross-label wrong-component detection is handled by the reverse pass only.
    """
    if not golden: return {}
    if not dets:   return {g.id: None for g in golden}
    G, D = len(golden), len(dets)
    if not HAS_NP: return {g.id: None for g in golden}
    cost = np.full((G,D), 1e9, dtype=np.float64)
    for i,gc in enumerate(golden):
        for j,dc in enumerate(dets):
            # Only consider same-label detections in forward match
            if dc.label.lower() != gc.label.lower(): continue
            d = _math.hypot(gc.cx-dc.cx, gc.cy-dc.cy)
            if d <= max_dist:
                cost[i,j] = d
    res = {g.id: None for g in golden}
    try:
        from scipy.optimize import linear_sum_assignment
        gi, di = linear_sum_assignment(cost)
        for g,d in zip(gi,di):
            if cost[g,d] < 1e8: res[golden[g].id] = dets[d].id
    except ImportError:
        # Greedy fallback (no scipy).
        # BUG-FIX: the original greedy loop matched by proximity alone without
        # checking labels.  Any detection close enough filled a golden slot even
        # if it was a completely different component type.  Same same-label-only
        # constraint as the scipy path is now enforced here too.
        used = set()
        for gc in sorted(golden, key=lambda x: x.id):
            best, bd = None, max_dist
            for dc in dets:
                if dc.id in used: continue
                if dc.label.lower() != gc.label.lower(): continue  # same-label only
                d = _math.hypot(gc.cx-dc.cx, gc.cy-dc.cy)
                if d < bd: best, bd = dc, d
            if best: res[gc.id] = best.id; used.add(best.id)
    return res

# ── Per-component diff — exact port ──────────────────────────────────────────
def _aoi_region_diff(diff_gray, c, W: int, H: int) -> float:
    """Works with _AOIComp objects (has .xyxy())."""
    x1,y1,x2,y2 = c.xyxy(W,H)
    if x2<=x1 or y2<=y1: return 0.0
    pad = max(1, int(min(x2-x1,y2-y1)*0.08))
    r = diff_gray[y1+pad:y2-pad, x1+pad:x2-pad]
    return float(r.mean()) if r.size > 0 else 0.0

# ── Calibration — exact port ─────────────────────────────────────────────────
def _aoi_calibrate(model, golden_img, golden: list, conf: float,
                   n_runs: int, rng_seed: int = 42,
                   model_path: str = "", use_sahi: bool = False,
                   _log=None) -> tuple:
    """Returns (pos_tol, per_diff_thr). Calibration always uses standard YOLO."""
    if not HAS_CV2 or not HAS_NP:
        return _MIN_POS_TOL, {c.id: _MIN_DIFF_THR for c in golden}
    import random as _rnd, time as _tc
    rng = _rnd.Random(rng_seed); np.random.seed(rng_seed)
    H,W = golden_img.shape[:2]
    jitters = []; comp_diffs = {c.id: [] for c in golden}

    for run_i in range(n_runs):
        if _log and run_i==0: _log(f'[CAL] Running {n_runs} augmented passes…')
        a = rng.uniform(0.93,1.07); b = rng.uniform(-7,7)
        aug = np.clip(golden_img.astype(np.float32)*a+b,0,255).astype(np.uint8)
        # Always standard YOLO for calibration (no SAHI tile-edge jitter)
        dets = _aoi_infer_arr(model, aug, conf, model_path="", use_sahi=False)
        match = _aoi_hungarian_match(golden, dets)
        det_by_id = {d.id:d for d in dets}
        for gc in golden:
            did = match.get(gc.id)
            if did is not None:
                dc = det_by_id[did]
                jitters.append(_math.hypot(dc.cx-gc.cx, dc.cy-gc.cy))
        diff_gray = cv2.cvtColor(cv2.absdiff(golden_img,aug),
                                  cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff_gray = cv2.GaussianBlur(diff_gray,(5,5),0)
        for gc in golden:
            comp_diffs[gc.id].append(_aoi_region_diff(diff_gray, gc, W, H))

    pos_tol = _MIN_POS_TOL
    if jitters:
        arr = np.array(jitters)
        q75,q25 = np.percentile(arr,[75,25]); iqr=q75-q25
        pos_tol = float(np.clip(np.median(arr)+2.5*iqr, _MIN_POS_TOL, 0.055))

    per_thr = {}
    for gc in golden:
        samples = comp_diffs.get(gc.id,[])
        if len(samples)>=2:
            arr = np.array(samples)
            thr = float(np.clip(arr.mean()+3.0*arr.std(), _MIN_DIFF_THR, 80.0))
        else:
            thr = _MIN_DIFF_THR * 2.0
        per_thr[gc.id] = thr
    return pos_tol, per_thr

# ── Defect checker — exact port ───────────────────────────────────────────────

def _aoi_ncc_polarity(golden_img, test_img, gc, dW: int, dH: int,
                      margin: float = 0.05) -> tuple:
    """
    NCC polarity check with multi-crop for tile-boundary robustness.
    Returns (is_flipped: bool, best_delta: float).

    Improvements over the original grayscale-only version:
    - Color NCC (per-channel average) — reduces false alarms on uniform regions.
    - Multi-angle: tests 90°, 180°, 270° rotations; reports the best delta
      so components with ambiguous orientation are caught correctly.
    - Gaussian-smoothed patches before NCC for illumination robustness.
    - Multi-crop scale pyramid (1.0, 0.80, 0.60) retained; best delta kept.
    """
    if not HAS_CV2 or not HAS_NP: return False, -1.0
    g_sm = cv2.resize(golden_img, (dW, dH), interpolation=cv2.INTER_AREA)
    t_sm = cv2.resize(test_img,   (dW, dH), interpolation=cv2.INTER_AREA)

    # Mild smoothing to reduce illumination-variation false alarms
    g_sm = cv2.GaussianBlur(g_sm, (3, 3), 0)
    t_sm = cv2.GaussianBlur(t_sm, (3, 3), 0)

    ROTATIONS = [
        cv2.ROTATE_180,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    ]

    def _zncc_gray(a, b):
        """Zero-mean NCC on 2D float arrays."""
        a = a.flatten() - a.mean()
        b = b.flatten() - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-6 else 0.0

    def _color_ncc(gp_bgr, tp_bgr):
        """Per-channel ZNCC averaged — more robust than grayscale NCC alone."""
        scores = []
        for ch in range(3):
            g_ch = gp_bgr[:, :, ch].astype(np.float32)
            t_ch = tp_bgr[:, :, ch].astype(np.float32)
            scores.append(_zncc_gray(g_ch, t_ch))
        return float(np.mean(scores))

    best_delta = -1.0
    for crop_frac in (1.0, 0.80, 0.60):
        hw = gc.w * crop_frac / 2;  hh = gc.h * crop_frac / 2
        sx1 = max(0, int((gc.cx - hw) * dW));  sx2 = min(dW, int((gc.cx + hw) * dW))
        sy1 = max(0, int((gc.cy - hh) * dH));  sy2 = min(dH, int((gc.cy + hh) * dH))
        if sx2 - sx1 < 4 or sy2 - sy1 < 4: continue

        gp_bgr = g_sm[sy1:sy2, sx1:sx2].astype(np.float32)
        tp_bgr = t_sm[sy1:sy2, sx1:sx2].astype(np.float32)
        if gp_bgr.size < 9: continue

        # Grayscale patches for the grayscale NCC path
        gp_gray = cv2.cvtColor(gp_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        tp_gray = cv2.cvtColor(tp_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Baseline score (unrotated) — weighted blend of color + gray NCC
        base_color = _color_ncc(gp_bgr, tp_bgr)
        base_gray  = _zncc_gray(gp_gray, tp_gray)
        base_score = 0.6 * base_color + 0.4 * base_gray

        for rot_code in ROTATIONS:
            gp_rot_bgr  = cv2.rotate(gp_bgr.astype(np.uint8), rot_code).astype(np.float32)
            gp_rot_gray = cv2.rotate(gp_gray.astype(np.uint8), rot_code).astype(np.float32)

            rot_color = _color_ncc(gp_rot_bgr, tp_bgr)
            rot_gray  = _zncc_gray(gp_rot_gray, tp_gray)
            rot_score = 0.6 * rot_color + 0.4 * rot_gray

            delta = rot_score - base_score
            if delta > best_delta:
                best_delta = delta

    return best_delta > margin, best_delta


def _aoi_patch_ssim(img_a, img_b, gc, W: int, H: int) -> float:
    """SSIM between golden and test patches at component gc (40 % crop for pad context).

    Empirical ranges:
      missing fill   → 0.05 – 0.30  (flat vs structured)
      wrong_component→ 0.35 – 0.75  (different component, still has edges/pads)
      correct        → 0.80 – 1.00
    """
    if not HAS_CV2 or not HAS_NP: return 1.0
    hw = gc.w * 0.40; hh = gc.h * 0.40
    x1 = max(0, int((gc.cx - hw) * W)); y1 = max(0, int((gc.cy - hh) * H))
    x2 = min(W, int((gc.cx + hw) * W)); y2 = min(H, int((gc.cy + hh) * H))
    if x2 - x1 < 4 or y2 - y1 < 4: return 1.0
    pa = cv2.cvtColor(img_a[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    pb = cv2.cvtColor(img_b[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if pa.shape != pb.shape:
        pb = cv2.resize(pb.astype(np.uint8), (pa.shape[1], pa.shape[0])).astype(np.float32)
    C1 = (0.01 * 255) ** 2; C2 = (0.03 * 255) ** 2
    mu_a = float(pa.mean()); mu_b = float(pb.mean())
    sig_a = float(pa.std());  sig_b = float(pb.std())
    cov = float(np.mean((pa - mu_a) * (pb - mu_b)))
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a ** 2 + sig_b ** 2 + C2)
    return float(num / den) if abs(den) > 1e-9 else 0.0


def _aoi_patch_variance(img, gc, W: int, H: int) -> float:
    """Pixel variance of the centre 20 % crop of a component slot.

    Empirical ranges:
      inject_missing Gaussian fill → var always < 5
      any real component           → var always > 80
    """
    if not HAS_CV2 or not HAS_NP: return 0.0
    hw = gc.w * 0.20; hh = gc.h * 0.20
    x1 = max(0, int((gc.cx - hw) * W)); y1 = max(0, int((gc.cy - hh) * H))
    x2 = min(W, int((gc.cx + hw) * W)); y2 = min(H, int((gc.cy + hh) * H))
    if x2 - x1 < 2 or y2 - y1 < 2: return 0.0
    return float(np.var(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)))


def _aoi_patch_tmpl(img_golden, img_test, gc, W: int, H: int) -> float:
    """TM_CCOEFF_NORMED of the golden patch against the test patch.

    Empirical ranges:
      missing fill   → near 0 or negative  (no matching structure)
      wrong_component→ 0.40 – 1.00         (edges/pads still spatially align)
      correct        → 0.70 – 1.00
    """
    if not HAS_CV2 or not HAS_NP: return 0.0
    hw = gc.w * 0.35; hh = gc.h * 0.35
    x1 = max(0, int((gc.cx - hw) * W)); y1 = max(0, int((gc.cy - hh) * H))
    x2 = min(W, int((gc.cx + hw) * W)); y2 = min(H, int((gc.cy + hh) * H))
    if x2 - x1 < 4 or y2 - y1 < 4: return 0.0
    tmpl  = cv2.cvtColor(img_golden[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    patch = cv2.cvtColor(img_test  [y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if tmpl.shape != patch.shape:
        patch = cv2.resize(patch, (tmpl.shape[1], tmpl.shape[0]))
    if tmpl.shape[0] < 2 or tmpl.shape[1] < 2: return 0.0
    try:
        res = cv2.matchTemplate(patch.astype(np.float32),
                                tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        return float(res.max())
    except Exception:
        return 0.0


def _aoi_same_det_is_local(dc, gc, golden_by_label: dict) -> bool:
    """True when dc is geometrically closest to gc among all same-label golden slots.

    Prevents a correctly-placed neighbouring component's detection from being claimed
    by a missing slot — which would produce wrong_component instead of missing.
    """
    dist_to_gc = _math.hypot(dc.cx - gc.cx, dc.cy - gc.cy)
    for og in golden_by_label.get(dc.label.lower(), []):
        if og.id == gc.id: continue
        if _math.hypot(dc.cx - og.cx, dc.cy - og.cy) < dist_to_gc * 0.85:
            return False
    return True



# -- v3.9 Fix P: Two-pass misalignment check (ported from evaluator v3.9) -----
def _aoi_misalignment_check(golden_img, test_img, gc, W: int, H: int) -> tuple:
    """Two-pass rotation recovery for misalignment detection (v3.9).
    Returns (is_misaligned: bool, best_angle: float, reason_str: str)
    """
    if not HAS_CV2 or not HAS_NP:
        return False, 0.0, "no_cv2"
    hw = gc.w * 0.45; hh = gc.h * 0.45
    x1 = max(0, int((gc.cx - hw) * W)); y1 = max(0, int((gc.cy - hh) * H))
    x2 = min(W, int((gc.cx + hw) * W)); y2 = min(H, int((gc.cy + hh) * H))
    ph, pw = y2 - y1, x2 - x1
    if pw < 10 or ph < 10:
        return False, 0.0, "patch_too_small"
    g_gray = cv2.cvtColor(golden_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    t_gray = cv2.cvtColor(test_img  [y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    center = (pw // 2, ph // 2)
    def _recover_ncc(angle):
        if angle == 0:
            return float(cv2.matchTemplate(t_gray, g_gray,
                                           cv2.TM_CCOEFF_NORMED)[0][0])
        M     = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot_t = cv2.warpAffine(t_gray, M, (pw, ph),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        return float(cv2.matchTemplate(rot_t, g_gray,
                                       cv2.TM_CCOEFF_NORMED)[0][0])
    coarse_best_ncc   = -1.0
    coarse_best_angle = 0
    for angle in range(-45, 50, 5):
        if angle == 0: continue
        ncc = _recover_ncc(angle)
        if ncc > coarse_best_ncc:
            coarse_best_ncc   = ncc
            coarse_best_angle = angle
    fine_best_ncc   = coarse_best_ncc
    fine_best_angle = coarse_best_angle
    fine_lo = coarse_best_angle - _MISALIGN_FINE_WINDOW
    fine_hi = coarse_best_angle + _MISALIGN_FINE_WINDOW + 1
    for angle in range(fine_lo, fine_hi, _MISALIGN_FINE_STEP):
        if angle == 0 or angle == coarse_best_angle: continue
        ncc = _recover_ncc(angle)
        if ncc > fine_best_ncc:
            fine_best_ncc   = ncc
            fine_best_angle = angle
    best_ncc     = fine_best_ncc
    best_angle   = fine_best_angle
    unrot_ncc    = _recover_ncc(0)
    recover_gain = best_ncc - unrot_ncc
    reason = (f"recov_ncc={best_ncc:.2f} gain={recover_gain:.2f} "
              f"(base={unrot_ncc:.2f}) at {best_angle} deg")
    if (best_ncc   >= _MISALIGN_RECOVER_NCC_MIN
            and recover_gain >= _MISALIGN_RECOVER_GAIN_MIN
            and abs(best_angle) >= _MISALIGN_RECOVER_ANGLE_MIN):
        return True, float(best_angle), (
            f"rotation_recovered ncc={best_ncc:.2f} "
            f"gain={recover_gain:.2f} at {best_angle} deg")
    return False, float(best_angle), reason


def _aoi_goh_polarity(golden_img, test_img, gc, W: int, H: int) -> tuple:
    """GOH polarity check (v3.9 Fix N). Returns (is_flipped, goh_delta)."""
    if not HAS_CV2 or not HAS_NP: return False, 0.0
    hw = gc.w * 0.35; hh = gc.h * 0.35
    x1 = max(0, int((gc.cx - hw) * W)); y1 = max(0, int((gc.cy - hh) * H))
    x2 = min(W, int((gc.cx + hw) * W)); y2 = min(H, int((gc.cy + hh) * H))
    if x2 - x1 < 8 or y2 - y1 < 8: return False, 0.0
    def _build_goh(img_bgr, _y1, _y2, _x1, _x2):
        gray  = cv2.cvtColor(img_bgr[_y1:_y2, _x1:_x2], cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
        gx    = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy    = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag   = np.sqrt(gx ** 2 + gy ** 2)
        ang   = np.arctan2(gy, gx)
        hist, _ = np.histogram(ang, bins=_GOH_POLAR_BINS,
                               range=(-_math.pi, _math.pi), weights=mag)
        total = hist.sum()
        return hist / (total + 1e-6)
    g_hist = _build_goh(golden_img, y1, y2, x1, x2)
    t_hist = _build_goh(test_img,   y1, y2, x1, x2)
    half = _GOH_POLAR_BINS // 2
    g_hist_rot = np.roll(g_hist, half)
    def _ncc_hist(a, b):
        a = a - a.mean(); b = b - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-6 else 0.0
    corr_orig = _ncc_hist(g_hist, t_hist)
    corr_rot  = _ncc_hist(g_hist_rot, t_hist)
    goh_delta = corr_rot - corr_orig
    return goh_delta >= _GOH_POLAR_MIN_DELTA, goh_delta


def _aoi_intensity_asymmetry_score(golden_img, test_img, gc, W: int, H: int) -> float:
    """Intensity asymmetry opposition score (v3.9 Fix N). Returns float."""
    if not HAS_CV2 or not HAS_NP: return 0.0
    hw = gc.w * 0.28; hh = gc.h * 0.28
    x1 = max(0, int((gc.cx - hw) * W)); y1 = max(0, int((gc.cy - hh) * H))
    x2 = min(W, int((gc.cx + hw) * W)); y2 = min(H, int((gc.cy + hh) * H))
    if x2 - x1 < 6 or y2 - y1 < 6: return 0.0
    def _bias(img_bgr, _y1, _y2, _x1, _x2):
        gray = cv2.cvtColor(img_bgr[_y1:_y2, _x1:_x2],
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape; mh, mw = h // 2, w // 2
        top = float(gray[:mh, :].mean()); bot = float(gray[mh:, :].mean())
        lft = float(gray[:, :mw].mean()); rgt = float(gray[:, mw:].mean())
        tb = (top - bot) / (top + bot + 1e-6)
        lr = (lft - rgt) / (lft + rgt + 1e-6)
        return tb, lr
    g_tb, g_lr = _bias(golden_img, y1, y2, x1, x2)
    t_tb, t_lr = _bias(test_img,   y1, y2, x1, x2)
    return float((-g_tb * t_tb + -g_lr * t_lr) / 2.0)


def _aoi_edge_density(img, gc, W: int, H: int) -> float:
    """Canny edge density within component crop (v3.9 Fix O)."""
    if not HAS_CV2 or not HAS_NP: return 0.0
    hw = gc.w * 0.32; hh = gc.h * 0.32
    x1 = max(0, int((gc.cx - hw) * W)); y1 = max(0, int((gc.cy - hh) * H))
    x2 = min(W, int((gc.cx + hw) * W)); y2 = min(H, int((gc.cy + hh) * H))
    if x2 - x1 < 4 or y2 - y1 < 4: return 0.0
    patch = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(patch, (3, 3), 0)
    edges = cv2.Canny(blur, 20, 60)
    return float(np.count_nonzero(edges)) / (edges.size + 1e-6)

def _aoi_check_board(golden: list, golden_img, test_img, dets: list,
                     pos_tol: float, per_thr: dict,
                     polarity_mult: float = 1.5,
                     match_radius: float = _AOI_MATCH_R) -> tuple:
    """
    Pixel-primary defect detection (v8 rewrite + v9 fixes).

    Returns (defects, match):
      defects – list of dicts {component_id, defect_type, expected_label, …}
      match   – {golden_id → det_id} from same-label Hungarian (overlay compat)

    ── Why the previous design was fragile ──────────────────────────────────────
    The two-pass architecture (forward same-label + reverse cross-label) coupled
    the MISSING and WRONG-COMPONENT decision boundaries through shared thresholds.
    Tightening MISS_SENS to catch more missing components widened the population
    of slots reaching the reverse pass, admitting more false WRONG-COMPONENT hits.
    Narrowing match_radius to reduce false WRONG-COMPONENT blocked legitimate
    wrong-component detections from reaching the reverse pass, leaving them as
    MISSING.  Both directions shared the same knob — fixing one broke the other.

    ── New single-pass pixel-primary approach ───────────────────────────────────
    region_diff  →  PRIMARY signal: is something physically different here?
    YOLO dets    →  SECONDARY evidence: what kind of component is it?

    Each golden slot is classified in one pass using three diff zones:

      ZONE A  diff < LOW_MULT × comp_thr
              Slot looks identical to golden.  Component is present and correct.
              Only run NCC polarity check for polarized component types.

      ZONE B  LOW_MULT × comp_thr ≤ diff < HIGH_MULT × comp_thr
              Moderate visual change.  Detection evidence disambiguates:
                • cross-det owning the slot    → WRONG COMPONENT  (checked first)
                • same-label det displaced     → MISALIGNED
                • polarized + same-label det   → check NCC polarity
                • no same-label det + polarized→ check NCC polarity (FIX 4)
                • owned cross-label det        → WRONG COMPONENT
                • neither                      → MISSING

      ZONE C  diff ≥ HIGH_MULT × comp_thr
              Strong physical change — pixel evidence is decisive.
              For polarized: run NCC first (FIX 3a) — a 180° flip gives high diff
              but is polarity, not missing.
                • polarized + NCC flipped      → WRONG POLARITY
                • owned cross-label det        → WRONG COMPONENT
                • same-label det very near slot→ WRONG COMPONENT (FIX 3b)
                • no owned detection           → MISSING

    Ownership rule (single knob that decouples both error directions):
      A cross-label detection "owns" a golden slot only when it is strictly
      CLOSER to that slot than to any golden slot of its own label.
    ─────────────────────────────────────────────────────────────────────────────
    """
    if not HAS_CV2 or not HAS_NP: return [], {}
    defects: list = []
    H, W = golden_img.shape[:2]

    # ── Diff image (scaled to ≤1280px for speed) ──────────────────────────────
    DIFF_MAX   = 1280
    diff_scale = min(1.0, DIFF_MAX / max(H, W))
    if diff_scale < 1.0:
        dW, dH = int(W * diff_scale), int(H * diff_scale)
        g_sm   = cv2.resize(golden_img, (dW, dH), interpolation=cv2.INTER_AREA)
        t_sm   = cv2.resize(test_img,   (dW, dH), interpolation=cv2.INTER_AREA)
    else:
        dW, dH, g_sm, t_sm = W, H, golden_img, test_img

    diff_gray = cv2.GaussianBlur(
        cv2.cvtColor(cv2.absdiff(g_sm, t_sm), cv2.COLOR_BGR2GRAY).astype(np.float32),
        (5, 5), 0)

    # ── Constants ─────────────────────────────────────────────────────────────
    POLARIZED = {
        "ic (u)", "ic (ic)", "transistor (q)", "transistor (qa)",
        "diode (d)", "led", "capacitor (c)", "cap array (cra)",
    }
    MIN_CONF   = 0.22   # below this → inpaint phantom, ignored
    LOW_MULT   = 0.50   # diff < LOW × thr  → Zone A (identical to golden)
    HIGH_MULT  = 2.0    # diff ≥ HIGH × thr → Zone C (strong physical change)
    MISALIGN_T = 1.6    # multiplier above detect_tol that qualifies as misaligned

    detect_tol = pos_tol * _AOI_DETECT_BAND

    # ── Filter phantoms once, build spatial index ─────────────────────────────
    real_dets = [d for d in dets if d.conf >= MIN_CONF]

    _golden_by_label: dict = {}
    for g in golden:
        _golden_by_label.setdefault(g.label.lower(), []).append(g)

    # ── Same-label Hungarian (kept for overlay/render compatibility) ──────────
    match = _aoi_hungarian_match(golden, dets, max_dist=match_radius)

    # ── Slot-detection lookup ─────────────────────────────────────────────────
    def _slot_dets(gc):
        """Return (same_label, cross_label) lists of (dist, det), nearest first.
        Search radius scales with component size: large ICs get slightly more
        room for YOLO jitter without opening the neighbour window for resistors."""
        _half_diag = _math.hypot(gc.w, gc.h) / 2
        r          = min(max(match_radius, _half_diag * 0.70), match_radius * 1.8)
        same: list = []; cross: list = []
        for dc in real_dets:
            d = _math.hypot(dc.cx - gc.cx, dc.cy - gc.cy)
            if d > r: continue
            (same if dc.label.lower() == gc.label.lower() else cross).append((d, dc))
        same.sort(key=lambda x: x[0])
        cross.sort(key=lambda x: x[0])
        return same, cross

    # ── Ownership check ───────────────────────────────────────────────────────
    def _owns_slot(dc, gc) -> bool:
        """True only when dc is unambiguously displaced to gc's position.

        Two conditions must BOTH hold:
          1. dc is strictly closer to gc than to any golden slot of dc's own label.
             (A correctly-placed component is always nearest its own slot.)
          2. dc is within detect_tol of gc — not just "nearer" but actually near.
             Without this gate a cross-label det 3 slots away could pass if its
             own-label golden slots happen to be even further.
        """
        dist_to_gc = _math.hypot(dc.cx - gc.cx, dc.cy - gc.cy)
        # Gate 2: must be physically close to this golden slot
        if dist_to_gc > detect_tol:
            return False
        own = _golden_by_label.get(dc.label.lower(), [])
        nearest_own = min(
            (_math.hypot(dc.cx - og.cx, dc.cy - og.cy) for og in own),
            default=float('inf'))
        # Gate 1: must be strictly closer to gc than to its own nearest slot,
        # with a 10% margin to avoid triggering on floating-point ties.
        return dist_to_gc < nearest_own * 0.90

    # ── Per-slot single-pass decision tree ────────────────────────────────────
    for gc in golden:
        comp_thr    = per_thr.get(gc.id, _MIN_DIFF_THR)
        region_diff = _aoi_region_diff(diff_gray, gc, dW, dH)
        low_thr     = comp_thr * LOW_MULT
        high_thr    = comp_thr * HIGH_MULT

        same_dets, cross_dets = _slot_dets(gc)

        # ── ZONE A: slot looks identical to golden ────────────────────────────
        # Pixel evidence says nothing changed at this location.  Only a polarity
        # flip can produce a meaningful NCC delta without a large diff signal.
        if region_diff < low_thr:
            if gc.label.lower() in POLARIZED and same_dets:
                is_flipped, ncc_delta = _aoi_ncc_polarity(
                    golden_img, test_img, gc, dW, dH, margin=0.05)
                if is_flipped:
                    defects.append({
                        "component_id":   gc.id,
                        "defect_type":    "wrong_polarity",
                        "expected_label": gc.label,
                        "found_label":    same_dets[0][1].label,
                        "cx": gc.cx, "cy": gc.cy,
                        "details": f"NCC zone-A diff={region_diff:.1f} ncc_delta={ncc_delta:.4f}"})
            continue   # slot is fine

        # ── ZONE B: moderate change — let detection evidence decide ───────────
        if region_diff < high_thr:
            # v3.9: rotation rescue -- check for misalignment before same-det logic
            _zb_mali_ok = False
            if same_dets:
                _zb_mali_ok, _zb_mali_angle, _zb_mali_reason = _aoi_misalignment_check(
                    golden_img, test_img, gc, W, H)
            if _zb_mali_ok:
                _mali_fl = (same_dets[0][1].label if same_dets else
                           (cross_dets[0][1].label if cross_dets else gc.label))
                defects.append({
                    "component_id":   gc.id,
                    "defect_type":    "misaligned",
                    "expected_label": gc.label,
                    "found_label":    _mali_fl,
                    "cx": gc.cx, "cy": gc.cy,
                    "details": f"zoneB_rotation_rescue {_zb_mali_reason}"})
                continue

            if same_dets:
                dist, tc = same_dets[0]

                # FIX: check cross-det ownership BEFORE misalign — a cross-det
                # owning the slot means it's WRONG_COMPONENT, not just misaligned.
                _zb_cross_owned = False
                for _zbd, _zbdc in cross_dets:
                    if _owns_slot(_zbdc, gc):
                        defects.append({
                            "component_id":   gc.id,
                            "defect_type":    "wrong_component",
                            "expected_label": gc.label,
                            "found_label":    _zbdc.label,
                            "cx": gc.cx, "cy": gc.cy,
                            "det_id": _zbdc.id,
                            "det_cx": _zbdc.cx, "det_cy": _zbdc.cy,
                            "det_w":  _zbdc.w,  "det_h":  _zbdc.h,
                            "details": f"diff={region_diff:.1f} zone-B cross-owns conf={_zbdc.conf:.2f}"})
                        _zb_cross_owned = True
                        break

                if not _zb_cross_owned:
                    if dist > detect_tol * MISALIGN_T and region_diff > comp_thr * 0.35:
                        # Component present but centroid has drifted outside normal jitter.
                        defects.append({
                            "component_id":   gc.id,
                            "defect_type":    "misaligned",
                            "expected_label": gc.label,
                            "found_label":    tc.label,
                            "cx": gc.cx, "cy": gc.cy,
                            "details": f"dist={dist:.4f} tol={detect_tol:.4f} diff={region_diff:.1f}"})
                    elif gc.label.lower() in POLARIZED:
                        # Same-label det in position, elevated diff: check for polarity flip.
                        is_flipped, ncc_delta = _aoi_ncc_polarity(
                            golden_img, test_img, gc, dW, dH, margin=_AOI_NCC_POL_B)
                        if is_flipped:
                            defects.append({
                                "component_id":   gc.id,
                                "defect_type":    "wrong_polarity",
                                "expected_label": gc.label,
                                "found_label":    tc.label,
                                "cx": gc.cx, "cy": gc.cy,
                                "details": f"NCC zone-B diff={region_diff:.1f} ncc_delta={ncc_delta:.4f}"})
                        else:
                            # Fix L: polar slot, same det present, NCC gave no flip signal.
                            # Zone B rdiff is elevated — something changed at this location.
                            # Since YOLO still fires the same label and NCC can't confirm
                            # rotation, most likely a same-class donor → wrong_component.
                            defects.append({
                                "component_id":   gc.id,
                                "defect_type":    "wrong_component",
                                "expected_label": gc.label,
                                "found_label":    tc.label,
                                "cx": gc.cx, "cy": gc.cy,
                                "details": f"zone-B polar noflip diff={region_diff:.1f} ncc_delta={ncc_delta:.4f}"})
                    else:
                        # Non-polarized, same det nearby, no cross ownership.
                        # Use SSIM to decide between wrong_component and normal variation.
                        _zb_ssim = _aoi_patch_ssim(g_sm, t_sm, gc, dW, dH)
                        if _zb_ssim < _AOI_SSIM_MISS_MAX:
                            defects.append({
                                "component_id":   gc.id,
                                "defect_type":    "missing",
                                "expected_label": gc.label,
                                "found_label":    "none",
                                "cx": gc.cx, "cy": gc.cy,
                                "details": f"zone-B ssim_low={_zb_ssim:.3f} diff={region_diff:.1f}"})
                        elif (dist < detect_tol and _aoi_same_det_is_local(tc, gc, _golden_by_label)
                              and _zb_ssim < _AOI_SSIM_MISS_MAX + 0.15):
                            defects.append({
                                "component_id":   gc.id,
                                "defect_type":    "wrong_component",
                                "expected_label": gc.label,
                                "found_label":    tc.label,
                                "cx": gc.cx, "cy": gc.cy,
                                "details": f"zone-B at-slot ssim={_zb_ssim:.3f} diff={region_diff:.1f}"})
                        elif _zb_ssim >= _AOI_SSIM_MISS_MAX + 0.15 and region_diff > comp_thr * 0.35:
                            # High SSIM + elevated diff: something structured occupies the slot.
                            defects.append({
                                "component_id":   gc.id,
                                "defect_type":    "wrong_component",
                                "expected_label": gc.label,
                                "found_label":    tc.label,
                                "cx": gc.cx, "cy": gc.cy,
                                "details": f"zone-B high-ssim={_zb_ssim:.3f} diff={region_diff:.1f}"})
                        # else: minor diff + low rdiff → normal board variation, OK
                continue

            # No same-label detection in slot.
            # FIX 4: also run NCC for polarized components here — when a polarized
            # component is rotated 180°, YOLO can still detect it but the diff lands
            # in Zone B and there may be no same-label det. Use tighter margin=0.40
            # to avoid false triggers on capacitors.
            if gc.label.lower() in POLARIZED:
                is_flipped, ncc_delta = _aoi_ncc_polarity(
                    golden_img, test_img, gc, dW, dH, margin=0.40)
                if is_flipped:
                    flbl = cross_dets[0][1].label if cross_dets else gc.label
                    defects.append({
                        "component_id":   gc.id,
                        "defect_type":    "wrong_polarity",
                        "expected_label": gc.label,
                        "found_label":    flbl,
                        "cx": gc.cx, "cy": gc.cy,
                        "details": f"NCC zone-B no-same diff={region_diff:.1f} ncc_delta={ncc_delta:.4f}"})
                    continue

            # Check for an owned cross-label det.
            for _dist, dc in cross_dets:
                if _owns_slot(dc, gc):
                    defects.append({
                        "component_id":   gc.id,
                        "defect_type":    "wrong_component",
                        "expected_label": gc.label,
                        "found_label":    dc.label,
                        "cx": gc.cx, "cy": gc.cy,
                        "det_id": dc.id,
                        "det_cx": dc.cx, "det_cy": dc.cy,
                        "det_w":  dc.w,  "det_h":  dc.h,
                        "details": f"diff={region_diff:.1f} conf={dc.conf:.2f}"})
                    break
            else:
                # Moderate diff, no usable detection → component absent.
                defects.append({
                    "component_id":   gc.id,
                    "defect_type":    "missing",
                    "expected_label": gc.label,
                    "found_label":    "none",
                    "cx": gc.cx, "cy": gc.cy,
                    "details": f"diff={region_diff:.1f} thr={comp_thr:.1f} no-det"})
            continue

        # ── ZONE C: strong physical change ────────────────────────────────────
        # Compute all three pixel signals upfront — they inform every step below.
        ssim_v   = _aoi_patch_ssim(g_sm, t_sm, gc, dW, dH)
        test_var = _aoi_patch_variance(t_sm, gc, dW, dH)
        tmpl_v   = _aoi_patch_tmpl(g_sm, t_sm, gc, dW, dH)

        slot_is_flat   = test_var < _AOI_VAR_FLAT and ssim_v < _AOI_SSIM_MISS_MAX
        var_structured = test_var >= _AOI_VAR_WC_CONFIRM
        cross_owns     = any(_owns_slot(dc, gc) for _, dc in cross_dets)

        # ── Step 1: flat slot — definitely missing ────────────────────────────
        # Skip if a cross-label det owns the slot (IC donor may look flat at 20%
        # crop but YOLO still fired) or if a same-label det is nearby.
        if slot_is_flat and not cross_owns and not same_dets:
            defects.append({
                "component_id":   gc.id,
                "defect_type":    "missing",
                "expected_label": gc.label,
                "found_label":    "none",
                "cx": gc.cx, "cy": gc.cy,
                "details": f"flat var={test_var:.1f} ssim={ssim_v:.3f}"})
            continue


        # -- Step 1.5: rotation-based misalignment rescue (v3.9) -----------
        _zc_mali_ok, _zc_mali_angle, _zc_mali_reason = _aoi_misalignment_check(
            golden_img, test_img, gc, W, H)
        if _zc_mali_ok:
            _mali_fl = (same_dets[0][1].label if same_dets else
                       (cross_dets[0][1].label if cross_dets else gc.label))
            defects.append({
                "component_id":   gc.id,
                "defect_type":    "misaligned",
                "expected_label": gc.label,
                "found_label":    _mali_fl,
                "cx": gc.cx, "cy": gc.cy,
                "details": f"zoneC_rotation_rescue {_zc_mali_reason}"})
            continue

        # ── Step 2: polarity check (non-flat, polarized components) ──────────
        # Compound acceptance logic validated across 100-board runs (v3.6):
        #   (a) delta > STRONG (0.95)         → unconditional WPOL (all ICs + strong caps)
        #   (b) delta > C margin (0.38)
        #       AND ssim < SSIM_NEG (-0.33)   → rotated cap (anti-correlated structure)
        # Structural fallback when NCC is blind (near-symmetric IC):
        #   delta >= STRUCT_MIN (0.25)
        #   AND ssim >= SSIM_STRUCT (0.28)
        #   AND tmpl >= TMPL_STRUCT (0.25)
        #     — WCOM FPs had delta 0.037–0.186; true WPOL via this path ≥ 0.393
        flipped, ncc_delta = False, -1.0
        if gc.label.lower() in POLARIZED:
            is_ic = "ic" in gc.label.lower()
            _ncc_margin = _AOI_NCC_POL_IC if is_ic else _AOI_NCC_POL_C
            flipped, ncc_delta = _aoi_ncc_polarity(
                g_sm, t_sm, gc, dW, dH, margin=_ncc_margin)

            accept_polarity = (flipped and
                               (ncc_delta > _AOI_NCC_POL_STRONG or
                                ssim_v < _AOI_NCC_POL_SSIM_NEG))

            # Structural fallback for near-symmetric ICs where NCC is blind
            if not accept_polarity and ncc_delta >= _AOI_NCC_POL_STRUCT_MIN:
                if ssim_v >= _AOI_SSIM_POL_STRUCT and tmpl_v >= _AOI_TMPL_POL_STRUCT:
                    accept_polarity = True

            # -- v3.9 Fix N: GOH + asymmetry fallback ---------------------
            if not accept_polarity:
                goh_flip, goh_delta = _aoi_goh_polarity(
                    golden_img, test_img, gc, W, H)
                if goh_delta >= _GOH_POLAR_MIN_DELTA:
                    accept_polarity = True
                elif (_GOH_ASYM_VOTE_BAND_LO <= goh_delta
                      < _GOH_ASYM_VOTE_BAND_HI):
                    asym_score = _aoi_intensity_asymmetry_score(
                        golden_img, test_img, gc, W, H)
                    if asym_score >= _ASYM_POLAR_MIN:
                        accept_polarity = True

            if accept_polarity:
                fl = (same_dets[0][1].label if same_dets else
                      (cross_dets[0][1].label if cross_dets else gc.label))
                defects.append({
                    "component_id":   gc.id,
                    "defect_type":    "wrong_polarity",
                    "expected_label": gc.label,
                    "found_label":    fl,
                    "cx": gc.cx, "cy": gc.cy,
                    "details": f"ncc_delta={ncc_delta:.4f} ssim={ssim_v:.3f} tmpl={tmpl_v:.3f}"})
                continue



        # ── Step 3: cross-label det owns slot ────────────────────────────────
        # Polarized components that failed the polarity check arrive here.
        # cross_polar_rescue: when a rotated component causes a cross-label YOLO
        # hit, NCC may have seen a weak flip (below full acceptance) — rescue it
        # when tmpl and NCC delta confirm the rotation rather than a donor swap.
        if cross_owns:
            if (gc.label.lower() in POLARIZED and flipped and var_structured
                    and ssim_v < _AOI_SSIM_CROSS_MAX
                    and tmpl_v >= _AOI_CROSS_RESCUE_TMPL
                    and ncc_delta > _AOI_CROSS_RESCUE_NCC):
                fl = (same_dets[0][1].label if same_dets else
                      (cross_dets[0][1].label if cross_dets else gc.label))
                defects.append({
                    "component_id":   gc.id,
                    "defect_type":    "wrong_polarity",
                    "expected_label": gc.label,
                    "found_label":    fl,
                    "cx": gc.cx, "cy": gc.cy,
                    "details": f"cross_polar_rescue ncc_delta={ncc_delta:.4f} tmpl={tmpl_v:.3f}"})
                continue
            # Stray cross-label YOLO hit on a truly flat/empty slot → pixel wins.
            if test_var < _AOI_VAR_FLAT and ssim_v < _AOI_SSIM_MISS_MAX:
                defects.append({
                    "component_id":   gc.id,
                    "defect_type":    "missing",
                    "expected_label": gc.label,
                    "found_label":    "none",
                    "cx": gc.cx, "cy": gc.cy,
                    "details": f"cross_flat_override var={test_var:.1f} ssim={ssim_v:.3f}"})
                continue
            for _dist, dc in cross_dets:
                if _owns_slot(dc, gc):
                    defects.append({
                        "component_id":   gc.id,
                        "defect_type":    "wrong_component",
                        "expected_label": gc.label,
                        "found_label":    dc.label,
                        "cx": gc.cx, "cy": gc.cy,
                        "det_id": dc.id, "det_cx": dc.cx, "det_cy": dc.cy,
                        "det_w": dc.w, "det_h": dc.h,
                        "details": f"cross_zoneC ssim={ssim_v:.3f} conf={dc.conf:.2f}"})
                    break
            continue

        # ── Step 4: MISSING vs WRONG_COMPONENT (no cross, not polarity) ──────
        # Primary discriminants validated on 100-board evaluator runs:
        #   var ≥ 80  → a real component is present   (var_structured)
        #   ssim ≥ 0.32 → structured patch, not fill  (ssim signal)
        # template match (tmpl) is diagnostic only — unreliable when donor
        # and golden are structurally dissimilar.
        if var_structured or ssim_v >= _AOI_SSIM_MISS_MAX:
            if same_dets:
                nearest_dist, nearest_dc = same_dets[0]
                at_slot  = nearest_dist < detect_tol
                is_local = _aoi_same_det_is_local(nearest_dc, gc, _golden_by_label)
                if at_slot and is_local:
                    # YOLO fired at the slot with the same label → wrong_component
                    reason = (f"var_rescue({test_var:.0f})" if var_structured
                              else f"at_slot_ssim({ssim_v:.2f})")
                    defects.append({
                        "component_id":   gc.id,
                        "defect_type":    "wrong_component",
                        "expected_label": gc.label,
                        "found_label":    nearest_dc.label,
                        "cx": gc.cx, "cy": gc.cy,
                        "det_id": nearest_dc.id, "det_cx": nearest_dc.cx,
                        "det_cy": nearest_dc.cy, "det_w": nearest_dc.w, "det_h": nearest_dc.h,
                        "details": f"{reason} dist={nearest_dist:.4f}"})
                elif ssim_v >= _AOI_SSIM_WC_NODET or var_structured:
                    # Neighbour det but var / SSIM still confirm something present
                    reason = (f"var_rescue({test_var:.0f})" if var_structured
                              else f"ssim_high_nolocal({ssim_v:.2f})")
                    defects.append({
                        "component_id":   gc.id,
                        "defect_type":    "wrong_component",
                        "expected_label": gc.label,
                        "found_label":    nearest_dc.label,
                        "cx": gc.cx, "cy": gc.cy,
                        "details": f"{reason} neighbour_dist={nearest_dist:.4f}"})
                else:
                    # Medium SSIM, var not high enough, det is a neighbour → missing
                    defects.append({
                        "component_id":   gc.id,
                        "defect_type":    "missing",
                        "expected_label": gc.label,
                        "found_label":    "none",
                        "cx": gc.cx, "cy": gc.cy,
                        "details": f"neighbour_det ssim={ssim_v:.3f} var={test_var:.1f}"})
            else:
                # No same-label det, but var/ssim say something is there
                defects.append({
                    "component_id":   gc.id,
                    "defect_type":    "wrong_component",
                    "expected_label": gc.label,
                    "found_label":    "unknown",
                    "cx": gc.cx, "cy": gc.cy,
                    "details": (f"var_rescue_nodet({test_var:.0f})" if var_structured
                                else f"ssim_only({ssim_v:.2f})")})
        else:
            # -- v3.9 Fix O: edge-density rescue ---------------------------
            edge_d = _aoi_edge_density(t_sm, gc, dW, dH)
            if edge_d >= _EDGE_DENSITY_WC_MIN:
                found_lbl = same_dets[0][1].label if same_dets else "unknown"
                defects.append({
                    "component_id":   gc.id,
                    "defect_type":    "wrong_component",
                    "expected_label": gc.label,
                    "found_label":    found_lbl,
                    "cx": gc.cx, "cy": gc.cy,
                    "details": f"edge_rescue({edge_d:.3f})"})
            else:
                defects.append({
                    "component_id":   gc.id,
                    "defect_type":    "missing",
                    "expected_label": gc.label,
                    "found_label":    "none",
                    "cx": gc.cx, "cy": gc.cy,
                    "details": f"ssim_low({ssim_v:.3f}) var({test_var:.1f}) tmpl({tmpl_v:.3f})"})

    return defects, match


# ── Overlay renderer — exact port ─────────────────────────────────────────────
def _aoi_render_overlay(test_img, golden: list, dets: list, defects: list,
                         golden_img=None, match: dict = None,
                         match_r: float = _AOI_MATCH_R):
    """
    match: pre-computed {golden_id -> det_id} from _aoi_check_board.
           If supplied, overlay uses it directly (no re-match).
           If None, falls back to fresh Hungarian match with match_r.
    Always pass 'match' from the worker so overlay and defect logic use the
    same pairing -- re-matching here with a different radius was the cause of
    boxes appearing at wrong positions.

    Rendering order (per golden slot):
      1. MISSING      — always drawn at the GOLDEN slot position, never at a
                        matched det.  A neighbouring same-label det can win the
                        Hungarian match for a missing slot; we must ignore that
                        match so the red cross appears exactly where the gap is.
      2. WRONG COMP   — pink box at the wrong-det position (det_cx/det_cy) +
                        thin pink outline on the golden slot.
      3. OTHER DEFECTS / OK — drawn at the matched det position.
                        Guarded by wrong_det_ids: if the matched det was already
                        consumed by a wrong_component defect (it is the actual
                        wrong part and therefore drawn in pink elsewhere), we
                        suppress the green box — this was the root cause of the
                        green + pink double-box artefact.
    """
    if not HAS_CV2 or not HAS_NP: return test_img
    img = test_img.copy(); H,W = img.shape[:2]
    defect_map = {d["component_id"]: d["defect_type"] for d in defects}
    if match is None:
        match = _aoi_hungarian_match(golden, dets, max_dist=match_r)
    det_by_id = {d.id: d for d in dets}

    # det_ids that ARE the wrong component — must not be drawn green by another slot
    wrong_det_ids = {d["det_id"] for d in defects if "det_id" in d}

    # ── Ghost outlines for all golden slots ───────────────────────────────────
    for gc in golden:
        x1,y1,x2,y2 = gc.xyxy(W,H)
        cv2.rectangle(img,(x1,y1),(x2,y2),_DEFECT_COLOURS["ghost"],1)

    for gc in golden:
        did   = match.get(gc.id)
        dtype = defect_map.get(gc.id, "ok")
        colour = _DEFECT_COLOURS.get(dtype, _DEFECT_COLOURS["ok"])

        # ── MISSING: ALWAYS draw at the golden slot, ignoring any match ───────
        # Bug fix: a neighbour's same-label det can be Hungarian-matched to a
        # missing slot (did is not None).  If we fell through to the matched-det
        # path the red box would appear at the neighbour's location, not the gap.
        if dtype == "missing":
            x1,y1,x2,y2 = gc.xyxy(W,H)
            cv2.rectangle(img,(x1,y1),(x2,y2),_DEFECT_COLOURS["missing"],2)
            cv2.line(img,(x1,y1),(x2,y2),_DEFECT_COLOURS["missing"],2)
            cv2.line(img,(x2,y1),(x1,y2),_DEFECT_COLOURS["missing"],2)
            miss_tag = f'{gc.label} [MISSING]'
            (mtw,mth),_ = cv2.getTextSize(miss_tag,cv2.FONT_HERSHEY_SIMPLEX,0.34,1)
            col_m = _DEFECT_COLOURS["missing"]
            cv2.rectangle(img,(x1,max(0,y1-mth-4)),(x1+mtw+2,y1),col_m,-1)
            cv2.putText(img,miss_tag,(x1+1,max(mth+2,y1-2)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.34,(255,255,255),1,cv2.LINE_AA)
            continue

        # ── WRONG COMPONENT: draw at the detection position ───────────────────
        if dtype == "wrong_component":
            d_rec = next((d for d in defects if d['component_id']==gc.id), None)
            col   = _DEFECT_COLOURS['wrong_component']
            if d_rec and 'det_cx' in d_rec:
                dx1=int((d_rec['det_cx']-d_rec['det_w']/2)*W)
                dy1=int((d_rec['det_cy']-d_rec['det_h']/2)*H)
                dx2=int((d_rec['det_cx']+d_rec['det_w']/2)*W)
                dy2=int((d_rec['det_cy']+d_rec['det_h']/2)*H)
                cv2.rectangle(img,(dx1,dy1),(dx2,dy2),col,2)
                tag=f"{d_rec.get('found_label','?')} [WRONG PART]"
                (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.34,1)
                cv2.rectangle(img,(dx1,dy1-th-4),(dx1+tw+2,dy1),col,-1)
                cv2.putText(img,tag,(dx1+1,dy1-2),cv2.FONT_HERSHEY_SIMPLEX,
                            0.34,(0,0,0),1,cv2.LINE_AA)
            # Thin outline on the golden slot so the viewer knows what was expected
            x1,y1,x2,y2 = gc.xyxy(W,H)
            cv2.rectangle(img,(x1,y1),(x2,y2),col,1)
            continue

        # ── OTHER DEFECTS / OK: draw at the matched detection ─────────────────
        if did is None:
            continue   # no match, nothing to draw (ghost outline already drawn)

        # Bug fix: if this det was already consumed as the "wrong part" for
        # another slot, suppress the green box — it is the wrong component,
        # not a legitimately-present correct component.
        if did in wrong_det_ids:
            continue

        tc = det_by_id.get(did)
        if tc is None: continue
        dx1,dy1 = int((tc.cx-tc.w/2)*W), int((tc.cy-tc.h/2)*H)
        dx2,dy2 = int((tc.cx+tc.w/2)*W), int((tc.cy+tc.h/2)*H)

        cv2.rectangle(img,(dx1,dy1),(dx2,dy2),colour,2)
        _OVL = {"misaligned":"[MISALIGNED]","wrong_polarity":"[WRONG POLARITY]",
                "roi_fail":"[ROI FAIL]"}
        tag = tc.label if dtype=="ok" else f'{tc.label} {_OVL.get(dtype,"["+dtype.upper()+"]")}'
        (tw,th),_ = cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.34,1)
        cv2.rectangle(img,(dx1,dy1-th-4),(dx1+tw+2,dy1),colour,-1)
        cv2.putText(img,tag,(dx1+1,dy1-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.34,(0,0,0),1,cv2.LINE_AA)

    # Draw unmatched detections (extra/spurious) in dim cyan
    # Exclude dets that are wrong_component — they are already drawn in pink
    assigned_det_ids = set(v for v in match.values() if v is not None)
    for dc in dets:
        if dc.id in assigned_det_ids: continue
        if dc.id in wrong_det_ids: continue  # already drawn in pink
        ex1,ey1 = int((dc.cx-dc.w/2)*W), int((dc.cy-dc.h/2)*H)
        ex2,ey2 = int((dc.cx+dc.w/2)*W), int((dc.cy+dc.h/2)*H)
        cv2.rectangle(img,(ex1,ey1),(ex2,ey2),(80,80,20),1)
        cv2.putText(img,f'?{dc.label}',(ex1+1,ey1-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.28,(80,80,20),1,cv2.LINE_AA)
    return img



# ─────────────────────────────────────────────────────────────────────────────
# ZOOMABLE IMAGE VIEWER  (scroll wheel zoom + middle-click/drag pan)
# ─────────────────────────────────────────────────────────────────────────────
class ZoomableImageView(QWidget):
    """
    Proper interactive image viewer:
      • Scroll wheel  → zoom in/out (centred on cursor)
      • Left drag     → pan
      • Double-click  → fit to window
      • +/- buttons   → zoom
      Displays a QPixmap at arbitrary zoom with smooth rendering.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ziv")
        self.setStyleSheet(f"#ziv{{background:{BG_CARD};border-radius:6px;border:1px solid {BG_BORDER2};}}")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(300)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMouseTracking(True)

        self._px: QPixmap = QPixmap()          # full-resolution source
        self._zoom: float = 1.0
        self._offset      = (0.0, 0.0)         # (ox, oy) pan offset in widget coords
        self._drag        = False
        self._drag_start  = (0, 0)
        self._drag_offset = (0.0, 0.0)

        # Peer-sync support: set both views' _peer to each other to sync zoom/pan
        self._peer         = None
        self._sync_blocked = False
        self._fit_zoom     = 1.0    # zoom level at last _fit() — used for sync ratio
        self._idle_text    = "No image loaded"   # override per instance if needed

        # Zoom controls overlay
        self._zoom_lbl = QLabel("100%", self)
        self._zoom_lbl.setStyleSheet(
            f"color:{CYAN};font-family:Consolas;font-size:10px;font-weight:bold;"
            f"background:{BG_CARD2}cc;border-radius:4px;padding:2px 6px;border:none;"
        )
        self._zoom_lbl.move(8, 8)
        self._zoom_lbl.adjustSize()

    def load_bytes(self, img_bytes: bytes):
        """Load image from JPEG/PNG bytes."""
        if not img_bytes:
            self._px = QPixmap(); self.update(); return
        qimg = QImage.fromData(img_bytes)
        if qimg.isNull(): return
        self._px = QPixmap.fromImage(qimg)
        self._fit()

    def load_pixmap(self, px: QPixmap):
        self._px = px; self._fit()

    def load_frame(self, bgr):
        if not HAS_CV2 or not HAS_NP: return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self._px = QPixmap.fromImage(qimg)
        self._fit()

    def load_image(self, path):
        self._px = QPixmap(path)
        self._fit()

    def clear(self):
        self._px = QPixmap(); self.update()

    def _fit(self):
        """Fit image to widget, centred."""
        if self._px.isNull(): return
        W, H = self.width(), self.height()
        if W < 2 or H < 2: return
        sx = W / self._px.width(); sy = H / self._px.height()
        self._zoom = min(sx, sy) * 0.97
        self._fit_zoom = self._zoom   # record for sync ratio conversion
        self._center()
        self.update()
        self._zoom_lbl.setText(f"{self._zoom*100:.0f}%")

    def _center(self):
        W, H = self.width(), self.height()
        pw = self._px.width() * self._zoom
        ph = self._px.height() * self._zoom
        self._offset = ((W - pw) / 2, (H - ph) / 2)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._px.isNull(): self._fit()
        self._zoom_lbl.move(8, 8)

    def paintEvent(self, e):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#1A1F2E"))
        if self._px.isNull():
            # Subtle dot-grid
            dot_col = QColor(60, 70, 100, 70)
            p.setPen(QPen(dot_col, 1))
            for gx in range(0, W, 28):
                for gy in range(0, H, 28):
                    p.drawPoint(gx, gy)
            # Faint crosshair
            cx, cy = W//2, H//2
            cross = QColor(CYAN); cross.setAlpha(28)
            p.setPen(QPen(cross, 1, Qt.PenStyle.DashLine))
            p.drawLine(0, cy, W, cy); p.drawLine(cx, 0, cx, H)
            # Corner brackets
            cs = 18; bdr = QColor(CYAN); bdr.setAlpha(60)
            p.setPen(QPen(bdr, 2))
            for bx,by,dx,dy in [(6,6,1,1),(W-6-cs,6,-1,1),(6,H-6-cs,1,-1),(W-6-cs,H-6-cs,-1,-1)]:
                p.drawLine(bx,by,bx+cs*dx,by); p.drawLine(bx,by,bx,by+cs*dy)
            # Centre target ring
            tgt = QColor(CYAN); tgt.setAlpha(20)
            p.setPen(QPen(tgt, 1)); p.setBrush(Qt.BrushStyle.NoBrush)
            r = 16; p.drawEllipse(cx-r, cy-r, r*2, r*2)
            # Text
            p.setPen(QPen(QColor(TEXT_DIM)))
            p.setFont(_F_MONO_11)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._idle_text)
            p.end(); return
        ox, oy = self._offset
        dw = int(self._px.width()  * self._zoom)
        dh = int(self._px.height() * self._zoom)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                        self._zoom < 2.0)
        p.drawPixmap(int(ox), int(oy), dw, dh, self._px)
        p.end()

    def _sync_from(self, zoom: float, offset: tuple):
        """Called by peer to set zoom/pan without triggering infinite loop."""
        if self._sync_blocked: return
        self._sync_blocked = True
        self._zoom = zoom
        self._offset = offset
        self._zoom_lbl.setText(f"{zoom*100:.0f}%")
        self.update()
        self._sync_blocked = False

    def _push_sync(self):
        """Push current zoom/pan state to peer (if any)."""
        if self._peer and not self._sync_blocked:
            self._peer._sync_from(self._zoom, self._offset)

    def zoom_to_component(self, cx_norm: float, cy_norm: float,
                           w_norm: float, h_norm: float):
        """Zoom so the component + generous context fills the view.
        We inflate the effective bbox by 4× so there is plenty of surrounding
        board visible — avoids the "face pressed against IC" problem."""
        if self._px.isNull(): return
        iw, ih = self._px.width(), self._px.height()
        # Expand the region of interest 4× around the component centre
        roi_w = min(1.0, w_norm * 4.0)
        roi_h = min(1.0, h_norm * 4.0)
        comp_w_px = max(1, roi_w * iw)
        comp_h_px = max(1, roi_h * ih)
        vW, vH = self.width(), self.height()
        zoom_x = vW / comp_w_px
        zoom_y = vH / comp_h_px
        self._zoom = min(zoom_x, zoom_y)
        self._zoom = max(0.1, min(12.0, self._zoom))
        cx_px = cx_norm * iw
        cy_px = cy_norm * ih
        self._offset = (vW / 2 - cx_px * self._zoom,
                        vH / 2 - cy_px * self._zoom)
        self._zoom_lbl.setText(f"{self._zoom*100:.0f}%")
        self.update()
        self._push_sync()

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        factor = 1.15 if delta > 0 else (1/1.15)
        mx, my = e.position().x(), e.position().y()
        ox, oy = self._offset
        # Zoom centred on cursor
        new_zoom = max(0.05, min(32.0, self._zoom * factor))
        ratio = new_zoom / self._zoom
        self._offset = (mx - (mx - ox) * ratio,
                        my - (my - oy) * ratio)
        self._zoom = new_zoom
        self._zoom_lbl.setText(f"{self._zoom*100:.0f}%")
        self.update()
        self._push_sync()

    def mousePressEvent(self, e):
        if e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag = True
            self._drag_start  = (e.position().x(), e.position().y())
            self._drag_offset = self._offset
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def mouseMoveEvent(self, e):
        if self._drag:
            dx = e.position().x() - self._drag_start[0]
            dy = e.position().y() - self._drag_start[1]
            self._offset = (self._drag_offset[0] + dx,
                            self._drag_offset[1] + dy)
            self.update()
            self._push_sync()

    def mouseReleaseEvent(self, e):
        if e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag = False
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def mouseDoubleClickEvent(self, e):
        self._fit()   # double-click resets to fit
        self._push_sync()

    def keyPressEvent(self, e):
        # Don't swallow navigation keys — let the parent tab handle them.
        e.ignore()

    def zoom_in(self):
        mx, my = self.width()/2, self.height()/2
        ox, oy = self._offset
        new_zoom = min(32.0, self._zoom * 1.3)
        ratio = new_zoom / self._zoom
        self._offset = (mx - (mx-ox)*ratio, my - (my-oy)*ratio)
        self._zoom = new_zoom
        self._zoom_lbl.setText(f"{self._zoom*100:.0f}%"); self.update()
        self._push_sync()

    def zoom_out(self):
        mx, my = self.width()/2, self.height()/2
        ox, oy = self._offset
        new_zoom = max(0.05, self._zoom / 1.3)
        ratio = new_zoom / self._zoom
        self._offset = (mx - (mx-ox)*ratio, my - (my-oy)*ratio)
        self._zoom = new_zoom
        self._zoom_lbl.setText(f"{self._zoom*100:.0f}%"); self.update()
        self._push_sync()

# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE AOI WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────
class OfflineAOIThread(QThread):
    """
    Runs the full offline AOI pipeline in background:
      golden_path → extract components → calibrate diff thresholds
      test_paths  → YOLO + diff check + render overlay → emit per board
    """
    progress    = Signal(int, str)       # (pct, message)
    board_done  = Signal(str, bytes, int, int, list, list)  # (name, img_bytes, W, H, defects, golden_comps)
    finished    = Signal(dict)           # summary dict

    def __init__(self, golden_path: str, test_paths: list,
                 model_path: str, conf: float,
                 use_sahi: bool, cal_runs: int,
                 filters: list = None, rois: list = None,
                 pipe_model=None):
        super().__init__()
        self._golden_path = golden_path
        self._test_paths  = list(test_paths)
        self._model_path  = model_path
        self._conf        = conf
        self._use_sahi    = use_sahi
        self._cal_runs    = cal_runs
        self._filters     = filters or []
        self._rois        = rois or []
        self._pipe_model  = pipe_model   # pre-built ROI YOLO model
        self._abort       = False

    def abort(self): self._abort = True

    def run(self):
        _t = time   # module-level alias
        def _emit(msg): self.progress.emit(-1, msg)

        if not HAS_YOLO or not HAS_CV2 or not HAS_NP:
            self.progress.emit(0,"[AOI] Missing dependency (ultralytics/cv2/numpy)")
            self.finished.emit({}); return

        # ── Load model ────────────────────────────────────────────────────────
        t0 = _t.time()
        self.progress.emit(2, f"[AOI] Loading model: {os.path.basename(self._model_path)}")
        try:
            model = _YOLO(self._model_path, task="detect")
            _emit(f"[AOI] Model loaded in {_t.time()-t0:.1f}s")
        except Exception as e:
            self.progress.emit(0, f"[AOI] Model load failed: {e}")
            self.finished.emit({}); return

        # ── Load golden ───────────────────────────────────────────────────────
        self.progress.emit(5, f"[AOI] Reading golden: {os.path.basename(self._golden_path)}")
        golden_img = cv2.imread(self._golden_path)
        if golden_img is None:
            self.progress.emit(0, "[AOI] Cannot read golden image")
            self.finished.emit({}); return

        # Apply filter pipeline to golden so diff and inference use the same
        # pre-processing as test boards (apples-to-apples comparison).
        if self._filters:
            golden_img = apply_filters(golden_img, self._filters)
            _emit(f"[AOI] Applied {len(self._filters)} filter(s) to golden image")

        gH, gW = golden_img.shape[:2]
        _emit(f"[AOI] Golden image: {gW}×{gH}  (standard YOLO — SAHI only applies to test boards if enabled)")

        t1 = _t.time()
        golden_dets = _aoi_infer_arr(model, golden_img, self._conf,
                                      model_path=self._model_path,
                                      use_sahi=False,
                                      _log=_emit)
        _counts = {}
        for _c in golden_dets: _counts[_c.label] = _counts.get(_c.label,0)+1
        _emit(f"[AOI] Golden: {len(golden_dets)} components in {_t.time()-t1:.1f}s")
        for _lbl,_n in sorted(_counts.items()):
            _emit(f"  [GOLDEN]   {_lbl:<28} x{_n}")

        if not golden_dets:
            self.progress.emit(0, "[AOI] No components on golden — lower confidence?")
            self.finished.emit({}); return

        # ── Calibrate ─────────────────────────────────────────────────────────
        self.progress.emit(8, f"[AOI] Calibrating ({self._cal_runs} runs)…")
        t1 = _t.time()
        # Use same inference method as golden extraction for calibration.
        # If SAHI was used for golden, calibrate with SAHI too — otherwise
        # pos_tol is calibrated from standard YOLO jitter which is smaller
        # than SAHI merge jitter → every SAHI test det looks 'misaligned'.
        # BUT: limit cal_runs to 3 when SAHI is active (each run is slow).
        # Calibration — always standard YOLO (fast, consistent jitter measurement)
        # SAHI jitter is corrected afterwards by scaling pos_tol
        t_cal = _t.time()
        cal_runs_eff = 3 if (self._use_sahi and max(gH,gW) > _AOI_SAHI_TRIGGER) else self._cal_runs
        if cal_runs_eff != self._cal_runs:
            _emit(f"[CAL] Large SAHI board: capping calibration runs {self._cal_runs}→{cal_runs_eff}")
        pos_tol, per_thr = _aoi_calibrate(
            model, golden_img, golden_dets, self._conf,
            cal_runs_eff, use_sahi=False, model_path=self._model_path)

        # ── SAHI jitter correction ─────────────────────────────────────────
        # Standard YOLO jitter on a 6330px board: pos_tol ≈ 0.010
        # SAHI tile-merge jitter on same board:   pos_tol ≈ 0.03–0.06
        # Without this correction: detect_tol = 0.010×2 = 0.020 but SAHI
        # places real components 0.04 away → MISALIGNED fires on every board.
        is_sahi_board = bool(self._use_sahi) and max(gH, gW) > _AOI_SAHI_TRIGGER
        if is_sahi_board:
            raw_pos_tol = pos_tol
            pos_tol     = float(min(pos_tol * _AOI_SAHI_JITTER_MULT, 0.08))
            match_r     = _AOI_MATCH_R_SAHI
            _emit(f"[CAL] SAHI jitter correction: pos_tol {raw_pos_tol:.4f} → {pos_tol:.4f} "
                  f"(×{_AOI_SAHI_JITTER_MULT})  match_r={match_r:.3f}")
        else:
            match_r = _AOI_MATCH_R

        thr_vals = list(per_thr.values())
        _emit(f"[CAL] Done in {_t.time()-t_cal:.1f}s  "
              f"pos_tol={pos_tol:.4f}  detect_tol={pos_tol*_AOI_DETECT_BAND:.4f}  "
              f"diff_thr min={min(thr_vals):.1f} med={float(np.median(thr_vals)):.1f} max={max(thr_vals):.1f}")
        # Log top-5 components by threshold (highest = hardest to trigger = most noise)
        _id_thr = sorted(per_thr.items(), key=lambda x:-x[1])[:5]
        _gc_map = {g.id:g for g in golden_dets}
        _emit(f"[CAL] Top-5 high-threshold components (noisy regions):")
        for _cid,_thr in _id_thr:
            _g = _gc_map.get(_cid)
            if _g: _emit(f"  thr={_thr:.1f}  {_g.label:<24} cx={_g.cx:.4f} cy={_g.cy:.4f}  px=({int(_g.cx*gW)},{int(_g.cy*gH)})")
        self.progress.emit(15, f"[AOI] Calibrated — {len(golden_dets)} components  pos_tol={pos_tol:.4f}")

        # ── Per-board loop ────────────────────────────────────────────────────
        n = len(self._test_paths)
        board_summaries = []

        for i, path in enumerate(self._test_paths):
            if self._abort:
                _emit("[AOI] Aborted by user"); break
            name = os.path.basename(path)
            pct  = 15 + int((i / max(n, 1)) * 82)
            tb   = _t.time()
            self.progress.emit(pct, f"[AOI] [{i+1}/{n}] {name}")

            test_img = cv2.imread(path)
            if test_img is None:
                _emit(f"[AOI] [{i+1}/{n}] SKIP — cannot read: {name}"); continue

            tH, tW = test_img.shape[:2]

            # Resize to golden size (diff needs pixel alignment)
            if (tW, tH) != (gW, gH):
                test_img = cv2.resize(test_img, (gW, gH))
                _emit(f"[AOI] [{i+1}/{n}] Resized {tW}×{tH} → {gW}×{gH}")

            # Apply filter pipeline
            proc_img = apply_filters(test_img, self._filters) if self._filters else test_img

            # ROI zones
            roi_results = []; roi_fail = False
            for roi in self._rois:
                res = run_roi(proc_img, roi, self._pipe_model)
                if not res.get('passed'): roi_fail = True
                roi_results.append({'name':roi.name,'type':roi.zone_type,
                                    'passed':res.get('passed',False),
                                    'info':res.get('info','')})

            # YOLO inference on test board
            ti = _t.time()
            dets = _aoi_infer_arr(model, proc_img, self._conf,
                                   model_path=self._model_path,
                                   use_sahi=self._use_sahi,
                                   _log=_emit)
            _test_counts = {}
            for _d in dets: _test_counts[_d.label] = _test_counts.get(_d.label,0)+1
            _golden_counts = {}
            for _g in golden_dets: _golden_counts[_g.label] = _golden_counts.get(_g.label,0)+1
            _delta_parts = []
            for _lbl in sorted(set(list(_test_counts)+list(_golden_counts))):
                _gt = _golden_counts.get(_lbl,0); _ts = _test_counts.get(_lbl,0)
                _diff = _ts - _gt
                _delta_parts.append(f"{_lbl[:8]}:{_gt}→{_ts}{'('+str(_diff)+')' if _diff else ''}")
            _emit(f"[AOI] [{i+1}/{n}] Inference: {len(dets)} dets in {_t.time()-ti:.2f}s  "
                  f"vs golden {len(golden_dets)}  delta=[{' | '.join(_delta_parts)}]")

            # Defect check
            # polarity_mult: 2.5 for SAHI/large images (JPEG+tile noise),
            # 1.5 for standard YOLO (less rendering variance)
            poly_mult = 2.5 if (self._use_sahi and max(gH,gW)>_AOI_SAHI_TRIGGER) else 1.5
            defects, board_match = _aoi_check_board(golden_dets, golden_img, proc_img,
                                        dets, pos_tol, per_thr, poly_mult,
                                        match_radius=match_r)
            if roi_fail:
                for rr in roi_results:
                    if not rr['passed']:
                        defects.append({'component_id':-1,'defect_type':'roi_fail',
                            'expected_label':rr['name'],'found_label':rr['type'],
                            'details':f"ROI {rr['name']}: {rr['info']}"})

            _DS = {"missing":"MISSING","wrong_component":"WRONG PART",
                   "misaligned":"MISALIGNED","wrong_polarity":"WRONG POLARITY",
                   "roi_fail":"ROI FAIL"}
            if defects:
                _emit(f"[AOI] [{i+1}/{n}] {len(defects)} DEFECT(S):")
                for _def in defects:
                    _dt  = _DS.get(_def["defect_type"], _def["defect_type"].upper())
                    _el  = _def.get("expected_label","?")
                    _fl  = _def.get("found_label","")
                    _det = _def.get("details","")
                    _cid = _def.get("component_id",-1)
                    _gc  = next((g for g in golden_dets if g.id==_cid), None)
                    if _gc:
                        _px=int(_gc.cx*gW); _py=int(_gc.cy*gH)
                        _coord=f"cx={_gc.cx:.4f} cy={_gc.cy:.4f}  px=({_px},{_py})  box={_gc.w:.3f}×{_gc.h:.3f}"
                    else:
                        _coord="(no location)"
                    _fl_str = f"  → {_fl}" if _fl and _fl!="none" else ""
                    _emit(f"    [{_dt}]  {_el}{_fl_str}")
                    _emit(f"      loc: {_coord}")
                    _emit(f"      det: {_det}")
            else:
                _emit(f"[AOI] [{i+1}/{n}] PASS")

            _emit(f"[AOI] [{i+1}/{n}] Board done in {_t.time()-tb:.1f}s")

            overlay = _aoi_render_overlay(proc_img, golden_dets, dets, defects, golden_img,
                                               match=board_match, match_r=match_r)

            # Encode overlay as PNG bytes for signal
            ok, buf  = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_bytes = bytes(buf.tobytes()) if ok else b""
            ov_h, ov_w = overlay.shape[:2]

            board_summaries.append({
                "name": name, "path": path,
                "n_comps": len(golden_dets),
                "n_defects": len(defects),
                "defects": defects,
                "roi_results": roi_results,
                "pass": len(defects)==0,
            })
            # Include golden component positions for click-zoom in viewer
            golden_for_result = [{"id":g.id,"cx":g.cx,"cy":g.cy,
                                   "w":g.w,"h":g.h,"label":g.label}
                                  for g in golden_dets]
            self.board_done.emit(name, img_bytes, ov_w, ov_h, defects, golden_for_result)

        # Summary
        pass_n = sum(1 for b in board_summaries if b["pass"])
        fail_n = len(board_summaries) - pass_n
        defect_counts = {}
        for b in board_summaries:
            for d in b["defects"]:
                t = d["defect_type"]
                defect_counts[t] = defect_counts.get(t, 0) + 1

        summary = {
            "total": len(board_summaries), "pass": pass_n, "fail": fail_n,
            "defect_counts": defect_counts, "boards": board_summaries,
            "golden_components": len(golden_dets),
            "pos_tol": pos_tol,
        }
        self.progress.emit(100, f"[AOI] Done — {pass_n}/{len(board_summaries)} PASS")
        self.finished.emit(summary)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — OFFLINE AOI INSPECTION
# ─────────────────────────────────────────────────────────────────────────────

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
            try: self._pipe_model = load_optimized_yolo(model_path, task="detect")
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
                try: self._pipe_model = load_optimized_yolo(mp, task="detect")
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
class _InferThread(QThread):
    result   = Signal(str, bytes, int, int, list)  # name, jpg_bytes, W, H, dets
    log      = Signal(str)
    finished = Signal()

    def __init__(self, paths, model_path, conf, use_sahi, grab_frame_fn=None):
        super().__init__()
        self._paths      = paths          # list of file paths OR empty for single frame
        self._model_path = model_path
        self._conf       = conf
        self._use_sahi   = use_sahi
        self._grab       = grab_frame_fn  # callable → np array, or None
        self._abort      = False

    def stop(self): self._abort = True

    def run(self):
        if not HAS_YOLO:
            self.log.emit("[INFER] ultralytics not available"); self.finished.emit(); return
        _t = time   # module-level alias — no frozen-module lookup overhead
        # _YOLO already imported at module level; no re-import needed
        try:
            self.log.emit(f"[INFER] Loading model: {os.path.basename(self._model_path)}")
            model = load_optimized_yolo(self._model_path, task="detect")
            self.log.emit(f"[INFER] Model loaded")
        except Exception as e:
            self.log.emit(f"[INFER] Model load failed: {e}"); self.finished.emit(); return

        # If grab_frame_fn supplied → infer on one live frame
        sources = self._paths if self._paths else ["__frame__"]
        for src_path in sources:
            if self._abort: break

            if src_path == "__frame__" and self._grab:
                img = self._grab()
                if img is None:
                    self.log.emit("[INFER] Could not grab frame"); continue
                name = "frame"
            else:
                img = cv2.imread(src_path)
                if img is None:
                    self.log.emit(f"[INFER] Cannot read: {src_path}"); continue
                name = os.path.basename(src_path)

            H, W = img.shape[:2]
            t0 = _t.time()

            large = max(H, W) > _AOI_SAHI_TRIGGER
            if self._use_sahi and large and HAS_SAHI:
                dets = _aoi_infer_arr(model, img, self._conf,
                                      model_path=self._model_path,
                                      use_sahi=True, _log=self.log.emit)
            else:
                if self._use_sahi and large and not HAS_SAHI:
                    self.log.emit("[INFER] SAHI not installed — using standard YOLO")
                dets = _aoi_infer_arr(model, img, self._conf,
                                      model_path=self._model_path,
                                      use_sahi=False)

            elapsed = _t.time() - t0
            counts = {}
            for d in dets: counts[d.label] = counts.get(d.label,0)+1
            self.log.emit(
                f"[INFER] {name}  {len(dets)} dets  {elapsed:.2f}s  "
                + "  ".join(f"{l}×{n}" for l,n in sorted(counts.items()))
            )

            # Render overlay
            overlay = img.copy()
            _INFER_COLS = {
                "resistor (r)": (0,200,100),
                "capacitor (c)": (0,150,255),
                "ic (u)": (200,100,0),
                "transistor (q)": (255,50,200),
            }
            for d in dets:
                x1,y1,x2,y2 = d.xyxy(W,H)
                col = _INFER_COLS.get(d.label.lower(), (100,220,100))
                cv2.rectangle(overlay,(x1,y1),(x2,y2),col,2)
                tag = f"{d.label} {d.conf:.2f}"
                (tw,th),_ = cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.30,1)
                cv2.rectangle(overlay,(x1,max(0,y1-th-4)),(x1+tw+2,y1),col,-1)
                cv2.putText(overlay,tag,(x1+1,y1-2),cv2.FONT_HERSHEY_SIMPLEX,
                            0.30,(0,0,0),1,cv2.LINE_AA)

            ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY,88])
            img_bytes = bytes(buf.tobytes()) if ok else b""
            self.result.emit(name, img_bytes, W, H, dets)

        self.finished.emit()


class InferTab(QWidget):
    """Standalone YOLO/SAHI inference — no AOI pipeline, no diff, just detections."""

    def __init__(self, cfg):
        super().__init__()
        self._cfg    = cfg
        self._thread = None
        self._results = []
        self._cur    = 0
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(10)

        # ── Top bar: model + conf + SAHI toggle + run ─────────────────────────
        top = QHBoxLayout(); top.setSpacing(6)

        top.addWidget(QLabel("Model:"))
        self._model_lbl = QLabel(os.path.basename(
            self._cfg.get("model_path", "no model")))
        self._model_lbl.setStyleSheet(f"color:{TEXT_SEC};font-size:11px;")
        top.addWidget(self._model_lbl, 1)

        top.addWidget(QLabel("Conf:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95); self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(float(self._cfg.get("conf", 0.25)))
        self._conf_spin.setFixedWidth(62)
        top.addWidget(self._conf_spin)

        self._sahi_cb = QCheckBox("SAHI")
        self._sahi_cb.setChecked(True)
        top.addWidget(self._sahi_cb)

        root.addLayout(top)

        # ── Source row ────────────────────────────────────────────────────────
        src_row = QHBoxLayout(); src_row.setSpacing(6)
        b_file = QPushButton("📂 File(s)"); b_file.setFixedHeight(28)
        b_dir  = QPushButton("📁 Folder");  b_dir.setFixedHeight(28)
        b_frame= QPushButton("📷 Grab Frame"); b_frame.setFixedHeight(28)
        b_file.clicked.connect(self._pick_files)
        b_dir.clicked.connect(self._pick_folder)
        b_frame.clicked.connect(self._grab_frame)
        src_row.addWidget(b_file); src_row.addWidget(b_dir)
        src_row.addWidget(b_frame); src_row.addStretch()
        self._run_btn = QPushButton("▶  Run Inference")
        self._run_btn.setFixedHeight(28)
        self._run_btn.setStyleSheet(
            f"QPushButton{{background:{CYAN};color:#FFFFFF;border:none;"
            f"border-radius:4px;font-weight:600;padding:0 12px;}}"
            f"QPushButton:disabled{{background:{BG_DEEP};color:{TEXT_DIM};}}"
        )
        self._run_btn.clicked.connect(self._run)
        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.setFixedHeight(28); self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        src_row.addWidget(self._run_btn); src_row.addWidget(self._stop_btn)
        root.addLayout(src_row)

        # ── File list ─────────────────────────────────────────────────────────
        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(70)
        self._file_list.setStyleSheet(
            f"background:{BG_DEEP};border:1px solid {BG_BORDER2};"
            f"border-radius:6px;font-size:11px;color:{TEXT_SEC};")
        root.addWidget(self._file_list)

        # ── Image viewer ──────────────────────────────────────────────────────
        self._img_view = ZoomableImageView()
        root.addWidget(self._img_view, 6)

        # ── Nav + stats bar ───────────────────────────────────────────────────
        nav = QHBoxLayout(); nav.setSpacing(6)
        self._prev_btn = QPushButton("\u2190  Previous"); self._prev_btn.setFixedHeight(28)
        self._next_btn = QPushButton("Next  \u2192"); self._next_btn.setFixedHeight(28)
        self._prev_btn.clicked.connect(self._prev)
        self._next_btn.clicked.connect(self._next)
        self._pos_lbl  = QLabel("—"); self._pos_lbl.setStyleSheet(f"color:{TEXT_SEC};font-size:12px;")
        self._det_lbl  = QLabel(""); self._det_lbl.setStyleSheet(f"color:{CYAN};font-size:12px;")
        nav.addWidget(self._prev_btn); nav.addWidget(self._pos_lbl)
        nav.addWidget(self._next_btn); nav.addStretch()
        nav.addWidget(self._det_lbl)
        root.addLayout(nav)

        # ── Log ───────────────────────────────────────────────────────────────
        self._log = FastLog()
        self._log.setMaximumHeight(80)
        self._log.setStyleSheet(
            f"background:{BG_DEEP};border:1px solid {BG_BORDER2};"
            f"border-radius:6px;padding:4px;font-family:Consolas;font-size:10px;"
            f"color:{TEXT_SEC};")
        root.addWidget(self._log)

        self._paths = []

        # Window-level shortcuts — work regardless of which child widget has focus
        sc_l = QShortcut(QKeySequence(Qt.Key.Key_Left),  self)
        sc_r = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        sc_f = QShortcut(QKeySequence(Qt.Key.Key_F6),    self)
        sc_l.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_r.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_f.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_l.activated.connect(self._prev)
        sc_r.activated.connect(self._next)
        sc_f.activated.connect(self._run)

    # ── File picking ──────────────────────────────────────────────────────────
    def _pick_files(self):
        files,_ = QFileDialog.getOpenFileNames(self,"Select images","",
                                               "Images (*.jpg *.jpeg *.png *.bmp *.tiff)")
        if files: self._set_paths(files)

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self,"Select folder")
        if d:
            exts = {".jpg",".jpeg",".png",".bmp",".tiff"}
            files = sorted([os.path.join(d,f) for f in os.listdir(d)
                            if os.path.splitext(f)[1].lower() in exts])
            self._set_paths(files)

    def _set_paths(self, paths):
        self._paths = paths
        self._file_list.clear()
        for p in paths: self._file_list.addItem(os.path.basename(p))

    def _grab_frame(self):
        """Grab one frame from the RunTab camera and infer on it."""
        self._paths = []
        self._run(frame_mode=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    def _run(self, frame_mode=False):
        if self._thread and self._thread.isRunning(): return
        model_path = self._cfg.get("model_path","")
        if not model_path or not os.path.exists(model_path):
            self._log.append("[INFER] No model — set model path in RUN tab first"); return

        paths = [] if frame_mode else self._paths
        if not paths and not frame_mode:
            self._log.append("[INFER] No images selected"); return

        # Grab frame from RunTab's camera if in frame mode.
        # FIX: _run._raw_px is a QPixmap (display cache), not a numpy array.
        # Using it caused AttributeError on .shape[:2] inside _InferThread.
        # get_latest_frame() returns the BGR ndarray directly from the camera thread.
        grab_fn = None
        if frame_mode:
            try:
                mw = self.window()
                frame = mw._run.get_latest_frame()   # → BGR ndarray or None
                if frame is not None:
                    frame = frame.copy()             # snapshot before camera thread overwrites
                    grab_fn = lambda: frame
                else:
                    self._log.append("[INFER] No camera frame available — start camera in RUN tab first")
                    return
            except Exception as ex:
                self._log.append(f"[INFER] Cannot grab frame: {ex}"); return

        self._results = []; self._cur = 0
        self._run_btn.setEnabled(False); self._stop_btn.setEnabled(True)

        self._thread = _InferThread(paths, model_path,
                                    self._conf_spin.value(),
                                    self._sahi_cb.isChecked(),
                                    grab_fn)
        self._thread.log.connect(self._log.append)
        self._thread.result.connect(self._on_result)
        self._thread.finished.connect(self._on_done)
        self._thread.start()

    def _stop(self):
        if self._thread: self._thread.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    def _on_result(self, name, img_bytes, W, H, dets):
        self._results.append({"name":name,"img_bytes":img_bytes,
                               "W":W,"H":H,"dets":dets})
        self._cur = len(self._results)-1
        self._show_current()

    def _on_done(self):
        self._run_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self._log.append(f"[INFER] Done — {len(self._results)} image(s)")

    def _show_current(self):
        if not self._results: return
        r = self._results[self._cur]
        if r["img_bytes"]:
            self._img_view.load_bytes(r["img_bytes"])
        counts = {}
        for d in r["dets"]: counts[d.label]=counts.get(d.label,0)+1
        self._pos_lbl.setText(f"{self._cur+1}/{len(self._results)}  {r['name']}")
        self._det_lbl.setText(
            f"{len(r['dets'])} det(s)  " +
            "  ".join(f"{l}:{n}" for l,n in sorted(counts.items())))

    def _prev(self):
        if self._results:
            self._cur = (self._cur-1) % len(self._results)
            self._show_current()

    def _next(self):
        if self._results:
            self._cur = (self._cur+1) % len(self._results)
            self._show_current()




# ─────────────────────────────────────────────────────────────────────────────
# TOAST NOTIFICATION SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
class ToastManager:
    """Bottom-right corner toast notifications. Call ToastManager.show(parent, msg, kind)."""
    _active: list = []

    @classmethod
    def show(cls, parent, message: str, kind: str = "info", duration: int = 3000):
        return  # toast notifications disabled

    @classmethod
    def _restack(cls, parent):
        cls._active = [w for w in cls._active if w.isVisible()]
        pw, ph = parent.width(), parent.height()
        y = ph - 48
        for toast in reversed(cls._active):
            toast.setFixedWidth(310)
            toast.adjustSize()
            h = max(44, toast.sizeHint().height())
            y -= h
            toast.move(pw - 326, y)
            y -= 8


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY DATABASE  (SQLite)
# ─────────────────────────────────────────────────────────────────────────────
class HistoryDB:
    """Thread-safe SQLite-backed inspection result log."""
    DB_PATH = os.path.join(DATA_ROOT, "history.db")

    def __init__(self):
        os.makedirs(DATA_ROOT, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(self.DB_PATH, check_same_thread=False)
        # WAL: readers don't block writers; NORMAL sync is safe and 3–5× faster
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        # 8 MB page cache → fewer disk reads on history scroll
        self._con.execute("PRAGMA cache_size=-8192")
        # 64 MB memory-mapped I/O → near-zero latency on SSD and HDD alike
        self._con.execute("PRAGMA mmap_size=67108864")
        self._create_tables()

    def _create_tables(self):
        with self._lock:
            self._con.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts            TEXT    NOT NULL,
                    project       TEXT    DEFAULT '',
                    mode          TEXT    DEFAULT 'RUN',
                    passed        INTEGER DEFAULT 0,
                    n_defects     INTEGER DEFAULT 0,
                    defect_types  TEXT    DEFAULT '',
                    model         TEXT    DEFAULT '',
                    board_serial  TEXT    DEFAULT '',
                    image_path    TEXT    DEFAULT ''
                )
            """)
            # Migration: silently add image_path to existing databases
            try:
                self._con.execute("ALTER TABLE results ADD COLUMN image_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass   # column already present
            # Indexes: history tab sorts/filters by ts and project —
            # without these, every refresh is a full-table scan.
            self._con.execute(
                "CREATE INDEX IF NOT EXISTS idx_results_ts      ON results(ts)"
            )
            self._con.execute(
                "CREATE INDEX IF NOT EXISTS idx_results_project ON results(project, ts)"
            )
            self._con.commit()

    def log_result(self, project="", mode="RUN", passed=True,
                   n_defects=0, defect_types="", model="", board_serial=""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cur = self._con.execute(
                "INSERT INTO results "
                "(ts,project,mode,passed,n_defects,defect_types,model,board_serial) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ts, project, mode, 1 if passed else 0,
                 n_defects, defect_types, model, board_serial)
            )
            self._con.commit()
            return cur.lastrowid   # caller can call update_image_path() once the file is saved

    def update_image_path(self, row_id: int, path: str):
        """Store the saved annotated image path for an existing result row."""
        if not row_id:
            return
        with self._lock:
            self._con.execute(
                "UPDATE results SET image_path=? WHERE id=?", (path, row_id))
            self._con.commit()

    def fetch(self, project=None, days=30, mode=None, passed=None, limit=500):
        q = "SELECT id,ts,project,mode,passed,n_defects,defect_types,model,image_path FROM results WHERE 1=1"
        p: list = []
        if project:      q += " AND project=?";                        p.append(project)
        if mode:         q += " AND mode=?";                           p.append(mode)
        if passed is not None: q += " AND passed=?";                   p.append(1 if passed else 0)
        if days:         q += " AND ts>=datetime('now',?)";            p.append(f"-{days} days")
        q += f" ORDER BY ts DESC LIMIT {limit}"
        with self._lock:
            return self._con.execute(q, p).fetchall()

    def yield_trend(self, project=None, days=14, bucket="day"):
        """Return [(date_str, total, pass_n)] bucketed by day."""
        fmt = "%Y-%m-%d" if bucket == "day" else "%Y-%m-%d %H"
        q = ("SELECT strftime(?,ts),COUNT(*),SUM(passed) FROM results WHERE 1=1"
             + (" AND project=?" if project else "")
             + (f" AND ts>=datetime('now','-{days} days')" if days else "")
             + " GROUP BY strftime(?,ts) ORDER BY 1")
        params = [fmt] + ([project] if project else []) + [fmt]
        with self._lock:
            return self._con.execute(q, params).fetchall()

    def projects(self):
        with self._lock:
            return [r[0] for r in self._con.execute(
                "SELECT DISTINCT project FROM results WHERE project!='' ORDER BY project"
            ).fetchall()]

    def export_csv(self, path):
        import csv
        rows = self.fetch(days=None, limit=100000)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ID","Timestamp","Project","Mode","Passed","Defects","DefectTypes","Model"])
            w.writerows(rows)

    def close(self):
        with self._lock:
            self._con.close()


# ─────────────────────────────────────────────────────────────────────────────
# RESULT STRIP WIDGET  (last-10 boards coloured squares)
# ─────────────────────────────────────────────────────────────────────────────
class ResultStrip(QWidget):
    """Horizontal row of 10 coloured squares showing last board results."""
    N = 10
    def __init__(self):
        super().__init__()
        self._results: deque = deque(maxlen=self.N)   # True=pass False=fail
        self.setFixedHeight(32)
        self.setToolTip("Last 10 board results (newest right)")

    def push(self, passed: bool):
        self._results.append(passed); self.update()

    def clear_results(self):
        self._results.clear(); self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        sq = 22; gap = 4
        total_w = self.N * sq + (self.N-1) * gap
        x0 = (W - total_w) // 2; y0 = (H - sq) // 2
        results_list = list(self._results)
        for i in range(self.N):
            x = x0 + i*(sq+gap)
            # align newest to right
            data_idx = i - (self.N - len(results_list))
            if 0 <= data_idx < len(results_list):
                col = QColor(GREEN) if results_list[data_idx] else QColor(RED)
                col.setAlpha(220)
                p.setBrush(QBrush(col))
                p.setPen(QPen(col.lighter(130), 1))
            else:
                p.setBrush(QBrush(QColor(BG_BORDER)))
                p.setPen(QPen(QColor(BG_BORDER2), 1))
            p.drawRoundedRect(x, y0, sq, sq, 4, 4)
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY TAB
# ─────────────────────────────────────────────────────────────────────────────
