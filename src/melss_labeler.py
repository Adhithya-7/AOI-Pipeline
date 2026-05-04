"""
MELSS VisionForge — Standalone Labeler  v2
Changes in v2:
  • Middle-mouse-button drag to pan when zoomed
  • Scroll-wheel zoom pivots toward cursor position
  • Labels auto-saved on every box add / edit / delete
  • Create Project button (makes images/ labels/ data.yaml)
  • Quick-select: Transistor (Q), Resistor (R), Capacitor (C), IC
"""
import sys, os, json, shutil, platform, time

try:    import cv2;                            HAS_CV2  = True
except: HAS_CV2  = False
try:    import numpy as np;                    HAS_NP   = True
except: HAS_NP   = False; np = None
try:    from ultralytics import YOLO as _YOLO; HAS_YOLO = True
except: HAS_YOLO = False

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QListWidget, QListWidgetItem,
    QFileDialog, QProgressBar, QDoubleSpinBox, QComboBox,
    QSizePolicy, QMessageBox, QInputDialog, QPlainTextEdit,
    QScrollArea, QLineEdit
)
from PySide6.QtCore  import Qt, QTimer, QThread, Signal, Slot, QSize, QPoint
from PySide6.QtGui   import (
    QFont, QPixmap, QPainter, QColor, QPen, QBrush, QCursor,
    QImage, QIcon, QPalette, QTextCursor
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG_DEEP="#080b0f"; BG_PANEL="#0d1017"; BG_CARD="#111620"; BG_CARD2="#161c28"
BG_BORDER="#1c2333"; BG_BORDER2="#243040"
CYAN="#00d4f5"; CYAN_DIM="#006b8a"
GREEN="#00e676"; GREEN_DIM="#00522a"
AMBER="#f5a623"; AMBER_DIM="#6b470f"
RED="#f5454a";   RED_DIM="#6b1517"
PURPLE="#a78bfa"
TEXT_PRI="#dde4ed"; TEXT_SEC="#6b7a8d"; TEXT_DIM="#38424f"

DATA_ROOT = "VisionForge_Data"
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "best.pt"
OG_MODEL  = str(DEFAULT_MODEL)

MASTER_CLASS_LIST = [
    "Connector (P)",       # 0
    "Resistor (R)",        # 1
    "Transformer (T)",     # 2
    "Diode (D)",           # 3
    "Capacitor (C)",       # 4
    "Transistor (Q)",      # 5
    "Jumper (J)",          # 6
    "Inductor (L)",        # 7
    "IC (U)",              # 8
    "Resistor Array (RA)", # 9
    "Resistor Net (RN)",   # 10
    "Crystal (CR)",        # 11
    "IC (IC)",             # 12
    "Jumper (JP)",         # 13
    "Varistor (V)",        # 14
    "Button (BTN)",        # 15
    "Switch (SW)",         # 16
    "Switch (S)",          # 17
    "Test Point (TP)",     # 18
    "LED",                 # 19
    "Transistor (QA)",     # 20
    "Cap Array (CRA)",     # 21
    "Motor (M)",           # 22
    "Fuse (F)",            # 23
    "Ferrite Bead (FB)",   # 24
]

BOX_COLORS = {
    0:"#FFA500", 1:"#FF3333", 2:"#AA00FF", 3:"#00FF88", 4:"#FFFF00",
    5:"#00FFFF", 6:"#FF88AA", 7:"#88FF88", 8:"#3366FF", 9:"#FF8800",
    10:"#00AAFF",11:"#FF00FF",12:"#66AAFF",13:"#FFAA66",14:"#AA66FF",
}
DEF_COL = "#CCCCCC"

_F_MONO_8  = QFont("Consolas", 8, QFont.Weight.Bold)
_F_MONO_9  = QFont("Consolas", 9, QFont.Weight.Bold)
_F_MONO_11 = QFont("Consolas", 11)
_F_MONO_12 = QFont("Consolas", 12, QFont.Weight.Bold)

GLOBAL_QSS = f"""
QWidget{{color:{TEXT_PRI};font-family:'Segoe UI',sans-serif;font-size:12px;}}
QPushButton{{
  background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {BG_CARD2},stop:1 {BG_CARD});
  color:{TEXT_PRI};border:1px solid {BG_BORDER2};border-radius:6px;
  padding:5px 14px;font-size:11px;font-weight:600;}}
QPushButton:hover{{border-color:{TEXT_DIM};}}
QPushButton:pressed{{background:{BG_CARD};border-color:{CYAN_DIM};color:{CYAN};}}
QPushButton:disabled{{background:{BG_CARD};color:{TEXT_DIM};border-color:{BG_BORDER};}}
QLineEdit{{background:{BG_DEEP};color:{TEXT_PRI};border:1px solid {BG_BORDER2};
  border-radius:6px;padding:3px 8px;}}
QLineEdit:focus{{border-color:{CYAN_DIM};}}
QComboBox,QDoubleSpinBox{{background:{BG_DEEP};color:{TEXT_PRI};
  border:1px solid {BG_BORDER2};border-radius:6px;padding:3px 8px;min-height:26px;}}
QComboBox:focus,QDoubleSpinBox:focus{{border-color:{CYAN_DIM};}}
QComboBox::drop-down{{border:none;width:20px;}}
QComboBox QAbstractItemView{{background:{BG_CARD2};color:{TEXT_PRI};
  border:1px solid {BG_BORDER2};selection-background-color:{CYAN_DIM};outline:none;}}
QListWidget{{background:{BG_DEEP};border:1px solid {BG_BORDER};
  border-radius:6px;outline:none;padding:2px;}}
QListWidget::item{{padding:5px 8px;border-radius:4px;color:{TEXT_SEC};font-size:11px;}}
QListWidget::item:selected{{background:{CYAN_DIM};color:{TEXT_PRI};}}
QListWidget::item:hover:!selected{{background:{BG_CARD2};color:{TEXT_PRI};}}
QProgressBar{{background:{BG_DEEP};border:1px solid {BG_BORDER};border-radius:4px;
  text-align:center;font-size:10px;color:{TEXT_DIM};}}
QProgressBar::chunk{{
  background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {CYAN_DIM},stop:1 {CYAN});
  border-radius:3px;}}
QScrollBar:vertical{{background:{BG_DEEP};width:8px;border-radius:4px;margin:0;}}
QScrollBar::handle:vertical{{background:{BG_BORDER2};border-radius:4px;min-height:30px;}}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}
QScrollBar:horizontal{{background:{BG_DEEP};height:8px;border-radius:4px;}}
QScrollBar::handle:horizontal{{background:{BG_BORDER2};border-radius:4px;min-width:30px;}}
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal{{width:0;}}
QScrollArea{{border:none;background:transparent;}}
QPlainTextEdit{{background:{BG_DEEP};color:{TEXT_PRI};border:1px solid {BG_BORDER2};
  border-radius:6px;padding:4px 8px;selection-background-color:{CYAN_DIM};}}
"""

# ── UI helpers ────────────────────────────────────────────────────────────────
def make_card(oid):
    f = QFrame(); f.setObjectName(oid)
    f.setStyleSheet(
        f"#{oid}{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
        f"stop:0 {BG_CARD2},stop:1 {BG_CARD});"
        f"border:1px solid {BG_BORDER};border-radius:10px;}}"
    )
    return f

def make_sep():
    f = QFrame(); f.setFrameShape(QFrame.Shape.HLine); f.setObjectName("sep")
    f.setStyleSheet(
        f"#sep{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
        f"stop:0 transparent,stop:0.2 {BG_BORDER2},stop:0.8 {BG_BORDER2},stop:1 transparent);"
        f"max-height:1px;border:none;margin:4px 0;}}"
    )
    return f

def sec_lbl(text):
    l = QLabel(text.upper()); l.setObjectName("seclbl")
    l.setStyleSheet(
        f"#seclbl{{color:{CYAN};font-size:9px;font-weight:bold;"
        f"font-family:'Consolas';letter-spacing:3px;border:none;padding:2px 0;}}"
    )
    return l


# ── FastLog ───────────────────────────────────────────────────────────────────
class FastLog(QPlainTextEdit):
    MAX_LINES = 500
    def __init__(self, parent=None):
        super().__init__(parent); self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)
        self._buf = []; self._timer = QTimer(self)
        self._timer.setSingleShot(True); self._timer.setInterval(100)
        self._timer.timeout.connect(self._flush)
    def append(self, text: str):
        self._buf.append(str(text))
        if not self._timer.isActive(): self._timer.start()
    def _flush(self):
        if not self._buf: return
        sb = self.verticalScrollBar(); at_bottom = sb.value() >= sb.maximum() - 4
        block = "\n".join(self._buf); self._buf.clear()
        cursor = self.textCursor(); cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(("\n" if self.document().blockCount() > 1 else "") + block)
        if at_bottom: sb.setValue(sb.maximum())
    def clear(self):
        self._buf.clear(); self._timer.stop(); super().clear()


# ── SAHI helpers ──────────────────────────────────────────────────────────────
def _sahi_slices(img_w, img_h, tile_sz, overlap):
    """Yield (x1,y1,x2,y2) tile rectangles covering the full image."""
    step = max(1, int(tile_sz * (1 - overlap)))
    xs = list(range(0, img_w - tile_sz, step)) + [max(0, img_w - tile_sz)]
    ys = list(range(0, img_h - tile_sz, step)) + [max(0, img_h - tile_sz)]
    seen = set()
    for y in ys:
        for x in xs:
            r = (x, y, min(x+tile_sz,img_w), min(y+tile_sz,img_h))
            if r not in seen:
                seen.add(r); yield r

def _iou(a, b):
    """IoU of two [x1,y1,x2,y2] boxes."""
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
    inter=iw*ih
    if inter==0: return 0.0
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def _nms(dets, iou_thr=0.5):
    """dets: list of [cls, score, x1,y1,x2,y2].  Returns filtered list."""
    if not dets: return []
    dets = sorted(dets, key=lambda d: -d[1])
    keep = []
    while dets:
        best = dets.pop(0); keep.append(best)
        dets = [d for d in dets if d[0]!=best[0] or _iou(best[2:],d[2:])<iou_thr]
    return keep


# ── AutoLabelThread ───────────────────────────────────────────────────────────
class AutoLabelThread(QThread):
    progress = Signal(int, str); finished = Signal(int)

    def __init__(self, model, inbox, out, conf,
                 use_sahi=False, tile_sz=640, overlap=0.2, iou_thr=0.5):
        super().__init__()
        self.model   = model
        self.inbox   = inbox
        self.out     = out
        self.conf    = conf
        self.use_sahi= use_sahi
        self.tile_sz = tile_sz
        self.overlap = overlap
        self.iou_thr = iou_thr

    # ── single full-image pass ────────────────────────────────────────────────
    def _predict_full(self, path):
        res = self.model.predict(path, conf=self.conf, verbose=False)[0]
        h, w = res.orig_shape
        dets = []
        for box in res.boxes:
            cls = int(box.cls[0])
            sc  = float(box.conf[0])
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            dets.append([cls, sc, x1, y1, x2, y2])
        return w, h, dets

    # ── SAHI tiled pass ───────────────────────────────────────────────────────
    def _predict_sahi(self, path):
        if not HAS_CV2 and not HAS_NP:
            return self._predict_full(path)    # fallback: no cv2/numpy
        import cv2 as _cv2, numpy as _np

        img = _cv2.imread(path)
        if img is None:
            return self._predict_full(path)
        img_h, img_w = img.shape[:2]

        all_dets = []

        # also run a full-image pass so large components aren't missed
        _, _, full_dets = self._predict_full(path)
        all_dets.extend(full_dets)

        tile_sz  = self.tile_sz
        # If image is smaller than tile, skip tiling
        if img_w <= tile_sz and img_h <= tile_sz:
            return img_w, img_h, _nms(all_dets, self.iou_thr)

        for (tx1,ty1,tx2,ty2) in _sahi_slices(img_w, img_h, tile_sz, self.overlap):
            tile = img[ty1:ty2, tx1:tx2]
            # Encode tile to jpg bytes for YOLO
            # Pass tile as RGB numpy array — YOLO accepts np.ndarray directly.
            # (raw bytes / encoded buffer are NOT supported by ultralytics predict)
            tile_rgb = _cv2.cvtColor(tile, _cv2.COLOR_BGR2RGB)
            res = self.model.predict(tile_rgb, conf=self.conf, verbose=False)[0]
            th, tw = res.orig_shape
            for box in res.boxes:
                cls = int(box.cls[0])
                sc  = float(box.conf[0])
                bx1,by1,bx2,by2 = box.xyxy[0].tolist()
                # remap tile-local → full-image coords
                all_dets.append([cls, sc,
                                  bx1+tx1, by1+ty1,
                                  bx2+tx1, by2+ty1])

        merged = _nms(all_dets, self.iou_thr)
        return img_w, img_h, merged

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        if not os.path.exists(self.inbox): self.finished.emit(0); return
        os.makedirs(self.out, exist_ok=True)
        files = [f for f in os.listdir(self.inbox)
                 if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        total = len(files); count = 0
        for i, fn in enumerate(files):
            ip = os.path.join(self.inbox, fn)
            try:
                if self.use_sahi:
                    w, h, dets = self._predict_sahi(ip)
                else:
                    w, h, dets = self._predict_full(ip)

                labels = []
                for d in dets:
                    cls,sc,x1,y1,x2,y2 = d
                    cx=(x1+x2)/2/w; cy=(y1+y2)/2/h
                    bw=(x2-x1)/w;   bh=(y2-y1)/h
                    labels.append([int(cls), cx, cy, bw, bh])

                if labels:
                    out_path = os.path.join(self.out, os.path.splitext(fn)[0]+".txt")
                    with open(out_path,"w") as f:
                        for l in labels:
                            f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
                    count += 1
                    mode = "SAHI" if self.use_sahi else "full"
                    self.progress.emit(int((i+1)/total*100),
                                       f"{fn}: {len(labels)} boxes [{mode}]")
                else:
                    self.progress.emit(int((i+1)/total*100),
                                       f"{fn}: 0 det (existing labels kept)")
            except Exception as e:
                self.progress.emit(int((i+1)/total*100), f"Err {fn}: {e}")
        self.finished.emit(count)


# ── ThumbThread ───────────────────────────────────────────────────────────────
class ThumbThread(QThread):
    thumb_done = Signal(int, bytes, int, int); all_done = Signal()
    def __init__(self, images, size=88):
        super().__init__(); self._images = list(images)
        self._size = size; self._abort = False
    def abort(self): self._abort = True; self.wait(1000)
    def run(self):
        sz = self._size
        for i, path in enumerate(self._images):
            if self._abort: return
            try:
                if HAS_CV2:
                    bgr = cv2.imread(path)
                    if bgr is None: time.sleep(0.06); continue
                    h, w = bgr.shape[:2]; scale = min(sz/w, sz/h)
                    nw, nh = max(1,int(w*scale)), max(1,int(h*scale))
                    small = cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_LINEAR)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    self.thumb_done.emit(i, bytes(rgb.tobytes()), nw, nh)
                else:
                    px = QPixmap(path)
                    if not px.isNull():
                        px = px.scaled(sz,sz,Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.FastTransformation)
                        img = px.toImage().convertToFormat(QImage.Format.Format_RGB888)
                        ptr = img.bits(); ptr.setsize(img.sizeInBytes())
                        self.thumb_done.emit(i, bytes(ptr), img.width(), img.height())
            except: pass
            time.sleep(0.06)
        self.all_done.emit()


# ── LabelCanvas ───────────────────────────────────────────────────────────────
#  Left drag (empty)  → draw box
#  Left drag (body)   → move box
#  Left drag (handle) → resize box
#  Middle drag        → pan image
#  Right click        → delete box
#  Scroll wheel       → zoom toward cursor
class LabelCanvas(QWidget):
    box_added   = Signal(list)
    box_changed = Signal()

    HS = 7   # handle half-size px

    # 0=TL 1=TC 2=TR  3=ML 4=MR  5=BL 6=BC 7=BR
    CURSORS = {
        0: Qt.CursorShape.SizeFDiagCursor,
        1: Qt.CursorShape.SizeVerCursor,
        2: Qt.CursorShape.SizeBDiagCursor,
        3: Qt.CursorShape.SizeHorCursor,
        4: Qt.CursorShape.SizeHorCursor,
        5: Qt.CursorShape.SizeBDiagCursor,
        6: Qt.CursorShape.SizeVerCursor,
        7: Qt.CursorShape.SizeFDiagCursor,
    }

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self._px     = QPixmap()
        self._scaled = None
        self._ow = self._oh = 1
        self._scale  = 1.0
        self._ox = self._oy = 0.0

        self._boxes   = []
        self._sel_cls = 0

        # Draw
        self._drawing = False
        self._sx = self._sy = self._mx = self._my = 0

        # Edit (resize / move)
        self._sel_box       = -1
        self._drag_handle   = -1
        self._drag_box_orig = None
        self._drag_start_px = None
        self._editing       = False

        # Pan (middle mouse)
        self._panning      = False
        self._pan_start_px = None
        self._pan_start_ox = 0.0
        self._pan_start_oy = 0.0

        self._cx_pos = QPoint(-1, -1)

        # 2-click draw mode (alternative to drag)
        # True = click-click mode, False = drag mode
        self._click_mode = False
        self._click1 = None   # QPoint of first click when waiting for 2nd

        self._rtimer = QTimer(); self._rtimer.setSingleShot(True)
        self._rtimer.timeout.connect(self._fit)

    # ── I/O ──────────────────────────────────────────────────────────────────
    def load_image(self, path):
        self._px = QPixmap(); self._px.load(path); self._scaled = None
        self._ow = self._px.width()  if not self._px.isNull() else 1
        self._oh = self._px.height() if not self._px.isNull() else 1
        self._boxes = []; self._sel_box = -1; self._fit()

    def load_labels(self, txt):
        self._boxes = []; self._sel_box = -1
        if os.path.exists(txt):
            for line in open(txt):
                try: self._boxes.append(list(map(float, line.strip().split())))
                except: pass
        self.update()

    def save_labels(self, txt):
        if not self._boxes:
            # Never write an empty file — it would destroy existing labels.
            # If the user explicitly cleared boxes, _clear_lbl removes the file directly.
            return
        os.makedirs(os.path.dirname(os.path.abspath(txt)), exist_ok=True)
        with open(txt, "w") as f:
            for b in self._boxes:
                f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    def clear_labels(self): self._boxes = []; self._sel_box = -1; self._click1 = None; self.update()
    def set_class(self, idx): self._sel_cls = idx
    def set_click_mode(self, enabled): self._click_mode = enabled; self._click1 = None; self.update()
    def get_boxes(self): return self._boxes

    # ── Layout ───────────────────────────────────────────────────────────────
    def _fit(self):
        if self._px.isNull(): self.update(); return
        cw, ch = max(1,self.width()), max(1,self.height())
        self._scale = min(cw/self._ow, ch/self._oh)
        self._ox = (cw - self._ow*self._scale)/2
        self._oy = (ch - self._oh*self._scale)/2
        self._scaled = None; self.update()

    def resizeEvent(self, e): self._rtimer.start(60)

    # ── Coordinate helpers ────────────────────────────────────────────────────
    def _box_to_widget(self, b):
        cls,cx,cy,bw,bh = b; sc = self._scale
        x1 = int((cx-bw/2)*self._ow*sc + self._ox)
        y1 = int((cy-bh/2)*self._oh*sc + self._oy)
        x2 = int((cx+bw/2)*self._ow*sc + self._ox)
        y2 = int((cy+bh/2)*self._oh*sc + self._oy)
        return x1,y1,x2,y2

    def _handles(self, x1,y1,x2,y2):
        mx=(x1+x2)//2; my=(y1+y2)//2
        return [(x1,y1),(mx,y1),(x2,y1),
                (x1,my),       (x2,my),
                (x1,y2),(mx,y2),(x2,y2)]

    def _hit_handle(self, px,py,x1,y1,x2,y2):
        HS=self.HS
        for i,(hx,hy) in enumerate(self._handles(x1,y1,x2,y2)):
            if abs(px-hx)<=HS and abs(py-hy)<=HS: return i
        return -1

    def _hit_box(self, px,py):
        for i in range(len(self._boxes)-1,-1,-1):
            x1,y1,x2,y2 = self._box_to_widget(self._boxes[i])
            h = self._hit_handle(px,py,x1,y1,x2,y2)
            if h>=0: return i,h
            if x1<=px<=x2 and y1<=py<=y2: return i,-1
        return -1,-1

    def _widget_to_norm(self, wx,wy):
        nx=(wx-self._ox)/(self._ow*self._scale)
        ny=(wy-self._oy)/(self._oh*self._scale)
        return max(0.,min(1.,nx)),max(0.,min(1.,ny))

    def _apply_handle_drag(self, bidx,hidx,wx,wy):
        b=list(self._boxes[bidx])
        cx,cy,bw,bh=b[1],b[2],b[3],b[4]
        nx,ny=self._widget_to_norm(wx,wy)
        lft=cx-bw/2; rgt=cx+bw/2; top=cy-bh/2; bot=cy+bh/2
        if   hidx==0: lft,top=nx,ny
        elif hidx==1: top=ny
        elif hidx==2: rgt,top=nx,ny
        elif hidx==3: lft=nx
        elif hidx==4: rgt=nx
        elif hidx==5: lft,bot=nx,ny
        elif hidx==6: bot=ny
        elif hidx==7: rgt,bot=nx,ny
        lft,rgt=min(lft,rgt),max(lft,rgt)
        top,bot=min(top,bot),max(top,bot)
        b[1]=(lft+rgt)/2; b[2]=(top+bot)/2
        b[3]=max(.005,rgt-lft); b[4]=max(.005,bot-top)
        self._boxes[bidx]=b

    def _apply_body_drag(self, bidx,dx_w,dy_w):
        b=list(self._boxes[bidx])
        dnx=dx_w/(self._ow*self._scale); dny=dy_w/(self._oh*self._scale)
        b[1]=max(b[3]/2,min(1-b[3]/2,b[1]+dnx))
        b[2]=max(b[4]/2,min(1-b[4]/2,b[2]+dny))
        self._boxes[bidx]=b

    def _clamp_viewport(self):
        dw=self._ow*self._scale; dh=self._oh*self._scale
        W,H=self.width(),self.height(); mg=40
        self._ox=max(mg-dw, min(W-mg, self._ox))
        self._oy=max(mg-dh, min(H-mg, self._oy))

    # ── Paint ─────────────────────────────────────────────────────────────────
    def paintEvent(self, e):
        p=QPainter(self); W,H=self.width(),self.height()
        p.fillRect(0,0,W,H,QColor(20,20,25))
        if self._px.isNull(): return

        dw=int(self._ow*self._scale); dh=int(self._oh*self._scale)
        ox=int(self._ox); oy=int(self._oy)

        if self._scaled is None or self._scaled.width()!=dw or self._scaled.height()!=dh:
            self._scaled=self._px.scaled(dw,dh,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation)
        p.drawPixmap(ox,oy,self._scaled)

        # Crosshair
        cpx,cpy=self._cx_pos.x(),self._cx_pos.y()
        if 0<=cpx<W and 0<=cpy<H:
            p.setPen(QPen(QColor(0,212,245,100),1))
            p.drawLine(ox,cpy,ox+dw,cpy)       # horizontal
            p.drawLine(cpx,oy,cpx,oy+dh)       # vertical
            p.setPen(QPen(QColor(0,212,245,210),1)); cs=10
            p.drawLine(cpx-cs,cpy,cpx+cs,cpy)
            p.drawLine(cpx,cpy-cs,cpx,cpy+cs)

        # Boxes
        p.setFont(_F_MONO_8)
        for i,b in enumerate(self._boxes):
            cls=int(b[0]); cx2,cy2,bw2,bh2=b[1],b[2],b[3],b[4]
            x1=int((cx2-bw2/2)*self._ow*self._scale+ox)
            y1=int((cy2-bh2/2)*self._oh*self._scale+oy)
            pw2=int(bw2*self._ow*self._scale); ph2=int(bh2*self._oh*self._scale)
            col=QColor(BOX_COLORS.get(cls,DEF_COL)); is_sel=(i==self._sel_box)
            p.setPen(QPen(col,3 if is_sel else 2))
            p.setBrush(QBrush(QColor(col.red(),col.green(),col.blue(),45 if is_sel else 22)))
            p.drawRect(x1,y1,pw2,ph2)
            name=MASTER_CLASS_LIST[cls] if cls<len(MASTER_CLASS_LIST) else str(cls)
            txt=f"{cls}:{name[:13]}"; cw2=len(txt)*6+4; ch2=13
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(col.red(),col.green(),col.blue(),180)))
            p.drawRect(x1,y1-ch2,cw2,ch2)
            p.setPen(QPen(QColor(0,0,0))); p.drawText(x1+2,y1-2,txt)
            if is_sel:
                x2h=x1+pw2; y2h=y1+ph2; hs=self.HS
                p.setPen(QPen(QColor(255,255,255,220),1))
                p.setBrush(QBrush(QColor(col.red(),col.green(),col.blue(),220)))
                for hx,hy in self._handles(x1,y1,x2h,y2h):
                    p.drawRect(hx-hs,hy-hs,hs*2,hs*2)

        # Rubber-band (drag mode)
        if self._drawing:
            x1r=min(self._sx,self._mx); y1r=min(self._sy,self._my)
            p.setPen(QPen(QColor(CYAN),1,Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(x1r,y1r,abs(self._mx-self._sx),abs(self._my-self._sy))

        # 2-click mode: first anchor set, previewing second corner
        if self._click_mode and self._click1 is not None:
            ax,ay=self._click1.x(),self._click1.y()
            cpx2,cpy2=self._cx_pos.x(),self._cx_pos.y()
            if cpx2>=0:
                x1r=min(ax,cpx2); y1r=min(ay,cpy2)
                p.setPen(QPen(QColor(AMBER),1,Qt.PenStyle.DashLine))
                p.setBrush(QBrush(QColor(245,166,35,18)))
                p.drawRect(x1r,y1r,abs(cpx2-ax),abs(cpy2-ay))
            # Draw anchor crosshair dot
            p.setPen(QPen(QColor(AMBER),2))
            p.setBrush(QBrush(QColor(AMBER)))
            p.drawEllipse(ax-5,ay-5,10,10)

    # ── Mouse ─────────────────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        mx,my=int(e.position().x()),int(e.position().y())

        if e.button()==Qt.MouseButton.MiddleButton:
            self._panning=True
            self._pan_start_px=QPoint(mx,my)
            self._pan_start_ox=self._ox; self._pan_start_oy=self._oy
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor)); return

        if e.button()==Qt.MouseButton.RightButton:
            bi,_=self._hit_box(mx,my)
            if bi>=0:
                self._boxes.pop(bi)
                if   self._sel_box==bi: self._sel_box=-1
                elif self._sel_box>bi:  self._sel_box-=1
                self.box_changed.emit(); self.update()
            return

        if e.button()==Qt.MouseButton.LeftButton:
            bi,hi=self._hit_box(mx,my)
            if bi>=0:
                # Always go straight into edit/move regardless of draw mode
                self._click1=None  # cancel any pending click-mode anchor
                self._sel_box=bi; self._editing=True
                self._drag_handle=hi; self._drag_box_orig=list(self._boxes[bi])
                self._drag_start_px=QPoint(mx,my)
                self._update_cursor(hi); self.update(); return

            # ── 2-click mode ───────────────────────────────────────────────
            if self._click_mode:
                if self._click1 is None:
                    # First click → store anchor
                    self._sel_box=-1
                    self._click1=QPoint(mx,my); self.update()
                else:
                    # Second click → commit box
                    ax,ay=self._click1.x(),self._click1.y()
                    self._click1=None
                    x1,y1=min(ax,mx),min(ay,my)
                    x2,y2=max(ax,mx),max(ay,my)
                    if (x2-x1)>8 and (y2-y1)>8:
                        sc=self._scale
                        cx=((x1+x2)/2-self._ox)/(self._ow*sc)
                        cy=((y1+y2)/2-self._oy)/(self._oh*sc)
                        bw=(x2-x1)/(self._ow*sc); bh=(y2-y1)/(self._oh*sc)
                        box=[self._sel_cls,
                             max(0,min(1,cx)),max(0,min(1,cy)),
                             max(0,min(1,bw)),max(0,min(1,bh))]
                        self._boxes.append(box)
                        self._sel_box=len(self._boxes)-1
                        self.box_added.emit(box)
                    self.update()
                return

            # ── Drag mode ──────────────────────────────────────────────────
            self._sel_box=-1; self._drawing=True
            self._sx=self._mx=mx; self._sy=self._my=my
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor)); self.update()

    def mouseMoveEvent(self, e):
        mx,my=int(e.position().x()),int(e.position().y())
        self._cx_pos=QPoint(mx,my)

        if self._panning:
            dx=mx-self._pan_start_px.x(); dy=my-self._pan_start_px.y()
            self._ox=self._pan_start_ox+dx; self._oy=self._pan_start_oy+dy
            self._clamp_viewport(); self._scaled=None; self.update(); return

        if self._editing:
            if self._drag_handle>=0:
                self._apply_handle_drag(self._sel_box,self._drag_handle,mx,my)
            else:
                dx=mx-self._drag_start_px.x(); dy=my-self._drag_start_px.y()
                self._boxes[self._sel_box]=list(self._drag_box_orig)
                self._apply_body_drag(self._sel_box,dx,dy)
            self.update(); return

        if self._drawing:
            self._mx=mx; self._my=my; self.update(); return

        bi,hi=self._hit_box(mx,my)
        if bi>=0: self._update_cursor(hi)
        else:     self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button()==Qt.MouseButton.MiddleButton:
            self._panning=False
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.update(); return

        if e.button()==Qt.MouseButton.LeftButton:
            if self._editing:
                self._editing=False; self._drag_handle=-1; self._drag_box_orig=None
                self.box_changed.emit()
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
                self.update(); return
            if self._drawing:
                self._drawing=False
                x1,y1=min(self._sx,self._mx),min(self._sy,self._my)
                x2,y2=max(self._sx,self._mx),max(self._sy,self._my)
                if (x2-x1)>8 and (y2-y1)>8:
                    sc=self._scale
                    cx=((x1+x2)/2-self._ox)/(self._ow*sc)
                    cy=((y1+y2)/2-self._oy)/(self._oh*sc)
                    bw=(x2-x1)/(self._ow*sc); bh=(y2-y1)/(self._oh*sc)
                    box=[self._sel_cls,
                         max(0,min(1,cx)),max(0,min(1,cy)),
                         max(0,min(1,bw)),max(0,min(1,bh))]
                    self._boxes.append(box)
                    self._sel_box=len(self._boxes)-1
                    self.box_added.emit(box)
                self.update()

    def wheelEvent(self, e):
        """Zoom pivoted on cursor position."""
        mx,my=e.position().x(),e.position().y()
        factor=1.12 if e.angleDelta().y()>0 else 1/1.12
        new_scale=max(0.05,min(20.0,self._scale*factor))
        # Keep the image pixel under cursor fixed
        img_x=(mx-self._ox)/(self._ow*self._scale)
        img_y=(my-self._oy)/(self._oh*self._scale)
        self._scale=new_scale
        self._ox=mx-img_x*self._ow*self._scale
        self._oy=my-img_y*self._oh*self._scale
        self._clamp_viewport(); self._scaled=None; self.update()

    def leaveEvent(self, e): self._cx_pos=QPoint(-1,-1); self.update()

    def keyPressEvent(self, e):
        if e.key()==Qt.Key.Key_Escape and self._click1 is not None:
            self._click1=None; self.update()
        else:
            super().keyPressEvent(e)

    def _update_cursor(self, handle_idx):
        if handle_idx<0: self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        else:            self.setCursor(QCursor(self.CURSORS.get(handle_idx,Qt.CursorShape.SizeAllCursor)))


# ── Main window ───────────────────────────────────────────────────────────────
class LabelerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MELSS VisionForge — Labeler v2")
        self.setMinimumSize(1140,740)

        self._project  = None
        self._images   = []
        self._cur      = 0
        self._label_cache = {}
        self._px_cache = {}
        self._auto_thread = self._thumb_thread = None
        self._cached_model = None; self._cached_model_path = None
        self._thumb_buf = []; self._thumb_flush_pending = False

        # Auto-save timer — debounce 300 ms after last box change
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True); self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._save_cur_silent)
        self._loading = False   # re-entry guard: True while _load_cur is running

        central=QWidget(); self.setCentralWidget(central)
        root=QHBoxLayout(central); root.setContentsMargins(10,10,10,10); root.setSpacing(10)
        self._build_left(root); self._build_centre(root); self._build_right(root)

        # Wire auto-save
        self._canvas.box_added.connect(self._schedule_autosave)
        self._canvas.box_changed.connect(self._schedule_autosave)
        # Wire box list refresh
        self._canvas.box_added.connect(lambda _: self._refresh_box_list())
        self._canvas.box_changed.connect(self._refresh_box_list)

    # ── Panel builders ────────────────────────────────────────────────────────
    def _build_left(self, root):
        lf=make_card("lf_lbl"); ll=QVBoxLayout(lf)
        ll.setContentsMargins(6,6,6,6); ll.setSpacing(6)
        ll.addWidget(sec_lbl("PROJECT"))

        pb=QHBoxLayout()
        b_new=QPushButton("＋ Create"); b_new.setFixedHeight(32); b_new.setObjectName("b_new")
        b_new.setStyleSheet(
            f"#b_new{{background:{GREEN_DIM};color:{GREEN};"
            f"border:1px solid {GREEN_DIM};border-radius:5px;font-weight:bold;}}"
            f"#b_new:hover{{background:#004d1a;border-color:{GREEN};}}")
        b_new.clicked.connect(self._create_project)

        b_open=QPushButton("Open"); b_open.setFixedHeight(32); b_open.setObjectName("b_open")
        b_open.setStyleSheet(
            f"#b_open{{background:{CYAN_DIM};color:#fff;border:none;border-radius:5px;}}"
            f"#b_open:hover{{background:{CYAN};color:#000;}}")
        b_open.clicked.connect(self._open_project)
        pb.addWidget(b_new,1); pb.addWidget(b_open,1); ll.addLayout(pb)

        self._proj_lbl=QLabel("No project loaded"); self._proj_lbl.setObjectName("projlbl")
        self._proj_lbl.setStyleSheet(
            f"#projlbl{{color:{TEXT_DIM};font-size:10px;font-family:Consolas;border:none;}}")
        self._proj_lbl.setWordWrap(True); ll.addWidget(self._proj_lbl)

        br=QHBoxLayout()
        b_up=QPushButton("Upload Images"); b_up.setObjectName("b_up"); b_up.setFixedHeight(30)
        b_up.setStyleSheet(
            f"#b_up{{background:{BG_CARD2};color:{TEXT_PRI};"
            f"border:1px solid {BG_BORDER2};border-radius:5px;font-size:10px;}}"
            f"#b_up:hover{{border-color:{CYAN_DIM};}}")
        b_up.clicked.connect(self._upload)
        b_fo=QPushButton("⬢ Folder"); b_fo.setFixedHeight(30)
        b_fo.setStyleSheet(
            f"QPushButton{{background:{BG_CARD2};color:{TEXT_SEC};"
            f"border:1px solid {BG_BORDER};border-radius:5px;font-size:10px;}}")
        b_fo.clicked.connect(self._open_folder)
        br.addWidget(b_up,2); br.addWidget(b_fo,1); ll.addLayout(br)

        self._thumbs=QListWidget(); self._thumbs.setObjectName("thumbs")
        self._thumbs.setStyleSheet(
            "#thumbs{background:#0a0c10;border:1px solid #1c2333;border-radius:6px;}")
        self._thumbs.setViewMode(QListWidget.ViewMode.IconMode)
        self._thumbs.setIconSize(QSize(88,88))
        self._thumbs.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._thumbs.setUniformItemSizes(True); self._thumbs.setSpacing(3)
        self._thumbs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._thumbs.currentRowChanged.connect(self._on_select)
        ll.addWidget(self._thumbs,1)

        self._stats=QLabel("0 images"); self._stats.setObjectName("stats_lbl")
        self._stats.setStyleSheet(
            f"#stats_lbl{{color:{TEXT_SEC};font-size:10px;font-family:Consolas;border:none;}}")
        ll.addWidget(self._stats)

        b_del_img=QPushButton("🗑  Delete Image from Disk")
        b_del_img.setObjectName("b_del_img"); b_del_img.setFixedHeight(30)
        b_del_img.setStyleSheet(
            f"#b_del_img{{background:{RED_DIM};color:{RED};"
            f"border:1px solid {RED_DIM};border-radius:5px;"
            f"font-size:10px;font-weight:bold;}}"
            f"#b_del_img:hover{{background:#8b0a0d;border-color:{RED};}}")
        b_del_img.clicked.connect(self._delete_cur_image)
        ll.addWidget(b_del_img)
        root.addWidget(lf,1)

    def _build_centre(self, root):
        cf=make_card("cf_panel"); cl=QVBoxLayout(cf)
        cl.setContentsMargins(4,4,4,4); cl.setSpacing(4)

        # ── Filename bar ──────────────────────────────────────────────────
        fbar=QFrame(); fbar.setObjectName("fbar")
        fbar.setStyleSheet(
            f"#fbar{{background:{BG_DEEP};border:1px solid {BG_BORDER2};"
            f"border-radius:6px;}}")
        fbl=QHBoxLayout(fbar); fbl.setContentsMargins(10,4,8,4); fbl.setSpacing(8)

        fil=QLabel("📄"); fil.setStyleSheet(f"color:{CYAN};border:none;font-size:13px;")
        fbl.addWidget(fil)

        self._fname_edit=QLineEdit("No image loaded")
        self._fname_edit.setReadOnly(True)
        self._fname_edit.setObjectName("fnedit")
        self._fname_edit.setStyleSheet(
            f"#fnedit{{background:transparent;color:{TEXT_PRI};"
            f"border:none;font-family:Consolas;font-size:12px;font-weight:bold;"
            f"selection-background-color:{CYAN_DIM};}}")
        fbl.addWidget(self._fname_edit,1)

        self._fsize_lbl=QLabel("")
        self._fsize_lbl.setStyleSheet(
            f"color:{TEXT_DIM};font-size:10px;font-family:Consolas;border:none;")
        fbl.addWidget(self._fsize_lbl)

        b_copy=QPushButton("⎘ Copy Name"); b_copy.setFixedHeight(24)
        b_copy.setObjectName("b_copy")
        b_copy.setStyleSheet(
            f"#b_copy{{background:{BG_CARD2};color:{CYAN};"
            f"border:1px solid {BG_BORDER2};border-radius:4px;"
            f"padding:0 8px;font-size:10px;font-weight:bold;}}"
            f"#b_copy:hover{{background:{CYAN_DIM};color:#fff;}}")
        b_copy.setToolTip("Copy full filename to clipboard")
        b_copy.clicked.connect(self._copy_filename)
        fbl.addWidget(b_copy)
        cl.addWidget(fbar)

        # Canvas
        self._canvas=LabelCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        cl.addWidget(self._canvas,1)

        self._autosave_lbl=QLabel("● AUTO-SAVE ON"); self._autosave_lbl.setObjectName("aslbl")
        self._autosave_lbl.setStyleSheet(
            f"#aslbl{{color:{GREEN};font-size:9px;font-family:Consolas;"
            f"letter-spacing:1px;border:none;}}")
        self._autosave_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

        ctrl=QHBoxLayout(); ctrl.setSpacing(4); ctrl.addWidget(QLabel("Class:"))
        self._cls_cb=QComboBox()
        for c in MASTER_CLASS_LIST: self._cls_cb.addItem(c)
        self._cls_cb.currentIndexChanged.connect(self._canvas.set_class)
        ctrl.addWidget(self._cls_cb,1)

        # Quick picks: Transistor Q=5, Resistor R=1, Capacitor C=4, IC=8
        QUICK=[(5,"Q","#00FFFF"),(1,"R","#FF3333"),(4,"C","#FFFF00"),(8,"IC","#3366FF")]
        for idx,lbl,col in QUICK:
            b=QPushButton(lbl); b.setFixedSize(32,28); oid=f"qc_{idx}"; b.setObjectName(oid)
            b.setStyleSheet(
                f"#{oid}{{background:{col};color:black;border-radius:4px;"
                f"font-size:9px;font-weight:bold;border:none;}}")
            b.clicked.connect(
                lambda _,i=idx:(self._canvas.set_class(i),
                                self._cls_cb.setCurrentIndex(i)))
            ctrl.addWidget(b)

        # Draw-mode toggle: Drag ↔ 2-Click
        self._draw_mode_btn=QPushButton("✎ Drag"); self._draw_mode_btn.setFixedHeight(28)
        self._draw_mode_btn.setObjectName("b_dm"); self._draw_mode_btn.setCheckable(True)
        self._draw_mode_btn.setToolTip(
            "Toggle draw mode:\n  Drag = click+drag to draw\n  2-Click = click corner1, click corner2")
        self._draw_mode_btn.setStyleSheet(
            f"#b_dm{{background:{BG_CARD2};color:{TEXT_SEC};"
            f"border:1px solid {BG_BORDER2};border-radius:5px;"
            f"font-size:10px;font-weight:bold;padding:0 8px;}}"
            f"#b_dm:checked{{background:#2d1f4e;color:{PURPLE};"
            f"border-color:{PURPLE};}}")
        self._draw_mode_btn.toggled.connect(self._toggle_draw_mode)
        ctrl.addWidget(self._draw_mode_btn)

        b_prev=QPushButton("◀ Prev"); b_prev.setFixedHeight(30); b_prev.clicked.connect(self._prev)
        self._nav=QLabel("0/0"); self._nav.setObjectName("navlbl")
        self._nav.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._nav.setStyleSheet(
            f"#navlbl{{color:{CYAN};font-family:Consolas;min-width:60px;border:none;}}")

        b_next=QPushButton("Next ▶"); b_next.setObjectName("b_next"); b_next.setFixedHeight(30)
        b_next.setStyleSheet(
            f"#b_next{{background:{GREEN_DIM};color:{GREEN};"
            f"border:1px solid {GREEN_DIM};border-radius:5px;}}"
            f"#b_next:hover{{background:#004d1a;}}")
        b_next.clicked.connect(self._next)

        b_clr=QPushButton("Clear"); b_clr.setObjectName("b_clr"); b_clr.setFixedHeight(30)
        b_clr.setStyleSheet(
            f"#b_clr{{background:{BG_CARD};color:{RED};"
            f"border:1px solid {RED_DIM};border-radius:5px;}}")
        b_clr.clicked.connect(self._clear_lbl)

        ctrl.addWidget(b_prev); ctrl.addWidget(self._nav)
        ctrl.addWidget(b_next); ctrl.addWidget(b_clr)
        ctrl.addWidget(self._autosave_lbl)
        cl.addLayout(ctrl)
        root.addWidget(cf,3)

    def _build_right(self, root):
        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(270); scroll.setMaximumWidth(310)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        scroll.setObjectName("rscroll"); scroll.setStyleSheet("#rscroll{border:none;}")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        rw=QWidget(); rw.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        rl=QVBoxLayout(rw); rl.setSpacing(6); rl.setContentsMargins(4,4,4,4)

        rl.addWidget(sec_lbl("AUTO-LABELER"))
        rl.addWidget(QLabel("Teacher model (.pt):"))
        tr=QHBoxLayout()
        self._teacher=QLineEdit(); self._teacher.setPlaceholderText("path/to/best.pt")
        if os.path.exists(OG_MODEL): self._teacher.setText(OG_MODEL)
        b_br=QPushButton("…"); b_br.setFixedWidth(28)
        b_br.clicked.connect(lambda: self._teacher.setText(
            QFileDialog.getOpenFileName(self,"Select model","","*.pt")[0]
            or self._teacher.text()))
        tr.addWidget(self._teacher,1); tr.addWidget(b_br); rl.addLayout(tr)

        cr=QHBoxLayout(); cr.addWidget(QLabel("Conf:"))
        self._auto_conf=QDoubleSpinBox()
        self._auto_conf.setRange(0.1,0.99); self._auto_conf.setValue(0.5)
        self._auto_conf.setFixedWidth(80); cr.addWidget(self._auto_conf); cr.addStretch()
        rl.addLayout(cr)

        # ── SAHI controls ─────────────────────────────────────────────────
        sahi_card=QFrame(); sahi_card.setObjectName("sahi_card")
        sahi_card.setStyleSheet(
            f"#sahi_card{{background:{BG_DEEP};border:1px solid {BG_BORDER2};"
            f"border-radius:6px;padding:2px;}}")
        sl=QVBoxLayout(sahi_card); sl.setContentsMargins(8,6,8,6); sl.setSpacing(4)

        sahi_hdr=QHBoxLayout()
        self._sahi_chk=QPushButton("⊞  Sliced Inference  (SAHI)")
        self._sahi_chk.setCheckable(True); self._sahi_chk.setObjectName("sahi_chk")
        self._sahi_chk.setToolTip(
            "Tile the image and run the model on each tile so\n"
            "small / dense components are not missed.\n"
            "Also runs a full-image pass to catch large items.")
        self._sahi_chk.setStyleSheet(
            f"#sahi_chk{{background:{BG_CARD2};color:{TEXT_SEC};"
            f"border:1px solid {BG_BORDER2};border-radius:5px;"
            f"font-size:10px;font-weight:bold;padding:4px 8px;text-align:left;}}"
            f"#sahi_chk:checked{{background:#1a1040;color:{PURPLE};"
            f"border-color:{PURPLE};}}")
        self._sahi_chk.toggled.connect(self._on_sahi_toggled)
        sahi_hdr.addWidget(self._sahi_chk,1); sl.addLayout(sahi_hdr)

        self._sahi_opts=QWidget(); sol=QVBoxLayout(self._sahi_opts)
        sol.setContentsMargins(0,2,0,0); sol.setSpacing(3)
        self._sahi_opts.setVisible(False)

        tr1=QHBoxLayout()
        tr1.addWidget(QLabel("Tile size:"))
        self._sahi_tile=QComboBox()
        for t in ["320","416","512","640","800","1024"]:
            self._sahi_tile.addItem(f"{t} px", int(t))
        self._sahi_tile.setCurrentIndex(3)   # 640 default
        self._sahi_tile.setFixedWidth(90)
        tr1.addWidget(self._sahi_tile); tr1.addStretch(); sol.addLayout(tr1)

        tr2=QHBoxLayout()
        tr2.addWidget(QLabel("Overlap:"))
        self._sahi_overlap=QDoubleSpinBox()
        self._sahi_overlap.setRange(0.05,0.5); self._sahi_overlap.setValue(0.2)
        self._sahi_overlap.setSingleStep(0.05); self._sahi_overlap.setFixedWidth(70)
        self._sahi_overlap.setToolTip("Fraction of tile overlapping the adjacent tile\n(0.2 = 20%)")
        tr2.addWidget(self._sahi_overlap); tr2.addStretch(); sol.addLayout(tr2)

        tr3=QHBoxLayout()
        tr3.addWidget(QLabel("Merge IoU:"))
        self._sahi_iou=QDoubleSpinBox()
        self._sahi_iou.setRange(0.1,0.9); self._sahi_iou.setValue(0.5)
        self._sahi_iou.setSingleStep(0.05); self._sahi_iou.setFixedWidth(70)
        self._sahi_iou.setToolTip("IoU threshold for NMS when merging boxes\nfrom different tiles")
        tr3.addWidget(self._sahi_iou); tr3.addStretch(); sol.addLayout(tr3)

        sl.addWidget(self._sahi_opts)
        rl.addWidget(sahi_card)

        b_al=QPushButton("▶  Auto-Label All"); b_al.setObjectName("b_al"); b_al.setFixedHeight(34)
        b_al.setStyleSheet(
            f"#b_al{{background:{AMBER_DIM};color:{AMBER};"
            f"border:1px solid {AMBER_DIM};border-radius:5px;font-weight:bold;}}"
            f"#b_al:hover{{background:#9a6600;border-color:{AMBER};}}")
        b_al.clicked.connect(self._auto_label); rl.addWidget(b_al)

        self._al_prog=QProgressBar(); self._al_prog.setFixedHeight(5)
        self._al_prog.setValue(0); rl.addWidget(self._al_prog)

        self._al_log=FastLog(); self._al_log.setMaximumHeight(80); self._al_log.setObjectName("allog")
        self._al_log.setStyleSheet(
            f"#allog{{background:{BG_CARD};border:1px solid {BG_BORDER};"
            f"border-radius:4px;padding:3px;color:{AMBER};"
            f"font-family:Consolas;font-size:10px;}}")
        rl.addWidget(self._al_log); rl.addWidget(make_sep())

        rl.addWidget(sec_lbl("CONTROLS"))
        hints=[("Draw (drag)",  "Left drag"),
               ("Draw (2-clk)", "Toggle → click×2"),
               ("Move box",     "Drag body"),
               ("Resize",       "Drag handle"),
               ("Delete box",   "Right-click"),
               ("Pan",          "Middle drag"),
               ("Zoom",         "Scroll wheel"),
               ("Navigate",     "← / →"),
               ("Cancel 2-clk", "Esc")]
        for action,key in hints:
            hr=QHBoxLayout()
            la=QLabel(action); la.setStyleSheet(f"color:{TEXT_SEC};font-size:10px;border:none;")
            lk=QLabel(key);   lk.setStyleSheet(
                f"color:{CYAN};font-size:10px;font-family:Consolas;border:none;")
            lk.setAlignment(Qt.AlignmentFlag.AlignRight)
            hr.addWidget(la,1); hr.addWidget(lk); rl.addLayout(hr)

        rl.addWidget(make_sep()); rl.addWidget(sec_lbl("CURRENT BOXES"))
        self._box_list=QListWidget(); self._box_list.setMaximumHeight(200)
        self._box_list.setObjectName("boxlist")
        self._box_list.setStyleSheet(
            "#boxlist{background:#0a0c10;border:1px solid #1c2333;border-radius:6px;}")
        self._box_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._box_list.currentRowChanged.connect(
            lambda i:(setattr(self._canvas,'_sel_box',i),self._canvas.update()))
        rl.addWidget(self._box_list)

        b_del=QPushButton("Delete Selected Box"); b_del.setFixedHeight(28)
        b_del.setStyleSheet(
            f"QPushButton{{background:{BG_CARD};color:{RED};"
            f"border:1px solid {RED_DIM};border-radius:5px;font-size:10px;}}")
        b_del.clicked.connect(self._del_selected_box); rl.addWidget(b_del)
        rl.addStretch(); scroll.setWidget(rw); root.addWidget(scroll)

    # ── Auto-save ─────────────────────────────────────────────────────────────
    def _schedule_autosave(self, *_): self._save_timer.start()

    def _save_cur_silent(self):
        if not self._images or self._cur>=len(self._images): return
        ip=self._images[self._cur]; lp=self._lbl_path(ip)
        try:
            self._canvas.save_labels(lp)
            has=bool(self._canvas.get_boxes())
            if self._label_cache.get(ip)!=has:
                self._label_cache[ip]=has
                item=self._thumbs.item(self._cur)
                if item:
                    item.setText(f"{'✓' if has else '·'} {os.path.basename(ip)[:18]}")
                    item.setForeground(QColor(GREEN if has else AMBER))
            self._update_stats()
            self._autosave_lbl.setText("✓ SAVED")
            self._autosave_lbl.setStyleSheet(
                f"#aslbl{{color:{GREEN};font-size:9px;font-family:Consolas;"
                f"font-weight:bold;letter-spacing:1px;border:none;}}")
            QTimer.singleShot(900,self._reset_autosave_badge)
        except Exception as e:
            self._autosave_lbl.setText("! ERR")
            self._autosave_lbl.setStyleSheet(
                f"#aslbl{{color:{RED};font-size:9px;font-family:Consolas;border:none;}}")

    def _reset_autosave_badge(self):
        self._autosave_lbl.setText("● AUTO-SAVE ON")
        self._autosave_lbl.setStyleSheet(
            f"#aslbl{{color:{GREEN};font-size:9px;font-family:Consolas;"
            f"letter-spacing:1px;border:none;}}")

    def _copy_filename(self):
        if not self._images or self._cur>=len(self._images): return
        ip=self._images[self._cur]
        QApplication.clipboard().setText(os.path.basename(ip))
        # brief visual feedback
        self._fname_edit.setStyleSheet(
            f"background:{CYAN_DIM}33;color:{CYAN};border:none;"
            f"font-family:Consolas;font-size:12px;font-weight:bold;")
        QTimer.singleShot(500, lambda: self._fname_edit.setStyleSheet(
            f"background:transparent;color:{TEXT_PRI};border:none;"
            f"font-family:Consolas;font-size:12px;font-weight:bold;"
            f"selection-background-color:{CYAN_DIM};"))

    def _toggle_draw_mode(self, checked):
        self._canvas.set_click_mode(checked)
        self._draw_mode_btn.setText("✦ 2-Click" if checked else "✎ Drag")

    def _on_sahi_toggled(self, checked):
        self._sahi_opts.setVisible(checked)
        self._sahi_chk.setText(
            "⊞  Sliced Inference  (ON)" if checked else "⊞  Sliced Inference  (SAHI)")

    def _delete_cur_image(self):
        if not self._images or self._cur>=len(self._images): return
        ip=self._images[self._cur]
        fname=os.path.basename(ip)
        r=QMessageBox.warning(self,"Delete Image",
            f"Permanently delete:\n{fname}\n\nThis also removes the label file.",
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.Cancel)
        if r!=QMessageBox.StandardButton.Yes: return

        # 1. Stop any pending auto-save FIRST — prevents stale boxes being
        #    written to whatever image becomes current after the delete.
        self._save_timer.stop()

        # 2. Delete the image file
        try: os.remove(ip)
        except Exception as e:
            QMessageBox.critical(self,"Error",f"Could not delete image:\n{e}"); return

        # 3. Delete the matching label file
        lp=self._lbl_path(ip)
        if os.path.exists(lp):
            try: os.remove(lp)
            except: pass

        # 4. Wipe the canvas immediately so nothing stale is on screen
        self._canvas.clear_labels()

        # 5. Work out where to land after reload (same index, clamped)
        self._cur = min(self._cur, max(0, len(self._images)-2))

        # 6. Abort thumb thread and discard pending in-flight results whose
        #    indices are now wrong (takeItem would have shifted them all).
        if self._thumb_thread and self._thumb_thread.isRunning():
            self._thumb_thread.abort()
        self._thumb_buf=[]; self._thumb_flush_pending=False

        # 7. Full clean reload — rebuilds image list, label cache, thumb list
        #    and pixmap cache from disk; no stale index mapping is possible.
        self._load_images_keep_cur()

    # ── Project management ────────────────────────────────────────────────────
    def _create_project(self):
        base=QFileDialog.getExistingDirectory(self,"Choose parent folder","")
        if not base: return
        name,ok=QInputDialog.getText(self,"New Project","Project name:")
        if not ok or not name.strip(): return
        path=os.path.join(base,name.strip())
        try:
            for sub in ["images","labels"]:
                os.makedirs(os.path.join(path,sub),exist_ok=True)
            with open(os.path.join(path,"data.yaml"),"w") as f:
                f.write(
                    f"path: {os.path.abspath(path).replace(chr(92),'/')}\n"
                    f"train: images\nval: images\n"
                    f"nc: {len(MASTER_CLASS_LIST)}\n"
                    f"names: {MASTER_CLASS_LIST}\n")
            self._load_project(path)
        except Exception as e:
            QMessageBox.critical(self,"Error",f"Could not create project:\n{e}")

    def _open_project(self):
        d=QFileDialog.getExistingDirectory(self,"Open Project Folder","")
        if d: self._load_project(d)

    def _load_project(self, path):
        self._project=path; name=os.path.basename(path)
        self._proj_lbl.setText(name)
        self.setWindowTitle(f"MELSS Labeler — {name}")
        os.makedirs(os.path.join(path,"images"),exist_ok=True)
        os.makedirs(os.path.join(path,"labels"),exist_ok=True)
        self._load_images()

    def _lbl_path(self, ip):
        if self._project:
            return os.path.join(self._project,"labels",
                                os.path.splitext(os.path.basename(ip))[0]+".txt")
        return os.path.splitext(ip)[0]+".txt"

    def _upload(self):
        if not self._project:
            QMessageBox.warning(self,"No Project","Create or open a project first"); return
        files,_=QFileDialog.getOpenFileNames(self,"Upload Images","",
                                             "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if not files: return
        d=os.path.join(self._project,"images")
        for f in files: shutil.copy(f,os.path.join(d,os.path.basename(f)))
        self._load_images()

    def _open_folder(self):
        d=os.path.join(self._project,"images") if self._project else ""
        if not os.path.isdir(d): return
        if   platform.system()=="Windows": os.startfile(d)
        elif platform.system()=="Darwin":  __import__("subprocess").Popen(["open",d])
        else:                              __import__("subprocess").Popen(["xdg-open",d])

    def _load_images(self):
        if not self._project: return
        self._loading = True
        try:
            img_d=os.path.join(self._project,"images")
            lbl_d=os.path.join(self._project,"labels")
            exts=('.jpg','.jpeg','.png','.bmp','.tif','.tiff')
            imgs=sorted([os.path.join(img_d,f) for f in os.listdir(img_d)
                         if f.lower().endswith(exts)])
            self._images=imgs; self._label_cache={}; self._px_cache={}
            self._thumbs.setUpdatesEnabled(False); self._thumbs.clear()
            for ip in self._images:
                bn=os.path.splitext(os.path.basename(ip))[0]
                has=os.path.exists(os.path.join(lbl_d,bn+".txt"))
                self._label_cache[ip]=has
                item=QListWidgetItem(f"{'✓' if has else '·'} {os.path.basename(ip)[:18]}")
                item.setForeground(QColor(GREEN if has else AMBER)); self._thumbs.addItem(item)
            self._thumbs.setUpdatesEnabled(True)
            if self._cur>=len(self._images): self._cur=max(0,len(self._images)-1)
        finally:
            self._loading = False
        if self._images: self._load_cur()
        self._update_stats()
        if self._thumb_thread and self._thumb_thread.isRunning(): self._thumb_thread.abort()
        self._thumb_thread=ThumbThread(self._images,88)
        self._thumb_thread.thumb_done.connect(self._on_thumb); self._thumb_thread.start()

    def _load_images_keep_cur(self):
        """Like _load_images but preserves self._cur (used after delete).
        The _loading guard is held for the ENTIRE rebuild so that any
        currentRowChanged signals fired by clear()/addItem() cannot
        trigger _on_select and jump to the wrong image."""
        if not self._project: return
        self._loading = True          # block _on_select for the whole rebuild
        try:
            img_d=os.path.join(self._project,"images")
            lbl_d=os.path.join(self._project,"labels")
            exts=('.jpg','.jpeg','.png','.bmp','.tif','.tiff')
            imgs=sorted([os.path.join(img_d,f) for f in os.listdir(img_d)
                         if f.lower().endswith(exts)])
            self._images=imgs; self._label_cache={}; self._px_cache={}
            # Clamp cursor to the intended position (set by caller before us)
            if self._cur>=len(self._images): self._cur=max(0,len(self._images)-1)
            self._thumbs.setUpdatesEnabled(False); self._thumbs.clear()
            for ip in self._images:
                bn=os.path.splitext(os.path.basename(ip))[0]
                has=os.path.exists(os.path.join(lbl_d,bn+".txt"))
                self._label_cache[ip]=has
                tick='\u2713' if has else '\u00b7'
                item=QListWidgetItem(f"{tick} {os.path.basename(ip)[:18]}")
                item.setForeground(QColor(GREEN if has else AMBER)); self._thumbs.addItem(item)
            self._thumbs.setUpdatesEnabled(True)
        finally:
            self._loading = False     # restore before _load_cur (which sets it again internally)

        if self._images:
            self._load_cur()          # correctly loads self._cur (the intended next image)
        else:
            self._fname_edit.setText("No image loaded"); self._fsize_lbl.setText("")
            self._nav.setText("0/0"); self._refresh_box_list()
        self._update_stats()
        # Fresh thumb thread — indices now match the rebuilt list exactly
        self._thumb_thread=ThumbThread(self._images,88)
        self._thumb_thread.thumb_done.connect(self._on_thumb); self._thumb_thread.start()

    @Slot(int,bytes,int,int)
    def _on_thumb(self,idx,raw,w,h):
        self._thumb_buf.append((idx,raw,w,h))
        if not self._thumb_flush_pending:
            self._thumb_flush_pending=True; QTimer.singleShot(150,self._flush_thumbs)

    def _flush_thumbs(self):
        self._thumb_flush_pending=False
        if not self.isVisible() or not self._thumb_buf: self._thumb_buf=[]; return
        self._thumbs.setUpdatesEnabled(False)
        for idx,raw,w,h in self._thumb_buf:
            item=self._thumbs.item(idx)
            if item:
                qimg=QImage(raw,w,h,w*3,QImage.Format.Format_RGB888).copy()
                item.setIcon(QIcon(QPixmap.fromImage(qimg))); item.setText("")
        self._thumb_buf=[]; self._thumbs.setUpdatesEnabled(True)
        self._thumbs.viewport().update()

    def _load_cur(self):
        if not self._images or self._cur>=len(self._images): return
        ip=self._images[self._cur]
        self._loading=True   # block _on_select from re-entering
        try:
            # Update filename bar
            fname=os.path.basename(ip)
            self._fname_edit.setText(fname)
            try:
                sz=os.path.getsize(ip)
                self._fsize_lbl.setText(f"{sz//1024} KB  {self._cur+1}/{len(self._images)}")
            except: self._fsize_lbl.setText("")
            if ip in self._px_cache:
                c=self._canvas; c._px=self._px_cache[ip]; c._scaled=None
                c._ow=c._px.width(); c._oh=c._px.height(); c._boxes=[]; c._fit()
            else: self._canvas.load_image(ip)
            self._canvas.load_labels(self._lbl_path(ip))
            self._nav.setText(f"{self._cur+1}/{len(self._images)}")
            self._thumbs.setCurrentRow(self._cur)
            self._refresh_box_list()
        finally:
            self._loading=False
        QTimer.singleShot(50,self._preload_adj)

    def _preload_adj(self):
        if not self._images: return
        idxs=list(range(max(0,self._cur-1),min(len(self._images),self._cur+3)))
        needed={self._images[i] for i in idxs}
        for k in list(self._px_cache):
            if k not in needed: del self._px_cache[k]
        for i in idxs:
            path=self._images[i]
            if path not in self._px_cache:
                px=QPixmap(); px.load(path)
                if not px.isNull(): self._px_cache[path]=px

    def _save_cur(self):
        self._save_timer.stop(); self._save_cur_silent()

    def _clear_lbl(self):
        if not self._images: return
        if QMessageBox.question(self,"Clear","Clear all boxes for this image?",
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No
        )==QMessageBox.StandardButton.No: return
        ip=self._images[self._cur]; lp=self._lbl_path(ip)
        if os.path.exists(lp): os.remove(lp)
        self._canvas.clear_labels(); self._label_cache[ip]=False
        item=self._thumbs.item(self._cur)
        if item:
            item.setText(f"· {os.path.basename(ip)[:18]}")
            item.setForeground(QColor(AMBER))
        self._refresh_box_list(); self._update_stats()

    def _next(self):
        self._save_cur()
        if self._images: self._cur=min(self._cur+1,len(self._images)-1)
        self._load_cur(); self._update_stats()

    def _prev(self):
        self._save_cur()
        if self._images: self._cur=max(0,self._cur-1)
        self._load_cur()

    def _on_select(self,idx):
        # Ignore programmatic setCurrentRow calls made inside _load_cur itself
        if self._loading: return
        if 0<=idx<len(self._images) and idx!=self._cur:
            self._save_cur(); self._cur=idx; self._load_cur()

    def _refresh_box_list(self):
        self._box_list.clear()
        for b in self._canvas.get_boxes():
            cls=int(b[0])
            name=MASTER_CLASS_LIST[cls] if cls<len(MASTER_CLASS_LIST) else str(cls)
            item=QListWidgetItem(f"[{cls}] {name[:22]}")
            item.setForeground(QColor(BOX_COLORS.get(cls,DEF_COL)))
            self._box_list.addItem(item)
        self._update_stats()

    def _del_selected_box(self):
        sel=self._canvas._sel_box
        if 0<=sel<len(self._canvas._boxes):
            self._canvas._boxes.pop(sel); self._canvas._sel_box=-1
            self._canvas.update(); self._canvas.box_changed.emit()

    def _update_stats(self):
        total=len(self._images)
        labeled=sum(1 for v in self._label_cache.values() if v)
        boxes=len(self._canvas.get_boxes()) if self._images else 0
        self._stats.setText(
            f"{total} imgs  {labeled} labeled  {total-labeled} todo  |  {boxes} boxes")

    # ── Auto-label ────────────────────────────────────────────────────────────
    def _auto_label(self):
        if not self._project:
            QMessageBox.warning(self,"No Project","Create or open a project first"); return
        tp=self._teacher.text().strip()
        if not tp or not os.path.exists(tp):
            tp=OG_MODEL if os.path.exists(OG_MODEL) else ""
            if tp: self._teacher.setText(tp)
        if not tp or not os.path.exists(tp):
            QMessageBox.warning(self,"Missing","Select a valid teacher model (.pt)"); return
        self._al_log.clear(); self._al_prog.setValue(0)
        if self._cached_model_path!=tp:
            if not HAS_YOLO:
                QMessageBox.warning(self,"Missing","ultralytics not installed"); return
            try: self._cached_model=_YOLO(tp); self._cached_model_path=tp
            except Exception as e: QMessageBox.critical(self,"Error",str(e)); return
        inbox=os.path.join(self._project,"images")
        out=os.path.join(self._project,"labels")
        use_sahi = self._sahi_chk.isChecked()
        tile_sz  = self._sahi_tile.currentData()
        overlap  = self._sahi_overlap.value()
        iou_thr  = self._sahi_iou.value()
        self._auto_thread=AutoLabelThread(
            self._cached_model, inbox, out, self._auto_conf.value(),
            use_sahi=use_sahi, tile_sz=tile_sz,
            overlap=overlap, iou_thr=iou_thr)
        self._auto_thread.progress.connect(
            lambda p,m:(self._al_prog.setValue(p),self._al_log.append(m)))
        self._auto_thread.finished.connect(
            lambda c:(self._al_log.append(f"✓ Done — {c} labeled"),
                      self._al_prog.setValue(100),self._load_images()))
        self._auto_thread.start()
        mode_str = f"SAHI tile={tile_sz} overlap={int(overlap*100)}%" if use_sahi else "full-image"
        self._al_log.append(f"Running {os.path.basename(tp)} [{mode_str}] …")
        if use_sahi and not (HAS_CV2 and HAS_NP):
            self._al_log.append("⚠ cv2/numpy not found — SAHI falling back to full-image")

    # ── Keyboard ──────────────────────────────────────────────────────────────
    def keyPressEvent(self,e):
        k=e.key()
        if k in (Qt.Key.Key_Right,Qt.Key.Key_D): self._next()
        elif k in (Qt.Key.Key_Left,Qt.Key.Key_A): self._prev()
        elif k==Qt.Key.Key_S and e.modifiers()&Qt.KeyboardModifier.ControlModifier:
            self._save_cur()
        elif k in (Qt.Key.Key_Delete,Qt.Key.Key_Backspace):
            self._del_selected_box()
        else: super().keyPressEvent(e)

    def closeEvent(self,e):
        self._save_cur()
        if self._thumb_thread and self._thumb_thread.isRunning():
            self._thumb_thread.abort()
        e.accept()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    app=QApplication(sys.argv)
    app.setApplicationName("MELSS Labeler"); app.setOrganizationName("MELSS")
    app.setStyle("Fusion")
    f=QFont()
    for family in ["Segoe UI","SF Pro Display","Helvetica Neue","Arial"]:
        f.setFamily(family); f.setPointSize(10)
        if f.exactMatch(): break
    app.setFont(f)
    p=QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(BG_DEEP))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT_PRI))
    p.setColor(QPalette.ColorRole.Base,            QColor(BG_CARD))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(BG_PANEL))
    p.setColor(QPalette.ColorRole.Text,            QColor(TEXT_PRI))
    p.setColor(QPalette.ColorRole.Button,          QColor(BG_CARD2))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT_PRI))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(CYAN_DIM))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXT_PRI))
    app.setPalette(p); app.setStyleSheet(GLOBAL_QSS)
    win=LabelerWindow(); win.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()