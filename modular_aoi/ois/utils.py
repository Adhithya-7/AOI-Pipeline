"""
utils.py — Data, configuration, project management, and pure computation.

No Qt widgets — only data classes, algorithms, and filesystem helpers.
(Exception: SettingsDialog and HistoryDB are here because they are data-centric.)
"""
import sys, os, json, shutil, random, glob, time, platform, threading, sqlite3
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4" # Conservative limit to avoid saturation
os.environ["MKL_DOMAIN_NUM_THREADS"] = "4"
import math, copy
from datetime import datetime
from collections import deque
from dataclasses import dataclass

if __package__ in (None, ""):
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    __package__ = "ois"

# ── Optional dependency flags ────────────────────────────────────────────────
# Imported once here; every other module imports these from utils.
try:    import cv2;                            HAS_CV2  = True
except: HAS_CV2  = False
try:    import numpy as np;                    HAS_NP   = True
except: HAS_NP   = False; np = None
try:    from PIL import Image as PILImage;     HAS_PIL  = True
except: HAS_PIL  = False
try:    from ultralytics import YOLO as _YOLO; HAS_YOLO = True
except: HAS_YOLO = False

try:
    from sahi import AutoDetectionModel as _SAHIModel
    from sahi.predict import get_sliced_prediction as _sahi_predict
    HAS_SAHI = True
except ImportError:
    HAS_SAHI = False
_SAHI_MODEL_CACHE: dict = {}   # (model_path, conf) → SAHIModel


# ── Portable data directory ──────────────────────────────────────────────────
try:
    from platformdirs import user_data_dir as _udd
    DATA_ROOT = _udd("OIS", "MELSS")
except ImportError:
    DATA_ROOT = os.path.join(os.path.expanduser("~"), ".ois_data")
os.makedirs(DATA_ROOT, exist_ok=True)

# ── PyInstaller-aware resource resolver ──────────────────────────────────────
def _res(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)

# ── Camera backend — OS-specific ─────────────────────────────────────────────
def _cam_backend() -> int:
    if not HAS_CV2:
        return 0
    s = platform.system()
    if s == "Windows":  return cv2.CAP_DSHOW
    if s == "Darwin":   return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2   # Linux

# ── Inference Constants ──────────────────────────────────────────────────────
_AOI_SAHI_TRIGGER = 1920
_AOI_SLICE_HW     = 640
_AOI_OVERLAP      = 0.20
_AOI_MATCH_R      = 0.06

# ── Component dataclass ──────────────────────────────────────────────────────
@dataclass
class _AOIComp:
    id:    int
    label: str
    conf:  float
    cx:    float
    cy:    float
    w:     float
    h:     float

    def xyxy(self, iw:int, ih:int):
        return (max(0,int((self.cx-self.w/2)*iw)),
                max(0,int((self.cy-self.h/2)*ih)),
                min(iw,int((self.cx+self.w/2)*iw)),
                min(ih,int((self.cy+self.h/2)*ih)))

    def centre_px(self, iw:int, ih:int):
        return int(self.cx*iw), int(self.cy*ih)

import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "best.pt"

OG_MODEL = str(DEFAULT_MODEL)

MASTER_CLASS_LIST=[
    "Connector (P)","Resistor (R)","Transformer (T)","Diode (D)","Capacitor (C)",
    "Transistor (Q)","Jumper (J)","Inductor (L)","IC (U)","Resistor Array (RA)",
    "Resistor Net (RN)","Crystal (CR)","IC (IC)","Jumper (JP)","Varistor (V)",
    "Button (BTN)","Switch (SW)","Switch (S)","Test Point (TP)","LED",
    "Transistor (QA)","Cap Array (CRA)","Motor (M)","Fuse (F)","Ferrite Bead (FB)"
]
DEFAULT_AUG={"mosaic":1.0,"mixup":0.15,"copy_paste":0.3,"hsv_h":0.015,"hsv_s":0.6,
    "hsv_v":0.4,"degrees":12.0,"translate":0.15,"scale":0.6,"shear":2.0,
    "perspective":0.0005,"erasing":0.3,"epochs":50,"batch":4,"imgsz":640,"patience":25,"workers":1}


# ── Utility functions ────────────────────────────────────────────────────────

def load_optimized_yolo(path, task="detect"):
    """
    Strictly load the specified model path. No auto-redirection to OpenVINO.
    """
    if not HAS_YOLO or not path: return None
    
    def _log(msg):
        print(f"[ModelLoader] {msg}")

    # Resolve relative paths (relative to app root or project dir)
    if not os.path.isabs(path):
        # We try to find it relative to current working directory first
        pass

    if os.path.exists(path):
        try:
            m = _YOLO(path, task=task)
            _log(f"LOADED MODEL: {path}")
            return m
        except Exception as e:
            _log(f"Model load failed: {e}")

    # Last resort fallback for absolute paths that might not be detected by exists() 
    # (sometimes happens with network drives or complex paths)
    try:
        return _YOLO(path, task=task)
    except:
        pass

    _log(f"CRITICAL: Could not load model at {path}")
    return None


def safe_predict(model, *args, **kwargs):
    """
    Run model.predict() safely for both standard and optimized models.
    Caches the format check to minimize per-frame logic overhead.
    """
    if not model: return []
    
    # Fast check via cached property
    is_exported = getattr(model, "_aoi_is_opt", None)
    if is_exported is None:
        _name = str(getattr(model, 'model_name', getattr(model, 'ckpt_path', '')))
        is_exported = hasattr(model, 'model') and not _name.endswith('.pt')
        model._aoi_is_opt = is_exported
    
    # OpenVINO / ONNX models handle device internally; passing 'device' often crashes.
    if is_exported and 'device' in kwargs:
        del kwargs['device']
        
    if "imgsz" not in kwargs: kwargs["imgsz"] = 640
    
    try:
        return model.predict(*args, **kwargs)
    except Exception as e:
        if "device" in str(e).lower() and 'device' in kwargs:
            del kwargs['device']
            return model.predict(*args, **kwargs)
        raise e

# ── Custom SAHI wrapper for OpenVINO/ONNX ─────────────────────────────────────
# Standard SAHI UltralyticsDetectionModel calls .to(device), which crashes exports.
if HAS_SAHI:
    from sahi.models.ultralytics import UltralyticsDetectionModel
    class _AOISahiModel(UltralyticsDetectionModel):
        def load_model(self):
            # Use our optimized loader instead of raw YOLO class
            self.model = load_optimized_yolo(self.model_path, task=self.task)
            # Skip the .to(device) call that causes crashes on exported formats
            is_exported = self.model_path.endswith("_openvino_model") or self.model_path.endswith(".onnx")
            if self.device and not is_exported:
                try: self.model.to(self.device)
                except: pass

def _aoi_get_sahi_model(model_path: str, conf: float):
    if not HAS_SAHI: return None
    key = (model_path, round(conf,4))
    if key not in _SAHI_MODEL_CACHE:
        try:
            # Use our custom wrapper instead of AutoDetectionModel.from_pretrained
            _SAHI_MODEL_CACHE[key] = _AOISahiModel(
                model_path=model_path,
                confidence_threshold=conf,
                device="cpu" # The wrapper now safely ignores this for exports
            )
        except:
            return None
    return _SAHI_MODEL_CACHE[key]

def _aoi_bbox_overlap(a, b):
    ax1,ay1 = a.cx-a.w/2, a.cy-a.h/2
    ax2,ay2 = a.cx+a.w/2, a.cy+a.h/2
    bx1,by1 = b.cx-b.w/2, b.cy-b.h/2
    bx2,by2 = b.cx+b.w/2, b.cy+b.h/2
    ix = max(0.0, min(ax2,bx2) - max(ax1,bx1))
    iy = max(0.0, min(ay2,by2) - max(ay1,by1))
    inter = ix * iy
    aa, ab = a.w*a.h, b.w*b.h
    if inter == 0:
        iou = iomin = 0.0
    else:
        iou   = inter / max(aa + ab - inter, 1e-9)
        iomin = inter / max(min(aa, ab),      1e-9)
    a_has_b = (ax1 <= b.cx <= ax2 and ay1 <= b.cy <= ay2)
    b_has_a = (bx1 <= a.cx <= bx2 and by1 <= a.cy <= by2)
    return iou, iomin, a_has_b, b_has_a

def _aoi_nms_by_centre(comps: list,
                        iou_thr:   float = 0.30,
                        iomin_thr: float = 0.70,
                        dist_thr:  float = 0.025) -> list:
    if len(comps) <= 1:
        return comps
    ordered = sorted(comps, key=lambda c: -c.conf)
    suppressed = set(); keep = []; SIZE_GUARD = 4.0
    for i, a in enumerate(ordered):
        if i in suppressed: continue
        keep.append(a); area_a = a.w * a.h
        for j in range(i+1, len(ordered)):
            if j in suppressed: continue
            b = ordered[j]; area_b = b.w * b.h
            iou, iomin, a_has_b, _ = _aoi_bbox_overlap(a, b)
            if iou > iou_thr:
                suppressed.add(j); continue
            if iomin > iomin_thr and area_a >= SIZE_GUARD * area_b:
                suppressed.add(j); continue
            if a_has_b and area_a >= SIZE_GUARD * area_b:
                suppressed.add(j); continue
            if b.label.lower() == a.label.lower():
                if math.hypot(a.cx-b.cx, a.cy-b.cy) < dist_thr:
                    suppressed.add(j)
    for uid, c in enumerate(keep): c.id = uid
    return keep

def _aoi_infer_arr(model, img_bgr, conf: float,
                   model_path: str = "", use_sahi: bool = False,
                   _log=None) -> list:
    if not HAS_CV2 or not HAS_NP or not HAS_YOLO: return []
    H, W = img_bgr.shape[:2]; img_max = max(H, W)
    if use_sahi and HAS_SAHI and model_path and img_max > _AOI_SAHI_TRIGGER:
        if img_max > 4000: tile_hw = 1280
        elif img_max > 2000: tile_hw = 960
        else: tile_hw = _AOI_SLICE_HW
        n_tiles_est = ((img_max // int(tile_hw*(1-_AOI_OVERLAP))) + 1) ** 2
        t0 = time.time()
        if _log: _log(f"[SAHI] {img_max}px → tile={tile_hw}px  ~{n_tiles_est} tiles")
        try:
            from sahi.predict import get_sliced_prediction
            from PIL import Image as _PIL
            det_m = _aoi_get_sahi_model(model_path, conf)
            if det_m is None: raise RuntimeError("SAHI model unavailable")
            pil = _PIL.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            result = get_sliced_prediction(
                image=pil, detection_model=det_m,
                slice_height=tile_hw, slice_width=tile_hw,
                overlap_height_ratio=_AOI_OVERLAP, overlap_width_ratio=_AOI_OVERLAP,
                postprocess_match_threshold=0.20, verbose=0)
            comps = []
            for uid, obj in enumerate(result.object_prediction_list):
                x1,y1,x2,y2 = obj.bbox.minx,obj.bbox.miny,obj.bbox.maxx,obj.bbox.maxy
                comps.append(_AOIComp(uid, obj.category.name, float(obj.score.value),
                                      ((x1+x2)/2)/W, ((y1+y2)/2)/H, (x2-x1)/W, (y2-y1)/H))
            before = len(comps)
            if len(comps) > 1:
                _ord = sorted(range(len(comps)), key=lambda i: -comps[i].conf)
                _keep, _supp = [], set()
                for _i in _ord:
                    if _i in _supp: continue
                    _keep.append(_i); _a = comps[_i]
                    _ax1,_ay1 = _a.cx-_a.w/2, _a.cy-_a.h/2
                    _ax2,_ay2 = _a.cx+_a.w/2, _a.cy+_a.h/2
                    for _j in _ord:
                        if _j in _supp or _j == _i: continue
                        _b = comps[_j]
                        _ix = max(0, min(_ax2,_b.cx+_b.w/2) - max(_ax1,_b.cx-_b.w/2))
                        _iy = max(0, min(_ay2,_b.cy+_b.h/2) - max(_ay1,_b.cy-_b.h/2))
                        _inter = _ix * _iy; _union = _a.w*_a.h + _b.w*_b.h - _inter
                        if _union > 0 and _inter/_union > 0.40: _supp.add(_j)
                comps = [comps[_i] for _i in _keep]
                for _uid, _c in enumerate(comps): _c.id = _uid
            if _log: _log(f"[SAHI] {before} raw → {len(comps)} after NMS in {time.time()-t0:.1f}s")
            return comps
        except Exception as _e:
            if _log: _log(f"[SAHI] FAILED ({_e}) → falling back to standard YOLO")
    t0 = time.time()
    try:
        results = safe_predict(model, img_bgr, conf=conf, iou=0.35, imgsz=640, verbose=False, device='cpu')
        
        # Robustness check: if standard YOLO returns 0 detections on a huge image,
        # it might be a stale OpenVINO runtime state. Try one fresh reload.
        # ONLY DO THIS FOR LARGE IMAGES (> 4000px) where dets are expected.
        is_exported = hasattr(model, 'model') and not str(getattr(model, 'model_name', getattr(model, 'ckpt_path', ''))).endswith('.pt')
        if len(results) > 0 and len(results[0].boxes) == 0 and is_exported and img_max >= 4000:
            if _log: _log("[YOLO] 0 dets detected on large image — attempting OpenVINO state refresh...")
            try:
                # Re-load model briefly for this one call
                fresh_m = _YOLO(model.ckpt_path, task="detect")
                results = safe_predict(fresh_m, img_bgr, conf=conf, iou=0.35, imgsz=640, verbose=False)
            except:
                pass

        comps, uid = [], 0
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                cx,cy,bw,bh = box.xywhn[0].tolist()
                label = r.names[int(box.cls[0])]
                comps.append(_AOIComp(uid,label,float(box.conf[0]),cx,cy,bw,bh))
                uid += 1
        if len(comps) > 1 and HAS_NP:
            _ord = sorted(range(len(comps)), key=lambda i: -comps[i].conf)
            _keep, _supp = [], set()
            for _i in _ord:
                if _i in _supp: continue
                _keep.append(_i); _a = comps[_i]
                _ax1,_ay1 = _a.cx-_a.w/2, _a.cy-_a.h/2
                _ax2,_ay2 = _a.cx+_a.w/2, _a.cy+_a.h/2
                for _j in _ord:
                    if _j in _supp or _j == _i: continue
                    _b = comps[_j]
                    _ix = max(0, min(_ax2,_b.cx+_b.w/2) - max(_ax1,_b.cx-_b.w/2))
                    _iy = max(0, min(_ay2,_b.cy+_b.h/2) - max(_ay1,_b.cy-_b.h/2))
                    _inter = _ix * _iy; _union = _a.w*_a.h + _b.w*_b.h - _inter
                    if _union > 0 and _inter/_union > 0.40: _supp.add(_j)
            comps = [comps[_i] for _i in _keep]
            for _uid, _c in enumerate(comps): _c.id = _uid
        if _log: _log(f"[YOLO] {len(comps)} dets in {time.time()-t0:.2f}s ({W}×{H})")
        return comps
    except Exception as _e:
        if _log: _log(f"[YOLO] inference failed: {_e}")
        return []

def calc_iou(a,b):
    xA,yA=max(a[0],b[0]),max(a[1],b[1]); xB,yB=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    if inter==0: return 0.0
    areaA=max(0,(a[2]-a[0]))*max(0,(a[3]-a[1]))
    areaB=max(0,(b[2]-b[0]))*max(0,(b[3]-b[1]))
    union=areaA+areaB-inter
    return inter/float(union) if union>0 else 0.0

def calc_center_dist_norm(a,b):
    """Euclidean center-distance normalised by the golden box diagonal.
    Returns 0.0 when centres coincide, >1.0 when very far apart."""
    cax,cay=(a[0]+a[2])/2,(a[1]+a[3])/2
    cbx,cby=(b[0]+b[2])/2,(b[1]+b[3])/2
    diag=max(1,((a[2]-a[0])**2+(a[3]-a[1])**2)**0.5)
    return ((cax-cbx)**2+(cay-cby)**2)**0.5/diag

def _classes_match(g,d):
    """True when the two detections represent the same component class.

    ROOT-CAUSE FIX (v7.1):
      Integer class-IDs are assigned by YOLO at training time and can be
      completely reshuffled after any retrain (e.g. Resistor was class-id 1,
      becomes class-id 4 in the new model).  The golden master is saved once
      and reused across multiple retrains, so comparing IDs first was causing
      ID=1 (old: Resistor) to match ID=1 (new: Capacitor) → genuinely missing
      components were marked "found" by the wrong-class detection sitting
      nearby, making them silently disappear from the missing list.

    Correct priority (name FIRST, ID only as last-resort fallback):
      1. Name strings match (case-insensitive, stripped) → True  ← PRIMARY
      2. Integer IDs match (only when BOTH names are absent)     ← FALLBACK
    Names are always stored in the golden master so the fast-ID path is now
    only reached when reading legacy JSON that has no "name" field.
    """
    # PRIMARY — name-string comparison survives retraining / class reordering
    gn = str(g.get("name","")).strip().lower()
    dn = str(d.get("name","")).strip().lower()
    if gn and dn:
        return gn == dn
    # FALLBACK — integer ID (legacy JSON without a "name" field)
    gid = g.get("class"); did = d.get("class")
    if gid is not None and did is not None:
        return int(gid) == int(did)
    return False

def match_detections(golden, live, iou_thresh, persist_cnt, persist_max):
    """
    Optimal class-aware slot assignment (v8 rewrite).

    The previous greedy two-pass design had a cascade problem: whichever golden
    slot was processed first got the best available detection, leaving later
    slots without a match even when the detection was actually more appropriate
    for the later slot.  This produced simultaneous false-found and false-missing
    errors on dense boards.

    New approach — global Hungarian-optimal assignment:
      • Cost matrix: INF for cross-class pairs, (1 − IoU) for same-class pairs.
      • scipy.optimize.linear_sum_assignment finds the globally optimal one-to-one
        assignment (greedy fallback when scipy is unavailable).
      • A slot is "found" when its assigned detection has IoU ≥ iou_thresh.
      • Centre-proximity fallback (< 0.40 × bbox diagonal) runs only on same-class
        slots that failed the IoU gate — preserves the camera-drift guard without
        letting it rescue cross-class assignments.

    Returns a list of golden slot indices that are persistently absent.
    """
    missing = []
    G = len(golden)
    if G == 0:
        return []

    L = len(live)
    # [DEBUG] Log IDs for clarification on "terminal to ids"
    # User inquired about terminal/ids; this log helps confirm what's being matched
    if G > 0 and L > 0:
        _gn = [str(g.get("name","")) or str(g.get("class","?")) for g in golden[:3]]
        _ln = [str(d.get("name","")) or str(d.get("class","?")) for d in live[:3]]
        print(f"[MatchSync] Matching {L} detections against {G} golden slots. Sample names: {', '.join(_gn)} vs {', '.join(_ln)}")

    if L == 0:
        for i in range(G):
            persist_cnt[i] = persist_cnt.get(i, 0) + 1
            if persist_cnt[i] > persist_max:
                missing.append(i)
        return missing

    INF      = 1e9
    iou_gate = 1.0 - iou_thresh   # cost threshold: cost < iou_gate ↔ IoU ≥ iou_thresh

    # ── Build G×L cost matrix ─────────────────────────────────────────────────
    if HAS_NP:
        cost = np.full((G, L), INF, dtype=np.float64)
        for i, g in enumerate(golden):
            for j, d in enumerate(live):
                if not _classes_match(g, d): continue
                cost[i, j] = 1.0 - calc_iou(g["xyxy"], d["xyxy"])
    else:
        cost = [[INF] * L for _ in range(G)]
        for i, g in enumerate(golden):
            for j, d in enumerate(live):
                if not _classes_match(g, d): continue
                cost[i][j] = 1.0 - calc_iou(g["xyxy"], d["xyxy"])

    # ── Optimal assignment ────────────────────────────────────────────────────
    assignment: dict = {}   # golden_idx → live_idx
    try:
        from scipy.optimize import linear_sum_assignment
        _c = cost if HAS_NP else np.array(cost)
        gi, di = linear_sum_assignment(_c)
        for g, d in zip(gi, di):
            if float(_c[g, d]) < iou_gate:
                assignment[g] = d
    except ImportError:
        # Greedy fallback: process pairs sorted by ascending cost
        pairs = []
        for i in range(G):
            for j in range(L):
                c = float(cost[i, j]) if HAS_NP else float(cost[i][j])
                if c < INF:
                    pairs.append((c, i, j))
        pairs.sort()
        used_g: set = set(); used_l: set = set()
        for c, i, j in pairs:
            if i in used_g or j in used_l: continue
            if c < iou_gate:
                assignment[i] = j
                used_g.add(i); used_l.add(j)

    # ── Centre-proximity fallback for still-unmatched same-class slots ────────
    assigned_live = set(assignment.values())
    for i, g in enumerate(golden):
        if i in assignment: continue
        for j, d in enumerate(live):
            if j in assigned_live: continue
            if not _classes_match(g, d): continue
            if calc_center_dist_norm(g["xyxy"], d["xyxy"]) < 0.40:
                assignment[i] = j
                assigned_live.add(j)
                break

    # ── Update persistence counters and collect missing indices ───────────────
    for i in range(G):
        if i in assignment:
            persist_cnt[i] = 0
        else:
            persist_cnt[i] = persist_cnt.get(i, 0) + 1
            if persist_cnt[i] > persist_max:
                missing.append(i)

    return missing   # list of golden indices that are persistently absent

def find_best_pt(out_dir,run_name):
    exact=os.path.join(out_dir,run_name,"weights","best.pt")
    if os.path.exists(exact): return exact
    candidates=[]
    for root,dirs,files in os.walk(out_dir):
        if "best.pt" in files:
            p=os.path.join(root,"best.pt"); candidates.append((os.path.getmtime(p),p))
    return sorted(candidates)[-1][1] if candidates else ""

def build_golden_dataset(project_path,class_list,split=0.8):
    src_img=os.path.join(project_path,"golden_boards"); src_lbl=os.path.join(project_path,"labels")
    aug_img=os.path.join(project_path,"augmented_images"); aug_lbl=os.path.join(project_path,"augmented_labels")
    out=os.path.join(project_path,"YOLO_Dataset")
    if os.path.exists(out): shutil.rmtree(out)
    for s in ["train","val"]:
        os.makedirs(os.path.join(out,s,"images"),exist_ok=True)
        os.makedirs(os.path.join(out,s,"labels"),exist_ok=True)
    pairs=[]
    if os.path.exists(src_img):
        for f in os.listdir(src_img):
            if f.lower().endswith((".jpg",".png",".jpeg")):
                lbl=os.path.join(src_lbl,os.path.splitext(f)[0]+".txt")
                if os.path.exists(lbl): pairs.append((os.path.join(src_img,f),lbl))
    aug_pairs=[]
    if os.path.exists(aug_img) and os.path.exists(aug_lbl):
        for f in os.listdir(aug_img):
            if f.lower().endswith((".jpg",".png",".jpeg")):
                lbl=os.path.join(aug_lbl,os.path.splitext(f)[0]+".txt")
                if os.path.exists(lbl): aug_pairs.append((os.path.join(aug_img,f),lbl))
    all_pairs=pairs+aug_pairs
    if not all_pairs: raise ValueError(f"No labeled images. Golden:{len(pairs)} Aug:{len(aug_pairs)}")
    random.shuffle(all_pairs); n=max(1,int(len(all_pairs)*split))
    train_p,val_p=all_pairs[:n],all_pairs[n:]
    for fset,sname in [(train_p,"train"),(val_p,"val")]:
        for ip,lp in fset:
            fn=os.path.basename(ip); shutil.copy(ip,os.path.join(out,sname,"images",fn))
            shutil.copy(lp,os.path.join(out,sname,"labels",os.path.splitext(fn)[0]+".txt"))
    yaml=os.path.join(out,"data.yaml"); abs_root=os.path.abspath(out).replace("\\","/")
    with open(yaml,"w") as f:
        f.write(f"path: {abs_root}\ntrain: train/images\nval: val/images\n")
        f.write(f"nc: {len(class_list)}\nnames: {class_list}\n")
    return yaml,len(train_p),len(val_p)


# ── ProjectConfig ────────────────────────────────────────────────────────────

class ProjectConfig:
    DEFAULTS={"model_path":OG_MODEL if os.path.exists(OG_MODEL) else "","og_model":OG_MODEL if os.path.exists(OG_MODEL) else "",
        "camera_index":0,"confidence":0.25,"autolabel_conf":0.50,"aoi_conf":0.25,"aoi_cal_runs":8,

        "match_iou":0.30,"persistence":5,"max_obj_width":0.30,
        "max_obj_height":0.30,"edge_margin":10,"golden_file":"golden_master.json",
        "autosave_flags":True,"log_history":True,"save_annot_history":True,
        "last_project":"","aug_params":DEFAULT_AUG,"trained_models":[],
        "pcb_green_desat":0.70,"pcb_comp_boost":1.30}
    GLOBAL_CFG=os.path.join(DATA_ROOT,"global_config.json")
    def __init__(self): self._path=None; self._d=dict(self.DEFAULTS); self._dirty=False; self._load_global()
    def _load_global(self):
        if os.path.exists(self.GLOBAL_CFG):
            try:
                with open(self.GLOBAL_CFG) as f: g=json.load(f); self._d["last_project"]=g.get("last_project","")
            except: pass
    def _save_global(self):
        os.makedirs(DATA_ROOT,exist_ok=True)
        try:
            with open(self.GLOBAL_CFG,"w") as f: json.dump({"last_project":self._path or ""},f)
        except: pass
    def load(self,p):
        self._path=p; fp=os.path.join(p,"config.json"); self._d=dict(self.DEFAULTS)
        if os.path.exists(fp):
            try:
                with open(fp) as f: self._d.update(json.load(f))
            except: pass
        self._dirty=False; self._save_global()
    def save(self):
        if self._path:
            try:
                with open(os.path.join(self._path,"config.json"),"w") as f: json.dump(self._d,f,indent=2)
            except: pass
    def flush(self):
        if self._dirty: self.save(); self._dirty=False
    def get(self,k,default=None):
        val = self._d.get(k, self.DEFAULTS.get(k, default))
        # Definitive Purge: If the value is an OpenVINO directory, find the OG model instead
        if k == "model_path" and isinstance(val, str) and val.endswith("_openvino_model"):
            pt = val.replace("_openvino_model", ".pt")
            if os.path.exists(pt): return pt
        return val


    def set(self,k,v):
        if self._d.get(k)==v: return
        self._d[k]=v; self._dirty=True
    def get_last_project(self): return self._d.get("last_project","")
    def record_trained_model(self,path):
        models=self.get("trained_models") or []
        if path not in models: models.append(path)
        self.set("trained_models",models)
    def get_next_base_model(self):
        for m in reversed(self.get("trained_models") or []):
            if os.path.exists(m): return m
        og=self.get("og_model") or ""
        return og if og and os.path.exists(og) else ""



# ── HistoryDB ────────────────────────────────────────────────────────────────

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


# ── SettingsDialog ───────────────────────────────────────────────────────────

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QDoubleSpinBox, QSpinBox, QCheckBox,
    QLineEdit, QFileDialog, QDialogButtonBox
)
from PySide6.QtCore import Signal

from .theme import (
    BG_PANEL, BG_BORDER, BG_BORDER2, CYAN, TEXT_SEC, TEXT_DIM,
    sec_lbl, make_sep, make_card
)

class SettingsDialog(QDialog):
    settings_changed = Signal()

    def __init__(self, cfg: "ProjectConfig", parent=None):
        super().__init__(parent); self._cfg = cfg
        self.setWindowTitle("Settings"); self.setMinimumSize(540, 640)
        self.setStyleSheet(f"QDialog{{background:{BG_PANEL};}}")
        lay = QVBoxLayout(self); lay.setContentsMargins(20,16,20,16); lay.setSpacing(12)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;}")
        inner = QWidget(); form = QVBoxLayout(inner)
        form.setSpacing(8); form.setContentsMargins(0,0,8,0)

        # ── helper builders ───────────────────────────────────────────────
        def dbl_row(label, key, lo, hi, dec, tip=""):
            r = QHBoxLayout(); lw = QLabel(label); lw.setMinimumWidth(220)
            r.addWidget(lw)
            w = QDoubleSpinBox(); w.setRange(lo,hi); w.setDecimals(dec)
            w.setSingleStep(0.05); w.setValue(float(cfg.get(key, lo)))
            w.setFixedWidth(90)
            if tip: w.setToolTip(tip)
            r.addWidget(w); r.addStretch(); form.addLayout(r)
            return w

        def int_row(label, key, lo, hi, tip=""):
            r = QHBoxLayout(); lw = QLabel(label); lw.setMinimumWidth(220)
            r.addWidget(lw)
            w = QSpinBox(); w.setRange(lo,hi)
            w.setValue(int(cfg.get(key, lo))); w.setFixedWidth(70)
            if tip: w.setToolTip(tip)
            r.addWidget(w); r.addStretch(); form.addLayout(r)
            return w

        def chk_row(label, key, default=True, tip=""):
            cb = QCheckBox(label); cb.setChecked(bool(cfg.get(key, default)))
            if tip: cb.setToolTip(tip)
            form.addWidget(cb); return cb

        # ── CAMERA ────────────────────────────────────────────────────────
        form.addWidget(sec_lbl("CAMERA"))
        self._cam_idx = int_row("Camera index:", "camera_index", 0, 9,
                                "Index of the capture device (0 = default)")

        # ── CONFIDENCE ───────────────────────────────────────────────────
        form.addWidget(make_sep()); form.addWidget(sec_lbl("CONFIDENCE THRESHOLDS"))
        self._conf        = dbl_row("Run / Inspect confidence:",   "confidence",    0.05, 0.99, 2,
                                    "Min YOLO score for live RUN and INSPECT NOW detection")
        self._autolbl_conf= dbl_row("Auto-Label confidence:",      "autolabel_conf",0.05, 0.99, 2,
                                    "Confidence used when Auto-Labelling boards in the TRAIN tab")
        self._aoi_conf    = dbl_row("Offline AOI confidence:",     "aoi_conf",      0.05, 0.99, 2,
                                    "Confidence used when running batch Offline AOI")

        # ── LIVE INSPECTION ───────────────────────────────────────────────
        form.addWidget(make_sep()); form.addWidget(sec_lbl("LIVE INSPECTION"))
        self._iou         = dbl_row("Golden match IoU threshold:",  "match_iou",     0.05, 0.99, 2,
                                    "Min overlap to match a detection to a golden component slot")
        self._persist     = int_row("Persistence frames:",          "persistence",   1, 60,
                                    "Frames a component must be absent before flagging MISSING")
        self._max_w       = dbl_row("Max object width (fraction):", "max_obj_width", 0.05, 1.0, 2,
                                    "Detections wider than this fraction of the frame are ignored")
        self._max_h       = dbl_row("Max object height (fraction):","max_obj_height",0.05, 1.0, 2,
                                    "Detections taller than this fraction of the frame are ignored")
        self._edge        = int_row("Edge margin (px):",            "edge_margin",   0, 100,
                                    "Detections this close to the frame edge are ignored")

        # ── OFFLINE AOI ───────────────────────────────────────────────────
        form.addWidget(make_sep()); form.addWidget(sec_lbl("OFFLINE AOI"))
        self._aoi_cal     = int_row("Calibration runs:",            "aoi_cal_runs",  4, 20,
                                    "Number of inference passes used during AOI calibration (higher = more stable)")

        # ── MODEL ─────────────────────────────────────────────────────────
        form.addWidget(make_sep()); form.addWidget(sec_lbl("MODEL"))
        mr = QHBoxLayout(); mr.addWidget(QLabel("Active model:"))
        self._model_le = QLineEdit(cfg.get("model_path",""))
        self._model_le.setReadOnly(True)
        self._model_le.setStyleSheet(f"color:{TEXT_SEC};font-family:Consolas;font-size:10px;")
        b_mbr = QPushButton("Browse…"); b_mbr.setFixedHeight(26)
        def _pick():
            p,_ = QFileDialog.getOpenFileName(self,"Select Model","","PyTorch (*.pt)")
            if p: self._model_le.setText(p)
        b_mbr.clicked.connect(_pick)
        mr.addWidget(self._model_le,1); mr.addWidget(b_mbr); form.addLayout(mr)

        # ── OUTPUT / HISTORY ──────────────────────────────────────────────
        form.addWidget(make_sep()); form.addWidget(sec_lbl("OUTPUT / HISTORY"))
        self._autosave_cb   = chk_row("Auto-save flagged board images",
                                      "autosave_flags", True,
                                      "Save images captured with the FLAG button to golden_boards/")
        self._log_history_cb= chk_row("Log results to history database",
                                      "log_history", True,
                                      "Write pass/fail records to the HISTORY tab database")
        self._save_annot_cb = chk_row("Save annotated history images (side-by-side)",
                                      "save_annot_history", True,
                                      "After each inspection/AOI run, save a golden-vs-scan comparison image to project/history/")

        # ── PCB COLOR NORM ─────────────────────────────────────────────────
        form.addWidget(make_sep()); form.addWidget(sec_lbl("PCB COLOR NORM"))
        _pcb_note = QLabel(
            "Default parameters used when the PCB Color Norm filter is added to a pipeline.\n"
            "Green Desat suppresses the FR-4 substrate; Component Boost lifts component colours."
        )
        _pcb_note.setWordWrap(True)
        _pcb_note.setStyleSheet(f"color:{TEXT_SEC};font-size:10px;border:none;")
        form.addWidget(_pcb_note)
        self._pcb_desat  = dbl_row("Green Desaturation strength:",  "pcb_green_desat", 0.10, 1.00, 2,
                                   "Fraction by which green FR-4 substrate saturation is suppressed (0.10–1.00)")
        self._pcb_boost  = dbl_row("Component Colour Boost:",        "pcb_comp_boost",  1.00, 2.00, 2,
                                   "Saturation multiplier applied to non-green (component) pixels (1.00–2.00)")

        form.addStretch()
        scroll.setWidget(inner); lay.addWidget(scroll, 1)

        bbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        bbox.accepted.connect(self._accept)
        bbox.rejected.connect(self.reject)
        lay.addWidget(bbox)

    def _accept(self):
        c = self._cfg
        c.set("camera_index",     self._cam_idx.value())
        c.set("confidence",       self._conf.value())
        c.set("autolabel_conf",   self._autolbl_conf.value())
        c.set("aoi_conf",         self._aoi_conf.value())
        c.set("match_iou",        self._iou.value())
        c.set("persistence",      self._persist.value())
        c.set("max_obj_width",    self._max_w.value())
        c.set("max_obj_height",   self._max_h.value())
        c.set("edge_margin",      self._edge.value())
        c.set("aoi_cal_runs",     self._aoi_cal.value())
        mp = self._model_le.text().strip()
        if mp: c.set("model_path", mp)
        c.set("autosave_flags",   self._autosave_cb.isChecked())
        c.set("log_history",      self._log_history_cb.isChecked())
        c.set("save_annot_history",self._save_annot_cb.isChecked())
        c.set("pcb_green_desat",  self._pcb_desat.value())
        c.set("pcb_comp_boost",   self._pcb_boost.value())
        c.flush()
        self.settings_changed.emit()
        self.accept()
