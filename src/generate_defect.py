"""
PCB Defect Image Generator  v3.0
==================================
Changes vs v2.0:
  ✓ Misaligned REMOVED — not supported by inspector
  ✓ Wrong polarity ONLY for polarized components (ICs, transistors, diodes, LEDs)
  ✓ Missing: solid PCB-colour fill + pad darkening (clear diff, no phantom detection)
  ✓ Wrong component: prefers cross-group donor for maximum visual difference
  ✓ Saved as PNG (lossless) — no JPEG compression noise in diff
  ✓ Defect mix: missing=0.40, wrong_component=0.35, wrong_polarity=0.25
"""

import argparse, copy, json, os, random, sys, csv
from pathlib import Path

def _pip(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:    import cv2
except: _pip("opencv-python"); import cv2
try:    import numpy as np
except: _pip("numpy"); import numpy as np
try:    from ultralytics import YOLO
except: _pip("ultralytics"); from ultralytics import YOLO

try:
    from sahi import AutoDetectionModel as _SAHIDetModel
    from sahi.predict import get_sliced_prediction as _sahi_pred
    HAS_SAHI = True
except ImportError:
    HAS_SAHI = False

SAHI_TRIGGER = 1280   # use SAHI when image longest edge > this
SAHI_TILE    = 1280   # tile size (matches training)
SAHI_OVERLAP = 0.20
_sahi_model_cache: dict = {}    

DEFAULT_GOLDEN     = r"D:\MELSS\AOI\NEW_TEST_IMGS\images\Riyan_20260324_124639_🍭DSLR Ultra pixel by Riyan (R18).jpg"
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CANONICAL = ROOT / "models" / "best.pt"
DEFAULT_MODEL      = str(DEFAULT_MODEL_CANONICAL)
DEFAULT_OUTPUT_DIR = "generated_defects"
DEFAULT_NUM        = 20
DEFAULT_CONF       = 0.15
DEFAULT_DEFECTS    = 7

DEFECT_MIX = {
    "missing":         0.40,
    "wrong_component": 0.35,
    "wrong_polarity":  0.25,
}

# Only components with a physical pin-1/anode-cathode orientation marker
POLARIZED_LABELS = {
    "ic (u)", "ic (ic)",
    "transistor (q)", "transistor (qa)",
    "diode (d)",
    "led",
    "capacitor (electrolytic)",
}

class Component:
    def __init__(self, id, label, conf, cx, cy, w, h):
        self.id=id; self.label=label; self.conf=conf
        self.cx=cx; self.cy=cy; self.w=w; self.h=h

    def xyxy(self, W, H):
        return (max(0, int((self.cx-self.w/2)*W)),
                max(0, int((self.cy-self.h/2)*H)),
                min(W, int((self.cx+self.w/2)*W)),
                min(H, int((self.cy+self.h/2)*H)))

    def area(self): return self.w * self.h

    def is_polarized(self):
        return self.label.lower() in POLARIZED_LABELS

    def overlaps(self, other, thr=0.05):
        ax1,ay1=self.cx-self.w/2, self.cy-self.h/2
        ax2,ay2=self.cx+self.w/2, self.cy+self.h/2
        bx1,by1=other.cx-other.w/2, other.cy-other.h/2
        bx2,by2=other.cx+other.w/2, other.cy+other.h/2
        ix=max(0,min(ax2,bx2)-max(ax1,bx1))
        iy=max(0,min(ay2,by2)-max(ay1,by1))
        inter=ix*iy
        if inter==0: return False
        return inter/(self.area()+other.area()-inter) > thr


def _nms(comps, iou_thr=0.30, iomin_thr=0.70, dist_thr=0.025):
    """Post-SAHI NMS — same logic as inspector."""
    import math as _m
    if len(comps)<=1: return comps
    ordered=sorted(comps,key=lambda c:-c.conf); keep=[]; supp=set()
    SIZE_GUARD=4.0
    for i,a in enumerate(ordered):
        if i in supp: continue
        keep.append(a); aa=a.w*a.h
        ax1,ay1=a.cx-a.w/2,a.cy-a.h/2; ax2,ay2=a.cx+a.w/2,a.cy+a.h/2
        for j in range(i+1,len(ordered)):
            if j in supp: continue
            b=ordered[j]; ab=b.w*b.h
            ix=max(0,min(ax2,b.cx+b.w/2)-max(ax1,b.cx-b.w/2))
            iy=max(0,min(ay2,b.cy+b.h/2)-max(ay1,b.cy-b.h/2))
            inter=ix*iy
            iou=inter/(aa+ab-inter) if inter>0 else 0
            iomin=inter/max(min(aa,ab),1e-9) if inter>0 else 0
            a_has_b=(ax1<=b.cx<=ax2 and ay1<=b.cy<=ay2)
            if iou>iou_thr: supp.add(j); continue
            if iomin>iomin_thr and aa>=SIZE_GUARD*ab: supp.add(j); continue
            if a_has_b and aa>=SIZE_GUARD*ab: supp.add(j); continue
            if b.label.lower()==a.label.lower():
                if _m.hypot(a.cx-b.cx,a.cy-b.cy)<dist_thr: supp.add(j)
    for uid,c in enumerate(keep): c.id=uid
    return keep


def _simple_nms(comps, iou_thr=0.40):
    """Simple IoU NMS — identical to AutoLabelThread's post-predict pass.
    Much more aggressive than _nms (which is conservative SAHI-tile NMS)
    and gives detection counts matching the autolabeler."""
    import math as _m
    if len(comps) <= 1: return comps
    ordered = sorted(range(len(comps)), key=lambda i: -comps[i].conf)
    keep, supp = [], set()
    for i in ordered:
        if i in supp: continue
        keep.append(i)
        a = comps[i]
        ax1,ay1 = a.cx-a.w/2, a.cy-a.h/2
        ax2,ay2 = a.cx+a.w/2, a.cy+a.h/2
        for j in ordered:
            if j in supp or j == i: continue
            b = comps[j]
            ix = max(0, min(ax2,b.cx+b.w/2) - max(ax1,b.cx-b.w/2))
            iy = max(0, min(ay2,b.cy+b.h/2) - max(ay1,b.cy-b.h/2))
            inter = ix * iy
            union = a.w*a.h + b.w*b.h - inter
            if union > 0 and inter/union > iou_thr:
                supp.add(j)
    comps = [comps[i] for i in keep]
    for uid, c in enumerate(comps): c.id = uid
    return comps


def detect(model, img_bgr, conf=0.15, model_path="", use_sahi=False):
    """Detection matching AutoLabelThread inference exactly.

    SAHI is only used when explicitly requested via use_sahi=True
    (i.e. --use-sahi flag). Default is standard YOLO at imgsz=640,
    which is fast and matches the autolabeler's detection quality.
    """
    H,W=img_bgr.shape[:2]
    comps=[]; uid=0

    # ── SAHI path — only when explicitly requested ──────────────────────────
    if use_sahi and HAS_SAHI and model_path and max(H,W)>SAHI_TRIGGER:
        key=(model_path,round(conf,4))
        if key not in _sahi_model_cache:
            print(f"  [SAHI] Loading detection model (once)…")
            _sahi_model_cache[key]=_SAHIDetModel.from_pretrained(
                model_type="ultralytics",model_path=model_path,
                confidence_threshold=conf,device="cpu")
        try:
            from PIL import Image as _PIL
            pil=_PIL.fromarray(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB))
            result=_sahi_pred(image=pil,detection_model=_sahi_model_cache[key],
                              slice_height=SAHI_TILE,slice_width=SAHI_TILE,
                              overlap_height_ratio=SAHI_OVERLAP,
                              overlap_width_ratio=SAHI_OVERLAP,
                              postprocess_match_threshold=0.20,verbose=0)
            for obj in result.object_prediction_list:
                x1,y1,x2,y2=obj.bbox.minx,obj.bbox.miny,obj.bbox.maxx,obj.bbox.maxy
                comps.append(Component(uid,obj.category.name,
                                       float(obj.score.value),
                                       ((x1+x2)/2)/W,((y1+y2)/2)/H,
                                       (x2-x1)/W,(y2-y1)/H))
                uid+=1
            before=len(comps); comps=_simple_nms(comps)
            print(f"  [SAHI] {before} raw → {len(comps)} after NMS  ({W}×{H})")
            return comps
        except Exception as e:
            print(f"  [SAHI] Failed ({e}) → falling back to standard YOLO")
            comps=[]; uid=0

    # ── Standard YOLO — matches AutoLabelThread exactly ─────────────────────
    # model.predict() with iou=0.35 + imgsz=640 (YOLO internal resize, fast).
    # Then simple IoU NMS at 0.40 — same as autolabeler post-predict pass.
    results = model.predict(img_bgr, conf=conf, iou=0.35, imgsz=640, verbose=False)
    for r in results:
        if r.boxes is None: continue
        for b in r.boxes:
            cx,cy,w,h=b.xywhn[0].tolist()
            comps.append(Component(uid,r.names[int(b.cls[0])],
                                   float(b.conf[0]),cx,cy,w,h))
            uid+=1
    comps=_simple_nms(comps)
    print(f"  [YOLO] {len(comps)} components  ({W}×{H})")
    return comps


def _rot(patch, angle):
    h,w=patch.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
    return cv2.warpAffine(patch,M,(w,h),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _sample_bg(img, x1, y1, x2, y2):
    """Sample median background colour from border ring around component."""
    H,W=img.shape[:2]
    pad=max(6,int(min(x2-x1,y2-y1)*0.4))
    bx1,by1=max(0,x1-pad),max(0,y1-pad)
    bx2,by2=min(W,x2+pad),min(H,y2+pad)
    region=img[by1:by2,bx1:bx2]
    mask=np.ones(region.shape[:2],dtype=bool)
    mask[y1-by1:y2-by1, x1-bx1:x2-bx1]=False
    px=region[mask]
    return np.median(px,axis=0).astype(np.float32) if len(px) else np.array([34,100,34],dtype=np.float32)


def _build_bg_mask(img, comps):
    """
    Boolean mask (H×W) that is True only where there is no component bbox.
    Uses known detections, so it works correctly on dense boards where every
    neighbouring strip around a slot will contain other components.
    """
    H, W = img.shape[:2]
    mask = np.ones((H, W), dtype=bool)
    for c in comps:
        bx1, by1, bx2, by2 = c.xyxy(W, H)
        # Small halo so we also skip the immediate edge of each component
        exp = max(2, int(min(bx2 - bx1, by2 - by1) * 0.10))
        mask[max(0, by1-exp):min(H, by2+exp),
             max(0, bx1-exp):min(W, bx2+exp)] = False
    return mask


def inject_missing(img, comp, comps):
    """
    Colour-matched bare-PCB fill.

    Samples the median + per-channel std of every confirmed bare-PCB pixel
    on the board (bg_mask), fills the slot with that colour + matching noise,
    then cosine-feathers the edge into the surrounding image.  Works at any
    component size because the colour is measured, not guessed.
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = comp.xyxy(W, H)
    if x2 <= x1 or y2 <= y1:
        return img, None

    ph, pw = y2 - y1, x2 - x1

    # ── 1. Board-wide bare-PCB mask ──────────────────────────────────────────
    bg_mask = _build_bg_mask(img, comps)
    bg_mask[y1:y2, x1:x2] = False

    # ── 2. Measure bare-PCB colour from safe pixels near the slot first,
    #       then the whole board if not enough nearby pixels exist ────────────
    def _safe_pixels(radius_factor):
        r = int(max(ph, pw) * radius_factor)
        ry1, ry2 = max(0, y1 - r), min(H, y2 + r)
        rx1, rx2 = max(0, x1 - r), min(W, x2 + r)
        region_mask = bg_mask[ry1:ry2, rx1:rx2]
        px = img[ry1:ry2, rx1:rx2][region_mask].astype(np.float32)
        return px

    safe_px = np.empty((0, 3), dtype=np.float32)
    for factor in (3, 8, 40):
        safe_px = _safe_pixels(factor)
        if len(safe_px) >= 64:
            break

    if len(safe_px) < 8:
        safe_px = img[bg_mask].astype(np.float32)
    if len(safe_px) == 0:
        safe_px = img.reshape(-1, 3).astype(np.float32)

    mean_c = np.median(safe_px, axis=0)          # robust colour centre
    std_c  = np.std(safe_px, axis=0).clip(2, 10) # realistic grain, not too noisy

    # ── 3. Fill with measured colour + per-channel noise ────────────────────
    fill = np.random.normal(mean_c, std_c, (ph, pw, 3)).astype(np.float32)

    # ── 4. Cosine feather — proportional to slot size ────────────────────────
    feather = min(4, min(ph, pw) // 4)
    ramp = np.ones((ph, pw), dtype=np.float32)
    for k in range(feather):
        t = 0.5 - 0.5 * np.cos(np.pi * k / feather)
        ramp[k, :]      = np.minimum(ramp[k, :],      t)
        ramp[ph-1-k, :] = np.minimum(ramp[ph-1-k, :], t)
        ramp[:, k]      = np.minimum(ramp[:, k],      t)
        ramp[:, pw-1-k] = np.minimum(ramp[:, pw-1-k], t)
    ramp = ramp[:, :, np.newaxis]

    orig    = img[y1:y2, x1:x2].astype(np.float32)
    blended = ramp * fill + (1.0 - ramp) * orig
    img[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return img, {"defect_type": "missing", "detail": "colour-matched-fill+cosine-feather"}


def inject_wrong_component(img, comp, comps, rng):
    """
    Swap rules — only swap within compatible groups:
      passive-small (resistor, ferrite)     ↔  passive-small
      passive-large (capacitor, inductor)   ↔  passive-large
      active-small  (transistor, diode)     ↔  active-small  (similar size only)
      active-large  (IC)                    ↔  active-large  (similar size only)

    Cross-group swaps (IC into resistor etc.) are NOT done because:
      - Extreme resize destroys visual features → YOLO sees nothing
      - Centre-crop also unreliable at >4× size difference
    If no compatible donor exists → return None (skip this defect, caller retries).

    Paste strategy:
      ratio ≤ 1.6×: resize to fit exactly
      ratio 1.6–3×: resize + random offset within slot (± 20% of slot size)
      ratio > 3×:   skip (too different, would distort beyond recognition)
    """
    H, W = img.shape[:2]
    ax1,ay1,ax2,ay2 = comp.xyxy(W, H)
    if ax2<=ax1 or ay2<=ay1: return img, None
    tw, th = ax2-ax1, ay2-ay1

    def _compat_group(c):
        """
        Compatibility groups — who can swap with whom:
          passive  : resistor <-> capacitor (user requirement)
          active_s : transistor <-> diode (small active)
          ic_N     : IC <-> IC of similar area bucket only
          other    : everything else with itself
        """
        lbl = c.label.lower()
        a   = c.area()
        if any(x in lbl for x in ("resistor","capacitor","ferrite","inductor")):
            return "passive"          # R↔C, ferrite↔inductor etc. all ok
        if any(x in lbl for x in ("transistor","diode")):
            return "active_s"
        if "ic" in lbl:
            # Bucket by area so only similar-sized ICs swap
            if   a < 0.004:  return "ic_0"
            elif a < 0.012:  return "ic_1"
            elif a < 0.030:  return "ic_2"
            else:            return "ic_3"
        return f"other_{lbl[:8]}"

    target_grp = _compat_group(comp)

    # Build donor list: same compat group, different label, size ratio ≤ 4×
    donors = []
    for c in comps:
        if c.id == comp.id: continue
        if c.label.lower() == comp.label.lower(): continue
        if c.conf <= 0.15: continue
        if _compat_group(c) != target_grp: continue
        bx1,by1,bx2,by2 = c.xyxy(W,H)
        if bx2<=bx1 or by2<=by1: continue
        dw,dh = bx2-bx1, by2-by1
        if dw<4 or dh<4: continue
        ratio = max(c.area(), comp.area()) / max(min(c.area(), comp.area()), 1e-9)
        if ratio > 4.0: continue
        donors.append((c, bx1,by1,bx2,by2, ratio))

    if not donors:
        return img, None   # no compatible donor — skip, caller will retry

    # Pick randomly from donors (not always the closest — adds variety)
    donors.sort(key=lambda x: x[5])            # sort by ratio
    top = donors[:max(1, len(donors)//2+1)]    # top half by similarity
    donor, bx1,by1,bx2,by2, ratio = rng.choice(top)

    donor_patch = img[by1:by2, bx1:bx2].copy()
    dw,dh = bx2-bx1, by2-by1

    if ratio <= 1.6:
        # Straightforward resize — sizes are close enough
        paste_patch = cv2.resize(donor_patch, (tw,th), interpolation=cv2.INTER_LINEAR)
        img[ay1:ay2, ax1:ax2] = paste_patch

    else:
        # Resize to fit, then apply a random small offset so it doesn't sit
        # perfectly centered — makes the defect look more like a real placement error
        paste_patch = cv2.resize(donor_patch, (tw,th), interpolation=cv2.INTER_LINEAR)
        # Random offset: ±20% of slot dimension
        max_dy = max(1, int(th * 0.20))
        max_dx = max(1, int(tw * 0.20))
        off_y  = rng.randint(-max_dy, max_dy)
        off_x  = rng.randint(-max_dx, max_dx)
        dst_y1 = max(0, ay1 + off_y);  dst_y2 = min(H, ay2 + off_y)
        dst_x1 = max(0, ax1 + off_x);  dst_x2 = min(W, ax2 + off_x)
        src_y1 = dst_y1 - (ay1+off_y); src_y2 = src_y1 + (dst_y2-dst_y1)
        src_x1 = dst_x1 - (ax1+off_x); src_x2 = src_x1 + (dst_x2-dst_x1)
        if dst_y2>dst_y1 and dst_x2>dst_x1:
            img[dst_y1:dst_y2, dst_x1:dst_x2] = paste_patch[src_y1:src_y2, src_x1:src_x2]

    return img, {"defect_type":  "wrong_component",
                 "detail":       f"{comp.label} -> {donor.label} (ratio={ratio:.1f} grp={target_grp})",
                 "found_label":  donor.label}


def inject_wrong_polarity(img, comp):
    """180° rotation. Only call for polarized components."""
    H,W=img.shape[:2]
    x1,y1,x2,y2=comp.xyxy(W,H)
    if x2<=x1 or y2<=y1: return img, None
    img[y1:y2,x1:x2]=_rot(img[y1:y2,x1:x2].copy(),180)
    return img, {"defect_type":"wrong_polarity","detail":"180-deg rotation"}


COLOURS={"missing":(0,0,255),"wrong_component":(255,0,255),"wrong_polarity":(255,128,0),"ok":(0,200,0)}
DLABELS={"missing":"MISSING","wrong_component":"WRONG PART","wrong_polarity":"WRONG POLARITY"}

def draw_overlay(img, comps, defect_logs=None):
    out=img.copy(); H,W=out.shape[:2]
    dm={d["component_id"]:d for d in (defect_logs or [])}
    for c in comps:
        x1,y1,x2,y2=c.xyxy(W,H)
        d=dm.get(c.id); dtype=d["defect_type"] if d else "ok"
        col=COLOURS.get(dtype,(0,200,0))
        cv2.rectangle(out,(x1,y1),(x2,y2),col,3 if d else 1)
        if d:
            tag=f"{c.label} [{DLABELS.get(dtype,dtype)}]"
            if dtype=="wrong_component": tag+=f" -> {d.get('found_label','?')}"
            (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.34,1)
            cv2.rectangle(out,(x1,max(0,y1-th-5)),(x1+tw+4,y1),col,-1)
            cv2.putText(out,tag,(x1+2,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.34,(0,0,0),1,cv2.LINE_AA)
        else:
            cv2.putText(out,c.label[:14],(x1+1,y1-2),cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,(0,180,0),1,cv2.LINE_AA)
    return out


def generate(args):
    rng=random.Random(args.seed); np.random.seed(args.seed)
    out_dir=Path(args.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"images").mkdir(exist_ok=True)

    if HAS_SAHI:
        print(f"[SAHI] Available — auto-enabled for images >{SAHI_TRIGGER}px")
    else:
        print("[WARN] sahi not installed — using standard YOLO only. Install: pip install sahi")
    print(f"[MODEL] {args.model}")
    model=YOLO(args.model)
    golden_img=cv2.imread(args.golden)
    if golden_img is None: sys.exit(f"[ERROR] Cannot read {args.golden}")
    H,W=golden_img.shape[:2]
    print(f"[GOLDEN] {W}x{H}  {args.golden}")

    comps=detect(model,golden_img,conf=args.conf,model_path=args.model,use_sahi=args.use_sahi)
    if not comps: sys.exit("[ERROR] Zero components detected.")

    counts={}
    for c in comps: counts[c.label]=counts.get(c.label,0)+1
    print(f"[GOLDEN] {len(comps)} components:")
    for lbl,n in sorted(counts.items()): print(f"  {lbl:<30} x{n}")

    polarized=[c for c in comps if c.is_polarized()]
    print(f"\n[POOLS]")
    print(f"  MISSING / WRONG PART : all {len(comps)} components")
    print(f"  WRONG POLARITY       : {len(polarized)} polarized "
          f"({', '.join(sorted(set(c.label for c in polarized))) or 'NONE'})")
    if not polarized:
        print(f"  [WARN] No polarized components — wrong_polarity redirected to missing/wrong_component")

    cv2.imwrite(str(out_dir/"golden_annotated.jpg"),draw_overlay(golden_img,comps))
    with open(out_dir/"golden_components.json","w") as f:
        json.dump([{"id":c.id,"label":c.label,"conf":round(c.conf,3),
                    "cx":round(c.cx,5),"cy":round(c.cy,5),
                    "w":round(c.w,5),"h":round(c.h,5)} for c in comps],f,indent=2)

    eff_mix=dict(DEFECT_MIX)
    if not polarized:
        extra=eff_mix.pop("wrong_polarity",0)
        eff_mix["missing"]=eff_mix.get("missing",0)+extra*0.6
        eff_mix["wrong_component"]=eff_mix.get("wrong_component",0)+extra*0.4

    dtypes=list(eff_mix.keys()); dprobs=[eff_mix[k] for k in dtypes]
    all_meta=[]; rows=[]
    print(f"\n[GEN] {args.num} boards, 1-{args.defects_per_board} defects, mix={eff_mix}\n")

    for i in range(args.num):
        img=golden_img.copy()
        wc=copy.deepcopy(comps)
        nd=rng.randint(1,max(1,args.defects_per_board))
        victims=rng.sample(range(len(wc)),min(nd,len(wc)))
        log=[]

        for vi in victims:
            c=wc[vi]
            # Pick defect type, then pick an appropriate victim if needed
            dtype=rng.choices(dtypes,weights=dprobs,k=1)[0]
            if dtype=="wrong_polarity":
                if not c.is_polarized():
                    # Current victim can't have polarity defect — swap to a polarized one
                    polar_victims=[i for i in range(len(wc))
                                   if wc[i].is_polarized() and wc[i].conf>0.15 and i!=vi
                                   and i not in victims[:victims.index(vi)]]
                    if polar_victims:
                        vi=rng.choice(polar_victims); c=wc[vi]
                    else:
                        dtype="missing"   # no polarized available

            meta=None
            rec={"component_id":c.id,"original_label":c.label}

            if dtype=="missing":
                img,meta=inject_missing(img,c,wc); wc[vi].conf=0.0
            elif dtype=="wrong_component":
                img,meta=inject_wrong_component(img,c,wc,rng)
                if meta is None:
                    # No compatible same-group donor — fall back to missing
                    img,meta=inject_missing(img,c,wc); wc[vi].conf=0.0
            elif dtype=="wrong_polarity":
                img,meta=inject_wrong_polarity(img,c)

            if meta: rec.update(meta); log.append(rec)

        name=f"board_{i:03d}"
        cv2.imwrite(str(out_dir/"images"/f"{name}.png"),img)   # lossless
        cv2.imwrite(str(out_dir/f"{name}_annotated.jpg"),draw_overlay(img,comps,log))

        all_meta.append({"board":name,"image":str(out_dir/"images"/f"{name}.png"),
                          "n_defects":len(log),"defects":log})
        types=[d["defect_type"] for d in log]
        labels=[d.get("original_label","?") for d in log]
        print(f"  [{i+1:3d}/{args.num}] {name}  "
              +"  ".join(("MISS" if t=="missing" else "POLARITY" if t=="wrong_polarity" else "WRONG_PART" if t=="wrong_component" else t[:5].upper())+f"({l})" for t,l in zip(types,labels)))
        rows.append([name,len(log),",".join(types),",".join(labels)])

    with open(out_dir/"manifest.json","w") as f:
        json.dump({"golden":args.golden,"model":args.model,"conf":args.conf,
                   "n_components":len(comps),"boards":all_meta},f,indent=2)
    with open(out_dir/"summary.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["board","n_defects","types","components"])
        w.writerows(rows)

    print(f"\n[DONE] -> {out_dir.resolve()}")
    print(f"  images/*.png       lossless boards")
    print(f"  *_annotated.jpg    overlays")
    print(f"  manifest.json      ground-truth")
    print(f"  golden_annotated.jpg  check this for YOLO blind spots")
    print(f"\n[TIP] Run the inspector on images/ then compare with:")
    print(f"       python defect_generator.py --compare <inspector_json>")



def compare_results(manifest_path: str, inspector_json: str):
    """
    Cross-reference injected defects vs detected defects.
    Matches by NORMALISED POSITION (cx,cy) — NOT by component_id,
    since generator and inspector assign IDs independently.
    """
    import json, math

    with open(manifest_path) as f: manifest = json.load(f)
    with open(inspector_json) as f: inspector = json.load(f)

    # Load golden components from manifest to get cx/cy for each id
    golden_path = os.path.join(os.path.dirname(manifest_path), "golden_components.json")
    with open(golden_path) as f: golden_comps = json.load(f)
    gen_comp_map = {c["id"]: c for c in golden_comps}   # gen_id → {cx,cy,label}

    # Index inspector results — try multiple name formats
    insp_idx = {}
    for entry in inspector:
        raw = entry.get("name", "")
        for key in [raw, os.path.splitext(raw)[0], raw.replace(".png","").replace(".jpg","")]:
            insp_idx[key] = entry

    # Inspector golden atlas: load from inspector defect details
    # Inspector defects have cx/cy embedded in details OR we must match by proximity
    # Strategy: for each injected defect position (cx,cy), find the closest
    # inspector defect within MATCH_R and check type
    MATCH_R   = 0.05   # 5% of image — if inspector flagged anything this close = matched
    COMPAT = {
        ("missing","wrong_component"), ("wrong_component","missing"),
        ("missing","wrong_polarity"),  ("wrong_polarity","missing"),
        ("missing","misaligned"),      ("misaligned","missing"),
        ("wrong_component","wrong_polarity"), ("wrong_polarity","wrong_component"),
    }
    TYPE_SHORT = {
        "missing":"MISS","wrong_component":"WRONG",
        "wrong_polarity":"POLARITY","misaligned":"MISALIGN",
    }

    total_inj=total_tp=total_fp=total_fn=0
    type_matrix={}

    print()
    print("═"*96)
    print(f"  {'Board':<15} {'Injected':<38} {'Detected':<30} {'Match'}")
    print("─"*96)

    for board_meta in manifest["boards"]:
        bname = board_meta["board"]

        # Find inspector entry — try multiple name variants
        det_entry = (insp_idx.get(bname) or
                     insp_idx.get(bname+".png") or
                     insp_idx.get(bname+".jpg") or {})
        det_defects = det_entry.get("defects", [])

        inj = board_meta["defects"]
        total_inj += len(inj)

        matched_det_indices = set()
        tps=[];fps=[];fns=[]

        for inj_def in inj:
            itype = inj_def["defect_type"]
            gcid  = inj_def.get("component_id", -1)
            gc    = gen_comp_map.get(gcid, {})
            icx, icy = gc.get("cx", -1), gc.get("cy", -1)

            # Find closest inspector defect by cx/cy proximity
            best_idx = -1; best_dist = MATCH_R
            for di, ddef in enumerate(det_defects):
                if di in matched_det_indices: continue
                # Inspector defects have component_id in its own space
                # Try to extract coords from details string, or use rough match
                det_type = ddef.get("defect_type","")
                # Check compatibility first (type must be same or compatible)
                type_ok = (det_type == itype or
                           (det_type,itype) in COMPAT or
                           (itype,det_type) in COMPAT)
                if not type_ok: continue
                # Position: use cx/cy saved in enriched export, or label fallback
                dcx = ddef.get("cx", -1)
                dcy = ddef.get("cy", -1)
                if dcx >= 0 and icx >= 0:
                    dist = math.hypot(icx-dcx, icy-dcy)
                    if dist < best_dist:
                        best_dist=dist; best_idx=di
                else:
                    # Fallback: match by expected label if no coords
                    exp_lbl = ddef.get("expected_label", ddef.get("label","")).lower()
                    inj_lbl = gc.get("label","").lower()
                    if exp_lbl == inj_lbl and best_idx < 0:
                        best_dist=0.04; best_idx=di

            if best_idx >= 0:
                total_tp += 1
                matched_det_indices.add(best_idx)
                ddef = det_defects[best_idx]
                det_type = ddef.get("defect_type","")
                tps.append(f"✓{TYPE_SHORT.get(itype,itype)}")
                t_key=(itype,det_type)
                type_matrix[t_key]=type_matrix.get(t_key,0)+1
            else:
                total_fn += 1
                fns.append(f"{TYPE_SHORT.get(itype,itype)}(id={gcid} cx={icx:.3f} cy={icy:.3f})")
                t_key=(itype,"MISSED")
                type_matrix[t_key]=type_matrix.get(t_key,0)+1

        # Unmatched inspector detections = FPs
        for di,ddef in enumerate(det_defects):
            if di not in matched_det_indices:
                total_fp+=1
                fps.append(f"FP:{TYPE_SHORT.get(ddef.get('defect_type','?'),'?')}")

        inj_str = "  ".join(
            f"{TYPE_SHORT.get(d['defect_type'],d['defect_type'][:5])}"
            f"({d.get('original_label','?')[:6]})" for d in inj)
        det_str = ("PASS" if not det_defects else
                   "  ".join(f"{TYPE_SHORT.get(d.get('defect_type','?'),'?')}"
                              f"({d.get('expected_label','?')[:6]})"
                              for d in det_defects[:4]))
        status = ("✓ALL" if tps and not fns and not fps else
                  ("FN" if fns else ("FP" if fps else "✓")))

        print(f"  {bname:<15} {inj_str:<38} {det_str:<30} {status}")
        if fns: print(f"  {'':15} FN: {' '.join(fns[:4])}")
        if fps: print(f"  {'':15} FP: {' '.join(fps[:4])}")

    prec = total_tp/(total_tp+total_fp) if total_tp+total_fp else 0
    rec  = total_tp/(total_tp+total_fn) if total_tp+total_fn else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0

    print("═"*96)
    print(f"  TOTAL  Injected={total_inj}  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    if type_matrix:
        print()
        print("  Type breakdown (injected → detected/missed):")
        for (it,dt),cnt in sorted(type_matrix.items(),key=lambda x:-x[1]):
            flag = "✓" if it==dt else ("~" if dt!="MISSED" else "✗")
            print(f"    {flag}  {TYPE_SHORT.get(it,it):<12} → {TYPE_SHORT.get(dt,dt):<12}  ×{cnt}")
    print()



def parse_args():
    p=argparse.ArgumentParser(description="PCB Defect Generator v3.0")
    p.add_argument("--golden",default=DEFAULT_GOLDEN)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--num",type=int,default=DEFAULT_NUM)
    p.add_argument("--conf",type=float,default=DEFAULT_CONF)
    p.add_argument("--defects-per-board",type=int,default=DEFAULT_DEFECTS)
    p.add_argument("--output-dir",default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--use-sahi",action="store_true",
                   help="SAHI tiled inference for large boards (>1280px). Recommended for big images.")
    p.add_argument("--compare",default="",
                   help="Path to inspector JSON to compare against manifest.json")
    return p.parse_args()

if __name__=="__main__":
    _args = parse_args()
    if _args.compare:
        manifest = os.path.join(_args.output_dir, "manifest.json")
        compare_results(manifest, _args.compare)
    else:
        generate(_args)