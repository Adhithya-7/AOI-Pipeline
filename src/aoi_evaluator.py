"""
PCB Defect Accuracy Evaluator  v3.9
====================================
Injects defects → runs AOI inference → reports accuracy + confusion matrix.
No GUI. No large file I/O. Just pure evaluation.

v3.9 changes (over v3.6):
  [ACCURACY — 3 targeted fixes from 1000-board v3.6 run (3004 defects)]

  Dominant remaining failure: 124× wrong_polarity → wrong_component.
  Root cause: NCC is intensity-based and cannot detect orientation
  direction for near-symmetric components. A 180° rotation reverses every
  edge direction — a signal NCC ignores entirely.

  Fix N — Gradient Orientation Histogram (GOH) polarity signal:
    A GOH over the Sobel-gradient angle field captures directional
    asymmetry that NCC misses. Under 180° rotation the histogram shifts
    by exactly N//2 bins (6 bins × 30° = 180°). A new NCC on the rolled
    golden histogram vs. the test histogram gives a reliable polarity
    delta even for components whose pixel-intensity pattern is symmetric.
    Threshold _GOH_POLAR_MIN_DELTA = 0.08 (gap: WCOM max ≈ 0.06,
    true WPOL min ≈ 0.15).
    Tie-breaker asymmetry vote (_GOH_ASYM_VOTE_BAND = 0.05–0.10):
    top/bottom + left/right intensity asymmetry flips sign under 180°
    rotation for components with a visible polarity marker.
    Together expected to fix 60–80% of the 124 WPOL→WCOM FNs.

  Fix O — Edge-density rescue for wrong_component → missing swaps:
    7 WCOM donors resized into small slots had var 0.8–2.0 (below
    _VAR_WC_CONFIRM=80), fooling the variance gate into "missing".
    Canny edge density (edges/pixel) is not confused by low contrast
    after resize because Canny adaptive-thresholds per-image.
    Gaussian fill: density 0.00–0.01. Donor component: 0.04–0.18.
    _EDGE_DENSITY_WC_MIN = 0.045 sits well above fill noise.
    Expected to fix 5–6 of the 7 WCOM→MISS cases.

  Fix P — Two-pass misalignment angle sweep (coarse 5° + fine 1°):
    12 MALI FNs traced to injected angles in the 10°–20° boundary region
    where the recovery NCC peaks narrowly between the 5° coarse sample
    points (e.g. a 12° injection recovers 0.58 at both 10° and 15° —
    just below the 0.60 acceptance threshold).
    Strategy: coarse pass at 5° steps (unchanged), then fine pass at 1°
    steps over ±8° around the coarse best. Adds ≤17 extra warpAffine
    calls after the coarse pass — negligible overhead.
    Expected to fix 4–6 of the 12 MALI FNs.

  [IRREDUCIBLE FAILURES REMAINING]
  • C8 IC WPOL: edge histogram is symmetric under 180° rotation at this
    component's size; GOH delta ≈ 0. No current signal discriminates.
  • C84 Cap WPOL: ssim=0.231, tmpl=0.099, GOH also fails (too small,
    symmetric body). ~8× per 100 boards. Irreducible.
  • ~2 WCOM→MISS: donors with var <0.8 AND edge density <0.04 (pixel-
    indistinguishable from Gaussian fill). Irreducible.
  • 11 MISS FP: calibration noise / edge components; no clean fix.

v3.6 changes (over v3.5):
  Fix M — add _NCC_POLAR_STRUCT_MIN_DELTA = 0.25 to structural fallback.
    WCOM FP cap deltas: 0.037, 0.040, 0.186 — all blocked.
    True WPOL deltas via this path ≥ 0.393. Gap is unambiguous.

v3.5 changes (over v3.3):
  Fix K — lower structural polarity thresholds (0.40/0.45 → 0.28/0.25).
  Fix L — Zone B polar no-decision bug: emit wrong_component when NCC
    gives no flip signal but rdiff is elevated.

v3.4 changes (over v3.3):
  Fix H — raise _NCC_POLARITY_STRONG 0.72 → 0.95.
  Fix I — tighten _NCC_POLARITY_SSIM_NEG -0.30 → -0.33.
  Fix J — add _POLAR_CROSS_RESCUE_NCC_MIN gate (0.50).

v3.3 changes (over v3.2):
  Fix G — add delta >= 0 to polar_struct_polarity fallback.

v3.2 changes (over v3.1):
  Fix E — cross_polar_rescue tmpl gate (0.20).
  Fix F — structural polarity fallback extended to all polarized.

v3.1 changes (over v3.0):
  Fix A/B/C/D — IC near-sym fallback, cross_zoneC flip rescue,
                 flat guard same-det rescue, verbose display fix.

v3.0 changes (over v2.3):
  Speed: NCC pre-blur, batch region diff, cached bg mask, faster calib.
  Accuracy: separate NCC margins IC/Cap, raised STRONG, cross_zoneC guard.

Usage:
  python aoi_evaluator_v39.py --golden <img> --model <.pt> [options]

Options:
  --boards   N        Number of test boards                 (default 1000)
  --defects  N        Max defects per board                 (default 5)
  --conf     F        YOLO confidence threshold             (default 0.15)
  --seed     N        RNG seed                              (default 42)
  --verbose           Print per-board details
  --debug             Per-slot SSIM/var/tmpl/path for FAILED slots
  --trace             Per-slot diagnostics for ALL Zone B/C slots
  --save_viz PATH     Save annotated diff image (PNG)
  --only_defect TYPE  Limit to one defect type for focused eval
"""

import argparse, copy, math, os, random, sys, time
from pathlib import Path
from collections import defaultdict

# ── Terminal colour helpers ───────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()
def _c(code, s): return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s
def _green(s):   return _c("32", s)
def _red(s):     return _c("31", s)
def _yellow(s):  return _c("33", s)
def _cyan(s):    return _c("36", s)
def _bold(s):    return _c("1",  s)
def _dim(s):     return _c("2",  s)

# ── Auto-install deps ─────────────────────────────────────────────────────────
def _pip(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:    import cv2
except: _pip("opencv-python"); import cv2
try:    import numpy as np
except: _pip("numpy"); import numpy as np
try:    from ultralytics import YOLO
except: _pip("ultralytics"); from ultralytics import YOLO
try:    from scipy.optimize import linear_sum_assignment; HAS_SCIPY = True
except: HAS_SCIPY = False

# ═════════════════════════════════════════════════════════════════════════════
# 1.  COMPONENT
# ═════════════════════════════════════════════════════════════════════════════
class Component:
    def __init__(self, id, label, conf, cx, cy, w, h):
        self.id=id; self.label=label; self.conf=conf
        self.cx=cx; self.cy=cy; self.w=w; self.h=h

    def xyxy(self, W, H):
        return (max(0,int((self.cx-self.w/2)*W)),
                max(0,int((self.cy-self.h/2)*H)),
                min(W,int((self.cx+self.w/2)*W)),
                min(H,int((self.cy+self.h/2)*H)))

    def area(self): return self.w * self.h

    def is_polarized(self):
        return self.label.lower() in {
            "ic (u)","ic (ic)","transistor (q)","transistor (qa)",
            "diode (d)","led","capacitor (electrolytic)",
            "capacitor (c)","cap array (cra)",
        }

# ═════════════════════════════════════════════════════════════════════════════
# 2.  DETECTION
# ═════════════════════════════════════════════════════════════════════════════
def _simple_nms(comps, iou_thr=0.40):
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
            if j in supp or j==i: continue
            b = comps[j]
            ix = max(0, min(ax2,b.cx+b.w/2)-max(ax1,b.cx-b.w/2))
            iy = max(0, min(ay2,b.cy+b.h/2)-max(ay1,b.cy-b.h/2))
            inter = ix*iy
            union = a.w*a.h + b.w*b.h - inter
            if union > 0 and inter/union > iou_thr:
                supp.add(j)
    comps = [comps[i] for i in keep]
    for uid,c in enumerate(comps): c.id = uid
    return comps

def detect(model, img_bgr, conf=0.15):
    results = model.predict(img_bgr, conf=conf, iou=0.35, imgsz=640, verbose=False)
    comps, uid = [], 0
    for r in results:
        if r.boxes is None: continue
        for b in r.boxes:
            cx,cy,w,h = b.xywhn[0].tolist()
            comps.append(Component(uid, r.names[int(b.cls[0])],
                                   float(b.conf[0]), cx, cy, w, h))
            uid += 1
    return _simple_nms(comps)

# ═════════════════════════════════════════════════════════════════════════════
# 3.  DEFECT INJECTION
# ═════════════════════════════════════════════════════════════════════════════
def _build_bg_mask(img, comps):
    H, W = img.shape[:2]
    mask = np.ones((H, W), dtype=bool)
    for c in comps:
        bx1,by1,bx2,by2 = c.xyxy(W, H)
        exp = max(2, int(min(bx2-bx1, by2-by1)*0.10))
        mask[max(0,by1-exp):min(H,by2+exp), max(0,bx1-exp):min(W,bx2+exp)] = False
    return mask

def inject_missing(img, comp, comps, cached_bg_mask=None):
    H, W = img.shape[:2]
    x1,y1,x2,y2 = comp.xyxy(W, H)
    if x2<=x1 or y2<=y1: return img, None, cached_bg_mask
    ph, pw = y2-y1, x2-x1

    if cached_bg_mask is None:
        bg_mask = _build_bg_mask(img, comps)
    else:
        bg_mask = cached_bg_mask
    bg_mask[y1:y2, x1:x2] = False

    def _safe_pixels(rf):
        r = int(max(ph, pw)*rf)
        ry1,ry2 = max(0,y1-r), min(H,y2+r)
        rx1,rx2 = max(0,x1-r), min(W,x2+r)
        px = img[ry1:ry2, rx1:rx2][bg_mask[ry1:ry2, rx1:rx2]].astype(np.float32)
        return px

    safe_px = np.empty((0,3), dtype=np.float32)
    for factor in (3, 8, 40):
        safe_px = _safe_pixels(factor)
        if len(safe_px) >= 64: break
    if len(safe_px) < 8: safe_px = img[bg_mask].astype(np.float32)
    if len(safe_px) == 0: safe_px = img.reshape(-1,3).astype(np.float32)

    mean_c = np.median(safe_px, axis=0)
    std_c  = np.std(safe_px, axis=0).clip(2, 10)
    fill   = np.random.normal(mean_c, std_c, (ph,pw,3)).astype(np.float32)

    feather = min(4, min(ph,pw)//4)
    if feather > 0:
        ks = np.arange(feather, dtype=np.float32)
        edge = (0.5 - 0.5 * np.cos(np.pi * ks / feather))
        ramp = np.ones((ph, pw), dtype=np.float32)
        ramp[:feather,  :]  = np.minimum(ramp[:feather,  :],  edge[:, np.newaxis])
        ramp[-feather:, :]  = np.minimum(ramp[-feather:, :],  edge[::-1, np.newaxis])
        ramp[:,  :feather]  = np.minimum(ramp[:,  :feather],  edge[np.newaxis, :])
        ramp[:, -feather:]  = np.minimum(ramp[:, -feather:],  edge[np.newaxis, ::-1])
    else:
        ramp = np.ones((ph, pw), dtype=np.float32)
    ramp = ramp[:,:,np.newaxis]

    orig    = img[y1:y2, x1:x2].astype(np.float32)
    blended = ramp*fill + (1.0-ramp)*orig
    img[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return img, {"defect_type": "missing"}, bg_mask

def inject_wrong_component(img, comp, comps, rng):
    H, W = img.shape[:2]
    ax1,ay1,ax2,ay2 = comp.xyxy(W, H)
    if ax2<=ax1 or ay2<=ay1: return img, None
    tw, th = ax2-ax1, ay2-ay1

    def _group(c):
        lbl, a = c.label.lower(), c.area()
        if any(x in lbl for x in ("resistor","capacitor","ferrite","inductor")):
            return "passive"
        if any(x in lbl for x in ("transistor","diode")):
            return "active_s"
        if "ic" in lbl:
            if a < 0.004: return "ic_0"
            elif a < 0.012: return "ic_1"
            elif a < 0.030: return "ic_2"
            else: return "ic_3"
        return f"other_{lbl[:8]}"

    tg = _group(comp)
    donors = []
    for c in comps:
        if c.id==comp.id or c.label.lower()==comp.label.lower() or c.conf<=0.15: continue
        if _group(c) != tg: continue
        bx1,by1,bx2,by2 = c.xyxy(W,H)
        if bx2<=bx1 or by2<=by1: continue
        ratio = max(c.area(),comp.area()) / max(min(c.area(),comp.area()),1e-9)
        if ratio > 4.0: continue
        donors.append((c, bx1,by1,bx2,by2, ratio))

    if not donors: return img, None

    donors.sort(key=lambda x: x[5])
    top = donors[:max(1, len(donors)//2+1)]
    donor,bx1,by1,bx2,by2,ratio = rng.choice(top)

    donor_patch = img[by1:by2, bx1:bx2].copy()
    orig_patch  = img[ay1:ay2, ax1:ax2].copy()
    paste = cv2.resize(donor_patch, (tw,th), interpolation=cv2.INTER_LINEAR)

    if ratio > 1.6:
        off_y = rng.randint(-max(1,int(th*0.20)), max(1,int(th*0.20)))
        off_x = rng.randint(-max(1,int(tw*0.20)), max(1,int(tw*0.20)))
        dy1=max(0,ay1+off_y); dy2=min(H,ay2+off_y)
        dx1=max(0,ax1+off_x); dx2=min(W,ax2+off_x)
        sy1=dy1-(ay1+off_y); sy2=sy1+(dy2-dy1)
        sx1=dx1-(ax1+off_x); sx2=sx1+(dx2-dx1)
        if dy2>dy1 and dx2>dx1:
            img[dy1:dy2, dx1:dx2] = paste[sy1:sy2, sx1:sx2]
    else:
        img[ay1:ay2, ax1:ax2] = paste

    if cv2.absdiff(orig_patch, img[ay1:ay2, ax1:ax2]).mean() < 8.0:
        img[ay1:ay2, ax1:ax2] = orig_patch
        return img, None

    return img, {"defect_type": "wrong_component", "found_label": donor.label}

def inject_wrong_polarity(img, comp):
    H,W = img.shape[:2]
    x1,y1,x2,y2 = comp.xyxy(W, H)
    if x2<=x1 or y2<=y1: return img, None
    patch   = img[y1:y2, x1:x2].copy()
    flipped = cv2.rotate(patch, cv2.ROTATE_180)
    if cv2.absdiff(patch, flipped).mean() < 8.0: return img, None
    img[y1:y2, x1:x2] = flipped
    return img, {"defect_type": "wrong_polarity"}

def inject_misaligned(img, comp, rng):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = comp.xyxy(W, H)
    if x2 <= x1 or y2 <= y1: return img, None
    patch = img[y1:y2, x1:x2].copy()
    ph, pw = patch.shape[:2]
    if ph < 6 or pw < 6: return img, None

    angle = rng.uniform(15, 45) * rng.choice([-1, 1])
    center = (pw // 2, ph // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(patch, M, (pw, ph), borderMode=cv2.BORDER_REPLICATE)

    mask = np.ones((ph, pw), dtype=np.float32)
    mask_rot = cv2.warpAffine(mask, M, (pw, ph), borderValue=0.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_rot = cv2.erode(mask_rot, kernel, iterations=1)
    mask_rot = cv2.GaussianBlur(mask_rot, (5, 5), 0)
    mask_3ch = mask_rot[:, :, np.newaxis]

    blended = (rotated.astype(np.float32) * mask_3ch +
               patch.astype(np.float32) * (1.0 - mask_3ch))
    img[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    if cv2.absdiff(patch, img[y1:y2, x1:x2]).mean() < 5.0:
        img[y1:y2, x1:x2] = patch
        return img, None

    return img, {"defect_type": "misaligned", "angle": round(angle, 1)}

DEFECT_MIX = {"missing":0.35, "wrong_component":0.30, "wrong_polarity":0.20, "misaligned":0.15}
DTYPES     = list(DEFECT_MIX.keys())
DPROBS     = [DEFECT_MIX[k] for k in DTYPES]

def create_defects(golden_img, golden_comps, n_defects, rng):
    img = golden_img.copy()
    wc  = copy.deepcopy(golden_comps)

    available = list(range(len(wc)))
    rng.shuffle(available)
    victims   = available[:min(n_defects, len(available))]

    log      = []
    used_ids = set()
    cached_mask = None

    for vi in victims:
        c     = wc[vi]
        dtype = rng.choices(DTYPES, weights=DPROBS, k=1)[0]

        if dtype == "wrong_polarity" and not c.is_polarized():
            others = [j for j in available if wc[j].is_polarized() and j not in used_ids and j != vi]
            if others:
                vi = rng.choice(others); c = wc[vi]
            else:
                dtype = "missing"

        if vi in used_ids: continue
        used_ids.add(vi)
        meta = None

        if dtype == "missing":
            img, meta, cached_mask = inject_missing(img, c, wc, cached_mask)
        elif dtype == "wrong_component":
            img, meta = inject_wrong_component(img, c, wc, rng)
            if not meta:
                img, meta, cached_mask = inject_missing(img, c, wc, cached_mask)
        elif dtype == "wrong_polarity":
            img, meta = inject_wrong_polarity(img, c)
            if not meta:
                img, meta, cached_mask = inject_missing(img, c, wc, cached_mask)
        elif dtype == "misaligned":
            img, meta = inject_misaligned(img, c, rng)
            if not meta:
                img, meta, cached_mask = inject_missing(img, c, wc, cached_mask)

        if meta:
            row = {"component_id": c.id, "original_label": c.label,
                   "injected_type": meta["defect_type"]}
            for k, v in meta.items():
                if k != "defect_type":
                    row[k] = v
            log.append(row)

    return img, log

# ═════════════════════════════════════════════════════════════════════════════
# 4.  AOI INFERENCE ENGINE
# ═════════════════════════════════════════════════════════════════════════════
_MIN_DIFF_THR    = 8.0
_MIN_POS_TOL     = 0.010
_AOI_MATCH_R     = 0.06
_AOI_DETECT_BAND = 2.0

# ── Disambiguation thresholds ────────────────────────────────────────────────
_SSIM_MISSING_MAX       = 0.32
_SSIM_WC_DET_MIN        = 0.32
_SSIM_WC_NODET          = 0.55
_VAR_FLAT               = 5.0
_VAR_WC_CONFIRM         = 80.0
_TMPL_WC_MIN            = 0.65
_NCC_POLARITY_B         = 0.05
_NCC_POLARITY_IC        = 0.28
_NCC_POLARITY_C         = 0.38
_NCC_POLARITY_STRONG    = 0.95
_NCC_POLARITY_SSIM_NEG  = -0.33
_SSIM_POLAR_STRUCT      = 0.28
_TMPL_POLAR_STRUCT      = 0.25
_NCC_POLAR_STRUCT_MIN_DELTA    = 0.25
_POLAR_CROSS_RESCUE_TMPL       = 0.20
_POLAR_CROSS_RESCUE_NCC_MIN    = 0.50
_SSIM_POLAR_CROSS_RESCUE       = 0.45

# ── v3.9 new thresholds ───────────────────────────────────────────────────────
# Fix N: Gradient Orientation Histogram polarity
_GOH_POLAR_BINS          = 12      # 30° bins over [−π, π]
_GOH_POLAR_MIN_DELTA     = 0.08    # gap: WCOM max ≈0.06, WPOL min ≈0.15
_GOH_ASYM_VOTE_BAND_LO   = 0.05   # uncertain band lower bound
_GOH_ASYM_VOTE_BAND_HI   = 0.10   # uncertain band upper bound
_ASYM_POLAR_MIN          = 0.025  # minimum opposition score for asym vote

# Fix O: Edge-density rescue
_EDGE_DENSITY_WC_MIN     = 0.045  # Gaussian fill: 0.00–0.01; donor: 0.04–0.18

# Fix P: Two-pass misalignment sweep
_MISALIGN_RECOVER_NCC_MIN    = 0.60
_MISALIGN_RECOVER_GAIN_MIN   = 0.06
_MISALIGN_RECOVER_ANGLE_MIN  = 10.0
_MISALIGN_FINE_WINDOW        = 8    # ±8° around coarse best
_MISALIGN_FINE_STEP          = 1    # 1° resolution

_POLARIZED = {
    "ic (u)","ic (ic)","transistor (q)","transistor (qa)",
    "diode (d)","led","capacitor (c)","cap array (cra)",
}

_DTYPE_ABBR = {
    "missing":         "MISS",
    "wrong_component": "WCOM",
    "wrong_polarity":  "WPOL",
    "misaligned":      "MALI",
}

# ═════════════════════════════════════════════════════════════════════════════
# 4a.  SIGNAL EXTRACTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _aoi_region_diff(diff_gray, c, W, H):
    x1,y1,x2,y2 = c.xyxy(W, H)
    if x2<=x1 or y2<=y1: return 0.0
    pad = max(1, int(min(x2-x1,y2-y1)*0.08))
    r   = diff_gray[y1+pad:y2-pad, x1+pad:x2-pad]
    return float(r.mean()) if r.size > 0 else 0.0

def _aoi_hungarian_match(golden, dets, max_dist=_AOI_MATCH_R):
    if not golden: return {}
    if not dets:   return {g.id: None for g in golden}
    G, D = len(golden), len(dets)
    cost = np.full((G,D), 1e9, dtype=np.float64)
    for i,gc in enumerate(golden):
        for j,dc in enumerate(dets):
            if dc.label.lower() != gc.label.lower(): continue
            d = math.hypot(gc.cx-dc.cx, gc.cy-dc.cy)
            if d <= max_dist: cost[i,j] = d
    res = {g.id: None for g in golden}
    if HAS_SCIPY:
        gi,di = linear_sum_assignment(cost)
        for g,d in zip(gi,di):
            if cost[g,d] < 1e8: res[golden[g].id] = dets[d].id
    else:
        used = set()
        for gc in sorted(golden, key=lambda x: x.id):
            best, bd = None, max_dist
            for dc in dets:
                if dc.id in used or dc.label.lower()!=gc.label.lower(): continue
                d = math.hypot(gc.cx-dc.cx, gc.cy-dc.cy)
                if d < bd: best, bd = dc, d
            if best: res[gc.id] = best.id; used.add(best.id)
    return res

def _ncc_polarity(golden_img, test_img, gc, W, H, margin=0.05,
                  g_blurred=None, t_blurred=None):
    if g_blurred is None:
        g_sm = cv2.resize(golden_img, (W,H), interpolation=cv2.INTER_AREA)
        g_sm = cv2.GaussianBlur(g_sm,(3,3),0)
    else:
        g_sm = g_blurred

    if t_blurred is None:
        t_sm = cv2.resize(test_img, (W,H), interpolation=cv2.INTER_AREA)
        t_sm = cv2.GaussianBlur(t_sm,(3,3),0)
    else:
        t_sm = t_blurred

    def _zncc(a,b):
        a=a.flatten()-a.mean(); b=b.flatten()-b.mean()
        d=np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.dot(a,b)/d) if d>1e-6 else 0.0

    best_delta = -1.0
    for crop_frac in (1.0, 0.80, 0.60):
        hw=gc.w*crop_frac/2; hh=gc.h*crop_frac/2
        sx1=max(0,int((gc.cx-hw)*W)); sx2=min(W,int((gc.cx+hw)*W))
        sy1=max(0,int((gc.cy-hh)*H)); sy2=min(H,int((gc.cy+hh)*H))
        if sx2-sx1<4 or sy2-sy1<4: continue
        gp = cv2.cvtColor(g_sm[sy1:sy2,sx1:sx2],cv2.COLOR_BGR2GRAY).astype(np.float32)
        tp = cv2.cvtColor(t_sm[sy1:sy2,sx1:sx2],cv2.COLOR_BGR2GRAY).astype(np.float32)
        if gp.size<9: continue
        base = _zncc(gp,tp)
        for rot in (cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            delta = _zncc(cv2.rotate(gp.astype(np.uint8),rot).astype(np.float32), tp) - base
            if delta > best_delta: best_delta = delta
    return best_delta > margin, best_delta

def _patch_variance(img, gc, W, H):
    hw=gc.w*0.20; hh=gc.h*0.20
    x1=max(0,int((gc.cx-hw)*W)); y1=max(0,int((gc.cy-hh)*H))
    x2=min(W,int((gc.cx+hw)*W)); y2=min(H,int((gc.cy+hh)*H))
    if x2-x1<2 or y2-y1<2: return 0.0
    return float(np.var(cv2.cvtColor(img[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)))

def _patch_ssim(img_a, img_b, gc, W, H):
    hw=gc.w*0.40; hh=gc.h*0.40
    x1=max(0,int((gc.cx-hw)*W)); y1=max(0,int((gc.cy-hh)*H))
    x2=min(W,int((gc.cx+hw)*W)); y2=min(H,int((gc.cy+hh)*H))
    if x2-x1<4 or y2-y1<4: return 1.0
    pa=cv2.cvtColor(img_a[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY).astype(np.float32)
    pb=cv2.cvtColor(img_b[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY).astype(np.float32)
    if pa.shape!=pb.shape:
        pb=cv2.resize(pb.astype(np.uint8),(pa.shape[1],pa.shape[0])).astype(np.float32)
    C1=(0.01*255)**2; C2=(0.03*255)**2
    mu_a=float(pa.mean()); mu_b=float(pb.mean())
    sig_a=float(pa.std());  sig_b=float(pb.std())
    cov=float(np.mean((pa-mu_a)*(pb-mu_b)))
    num=(2*mu_a*mu_b+C1)*(2*cov+C2)
    den=(mu_a**2+mu_b**2+C1)*(sig_a**2+sig_b**2+C2)
    return float(num/den) if abs(den)>1e-9 else 0.0

def _patch_tmpl_score(img_golden, img_test, gc, W, H):
    hw=gc.w*0.35; hh=gc.h*0.35
    x1=max(0,int((gc.cx-hw)*W)); y1=max(0,int((gc.cy-hh)*H))
    x2=min(W,int((gc.cx+hw)*W)); y2=min(H,int((gc.cy+hh)*H))
    if x2-x1<4 or y2-y1<4: return 0.0
    tmpl = cv2.cvtColor(img_golden[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)
    patch= cv2.cvtColor(img_test  [y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)
    if tmpl.shape != patch.shape:
        patch = cv2.resize(patch, (tmpl.shape[1], tmpl.shape[0]))
    if tmpl.shape[0]<2 or tmpl.shape[1]<2: return 0.0
    try:
        res = cv2.matchTemplate(patch.astype(np.float32),
                                tmpl.astype(np.float32),
                                cv2.TM_CCOEFF_NORMED)
        return float(res.max())
    except Exception:
        return 0.0

# ── v3.9 Fix N: Gradient Orientation Histogram polarity ──────────────────────
def _goh_polarity(golden_img, test_img, gc, W, H):
    """
    Gradient Orientation Histogram (GOH) polarity check.

    A 180° rotation reverses every edge direction. The GOH of the rotated
    golden patch (histogram rolled by N//2 bins) should correlate strongly
    with the test patch's GOH if the component IS rotated.

    NCC is intensity-based and cannot detect this directional flip for
    near-symmetric components. GOH catches it because it explicitly encodes
    *direction*, not just magnitude.

    Returns (is_flipped: bool, goh_delta: float)
      goh_delta = corr(rolled_golden_GOH, test_GOH) − corr(golden_GOH, test_GOH)
      Positive and ≥ _GOH_POLAR_MIN_DELTA → likely rotated → WPOL.
    """
    hw = gc.w * 0.35; hh = gc.h * 0.35
    x1 = max(0, int((gc.cx - hw) * W)); y1 = max(0, int((gc.cy - hh) * H))
    x2 = min(W, int((gc.cx + hw) * W)); y2 = min(H, int((gc.cy + hh) * H))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return False, 0.0

    def _build_goh(img_bgr, y1, y2, x1, x2):
        gray  = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
        gx    = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy    = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag   = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.arctan2(gy, gx)  # −π to +π
        hist, _ = np.histogram(angle,
                               bins=_GOH_POLAR_BINS,
                               range=(-math.pi, math.pi),
                               weights=mag)
        total = hist.sum()
        return hist / (total + 1e-6)

    g_hist = _build_goh(golden_img, y1, y2, x1, x2)
    t_hist = _build_goh(test_img,   y1, y2, x1, x2)

    half = _GOH_POLAR_BINS // 2  # 6 bins = 180°
    g_hist_rot = np.roll(g_hist, half)

    def _ncc_hist(a, b):
        a = a - a.mean(); b = b - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-6 else 0.0

    corr_orig = _ncc_hist(g_hist, t_hist)
    corr_rot  = _ncc_hist(g_hist_rot, t_hist)
    goh_delta = corr_rot - corr_orig

    return goh_delta >= _GOH_POLAR_MIN_DELTA, goh_delta

# ── v3.9 Fix N (tie-breaker): Intensity asymmetry opposition score ────────────
def _intensity_asymmetry_score(golden_img, test_img, gc, W, H):
    """
    Top/bottom + left/right intensity asymmetry opposition score.

    A polarity marker (capacitor stripe, IC pin-1 chamfer) creates a
    directional bias in the golden patch. Under 180° rotation that bias
    reverses sign. The opposition score is positive when golden and test
    biases oppose each other — consistent with rotation.

    Used only as a tie-breaker when GOH delta is in the uncertain band
    [_GOH_ASYM_VOTE_BAND_LO, _GOH_ASYM_VOTE_BAND_HI].

    Returns float: opposition score in [−1, 1].
      Positive → asymmetries oppose → vote WPOL.
      ≥ _ASYM_POLAR_MIN → cast positive vote.
    """
    hw = gc.w * 0.28; hh = gc.h * 0.28
    x1 = max(0, int((gc.cx - hw) * W)); y1 = max(0, int((gc.cy - hh) * H))
    x2 = min(W, int((gc.cx + hw) * W)); y2 = min(H, int((gc.cy + hh) * H))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return 0.0

    def _bias(img_bgr, y1, y2, x1, x2):
        gray = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        mh, mw = h // 2, w // 2
        top = float(gray[:mh, :].mean()); bot = float(gray[mh:, :].mean())
        lft = float(gray[:, :mw].mean()); rgt = float(gray[:, mw:].mean())
        tb  = (top - bot) / (top + bot + 1e-6)
        lr  = (lft - rgt) / (lft + rgt + 1e-6)
        return tb, lr

    g_tb, g_lr = _bias(golden_img, y1, y2, x1, x2)
    t_tb, t_lr = _bias(test_img,   y1, y2, x1, x2)

    tb_opp = -g_tb * t_tb   # positive when signs oppose
    lr_opp = -g_lr * t_lr
    return float((tb_opp + lr_opp) / 2.0)

# ── v3.9 Fix O: Edge density (wrong_component rescue from missing path) ────────
def _edge_density(img, gc, W, H):
    """
    Fraction of pixels that are Canny edges within the component crop.

    Gaussian fill: density 0.00–0.01 (no structural content).
    Donor component (even after low-contrast resize): 0.04–0.18.

    Used in Zone C Step 4 as a third guard before emitting "missing":
    if density ≥ _EDGE_DENSITY_WC_MIN classify as wrong_component.

    Returns float: edge_density in [0, 1].
    """
    hw = gc.w * 0.32; hh = gc.h * 0.32
    x1 = max(0, int((gc.cx - hw) * W)); y1 = max(0, int((gc.cy - hh) * H))
    x2 = min(W, int((gc.cx + hw) * W)); y2 = min(H, int((gc.cy + hh) * H))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    patch = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(patch, (3, 3), 0)
    # Low Canny thresholds to catch faint edges in resized donors
    edges = cv2.Canny(blur, 20, 60)
    return float(np.count_nonzero(edges)) / (edges.size + 1e-6)

# ── v3.9 Fix P: Two-pass misalignment check ───────────────────────────────────
def _misalignment_check(golden_img, test_img, gc, W, H):
    """
    Two-pass rotation recovery for misalignment detection (v3.9c).

    Pass 1 (coarse): 5° steps over [−45, +45].
    Pass 2 (fine):   1° steps over [coarse_best ± 8°].

    The fine pass closes the gap at boundary injection angles (10°–20°)
    where the coarse-only sweep undersamples the NCC recovery peak.

    Returns (is_misaligned: bool, best_angle: float, reason_str: str)
    """
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

    # ── Pass 1: coarse 5° sweep ───────────────────────────────────────────
    coarse_best_ncc   = -1.0
    coarse_best_angle = 0
    for angle in range(-45, 50, 5):
        if angle == 0:
            continue
        ncc = _recover_ncc(angle)
        if ncc > coarse_best_ncc:
            coarse_best_ncc   = ncc
            coarse_best_angle = angle

    # ── Pass 2: fine 1° sweep around coarse best ─────────────────────────
    fine_best_ncc   = coarse_best_ncc
    fine_best_angle = coarse_best_angle
    fine_lo = coarse_best_angle - _MISALIGN_FINE_WINDOW
    fine_hi = coarse_best_angle + _MISALIGN_FINE_WINDOW + 1
    for angle in range(fine_lo, fine_hi, _MISALIGN_FINE_STEP):
        if angle == 0 or angle == coarse_best_angle:
            continue
        ncc = _recover_ncc(angle)
        if ncc > fine_best_ncc:
            fine_best_ncc   = ncc
            fine_best_angle = angle

    best_ncc     = fine_best_ncc
    best_angle   = fine_best_angle
    unrot_ncc    = _recover_ncc(0)
    recover_gain = best_ncc - unrot_ncc

    reason = (f"recov_ncc={best_ncc:.2f} gain={recover_gain:.2f} "
              f"(base={unrot_ncc:.2f}) at {best_angle}°")

    if (best_ncc   >= _MISALIGN_RECOVER_NCC_MIN
            and recover_gain >= _MISALIGN_RECOVER_GAIN_MIN
            and abs(best_angle) >= _MISALIGN_RECOVER_ANGLE_MIN):
        return True, float(best_angle), (
            f"rotation_recovered ncc={best_ncc:.2f} "
            f"gain={recover_gain:.2f} at {best_angle}°"
        )

    return False, float(best_angle), reason

def _same_det_is_local(dc, gc, golden_by_label):
    dist_to_gc=math.hypot(dc.cx-gc.cx, dc.cy-gc.cy)
    for og in golden_by_label.get(dc.label.lower(),[]):
        if og.id==gc.id: continue
        if math.hypot(dc.cx-og.cx, dc.cy-og.cy) < dist_to_gc*0.85:
            return False
    return True

def _aoi_calibrate(model, golden_img, golden, conf, n_runs=2):
    rng = random.Random(42); np.random.seed(42)
    H,W = golden_img.shape[:2]
    jitters=[]; comp_diffs={c.id:[] for c in golden}
    for _ in range(n_runs):
        a=rng.uniform(0.93,1.07); b=rng.uniform(-7,7)
        aug=np.clip(golden_img.astype(np.float32)*a+b,0,255).astype(np.uint8)
        dets=detect(model,aug,conf)
        match=_aoi_hungarian_match(golden,dets)
        det_by_id={d.id:d for d in dets}
        for gc in golden:
            did=match.get(gc.id)
            if did is not None:
                dc=det_by_id[did]
                jitters.append(math.hypot(dc.cx-gc.cx,dc.cy-gc.cy))
        diff_gray=cv2.GaussianBlur(
            cv2.cvtColor(cv2.absdiff(golden_img,aug),cv2.COLOR_BGR2GRAY).astype(np.float32),
            (5,5),0)
        for gc in golden:
            comp_diffs[gc.id].append(_aoi_region_diff(diff_gray,gc,W,H))

    pos_tol = _MIN_POS_TOL
    if jitters:
        arr=np.array(jitters)
        q75,q25=np.percentile(arr,[75,25]); iqr=q75-q25
        pos_tol=float(np.clip(np.median(arr)+2.5*iqr, _MIN_POS_TOL, 0.055))

    per_thr={}
    for gc in golden:
        samples=comp_diffs.get(gc.id,[])
        if len(samples)>=2:
            arr=np.array(samples)
            thr=float(np.clip(arr.mean()+3.0*arr.std(),_MIN_DIFF_THR,80.0))
        else:
            thr=_MIN_DIFF_THR*2.0
        per_thr[gc.id]=thr
    return pos_tol, per_thr

# ═════════════════════════════════════════════════════════════════════════════
# 4b.  BOARD INSPECTION
# ═════════════════════════════════════════════════════════════════════════════
def _aoi_check_board(golden, golden_img, test_img, dets, pos_tol, per_thr, debug=False):
    """
    Single-pass pixel-primary defect detection.
    Returns list of {component_id, defect_type, expected_label, ...}
    If debug=True, also returns a parallel list of diagnostic dicts.
    """
    defects=[]
    diag_rows=[] if debug else None
    H,W = golden_img.shape[:2]
    DIFF_MAX=1280; ds=min(1.0,DIFF_MAX/max(H,W))
    if ds < 1.0:
        dW,dH=int(W*ds),int(H*ds)
        g_sm=cv2.resize(golden_img,(dW,dH),interpolation=cv2.INTER_AREA)
        t_sm=cv2.resize(test_img,  (dW,dH),interpolation=cv2.INTER_AREA)
    else:
        dW,dH,g_sm,t_sm=W,H,golden_img,test_img

    # Pre-blur once; reused by every _ncc_polarity call this board
    g_sm_blur = cv2.GaussianBlur(g_sm, (3,3), 0)
    t_sm_blur = cv2.GaussianBlur(t_sm, (3,3), 0)

    diff_gray=cv2.GaussianBlur(
        cv2.cvtColor(cv2.absdiff(g_sm,t_sm),cv2.COLOR_BGR2GRAY).astype(np.float32),
        (5,5),0)

    MIN_CONF=0.22; LOW_MULT=0.50; HIGH_MULT=2.0; MISALIGN_T=1.6
    detect_tol=pos_tol*_AOI_DETECT_BAND

    real_dets=[d for d in dets if d.conf>=MIN_CONF]
    _golden_by_label={}
    for g in golden: _golden_by_label.setdefault(g.label.lower(),[]).append(g)

    def _slot_dets(gc):
        _hd=math.hypot(gc.w,gc.h)/2
        r=min(max(_AOI_MATCH_R,_hd*0.70),_AOI_MATCH_R*1.8)
        same,cross=[],[]
        for dc in real_dets:
            d=math.hypot(dc.cx-gc.cx,dc.cy-gc.cy)
            if d>r: continue
            (same if dc.label.lower()==gc.label.lower() else cross).append((d,dc))
        same.sort(key=lambda x:x[0]); cross.sort(key=lambda x:x[0])
        return same,cross

    def _owns_slot(dc,gc):
        dist=math.hypot(dc.cx-gc.cx,dc.cy-gc.cy)
        if dist>detect_tol: return False
        own=_golden_by_label.get(dc.label.lower(),[])
        nearest=min((math.hypot(dc.cx-og.cx,dc.cy-og.cy) for og in own),default=float('inf'))
        return dist < nearest*0.90

    def _rotation_rescue(gc, same, cross):
        is_mali, angle_d, mali_reason = _misalignment_check(golden_img, test_img, gc, W, H)
        if not is_mali:
            return None
        found_label = same[0][1].label if same else (cross[0][1].label if cross else gc.label)
        defect = {"component_id":gc.id,"defect_type":"misaligned",
                  "expected_label":gc.label,"found_label":found_label}
        return defect, angle_d, mali_reason

    for gc in golden:
        comp_thr=per_thr.get(gc.id,_MIN_DIFF_THR)
        rdiff=_aoi_region_diff(diff_gray,gc,dW,dH)
        low_thr=comp_thr*LOW_MULT; high_thr=comp_thr*HIGH_MULT
        same,cross=_slot_dets(gc)
        is_polar = gc.label.lower() in _POLARIZED

        d_info = {"cid":gc.id,"lbl":gc.label,"rdiff":round(rdiff,2),
                  "low_thr":round(low_thr,2),"high_thr":round(high_thr,2),
                  "n_same":len(same),"n_cross":len(cross),
                  "zone":"?","path":"?","decision":"none",
                  "tmpl":None,"ssim":None,"test_var":None,
                  "goh_delta":None,"asym_score":None,"edge_density":None
                  } if debug else None

        # ── ZONE A: looks identical ──────────────────────────────────────────
        if rdiff < low_thr:
            if d_info is not None: d_info["zone"]="A"
            if is_polar and same:
                flipped,delta=_ncc_polarity(golden_img,test_img,gc,dW,dH,
                                            margin=_NCC_POLARITY_B,
                                            g_blurred=g_sm_blur,t_blurred=t_sm_blur)
                if flipped:
                    defects.append({"component_id":gc.id,"defect_type":"wrong_polarity",
                                    "expected_label":gc.label})
                    if d_info is not None:
                        d_info.update({"path":"ncc_polarity","decision":"wrong_polarity",
                                       "ncc_delta":round(delta,3)})
            if d_info is not None:
                if d_info["decision"]=="none": d_info.update({"path":"clean","decision":"ok"})
                diag_rows.append(d_info)
            continue

        # ── ZONE B: moderate change ──────────────────────────────────────────
        if rdiff < high_thr:
            if d_info is not None: d_info["zone"]="B"
            mali_hit = _rotation_rescue(gc, same, cross)
            if mali_hit is not None:
                defect, angle_d, mali_reason = mali_hit
                defects.append(defect)
                if d_info is not None:
                    d_info.update({"path":f"zoneB_misalign({mali_reason})",
                                   "decision":"misaligned",
                                   "angle_delta":round(angle_d,1)})
                    diag_rows.append(d_info)
                continue
            if same:
                dist,tc=same[0]
                cross_owned=False
                for _,dc in cross:
                    if _owns_slot(dc,gc):
                        defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                        "expected_label":gc.label,"found_label":dc.label})
                        cross_owned=True
                        if d_info is not None:
                            d_info.update({"path":"cross_owned","decision":"wrong_component",
                                           "cross_lbl":dc.label,"cross_dist":round(math.hypot(dc.cx-gc.cx,dc.cy-gc.cy),4)})
                        break
                if not cross_owned:
                    if dist>detect_tol*MISALIGN_T and rdiff>comp_thr*0.35:
                        defects.append({"component_id":gc.id,"defect_type":"misaligned",
                                        "expected_label":gc.label})
                        if d_info is not None: d_info.update({"path":"misalign_offset","decision":"misaligned"})
                    elif dist<=detect_tol*MISALIGN_T:
                        is_mali, angle_d, mali_reason = _misalignment_check(
                            golden_img, test_img, gc, W, H)
                        if is_mali:
                            defects.append({"component_id":gc.id,"defect_type":"misaligned",
                                            "expected_label":gc.label,"found_label":tc.label})
                            if d_info is not None:
                                d_info.update({"path":f"misalign_rot({mali_reason})",
                                               "decision":"misaligned","angle_delta":round(angle_d,1)})
                    elif is_polar:
                        flipped,delta=_ncc_polarity(golden_img,test_img,gc,dW,dH,
                                                    margin=_NCC_POLARITY_B,
                                                    g_blurred=g_sm_blur,t_blurred=t_sm_blur)
                        if flipped:
                            defects.append({"component_id":gc.id,"defect_type":"wrong_polarity",
                                            "expected_label":gc.label,"found_label":tc.label})
                            if d_info is not None:
                                d_info.update({"path":"ncc_polarity","decision":"wrong_polarity",
                                               "ncc_delta":round(delta,3)})
                        else:
                            # v3.5 Fix L: polar slot, same det, NCC no flip → wrong_component
                            defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                            "expected_label":gc.label,"found_label":tc.label})
                            if d_info is not None:
                                d_info.update({"path":"zone_b_polar_noflip","decision":"wrong_component",
                                               "ncc_delta":round(delta,3)})
                    else:
                        ssim_v=_patch_ssim(g_sm,t_sm,gc,dW,dH)
                        if d_info is not None: d_info["ssim"]=round(ssim_v,3)
                        if ssim_v < _SSIM_MISSING_MAX:
                            defects.append({"component_id":gc.id,"defect_type":"missing",
                                            "expected_label":gc.label})
                            if d_info is not None: d_info.update({"path":"zone_b_ssim_low","decision":"missing"})
                        elif (dist<detect_tol and _same_det_is_local(tc,gc,_golden_by_label)
                              and ssim_v<_SSIM_WC_DET_MIN+0.15):
                            defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                            "expected_label":gc.label,"found_label":tc.label})
                            if d_info is not None: d_info.update({"path":"zone_b_ssim_wc","decision":"wrong_component"})
                        elif ssim_v >= _SSIM_WC_DET_MIN+0.15 and rdiff > comp_thr*0.35:
                            defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                            "expected_label":gc.label,"found_label":tc.label})
                            if d_info is not None: d_info.update({"path":"zone_b_ssim_high_wc","decision":"wrong_component"})
            else:
                # No same-label det in radius
                if is_polar:
                    flipped,delta=_ncc_polarity(golden_img,test_img,gc,dW,dH,
                                                margin=0.40,
                                                g_blurred=g_sm_blur,t_blurred=t_sm_blur)
                    if flipped:
                        fl=cross[0][1].label if cross else gc.label
                        defects.append({"component_id":gc.id,"defect_type":"wrong_polarity",
                                        "expected_label":gc.label,"found_label":fl})
                        if d_info is not None:
                            d_info.update({"path":"ncc_nosame","decision":"wrong_polarity",
                                           "ncc_delta":round(delta,3)})
                        if d_info is not None: diag_rows.append(d_info)
                        continue
                for _,dc in cross:
                    if _owns_slot(dc,gc):
                        defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                        "expected_label":gc.label,"found_label":dc.label})
                        if d_info is not None:
                            d_info.update({"path":"cross_nosame","decision":"wrong_component",
                                           "cross_lbl":dc.label})
                        break
                else:
                    defects.append({"component_id":gc.id,"defect_type":"missing",
                                    "expected_label":gc.label})
                    if d_info is not None: d_info.update({"path":"nosame_nocross","decision":"missing"})
            if d_info is not None:
                if d_info["decision"]=="none": d_info.update({"path":"zone_b_no_decision","decision":"ok"})
                diag_rows.append(d_info)
            continue

        # ── ZONE C: strong change ────────────────────────────────────────────
        if d_info is not None: d_info["zone"]="C"

        ssim_v   = _patch_ssim(g_sm, t_sm, gc, dW, dH)
        test_var = _patch_variance(t_sm, gc, dW, dH)
        tmpl_v   = _patch_tmpl_score(g_sm, t_sm, gc, dW, dH)
        if d_info is not None:
            d_info.update({"ssim":round(ssim_v,3),
                           "test_var":round(test_var,1),
                           "tmpl":round(tmpl_v,3)})

        slot_is_flat   = test_var < _VAR_FLAT and ssim_v < _SSIM_MISSING_MAX
        var_structured = test_var >= _VAR_WC_CONFIRM
        cross_owns     = any(_owns_slot(dc, gc) for _, dc in cross)

        # ── Step 1: flat slot → definitely missing ───────────────────────────
        if slot_is_flat and not cross_owns and not same:
            defects.append({"component_id":gc.id,"defect_type":"missing",
                            "expected_label":gc.label})
            if d_info is not None:
                d_info.update({"path":f"flat(var={test_var:.1f})","decision":"missing"})
            if d_info is not None: diag_rows.append(d_info)
            continue

        # ── Step 1.5: rotation-based misalignment rescue ─────────────────────
        mali_hit = _rotation_rescue(gc, same, cross)
        if mali_hit is not None:
            defect, angle_d, mali_reason = mali_hit
            defects.append(defect)
            if d_info is not None:
                d_info.update({"path":f"zoneC_misalign({mali_reason})",
                               "decision":"misaligned",
                               "angle_delta":round(angle_d,1)})
                diag_rows.append(d_info)
            continue

        # ── Step 2: polarity check ───────────────────────────────────────────
        flipped, delta = False, -1.0
        if is_polar:
            is_ic = "ic" in gc.label.lower()
            ncc_margin = _NCC_POLARITY_IC if is_ic else _NCC_POLARITY_C
            flipped, delta = _ncc_polarity(golden_img, test_img, gc, dW, dH,
                                           margin=ncc_margin,
                                           g_blurred=g_sm_blur, t_blurred=t_sm_blur)
            accept_polarity = (flipped and
                               (delta > _NCC_POLARITY_STRONG or
                                ssim_v < _NCC_POLARITY_SSIM_NEG))

            # v3.1–v3.6 structural fallback (ssim + tmpl + delta gates)
            if not accept_polarity and delta >= _NCC_POLAR_STRUCT_MIN_DELTA:
                if ssim_v >= _SSIM_POLAR_STRUCT and tmpl_v >= _TMPL_POLAR_STRUCT:
                    accept_polarity = True
                    if d_info is not None:
                        d_info["path"] = "polar_struct_polarity"

            # ── v3.9 Fix N: GOH + asymmetry fallback ─────────────────────────
            # Consult when all NCC/structural gates have failed.
            # GOH captures directional edge asymmetry invisible to NCC.
            # Asymmetry vote is a tie-breaker in the uncertain GOH band.
            if not accept_polarity:
                goh_flip, goh_delta = _goh_polarity(golden_img, test_img, gc, W, H)

                if goh_delta >= _GOH_POLAR_MIN_DELTA:
                    # Strong GOH signal — accept directly
                    accept_polarity = True
                    if d_info is not None:
                        d_info["path"]      = "goh_polarity"
                        d_info["goh_delta"] = round(goh_delta, 3)

                elif _GOH_ASYM_VOTE_BAND_LO <= goh_delta < _GOH_ASYM_VOTE_BAND_HI:
                    # Uncertain band: consult asymmetry vote as tie-breaker
                    asym_score = _intensity_asymmetry_score(
                        golden_img, test_img, gc, W, H)
                    if asym_score >= _ASYM_POLAR_MIN:
                        accept_polarity = True
                        if d_info is not None:
                            d_info["path"]       = "goh_asym_polarity"
                            d_info["goh_delta"]  = round(goh_delta, 3)
                            d_info["asym_score"] = round(asym_score, 3)

            if accept_polarity:
                fl = same[0][1].label if same else (cross[0][1].label if cross else gc.label)
                defects.append({"component_id":gc.id,"defect_type":"wrong_polarity",
                                "expected_label":gc.label,"found_label":fl})
                path_label = (d_info.get("path","") if d_info and d_info.get("path","") in
                              ("polar_struct_polarity","goh_polarity","goh_asym_polarity")
                              else "ncc_zoneC")
                if d_info is not None:
                    d_info.update({"path": path_label, "decision":"wrong_polarity",
                                   "ncc_delta":round(delta,3)})
                if d_info is not None: diag_rows.append(d_info)
                continue

        # ── Step 2.5: rotation check for polarized components ────────────────
        if is_polar and same and not cross_owns:
            _mali_dist, _mali_dc = same[0]
            _mali_at_slot = _mali_dist < detect_tol
            _mali_local   = _same_det_is_local(_mali_dc, gc, _golden_by_label)
            if _mali_at_slot and _mali_local:
                is_mali, mali_metric, mali_reason = _misalignment_check(
                    golden_img, test_img, gc, W, H)
                if is_mali:
                    defects.append({"component_id":gc.id,"defect_type":"misaligned",
                                    "expected_label":gc.label,"found_label":_mali_dc.label})
                    if d_info is not None:
                        d_info.update({"path":f"zoneC_polar_misalign({mali_reason})",
                                       "decision":"misaligned"})
                    if d_info is not None: diag_rows.append(d_info)
                    continue

        # ── Step 3: cross-label det owns slot → wrong_component ──────────────
        if cross_owns:
            if (is_polar and flipped and var_structured
                    and ssim_v < _SSIM_POLAR_CROSS_RESCUE
                    and tmpl_v >= _POLAR_CROSS_RESCUE_TMPL
                    and delta > _POLAR_CROSS_RESCUE_NCC_MIN):
                fl = same[0][1].label if same else (cross[0][1].label if cross else gc.label)
                defects.append({"component_id":gc.id,"defect_type":"wrong_polarity",
                                "expected_label":gc.label,"found_label":fl})
                if d_info is not None:
                    d_info.update({"path":"cross_polar_rescue","decision":"wrong_polarity",
                                   "ncc_delta":round(delta,3)})
                if d_info is not None: diag_rows.append(d_info)
                continue
            slot_looks_missing = test_var < _VAR_FLAT and ssim_v < _SSIM_MISSING_MAX
            if slot_looks_missing:
                defects.append({"component_id":gc.id,"defect_type":"missing",
                                "expected_label":gc.label})
                if d_info is not None:
                    d_info.update({"path":"cross_zoneC_flat_override","decision":"missing"})
                if d_info is not None: diag_rows.append(d_info)
                continue
            for _,dc in cross:
                if _owns_slot(dc,gc):
                    defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                    "expected_label":gc.label,"found_label":dc.label})
                    if d_info is not None:
                        d_info.update({"path":"cross_zoneC","decision":"wrong_component",
                                       "cross_lbl":dc.label})
                    break
            if d_info is not None: diag_rows.append(d_info)
            continue

        # ── Step 4: MISSING vs WRONG_COMPONENT ───────────────────────────────
        if var_structured or ssim_v >= _SSIM_MISSING_MAX:
            if same:
                nearest_dist, nearest_dc = same[0]
                at_slot  = nearest_dist < detect_tol
                is_local = _same_det_is_local(nearest_dc, gc, _golden_by_label)
                if d_info is not None:
                    d_info.update({"same_dist":round(nearest_dist,4),
                                   "at_slot":at_slot,"is_local":is_local})

                if at_slot and is_local:
                    defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                    "expected_label":gc.label,
                                    "found_label":nearest_dc.label})
                    reason = (f"var_rescue({test_var:.0f})"
                              if (var_structured and ssim_v < _SSIM_MISSING_MAX)
                              else f"at_slot_ssim({ssim_v:.2f})")
                    if d_info is not None:
                        d_info.update({"path":reason,"decision":"wrong_component"})
                elif ssim_v >= _SSIM_WC_NODET or var_structured:
                    defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                    "expected_label":gc.label,
                                    "found_label":nearest_dc.label})
                    reason = (f"var_rescue({test_var:.0f})"
                              if (var_structured and ssim_v < _SSIM_MISSING_MAX)
                              else f"ssim_high_nodet({ssim_v:.2f})")
                    if d_info is not None:
                        d_info.update({"path":reason,"decision":"wrong_component"})
                else:
                    defects.append({"component_id":gc.id,"defect_type":"missing",
                                    "expected_label":gc.label})
                    if d_info is not None:
                        d_info.update({"path":f"neighbour_det_ssim({ssim_v:.2f})",
                                       "decision":"missing"})
            else:
                defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                "expected_label":gc.label,"found_label":"unknown"})
                reason = (f"var_rescue_nodet({test_var:.0f})"
                          if (var_structured and ssim_v < _SSIM_MISSING_MAX)
                          else f"ssim_only({ssim_v:.2f})")
                if d_info is not None:
                    d_info.update({"path":reason,"decision":"wrong_component"})
        else:
            # ── v3.9 Fix O: edge-density rescue ──────────────────────────────
            # Catches low-variance donors that fool the var/SSIM gates but
            # still contain structural edges after resize.
            edge_d = _edge_density(t_sm, gc, dW, dH)
            if edge_d >= _EDGE_DENSITY_WC_MIN:
                found_lbl = same[0][1].label if same else "unknown"
                defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                "expected_label":gc.label,"found_label":found_lbl})
                if d_info is not None:
                    d_info.update({"path":f"edge_rescue({edge_d:.3f})",
                                   "decision":"wrong_component",
                                   "edge_density":round(edge_d, 3)})
            else:
                defects.append({"component_id":gc.id,"defect_type":"missing",
                                "expected_label":gc.label})
                if d_info is not None:
                    d_info.update({"path":f"ssim_low({ssim_v:.2f})_var({test_var:.0f})",
                                   "decision":"missing"})

        if d_info is not None:
            if d_info["decision"]=="none": d_info.update({"path":"zone_c_no_decision","decision":"ok"})
            diag_rows.append(d_info)

    if debug:
        return defects, diag_rows
    return defects

# ═════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATOR
# ═════════════════════════════════════════════════════════════════════════════
COMPAT = {
    ("missing","wrong_component"), ("wrong_component","missing"),
    ("missing","wrong_polarity"),  ("wrong_polarity","missing"),
    ("missing","misaligned"),      ("misaligned","missing"),
    ("wrong_component","wrong_polarity"), ("wrong_polarity","wrong_component"),
    ("wrong_component","misaligned"),     ("misaligned","wrong_component"),
    ("wrong_polarity","misaligned"),      ("misaligned","wrong_polarity"),
}

def run_evaluation(args):
    global DTYPES, DPROBS

    print("\n" + "═"*72)
    print(_bold("  PCB DEFECT ACCURACY EVALUATOR  v3.9"))
    print("═"*72)

    if args.only_defect:
        DTYPES = [args.only_defect]
        DPROBS = [1.0]
    else:
        DTYPES = list(DEFECT_MIX.keys())
        DPROBS = [DEFECT_MIX[k] for k in DTYPES]

    print(f"\n[MODEL]  {args.model}")
    model = YOLO(args.model)

    golden_img = cv2.imread(args.golden)
    if golden_img is None:
        sys.exit(f"[ERROR] Cannot read golden image: {args.golden}")
    H,W = golden_img.shape[:2]
    print(f"[GOLDEN] {W}×{H}  {args.golden}")

    golden = detect(model, golden_img, conf=args.conf)
    if not golden:
        sys.exit("[ERROR] Zero components detected on golden image.")

    counts = defaultdict(int)
    for c in golden: counts[c.label] += 1
    print(f"[GOLDEN] {len(golden)} components detected:")
    for lbl,n in sorted(counts.items()):
        print(f"    {lbl:<30}  ×{n}")

    polarized = [c for c in golden if c.is_polarized()]
    print(f"\n  Polarized : {len(polarized)} ({', '.join(sorted({c.label for c in polarized})) or 'NONE'})")
    print(f"  Boards    : {args.boards}")
    print(f"  Defects   : 1–{args.defects} per board")
    print(f"  Mix       : {args.only_defect or 'default'}")
    print(f"\n  [THRESH] SSIM_MISSING_MAX={_SSIM_MISSING_MAX}  SSIM_WC_NODET={_SSIM_WC_NODET}")
    print(f"  [THRESH] VAR_FLAT={_VAR_FLAT}  VAR_WC_CONFIRM={_VAR_WC_CONFIRM}")
    print(f"  [THRESH] NCC_POLARITY_IC={_NCC_POLARITY_IC}  NCC_POLARITY_C={_NCC_POLARITY_C}  "
          f"NCC_POLARITY_STRONG={_NCC_POLARITY_STRONG}  NCC_POLARITY_SSIM_NEG={_NCC_POLARITY_SSIM_NEG}")
    print(f"  [THRESH] POLAR_STRUCT: ssim≥{_SSIM_POLAR_STRUCT} tmpl≥{_TMPL_POLAR_STRUCT}  "
          f"CROSS_RESCUE: ssim<{_SSIM_POLAR_CROSS_RESCUE} tmpl≥{_POLAR_CROSS_RESCUE_TMPL}")
    print(f"  [THRESH v3.9] GOH_MIN_DELTA={_GOH_POLAR_MIN_DELTA}  "
          f"ASYM_BAND=[{_GOH_ASYM_VOTE_BAND_LO},{_GOH_ASYM_VOTE_BAND_HI}]  "
          f"ASYM_MIN={_ASYM_POLAR_MIN}")
    print(f"  [THRESH v3.9] EDGE_DENSITY_WC_MIN={_EDGE_DENSITY_WC_MIN}  "
          f"MISALIGN_FINE_WINDOW=±{_MISALIGN_FINE_WINDOW}° step={_MISALIGN_FINE_STEP}°\n")

    print("[CAL]  Calibrating diff thresholds…")
    t0 = time.time()
    pos_tol, per_thr = _aoi_calibrate(model, golden_img, golden, args.conf, n_runs=2)
    detect_tol = pos_tol * _AOI_DETECT_BAND
    print(f"[CAL]  pos_tol={pos_tol:.4f}  detect_tol={detect_tol:.4f}  "
          f"({time.time()-t0:.1f}s)\n")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    total_inj  = 0
    l_tp=l_fp=l_fn = 0
    s_tp=s_fp=s_fn = 0
    confusion   = defaultdict(lambda: defaultdict(int))
    path_counts = defaultdict(int)
    label_stats = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})
    wc_fail_signals = []

    board_rows = []

    for b in range(args.boards):
        n = rng.randint(1, max(1, args.defects))
        test_img, inj_log = create_defects(golden_img, golden, n, rng)

        dets_test = detect(model, test_img, conf=args.conf)
        need_diag = args.debug or args.trace
        if need_diag:
            found, diag = _aoi_check_board(golden, golden_img, test_img, dets_test,
                                            pos_tol, per_thr, debug=True)
        else:
            found = _aoi_check_board(golden, golden_img, test_img, dets_test,
                                     pos_tol, per_thr)
            diag = None

        inj_map   = {d["component_id"]: d for d in inj_log}
        found_map = {d["component_id"]: d for d in found}

        total_inj += len(inj_map)
        board_mm=board_fn=board_fp=0

        for cid, inj in inj_map.items():
            itype = inj["injected_type"]
            ilbl  = inj["original_label"]
            if cid in found_map:
                dtype = found_map[cid]["defect_type"]
                l_tp += 1
                confusion[itype][dtype] += 1
                if dtype == itype:
                    s_tp += 1
                    label_stats[ilbl]["tp"] += 1
                else:
                    s_fn += 1; s_fp += 1; board_mm += 1
                    label_stats[ilbl]["fn"] += 1
                    if itype == "wrong_component" and dtype == "missing" and diag:
                        for row in diag:
                            if row["cid"] == cid:
                                wc_fail_signals.append({
                                    "ssim": row.get("ssim"),
                                    "tmpl": row.get("tmpl"),
                                    "var":  row.get("test_var"),
                                    "edge_density": row.get("edge_density"),
                                    "path": row.get("path"),
                                })
                                break
            else:
                l_fn += 1; s_fn += 1; board_fn += 1
                confusion[itype]["MISSED"] += 1
                label_stats[ilbl]["fn"] += 1

        for cid, det in found_map.items():
            if cid not in inj_map:
                l_fp += 1; s_fp += 1; board_fp += 1
                confusion["FP_actual"][det["defect_type"]] += 1
                label_stats[det.get("expected_label","?")]["fp"] += 1
    
        if diag:
            for row in diag:
                path_base = row.get("path","?").split("(")[0]
                path_counts[path_base] += 1

        status = "PASS" if (board_mm+board_fn+board_fp)==0 else "FAIL"
        board_rows.append((b+1, len(inj_map), board_mm, board_fn, board_fp, status,
                           inj_log, found, inj_map, found_map))

        if args.verbose:
            def _abbr(dtype): return _DTYPE_ABBR.get(dtype, dtype[:4].upper())
            inj_str = ", ".join(
                f"{_abbr(d['injected_type'])}({d['original_label'][:8]})"
                for d in inj_log)
            fnd_str = ", ".join(
                f"{_abbr(d['defect_type'])}({d.get('expected_label','?')[:8]})"
                for d in found if d["component_id"] in inj_map)
            mark = _green("✓") if status=="PASS" else _red("✗")
            suffix = (f"  mm={board_mm} fn={board_fn} fp={board_fp}"
                      if status=="FAIL" else "")
            print(f"  {mark} Board {b+1:3d}  inj=[{inj_str}]  found=[{fnd_str}]{suffix}")

        if need_diag and status=="FAIL" and diag:
            err_cids = set()
            for cid, inj in inj_map.items():
                if cid not in found_map or found_map[cid]["defect_type"] != inj["injected_type"]:
                    err_cids.add(cid)
            for cid in found_map:
                if cid not in inj_map:
                    err_cids.add(cid)

            print(f"\n  {_bold('[DBG]')} Board {b+1} failures:")
            for row in diag:
                is_err = row["cid"] in err_cids
                if not args.trace and not is_err:
                    continue
                if args.trace and not is_err and row["zone"] not in ("B","C"):
                    continue
                inj_type = inj_map.get(row["cid"],{}).get("injected_type","fp")
                det_type = found_map.get(row["cid"],{}).get("defect_type","none")
                correct  = inj_type == det_type
                match    = _green("✓") if correct else _red("✗")
                prefix   = "    " if is_err else "    " + _dim("~")

                ssim_s = (f" ssim={_yellow(str(round(row['ssim'],3)))}"
                          if row.get("ssim") is not None else "")
                var_s  = (f" var={row['test_var']:.1f}"
                          if row.get("test_var") is not None else "")
                tmpl_s = (f" tmpl={_cyan(str(round(row['tmpl'],3)))}"
                          if row.get("tmpl") is not None else "")
                ncc_s  = (f" ncc_d={row['ncc_delta']:.3f}"
                          if "ncc_delta" in row else "")
                goh_s  = (f" goh_d={row['goh_delta']:.3f}"
                          if row.get("goh_delta") is not None else "")
                asym_s = (f" asym={row['asym_score']:.3f}"
                          if row.get("asym_score") is not None else "")
                edge_s = (f" edge={row['edge_density']:.3f}"
                          if row.get("edge_density") is not None else "")
                dist_s = (f" sdist={row['same_dist']:.4f}"
                          if "same_dist" in row else "")
                atsl_s = (f" at_slot={row['at_slot']} local={row['is_local']}"
                          if "at_slot" in row else "")
                thr_s  = _dim(f" [low={row['low_thr']} high={row['high_thr']}]")
                inj_abbr = _DTYPE_ABBR.get(inj_type, inj_type[:4])
                det_abbr = _DTYPE_ABBR.get(det_type, det_type[:4])

                print(f"{prefix}{match} C{row['cid']:3d} {row['lbl'][:10]:<10}  "
                      f"Zone={row['zone']}  rdiff={row['rdiff']:.1f}"
                      f"  inj={inj_abbr}→det={det_abbr}"
                      f"{ssim_s}{tmpl_s}{var_s}{ncc_s}{goh_s}{asym_s}{edge_s}{dist_s}{atsl_s}")
                print(f"         path={_bold(row['path'])}{thr_s}")
                if is_err and not correct:
                    _explain_failure(row, inj_type, det_type)
            print()

    if args.save_viz:
        _save_viz(args.save_viz, golden_img, golden, board_rows, per_thr)

    # ── Results ───────────────────────────────────────────────────────────────
    def prf(tp,fp,fn):
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        f = 2*p*r/(p+r) if p+r else 0.0
        return p,r,f

    l_p,l_r,l_f = prf(l_tp,l_fp,l_fn)
    s_p,s_r,s_f = prf(s_tp,s_fp,s_fn)

    pass_boards = sum(1 for row in board_rows if row[5]=="PASS")
    board_acc   = pass_boards / args.boards

    print("\n" + "═"*72)
    print(f"  RESULTS  ({args.boards} boards, {total_inj} injected defects)")
    print("═"*72)
    print(f"\n  Board accuracy (perfect match) : {board_acc:.1%}  "
          f"({pass_boards}/{args.boards} boards fully correct)\n")

    print(f"  LENIENT (flagged the right component — any type)")
    print(f"    TP={l_tp}  FP={l_fp}  FN={l_fn}")
    print(f"    Precision={l_p:.3f}  Recall={l_r:.3f}  F1={l_f:.3f}\n")

    print(f"  STRICT (correct defect type)")
    print(f"    TP={s_tp}  FP={s_fp}  FN={s_fn}")
    print(f"    Precision={s_p:.3f}  Recall={s_r:.3f}  F1={s_f:.3f}")

    print("\n" + "─"*72)
    print("  PER-DEFECT-TYPE METRICS  (TP / FP / FN / Precision / Recall / F1)")
    print("─"*72)
    print(f"  {'Defect Type':<22}  {'TP':>5}  {'FP':>5}  {'FN':>5}  "
          f"{'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-"*70)

    _dtype_keys = ["missing", "wrong_component", "wrong_polarity", "misaligned"]

    for dtype in _dtype_keys:
        tp_d = confusion[dtype].get(dtype, 0)
        fn_d = sum(v for k, v in confusion[dtype].items() if k != dtype)
        fp_from_others = sum(
            confusion[other].get(dtype, 0)
            for other in _dtype_keys if other != dtype
        )
        fp_spurious = confusion["FP_actual"].get(dtype, 0)
        fp_d = fp_from_others + fp_spurious

        prec_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) else 0.0
        rec_d  = tp_d / (tp_d + fn_d) if (tp_d + fn_d) else 0.0
        f1_d   = 2 * prec_d * rec_d / (prec_d + rec_d) if (prec_d + rec_d) else 0.0

        f1_str = f"{f1_d:.3f}"
        f1_col = _green(f1_str) if f1_d>=0.90 else (_yellow(f1_str) if f1_d>=0.70 else _red(f1_str))
        rec_str = f"{rec_d:.3f}"
        rec_col = _green(rec_str) if rec_d>=0.90 else (_yellow(rec_str) if rec_d>=0.70 else _red(rec_str))
        bar = _green("▓" * int(f1_d * 10)) + _dim("░" * (10 - int(f1_d * 10)))
        print(f"  {dtype:<22}  {tp_d:>5}  {fp_d:>5}  {fn_d:>5}  "
              f"  {prec_d:>8.3f}  {rec_col:>8}  {f1_col:>8}  {bar}")

    print()

    injected_types = ["missing","wrong_component","wrong_polarity","misaligned"]
    det_cols       = ["missing","wrong_component","wrong_polarity","misaligned","MISSED"]

    print("\n" + "─"*72)
    print("  CONFUSION MATRIX  (rows=injected, cols=detected/missed)")
    print("─"*72)
    col_header = "Injected \\ Detected"
    hdr = f"  {col_header:<22}" + "".join(f"{c[:10]:>12}" for c in det_cols)
    print(hdr); print("  " + "-"*70)
    for itype in injected_types:
        row_total = sum(confusion[itype].values())
        if row_total == 0: continue
        row = f"  {itype:<22}"
        for dcol in det_cols:
            n = confusion[itype].get(dcol, 0)
            pct = f"{n/row_total:.0%}" if row_total else "  -"
            mark = "✓" if dcol==itype else (" " if n==0 else "✗")
            cell = f"  {mark}{n:4d}({pct:>4})"
            row += _green(cell) if mark=="✓" else (_red(cell) if mark=="✗" else cell)
        print(row)

    fp_total = sum(confusion["FP_actual"].values())
    if fp_total:
        print(f"\n  False Positives by type: ", end="")
        for dt,n in sorted(confusion["FP_actual"].items()):
            print(f"{dt}×{n}", end="  ")
        print()

    miss_as_wc    = confusion["missing"].get("wrong_component",0)
    wc_as_miss    = confusion["wrong_component"].get("missing",0)
    total_swapped = miss_as_wc + wc_as_miss
    miss_inj      = sum(confusion["missing"].values())
    wc_inj        = sum(confusion["wrong_component"].values())

    print("\n" + "─"*72)
    print("  MISSING ↔ WRONG_COMPONENT SWAP ANALYSIS")
    print("─"*72)
    if miss_inj:
        print(f"  missing → wrong_component  : {miss_as_wc:4d}  "
              f"({miss_as_wc/miss_inj:.1%} of {miss_inj} missing)")
    if wc_inj:
        print(f"  wrong_component → missing  : {wc_as_miss:4d}  "
              f"({wc_as_miss/wc_inj:.1%} of {wc_inj} wrong_component)")
    print(f"  Total swapped              : {total_swapped}")

    if label_stats:
        print("\n" + "─"*72)
        print("  PER-LABEL STRICT ACCURACY  (v2.0)")
        print("─"*72)
        print(f"  {'Label':<28}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'Recall':>8}  {'Prec':>8}  {'F1':>8}")
        print("  " + "-"*78)
        for lbl in sorted(label_stats):
            s = label_stats[lbl]
            tp,fp,fn = s["tp"],s["fp"],s["fn"]
            rec  = tp/(tp+fn) if tp+fn else 0.0
            prec = tp/(tp+fp) if tp+fp else 0.0
            f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
            bar = _green("▓" * int(rec*10)) + _dim("░" * (10-int(rec*10)))
            f1_str = f"{f1:.3f}"
            f1_col = _green(f1_str) if f1>=0.90 else (_yellow(f1_str) if f1>=0.70 else _red(f1_str))
            print(f"  {lbl:<28}  {tp:>5}  {fp:>5}  {fn:>5}  "
                  f"{rec:>7.1%}  {prec:>7.1%}  {f1_col:>8}  {bar}")

    if path_counts and (args.debug or args.trace):
        print("\n" + "─"*72)
        print("  DECISION PATH FREQUENCY  (debug/trace only)")
        print("─"*72)
        total_paths = sum(path_counts.values())
        for path, cnt in sorted(path_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(40, int(cnt/total_paths*40))
            print(f"  {path:<40}  {cnt:4d}  {bar}")

    if wc_fail_signals:
        print("\n" + "─"*72)
        print(f"  REMAINING wrong_component→missing FAILURES: {len(wc_fail_signals)}")
        print("─"*72)
        print(f"  {'Path':<45}  {'ssim':>7}  {'tmpl':>7}  {'var':>8}  {'edge':>7}")
        print("  " + "-"*78)
        for sig in wc_fail_signals:
            ss = f"{sig['ssim']:.3f}"    if sig['ssim']         is not None else "  n/a"
            tm = f"{sig['tmpl']:.3f}"    if sig['tmpl']         is not None else "  n/a"
            vv = f"{sig['var']:.1f}"     if sig['var']          is not None else "    n/a"
            ed = f"{sig['edge_density']:.3f}" if sig.get('edge_density') is not None else "  n/a"
            path = sig['path'] or "?"
            print(f"  {path:<45}  {ss:>7}  {tm:>7}  {vv:>8}  {ed:>7}")
        if wc_fail_signals:
            ssims = [s['ssim'] for s in wc_fail_signals if s['ssim'] is not None]
            tmpls = [s['tmpl'] for s in wc_fail_signals if s['tmpl'] is not None]
            vars_ = [s['var']  for s in wc_fail_signals if s['var']  is not None]
            edges = [s['edge_density'] for s in wc_fail_signals if s.get('edge_density') is not None]
            if ssims: print(f"\n  ssim range  : {min(ssims):.3f} – {max(ssims):.3f}  (mean {sum(ssims)/len(ssims):.3f})")
            if tmpls: print(f"  tmpl range  : {min(tmpls):.3f} – {max(tmpls):.3f}  (mean {sum(tmpls)/len(tmpls):.3f})")
            if vars_: print(f"  var  range  : {min(vars_):.1f} – {max(vars_):.1f}  (mean {sum(vars_)/len(vars_):.1f})")
            if edges: print(f"  edge range  : {min(edges):.3f} – {max(edges):.3f}  (mean {sum(edges)/len(edges):.3f})")
            print(f"\n  Irreducible if edge_density < {_EDGE_DENSITY_WC_MIN} AND var < {_VAR_WC_CONFIRM}")

    print("\n" + "═"*72 + "\n")


def _explain_failure(row, inj_type, det_type):
    ssim = row.get("ssim"); tmpl = row.get("tmpl"); var = row.get("test_var")
    goh  = row.get("goh_delta"); edge = row.get("edge_density")
    if inj_type == "wrong_component" and det_type == "missing":
        reason = []
        if ssim is not None and ssim < _SSIM_MISSING_MAX:
            reason.append(f"SSIM={ssim:.3f} < {_SSIM_MISSING_MAX}")
        if var  is not None and var  < _VAR_WC_CONFIRM:
            reason.append(f"var={var:.1f} < {_VAR_WC_CONFIRM}  ← PRIMARY SIGNAL")
        if edge is not None:
            reason.append(f"edge_density={edge:.3f} (below {_EDGE_DENSITY_WC_MIN} rescue threshold)")
        if tmpl is not None:
            reason.append(f"tmpl={tmpl:.3f} (diagnostic only)")
        print("         " + _red("→ wc→miss: ") + "; ".join(reason)
              if reason else "         → wc→miss: unknown reason")
    elif inj_type == "wrong_polarity" and det_type == "wrong_component":
        lbl = row.get("lbl","")
        is_ic = "ic" in lbl.lower()
        margin = _NCC_POLARITY_IC if is_ic else _NCC_POLARITY_C
        thresh_name = "_NCC_POLARITY_IC" if is_ic else "_NCC_POLARITY_C"
        goh_str = f"; goh_delta={goh:.3f}" if goh is not None else ""
        print("         " + _red("→ polarity→wc: ") +
              f"NCC delta below margin ({thresh_name}={margin}); "
              f"structural + GOH fallbacks also missed"
              f"(ssim={ssim:.3f} tmpl={tmpl:.3f}{goh_str})"
              if ssim is not None and tmpl is not None else
              "         → polarity→wc: insufficient signal for WPOL classification")
    elif inj_type == "missing" and det_type == "wrong_component":
        print("         " + _red("→ miss→wc: ") +
              f"SSIM={ssim:.3f} or tmpl={tmpl:.3f} above threshold; fill was too structured.")


def _save_viz(out_path, golden_img, golden, board_rows, per_thr):
    try:
        H, W = golden_img.shape[:2]
        scale = min(1.0, 1600/max(H,W))
        vis_w, vis_h = int(W*scale), int(H*scale)

        failed = [r for r in board_rows if r[5]=="FAIL"][:4]
        if not failed:
            print(f"[VIZ]  No failed boards — skipping visualisation.")
            return

        cols = 1; rows = len(failed)
        canvas = np.zeros((rows*vis_h, cols*vis_w, 3), dtype=np.uint8)

        for row_idx, br in enumerate(failed):
            _, _, _, _, _, _, inj_log, found, inj_map, found_map = br
            vis = cv2.resize(golden_img.copy(), (vis_w, vis_h))

            for gc in golden:
                x1,y1,x2,y2 = gc.xyxy(vis_w, vis_h)
                cv2.rectangle(vis, (x1,y1),(x2,y2), (80,80,80), 1)

            for d in inj_log:
                cid = d["component_id"]
                gc = next((c for c in golden if c.id==cid), None)
                if gc is None: continue
                x1,y1,x2,y2 = gc.xyxy(vis_w, vis_h)
                det = found_map.get(cid)
                correct = det is not None and det["defect_type"]==d["injected_type"]
                color = (0,200,0) if correct else (0,0,220)
                cv2.rectangle(vis,(x1,y1),(x2,y2), color, 2)
                label = f"inj:{d['injected_type'][:4]}"
                if det: label += f"|det:{det['defect_type'][:4]}"
                cv2.putText(vis, label, (x1, max(0,y1-3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

            y0 = row_idx * vis_h
            canvas[y0:y0+vis_h, 0:vis_w] = vis

        cv2.imwrite(out_path, canvas)
        print(f"[VIZ]  Saved annotated visualisation → {out_path}")
    except Exception as e:
        print(f"[VIZ]  Warning: could not save visualisation: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="PCB Defect Accuracy Evaluator v3.9")
    p.add_argument("--golden",   default=r"D:\MELSS\AOI\NEW_TEST_IMGS\test.jpg")
    import pathlib
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    DEFAULT_MODEL = ROOT / "models" / "best.pt"
    p.add_argument("--model",    default=str(DEFAULT_MODEL))
    p.add_argument("--boards",   type=int,   default=100)
    p.add_argument("--defects",  type=int,   default=5)
    p.add_argument("--conf",     type=float, default=0.15)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--verbose",  action="store_true")
    p.add_argument("--debug",    action="store_true")
    p.add_argument("--trace",    action="store_true")
    p.add_argument("--save_viz", default="")
    p.add_argument("--only_defect", choices=sorted(DEFECT_MIX))
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.trace:
        args.debug   = True
        args.verbose = True
    if args.debug:
        args.verbose = True
    run_evaluation(args)