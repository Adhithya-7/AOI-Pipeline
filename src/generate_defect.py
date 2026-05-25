"""
PCB Defect Generator  (extracted from aoi_evaluator_v39.py)
============================================================
Outputs two folders:

  <out_dir>/images/      -- clean defected images, ZERO markings
  <out_dir>/annotated/   -- same images with coloured boxes showing changes

Usage
-----
  python pcb_defect_generator.py [options]

Options
-------
  --golden        PATH   Golden PCB image             (default: evaluator default)
  --model         PATH   YOLO .pt model               (default: evaluator default)
  --boards        N      Number of boards to generate (default: 10)
  --defects       N      Max defects per board        (default: 5)
  --conf          F      YOLO confidence threshold    (default: 0.15)
  --seed          N      RNG seed                     (default: 42)
  --out_dir       PATH   Root output folder           (default: ./defect_output)
  --only_defect   TYPE   Restrict to one defect type
  --verbose              Print per-board details
  --save_json            Write JSON sidecar in annotated/ folder
"""

import argparse, copy, json, pathlib, random, sys, time
from collections import defaultdict


def _pip(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:    import cv2
except: _pip("opencv-python"); import cv2
try:    import numpy as np
except: _pip("numpy"); import numpy as np
try:    from ultralytics import YOLO
except: _pip("ultralytics"); from ultralytics import YOLO

_USE_COLOR = sys.stdout.isatty()
def _c(code, s): return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s
def _green(s):   return _c("32", s)
def _bold(s):    return _c("1",  s)


# =============================================================================
# 1.  COMPONENT
# =============================================================================
class Component:
    def __init__(self, cid, label, conf, cx, cy, w, h):
        self.id = cid; self.label = label; self.conf = conf
        self.cx = cx; self.cy = cy; self.w = w; self.h = h

    def xyxy(self, W, H):
        return (max(0, int((self.cx - self.w/2)*W)),
                max(0, int((self.cy - self.h/2)*H)),
                min(W, int((self.cx + self.w/2)*W)),
                min(H, int((self.cy + self.h/2)*H)))

    def area(self): return self.w * self.h

    def is_polarized(self):
        return self.label.lower() in {
            "ic (u)", "ic (ic)", "transistor (q)", "transistor (qa)",
            "diode (d)", "led", "capacitor (electrolytic)",
            "capacitor (c)", "cap array (cra)",
        }


# =============================================================================
# 2.  DETECTION
# =============================================================================
def _simple_nms(comps, iou_thr=0.40):
    if len(comps) <= 1:
        return comps
    ordered = sorted(range(len(comps)), key=lambda i: -comps[i].conf)
    keep, supp = [], set()
    for i in ordered:
        if i in supp:
            continue
        keep.append(i)
        a = comps[i]
        ax1, ay1 = a.cx - a.w/2, a.cy - a.h/2
        ax2, ay2 = a.cx + a.w/2, a.cy + a.h/2
        for j in ordered:
            if j in supp or j == i:
                continue
            b = comps[j]
            ix = max(0, min(ax2, b.cx+b.w/2) - max(ax1, b.cx-b.w/2))
            iy = max(0, min(ay2, b.cy+b.h/2) - max(ay1, b.cy-b.h/2))
            inter = ix * iy
            union = a.w*a.h + b.w*b.h - inter
            if union > 0 and inter/union > iou_thr:
                supp.add(j)
    comps = [comps[i] for i in keep]
    for uid, c in enumerate(comps):
        c.id = uid
    return comps


def detect(model, img_bgr, conf=0.15):
    results = model.predict(img_bgr, conf=conf, iou=0.35, imgsz=640, verbose=False)
    comps, uid = [], 0
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cx, cy, w, h = b.xywhn[0].tolist()
            comps.append(Component(uid, r.names[int(b.cls[0])],
                                   float(b.conf[0]), cx, cy, w, h))
            uid += 1
    return _simple_nms(comps)


# =============================================================================
# 3.  INJECTORS  (verbatim from evaluator -- zero drawing inside)
# =============================================================================
def _build_bg_mask(img, comps):
    H, W = img.shape[:2]
    mask = np.ones((H, W), dtype=bool)
    for c in comps:
        bx1, by1, bx2, by2 = c.xyxy(W, H)
        exp = max(2, int(min(bx2-bx1, by2-by1)*0.10))
        mask[max(0,by1-exp):min(H,by2+exp),
             max(0,bx1-exp):min(W,bx2+exp)] = False
    return mask


def inject_missing(img, comp, comps, cached_bg_mask=None):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = comp.xyxy(W, H)
    if x2 <= x1 or y2 <= y1:
        return img, None, cached_bg_mask
    ph, pw = y2-y1, x2-x1
    bg_mask = _build_bg_mask(img, comps) if cached_bg_mask is None else cached_bg_mask
    bg_mask[y1:y2, x1:x2] = False

    def _safe(rf):
        r = int(max(ph, pw)*rf)
        ry1, ry2 = max(0, y1-r), min(H, y2+r)
        rx1, rx2 = max(0, x1-r), min(W, x2+r)
        return img[ry1:ry2, rx1:rx2][bg_mask[ry1:ry2, rx1:rx2]].astype(np.float32)

    safe = np.empty((0, 3), dtype=np.float32)
    for f in (3, 8, 40):
        safe = _safe(f)
        if len(safe) >= 64:
            break
    if len(safe) < 8:
        safe = img[bg_mask].astype(np.float32)
    if len(safe) == 0:
        safe = img.reshape(-1, 3).astype(np.float32)

    mean_c = np.median(safe, axis=0)
    std_c  = np.std(safe, axis=0).clip(2, 10)
    fill   = np.random.normal(mean_c, std_c, (ph, pw, 3)).astype(np.float32)

    feather = min(4, min(ph, pw)//4)
    if feather > 0:
        ks   = np.arange(feather, dtype=np.float32)
        edge = 0.5 - 0.5 * np.cos(np.pi * ks / feather)
        ramp = np.ones((ph, pw), dtype=np.float32)
        ramp[:feather,  :]  = np.minimum(ramp[:feather,  :],  edge[:, np.newaxis])
        ramp[-feather:, :]  = np.minimum(ramp[-feather:, :],  edge[::-1, np.newaxis])
        ramp[:,  :feather]  = np.minimum(ramp[:,  :feather],  edge[np.newaxis, :])
        ramp[:, -feather:]  = np.minimum(ramp[:, -feather:],  edge[np.newaxis, ::-1])
    else:
        ramp = np.ones((ph, pw), dtype=np.float32)
    ramp = ramp[:, :, np.newaxis]

    orig = img[y1:y2, x1:x2].astype(np.float32)
    img[y1:y2, x1:x2] = np.clip(ramp*fill + (1-ramp)*orig, 0, 255).astype(np.uint8)
    return img, {"defect_type": "missing"}, bg_mask


def inject_wrong_component(img, comp, comps, rng):
    H, W = img.shape[:2]
    ax1, ay1, ax2, ay2 = comp.xyxy(W, H)
    if ax2 <= ax1 or ay2 <= ay1:
        return img, None
    tw, th = ax2-ax1, ay2-ay1

    def _grp(c):
        lbl, a = c.label.lower(), c.area()
        if any(x in lbl for x in ("resistor","capacitor","ferrite","inductor")):
            return "passive"
        if any(x in lbl for x in ("transistor","diode")):
            return "active_s"
        if "ic" in lbl:
            if   a < 0.004:  return "ic_0"
            elif a < 0.012:  return "ic_1"
            elif a < 0.030:  return "ic_2"
            else:            return "ic_3"
        return "other_" + lbl[:8]

    tg = _grp(comp)
    donors = []
    for c in comps:
        if c.id == comp.id or c.label.lower() == comp.label.lower() or c.conf <= 0.15:
            continue
        if _grp(c) != tg:
            continue
        bx1, by1, bx2, by2 = c.xyxy(W, H)
        if bx2 <= bx1 or by2 <= by1:
            continue
        ratio = max(c.area(), comp.area()) / max(min(c.area(), comp.area()), 1e-9)
        if ratio > 4.0:
            continue
        donors.append((c, bx1, by1, bx2, by2, ratio))

    if not donors:
        return img, None

    donors.sort(key=lambda x: x[5])
    donor, bx1, by1, bx2, by2, ratio = rng.choice(donors[:max(1, len(donors)//2+1)])
    dp   = img[by1:by2, bx1:bx2].copy()
    orig = img[ay1:ay2, ax1:ax2].copy()
    paste = cv2.resize(dp, (tw, th), interpolation=cv2.INTER_LINEAR)

    if ratio > 1.6:
        oy = rng.randint(-max(1, int(th*0.20)), max(1, int(th*0.20)))
        ox = rng.randint(-max(1, int(tw*0.20)), max(1, int(tw*0.20)))
        dy1 = max(0, ay1+oy); dy2 = min(H, ay2+oy)
        dx1 = max(0, ax1+ox); dx2 = min(W, ax2+ox)
        sy1 = dy1-(ay1+oy);   sy2 = sy1+(dy2-dy1)
        sx1 = dx1-(ax1+ox);   sx2 = sx1+(dx2-dx1)
        if dy2 > dy1 and dx2 > dx1:
            img[dy1:dy2, dx1:dx2] = paste[sy1:sy2, sx1:sx2]
    else:
        img[ay1:ay2, ax1:ax2] = paste

    if cv2.absdiff(orig, img[ay1:ay2, ax1:ax2]).mean() < 8.0:
        img[ay1:ay2, ax1:ax2] = orig
        return img, None

    return img, {"defect_type": "wrong_component", "found_label": donor.label}


def inject_wrong_polarity(img, comp):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = comp.xyxy(W, H)
    if x2 <= x1 or y2 <= y1:
        return img, None
    patch   = img[y1:y2, x1:x2].copy()
    flipped = cv2.rotate(patch, cv2.ROTATE_180)
    if cv2.absdiff(patch, flipped).mean() < 8.0:
        return img, None
    img[y1:y2, x1:x2] = flipped
    return img, {"defect_type": "wrong_polarity"}


def inject_misaligned(img, comp, rng):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = comp.xyxy(W, H)
    if x2 <= x1 or y2 <= y1:
        return img, None
    patch = img[y1:y2, x1:x2].copy()
    ph, pw = patch.shape[:2]
    if ph < 6 or pw < 6:
        return img, None

    angle  = rng.uniform(15, 45) * rng.choice([-1, 1])
    ctr    = (pw//2, ph//2)
    M      = cv2.getRotationMatrix2D(ctr, angle, 1.0)
    rotated = cv2.warpAffine(patch, M, (pw, ph), borderMode=cv2.BORDER_REPLICATE)

    msk  = np.ones((ph, pw), dtype=np.float32)
    mr   = cv2.warpAffine(msk, M, (pw, ph), borderValue=0.0)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mr   = cv2.erode(mr, kern, iterations=1)
    mr   = cv2.GaussianBlur(mr, (5, 5), 0)[:, :, np.newaxis]

    blended = (rotated.astype(np.float32)*mr +
               patch.astype(np.float32)*(1.0-mr))
    img[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    if cv2.absdiff(patch, img[y1:y2, x1:x2]).mean() < 5.0:
        img[y1:y2, x1:x2] = patch
        return img, None

    return img, {"defect_type": "misaligned", "angle": round(angle, 1)}


DEFECT_MIX = {
    "missing":         0.35,
    "wrong_component": 0.30,
    "wrong_polarity":  0.20,
    "misaligned":      0.15,
}
DTYPES = list(DEFECT_MIX.keys())
DPROBS = [DEFECT_MIX[k] for k in DTYPES]


def create_defects(golden_img, golden_comps, n_defects, rng):
    """Returns (defected_img, log). golden_img is NEVER modified."""
    img = golden_img.copy()
    wc  = copy.deepcopy(golden_comps)

    available = list(range(len(wc)))
    rng.shuffle(available)
    victims = available[:min(n_defects, len(available))]

    log, used, cached = [], set(), None

    for vi in victims:
        c     = wc[vi]
        dtype = rng.choices(DTYPES, weights=DPROBS, k=1)[0]

        if dtype == "wrong_polarity" and not c.is_polarized():
            others = [j for j in available
                      if wc[j].is_polarized() and j not in used and j != vi]
            if others:
                vi = rng.choice(others)
                c  = wc[vi]
            else:
                dtype = "missing"

        if vi in used:
            continue
        used.add(vi)
        meta = None

        if dtype == "missing":
            img, meta, cached = inject_missing(img, c, wc, cached)
        elif dtype == "wrong_component":
            img, meta = inject_wrong_component(img, c, wc, rng)
            if not meta:
                img, meta, cached = inject_missing(img, c, wc, cached)
        elif dtype == "wrong_polarity":
            img, meta = inject_wrong_polarity(img, c)
            if not meta:
                img, meta, cached = inject_missing(img, c, wc, cached)
        elif dtype == "misaligned":
            img, meta = inject_misaligned(img, c, rng)
            if not meta:
                img, meta, cached = inject_missing(img, c, wc, cached)

        if meta:
            row = {"component_id":   c.id,
                   "original_label": c.label,
                   "injected_type":  meta["defect_type"]}
            for k, v in meta.items():
                if k != "defect_type":
                    row[k] = v
            log.append(row)

    return img, log


# =============================================================================
# 4.  ANNOTATOR -- draws on a separate copy ONLY, never touches the clean image
# =============================================================================
_COLORS = {
    "missing":         (0,   0,   220),   # red
    "wrong_component": (0,   165, 255),   # orange
    "wrong_polarity":  (255, 0,   200),   # magenta
    "misaligned":      (0,   220, 220),   # yellow
}
_ABBR = {
    "missing":         "MISS",
    "wrong_component": "WCOM",
    "wrong_polarity":  "WPOL",
    "misaligned":      "MALI",
}


def annotate(clean_img, inj_log, golden_comps):
    """
    Returns an annotated copy of clean_img.
    clean_img itself is NOT modified -- a new copy() is made first.
    """
    vis     = clean_img.copy()          # <-- only copy is drawn on
    H, W    = vis.shape[:2]
    by_id   = {c.id: c for c in golden_comps}

    for d in inj_log:
        gc = by_id.get(d["component_id"])
        if gc is None:
            continue

        x1, y1, x2, y2 = gc.xyxy(W, H)
        dtype  = d["injected_type"]
        color  = _COLORS.get(dtype, (180, 180, 180))
        abbr   = _ABBR.get(dtype, dtype[:4].upper())

        # Box around the changed region
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label
        txt = abbr + " " + d["original_label"][:14]
        if dtype == "misaligned":
            txt += " " + str(d.get("angle", "")) + "deg"

        font  = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.40
        thick  = 1
        (tw, txh), _ = cv2.getTextSize(txt, font, fscale, thick)
        pad = 2
        ty  = max(txh + pad*2, y1 - 2)
        cv2.rectangle(vis, (x1, ty - txh - pad*2), (x1 + tw + pad*2, ty),
                      color, cv2.FILLED)
        cv2.putText(vis, txt, (x1 + pad, ty - pad),
                    font, fscale, (255, 255, 255), thick, cv2.LINE_AA)

    return vis


# =============================================================================
# 5.  MAIN LOOP
# =============================================================================
def run_generator(args):
    global DTYPES, DPROBS

    print("\n" + "="*64)
    print(_bold("  PCB DEFECT GENERATOR"))
    print("="*64)

    if args.only_defect:
        DTYPES = [args.only_defect]
        DPROBS = [1.0]

    print(f"\n[MODEL]  {args.model}")
    model = YOLO(args.model)

    golden_img = cv2.imread(args.golden)
    if golden_img is None:
        sys.exit(f"[ERROR] Cannot read golden image: {args.golden}")
    H, W = golden_img.shape[:2]
    print(f"[GOLDEN] {W}x{H}  {args.golden}")

    golden = detect(model, golden_img, conf=args.conf)
    if not golden:
        sys.exit("[ERROR] Zero components detected on golden image.")

    counts = defaultdict(int)
    for c in golden:
        counts[c.label] += 1
    print(f"[GOLDEN] {len(golden)} components detected:")
    for lbl, n in sorted(counts.items()):
        print(f"    {lbl:<30}  x{n}")

    # ── Output folders ────────────────────────────────────────────────────────
    root    = pathlib.Path(args.out_dir)
    img_dir = root / "images"       # clean, zero markings
    ann_dir = root / "annotated"    # coloured boxes showing what changed
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  images/    -> {img_dir.resolve()}")
    print(f"               (pure defected image -- NO boxes, NO text, NOTHING drawn)")
    print(f"  annotated/ -> {ann_dir.resolve()}")
    print(f"               (coloured boxes marking every changed component)")
    print(f"\n  Boards  : {args.boards}")
    print(f"  Defects : 1-{args.defects} per board")
    mix_str = args.only_defect if args.only_defect else "default " + str(DEFECT_MIX)
    print(f"  Mix     : {mix_str}\n")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    type_counts = defaultdict(int)
    t0 = time.time()

    for b in range(args.boards):
        n_defects = rng.randint(1, max(1, args.defects))

        # Pixel-only injection -- absolutely no drawing inside create_defects
        defected, log = create_defects(golden_img, golden, n_defects, rng)

        types_str = "_".join(
            sorted({_ABBR.get(d["injected_type"], "UNK") for d in log})
        ) or "NONE"
        stem = f"board_{b+1:04d}_{types_str}"

        # 1. CLEAN -- only pixel changes from injection, nothing drawn
        cv2.imwrite(str(img_dir / (stem + ".png")), defected)

        # 2. ANNOTATED -- boxes drawn on a separate copy
        ann_img = annotate(defected, log, golden)
        cv2.imwrite(str(ann_dir / (stem + ".png")), ann_img)

        # 3. Optional JSON sidecar (annotated folder only)
        if args.save_json:
            with open(ann_dir / (stem + ".json"), "w") as fh:
                json.dump({"board": b+1, "defects": log}, fh, indent=2)

        for d in log:
            type_counts[d["injected_type"]] += 1

        if args.verbose:
            parts = []
            for d in log:
                ab = _ABBR.get(d["injected_type"], "?")
                ex = " (" + str(d.get("angle", "")) + "deg)" \
                     if d["injected_type"] == "misaligned" else ""
                parts.append(ab + "@C" + str(d["component_id"]) +
                             "(" + d["original_label"][:10] + ")" + ex)
            print("  " + _green("OK") + f" Board {b+1:4d}  {stem}.png"
                  "  [" + ", ".join(parts) + "]")
        elif (b+1) % max(1, args.boards//10) == 0:
            print(f"  ... {b+1}/{args.boards} ({(b+1)/args.boards*100:.0f}%)")

    elapsed = time.time() - t0
    total   = sum(type_counts.values())

    print("\n" + "="*64)
    print(f"  Done -- {args.boards} boards, {total} defects injected in {elapsed:.1f}s")
    print("-"*64)
    for dtype in ["missing", "wrong_component", "wrong_polarity", "misaligned"]:
        n   = type_counts.get(dtype, 0)
        pct = n/total*100 if total else 0
        bar = _green("#" * int(pct/2.5)) + "." * (40 - int(pct/2.5))
        print(f"  {dtype:<22}  {n:5d}  ({pct:5.1f}%)  {bar}")
    print(f"\n  Clean images  -> {img_dir.resolve()}")
    print(f"  Annotated     -> {ann_dir.resolve()}")
    print("="*64 + "\n")


# =============================================================================
# 6.  ENTRY POINT
# =============================================================================
def _parse_args():
    ROOT = pathlib.Path(__file__).resolve().parents[1]

    p = argparse.ArgumentParser(description="PCB Defect Generator")
    p.add_argument("--golden",
                   default=r"D:\MELSS\AOI\NEW_TEST_IMGS\images\v2.jpg",
                   help="Golden PCB image path")
    p.add_argument("--model",
                   default=str(ROOT / "models" / "best.pt"),
                   help="YOLO .pt model path")
    p.add_argument("--boards",      type=int,   default=10,
                   help="Number of boards to generate")
    p.add_argument("--defects",     type=int,   default=5,
                   help="Max defects per board (actual count is random 1..N)")
    p.add_argument("--conf",        type=float, default=0.15,
                   help="YOLO detection confidence threshold")
    p.add_argument("--seed",        type=int,   default=42,
                   help="RNG seed for reproducibility")
    p.add_argument("--out_dir",     default="defect_output",
                   help="Root output folder (images/ and annotated/ created inside)")
    p.add_argument("--only_defect", choices=sorted(DEFECT_MIX), default=None,
                   help="Restrict injection to one defect type")
    p.add_argument("--verbose",     action="store_true",
                   help="Print one line per board")
    p.add_argument("--save_json",   action="store_true",
                   help="Write JSON sidecar files into annotated/")
    return p.parse_args()


if __name__ == "__main__":
    run_generator(_parse_args())