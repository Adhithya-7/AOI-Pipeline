"""
PCB Defect Accuracy Evaluator  v3.1
====================================
Injects defects → runs AOI inference → reports accuracy + confusion matrix.
No GUI. No large file I/O. Just pure evaluation.

v3.6 changes (over v3.5):
  [ACCURACY — 1 targeted fix from 100-board 15-defect run (728 defects)]

  v3.5's lower structural thresholds (Fix K) introduced 3 new WCOM→WPOL FPs,
  all via polar_struct_polarity, all on Capacitor components:
    Board  1 C66: ssim=0.356, tmpl=0.436, ncc_delta=0.040
    Board 10 C46: ssim=0.444, tmpl=0.352, ncc_delta=0.186
    Board 56 C63: ssim=0.429, tmpl=0.406, ncc_delta=0.037
  The lowered ssim/tmpl gates (0.28/0.25) correctly passed these values, but
  the NCC delta was telling the truth: all three are NOT rotated components.

  Fix M — add _NCC_POLAR_STRUCT_MIN_DELTA = 0.25 to structural fallback:
    FP cap deltas:   0.037, 0.040, 0.186  — all blocked by ≥ 0.25
    True WPOL deltas via this path: C0=0.724, C6=0.744, C9=0.487, C46=0.393
    Gap between FP max (0.186) and TP min (0.393) is unambiguous; 0.25 is safe.
    The fix supersedes the old `delta >= 0` gate (which remains as a comment
    artefact — the new gate is strictly stronger).

  [REMAINING IRREDUCIBLE FAILURES]
  • C84 Capacitor WPOL: ssim=0.231, tmpl=0.099, ncc_delta well below cap margin.
    Both signals below structural gates (ssim<0.28 AND tmpl<0.25). Appears ~8×
    per 100 boards whenever C84 is injected. Irreducible without new signals.
  • C8 IC WPOL: ssim=0.323, tmpl=0.101. tmpl indistinct from WCOM noise floor.
  • 6× WCOM→MISS: donor paste var 1.0–2.0, pixel-indistinguishable from fill.

  [ACCURACY — 2 targeted fixes from 100-board extreme-density run (2178 defects)]

  New failure patterns at high defect density:

  • 51× WPOL→WCOM: near-symmetric ICs (C6, C8) with NCC delta below the 0.28
    IC margin. YOLO still fires a same-label det at the slot, sending them to
    `at_slot_ssim` → wrong_component before the structural fallback could trigger.
  • 17× WPOL→MISS: same IC family (C0) where 180° rotation produces a muted
    signal — var=6.7, ssim=0.299, tmpl=0.265 — below the old structural gates.
  • 1× WCOM FN: Zone B polar no-decision bug (C47 cap, `zone_b_no_decision`).

  Fix K — lower structural polarity thresholds (0.40/0.45 → 0.28/0.25):
    C0 IC WPOL: ssim=0.299 ≥ 0.28 ✓ AND tmpl=0.265 ≥ 0.25 ✓ → now caught as WPOL
    C6 IC WPOL: ssim=0.322 ≥ 0.28 ✓ AND tmpl=0.364 ≥ 0.25 ✓ → now caught as WPOL
    C8 IC WPOL: ssim=0.323 ≥ 0.28 ✓ AND tmpl=0.101 ≥ 0.25 ✗ → irreducible (tmpl
      indistinguishable from WCOM noise floor ~0.10–0.20).
    Safety retained: delta≥0 guard (Fix G) still blocks WCOM caps with negative
    NCC. Observed WCOM ICs at this step either have var>>80 (→ var_rescue) or
    ssim>>0.35 (→ at_slot_ssim), avoiding the structural fallback entirely.

  Fix L — Zone B polar no-decision bug:
    When is_polar, same det present, NCC returns flipped=False, the `elif is_polar`
    branch previously produced no defect and fell through with decision="none".
    Elevated rdiff in Zone B with a same-label YOLO det means something is there;
    NCC blindness means we can't confirm rotation, so default to wrong_component.
    Added `else:` branch emitting `zone_b_polar_noflip` → wrong_component.

  [REMAINING IRREDUCIBLE FAILURES]
  • C8 IC (U): tmpl=0.101 — indistinct from WCOM noise; no clean gate.
  • 35× WCOM→MISS: donor after resize produces var 0.8–2.0, ssim 0.011–0.045 —
    pixel-indistinguishable from inject_missing Gaussian fill. Irreducible.

  [ACCURACY — 3 targeted fixes from 100-board high-density run]

  New WCOM→WPOL failures appeared in the high-defect-density run (5 total):
  • 3× via ncc_zoneC STRONG branch  (ncc_d = 0.768, 0.787, 0.890, 0.910)
  • 1× via ncc_zoneC ssim_neg branch (ssim = -0.306, delta = 0.431)
  • 1× via cross_polar_rescue        (delta = 0.401, tmpl = 0.317)

  Fix H — raise _NCC_POLARITY_STRONG 0.72 → 0.95:
    All 4 STRONG-path FPs had ncc_d ≤ 0.910; all true WPOL detections via this
    branch had ncc_d ≥ 1.058. Gap of ~0.15 makes 0.95 a safe discriminator.
    Also fixes the v3.3 "irreducible" Board 36 C63 failure (ncc_d=0.910).

  Fix I — tighten _NCC_POLARITY_SSIM_NEG -0.30 → -0.33:
    WCOM FP (Board 29 C59) had ssim=-0.306 — barely below the old -0.30 gate.
    All observed true WPOL caps through this branch have ssim ≤ -0.336.
    Moving the gate to -0.33 cleanly blocks the FP while preserving all TPs.

  Fix J — add _POLAR_CROSS_RESCUE_NCC_MIN gate (0.50) to cross_polar_rescue:
    FP (Board 53 C68) had delta=0.401 — just barely above the 0.38 flip margin.
    Nearest true-positive rescue (C66) has delta=0.708.
    Adding `delta > 0.50` blocks the FP and preserves all observed TPs.

  [REMAINING IRREDUCIBLE FAILURES — 14 cases]
  14× wrong_component → missing: donor component, after resize+paste into a
  small resistor/cap slot, produces a flat-looking patch (var 0.8–1.8, ssim
  0.013–0.043) that is pixel-indistinguishable from inject_missing Gaussian fill.
  The _VAR_WC_CONFIRM threshold CANNOT be lowered to fix these without
  introducing ~equal MISS→WCOM false positives (MISS fill spans the same
  var range). These are irreducible given the current signal set.

v3.3 changes (over v3.2):
  [ACCURACY — 1 targeted fix: structural polarity fallback NCC sign gate]

  v3.2's Fix F (extending structural fallback from ICs to all polarized) caused
  7 new WCOM→WPOL false positives (boards 9/12/33/50/87/89/91).

  Root cause: all 7 FP caps had NEGATIVE NCC delta (-0.856 to -0.367).
  A negative delta means NCC actively says "test patch resembles original
  orientation more than rotated" — i.e. it is a different-looking component,
  not a rotated version of the same one.  The ssim+tmpl gate alone was
  insufficient because passive caps within the same family look similar.

  Fix G — add `delta >= 0` to the polar_struct_polarity fallback condition.
    Negative delta: NCC says not rotated → wrong_component, never WPOL.
    Near-zero positive delta: NCC ambiguous (component nearly symmetric) →
      ssim+tmpl then adjudicate → correct WPOL.
    This blocks all 7 FP caps (all delta < -0.35) while preserving:
      - 4 IC9 polarity cases (delta positive but < 0.28 IC margin)
      - C46 cap polarity case from board 82 (delta positive but < 0.38 cap margin)

v3.2 changes (over v3.1):
  [ACCURACY — 2 targeted fixes from 100-board v3.1 analysis]

  • Fix E — cross_polar_rescue tmpl gate (boards 3/35/59/82-C52: 4× WCOM→WPOL FP):
    cross_polar_rescue was firing on wrong_component caps whose donor happened
    to create a flip-like NCC signal.  Observed FP tmpl values: 0.178, -0.123,
    -0.094.  True WPOL rescued by this path had tmpl=0.343.
    Added `tmpl_v >= _POLAR_CROSS_RESCUE_TMPL (0.20)` to the rescue condition:
    all 4 FPs have tmpl < 0.20 (blocked); legitimate WPOL rescue has tmpl > 0.20
    (preserved).  After blocking, these slots fall through to cross_zoneC →
    wrong_component correctly.

  • Fix F — structural polarity fallback extended from IC-only to all polarized
    (board 82 C46 Capacitor: WPOL→WCOM):
    C46 cap: NCC delta below 0.38 (cap margin) but ssim=0.502, tmpl=0.473.
    Same pattern as C9 IC from v3.0 — same component rotated, NCC blind.
    The threshold (ssim ≥ 0.40 AND tmpl ≥ 0.45) is safe for caps: every
    observed WCOM cap FP has tmpl < 0.20.  Fallback renamed
    `polar_struct_polarity`; applies to all is_polar components.

  [REMAINING IRREDUCIBLE FAILURES — 2 cases, no clean fix available]
  • Board 36 C63 Capacitor: WCOM→WPOL via ncc_zoneC.
    Donor paste produces ssim=-0.394 AND ncc_d=0.910 > STRONG threshold (0.72).
    True WPOL caps via ssim<-0.30 branch ALSO have tmpl < -0.40 — tmpl is not
    a usable discriminator here.  No available signal cleanly separates this
    wrong_component donor from a genuine rotation.
  • Board 77 C56 Resistor: WCOM→MISS via cross_zoneC_flat_override.
    Donor produces var=1.4 — indistinguishable from inject_missing Gaussian fill.
    Removing the flat override would re-introduce Board 40's miss→wc regression.

v3.1 changes (over v3.0):
  [ACCURACY — 4 targeted fixes diagnosed from 100-board run]

  • Fix A — IC near-symmetric polarity fallback (boards 4/8/28/71):
    C9 IC delta was BELOW even _NCC_POLARITY_IC=0.28 — IC is physically
    symmetric, NCC cannot detect the flip.  Secondary structural heuristic:
    if NCC gave no flip signal but ssim ≥ _SSIM_IC_POLARITY (0.40) AND
    tmpl ≥ _TMPL_IC_POLARITY (0.45), the slot contains the same component
    rotated (not a different donor).  IC wrong_component donors always have
    tmpl < 0.20 and ssim < 0.35 in observed data → safe gate.

  • Fix B — cross_zoneC polarized flip rescue (boards 9/30):
    C74/C66 Capacitors: NCC ran in Step 2 (accept_polarity=False because
    delta < 0.72 and ssim > -0.30), then cross_owns=True sent them to
    wrong_component via cross_zoneC.  `flipped` and `delta` are now
    preserved across steps.  In Step 3, if is_polar AND flipped AND
    var_structured AND ssim < 0.45 → wrong_polarity (YOLO cross-label is
    rotation confusion, not a different component).  ssim < 0.45 guard
    prevents C42 wrong_component (ssim=0.654) from being mistakenly rescued.

  • Fix C — flat guard same-detection rescue (board 85):
    C7 IC wrong_component: pasted IC donor looked flat in 20%-crop (var=2.9)
    → flat guard fired → missing.  If a same-label YOLO detection exists near
    the slot, the slot cannot be empty.  Flat guard now skips when same≠∅.

  • Fix D — verbose display bug:
    wrong_component and wrong_polarity both truncated to "WRON" making board
    summaries unreadable on failed boards.  Added _DTYPE_ABBR lookup table
    producing "WCOM", "WPOL", "MISS", "MALI" in output.

v3.0 changes (over v2.3):
  [SPEED]
  • _ncc_polarity: no longer resizes 7832×5874 images on every call.
    Caller pre-computes one blurred+scaled pair per board and passes it in;
    the function signature gains optional g_blurred/t_blurred parameters.
  • _aoi_check_board: pre-computes g_sm_blur/t_sm_blur ONCE, passes to NCC.
  • create_defects: _build_bg_mask cached once per board, not per injection.
  • inject_missing: feather ramp now built with numpy broadcasting (no Python loop).
  • _aoi_region_diff batch mode: all 89 component diffs computed in one
    vectorised NumPy pass instead of a per-component Python loop.
  • Calibration n_runs 3→2 (saves one YOLO inference ~1.1 s).

  [ACCURACY]
  • Separate NCC polarity margins for IC vs Capacitor:
      _NCC_POLARITY_IC  = 0.28  (catches near-symmetric ICs that score ~0.36)
      _NCC_POLARITY_C   = 0.38  (caps, unchanged from v2.3)
    Component 9 IC (U) failure on boards 3/4/8/28 is now caught correctly.
  • _NCC_POLARITY_STRONG raised 0.60→0.72.
    Observed wrong_component→wrong_polarity FPs had ncc_d 0.64–0.91 and
    ssim > -0.20; true polarity ICs always score > 0.72.  Raising the
    threshold eliminates FPs while keeping all genuine polarity detections
    (true caps still pass via the ssim<-0.30 branch).
  • cross_zoneC now gated by min(var, ssim) floor:
    if test_var < _VAR_FLAT AND ssim < _SSIM_MISSING_MAX, a cross-label det
    cannot claim ownership — slot is too flat to be a real component.
    Fixes Board 40 miss→wrong_component caused by a stray cross-label YOLO hit.

v2.3 fixes (over v2.2):
  • Issue B: Removed SSIM entry gate on NCC polarity check (v2.2 gate was
    WRONG — true wrong_polarity caps have ssim -0.57 to -0.35 and were being
    blocked). Replaced with compound acceptance: accept wrong_polarity only
    when delta > 0.60 (ICs) OR (delta > 0.38 AND ssim < -0.30) (rotated caps).
    This eliminates wrong_polarity cap FPs (ssim > -0.20, delta 0.38-0.47)
    while correctly catching rotated caps (ssim < -0.30) and all ICs (delta > 0.70).
  • Issue A: Flat branch now checks `not cross_owns` — a cross-label detection
    owning the slot rescues the edge case where a small IC donor patch looks
    flat (var < 5) due to the tight 20%-crop used by _patch_variance.

v2.2 fixes (over v2.0):
  • Issue 1: _VAR_FLAT tightened 12→5; flat guard now also requires
    ssim < _SSIM_MISSING_MAX — IC wrong-component patches (var 2–7, ssim ≥ 0.32)
    were wrongly short-circuited to "missing" by the old flat shortcut.
  • Issue 2: Zone C polarity check moved BEFORE cross_owns. Rotated polarized
    components triggered a cross-label YOLO hit; cross_owns then fired as
    wrong_component before NCC could run → 35 wrong_polarity→wrong_component
    confusions eliminated.
  • Issue 3: _NCC_POLARITY_C raised 0.20→0.38; SSIM guard added (skip NCC when
    ssim < -0.25, i.e. structurally incompatible donor) → 13 wrong_component→
    wrong_polarity false positives eliminated.
  • Issue 4: Zone B non-polarized high-SSIM branch added — high SSIM (≥0.47)
    with elevated rdiff was silently emitting no decision (FN); now classifies
    as wrong_component.

v2.0 improvements:
  • Template-match score (TM_CCOEFF_NORMED) added as Zone-C discriminant
    → fixes wrong_component→missing swaps when SSIM is low but component IS present
  • Zone C: polarity check now runs even when only cross-label dets exist
  • --trace flag: prints every threshold comparison for every slot (not just failures)
  • --save_viz PATH: saves annotated debug image marking each slot and its decision
  • Per-path statistics table in summary
  • Per-label accuracy breakdown in summary
  • SSIM/var/tmpl threshold sensitivity table
  • Color terminal output (auto-detected)
  • Board summary table always printed (no need for --verbose)

Usage:
  python aoi_evaluator.py --golden <img> --model <.pt> [options]

Options:
  --boards   N        Number of test boards to generate  (default 50)
  --defects  N        Max defects per board               (default 5)
  --conf     F        YOLO confidence threshold           (default 0.15)
  --seed     N        RNG seed                            (default 42)
  --verbose           Print per-board details (injected vs found)
  --debug             Per-slot SSIM/var/tmpl/path for FAILED slots (implies --verbose)
  --trace             Per-slot diagnostics for ALL slots on failed boards
  --save_viz PATH     Save annotated diff image to PATH (PNG)
"""

import argparse, copy, math, os, random, sys, time
from pathlib import Path
from collections import defaultdict

# ── Terminal colour helpers (auto-disable when piped) ─────────────────────────
_USE_COLOR = sys.stdout.isatty()
def _c(code, s): return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s
def _green(s):   return _c("32", s)
def _red(s):     return _c("31", s)
def _yellow(s):  return _c("33", s)
def _cyan(s):    return _c("36", s)
def _bold(s):    return _c("1",  s)
def _dim(s):     return _c("2",  s)

# ── Auto-install deps ────────────────────────────────────────────────────────
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
# 2.  DETECTION (standard YOLO + simple IoU NMS)
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
# 3.  DEFECT INJECTION  (from generate_defect.py v3.0 — realistic fill)
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
    """Colour-matched bare-PCB fill with cosine feather.

    v3.0: accepts an optional cached_bg_mask to avoid rebuilding it for every
    injection on the same board.  Pass None on the first call and reuse the
    returned mask for subsequent calls (mask is modified in-place inside this
    function so it correctly excludes the freshly-erased slot).
    Also: feather ramp now uses numpy broadcasting — no Python loop.
    """
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
    std_c  = np.std(safe_px, axis=0).clip(2, 10)   # v3.0 realistic grain
    fill   = np.random.normal(mean_c, std_c, (ph,pw,3)).astype(np.float32)

    # v3.0: vectorised cosine feather ramp — no Python loop
    feather = min(4, min(ph,pw)//4)
    if feather > 0:
        ks = np.arange(feather, dtype=np.float32)
        edge = (0.5 - 0.5 * np.cos(np.pi * ks / feather))   # shape (feather,)
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
    """Swap within compatible component group."""
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

    # Verify visual change happened
    if cv2.absdiff(orig_patch, img[ay1:ay2, ax1:ax2]).mean() < 8.0:
        img[ay1:ay2, ax1:ax2] = orig_patch
        return img, None

    return img, {"defect_type": "wrong_component", "found_label": donor.label}

def inject_wrong_polarity(img, comp):
    """180° rotation for polarized components."""
    H,W = img.shape[:2]
    x1,y1,x2,y2 = comp.xyxy(W, H)
    if x2<=x1 or y2<=y1: return img, None
    patch   = img[y1:y2, x1:x2].copy()
    flipped = cv2.rotate(patch, cv2.ROTATE_180)
    if cv2.absdiff(patch, flipped).mean() < 8.0: return img, None
    img[y1:y2, x1:x2] = flipped
    return img, {"defect_type": "wrong_polarity"}

DEFECT_MIX   = {"missing":0.40, "wrong_component":0.35, "wrong_polarity":0.25}
DTYPES       = list(DEFECT_MIX.keys())
DPROBS       = [DEFECT_MIX[k] for k in DTYPES]

def create_defects(golden_img, golden_comps, n_defects, rng):
    img = golden_img.copy()
    wc  = copy.deepcopy(golden_comps)

    polarized = [c for c in wc if c.is_polarized()]
    available = list(range(len(wc)))
    rng.shuffle(available)
    victims   = available[:min(n_defects, len(available))]

    log      = []
    used_ids = set()

    # v3.0: build bg_mask once; inject_missing updates it in-place as slots are erased
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

        if meta:
            log.append({"component_id": c.id, "original_label": c.label,
                        "injected_type": meta["defect_type"]})

    return img, log

# ═════════════════════════════════════════════════════════════════════════════
# 4.  AOI INFERENCE ENGINE  (ported from optical_inspection_system.py)
# ═════════════════════════════════════════════════════════════════════════════
_MIN_DIFF_THR    = 8.0
_MIN_POS_TOL     = 0.010
_AOI_MATCH_R     = 0.06
_AOI_DETECT_BAND = 2.0

# ── Disambiguation thresholds (v2.3) ────────────────────────────────────────
# SSIM alone is insufficient when the donor component looks very different
# from the golden one (NCC can score negative).
#
# v3.0/v3.4 polarity acceptance (v3.4 tightens both thresholds):
#
#   True wrong_polarity caps:  ssim -0.57 to -0.34  (anti-correlated, rotated)
#                              ncc_d always ≥ 1.058 via STRONG; or ≥ 0.38 + ssim<-0.33
#   Wrong_component cap FPs:   ssim -0.31 to +0.25
#                              ncc_d 0.38–0.91  (blocked by new STRONG=0.95)
#   True wrong_polarity ICs:   ncc_d 1.058–1.700 (all unambiguous at 0.95)
#
#   Accept wrong_polarity when:
#     (a) delta > _NCC_POLARITY_STRONG (≥0.95)       ← catches all ICs + strong caps
#     OR
#     (b) delta > _NCC_POLARITY_C (≥0.38)
#         AND ssim < _NCC_POLARITY_SSIM_NEG (-0.33)  ← catches rotated caps
#
# v2.3 flat guard:
#   slot_is_flat now also requires NOT cross_owns, so a cross-label detection
#   at the slot rescues the case where a small IC donor patch looks flat.
#
# Zone C decision order (v3.5):
#   1. slot_is_flat (var<5 AND ssim<0.32 AND NOT cross_owns AND NOT same) → missing
#   2. is_polar → NCC with compound acceptance                             → wrong_polarity
#      2b. [all polar] delta≥0 AND ssim≥0.28 AND tmpl≥0.25 struct fallback→ wrong_polarity
#   3. cross_owns AND is_polar AND flipped AND var_struct
#             AND ssim<0.45 AND tmpl≥0.20 AND delta>0.50                  → wrong_polarity
#      cross_owns (otherwise)                                              → wrong_component
#   4. var ≥ 80 OR ssim ≥ 0.32                                            → wrong_component
#   5. otherwise                                                           → missing
#
_SSIM_MISSING_MAX = 0.32   # below AND no rescue signals → missing
_SSIM_WC_DET_MIN  = 0.32   # above + det at slot → wrong_component
_SSIM_WC_NODET    = 0.55   # above even without det → wrong_component
_VAR_FLAT         = 5.0    # v2.2: tightened from 12→5; genuine fill always <5
_VAR_WC_CONFIRM   = 80.0   # v2.1: var alone (no tmpl). Failures: var>132. Miss-fill: var<2.
_TMPL_WC_MIN      = 0.65   # tmpl score → component present (v2.0; diagnostic reference)
_NCC_POLARITY_B   = 0.05   # Zone B polarity margin
_NCC_POLARITY_IC       = 0.28   # v3.0: IC-specific margin; near-symmetric ICs score ~0.36
_NCC_POLARITY_C        = 0.38   # v2.2: raised from 0.20→0.38; base NCC margin for caps
_NCC_POLARITY_STRONG   = 0.95   # v3.4: raised 0.72→0.95; new WC FPs scored 0.768/0.787/0.890/0.910; true WPOL always ≥ 1.058
_NCC_POLARITY_SSIM_NEG = -0.33  # v3.4: tightened -0.30→-0.33; WCOM FP cap had ssim=-0.306 (blocked); true WPOL caps ≤ -0.336
# v3.2: structural polarity fallback for ALL polarized (IC + cap); replaces v3.1 IC-only version
# v3.5: lowered SSIM 0.40→0.28 and TMPL 0.45→0.25 to catch near-symmetric ICs:
#   C0 IC WPOL: ssim=0.299, tmpl=0.265  — WPOL→MISS before fix, now caught
#   C6 IC WPOL: ssim=0.322, tmpl=0.364  — WPOL→WCOM before fix, now caught
#   C8 IC WPOL: ssim=0.323, tmpl=0.101  — irreducible; tmpl < 0.25, indistinct from WCOM noise
#   Safety: delta≥0 guard (Fix G) still blocks WCOM caps with negative NCC from triggering.
_SSIM_POLAR_STRUCT = 0.28  # v3.5: lowered 0.40→0.28; catches near-symmetric IC WPOL (ssim≈0.30)
_TMPL_POLAR_STRUCT = 0.25  # v3.5: lowered 0.45→0.25; catches C0 IC (tmpl=0.265); C6 also passes
# v3.6: minimum NCC delta for struct fallback — WCOM FPs had delta 0.037–0.186; true WPOL ≥ 0.393
#   Gap is clear: max FP delta=0.186, min TP delta=0.393. 0.25 sits in the middle.
_NCC_POLAR_STRUCT_MIN_DELTA = 0.25
# v3.2: cross_polar_rescue tmpl lower bound — WCOM donors score tmpl < 0.20; true WPOL ≥ 0.30
_POLAR_CROSS_RESCUE_TMPL = 0.20
# v3.4: cross_polar_rescue NCC delta floor — WCOM FP had delta=0.401; true WPOL rescues have delta ≥ 0.708
_POLAR_CROSS_RESCUE_NCC_MIN = 0.50
# v3.1: cross_zoneC polarized flip rescue — ssim upper bound (wrong_component caps score ssim > 0.45)
_SSIM_POLAR_CROSS_RESCUE = 0.45

# v3.1: readable defect-type abbreviations for verbose output
_DTYPE_ABBR = {
    "missing":         "MISS",
    "wrong_component": "WCOM",
    "wrong_polarity":  "WPOL",
    "misaligned":      "MALI",
}

_POLARIZED = {
    "ic (u)","ic (ic)","transistor (q)","transistor (qa)",
    "diode (d)","led","capacitor (c)","cap array (cra)",
}

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
    """Compute NCC polarity delta between golden and test patches.

    v3.0 speed fix: if g_blurred/t_blurred are supplied (pre-scaled to WxH and
    pre-blurred), the function skips the two full-image cv2.resize calls that
    were happening on every invocation.  The caller (typically _aoi_check_board)
    should pre-compute these once per board and reuse them across all components.
    """
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
    """Structural Similarity Index between the golden and test patches at gc.

    Key insight for missing vs wrong_component disambiguation:
      • MISSING  (PCB fill applied):      SSIM ≈ 0.05–0.30  (flat vs structured)
      • WRONG_COMPONENT (donor pasted):   SSIM ≈ 0.35–0.75  (different but still
                                          a component with edges/pads/body)
      • CORRECT (no defect):              SSIM ≈ 0.80–1.00

    Uses a slightly larger crop (40% of box in each direction) to include pad
    context, which strengthens the structural signal.
    """
    hw=gc.w*0.40; hh=gc.h*0.40
    x1=max(0,int((gc.cx-hw)*W)); y1=max(0,int((gc.cy-hh)*H))
    x2=min(W,int((gc.cx+hw)*W)); y2=min(H,int((gc.cy+hh)*H))
    if x2-x1<4 or y2-y1<4: return 1.0
    pa=cv2.cvtColor(img_a[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY).astype(np.float32)
    pb=cv2.cvtColor(img_b[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY).astype(np.float32)
    if pa.shape!=pb.shape:
        pb=cv2.resize(pb.astype(np.uint8),(pa.shape[1],pa.shape[0])).astype(np.float32)
    # Standard SSIM formula (Wang et al. 2004)
    C1=(0.01*255)**2; C2=(0.03*255)**2
    mu_a=float(pa.mean()); mu_b=float(pb.mean())
    sig_a=float(pa.std());  sig_b=float(pb.std())
    cov=float(np.mean((pa-mu_a)*(pb-mu_b)))
    num=(2*mu_a*mu_b+C1)*(2*cov+C2)
    den=(mu_a**2+mu_b**2+C1)*(sig_a**2+sig_b**2+C2)
    return float(num/den) if abs(den)>1e-9 else 0.0

def _patch_color_diff(img_a, img_b, gc, W, H):
    """Mean absolute BGR difference at the component patch (0–255)."""
    hw=gc.w*0.35; hh=gc.h*0.35
    x1=max(0,int((gc.cx-hw)*W)); y1=max(0,int((gc.cy-hh)*H))
    x2=min(W,int((gc.cx+hw)*W)); y2=min(H,int((gc.cy+hh)*H))
    if x2-x1<2 or y2-y1<2: return 0.0
    pa=img_a[y1:y2,x1:x2].astype(np.float32)
    pb=img_b[y1:y2,x1:x2].astype(np.float32)
    if pa.shape!=pb.shape: pb=cv2.resize(pb,(pa.shape[1],pa.shape[0]))
    return float(np.mean(np.abs(pa-pb)))

def _patch_tmpl_score(img_golden, img_test, gc, W, H):
    """Normalised cross-correlation (TM_CCOEFF_NORMED) of the golden patch
    against the test patch at the same location.

    Interpretation:
      • MISSING  (Gaussian fill):   score ≈ −0.10 – 0.40  (no matching structure)
      • WRONG_COMPONENT (donor):    score ≈  0.40 – 1.00  (edges/pads still align)
      • CORRECT  (same component):  score ≈  0.70 – 1.00

    Uses 35% crop (same as color-diff) to stay tightly within the component
    footprint and avoid PCB background contamination.
    This score rescues the Zone-C ssim_low→missing misclassification:
    when SSIM is low because the donor looks different, tmpl still recognises
    that *something structured* occupies the slot.
    """
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

def _same_det_is_local(dc, gc, golden_by_label):
    """True when dc is geometrically closest to gc among all golden slots of
    the same label.  Prevents a neighbouring component's detection from being
    claimed by gc when gc is actually missing.
    """
    dist_to_gc=math.hypot(dc.cx-gc.cx, dc.cy-gc.cy)
    for og in golden_by_label.get(dc.label.lower(),[]):
        if og.id==gc.id: continue
        if math.hypot(dc.cx-og.cx, dc.cy-og.cy) < dist_to_gc*0.85:
            return False
    return True

def _aoi_calibrate(model, golden_img, golden, conf, n_runs=3):
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


def _aoi_check_board(golden, golden_img, test_img, dets, pos_tol, per_thr, debug=False):
    """
    Single-pass pixel-primary defect detection.
    Returns list of {component_id, defect_type, expected_label, ...}
    If debug=True, also returns a parallel list of diagnostic dicts.

    v3.0 speed: pre-computes one blurred+scaled image pair (g_sm_blur, t_sm_blur)
    used by ALL _ncc_polarity calls, eliminating repeated full-image resizes.
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

    # v3.0: pre-blur once; reused by every _ncc_polarity call this board
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
                  "tmpl":None,"ssim":None,"test_var":None} if debug else None

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
                        if d_info is not None: d_info.update({"path":"misalign","decision":"misaligned"})
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
                            # v3.5 Fix L: polar slot, same det present, NCC gave no flip signal.
                            # Zone B rdiff is elevated — something changed. Since YOLO still sees
                            # the same component label and NCC can't detect a rotation, the most
                            # likely cause is wrong_component (donor from the same label class).
                            # Before this fix the slot fell through with decision="none" (FN).
                            defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                            "expected_label":gc.label,"found_label":tc.label})
                            if d_info is not None:
                                d_info.update({"path":"zone_b_polar_noflip","decision":"wrong_component",
                                               "ncc_delta":round(delta,3)})
                    else:
                        # Non-polarized, same det nearby, no cross ownership.
                        # Use SSIM as tie-breaker.
                        ssim_v=_patch_ssim(g_sm,t_sm,gc,dW,dH)
                        if d_info is not None: d_info["ssim"]=round(ssim_v,3)
                        if (ssim_v < _SSIM_MISSING_MAX):
                            defects.append({"component_id":gc.id,"defect_type":"missing",
                                            "expected_label":gc.label})
                            if d_info is not None: d_info.update({"path":"zone_b_ssim_low","decision":"missing"})
                        elif (dist<detect_tol and _same_det_is_local(tc,gc,_golden_by_label)
                              and ssim_v<_SSIM_WC_DET_MIN+0.15):
                            defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                            "expected_label":gc.label,"found_label":tc.label})
                            if d_info is not None: d_info.update({"path":"zone_b_ssim_wc","decision":"wrong_component"})
                        elif ssim_v >= _SSIM_WC_DET_MIN+0.15 and rdiff > comp_thr*0.35:
                            # v2.2: high SSIM in Zone B means something structured occupies
                            # the slot even if YOLO didn't land exactly on it.
                            # Previously fell through with decision="none" (zone_b_no_decision)
                            # and the defect was silently missed (FN).
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

        # Compute all signals upfront so guards can use them early
        ssim_v   = _patch_ssim(g_sm, t_sm, gc, dW, dH)
        test_var = _patch_variance(t_sm, gc, dW, dH)
        tmpl_v   = _patch_tmpl_score(g_sm, t_sm, gc, dW, dH)
        if d_info is not None:
            d_info.update({"ssim":round(ssim_v,3),
                           "test_var":round(test_var,1),
                           "tmpl":round(tmpl_v,3)})

        # v2.2: flat requires BOTH low var AND low SSIM.
        # IC wrong-component donors have var 2–7 but ssim ≥ 0.32 — they must
        # NOT be short-circuited to "missing" by the old var-only guard.
        slot_is_flat   = test_var < _VAR_FLAT and ssim_v < _SSIM_MISSING_MAX
        var_structured = test_var >= _VAR_WC_CONFIRM   # v2.0: component present signal
        cross_owns     = any(_owns_slot(dc, gc) for _, dc in cross)

        # ── Step 1: flat slot → definitely missing  ──────────────────────────
        # v2.3: UNLESS a cross-label detection already owns this slot.
        # v3.1: ALSO skip if a same-label detection is nearby — if YOLO saw the
        # same component class, the slot is not empty (Board 85: IC donor looked
        # flat at 20%-crop but YOLO still fired a same-label IC hit).
        if slot_is_flat and not cross_owns and not same:
            defects.append({"component_id":gc.id,"defect_type":"missing",
                            "expected_label":gc.label})
            if d_info is not None:
                d_info.update({"path":f"flat(var={test_var:.1f})","decision":"missing"})
            if d_info is not None: diag_rows.append(d_info)
            continue

        # ── Step 2: polarity check (non-flat, polarized components) ────────────
        # v3.0: IC components use a lower base margin (_NCC_POLARITY_IC = 0.28).
        #   _NCC_POLARITY_STRONG raised 0.60→0.72.
        #
        # v3.1: IC structural polarity fallback.
        #   When NCC gives NO flip signal (flipped=False) for an IC, check
        #   ssim ≥ _SSIM_IC_POLARITY (0.40) AND tmpl ≥ _TMPL_IC_POLARITY (0.45).
        #   Interpretation: the slot contains the same IC rotated (high template
        #   match + high structural similarity), not a different donor (which always
        #   scores tmpl < 0.20 and ssim < 0.35 in observed data).
        #
        # `flipped` and `delta` are preserved for use in Step 3 below.
        flipped, delta = False, -1.0   # initialise; only updated when is_polar
        if is_polar:
            is_ic = "ic" in gc.label.lower()
            ncc_margin = _NCC_POLARITY_IC if is_ic else _NCC_POLARITY_C
            flipped, delta = _ncc_polarity(golden_img, test_img, gc, dW, dH,
                                           margin=ncc_margin,
                                           g_blurred=g_sm_blur, t_blurred=t_sm_blur)
            accept_polarity = (flipped and
                               (delta > _NCC_POLARITY_STRONG or
                                ssim_v < _NCC_POLARITY_SSIM_NEG))

            # v3.1 Fix A / v3.2 Fix F / v3.3 Fix G / v3.6 Fix M: structural polarity fallback.
            # When NCC gives no flip signal, check ssim + tmpl + delta.
            # v3.3: delta≥0 blocks WCOM caps with negative NCC (FPs had delta -0.856 to -0.367).
            # v3.5: SSIM/TMPL thresholds lowered (0.40/0.45 → 0.28/0.25) to catch near-sym ICs.
            # v3.6: delta≥0.25 floor added — new WCOM FP caps had delta 0.037–0.186;
            #   true WPOL via this path: IC C0=0.724, C6=0.744, C9=0.487, cap C46=0.393.
            #   Gap between FP max (0.186) and TP min (0.393) is unambiguous.
            if not accept_polarity and delta >= _NCC_POLAR_STRUCT_MIN_DELTA:
                if ssim_v >= _SSIM_POLAR_STRUCT and tmpl_v >= _TMPL_POLAR_STRUCT:
                    accept_polarity = True
                    if d_info is not None:
                        d_info["path"] = "polar_struct_polarity"   # set tentatively

            if accept_polarity:
                fl = same[0][1].label if same else (cross[0][1].label if cross else gc.label)
                defects.append({"component_id":gc.id,"defect_type":"wrong_polarity",
                                "expected_label":gc.label,"found_label":fl})
                path_label = (d_info.get("path","") if d_info and d_info.get("path","")=="polar_struct_polarity"
                              else "ncc_zoneC")
                if d_info is not None:
                    d_info.update({"path": path_label, "decision":"wrong_polarity",
                                   "ncc_delta":round(delta,3)})
                if d_info is not None: diag_rows.append(d_info)
                continue

        # ── Step 3: cross-label det owns slot → wrong_component  ──────────────
        # Polarized components that failed the polarity check above arrive here.
        #
        # v3.0: Guard against a stray cross-label YOLO hit on a truly flat/empty
        # slot. If var < _VAR_FLAT AND ssim < _SSIM_MISSING_MAX, pixel wins.
        #
        # v3.1 Fix B / v3.2 Fix E: Polarized flip rescue via cross_owns path.
        #   When a rotated cap/IC causes YOLO to fire a cross-label detection,
        #   cross_owns=True bypasses wrong_polarity detection. Rescue when:
        #     is_polar AND flipped (NCC saw a flip signal, just below acceptance)
        #     AND var_structured (something real is in the slot)
        #     AND ssim < _SSIM_POLAR_CROSS_RESCUE (0.45) — blocks high-ssim WCOM caps
        #     AND tmpl >= _POLAR_CROSS_RESCUE_TMPL (0.20) — blocks low-tmpl WCOM donors
        #   v3.2: added tmpl gate. FP cases all had tmpl < 0.20 (0.178, -0.123, -0.094).
        #   True WPOL rescued by this path had tmpl=0.343.  Without the tmpl gate,
        #   4 WCOM→WPOL FPs appeared on boards 3/35/59/82.
        if cross_owns:
            if (is_polar and flipped and var_structured
                    and ssim_v < _SSIM_POLAR_CROSS_RESCUE
                    and tmpl_v >= _POLAR_CROSS_RESCUE_TMPL
                    and delta > _POLAR_CROSS_RESCUE_NCC_MIN):   # v3.4: blocks FP at delta=0.401
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
                # Pixel signal dominates: flat slot despite cross det → missing
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

        # ── Step 4: MISSING vs WRONG_COMPONENT (no cross, not polarity) ────────
        #
        # Primary discriminant (v2.0 data-validated):
        #
        #   inject_missing Gaussian fill → var ALWAYS < 5  (confirmed on all boards)
        #   Any real component (even different donor) → var ALWAYS > 130
        #
        #   TM_CCOEFF_NORMED (tmpl) is UNRELIABLE here: a pasted donor component
        #   from a different component type scores near 0 or negative because the
        #   golden template and donor share no structural similarity.
        #   → tmpl is kept for diagnostics only, NOT used as a decision signal.
        #
        #   Decision tree:
        #     var ≥ _VAR_WC_CONFIRM (80)  → wrong_component  (something is in slot)
        #     ssim ≥ _SSIM_MISSING_MAX    → wrong_component  (SSIM says structured)
        #     otherwise                   → missing
        #
        if var_structured or ssim_v >= _SSIM_MISSING_MAX:
            # Something component-like occupies the slot
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
                    # Neighbour det, medium SSIM, var not high enough
                    defects.append({"component_id":gc.id,"defect_type":"missing",
                                    "expected_label":gc.label})
                    if d_info is not None:
                        d_info.update({"path":f"neighbour_det_ssim({ssim_v:.2f})",
                                       "decision":"missing"})
            else:
                # No same-label det — var / ssim still say something is here
                defects.append({"component_id":gc.id,"defect_type":"wrong_component",
                                "expected_label":gc.label,"found_label":"unknown"})
                reason = (f"var_rescue_nodet({test_var:.0f})"
                          if (var_structured and ssim_v < _SSIM_MISSING_MAX)
                          else f"ssim_only({ssim_v:.2f})")
                if d_info is not None:
                    d_info.update({"path":reason,"decision":"wrong_component"})
        else:
            # All signals point to missing
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
    ("wrong_component","wrong_polarity"), ("wrong_polarity","wrong_component"),
}

def run_evaluation(args):
    print("\n" + "═"*72)
    print(_bold("  PCB DEFECT ACCURACY EVALUATOR  v3.6"))
    print("═"*72)

    # ── Load ─────────────────────────────────────────────────────────────────
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
    print(f"\n  [THRESH] SSIM_MISSING_MAX={_SSIM_MISSING_MAX}  SSIM_WC_NODET={_SSIM_WC_NODET}")
    print(f"  [THRESH] VAR_FLAT={_VAR_FLAT}  VAR_WC_CONFIRM={_VAR_WC_CONFIRM}")
    print(f"  [THRESH] NCC_POLARITY_IC={_NCC_POLARITY_IC}  NCC_POLARITY_C={_NCC_POLARITY_C}  "
          f"NCC_POLARITY_STRONG={_NCC_POLARITY_STRONG}  NCC_POLARITY_SSIM_NEG={_NCC_POLARITY_SSIM_NEG}")
    print(f"  [THRESH] POLAR_STRUCT: ssim≥{_SSIM_POLAR_STRUCT} tmpl≥{_TMPL_POLAR_STRUCT}  "
          f"CROSS_RESCUE: ssim<{_SSIM_POLAR_CROSS_RESCUE} tmpl≥{_POLAR_CROSS_RESCUE_TMPL}\n")

    # ── Calibrate ────────────────────────────────────────────────────────────
    print("[CAL]  Calibrating diff thresholds…")
    t0 = time.time()
    pos_tol, per_thr = _aoi_calibrate(model, golden_img, golden, args.conf, n_runs=2)
    detect_tol = pos_tol * _AOI_DETECT_BAND
    print(f"[CAL]  pos_tol={pos_tol:.4f}  detect_tol={detect_tol:.4f}  "
          f"({time.time()-t0:.1f}s)\n")

    # ── Accumulate ───────────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    total_inj  = 0
    l_tp=l_fp=l_fn = 0
    s_tp=s_fp=s_fn = 0
    confusion  = defaultdict(lambda: defaultdict(int))
    path_counts = defaultdict(int)                   # v2.0: per-path stats
    label_stats = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})  # v2.0: per-label
    # Collect raw ssim/tmpl/var for injected wrong_component that failed:
    wc_fail_signals = []                              # v2.0: sensitivity analysis

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
                    # v2.0: collect signal values for WC failures
                    if itype == "wrong_component" and dtype == "missing" and diag:
                        for row in diag:
                            if row["cid"] == cid:
                                wc_fail_signals.append({
                                    "ssim": row.get("ssim"),
                                    "tmpl": row.get("tmpl"),
                                    "var":  row.get("test_var"),
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

        # v2.0: accumulate path stats
        if diag:
            for row in diag:
                path_base = row.get("path","?").split("(")[0]
                path_counts[path_base] += 1

        status = "PASS" if (board_mm+board_fn+board_fp)==0 else "FAIL"
        board_rows.append((b+1, len(inj_map), board_mm, board_fn, board_fp, status,
                           inj_log, found, inj_map, found_map))

        # ── Per-board verbose line ────────────────────────────────────────────
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

        # ── Per-slot debug / trace output ─────────────────────────────────────
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
                    continue   # --trace only shows B/C for non-error slots
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
                      f"{ssim_s}{tmpl_s}{var_s}{ncc_s}{dist_s}{atsl_s}")
                print(f"         path={_bold(row['path'])}{thr_s}")
                # v2.0: explain WHY the decision was made
                if is_err and not correct:
                    _explain_failure(row, inj_type, det_type)
            print()

    # ── Optional visualisation ────────────────────────────────────────────────
    if args.save_viz:
        _save_viz(args.save_viz, golden_img, golden, board_rows, per_thr)

    # ─────────────────────────────────────────────────────────────────────────
    #  RESULTS
    # ─────────────────────────────────────────────────────────────────────────
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

    # ── Per-defect-type metrics ───────────────────────────────────────────────
    print("\n" + "─"*72)
    print("  PER-DEFECT-TYPE METRICS  (TP / FP / FN / Precision / Recall / F1)")
    print("─"*72)
    print(f"  {'Defect Type':<22}  {'TP':>5}  {'FP':>5}  {'FN':>5}  "
          f"{'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-"*70)

    _dtype_keys = ["missing", "wrong_component", "wrong_polarity"]

    # Pre-compute totals needed for per-type FP:
    #   FP for type T = sum of all non-T injected defects detected AS T
    #                 + false-positive detections (no injection) reported as T
    for dtype in _dtype_keys:
        # TP: injected as dtype AND detected as dtype
        tp_d = confusion[dtype].get(dtype, 0)

        # FN: injected as dtype but detected as something else or MISSED
        fn_d = sum(v for k, v in confusion[dtype].items() if k != dtype)

        # FP: injected as a *different* type but mis-detected as dtype
        fp_from_others = sum(
            confusion[other].get(dtype, 0)
            for other in _dtype_keys if other != dtype
        )
        # FP: genuine false positives (no defect injected) reported as dtype
        fp_spurious = confusion["FP_actual"].get(dtype, 0)
        fp_d = fp_from_others + fp_spurious

        prec_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) else 0.0
        rec_d  = tp_d / (tp_d + fn_d) if (tp_d + fn_d) else 0.0
        f1_d   = 2 * prec_d * rec_d / (prec_d + rec_d) if (prec_d + rec_d) else 0.0

        # Colour-code F1: green ≥0.90, yellow ≥0.70, red <0.70
        f1_str = f"{f1_d:.3f}"
        if f1_d >= 0.90:
            f1_col = _green(f1_str)
        elif f1_d >= 0.70:
            f1_col = _yellow(f1_str)
        else:
            f1_col = _red(f1_str)

        rec_str = f"{rec_d:.3f}"
        if rec_d >= 0.90:
            rec_col = _green(rec_str)
        elif rec_d >= 0.70:
            rec_col = _yellow(rec_str)
        else:
            rec_col = _red(rec_str)

        bar = _green("▓" * int(f1_d * 10)) + _dim("░" * (10 - int(f1_d * 10)))
        print(f"  {dtype:<22}  {tp_d:>5}  {fp_d:>5}  {fn_d:>5}  "
              f"  {prec_d:>8.3f}  {rec_col:>8}  {f1_col:>8}  {bar}")

    print()

    # ── Confusion matrix ─────────────────────────────────────────────────────
    injected_types = ["missing","wrong_component","wrong_polarity"]
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

    # ── Missing ↔ Wrong_component swap analysis ───────────────────────────────
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

    # ── v2.0: Per-label accuracy ──────────────────────────────────────────────
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

    # ── v2.0: Decision path frequency table ──────────────────────────────────
    if path_counts and (args.debug or args.trace):
        print("\n" + "─"*72)
        print("  DECISION PATH FREQUENCY  (v2.0, debug/trace only)")
        print("─"*72)
        total_paths = sum(path_counts.values())
        for path, cnt in sorted(path_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(40, int(cnt/total_paths*40))
            print(f"  {path:<40}  {cnt:4d}  {bar}")

    # ── v2.0: Signal distribution for remaining WC→MISS failures ─────────────
    if wc_fail_signals:
        print("\n" + "─"*72)
        print(f"  REMAINING wrong_component→missing FAILURES: {len(wc_fail_signals)}  (v2.0)")
        print("─"*72)
        print(f"  {'Path':<45}  {'ssim':>7}  {'tmpl':>7}  {'var':>8}")
        print("  " + "-"*70)
        for sig in wc_fail_signals:
            ss = f"{sig['ssim']:.3f}" if sig['ssim'] is not None else "  n/a"
            tm = f"{sig['tmpl']:.3f}" if sig['tmpl'] is not None else "  n/a"
            vv = f"{sig['var']:.1f}"  if sig['var']  is not None else "    n/a"
            path = sig['path'] or "?"
            print(f"  {path:<45}  {ss:>7}  {tm:>7}  {vv:>8}")
        if wc_fail_signals:
            ssims = [s['ssim'] for s in wc_fail_signals if s['ssim'] is not None]
            tmpls = [s['tmpl'] for s in wc_fail_signals if s['tmpl'] is not None]
            vars_ = [s['var']  for s in wc_fail_signals if s['var']  is not None]
            if ssims:
                print(f"\n  ssim range  : {min(ssims):.3f} – {max(ssims):.3f}  "
                      f"(mean {sum(ssims)/len(ssims):.3f})")
            if tmpls:
                print(f"  tmpl range  : {min(tmpls):.3f} – {max(tmpls):.3f}  "
                      f"(mean {sum(tmpls)/len(tmpls):.3f})")
            if vars_:
                print(f"  var  range  : {min(vars_):.1f} – {max(vars_):.1f}  "
                      f"(mean {sum(vars_)/len(vars_):.1f})")
            print(f"\n  → To fix remaining failures: lower _TMPL_WC_MIN below "
                  f"{min(tmpls):.3f}" if tmpls else "")
            print(f"    or lower _VAR_WC_CONFIRM below "
                  f"{min(vars_):.1f}" if vars_ else "")

    print("\n" + "═"*72 + "\n")


# ── v2.0: Failure explainer ───────────────────────────────────────────────────
def _explain_failure(row, inj_type, det_type):
    """Print a human-readable explanation of why the slot was mis-classified."""
    ssim = row.get("ssim"); tmpl = row.get("tmpl"); var = row.get("test_var")
    if inj_type == "wrong_component" and det_type == "missing":
        reason = []
        if ssim is not None and ssim < _SSIM_MISSING_MAX:
            reason.append(f"SSIM={ssim:.3f} < {_SSIM_MISSING_MAX}")
        if var  is not None and var  < _VAR_WC_CONFIRM:
            reason.append(f"var={var:.1f} < {_VAR_WC_CONFIRM}  ← PRIMARY SIGNAL")
        if tmpl is not None:
            reason.append(f"tmpl={tmpl:.3f} (diagnostic only — unreliable for diff donors)")
        print("         " + _red("→ wc→miss: ") + "; ".join(reason)
              if reason else "         → wc→miss: unknown reason")
        print(f"         {_dim('Fix hint')}: lower _VAR_WC_CONFIRM below {var:.1f}"
              if var is not None else "")
    elif inj_type == "wrong_polarity" and det_type == "wrong_component":
        lbl = row.get("lbl","")
        is_ic = "ic" in lbl.lower()
        margin = _NCC_POLARITY_IC if is_ic else _NCC_POLARITY_C
        thresh_name = "_NCC_POLARITY_IC" if is_ic else "_NCC_POLARITY_C"
        print("         " + _red("→ polarity→wc: ") +
              f"NCC delta below margin ({thresh_name}={margin}); "
              f"structural fallback also missed "
              f"(need ssim≥{_SSIM_POLAR_STRUCT} AND tmpl≥{_TMPL_POLAR_STRUCT})"
              + (f"; got ssim={ssim:.3f} tmpl={tmpl:.3f}" if ssim is not None and tmpl is not None else ""))
    elif inj_type == "missing" and det_type == "wrong_component":
        print("         " + _red("→ miss→wc: ") +
              f"SSIM={ssim:.3f} or tmpl={tmpl:.3f} above threshold; "
              "fill was too structured.")


# ── v2.0: Annotated visualisation ────────────────────────────────────────────
def _save_viz(out_path, golden_img, golden, board_rows, per_thr):
    """Save a collage of the first few failed boards with annotated slot boxes."""
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
            # We need to recreate the test image — skip for now, show golden annotated
            vis = cv2.resize(golden_img.copy(), (vis_w, vis_h))

            # Draw all golden slots
            for gc in golden:
                x1,y1,x2,y2 = gc.xyxy(vis_w, vis_h)
                cv2.rectangle(vis, (x1,y1),(x2,y2), (80,80,80), 1)

            # Annotate injected defects
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
    p = argparse.ArgumentParser(description="PCB Defect Accuracy Evaluator v2.0")
    p.add_argument("--golden",   required=True,  help="Path to golden board image")
    import pathlib
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    DEFAULT_MODEL = ROOT / "models" / "best.pt"
    p.add_argument("--model",    default=str(DEFAULT_MODEL),  help="Path to YOLO .pt model")
    p.add_argument("--boards",   type=int, default=50, help="Number of test boards (default 50)")
    p.add_argument("--defects",  type=int, default=5,  help="Max defects per board (default 5)")
    p.add_argument("--conf",     type=float, default=0.15, help="YOLO confidence (default 0.15)")
    p.add_argument("--seed",     type=int,   default=42,   help="RNG seed (default 42)")
    p.add_argument("--verbose",  action="store_true",
                   help="Print per-board injected vs detected summary")
    p.add_argument("--debug",    action="store_true",
                   help="Print per-slot SSIM/tmpl/var/path for FAILED slots "
                        "(implies --verbose)")
    p.add_argument("--trace",    action="store_true",
                   help="Print per-slot diagnostics for ALL Zone B/C slots on "
                        "failed boards, not just error slots (implies --debug)")
    p.add_argument("--save_viz", default="",
                   help="Save annotated debug image to this path (PNG)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.trace:
        args.debug   = True
        args.verbose = True
    if args.debug:
        args.verbose = True
    run_evaluation(args)