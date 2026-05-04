"""
aoi_engine.py — Core AOI inspection algorithms.

_AOIComp dataclass, all _aoi_* functions (bbox_overlap, nms_by_centre,
infer_arr, hungarian_match, region_diff, calibrate, ncc_polarity,
patch_ssim, patch_variance, patch_tmpl, same_det_is_local, check_board,
render_overlay), and OfflineAOIThread.
"""
import os, sys, time, math, copy
from dataclasses import dataclass as _dataclass

if __package__ in (None, ""):
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    __package__ = "ois"

from PySide6.QtCore import QThread, Signal

from .utils import (
    HAS_CV2, HAS_NP, HAS_YOLO, HAS_SAHI, _SAHI_MODEL_CACHE, safe_predict,
    load_optimized_yolo,
    _AOIComp, _AOI_SAHI_TRIGGER, _AOI_SLICE_HW, _AOI_OVERLAP, _AOI_MATCH_R,
    _aoi_get_sahi_model, _aoi_bbox_overlap, _aoi_nms_by_centre, _aoi_infer_arr
)
from .filters import apply_filters, run_roi

if HAS_CV2:
    import cv2
if HAS_NP:
    import numpy as np
if HAS_YOLO:
    from ultralytics import YOLO as _YOLO


_math = math   # local alias keeps all AOI engine call-sites unchanged


# ── Constants ────────────────────────────────────────────────────────────────
_MIN_DIFF_THR     = 8.0
_MIN_POS_TOL      = 0.010
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
    "misaligned":      (  0,165,255),
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
    DIFF_MAX   = 960
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
            # All signals say nothing is here → missing
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
            model = load_optimized_yolo(self._model_path, task="detect")
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
