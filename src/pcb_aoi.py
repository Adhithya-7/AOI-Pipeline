"""
PCB AOI - Automated Optical Inspection
Detects MISSING, MISALIGNED, and WRONG components against a golden board.
All paths are entered interactively - no command-line arguments needed.

Requirements:
    pip install ultralytics opencv-python numpy scipy
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

# ──────────────────────────────────────────────
#  DEFAULTS  (edit here or just press Enter at prompts)
# ──────────────────────────────────────────────
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = str(ROOT / "models" / "best.pt")
DEFAULT_OUTPUT_DIR    = r"D:\MELSS\AOI\aoi_output"
DEFAULT_CONF          = "0.25"
DEFAULT_IOU_MATCH     = "0.30"   # min IoU to consider two boxes the "same" component
DEFAULT_MISALIGN_PX   = "15"     # pixel distance threshold to flag misalignment
DEFAULT_MISALIGN_ANGLE= "8"      # rotation angle threshold (degrees) to flag misalignment
DEFAULT_WRONG_IOU     = "0.40"   # tighter IoU to flag wrong-class at a position


# ══════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════

def ask(prompt, default=""):
    if default:
        print(f"  {prompt}")
        print(f"    default: {default}")
        val = input("    → ").strip()
    else:
        val = input(f"  {prompt}: ").strip()
    return val if val else default


def ask_file(prompt, default="", must_exist=True):
    """Ask for a file path with existence check."""
    while True:
        path = ask(prompt, default).strip('"').strip("'")
        if not must_exist:
            return path
        if os.path.isfile(path):
            return path
        print(f"    [ERROR] File not found: {path}")
        print(f"    Please enter a valid path.")


def ask_dir(prompt, default=""):
    """Ask for a directory path (will be created if missing)."""
    path = ask(prompt, default).strip('"').strip("'")
    return path


def iou(a, b):
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def _patch_orientation_px(img, box):
    """Compute principal orientation angle of a component from its pixel bounding box.

    Uses Canny edge detection + cv2.minAreaRect on the largest contour.
    Falls back to image moments when contour analysis fails.

    Args:
        img:  BGR image (numpy array)
        box:  [x1, y1, x2, y2] pixel coordinates

    Returns:
        angle in degrees [0, 180), or None if patch is too small/featureless.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    patch = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: image moments (PCA-like orientation)
        import math
        M = cv2.moments(blurred)
        if abs(M["mu20"] - M["mu02"]) < 1e-6:
            return None
        angle = 0.5 * math.degrees(math.atan2(2 * M["mu11"],
                                                M["mu20"] - M["mu02"]))
        return angle % 180

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:
        return None

    rect = cv2.minAreaRect(largest)
    angle = rect[2]
    w_r, h_r = rect[1]

    # Normalise: minAreaRect angle of shorter side → [0, 180)
    if w_r < h_r:
        angle = angle + 90
    angle = angle % 180
    return angle


def _compute_angle_delta(golden_bgr, test_bgr, g_box, t_box):
    """Compute angular difference between golden and test component patches.

    Returns angle delta in degrees [0, 90], or 0.0 if orientation cannot be determined.
    """
    g_angle = _patch_orientation_px(golden_bgr, g_box)
    t_angle = _patch_orientation_px(test_bgr, t_box)

    if g_angle is None or t_angle is None:
        return 0.0

    delta = abs(g_angle - t_angle)
    if delta > 90:
        delta = 180 - delta
    return delta


# ══════════════════════════════════════════════
#  DETECTION
# ══════════════════════════════════════════════

def detect(model, img_path, conf):
    """Return list of dicts: {class, box:[x1,y1,x2,y2], conf}"""
    results = model(img_path, conf=conf, verbose=False)[0]
    comps = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        cls_id = int(box.cls[0])
        comps.append({
            "class"   : results.names[cls_id],
            "cls_id"  : cls_id,
            "box"     : [x1, y1, x2, y2],
            "conf"    : float(box.conf[0]),
        })
    return comps


# ══════════════════════════════════════════════
#  IMAGE ALIGNMENT  (ORB + Homography)
# ══════════════════════════════════════════════

def align_to_golden(golden_bgr, test_bgr):
    """Warp test_bgr so it lines up with golden_bgr. Returns (aligned, H)."""
    g_gray = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2GRAY)
    t_gray = cv2.cvtColor(test_bgr,   cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(6000)
    kp_g, des_g = orb.detectAndCompute(g_gray, None)
    kp_t, des_t = orb.detectAndCompute(t_gray, None)

    if des_g is None or des_t is None or len(des_g) < 10 or len(des_t) < 10:
        print("    [WARN] Not enough features for alignment – using raw image.")
        return test_bgr, np.eye(3)

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_g, des_t), key=lambda m: m.distance)[:300]

    if len(matches) < 10:
        print("    [WARN] Too few feature matches – using raw image.")
        return test_bgr, np.eye(3)

    pts_g = np.float32([kp_g[m.queryIdx].pt for m in matches])
    pts_t = np.float32([kp_t[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts_t, pts_g, cv2.RANSAC, 5.0)
    if H is None:
        print("    [WARN] Homography failed – using raw image.")
        return test_bgr, np.eye(3)

    h, w = golden_bgr.shape[:2]
    aligned = cv2.warpPerspective(test_bgr, H, (w, h))
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"    Aligned using {inliers}/{len(matches)} inlier matches.")
    return aligned, H


# ══════════════════════════════════════════════
#  COMPARISON LOGIC
# ══════════════════════════════════════════════

def compare(golden_comps, test_comps, iou_match, misalign_px, wrong_iou,
            golden_bgr=None, test_bgr=None, misalign_angle=8.0):
    """
    Returns four lists:
        missing    – in golden but nothing nearby in test
        misaligned – correct class found but center offset > threshold
                     OR rotation angle > misalign_angle threshold
        wrong      – something is at that location but different class
        extra      – in test but nothing nearby in golden
    """
    missing    = []
    misaligned = []
    wrong      = []
    extra      = []

    used_test = set()

    for g in golden_comps:
        # Find best-IoU test component
        best_iou_val  = 0.0
        best_idx      = -1
        for i, t in enumerate(test_comps):
            v = iou(g["box"], t["box"])
            if v > best_iou_val:
                best_iou_val = v
                best_idx     = i

        if best_iou_val < iou_match:
            # Nothing close enough → MISSING
            missing.append({"golden": g})
        else:
            t = test_comps[best_idx]
            used_test.add(best_idx)

            g_cx, g_cy = center(g["box"])
            t_cx, t_cy = center(t["box"])
            dist = float(np.hypot(g_cx - t_cx, g_cy - t_cy))

            if t["class"] != g["class"]:
                # Spatially overlapping but different label → WRONG
                wrong.append({
                    "golden"  : g,
                    "test"    : t,
                    "expected": g["class"],
                    "found"   : t["class"],
                    "iou"     : round(best_iou_val, 3),
                    "offset_px": round(dist, 1),
                })
            elif dist > misalign_px:
                # Correct class but shifted → MISALIGNED (offset)
                misaligned.append({
                    "golden"    : g,
                    "test"      : t,
                    "offset_px" : round(dist, 1),
                    "type"      : "offset",
                })
            else:
                # Same class, within centroid tolerance — check rotation
                if golden_bgr is not None and test_bgr is not None:
                    angle_delta = _compute_angle_delta(
                        golden_bgr, test_bgr, g["box"], t["box"])
                    if angle_delta > misalign_angle:
                        misaligned.append({
                            "golden"      : g,
                            "test"        : t,
                            "offset_px"   : round(dist, 1),
                            "angle_delta" : round(angle_delta, 1),
                            "type"        : "rotation",
                        })

    for i, t in enumerate(test_comps):
        if i not in used_test:
            extra.append({"test": t})

    return missing, misaligned, wrong, extra


# ══════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════

COLORS = {
    "MISSING"   : (0,   0,   255),   # red
    "MISALIGNED": (0,   140, 255),   # orange
    "WRONG"     : (255, 0,   200),   # magenta
    "EXTRA"     : (0,   220, 220),   # cyan
    "OK"        : (0,   200, 0  ),   # green
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, box, text, color, thickness=2, font_scale=0.45):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # background pill for text
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, 1)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(img, (x1, ty - th - 3), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(img, text, (x1 + 2, ty), FONT, font_scale, (255,255,255), 1, cv2.LINE_AA)


def annotate(base_img, golden_comps, missing, misaligned, wrong, extra, board_name):
    img = base_img.copy()

    # First draw all golden boxes light green (OK) so we have a background reference
    for g in golden_comps:
        x1,y1,x2,y2 = [int(v) for v in g["box"]]
        cv2.rectangle(img, (x1,y1),(x2,y2), (0,180,0), 1)

    for item in missing:
        g = item["golden"]
        _label(img, g["box"], f"MISSING {g['class']}", COLORS["MISSING"], 2)

    for item in misaligned:
        t  = item["test"]
        g  = item["golden"]
        mali_type = item.get("type", "offset")
        if mali_type == "rotation":
            angle = item.get("angle_delta", 0)
            label_text = f"MISALIGN(rot) {t['class']} {angle:.0f}deg"
        else:
            label_text = f"MISALIGN {t['class']} {item['offset_px']}px"
        _label(img, t["box"], label_text, COLORS["MISALIGNED"], 2)
        gc = tuple([int(v) for v in center(g["box"])])
        tc = tuple([int(v) for v in center(t["box"])])
        cv2.arrowedLine(img, gc, tc, COLORS["MISALIGNED"], 2, tipLength=0.3)

    for item in wrong:
        t = item["test"]
        _label(img, t["box"], f"WRONG exp:{item['expected']} got:{item['found']}", COLORS["WRONG"], 2)

    for item in extra:
        t = item["test"]
        _label(img, t["box"], f"EXTRA {t['class']}", COLORS["EXTRA"], 2)

    # HUD overlay
    status = "PASS" if (len(missing) + len(misaligned) + len(wrong)) == 0 else "FAIL"
    s_col  = COLORS["OK"] if status == "PASS" else COLORS["MISSING"]

    hud_h = 135
    hud = img[:hud_h, :420].copy()
    cv2.rectangle(img, (0,0), (420, hud_h), (20,20,20), -1)
    cv2.addWeighted(hud, 0.25, img[:hud_h, :420], 0.75, 0, img[:hud_h, :420])

    lines = [
        (f"Board : {board_name}",             (255,255,255)),
        (f"MISSING    : {len(missing)}",       COLORS["MISSING"]),
        (f"MISALIGNED : {len(misaligned)}",    COLORS["MISALIGNED"]),
        (f"WRONG      : {len(wrong)}",         COLORS["WRONG"]),
        (f"EXTRA      : {len(extra)}",         COLORS["EXTRA"]),
    ]
    for idx, (txt, col) in enumerate(lines):
        cv2.putText(img, txt, (12, 22 + idx*22), FONT, 0.55, col, 1, cv2.LINE_AA)

    cv2.putText(img, status, (320, 100), FONT, 2.0, s_col, 3, cv2.LINE_AA)

    # Legend bottom-right
    legend = [
        ("MISSING",    COLORS["MISSING"]),
        ("MISALIGNED", COLORS["MISALIGNED"]),
        ("WRONG",      COLORS["WRONG"]),
        ("EXTRA",      COLORS["EXTRA"]),
    ]
    lx, ly = img.shape[1] - 160, img.shape[0] - 100
    for i, (lbl, col) in enumerate(legend):
        cv2.rectangle(img, (lx, ly + i*22), (lx+14, ly + i*22 + 14), col, -1)
        cv2.putText(img, lbl, (lx+18, ly + i*22 + 12), FONT, 0.42, col, 1, cv2.LINE_AA)

    return img


# ══════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════

def build_report(board_name, status, golden_comps, test_comps,
                 missing, misaligned, wrong, extra):
    def comp_summary(c):
        return {"class": c["class"], "box": [round(v,1) for v in c["box"]], "conf": round(c["conf"],3)}

    return {
        "board" : board_name,
        "status": status,
        "summary": {
            "golden_total"  : len(golden_comps),
            "test_total"    : len(test_comps),
            "missing"       : len(missing),
            "misaligned"    : len(misaligned),
            "wrong"         : len(wrong),
            "extra"         : len(extra),
        },
        "details": {
            "missing"    : [{"expected": comp_summary(m["golden"])} for m in missing],
            "misaligned" : [{"component": comp_summary(m["test"]),
                             "offset_px": m["offset_px"]} for m in misaligned],
            "wrong"      : [{"expected_class": m["expected"],
                             "found_class"   : m["found"],
                             "box"           : [round(v,1) for v in m["test"]["box"]],
                             "iou"           : m["iou"]} for m in wrong],
            "extra"      : [{"component": comp_summary(e["test"])} for e in extra],
        }
    }


# ══════════════════════════════════════════════
#  GOLDEN BOARD ANNOTATION HELPER
# ══════════════════════════════════════════════

def draw_golden(img, comps):
    out = img.copy()
    for c in comps:
        x1,y1,x2,y2 = [int(v) for v in c["box"]]
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,220,0),2)
        cv2.putText(out, c["class"], (x1, max(y1-4,10)), FONT, 0.45, (0,220,0),1, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PCB AOI  –  Missing / Misaligned / Wrong Component     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ── CONFIG ──────────────────────────────────────────────────────────
    model_path   = ask_file("YOLO model (.pt) path", DEFAULT_MODEL_PATH)
    output_dir   = ask_dir("Output folder",          DEFAULT_OUTPUT_DIR)
    conf         = float(ask("Detection confidence (e.g. 0.25)", DEFAULT_CONF))
    iou_match    = float(ask("Component match IoU   (e.g. 0.30)", DEFAULT_IOU_MATCH))
    misalign_px  = float(ask("Misalign threshold px (e.g. 15)",   DEFAULT_MISALIGN_PX))
    misalign_ang = float(ask("Misalign angle thresh (e.g. 8)",    DEFAULT_MISALIGN_ANGLE))

    os.makedirs(output_dir, exist_ok=True)

    # ── MODE ─────────────────────────────────────────────────────────────
    print()
    print("  Mode:")
    print("    1  Extract golden only  (save reference for later)")
    print("    2  Check test boards    (load saved golden reference)")
    print("    3  Extract golden + check test boards  [default]")
    mode = ask("Select", "3")

    # ── LOAD MODEL ───────────────────────────────────────────────────────
    from ultralytics import YOLO
    print(f"\n  Loading model: {model_path}")
    model = YOLO(model_path)
    print("  Model loaded.")

    golden_json = os.path.join(output_dir, "golden_reference.json")
    golden_img_out = os.path.join(output_dir, "golden_annotated.jpg")
    golden_raw_out = os.path.join(output_dir, "golden_raw.jpg")

    golden_comps = None
    golden_bgr   = None

    # ── GOLDEN ───────────────────────────────────────────────────────────
    if mode in ("1", "3"):
        print()
        golden_path = ask_file("Golden board image path")

        print(f"\n  Detecting on golden board …")
        golden_bgr   = cv2.imread(golden_path)
        if golden_bgr is None:
            raise FileNotFoundError(f"Cannot read: {golden_path}")

        golden_comps = detect(model, golden_path, conf)

        # Count by class
        counts = {}
        for c in golden_comps:
            counts[c["class"]] = counts.get(c["class"], 0) + 1
        print(f"  Found {len(golden_comps)} components:")
        for cls, n in sorted(counts.items()):
            print(f"    {n:4d}  {cls}")

        # Save reference JSON + annotated image
        with open(golden_json, "w") as f:
            json.dump(golden_comps, f, indent=2)
        cv2.imwrite(golden_raw_out, golden_bgr)
        cv2.imwrite(golden_img_out, draw_golden(golden_bgr, golden_comps))
        print(f"\n  Golden reference saved → {golden_json}")
        print(f"  Annotated image saved  → {golden_img_out}")

        if mode == "1":
            print("\n  Done. Run again in mode 2 to check test boards.")
            input("  Press Enter …")
            return

    # ── LOAD SAVED GOLDEN (mode 2) ────────────────────────────────────────
    if mode == "2":
        gj = ask_file("Golden reference JSON",               golden_json)
        gi = ask_file("Golden reference image (alignment)",  golden_raw_out, must_exist=False)
        with open(gj) as f:
            golden_comps = json.load(f)
        golden_bgr = cv2.imread(gi)
        if golden_bgr is None:
            print("  [WARN] Golden image not found – alignment disabled.")
        print(f"  Loaded {len(golden_comps)} components from golden reference.")

    # ── TEST BOARDS ───────────────────────────────────────────────────────
    print()
    print("  Enter test board image paths. Empty line when done.")
    test_paths = []
    while True:
        p = input("    Test image path: ").strip().strip('"').strip("'")
        if not p:
            break
        test_paths.append(p)

    if not test_paths:
        print("  No test images provided. Exiting.")
        input("  Press Enter …")
        return

    all_results = []

    for tp in test_paths:
        board_name = Path(tp).stem
        print(f"\n{'─'*60}")
        print(f"  Board: {board_name}")

        test_bgr = cv2.imread(tp)
        if test_bgr is None:
            print(f"  [ERROR] Cannot read: {tp}")
            continue

        # Alignment
        if golden_bgr is not None:
            print("  Aligning to golden …")
            aligned_bgr, H = align_to_golden(golden_bgr, test_bgr)
        else:
            aligned_bgr, H = test_bgr, np.eye(3)

        # Save aligned image for inspection
        al_path = os.path.join(output_dir, f"{board_name}_aligned.jpg")
        cv2.imwrite(al_path, aligned_bgr)

        # Detect on aligned image (write to temp, detect, delete)
        tmp_path = os.path.join(output_dir, "_tmp_detect.jpg")
        cv2.imwrite(tmp_path, aligned_bgr)
        test_comps = detect(model, tmp_path, conf)
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        print(f"  Detected {len(test_comps)} components on test board.")

        # Compare
        missing, misaligned, wrong, extra = compare(
            golden_comps, test_comps, iou_match, misalign_px,
            float(DEFAULT_WRONG_IOU),
            golden_bgr=golden_bgr, test_bgr=aligned_bgr,
            misalign_angle=misalign_ang
        )

        status = "PASS" if (len(missing) + len(misaligned) + len(wrong)) == 0 else "FAIL"

        print(f"  ┌─ RESULT : {status}")
        print(f"  │  MISSING    {len(missing)}")
        if missing:
            for m in missing:
                cx,cy = center(m["golden"]["box"])
                print(f"  │    - {m['golden']['class']}  @ ({cx:.0f},{cy:.0f})")
        print(f"  │  MISALIGNED {len(misaligned)}")
        if misaligned:
            for m in misaligned:
                print(f"  │    - {m['golden']['class']}  offset={m['offset_px']}px")
        print(f"  │  WRONG      {len(wrong)}")
        if wrong:
            for m in wrong:
                print(f"  │    - expected {m['expected']}  found {m['found']}")
        print(f"  └  EXTRA      {len(extra)}")

        # Annotated result image
        result_img = annotate(aligned_bgr, golden_comps,
                              missing, misaligned, wrong, extra, board_name)
        res_path   = os.path.join(output_dir, f"{board_name}_result.jpg")
        cv2.imwrite(res_path, result_img)
        print(f"  Result image → {res_path}")

        # JSON report
        report = build_report(board_name, status, golden_comps, test_comps,
                               missing, misaligned, wrong, extra)
        rpt_path = os.path.join(output_dir, f"{board_name}_report.json")
        with open(rpt_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report       → {rpt_path}")

        all_results.append(report)

    # ── BATCH SUMMARY ──────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  BATCH SUMMARY")
    print(f"  {'Board':<30} {'Status':<8}  MISS  MISAL  WRONG  EXTRA")
    print(f"  {'─'*30} {'─'*7}  {'─'*4}  {'─'*5}  {'─'*5}  {'─'*5}")
    passed = failed = 0
    for r in all_results:
        s  = r["summary"]
        st = r["status"]
        if st == "PASS": passed += 1
        else:            failed  += 1
        print(f"  {r['board']:<30} {st:<8}  "
              f"{s['missing']:>4}  {s['misaligned']:>5}  "
              f"{s['wrong']:>5}  {s['extra']:>5}")
    print(f"\n  Boards checked: {len(all_results)}   PASS: {passed}   FAIL: {failed}")
    print(f"  All outputs in: {output_dir}")

    # Save batch summary JSON
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Batch summary  → {summary_path}")

    print()
    input("  Press Enter to exit …")


if __name__ == "__main__":
    main()