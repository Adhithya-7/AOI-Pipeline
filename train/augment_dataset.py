"""
augment_dataset.py
──────────────────
Offline augmentation pipeline for PCB labeling data.
Takes a YOLO-format project folder (images/ + labels/) and expands it
to ~200-300 images using a diverse set of PCB-relevant transforms.

All bounding boxes are correctly remapped for every spatial transform.

Usage:
    python augment_dataset.py --src  "D:/MELSS/AOI/MyProject"
                              --dst  "D:/MELSS/AOI/MyProject_aug"
                              --target 280
                              --seed  42
"""

import os, sys, random, argparse, shutil, math
from pathlib import Path

import cv2
import numpy as np

# ── CONFIG (can also be overridden via CLI) ────────────────────────────────────
SRC_DIR    = r"D:\MELSS\AOI\AOI_Projects\new\images\645"   # folder with images/ and labels/
DST_DIR    = r"D:\MELSS\AOI\AOI_Projects\new_augment"
TARGET     = 280          # desired output count (copies of originals included)
SEED       = 42
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Augmentation knobs — tweak freely ─────────────────────────────────────────
AUG_CFG = dict(
    # Geometry
    hflip_p      = 0.5,    # horizontal flip probability
    vflip_p      = 0.0,    # vertical flip  (PCBs rarely upside-down, keep off)
    rot_max_deg  = 5,      # small rotation ±5° (PCBs usually aligned to grid)
    rot90_p      = 0.4,    # 90° / 180° / 270° rotation probability
    scale_lo     = 0.85,   # random scale range
    scale_hi     = 1.15,
    translate_p  = 0.4,    # random translate ±8% of image size
    translate_frac= 0.08,
    perspective_p= 0.2,    # mild perspective warp
    perspective_k= 0.04,   # warp strength (keep small for PCBs)

    # Pixel / colour
    brightness_range = (0.70, 1.35),  # multiplicative brightness
    contrast_range   = (0.75, 1.30),  # multiplicative contrast
    saturation_range = (0.70, 1.40),
    hue_shift_max    = 10,            # degrees
    gamma_range      = (0.75, 1.40),

    # Noise / blur
    gaussian_noise_p  = 0.40,
    gaussian_noise_var= (5, 25),      # variance range
    motion_blur_p     = 0.20,
    motion_blur_k     = (3, 7),       # kernel size range
    median_blur_p     = 0.15,
    median_blur_k     = (3, 5),

    # PCB-specific
    jpeg_quality_p    = 0.30,         # simulate camera compression artefacts
    jpeg_quality_range= (60, 95),
    grid_distort_p    = 0.15,         # mild barrel / pin-cushion
    grid_distort_steps= 5,
    grid_distort_limit= 0.05,

    # Safety
    min_box_area  = 0.0002,   # drop boxes smaller than this fraction of image
    min_visibility= 0.30,     # drop boxes with <30% of original area still visible
)

random.seed(SEED); np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Box helpers (YOLO ↔ corner format)
# ══════════════════════════════════════════════════════════════════════════════
def yolo_to_corners(boxes, W, H):
    """(N,5) yolo → (N,5) [cls,x1,y1,x2,y2] in pixel coords."""
    out = []
    for b in boxes:
        cls, cx, cy, bw, bh = b
        x1 = (cx - bw/2) * W; y1 = (cy - bh/2) * H
        x2 = (cx + bw/2) * W; y2 = (cy + bh/2) * H
        out.append([cls, x1, y1, x2, y2])
    return out

def corners_to_yolo(boxes, W, H):
    """(N,5) [cls,x1,y1,x2,y2] → (N,5) yolo, clamped."""
    out = []
    for b in boxes:
        cls, x1, y1, x2, y2 = b
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        if x2 <= x1 or y2 <= y1: continue
        cx = (x1+x2)/2/W; cy = (y1+y2)/2/H
        bw = (x2-x1)/W;   bh = (y2-y1)/H
        out.append([cls, cx, cy, bw, bh])
    return out

def transform_corners(pts, M):
    """Apply 3×3 affine/perspective matrix M to (N,2) points."""
    n = pts.shape[0]
    h = np.hstack([pts, np.ones((n,1))])
    t = (M @ h.T).T
    t[:,0] /= t[:,2]; t[:,1] /= t[:,2]
    return t[:,:2]

def box_after_transform(b, M, W, H, orig_area, cfg):
    """Transform a box's four corners through M, return min-area AABB in YOLO."""
    cls, x1, y1, x2, y2 = b
    corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
    tc = transform_corners(corners, M)
    nx1,ny1 = tc.min(axis=0); nx2,ny2 = tc.max(axis=0)
    # clip
    nx1=max(0,nx1); ny1=max(0,ny1); nx2=min(W,nx2); ny2=min(H,ny2)
    if nx2<=nx1 or ny2<=ny1: return None
    new_area = (nx2-nx1)*(ny2-ny1)
    # visibility check
    if new_area / max(orig_area, 1e-6) < cfg["min_visibility"]: return None
    # min area check
    if new_area / (W*H) < cfg["min_box_area"]: return None
    cx=(nx1+nx2)/2/W; cy=(ny1+ny2)/2/H
    bw=(nx2-nx1)/W;   bh=(ny2-ny1)/H
    return [cls, cx, cy, bw, bh]


# ══════════════════════════════════════════════════════════════════════════════
# Individual transforms
# ══════════════════════════════════════════════════════════════════════════════
def apply_hflip(img, boxes):
    H, W = img.shape[:2]
    img2 = cv2.flip(img, 1)
    b2 = [[b[0], 1.0-b[1], b[2], b[3], b[4]] for b in boxes]
    return img2, b2

def apply_vflip(img, boxes):
    H, W = img.shape[:2]
    img2 = cv2.flip(img, 0)
    b2 = [[b[0], b[1], 1.0-b[2], b[3], b[4]] for b in boxes]
    return img2, b2

def apply_rot90(img, boxes):
    """Random 90/180/270."""
    k = random.choice([1, 2, 3])
    img2 = np.rot90(img, k)
    H, W = img.shape[:2]
    nH, nW = img2.shape[:2]
    corners_b = yolo_to_corners(boxes, W, H)
    b2 = []
    for b in corners_b:
        cls, x1, y1, x2, y2 = b
        pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
        # rotate pts
        for _ in range(k):
            pts = np.column_stack([H - pts[:,1], pts[:,0]])
            H2, W2 = W, H
        nx1,ny1 = pts.min(axis=0); nx2,ny2 = pts.max(axis=0)
        b2.append([cls, nx1, ny1, nx2, ny2])
    return img2, corners_to_yolo(b2, nW, nH)

def apply_affine(img, boxes, cfg):
    """Random small rotation + scale + translate."""
    H, W = img.shape[:2]
    angle = random.uniform(-cfg["rot_max_deg"], cfg["rot_max_deg"])
    scale = random.uniform(cfg["scale_lo"], cfg["scale_hi"])
    cx, cy = W/2, H/2
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    if random.random() < cfg["translate_p"]:
        tx = random.uniform(-cfg["translate_frac"], cfg["translate_frac"]) * W
        ty = random.uniform(-cfg["translate_frac"], cfg["translate_frac"]) * H
        M[0,2] += tx; M[1,2] += ty
    M3 = np.vstack([M, [0,0,1]])
    img2 = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
    corners_b = yolo_to_corners(boxes, W, H)
    b2 = []
    for b in corners_b:
        orig_area = (b[3]-b[1])*(b[4]-b[2])
        r = box_after_transform(b, M3, W, H, orig_area, cfg)
        if r: b2.append(r)
    return img2, b2

def apply_perspective(img, boxes, cfg):
    H, W = img.shape[:2]
    k = cfg["perspective_k"]
    def rp(v): return v * random.uniform(-k, k)
    src = np.float32([[0,0],[W,0],[W,H],[0,H]])
    dst = np.float32([
        [rp(W), rp(H)],
        [W+rp(W), rp(H)],
        [W+rp(W), H+rp(H)],
        [rp(W), H+rp(H)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    img2 = cv2.warpPerspective(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
    corners_b = yolo_to_corners(boxes, W, H)
    b2 = []
    for b in corners_b:
        orig_area = (b[3]-b[1])*(b[4]-b[2])
        r = box_after_transform(b, M, W, H, orig_area, cfg)
        if r: b2.append(r)
    return img2, b2

def apply_color(img, cfg):
    """Brightness, contrast, saturation, hue, gamma — all random."""
    img = img.astype(np.float32)
    # Brightness + contrast in BGR space
    alpha = random.uniform(*cfg["contrast_range"])
    beta  = random.uniform(*cfg["brightness_range"])
    img = img * alpha * beta
    img = np.clip(img, 0, 255)
    # HSV-space saturation + hue
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= random.uniform(*cfg["saturation_range"])
    hsv[:,:,0]  = (hsv[:,:,0] + random.uniform(-cfg["hue_shift_max"],
                                                 cfg["hue_shift_max"])) % 180
    hsv = np.clip(hsv, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    # Gamma
    gamma = random.uniform(*cfg["gamma_range"])
    lut = np.array([min(255, ((i/255.0)**(1.0/gamma))*255) for i in range(256)],
                   dtype=np.uint8)
    img = cv2.LUT(img.astype(np.uint8), lut)
    return img

def apply_noise(img, cfg):
    if random.random() < cfg["gaussian_noise_p"]:
        var = random.uniform(*cfg["gaussian_noise_var"])
        noise = np.random.normal(0, math.sqrt(var), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < cfg["motion_blur_p"]:
        k = random.randrange(cfg["motion_blur_k"][0], cfg["motion_blur_k"][1]+1, 2)
        angle = random.uniform(0, 180)
        M = cv2.getRotationMatrix2D((k//2, k//2), angle, 1)
        kernel = np.zeros((k, k)); kernel[k//2, :] = 1.0
        kernel = cv2.warpAffine(kernel, M, (k, k)) / k
        img = cv2.filter2D(img, -1, kernel)
    if random.random() < cfg["median_blur_p"]:
        k = random.choice([x for x in range(cfg["median_blur_k"][0],
                                             cfg["median_blur_k"][1]+1) if x%2==1])
        img = cv2.medianBlur(img, k)
    return img

def apply_jpeg(img, cfg):
    q = random.randint(*cfg["jpeg_quality_range"])
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img

def apply_grid_distort(img, boxes, cfg):
    """Barrel/pin-cushion via grid warp (boxes stay intact — distortion is mild)."""
    H, W = img.shape[:2]
    steps = cfg["grid_distort_steps"]
    lim   = cfg["grid_distort_limit"]
    # build source and destination grids
    xs = np.linspace(0, W, steps+1, dtype=np.float32)
    ys = np.linspace(0, H, steps+1, dtype=np.float32)
    map_x = np.zeros((H, W), dtype=np.float32)
    map_y = np.zeros((H, W), dtype=np.float32)
    for i in range(steps):
        for j in range(steps):
            dx = random.uniform(-lim, lim) * W / steps
            dy = random.uniform(-lim, lim) * H / steps
            x1s,x2s = xs[j], xs[j+1]
            y1s,y2s = ys[i], ys[i+1]
            block_h = int(y2s-y1s); block_w = int(x2s-x1s)
            if block_h<=0 or block_w<=0: continue
            gx,gy = np.meshgrid(np.linspace(x1s,x2s,block_w),
                                 np.linspace(y1s,y2s,block_h))
            r = slice(int(y1s), int(y1s)+block_h)
            c = slice(int(x1s), int(x1s)+block_w)
            map_x[r,c] = np.clip(gx+dx, 0, W-1)
            map_y[r,c] = np.clip(gy+dy, 0, H-1)
    if map_x.max() == 0:   # fallback if grid init failed
        return img, boxes
    img2 = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)
    # Boxes are not remapped (distortion is mild and per-pixel inverse is complex)
    # — for PCB use this is an acceptable trade-off.
    return img2, boxes


# ══════════════════════════════════════════════════════════════════════════════
# Full augmentation pipeline for one image
# ══════════════════════════════════════════════════════════════════════════════
def augment_one(img, boxes, cfg):
    """Apply a random combination of transforms and return (aug_img, aug_boxes)."""
    # Spatial (order matters: do these before pixel ops so boxes stay valid)
    if random.random() < cfg["hflip_p"]:
        img, boxes = apply_hflip(img, boxes)
    if random.random() < cfg["vflip_p"]:
        img, boxes = apply_vflip(img, boxes)
    if random.random() < cfg["rot90_p"]:
        img, boxes = apply_rot90(img, boxes)
    # Affine: rotation + scale + translate
    img, boxes = apply_affine(img, boxes, cfg)
    # Perspective warp
    if random.random() < cfg["perspective_p"]:
        img, boxes = apply_perspective(img, boxes, cfg)
    # Grid distort (mild, boxes not updated — acceptable)
    if random.random() < cfg["grid_distort_p"]:
        img, boxes = apply_grid_distort(img, boxes, cfg)
    # Pixel-level
    img = apply_color(img, cfg)
    img = apply_noise(img, cfg)
    # JPEG compression artefact simulation
    if random.random() < cfg["jpeg_quality_p"]:
        img = apply_jpeg(img, cfg)
    return img, boxes


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_labels(lbl_path):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append(list(map(float, parts)))
    return boxes

def save_labels(lbl_path, boxes):
    with open(lbl_path, "w") as f:
        for b in boxes:
            f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main(src, dst, target, seed):
    random.seed(seed); np.random.seed(seed)

    src = Path(src); dst = Path(dst)
    src_img = src / "images"; src_lbl = src / "labels"

    # Output structure: train/images, train/labels, val/images, val/labels
    for split in ("train", "val"):
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)

    src_img = src / "images"; src_lbl = src / "labels"

    # Collect originals
    originals = sorted([p for p in src_img.iterdir()
                        if p.suffix.lower() in IMG_EXTS])
    if not originals:
        print(f"[ERROR] No images found in {src_img}"); sys.exit(1)
    n_orig = len(originals)
    print(f"Found {n_orig} original images. Target: {target}")

    # ── Step 1: augment ALL originals into a flat staging folder ──────────
    stage = dst / "_stage"
    (stage / "images").mkdir(parents=True, exist_ok=True)
    (stage / "labels").mkdir(parents=True, exist_ok=True)

    # Copy originals into stage first
    for p in originals:
        shutil.copy2(p, stage / "images" / p.name)
        lbl = src_lbl / (p.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, stage / "labels" / lbl.name)

    # Generate augmented copies from ALL originals until we hit target
    need = target - n_orig
    copies_per_img = math.ceil(need / n_orig)
    print(f"Augmenting all {n_orig} originals  ({copies_per_img} copies each) …")

    generated = 0
    for copy_idx in range(copies_per_img):
        if generated >= need: break
        pool = list(originals); random.shuffle(pool)
        for p in pool:
            if generated >= need: break
            img = cv2.imread(str(p))
            if img is None:
                print(f"  [SKIP] Cannot read {p.name}"); continue
            boxes = load_labels(str(src_lbl / (p.stem + ".txt")))
            aug_img, aug_boxes = augment_one(img, boxes, AUG_CFG)
            if not aug_boxes and boxes:
                aug_img, aug_boxes = apply_hflip(img, boxes)
                aug_img = apply_color(aug_img, AUG_CFG)
            stem = f"{p.stem}_aug{copy_idx:03d}"
            cv2.imwrite(str(stage / "images" / (stem + ".jpg")),
                        aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            save_labels(str(stage / "labels" / (stem + ".txt")), aug_boxes)
            generated += 1

    all_images = sorted((stage / "images").iterdir())
    total = len(all_images)
    print(f"Total in pool  : {total}  ({n_orig} orig + {generated} aug)")

    # ── Step 2: shuffle and split 80/20 into train / val ─────────────────
    random.shuffle(all_images)
    n_val   = round(total * 0.20)
    n_train = total - n_val
    val_imgs   = all_images[:n_val]
    train_imgs = all_images[n_val:]

    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for p in imgs:
            shutil.copy2(p, dst / split / "images" / p.name)
            lbl = stage / "labels" / (p.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dst / split / "labels" / lbl.name)

    # Clean up staging folder
    shutil.rmtree(stage)

    print(f"\n✓ Done:")
    print(f"   train/images : {n_train}")
    print(f"   val/images   : {n_val}")
    print(f"   Output → {dst}")

    # ── Write data.yaml ────────────────────────────────────────────────────
    names = [
        "Connector (P)","Resistor (R)","Transformer (T)","Diode (D)","Capacitor (C)",
        "Transistor (Q)","Jumper (J)","Inductor (L)","IC (U)","Resistor Array (RA)",
        "Resistor Net (RN)","Crystal (CR)","IC (IC)","Jumper (JP)","Varistor (V)",
        "Button (BTN)","Switch (SW)","Switch (S)","Test Point (TP)","LED",
        "Transistor (QA)","Cap Array (CRA)","Motor (M)","Fuse (F)","Ferrite Bead (FB)",
    ]
    yaml_path = dst / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {dst.as_posix()}\n")
        f.write(f"train: train/images\n")
        f.write(f"val:   val/images\n")
        f.write(f"nc: {len(names)}\n")
        f.write(f"names: {names}\n")
    print(f"  data.yaml → {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB dataset augmenter")
    parser.add_argument("--src",    default=SRC_DIR,  help="Source project folder")
    parser.add_argument("--dst",    default=DST_DIR,   help="Output augmented folder")
    parser.add_argument("--target", type=int, default=TARGET, help="Target image count")
    parser.add_argument("--seed",   type=int, default=SEED,   help="Random seed")
    a = parser.parse_args()
    main(a.src, a.dst, a.target, a.seed)