# -*- coding: utf-8 -*-
"""
pipeline.py  —  MELSS PCB Full Training Pipeline
═════════════════════════════════════════════════
Runs all 4 stages end-to-end:
  Stage 1 — Augment   : offline augmentation of the raw labelled dataset
  Stage 2 — Remap     : compact class IDs (remove unused classes)
  Stage 3 — Tile      : slice high-res images → small overlapping tiles
  Stage 4 — Train     : 2-phase frozen→full fine-tune with YOLO

Usage (full run):
    python pipeline.py

Skip stages you already completed:
    python pipeline.py --skip-augment --skip-remap

Start from a specific stage:
    python pipeline.py --from-stage tile
    python pipeline.py --from-stage train

Override key paths:
    python pipeline.py --src "D:/..." --dst-aug "D:/..." --dst-tile "D:/..."
                       --model "D:/..." --name "my_run"
"""

import os, sys, gc, math, random, shutil, argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  C O N F I G  — edit these defaults; all are also settable via CLI
# ══════════════════════════════════════════════════════════════════════════════

# Paths
SRC_DIR     = r"D:\MELSS\AOI\AOI_Projects\new\images\645"   # raw images + labels
DST_AUG     = r"D:\MELSS\AOI\AOI_Projects\new_augment"      # augmented dataset
DST_TILE    = r"D:\MELSS\AOI\AOI_Projects\new_tiled"        # tiled dataset
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "best.pt"
BASE_MODEL  = str(DEFAULT_MODEL)
PROJECT_DIR = r"D:\MELSS\AOI\runs"
RUN_NAME    = "gatekeeper_v4_tiled"

# Stage 1 — Augmentation
AUG_TARGET = 280
AUG_SEED   = 42

# Stage 3 — Tiling
TILE_SZ      = 960
TILE_OVERLAP = 0.25
BG_KEEP_RATE = 0.05

# Stage 4 — Training
EPOCHS        = 120
IMGSZ         = 960
BATCH         = 4
WORKERS       = 4
FREEZE_EPOCHS = 10
PATIENCE      = 20
DEVICE        = "0"
NBS           = BATCH

# Full class list (index = original class ID)
ORIGINAL_CLASSES = [
    "Connector (P)", "Resistor (R)", "Transformer (T)", "Diode (D)",
    "Capacitor (C)", "Transistor (Q)", "Jumper (J)", "Inductor (L)",
    "IC (U)", "Resistor Array (RA)", "Resistor Net (RN)", "Crystal (CR)",
    "IC (IC)", "Jumper (JP)", "Varistor (V)", "Button (BTN)",
    "Switch (SW)", "Switch (S)", "Test Point (TP)", "LED",
    "Transistor (QA)", "Cap Array (CRA)", "Motor (M)", "Fuse (F)",
    "Ferrite Bead (FB)",
]

# Hyperparameters (v4 — all training-relevant only; inference params removed)
HYPER = dict(
    optimizer        = "AdamW",
    lr0              = 0.0005,
    lrf              = 0.05,
    momentum         = 0.937,
    weight_decay     = 0.0005,
    warmup_epochs    = 3.0,
    warmup_momentum  = 0.8,
    warmup_bias_lr   = 0.05,
    box              = 7.5,
    cls              = 0.5,
    dfl              = 1.5,
    hsv_h            = 0.010,
    hsv_s            = 0.50,
    hsv_v            = 0.30,
    degrees          = 3.0,
    translate        = 0.05,
    scale            = 0.20,
    shear            = 0.0,
    perspective      = 0.0001,
    flipud           = 0.0,
    fliplr           = 0.5,
    mosaic           = 0.4,   # FIX #13: was hardcoded 0.0 in train calls
    mixup            = 0.0,
    copy_paste       = 0.0,
    erasing          = 0.20,
    iou              = 0.60,  # training IoU threshold (anchor assignment)
)

# Augmentation config
AUG_CFG = dict(
    hflip_p=0.5, vflip_p=0.0,
    rot_max_deg=5, rot90_p=0.4,
    scale_lo=0.85, scale_hi=1.15,
    translate_p=0.4, translate_frac=0.08,
    perspective_p=0.2, perspective_k=0.04,
    brightness_range=(0.70, 1.35),
    contrast_range=(0.75, 1.30),
    saturation_range=(0.70, 1.40),
    hue_shift_max=10, gamma_range=(0.75, 1.40),
    gaussian_noise_p=0.40, gaussian_noise_var=(5, 25),
    motion_blur_p=0.20, motion_blur_k=(3, 7),
    median_blur_p=0.15, median_blur_k=(3, 5),
    jpeg_quality_p=0.30, jpeg_quality_range=(60, 95),
    grid_distort_p=0.15, grid_distort_steps=5, grid_distort_limit=0.05,
    min_box_area=0.0002, min_visibility=0.30,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ══════════════════════════════════════════════════════════════════════════════
#  S T A G E  1  —  A U G M E N T A T I O N
# ══════════════════════════════════════════════════════════════════════════════

def _yolo_to_corners(boxes, W, H):
    out = []
    for b in boxes:
        cls, cx, cy, bw, bh = b
        out.append([cls, (cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H])
    return out

def _corners_to_yolo(boxes, W, H):
    out = []
    for b in boxes:
        cls, x1, y1, x2, y2 = b
        x1=max(0,min(W-1,x1)); x2=max(0,min(W-1,x2))
        y1=max(0,min(H-1,y1)); y2=max(0,min(H-1,y2))
        if x2<=x1 or y2<=y1: continue
        out.append([cls,(x1+x2)/2/W,(y1+y2)/2/H,(x2-x1)/W,(y2-y1)/H])
    return out

def _transform_corners(pts, M):
    n = pts.shape[0]
    h = np.hstack([pts, np.ones((n,1))])
    t = (M @ h.T).T
    t[:,0] /= t[:,2]; t[:,1] /= t[:,2]
    return t[:,:2]

def _box_after_M(b, M, W, H, orig_area, cfg):
    cls, x1, y1, x2, y2 = b
    corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
    tc = _transform_corners(corners, M)
    nx1,ny1 = tc.min(0); nx2,ny2 = tc.max(0)
    nx1=max(0,nx1); ny1=max(0,ny1); nx2=min(W,nx2); ny2=min(H,ny2)
    if nx2<=nx1 or ny2<=ny1: return None
    new_area = (nx2-nx1)*(ny2-ny1)
    if new_area/max(orig_area,1e-6) < cfg["min_visibility"]: return None
    if new_area/(W*H) < cfg["min_box_area"]: return None
    return [cls,(nx1+nx2)/2/W,(ny1+ny2)/2/H,(nx2-nx1)/W,(ny2-ny1)/H]

def _apply_hflip(img, boxes):
    return cv2.flip(img,1), [[b[0],1.0-b[1],b[2],b[3],b[4]] for b in boxes]

def _apply_vflip(img, boxes):
    return cv2.flip(img,0), [[b[0],b[1],1.0-b[2],b[3],b[4]] for b in boxes]

def _apply_rot90(img, boxes):
    """FIX #2: correct cumulative H/W swap for k rotations."""
    k = random.choice([1,2,3])
    img2 = np.rot90(img, k)
    H, W = img.shape[:2]
    corners_b = _yolo_to_corners(boxes, W, H)
    b2 = []
    for b in corners_b:
        cls, x1, y1, x2, y2 = b
        pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
        cH, cW = H, W
        for _ in range(k):
            pts = np.column_stack([cH - pts[:,1], pts[:,0]])
            cH, cW = cW, cH
        nx1,ny1 = pts.min(0); nx2,ny2 = pts.max(0)
        b2.append([cls, nx1, ny1, nx2, ny2])
    nH, nW = img2.shape[:2]
    return img2, _corners_to_yolo(b2, nW, nH)

def _apply_affine(img, boxes, cfg):
    H, W = img.shape[:2]
    angle = random.uniform(-cfg["rot_max_deg"], cfg["rot_max_deg"])
    scale = random.uniform(cfg["scale_lo"], cfg["scale_hi"])
    M2 = cv2.getRotationMatrix2D((W/2,H/2), angle, scale)
    if random.random() < cfg["translate_p"]:
        M2[0,2] += random.uniform(-cfg["translate_frac"],cfg["translate_frac"])*W
        M2[1,2] += random.uniform(-cfg["translate_frac"],cfg["translate_frac"])*H
    M3 = np.vstack([M2,[0,0,1]])
    img2 = cv2.warpAffine(img, M2, (W,H), borderMode=cv2.BORDER_REFLECT_101)
    b2 = []
    for b in _yolo_to_corners(boxes,W,H):
        r = _box_after_M(b, M3, W, H, (b[3]-b[1])*(b[4]-b[2]), cfg)
        if r: b2.append(r)
    return img2, b2

def _apply_perspective(img, boxes, cfg):
    H, W = img.shape[:2]
    k = cfg["perspective_k"]
    rp = lambda v: v*random.uniform(-k,k)
    src = np.float32([[0,0],[W,0],[W,H],[0,H]])
    dst = np.float32([[rp(W),rp(H)],[W+rp(W),rp(H)],[W+rp(W),H+rp(H)],[rp(W),H+rp(H)]])
    M = cv2.getPerspectiveTransform(src, dst)
    img2 = cv2.warpPerspective(img, M, (W,H), borderMode=cv2.BORDER_REFLECT_101)
    b2 = []
    for b in _yolo_to_corners(boxes,W,H):
        r = _box_after_M(b, M, W, H, (b[3]-b[1])*(b[4]-b[2]), cfg)
        if r: b2.append(r)
    return img2, b2

def _apply_color(img, cfg):
    """FIX #4: separate brightness and contrast operations."""
    img = img.astype(np.float32)
    alpha = random.uniform(*cfg["contrast_range"])    # contrast
    beta  = random.uniform(*cfg["brightness_range"])  # brightness scale
    img = np.clip(img * alpha * beta, 0, 255)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*random.uniform(*cfg["saturation_range"]),0,255)
    hsv[:,:,0] = (hsv[:,:,0]+random.uniform(-cfg["hue_shift_max"],cfg["hue_shift_max"]))%180
    img = cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    gamma = random.uniform(*cfg["gamma_range"])
    lut = np.array([min(255,((i/255.0)**(1.0/gamma))*255) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img.astype(np.uint8), lut)

def _apply_noise(img, cfg):
    if random.random() < cfg["gaussian_noise_p"]:
        var = random.uniform(*cfg["gaussian_noise_var"])
        noise = np.random.normal(0, math.sqrt(var), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32)+noise, 0, 255).astype(np.uint8)
    if random.random() < cfg["motion_blur_p"]:
        k = random.randrange(cfg["motion_blur_k"][0], cfg["motion_blur_k"][1]+1, 2)
        angle = random.uniform(0, 180)
        Mk = cv2.getRotationMatrix2D((k//2,k//2), angle, 1)
        kernel = np.zeros((k,k)); kernel[k//2,:] = 1.0
        kernel = cv2.warpAffine(kernel, Mk, (k,k))
        s = kernel.sum()
        if s > 1e-6: kernel /= s          # FIX #3: safe normalise
        img = cv2.filter2D(img, -1, kernel)
    if random.random() < cfg["median_blur_p"]:
        k = random.choice([x for x in range(cfg["median_blur_k"][0],
                                             cfg["median_blur_k"][1]+1) if x%2==1])
        img = cv2.medianBlur(img, k)
    return img

def _apply_jpeg(img, cfg):
    q = random.randint(*cfg["jpeg_quality_range"])
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img

def _apply_grid_distort(img, boxes, cfg):
    H, W = img.shape[:2]
    steps = cfg["grid_distort_steps"]; lim = cfg["grid_distort_limit"]
    xs = np.linspace(0, W, steps+1, dtype=np.float32)
    ys = np.linspace(0, H, steps+1, dtype=np.float32)
    map_x = np.zeros((H,W), dtype=np.float32)
    map_y = np.zeros((H,W), dtype=np.float32)
    for i in range(steps):
        for j in range(steps):
            dx = random.uniform(-lim,lim)*W/steps
            dy = random.uniform(-lim,lim)*H/steps
            bh = int(ys[i+1]-ys[i]); bw = int(xs[j+1]-xs[j])
            if bh<=0 or bw<=0: continue
            gx, gy = np.meshgrid(np.linspace(xs[j],xs[j+1],bw),
                                  np.linspace(ys[i],ys[i+1],bh))
            r = slice(int(ys[i]),int(ys[i])+bh)
            c = slice(int(xs[j]),int(xs[j])+bw)
            map_x[r,c] = np.clip(gx+dx,0,W-1)
            map_y[r,c] = np.clip(gy+dy,0,H-1)
    # FIX #5: check all zeros rather than just max==0
    if (map_x == 0).all(): return img, boxes
    img2 = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img2, boxes

def _augment_one(img, boxes, cfg):
    if random.random() < cfg["hflip_p"]:       img, boxes = _apply_hflip(img, boxes)
    if random.random() < cfg["vflip_p"]:       img, boxes = _apply_vflip(img, boxes)
    if random.random() < cfg["rot90_p"]:       img, boxes = _apply_rot90(img, boxes)
    img, boxes = _apply_affine(img, boxes, cfg)
    if random.random() < cfg["perspective_p"]: img, boxes = _apply_perspective(img, boxes, cfg)
    if random.random() < cfg["grid_distort_p"]:img, boxes = _apply_grid_distort(img, boxes, cfg)
    img = _apply_color(img, cfg)
    img = _apply_noise(img, cfg)
    if random.random() < cfg["jpeg_quality_p"]: img = _apply_jpeg(img, cfg)
    return img, boxes

def _load_labels(path):
    boxes = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append(list(map(float, parts)))
    return boxes

def _save_labels(path, boxes):
    with open(path, "w") as f:
        for b in boxes:
            f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

def stage_augment(src, dst, target, seed):
    """Stage 1: Offline augmentation."""
    print("\n" + "="*60)
    print("  STAGE 1 — Augmentation")
    print("="*60)
    # FIX #1: seed only here, not at module level
    random.seed(seed); np.random.seed(seed)

    src = Path(src); dst = Path(dst)
    src_img = src / "images"; src_lbl = src / "labels"

    for split in ("train", "val"):
        (dst/split/"images").mkdir(parents=True, exist_ok=True)
        (dst/split/"labels").mkdir(parents=True, exist_ok=True)

    originals = sorted(p for p in src_img.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not originals:
        print(f"[ERROR] No images in {src_img}"); sys.exit(1)
    n_orig = len(originals)
    print(f"  Found {n_orig} originals → target {target}")

    stage = dst / "_stage"
    (stage/"images").mkdir(parents=True, exist_ok=True)
    (stage/"labels").mkdir(parents=True, exist_ok=True)

    for p in originals:
        shutil.copy2(p, stage/"images"/p.name)
        lbl = src_lbl/(p.stem+".txt")
        if lbl.exists(): shutil.copy2(lbl, stage/"labels"/lbl.name)

    need = max(0, target - n_orig)
    copies_per = math.ceil(need / max(n_orig,1))
    generated  = 0
    for ci in range(copies_per):
        if generated >= need: break
        pool = list(originals); random.shuffle(pool)
        for p in pool:
            if generated >= need: break
            img = cv2.imread(str(p))
            if img is None: continue
            boxes = _load_labels(str(src_lbl/(p.stem+".txt")))
            aug_img, aug_boxes = _augment_one(img, boxes, AUG_CFG)
            if not aug_boxes and boxes:
                aug_img, aug_boxes = _apply_hflip(img, boxes)
                aug_img = _apply_color(aug_img, AUG_CFG)
            stem = f"{p.stem}_aug{ci:03d}"
            cv2.imwrite(str(stage/"images"/(stem+".jpg")), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            _save_labels(str(stage/"labels"/(stem+".txt")), aug_boxes)
            generated += 1

    all_images = sorted((stage/"images").iterdir())
    total = len(all_images)
    random.shuffle(all_images)
    n_val   = round(total * 0.20)
    n_train = total - n_val
    val_imgs   = all_images[:n_val]
    train_imgs = all_images[n_val:]

    for split, imgs in [("train",train_imgs),("val",val_imgs)]:
        (dst/split/"images").mkdir(parents=True, exist_ok=True)
        (dst/split/"labels").mkdir(parents=True, exist_ok=True)
        for p in imgs:
            shutil.copy2(p, dst/split/"images"/p.name)
            lbl = stage/"labels"/(p.stem+".txt")
            if lbl.exists(): shutil.copy2(lbl, dst/split/"labels"/lbl.name)

    shutil.rmtree(stage)

    names = ORIGINAL_CLASSES
    yaml_path = dst/"data.yaml"
    with open(yaml_path,"w") as f:
        f.write(f"path: {dst.as_posix()}\n")
        f.write(f"train: train/images\nval:   val/images\n")
        f.write(f"nc: {len(names)}\nnames: {names}\n")

    print(f"  ✓ train={n_train}  val={n_val}  → {dst}")
    return str(yaml_path)


# ══════════════════════════════════════════════════════════════════════════════
#  S T A G E  2  —  R E M A P  C L A S S E S
# ══════════════════════════════════════════════════════════════════════════════

def _find_label_dirs(data_yaml):
    import yaml
    with open(data_yaml) as f: cfg = yaml.safe_load(f)
    base = Path(cfg.get("path", Path(data_yaml).parent))
    dirs = []
    for key in ("train","val","test"):
        rel = cfg.get(key)
        if not rel: continue
        # FIX #11: single clean derivation
        lbl_dir = Path(str((base/rel)).replace("images","labels"))
        if lbl_dir.is_dir(): dirs.append(lbl_dir)
    return dirs

def stage_remap(data_yaml, dry_run=False):
    """Stage 2: Compact class IDs."""
    import yaml
    print("\n" + "="*60)
    print("  STAGE 2 — Remap Classes")
    print("="*60)
    data_yaml = Path(data_yaml)
    label_dirs = _find_label_dirs(data_yaml)
    if not label_dirs:
        print("[WARN] No label dirs found — skipping remap"); return

    counts = defaultdict(int)
    for d in label_dirs:
        for f in d.glob("*.txt"):
            for line in f.read_text().splitlines():
                parts = line.strip().split()
                if parts: counts[int(parts[0])] += 1

    if not counts:
        print("[WARN] No annotations found — skipping remap"); return

    active = sorted(counts.keys())
    old_to_new = {old:new for new,old in enumerate(active)}
    new_names  = [ORIGINAL_CLASSES[i] if i<len(ORIGINAL_CLASSES) else str(i) for i in active]

    print(f"  {'Old':>6}  {'New':>6}  {'Count':>8}  Name")
    for old,new in old_to_new.items():
        name = ORIGINAL_CLASSES[old] if old<len(ORIGINAL_CLASSES) else str(old)
        print(f"  {old:>6}  {new:>6}  {counts[old]:>8}  {name}")

    if dry_run:
        print("  [DRY RUN] No files written."); return

    bak = data_yaml.with_suffix(".yaml.bak")
    shutil.copy2(data_yaml, bak)

    total_files=0; total_boxes=0; skipped=0
    for d in label_dirs:
        for f in sorted(d.glob("*.txt")):
            lines = f.read_text().splitlines()
            new_lines=[]; changed=False
            for line in lines:
                parts=line.strip().split()
                if not parts: continue
                old_cls=int(parts[0])
                if old_cls not in old_to_new: skipped+=1; changed=True; continue
                nc = old_to_new[old_cls]
                if nc!=old_cls: changed=True
                new_lines.append(f"{nc} "+" ".join(parts[1:]))
                total_boxes+=1
            if changed:
                f.write_text("\n".join(new_lines)+("\n" if new_lines else ""))
            total_files+=1

    with open(data_yaml) as f: cfg=yaml.safe_load(f)
    cfg["nc"]=len(new_names); cfg["names"]=new_names
    with open(data_yaml,"w") as f:
        yaml.dump(cfg,f,default_flow_style=False,allow_unicode=True,sort_keys=False)

    print(f"  ✓ {total_files} files rewritten  ({total_boxes} boxes, {skipped} dropped)")
    print(f"  Updated data.yaml → nc={len(new_names)}")


# ══════════════════════════════════════════════════════════════════════════════
#  S T A G E  3  —  T I L I N G
# ══════════════════════════════════════════════════════════════════════════════

def _remap_to_tile(boxes, img_w, img_h, tx1, ty1, tx2, ty2,
                   min_vis=0.25, min_area=16):
    tw, th = tx2-tx1, ty2-ty1
    result = []
    for cls,cx,cy,bw,bh in boxes:
        ax=cx*img_w; ay=cy*img_h; aw=bw*img_w; ah=bh*img_h
        bx1=ax-aw/2; by1=ay-ah/2; bx2=ax+aw/2; by2=ay+ah/2
        orig_a=aw*ah
        cx1=max(bx1,tx1); cy1=max(by1,ty1); cx2=min(bx2,tx2); cy2=min(by2,ty2)
        if cx2<=cx1 or cy2<=cy1: continue
        ca=(cx2-cx1)*(cy2-cy1)
        if orig_a>0 and ca/orig_a<min_vis: continue
        if ca<min_area: continue
        ncx=max(0.0,min(1.0,((cx1+cx2)/2-tx1)/tw))
        ncy=max(0.0,min(1.0,((cy1+cy2)/2-ty1)/th))
        nbw=max(0.0,min(1.0,(cx2-cx1)/tw))
        nbh=max(0.0,min(1.0,(cy2-cy1)/th))
        if nbw>0 and nbh>0: result.append([cls,ncx,ncy,nbw,nbh])
    return result

def _tile_coords(W, H, sz, overlap):
    stride=max(int(sz*(1-overlap)),1)
    xs=list(range(0,W-sz+1,stride))
    if not xs or xs[-1]+sz<W: xs.append(max(0,W-sz))
    ys=list(range(0,H-sz+1,stride))
    if not ys or ys[-1]+sz<H: ys.append(max(0,H-sz))
    seen=set()
    for x in xs:
        for y in ys:
            x2=min(x+sz,W); y2=min(y+sz,H)
            x1=max(0,x2-sz); y1=max(0,y2-sz)
            if (x1,y1,x2,y2) not in seen:
                seen.add((x1,y1,x2,y2)); yield x1,y1,x2,y2

def _process_tile_split(img_paths, lbl_dir, out_img, out_lbl,
                         tile_sz, overlap, bg_keep, min_vis, min_area):
    out_img.mkdir(parents=True,exist_ok=True)
    out_lbl.mkdir(parents=True,exist_ok=True)
    total_tiles=0; total_boxes=0; skipped=0
    cls_counts=defaultdict(int)
    for img_path in img_paths:
        img=cv2.imread(str(img_path))
        if img is None: print(f"  [WARN] Cannot read {img_path.name}"); continue
        H,W=img.shape[:2]
        lp=lbl_dir/(img_path.stem+".txt")
        boxes=[]
        if lp.exists():
            for line in lp.read_text().splitlines():
                p=line.strip().split()
                if len(p)==5: boxes.append([int(p[0])]+[float(x) for x in p[1:]])
        if W<tile_sz or H<tile_sz:
            img=cv2.copyMakeBorder(img,0,max(0,tile_sz-H),0,max(0,tile_sz-W),
                                   cv2.BORDER_REFLECT_101)
            H,W=img.shape[:2]
        for ti,(x1,y1,x2,y2) in enumerate(_tile_coords(W,H,tile_sz,overlap)):
            tb=_remap_to_tile(boxes,W,H,x1,y1,x2,y2,min_vis,min_area)
            if not tb:
                if random.random()>bg_keep: skipped+=1; continue
            tile=img[y1:y2,x1:x2]
            if tile.shape[0]!=tile_sz or tile.shape[1]!=tile_sz:
                tile=cv2.resize(tile,(tile_sz,tile_sz),interpolation=cv2.INTER_LINEAR)
            stem=f"{img_path.stem}_t{ti:04d}"
            cv2.imwrite(str(out_img/(stem+".jpg")),tile,[cv2.IMWRITE_JPEG_QUALITY,95])
            (out_lbl/(stem+".txt")).write_text(
                "\n".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in tb))
            total_tiles+=1; total_boxes+=len(tb)
            for b in tb: cls_counts[b[0]]+=1
    return total_tiles, total_boxes, skipped, cls_counts

def stage_tile(src, aug, dst, tile_sz, overlap, bg_keep, min_vis, min_area, seed):
    """Stage 3: Tile high-res images."""
    import yaml
    print("\n" + "="*60)
    print("  STAGE 3 — Tiling")
    print("="*60)
    random.seed(seed)
    src=Path(src); aug=Path(aug) if aug else None; dst=Path(dst)
    dst.mkdir(parents=True,exist_ok=True)

    nc,names=None,None
    for yp in ([aug/"data.yaml"] if aug else []) + [src/"data.yaml"]:
        if yp and yp.exists():
            d=yaml.safe_load(open(yp))
            nc=d.get("nc",len(ORIGINAL_CLASSES)); names=d.get("names",[]); break

    # Find splits
    splits_found={}
    for split in ("train","val"):
        idir=src/split/"images"; ldir=src/split/"labels"
        if idir.is_dir() and ldir.is_dir():
            imgs=sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXTS)
            if imgs: splits_found[split]=(imgs,ldir)

    if not splits_found:
        candidates=[src/"images",src]
        idir=next((p for p in candidates if p.is_dir()),src)
        ldir=src/"labels" if (src/"labels").is_dir() else src
        all_imgs=sorted(p for p in idir.rglob("*")
                        if p.suffix.lower() in IMG_EXTS and "labels" not in str(p))
        if not all_imgs: print(f"[ERROR] No images in {src}"); sys.exit(1)
        random.shuffle(all_imgs); sp=max(1,int(len(all_imgs)*0.8))
        splits_found["train"]=(all_imgs[:sp],ldir)
        splits_found["val"]=(all_imgs[sp:],ldir)

    grand=defaultdict(int)
    # FIX #10: collect class counts from BOTH splits for nc computation
    for split,(img_paths,lbl_dir) in splits_found.items():
        out_img=dst/split/"images"; out_lbl=dst/split/"labels"
        # For val: copy from aug set untiled if available
        if split=="val" and aug and (aug/"val"/"images").is_dir():
            print(f"  [val] Copying from augmented val (no tiling) …")
            out_img.mkdir(parents=True,exist_ok=True)
            out_lbl.mkdir(parents=True,exist_ok=True)
            n=0
            for p in (aug/"val"/"images").iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    shutil.copy2(p, out_img/p.name)
                    lp=aug/"val"/"labels"/(p.stem+".txt")
                    if lp.exists(): shutil.copy2(lp, out_lbl/lp.name)
                    n+=1
            print(f"       → {n} val images")
            continue
        print(f"  [{split}] Tiling {len(img_paths)} images …")
        nt,nb,ns,cc=_process_tile_split(img_paths,lbl_dir,out_img,out_lbl,
                                         tile_sz,overlap,bg_keep,min_vis,min_area)
        for k,v in cc.items(): grand[k]+=v
        print(f"       → {nt} tiles  {nb} boxes  {ns} bg skipped")

    # Write data.yaml
    has_val=(dst/"val"/"images").is_dir() and any((dst/"val"/"images").iterdir())
    # FIX #9: default nc to full class count, not 7
    cfg_out={
        "path": str(dst).replace("\\","/"),
        "train": "train/images",
        "val": "val/images" if has_val else "train/images",
        "nc": nc or (max(grand.keys(),default=0)+1),
        "names": names or ORIGINAL_CLASSES,
    }
    with open(dst/"data.yaml","w") as f:
        yaml.dump(cfg_out,f,default_flow_style=False,allow_unicode=True,sort_keys=False)

    print(f"\n  Class distribution (train tiles):")
    for idx,name in enumerate(cfg_out["names"]):
        cnt=grand.get(idx,0)
        if cnt>0: print(f"    {idx}: {name:28s}  {cnt:6d}")
    print(f"  ✓ Tiled dataset → {dst}")
    return str(dst/"data.yaml")


# ══════════════════════════════════════════════════════════════════════════════
#  S T A G E  4  —  T R A I N I N G
# ══════════════════════════════════════════════════════════════════════════════

def _check_env():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. pip install ultralytics"); sys.exit(1)
    try:
        import torch
        if torch.cuda.is_available():
            name=torch.cuda.get_device_name(0)
            vram=torch.cuda.get_device_properties(0).total_memory/1e9
            print(f"  [GPU] {name}  ({vram:.1f} GB VRAM)")
        else:
            print("  [WARN] No CUDA GPU — training on CPU will be very slow.")
    except Exception as e:
        print(f"  [WARN] GPU query failed: {e}")

def _count_images(data_yaml):
    import yaml
    try:
        with open(data_yaml) as f: d=yaml.safe_load(f)
        base=Path(d.get("path",Path(data_yaml).parent))
        train_val=d.get("train","images")
        if isinstance(train_val,str): train_val=[train_val]
        n=0
        for p in train_val:
            td=base/p
            if td.is_dir():
                n+=sum(1 for f in td.iterdir() if f.suffix.lower() in IMG_EXTS)
        return n
    except Exception as e:
        print(f"  [WARN] Could not count images: {e}"); return 0

def stage_train(data, model_path, name, epochs, imgsz, batch, device):
    """Stage 4: 2-phase YOLO fine-tune."""
    import torch
    from ultralytics import YOLO

    print("\n" + "="*60)
    print("  STAGE 4 — Training")
    print("="*60)
    _check_env()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")

    if not os.path.exists(data):
        print(f"[ERROR] data.yaml not found: {data}"); sys.exit(1)
    if not os.path.exists(model_path):
        print(f"[ERROR] Base model not found: {model_path}"); sys.exit(1)

    n_imgs=_count_images(data)
    print(f"  [DATA ] {n_imgs} training images  → {data}")
    print(f"  [MODEL] {model_path}")
    print(f"  [RUN  ] {name}  epochs={epochs}  imgsz={imgsz}  batch={batch}")

    # ── Phase 1: Frozen backbone ──────────────────────────────────────────────
    print(f"\n  Phase 1 — Frozen backbone ({FREEZE_EPOCHS} epochs) …")
    model = YOLO(model_path)
    model.train(
        data=data, epochs=FREEZE_EPOCHS, imgsz=imgsz, batch=batch,
        device=device, workers=WORKERS,
        project=PROJECT_DIR, name=name+"_phase1", exist_ok=True,
        patience=0, save=True, save_period=5, plots=True,
        amp=True, cos_lr=True, freeze=10, close_mosaic=0, verbose=True,
        nbs=NBS,
        lr0        = HYPER["lr0"]*2,
        lrf        = HYPER["lrf"],
        optimizer  = HYPER["optimizer"],
        weight_decay=HYPER["weight_decay"],
        warmup_epochs   = HYPER["warmup_epochs"],
        warmup_momentum = HYPER["warmup_momentum"],
        warmup_bias_lr  = HYPER["warmup_bias_lr"],
        momentum   = HYPER["momentum"],
        box=HYPER["box"], cls=HYPER["cls"], dfl=HYPER["dfl"],
        hsv_h=HYPER["hsv_h"], hsv_s=HYPER["hsv_s"], hsv_v=HYPER["hsv_v"],
        degrees=HYPER["degrees"], translate=HYPER["translate"],
        scale=HYPER["scale"], fliplr=HYPER["fliplr"],
        mosaic=0.0, mixup=0.0, copy_paste=0.0,   # minimal aug in warm-up
        erasing=HYPER["erasing"],
    )

    # Free GPU memory between phases
    del model; gc.collect()
    # FIX #14: guard cuda calls
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        free=torch.cuda.mem_get_info()[0]/1e9
        print(f"  GPU memory freed.  Free: {free:.2f} GB")

    p1_best = Path(PROJECT_DIR)/(name+"_phase1")/"weights"/"best.pt"
    if not p1_best.exists():
        p1_best = Path(PROJECT_DIR)/(name+"_phase1")/"weights"/"last.pt"
    remaining = epochs - FREEZE_EPOCHS
    print(f"\n  Phase 2 — Full fine-tune from {p1_best}  ({remaining} epochs) …")

    # ── Phase 2: Full fine-tune ───────────────────────────────────────────────
    model2 = YOLO(str(p1_best))
    model2.train(
        data=data, epochs=remaining, imgsz=imgsz, batch=batch,
        device=device, workers=WORKERS,
        project=PROJECT_DIR, name=name, exist_ok=True,
        patience=PATIENCE, save=True, save_period=10, plots=True,
        amp=True, cos_lr=True, freeze=0,
        # FIX #15: close_mosaic=10 (disable mosaic for final 10 epochs — standard practice)
        close_mosaic=10, verbose=True,
        nbs=NBS,
        lr0        = HYPER["lr0"],
        lrf        = HYPER["lrf"],
        optimizer  = HYPER["optimizer"],
        weight_decay=HYPER["weight_decay"],
        warmup_epochs   = 1.0,
        warmup_momentum = HYPER["warmup_momentum"],
        warmup_bias_lr  = HYPER["warmup_bias_lr"],
        momentum   = HYPER["momentum"],
        box=HYPER["box"], cls=HYPER["cls"], dfl=HYPER["dfl"],
        hsv_h=HYPER["hsv_h"], hsv_s=HYPER["hsv_s"], hsv_v=HYPER["hsv_v"],
        degrees=HYPER["degrees"], translate=HYPER["translate"],
        scale=HYPER["scale"], shear=HYPER["shear"],
        perspective=HYPER["perspective"],
        flipud=HYPER["flipud"], fliplr=HYPER["fliplr"],
        # FIX #13: use HYPER["mosaic"] instead of hardcoded 0.0
        mosaic=HYPER["mosaic"],
        mixup=HYPER["mixup"], copy_paste=HYPER["copy_paste"],
        erasing=HYPER["erasing"],
        iou=HYPER["iou"],
        # FIX #16/#17: removed nms/conf — they are inference params, not training params
    )

    # ── Export ────────────────────────────────────────────────────────────────
    best = Path(PROJECT_DIR)/name/"weights"/"best.pt"
    print(f"\n  ✓ Training complete.  Best weights → {best}")

    if best.exists():
        print("\n  Exporting to ONNX …")
        try:
            YOLO(str(best)).export(format="onnx",imgsz=imgsz,simplify=True,opset=17,dynamic=False)
            print("    → ONNX export done")
        except Exception as e:
            print(f"    [WARN] ONNX: {e}")
        print("  Exporting to TorchScript …")
        try:
            YOLO(str(best)).export(format="torchscript",imgsz=imgsz,optimize=True)
            print("    → TorchScript export done")
        except Exception as e:
            print(f"    [WARN] TorchScript: {e}")

    print("\n" + "="*60)
    print(f"  Run dir  :  {Path(PROJECT_DIR)/name}")
    print(f"  Best .pt :  {best}")
    onnx=best.with_suffix(".onnx")
    if onnx.exists(): print(f"  ONNX     :  {onnx}")
    print("="*60)


# ══════════════════════════════════════════════════════════════════════════════
#  E N T R Y   P O I N T
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="MELSS PCB Full Training Pipeline  (augment → remap → tile → train)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Stage control
    p.add_argument("--from-stage", choices=["augment","remap","tile","train"],
                   help="Start from this stage (skips all previous stages)")
    p.add_argument("--skip-augment", action="store_true")
    p.add_argument("--skip-remap",   action="store_true")
    p.add_argument("--skip-tile",    action="store_true")
    p.add_argument("--skip-train",   action="store_true")
    p.add_argument("--dry-run-remap",action="store_true",
                   help="Print remap plan without writing files")

    # Paths
    p.add_argument("--src",      default=SRC_DIR,    help="Raw source images/labels dir")
    p.add_argument("--dst-aug",  default=DST_AUG,    help="Augmentation output dir")
    p.add_argument("--dst-tile", default=DST_TILE,   help="Tiling output dir")
    p.add_argument("--model",    default=BASE_MODEL,  help="Base .pt to warm-start from")
    p.add_argument("--name",     default=RUN_NAME,    help="YOLO run name")

    # Stage 1
    p.add_argument("--target", type=int, default=AUG_TARGET, help="Augmentation target count")
    p.add_argument("--seed",   type=int, default=AUG_SEED)

    # Stage 3
    p.add_argument("--tile",    type=int,   default=TILE_SZ)
    p.add_argument("--overlap", type=float, default=TILE_OVERLAP)
    p.add_argument("--bg-keep", type=float, default=BG_KEEP_RATE)
    p.add_argument("--min-vis", type=float, default=0.25)
    p.add_argument("--min-area",type=int,   default=16)

    # Stage 4
    p.add_argument("--epochs",  type=int,   default=EPOCHS)
    p.add_argument("--imgsz",   type=int,   default=IMGSZ)
    p.add_argument("--batch",   type=int,   default=BATCH)
    p.add_argument("--device",  default=DEVICE)

    return p.parse_args()


def main():
    a = parse_args()

    # --from-stage sets skip flags for everything before it
    order = ["augment","remap","tile","train"]
    if a.from_stage:
        idx = order.index(a.from_stage)
        if idx > 0: a.skip_augment = True
        if idx > 1: a.skip_remap   = True
        if idx > 2: a.skip_tile    = True

    print("\n" + "█"*60)
    print("  MELSS PCB  Full Training Pipeline")
    print("█"*60)
    print(f"  src      : {a.src}")
    print(f"  dst_aug  : {a.dst_aug}")
    print(f"  dst_tile : {a.dst_tile}")
    print(f"  model    : {a.model}")
    print(f"  run name : {a.name}")
    status = lambda s, skip: f"  [{'SKIP' if skip else 'RUN '}]  Stage: {s}"
    print(status("1 — Augment", a.skip_augment))
    print(status("2 — Remap",   a.skip_remap))
    print(status("3 — Tile",    a.skip_tile))
    print(status("4 — Train",   a.skip_train))

    # Stage 1
    aug_yaml = str(Path(a.dst_aug) / "data.yaml")
    if not a.skip_augment:
        stage_augment(a.src, a.dst_aug, a.target, a.seed)
    else:
        print(f"\n[SKIP] Augmentation  (using {aug_yaml})")

    # Stage 2
    if not a.skip_remap:
        stage_remap(aug_yaml, dry_run=a.dry_run_remap)
    else:
        print("\n[SKIP] Class remap")

    # Stage 3
    tile_yaml = str(Path(a.dst_tile) / "data.yaml")
    if not a.skip_tile:
        stage_tile(a.dst_aug, a.dst_aug, a.dst_tile,
                   a.tile, a.overlap, a.bg_keep, a.min_vis, a.min_area, a.seed)
    else:
        print(f"\n[SKIP] Tiling  (using {tile_yaml})")

    # Stage 4
    if not a.skip_train:
        stage_train(tile_yaml, a.model, a.name, a.epochs, a.imgsz, a.batch, a.device)
    else:
        print("\n[SKIP] Training")

    print("\n" + "█"*60)
    print("  Pipeline complete!")
    print("█"*60)


if __name__ == "__main__":
    main()
