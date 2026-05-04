"""
tile_dataset.py
───────────────
Slices a YOLO dataset into overlapping tiles so small PCB components
(resistors, capacitors, etc.) are trained at their ACTUAL pixel size
instead of being shrunk down to near-invisible blobs.

Why this works:
  A 4000×3000px PCB photo shrunk to 960px makes a 2mm resistor ~6px wide.
  Tiling at 960px extracts the same resistor at ~40px wide — the model
  can actually learn what it looks like.

Strategy:
  • Tiles from the ORIGINAL full-resolution source images (not the 
    already-augmented ones which are already smaller).
  • If source images aren't available, falls back to tiling the augmented set.
  • Tile size = 960px (matches training imgsz exactly → no double resize)
  • Overlap = 25% so components on tile borders still get trained
  • Tiles with no boxes after clipping are kept at a lower rate (background tiles)
  • Result goes into a new dataset folder ready to train on

Usage:
  # Tile from original source images (best quality):
  python tile_dataset.py --src  "D:/MELSS/AOI/AOI_Projects/new"
                         --aug  "D:/MELSS/AOI/AOI_Projects/new_augment"
                         --dst  "D:/MELSS/AOI/AOI_Projects/new_tiled"
                         --tile 960 --overlap 0.25

  # Or tile from the already-augmented set:
  python tile_dataset.py --src  "D:/MELSS/AOI/AOI_Projects/new_augment"
                         --dst  "D:/MELSS/AOI/AOI_Projects/new_tiled"
                         --tile 960 --overlap 0.25

  Then train:
  python train_finetune.py --data "D:/MELSS/AOI/AOI_Projects/new_tiled/data.yaml"
                           --model "D:/MELSS/AOI/runs/gatekeeper_v2/weights/best.pt"
                           --name  "gatekeeper_v3"
"""

import argparse, shutil, random, math
from pathlib import Path
from collections import defaultdict


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_labels(label_path):
    """Return list of [cls, cx, cy, w, h] normalised YOLO boxes."""
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return boxes


def remap_boxes_to_tile(boxes, img_w, img_h, tx1, ty1, tx2, ty2,
                        min_visibility=0.25, min_area_px=16):
    """
    Clip YOLO boxes to a tile region and remap to tile-local normalised coords.
    Drops boxes with < min_visibility fraction remaining or < min_area_px area.

    boxes : list of [cls, cx, cy, w, h]  (normalised to full image)
    tx1,ty1,tx2,ty2 : tile corners in PIXEL coords
    """
    tile_w = tx2 - tx1
    tile_h = ty2 - ty1
    result = []

    for cls, cx, cy, bw, bh in boxes:
        # Convert to absolute pixel coords in full image
        abs_cx = cx * img_w
        abs_cy = cy * img_h
        abs_bw = bw * img_w
        abs_bh = bh * img_h

        # Box corners (absolute)
        bx1 = abs_cx - abs_bw / 2
        by1 = abs_cy - abs_bh / 2
        bx2 = abs_cx + abs_bw / 2
        by2 = abs_cy + abs_bh / 2

        orig_area = abs_bw * abs_bh

        # Clip to tile
        cx1 = max(bx1, tx1)
        cy1 = max(by1, ty1)
        cx2 = min(bx2, tx2)
        cy2 = min(by2, ty2)

        if cx2 <= cx1 or cy2 <= cy1:
            continue  # fully outside tile

        clipped_area = (cx2 - cx1) * (cy2 - cy1)
        if orig_area > 0 and clipped_area / orig_area < min_visibility:
            continue  # too little of the box survives

        if clipped_area < min_area_px:
            continue  # too small to be useful

        # Remap to tile-local normalised coords
        new_cx = ((cx1 + cx2) / 2 - tx1) / tile_w
        new_cy = ((cy1 + cy2) / 2 - ty1) / tile_h
        new_bw = (cx2 - cx1) / tile_w
        new_bh = (cy2 - cy1) / tile_h

        # Clamp
        new_cx = max(0.0, min(1.0, new_cx))
        new_cy = max(0.0, min(1.0, new_cy))
        new_bw = max(0.0, min(1.0, new_bw))
        new_bh = max(0.0, min(1.0, new_bh))

        if new_bw > 0 and new_bh > 0:
            result.append([cls, new_cx, new_cy, new_bw, new_bh])

    return result


def tile_coords(img_w, img_h, tile_sz, overlap):
    """Yield (x1, y1, x2, y2) pixel tile rectangles covering the full image."""
    stride = int(tile_sz * (1.0 - overlap))
    stride = max(stride, 1)

    xs = list(range(0, img_w - tile_sz + 1, stride))
    if not xs or xs[-1] + tile_sz < img_w:
        xs.append(max(0, img_w - tile_sz))

    ys = list(range(0, img_h - tile_sz + 1, stride))
    if not ys or ys[-1] + tile_sz < img_h:
        ys.append(max(0, img_h - tile_sz))

    seen = set()
    for x in xs:
        for y in ys:
            x2 = min(x + tile_sz, img_w)
            y2 = min(y + tile_sz, img_h)
            x1 = max(0, x2 - tile_sz)
            y1 = max(0, y2 - tile_sz)
            key = (x1, y1, x2, y2)
            if key not in seen:
                seen.add(key)
                yield x1, y1, x2, y2


def process_split(img_paths, label_dir, out_img_dir, out_lbl_dir,
                  tile_sz, overlap, bg_keep_rate,
                  min_visibility, min_area_px):
    """Tile all images in a split and write results to output dirs."""
    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python not installed: pip install opencv-python")
        raise

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    total_tiles = 0
    total_boxes = 0
    skipped_bg  = 0
    class_counts = defaultdict(int)

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read {img_path.name}")
            continue

        img_h, img_w = img.shape[:2]
        label_path = label_dir / (img_path.stem + ".txt")
        boxes = read_labels(label_path)

        # If image is smaller than tile_sz in any dim, pad it
        if img_w < tile_sz or img_h < tile_sz:
            pad_w = max(0, tile_sz - img_w)
            pad_h = max(0, tile_sz - img_h)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_REFLECT_101)
            img_h, img_w = img.shape[:2]

        for tile_idx, (x1, y1, x2, y2) in enumerate(
                tile_coords(img_w, img_h, tile_sz, overlap)):

            tile_boxes = remap_boxes_to_tile(
                boxes, img_w, img_h, x1, y1, x2, y2,
                min_visibility=min_visibility,
                min_area_px=min_area_px)

            # Optionally drop background tiles
            if not tile_boxes:
                if random.random() > bg_keep_rate:
                    skipped_bg += 1
                    continue

            # Crop tile
            tile_img = img[y1:y2, x1:x2]
            if tile_img.shape[0] != tile_sz or tile_img.shape[1] != tile_sz:
                tile_img = cv2.resize(tile_img, (tile_sz, tile_sz),
                                      interpolation=cv2.INTER_LINEAR)

            stem = f"{img_path.stem}_t{tile_idx:04d}"
            cv2.imwrite(str(out_img_dir / (stem + ".jpg")), tile_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Write label
            lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
                     for b in tile_boxes]
            (out_lbl_dir / (stem + ".txt")).write_text("\n".join(lines))

            total_tiles += 1
            total_boxes += len(tile_boxes)
            for b in tile_boxes:
                class_counts[b[0]] += 1

    return total_tiles, total_boxes, skipped_bg, class_counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main(src, aug, dst, tile_sz, overlap, bg_keep_rate,
         min_visibility, min_area_px, val_from_aug):
    try:
        import yaml
        import cv2
    except ImportError as e:
        print(f"[ERROR] Missing package: {e}"); return

    src  = Path(src)
    dst  = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # ── Decide where to get images from ───────────────────────────────────
    # Priority: original source images (highest quality)
    # Fallback: augmented dataset

    aug_path = Path(aug) if aug else None

    # Load class names from the augment data.yaml if available
    nc, names = None, None
    if aug_path and (aug_path / "data.yaml").exists():
        with open(aug_path / "data.yaml") as f:
            d = yaml.safe_load(f)
        nc    = d.get("nc", 7)
        names = d.get("names", [])
    elif (src / "data.yaml").exists():
        with open(src / "data.yaml") as f:
            d = yaml.safe_load(f)
        nc    = d.get("nc", 7)
        names = d.get("names", [])

    print(f"\n{'='*60}")
    print(f"  Tile Dataset Generator")
    print(f"{'='*60}")
    print(f"  Source      : {src}")
    print(f"  Output      : {dst}")
    print(f"  Tile size   : {tile_sz}px")
    print(f"  Overlap     : {overlap*100:.0f}%")
    print(f"  BG keep rate: {bg_keep_rate*100:.0f}%")
    print(f"  nc={nc}  classes: {names}")
    print()

    # ── Find source images + labels ───────────────────────────────────────
    # Two possible layouts:
    # A) src/train/images/, src/train/labels/, src/val/images/, src/val/labels/
    # B) src/images/, src/labels/  (flat, original raw images)
    # C) src/ directly contains images (also original)

    splits_found = {}
    for split in ("train", "val"):
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            imgs = sorted(p for p in img_dir.iterdir()
                          if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})
            if imgs:
                splits_found[split] = (imgs, lbl_dir)

    if not splits_found:
        # Flat layout — tile everything and do a fresh 80/20 split
        img_extensions = {".jpg",".jpeg",".png",".bmp"}
        candidates = [src / "images", src]
        img_dir = next((p for p in candidates if p.is_dir()), src)
        lbl_dir = src / "labels" if (src / "labels").is_dir() else src

        all_imgs = sorted(p for p in img_dir.rglob("*")
                          if p.suffix.lower() in img_extensions
                          and "labels" not in str(p))
        if not all_imgs:
            print(f"[ERROR] No images found in {src}"); return

        random.shuffle(all_imgs)
        split_n = max(1, int(len(all_imgs) * 0.8))
        splits_found["train"] = (all_imgs[:split_n], lbl_dir)
        splits_found["val"]   = (all_imgs[split_n:], lbl_dir)
        print(f"  Flat layout detected — {len(all_imgs)} images → "
              f"{split_n} train / {len(all_imgs)-split_n} val")

    # ── If val_from_aug: use augmented val set as-is (untiled) ────────────
    # Rationale: val set should reflect real inference conditions (full image)
    # But if aug not provided, tile val too.

    grand_counts = defaultdict(int)

    for split, (img_paths, lbl_dir) in splits_found.items():
        out_img = dst / split / "images"
        out_lbl = dst / split / "labels"

        if split == "val" and val_from_aug and aug_path:
            # Copy val from augmented set unchanged
            aug_val_img = aug_path / "val" / "images"
            aug_val_lbl = aug_path / "val" / "labels"
            if aug_val_img.is_dir():
                print(f"  [val] Copying from augmented val set (no tiling) …")
                out_img.mkdir(parents=True, exist_ok=True)
                out_lbl.mkdir(parents=True, exist_ok=True)
                n = 0
                for p in aug_val_img.iterdir():
                    if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}:
                        shutil.copy2(p, out_img / p.name)
                        lp = aug_val_lbl / (p.stem + ".txt")
                        if lp.exists():
                            shutil.copy2(lp, out_lbl / lp.name)
                        n += 1
                print(f"         → {n} val images copied")
                continue

        print(f"  [{split}] Tiling {len(img_paths)} images …")
        n_tiles, n_boxes, n_skip, cls_counts = process_split(
            img_paths, lbl_dir, out_img, out_lbl,
            tile_sz, overlap, bg_keep_rate, min_visibility, min_area_px)

        for k, v in cls_counts.items():
            grand_counts[k] += v

        tiles_per_img = n_tiles / max(len(img_paths), 1)
        print(f"         → {n_tiles} tiles  ({tiles_per_img:.1f}×/img)  "
              f"{n_boxes} boxes  {n_skip} bg tiles skipped")

    # ── Write data.yaml ───────────────────────────────────────────────────
    has_val = (dst / "val" / "images").is_dir() and \
              any((dst / "val" / "images").iterdir())

    cfg = {
        "path"  : str(dst).replace("\\", "/"),
        "train" : "train/images",
        "val"   : "val/images" if has_val else "train/images",
        "nc"    : nc or max(grand_counts.keys(), default=0) + 1,
        "names" : names or [str(i) for i in range(nc or 7)],
    }
    import yaml
    with open(dst / "data.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)

    print(f"\n  data.yaml written → nc={cfg['nc']}")
    print(f"\n  Class distribution in tiled train set:")
    for idx, name in enumerate(cfg["names"]):
        print(f"    {idx}: {name:25s}  {grand_counts.get(idx, 0):6d} boxes")

    print(f"\n✓ Done!  Tiled dataset at: {dst}")
    print(f"\n  Next step — train on tiles:")
    print(f"    python train_finetune.py \\")
    print(f'      --data  "{dst / "data.yaml"}" \\')
    print(f'      --model "D:/MELSS/AOI/runs/gatekeeper_v2/weights/best.pt" \\')
    print(f'      --name  "gatekeeper_v3" \\')
    print(f'      --imgsz {tile_sz}')


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Tile a YOLO PCB dataset for small-component training")
    p.add_argument("--src",  required=True,
                   help="Source dataset dir (original images preferred) or augmented set")
    p.add_argument("--aug",  default=None,
                   help="Augmented dataset dir (for data.yaml class names and val set)")
    p.add_argument("--dst",  required=True,
                   help="Output tiled dataset dir")
    p.add_argument("--tile", type=int, default=960,
                   help="Tile size in pixels (default 960 — matches training imgsz)")
    p.add_argument("--overlap", type=float, default=0.25,
                   help="Tile overlap fraction (default 0.25 = 25%%)")
    p.add_argument("--bg-keep", type=float, default=0.05,
                   help="Fraction of empty (background) tiles to keep (default 0.05)")
    p.add_argument("--min-vis", type=float, default=0.25,
                   help="Min box visibility fraction after clipping (default 0.25)")
    p.add_argument("--min-area", type=int, default=16,
                   help="Min box area in pixels after clipping (default 16)")
    p.add_argument("--no-aug-val", action="store_true",
                   help="Also tile the val set (default: copy val from --aug untiled)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    random.seed(a.seed)
    main(
        src          = a.src,
        aug          = a.aug,
        dst          = a.dst,
        tile_sz      = a.tile,
        overlap      = a.overlap,
        bg_keep_rate = a.bg_keep,
        min_visibility = a.min_vis,
        min_area_px  = a.min_area,
        val_from_aug = not a.no_aug_val,
    )
