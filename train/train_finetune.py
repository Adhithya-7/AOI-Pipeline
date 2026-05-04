"""
train_finetune.py
─────────────────
Fine-tune the existing gatekeeper model on the new augmented dataset.
Uses the best known hyperparameters for a small high-res PCB dataset:
  • Warm-started from previous best.pt weights
  • Frozen backbone for the first N epochs, then full unfreeze
  • Cosine LR schedule, mixed precision, mosaic+mixup off (already augmented)
  • Large imgsz to preserve PCB detail
  • Early stopping with patience
  • Exports to ONNX + TorchScript after training

Usage:
    python train_finetune.py --data  "D:/MELSS/AOI/MyProject_aug/data.yaml"
                             --model "D:/MELSS/AOI/runs/gatekeeper_hires_finetune/weights/best.pt"
                             --name  "gatekeeper_v2"
                             --epochs 120
                             --imgsz  1280
"""

import os, sys, argparse
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_YAML   = r"D:\MELSS\AOI\AOI_Projects\new_tiled\data.yaml"
# Warm-start from the v3_tiled run (already 40 epochs trained on this dataset)
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "best.pt"
BASE_MODEL  = str(DEFAULT_MODEL)
RUN_NAME    = "gatekeeper_v4_tiled"
PROJECT_DIR = r"D:\MELSS\AOI\runs"

# FIX 1: More epochs — v3 loss was still declining at ep35, model undertrained.
EPOCHS        = 120
IMGSZ         = 960   # 960 fits batch=4 cleanly on 4GB; still high-res for PCBs
BATCH         = 4

WORKERS       = 4     # dataloader workers; lower if RAM is tight
# FIX 2: Longer freeze warm-up — head needs more time before unfreezing backbone.
FREEZE_EPOCHS = 10
# FIX 3: More patience — slow convergence was causing premature early-stop.
PATIENCE      = 20
DEVICE        = "0"   # "0" = GPU 0, "cpu" = CPU, "0,1" = multi-GPU


# ── Hyperparameters (fixed for v4) ────────────────────────────────────────────
#
#   Key fixes vs v3:
#     • lr0 halved to 0.0005 — NaN spike in cls_loss at ep3 was LR instability.
#     • lrf raised to 0.05 — floor is now lr0*lrf=2.5e-5, avoids LR collapsing.
#     • mosaic = 0.4 — tiled images ARE small crops; mosaic teaches the model
#       to assemble context from adjacent tiles, which is exactly what inference
#       over a large PCB requires. Previous logic of "already augmented" was
#       wrong for a TILED dataset.
#     • iou = 0.60 — previous threshold of 0.70 aggressively suppressed valid
#       predictions on small tiled components, causing recall=24%. Lower IoU
#       threshold increases positive assignments during training → better recall.
#     • scale reduced to 0.2 — large scale jitter (±30%) distorts small
#       components in tiles; ±20% is safer for this tile size.
#     • erasing reduced to 0.2 — random erasing at 0.3 was too aggressive for
#       small partially-visible components in border tiles.
#
# NBS must equal BATCH so ultralytics LR scaling (lr × batch/nbs) = 1.0.
NBS = BATCH   # always keep in sync with BATCH

HYPER = dict(
    # ── Optimiser ──────────────────────────────────────────────────────────
    optimizer   = "AdamW",
    lr0         = 0.0005,       # FIX: halved — v3 had NaN cls_loss at ep3
    lrf         = 0.05,         # FIX: higher floor (lr0*lrf=2.5e-5)
    momentum    = 0.937,
    weight_decay= 0.0005,
    warmup_epochs   = 3.0,
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.05,     # FIX: lowered from 0.1 to prevent initial overshoot

    # ── Loss ───────────────────────────────────────────────────────────────
    box         = 7.5,
    cls         = 0.5,
    dfl         = 1.5,

    # ── Online augmentation ────────────────────────────────────────────────
    hsv_h       = 0.010,
    hsv_s       = 0.50,
    hsv_v       = 0.30,
    degrees     = 3.0,
    translate   = 0.05,
    scale       = 0.20,         # FIX: reduced from 0.30 — less distortion on tiles
    shear       = 0.0,
    perspective = 0.0001,
    flipud      = 0.0,
    fliplr      = 0.5,
    mosaic      = 0.4,          # FIX: re-enabled — tiles BENEFIT from mosaic context
    mixup       = 0.0,
    copy_paste  = 0.0,
    erasing     = 0.20,         # FIX: reduced from 0.3 — safer for small tile objects

    # ── NMS / IoU ──────────────────────────────────────────────────────────
    iou         = 0.60,         # FIX: lowered from 0.70 — v3 recall was only 24%
    conf        = 0.001,
    nms         = True,
)


# ══════════════════════════════════════════════════════════════════════════════
def check_env():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.  pip install ultralytics")
        sys.exit(1)
    try:
        import torch
        gpu = torch.cuda.is_available()
        if gpu:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] {name}  ({vram:.1f} GB VRAM)")
        else:
            print("[WARN] No CUDA GPU found — training on CPU will be very slow.")
    except Exception as e:
        print(f"[WARN] Could not query GPU: {e}")


def count_images(data_yaml):
    """Quick sanity check — count train images."""
    import yaml
    try:
        with open(data_yaml) as f:
            d = yaml.safe_load(f)
        base = Path(d.get("path", Path(data_yaml).parent))
        train_val = d.get("train", "images")
        
        if isinstance(train_val, str):
            train_val = [train_val]
            
        n = 0
        for p in train_val:
            train_dir = base / p
            if train_dir.is_dir():
                n += sum(1 for f in train_dir.iterdir()
                         if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})
        return n
    except Exception as e:
        print(f"[WARN] Could not count images: {e}")
        return 0


def main(data, model_path, name, epochs, imgsz, batch, device):
    import os, gc
    # Reduces VRAM fragmentation — critical for 4GB GPUs at large imgsz
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from ultralytics import YOLO
    import torch

    print("=" * 60)
    print("  MELSS PCB Fine-tune Trainer")
    print("=" * 60)
    check_env()

    # Validate inputs
    if not os.path.exists(data):
        print(f"[ERROR] data.yaml not found: {data}"); sys.exit(1)
    if not os.path.exists(model_path):
        print(f"[ERROR] base model not found: {model_path}"); sys.exit(1)

    n_imgs = count_images(data)
    print(f"[DATA] {n_imgs} training images  →  {data}")
    print(f"[MODEL] {model_path}")
    print(f"[RUN] {name}  epochs={epochs}  imgsz={imgsz}  batch={batch}")
    print()

    # ── Phase 1: Frozen backbone warm-up ──────────────────────────────────
    print(f"Phase 1 — Frozen backbone ({FREEZE_EPOCHS} epochs) …")
    model = YOLO(model_path)

    # Determine number of backbone layers to freeze (first ~10 layers = backbone)
    # For YOLOv8 the backbone ends around layer index 9-10
    BACKBONE_FREEZE = 10

    results_p1 = model.train(
        data        = data,
        epochs      = FREEZE_EPOCHS,
        imgsz       = imgsz,
        batch       = batch,
        device      = device,
        workers     = WORKERS,
        project     = PROJECT_DIR,
        name        = name + "_phase1",
        exist_ok    = True,
        patience    = 0,            # no early stop during warm-up
        save        = True,
        save_period = 5,
        plots       = True,
        amp         = True,         # mixed precision
        cos_lr      = True,
        freeze      = BACKBONE_FREEZE,
        close_mosaic= 0,            # mosaic already off via hyp, this kills it in scheduler
        verbose     = True,
        # Warm-up hyps only
        nbs         = NBS,              # must match batch to prevent LR scaling
        lr0         = HYPER["lr0"] * 2,   # slightly higher for head warm-up
        lrf         = HYPER["lrf"],
        optimizer   = HYPER["optimizer"],
        weight_decay= HYPER["weight_decay"],
        warmup_epochs   = HYPER["warmup_epochs"],
        warmup_momentum = HYPER["warmup_momentum"],
        warmup_bias_lr  = HYPER["warmup_bias_lr"],
        momentum    = HYPER["momentum"],
        box         = HYPER["box"],
        cls         = HYPER["cls"],
        dfl         = HYPER["dfl"],
        # Augmentation (minimal during warm-up)
        hsv_h       = HYPER["hsv_h"],
        hsv_s       = HYPER["hsv_s"],
        hsv_v       = HYPER["hsv_v"],
        degrees     = HYPER["degrees"],
        translate   = HYPER["translate"],
        scale       = HYPER["scale"],
        fliplr      = HYPER["fliplr"],
        mosaic      = 0.0,
        mixup       = 0.0,
        copy_paste  = 0.0,
        erasing     = HYPER["erasing"],
    )

    # ── Phase 2: Full fine-tune with early stopping ────────────────────────
    # Explicitly free all GPU memory from phase 1 before loading phase 2.
    # On 4GB GPUs, leftover allocations from the trainer cause autobatch to OOM.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  GPU memory freed. Free: {(torch.cuda.mem_get_info()[0]/1e9):.2f} GB")

    p1_best = Path(PROJECT_DIR) / (name + "_phase1") / "weights" / "best.pt"
    if not p1_best.exists():
        p1_best = Path(PROJECT_DIR) / (name + "_phase1") / "weights" / "last.pt"
    print(f"\nPhase 2 — Full fine-tune from {p1_best}  ({epochs - FREEZE_EPOCHS} epochs) …")

    model2 = YOLO(str(p1_best))

    remaining = epochs - FREEZE_EPOCHS
    results_p2 = model2.train(
        data        = data,
        epochs      = remaining,
        imgsz       = imgsz,
        batch       = batch,
        device      = device,
        workers     = WORKERS,
        project     = PROJECT_DIR,
        name        = name,
        exist_ok    = True,
        patience    = PATIENCE,
        save        = True,
        save_period = 10,
        plots       = True,
        amp         = True,
        cos_lr      = True,
        freeze      = 0,            # unfreeze everything
        close_mosaic= 0,
        verbose     = True,
        # Full hyps
        nbs         = NBS,
        lr0         = HYPER["lr0"],
        lrf         = HYPER["lrf"],
        optimizer   = HYPER["optimizer"],
        weight_decay= HYPER["weight_decay"],
        warmup_epochs   = 1.0,     # short re-warm after phase transition
        warmup_momentum = HYPER["warmup_momentum"],
        warmup_bias_lr  = HYPER["warmup_bias_lr"],
        momentum    = HYPER["momentum"],
        box         = HYPER["box"],
        cls         = HYPER["cls"],
        dfl         = HYPER["dfl"],
        hsv_h       = HYPER["hsv_h"],
        hsv_s       = HYPER["hsv_s"],
        hsv_v       = HYPER["hsv_v"],
        degrees     = HYPER["degrees"],
        translate   = HYPER["translate"],
        scale       = HYPER["scale"],
        shear       = HYPER["shear"],
        perspective = HYPER["perspective"],
        flipud      = HYPER["flipud"],
        fliplr      = HYPER["fliplr"],
        mosaic      = 0.0,
        mixup       = 0.0,
        copy_paste  = 0.0,
        erasing     = HYPER["erasing"],
        iou         = HYPER["iou"],
        conf        = HYPER["conf"],
    )

    # ── Export ─────────────────────────────────────────────────────────────
    best = Path(PROJECT_DIR) / name / "weights" / "best.pt"
    print(f"\n✓ Training complete.  Best weights → {best}")

    if best.exists():
        print("\nExporting to ONNX …")
        try:
            export_model = YOLO(str(best))
            export_model.export(
                format  = "onnx",
                imgsz   = imgsz,
                simplify= True,
                opset   = 17,
                dynamic = False,
            )
            print("  → ONNX export done")
        except Exception as e:
            print(f"  [WARN] ONNX export failed: {e}")

        print("Exporting to TorchScript …")
        try:
            export_model2 = YOLO(str(best))
            export_model2.export(format="torchscript", imgsz=imgsz, optimize=True)
            print("  → TorchScript export done")
        except Exception as e:
            print(f"  [WARN] TorchScript export failed: {e}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Run dir  :  {Path(PROJECT_DIR) / name}")
    print(f"  Best .pt :  {best}")
    onnx = best.with_suffix(".onnx")
    if onnx.exists(): print(f"  ONNX     :  {onnx}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB model fine-tuner")
    parser.add_argument("--data",   default=DATA_YAML,  help="Path to data.yaml")
    parser.add_argument("--model",  default=BASE_MODEL, help="Base .pt to start from")
    parser.add_argument("--name",   default=RUN_NAME,   help="Run name")
    parser.add_argument("--epochs", type=int, default=EPOCHS,  help="Total epochs (both phases)")
    parser.add_argument("--imgsz",  type=int, default=IMGSZ,   help="Image size")
    parser.add_argument("--batch",  type=int, default=BATCH,   help="Batch size (-1=auto)")
    parser.add_argument("--device", default=DEVICE, help="Device: 0, cpu, 0,1 …")
    a = parser.parse_args()
    main(a.data, a.model, a.name, a.epochs, a.imgsz, a.batch, a.device)