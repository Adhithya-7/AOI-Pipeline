import os
import sys
import time
import cv2
from ultralytics import YOLO

import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = str(ROOT / "models" / "best.pt")

print(f"Testing model: {MODEL_PATH}")
print(f"Exists: {os.path.exists(MODEL_PATH)}")

try:
    model = YOLO(MODEL_PATH, task="detect")
    print("Model loaded successfully")
except Exception as e:
    print(f"Model load failed: {e}")
    sys.exit(1)

# Dummy image
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)

for i in range(3):
    t0 = time.time()
    try:
        res = model.predict(img, imgsz=640, verbose=False)
        print(f"Call {i}: {len(res[0].boxes)} boxes in {time.time()-t0:.4f}s")
    except Exception as e:
        print(f"Call {i} FAILED: {e}")

print("\nTesting SAHI loading...")
try:
    from sahi import AutoDetectionModel
    is_exported = MODEL_PATH.endswith("_openvino_model") or MODEL_PATH.endswith(".onnx")
    _dev = "cpu" if not is_exported else None
    print(f"Using device: {_dev}")
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=MODEL_PATH,
        confidence_threshold=0.25, device=_dev)
    print("SAHI model loaded successfully")
except Exception as e:
    print(f"SAHI model load failed: {e}")
    import traceback
    traceback.print_exc()
