# MELSS AOI Training & Inspection Pipeline

## 1. Project Overview
This project provides a comprehensive, professional-grade Automated Optical Inspection (AOI) framework tailored for Printed Circuit Board (PCB) defect detection. The system detects components on a PCB and identifies manufacturing defects by comparing test boards against a known "golden" reference board.

It utilizes a hybrid approach:
1. **Deep Learning (YOLOv8)**: For robust, real-time component localization and bounding box prediction.
2. **Heuristic-based Analytical Logic**: For high-precision defect categorization using structural signals like SSIM (Structural Similarity Index), ZNCC (Zero-normalized cross-correlation), and pixel variance.

The codebase is split into two primary domains:
- **Training Pipeline (`train/`)**: Designed to handle small PCB components. It includes tools for offline dataset augmentation, high-resolution image tiling (to detect tiny components without losing pixel clarity during model downscaling), and a customized two-phase YOLOv8 fine-tuning process.
- **Evaluation & Inspection System (`src/` & `modular_aoi/`)**: Categorizes defects into `missing`, `misaligned`, `wrong_component`, and `wrong_polarity`. It includes command-line evaluation tools for regression testing and a full graphical user interface (GUI) for live camera inspections, offline batch processing, and continuous model retraining.

## 2. Setup & Installation

**Requirements:** 
- Python >= 3.8
- Optional: CUDA-compatible GPU for accelerated model training and real-time inference.

**Install Dependencies:**
```bash
pip install opencv-python numpy torch ultralytics scipy PySide6 platformdirs pytesseract pyzbar sahi
```
*(Note: `pytesseract` and `pyzbar` may require additional system-level libraries installed via your package manager depending on your OS for OCR and barcode reading capabilities. For example, installing Tesseract-OCR on Windows.)*

## 3. Recommended Workflow
1. **Labeling**: Use `src/melss_labeler.py` to annotate raw golden boards.
2. **Augmentation**: Run `train/augment_dataset.py` to artificially expand your dataset with lighting and geometric variations.
3. **Tiling**: Execute `train/tile_dataset.py` to slice large boards into 960x960 overlapping tiles, preserving small component details.
4. **Training**: Use `train/train_finetune.py` to fine-tune the YOLOv8 model on the tiled dataset.
5. **Validation**: Generate synthetic defects with `src/generate_defect.py` and score the model using `src/aoi_evaluator.py`.
6. **Deployment**: Run the full modular interface via `modular_aoi/optical_inspection_system.py` to inspect real boards.

---

## 4. File Reference

### 🧠 Training Pipeline (`train/`)

#### `train/pipeline.py`
The master orchestrator for the training workflow. It automatically runs augmentation, class remapping, dataset tiling, and YOLO fine-tuning sequentially based on a centralized configuration dictionary.
- **Input**: Hardcoded paths and configurations defined at the top of the script.
- **Output**: Generates augmented datasets, tiled datasets, and triggers the `train_finetune.py` process.
- **Sample Command**:
  ```bash
  python train/pipeline.py
  ```

#### `train/augment_dataset.py`
Performs robust offline data augmentation on a YOLO-format dataset. It applies geometric transforms (flips, mild rotations, scaling, perspective warping) and color modifications (brightness, contrast, HSV shifts, noise). Crucially, it accurately remaps normalized YOLO bounding boxes during these spatial transformations and splits the data into 80/20 train/val sets.
- **Input**: A YOLO-format project folder (`--src`), desired target image count (`--target`).
- **Output**: A new folder with augmented images, updated YOLO `.txt` labels, and a `data.yaml` file.
- **Sample Command**:
  ```bash
  python train/augment_dataset.py --src "D:/MELSS/AOI/MyProject" --dst "D:/MELSS/AOI/MyProject_aug" --target 280 --seed 42
  ```

#### `train/tile_dataset.py`
Solves the "small object detection" problem for PCBs. It slices high-resolution PCB images into smaller, overlapping tiles (e.g., 960x960 pixels with 25% overlap) to preserve the actual pixel size of tiny resistors and capacitors. It remaps normalized bounding boxes to local tile coordinates and smartly drops empty background tiles.
- **Input**: Source dataset (`--src`), optional augmented dataset (`--aug`), tile size (`--tile`), overlap (`--overlap`).
- **Output**: Tiled images and labels saved into a new output dataset directory ready for YOLO.
- **Sample Command**:
  ```bash
  python train/tile_dataset.py --src "D:/MELSS/AOI/AOI_Projects/new" --aug "D:/MELSS/AOI/AOI_Projects/new_augment" --dst "D:/MELSS/AOI/AOI_Projects/new_tiled" --tile 960 --overlap 0.25
  ```

#### `train/train_finetune.py`
Fine-tunes a base YOLOv8 model on the generated dataset using a specialized two-phase training strategy: a frozen backbone warm-up (to protect pre-trained features) followed by a full unfreeze. It employs specific hyperparameters optimized for tiled data, such as restoring mosaic context and lowering IoU thresholds to improve recall on cut-off components.
- **Input**: Path to the dataset `data.yaml` (`--data`), base model weights (`--model`), run name (`--name`), epochs (`--epochs`), and image size (`--imgsz`).
- **Output**: Trained best `.pt` weights, exported ONNX, and TorchScript models.
- **Sample Command**:
  ```bash
  python train/train_finetune.py --data "D:/MELSS/AOI/AOI_Projects/new_tiled/data.yaml" --model "D:/MELSS/AOI/runs/gatekeeper_v2/weights/best.pt" --name "gatekeeper_v3" --epochs 120 --imgsz 960
  ```

---

### 🔬 Evaluation & Core Scripts (`src/`)

#### `src/aoi_evaluator.py`
The analytical heart of the AOI system. It analyzes detection results on test boards against a golden reference to pinpoint defects. Rather than relying solely on YOLO classes, it uses structural heuristics—SSIM for structural presence, ZNCC (Zero-normalized cross-correlation) for polarity rotation, and pixel variance—to accurately classify defects and prevent false positives.
- **Input**: Golden board image, golden annotations, and test images loaded within the script.
- **Output**: Prints a detailed evaluation report showing correct matches and categorized defect counts.
- **Sample Command**:
  ```bash
  python src/aoi_evaluator.py
  ```

#### `src/generate_defect.py`
A crucial tool for regression testing. It creates synthetic defect images by algorithmically modifying a golden board image. It injects defects via color-matched background fills (simulating missing components), donor component swaps (simulating wrong components), and 180-degree rotations (simulating wrong polarity).
- **Input**: Golden board image and baseline component bounding box coordinates.
- **Output**: A suite of synthetic test images with injected defects and a `manifest.json` cataloging every anomaly.
- **Sample Command**:
  ```bash
  python src/generate_defect.py
  ```

#### `src/melss_labeler.py`
A custom GUI-based dataset labeling tool built with PySide6 for creating and managing YOLO-format annotations. It provides an intuitive interface to draw bounding boxes and assign PCB component classes.
- **Input**: Folder of raw images selected via GUI.
- **Output**: Corresponding YOLO format `.txt` label files and a `classes.txt` file.
- **Sample Command**:
  ```bash
  python src/melss_labeler.py
  ```

#### `src/pcb_aoi.py`
A standalone interactive command-line/GUI tool for running a localized AOI inspection workflow. It aligns a test board to the golden reference using ORB features and RANSAC homography, detects components, and executes the comparison logic to find anomalies.
- **Input**: Paths to the golden board, test board, and the YOLO model weights.
- **Output**: A visual overlay window highlighting the detected defects with colored bounding boxes.
- **Sample Command**:
  ```bash
  python src/pcb_aoi.py --golden "D:/MELSS/AOI/golden.jpg" --test "D:/MELSS/AOI/test.jpg" --model "D:/MELSS/AOI/models/best.pt"
  ```

---

### 🖥️ Main Application (`modular_aoi/`)

#### `modular_aoi/optical_inspection_system.py`
The primary entry point to launch the modular Optical Inspection System GUI. This robust PyQt6 application integrates live camera feeds, offline processing tabs, model training UI, history tracking, and the core AOI inference logic into a single cohesive product.
- **Input**: User interactions via GUI, pre-trained model weights, and live camera feeds.
- **Output**: Real-time UI updates, inspection reports logged to a SQLite database, and exported project data.
- **Sample Command**:
  ```bash
  python modular_aoi/optical_inspection_system.py
  ```

#### The `modular_aoi/ois/` Package
This internal package contains the refactored, modular components of the optical inspection system, ensuring clean architecture and maintainability:
- **`__init__.py`**: Exposes the primary interface of the `ois` package.
- **`aoi_engine.py`**: Houses the core mathematical and heuristic AOI logic completely decoupled from the UI. It calculates bounding box overlaps, performs Hungarian matching for optimal component pairing, and runs the multi-stage pixel heuristic checks using golden and test pairs.
- **`filters.py`**: Implements high-speed image processing filters (CLAHE, gamma correction, unsharp masking) to normalize board lighting and contrast before YOLO inference.
- **`main_window.py`**: Defines the main application GUI window, side navigation bar, top toolbar, and the overall widget layout structure.
- **`theme.py`**: Centralizes UI styling constants (color palettes, dynamic fonts) and PyQt stylesheets for standardizing the app's appearance.
- **`threads.py`**: Manages background worker `QThread` classes to keep the UI responsive, including `CameraThread` for video capture, `InferenceThread` for real-time object detection, and `ThumbThread` for asynchronous image thumbnail loading.
- **`utils.py`**: Provides shared utility functions, data structures (`_AOIComp`), global configuration singletons (`ProjectConfig`), database management (`HistoryDB`), and YOLO/SAHI inference wrappers.
- **`widgets.py`**: Implements custom reusable PyQt6 widgets such as fast-logging text areas, modern statistical cards, custom sliders, and component list items.

#### UI Tabs (`modular_aoi/ois/tabs/`)
Individual modules defining the specific UI layouts and interaction logic for each main section of the application:
- **`aoi_tab.py`**: Live camera and board inspection functionality.
- **`filters_tab.py`**: Real-time adjustment of image preprocessing pipelines.
- **`history_tab.py`**: SQLite-backed review of past inspection results.
- **`infer_tab.py`**: Isolated model inference testing and visualization.
- **`run_tab.py`**: Offline batch processing of captured board images.
- **`train_tab.py`**: UI for triggering the YOLO model retraining pipeline.

---

### 🗂️ Data & Configurations

#### `test_defects/`
- **`manifest.json`**: A catalog mapping each synthetic test board image (generated by `generate_defect.py`) to its specifically injected defects. Used as ground truth for evaluating the AOI engine.
- **`golden_components.json`**: Defines the exact bounding boxes, class labels, and confidence scores for every component on the original golden reference board.

#### `data/Main_dataset/` & `data/Real_images/TEST/`
- **`data.yaml`**: Standard YOLO dataset configuration files. Created by the augmentation or tiling pipelines, these files are parsed by YOLO during training to map integer class indices to human-readable strings and to specify training/validation image directories.
