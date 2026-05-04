"""
filters.py — Image processing filter system and auto-calibration.

FilterNode hierarchy, apply_filters, analysis, auto-pipeline builder,
AutoCalibrateWorker (QThread), and run_roi.
"""
import os, sys, copy, math, time

if __package__ in (None, ""):
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    __package__ = "ois"

from PySide6.QtCore import QThread, Signal

from .utils import HAS_CV2, HAS_NP, HAS_YOLO, HAS_OCR, HAS_ZBAR, safe_predict

if HAS_CV2:
    import cv2
if HAS_NP:
    import numpy as np
if HAS_YOLO:
    from ultralytics import YOLO as _YOLO
if HAS_OCR:
    import pytesseract
if HAS_ZBAR:
    from pyzbar import pyzbar


# ── Filter nodes ─────────────────────────────────────────────────────────────

class FilterNode:
    def __init__(self,name,params=None): self.name=name; self.params=params or {}; self.enabled=True
    def apply(self,img): return img
    def get_config(self): return {"name":self.name,"params":self.params,"enabled":self.enabled}

class CLAHEFilter(FilterNode):
    def __init__(self,clip=2.0,tile=8):
        super().__init__("CLAHE",{"clip_limit":clip,"tile_size":tile})
        self._clahe    = None  # cached cv2.CLAHE instance
        self._clahe_cl = None  # clip_limit at cache creation time
        self._clahe_ts = None  # tile_size at cache creation time
    def _get_clahe(self):
        cl = self.params["clip_limit"]
        ts = self.params["tile_size"]
        if self._clahe is None or self._clahe_cl != cl or self._clahe_ts != ts:
            self._clahe    = cv2.createCLAHE(clipLimit=cl,
                                              tileGridSize=(ts, ts))
            self._clahe_cl = cl
            self._clahe_ts = ts
        return self._clahe
    def apply(self,img):
        if not self.enabled or not HAS_CV2: return img
        lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        cl=self._get_clahe().apply(l)
        return cv2.cvtColor(cv2.merge((cl,a,b)),cv2.COLOR_LAB2BGR)

class GaussBlurFilter(FilterNode):
    def __init__(self,k=5): super().__init__("Gaussian Blur",{"kernel":k})
    def apply(self,img):
        if not self.enabled or not HAS_CV2: return img
        k=self.params["kernel"]; k=k if k%2==1 else k+1
        return cv2.GaussianBlur(img,(k,k),0)

class CannyFilter(FilterNode):
    def __init__(self,t1=50,t2=150): super().__init__("Canny Edge",{"thresh1":t1,"thresh2":t2})
    def apply(self,img):
        if not self.enabled or not HAS_CV2: return img
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(cv2.Canny(gray,self.params["thresh1"],self.params["thresh2"]),cv2.COLOR_GRAY2BGR)

class AdaptThreshFilter(FilterNode):
    def __init__(self,bs=11,c=2): super().__init__("Adaptive Thresh",{"block_size":bs,"c":c})
    def apply(self,img):
        if not self.enabled or not HAS_CV2: return img
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); bs=self.params["block_size"]
        bs=bs if bs%2==1 else bs+1
        return cv2.cvtColor(cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,bs,self.params["c"]),cv2.COLOR_GRAY2BGR)

class MedianBlurFilter(FilterNode):
    def __init__(self,k=5): super().__init__("Median Blur",{"kernel":k})
    def apply(self,img):
        if not self.enabled or not HAS_CV2: return img
        k=self.params["kernel"]; k=k if k%2==1 else k+1
        return cv2.medianBlur(img,max(1,k))

class BilateralDenoiseFilter(FilterNode):
    def __init__(self,d=9,sc=60,ss=60): super().__init__("Bilateral Denoise",{"d":d,"sigma_color":sc,"sigma_space":ss})
    def apply(self,img):
        if not self.enabled or not HAS_CV2: return img
        p=self.params
        return cv2.bilateralFilter(img,int(p["d"]),int(p["sigma_color"]),int(p["sigma_space"]))

class BrightnessContrastFilter(FilterNode):
    def __init__(self,a=1.2,b=0): super().__init__("Brightness/Contrast",{"alpha":a,"beta":b})
    def apply(self,img):
        if not self.enabled or not HAS_CV2: return img
        return cv2.convertScaleAbs(img,alpha=float(self.params["alpha"]),beta=int(self.params["beta"]))

class GammaFilter(FilterNode):
    def __init__(self,g=1.2):
        super().__init__("Gamma",{"gamma":g})
        self._lut = None; self._lut_g = None
    def apply(self,img):
        if not self.enabled or not HAS_CV2: return img
        g=max(0.05,float(self.params["gamma"]))
        if self._lut is None or self._lut_g != g:
            self._lut_g = g
            self._lut = np.array([((i/255.0)**(1.0/g))*255 for i in range(256)],dtype=np.uint8)
        return cv2.LUT(img,self._lut)

class SharpenFilter(FilterNode):
    _KERNEL = None   # shared class-level kernel cache
    def __init__(self,a=1.0): super().__init__("Sharpen",{"amount":a})
    def apply(self,img):
        if not self.enabled or not HAS_CV2: return img
        a=float(self.params["amount"])
        if SharpenFilter._KERNEL is None:
            SharpenFilter._KERNEL = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)
        sh=cv2.filter2D(img,-1,SharpenFilter._KERNEL)
        return cv2.addWeighted(img,max(0.0,1.0-a),sh,max(0.1,a),0)

class UnsharpMaskFilter(FilterNode):
    def __init__(self,s=2.0,a=1.4): super().__init__("Unsharp Mask",{"sigma":s,"amount":a})
    def apply(self,img):
        if not self.enabled or not HAS_CV2: return img
        s=max(0.1,float(self.params["sigma"])); a=float(self.params["amount"])
        gb=cv2.GaussianBlur(img,(0,0),s)
        return cv2.addWeighted(img,1.0+a,gb,-a,0)

class MorphCloseFilter(FilterNode):
    def __init__(self,k=5,it=1): super().__init__("Morph Close",{"kernel":k,"iters":it})
    def apply(self,img):
        if not self.enabled or not HAS_CV2: return img
        k=int(self.params["kernel"]); k=k if k%2==1 else k+1
        it=max(1,int(self.params["iters"]))
        ker=cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
        return cv2.morphologyEx(img,cv2.MORPH_CLOSE,ker,iterations=it)

class PCBColorNormFilter(FilterNode):
    """Suppress green FR-4 substrate and boost component-colour contrast."""
    def __init__(self, green_desat: float = 0.70, comp_boost: float = 1.30):
        super().__init__("PCB Color Norm",
                         {"green_desat": green_desat, "comp_boost": comp_boost})

    def apply(self, img):
        if not self.enabled or not (HAS_CV2 and HAS_NP):
            return img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        green_mask = (H >= 17) & (H <= 45) & (S > 40) & (V > 40)
        desat = float(self.params["green_desat"])
        boost = float(self.params["comp_boost"])
        # Suppress substrate
        S[green_mask]  = S[green_mask]  * (1.0 - desat)
        V[green_mask]  = V[green_mask]  * 0.75
        # Lift component colours
        non_green = ~green_mask
        S[non_green] = np.clip(S[non_green] * boost, 0, 255)
        hsv[:, :, 1] = S
        hsv[:, :, 2] = V
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                            cv2.COLOR_HSV2BGR)


FILTER_REGISTRY={
    "CLAHE":CLAHEFilter,
    "Gaussian Blur":GaussBlurFilter,
    "Median Blur":MedianBlurFilter,
    "Bilateral Denoise":BilateralDenoiseFilter,
    "Canny Edge":CannyFilter,
    "Adaptive Thresh":AdaptThreshFilter,
    "Brightness/Contrast":BrightnessContrastFilter,
    "Gamma":GammaFilter,
    "Sharpen":SharpenFilter,
    "Unsharp Mask":UnsharpMaskFilter,
    "Morph Close":MorphCloseFilter,
}

def apply_filters(img, filters):
    if not filters or not any(f.enabled for f in filters):
        return img.copy()
    out = img.copy()
    for f in filters:
        if f.enabled:
            try: out = f.apply(out)
            except: pass
    return out


# ── Image analysis ───────────────────────────────────────────────────────────

def _analyse_image(img):
    """Return comprehensive image quality metrics used by auto-calibrate."""
    if not HAS_CV2 or not HAS_NP or img is None:
        return {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_u8 = gray.astype(np.uint8)

    mean_bright  = float(gray.mean())
    std_contrast = float(gray.std())
    lap_var      = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())

    _NOISE_MAX = 480
    _noise_scale = _NOISE_MAX / max(1, max(img.shape[:2]))
    if _noise_scale < 1.0:
        _noise_img = cv2.resize(img, (0, 0),
                                fx=_noise_scale, fy=_noise_scale,
                                interpolation=cv2.INTER_AREA)
    else:
        _noise_img = img
    bl        = cv2.bilateralFilter(_noise_img, 5, 30, 30)
    noise_est = float(np.abs(_noise_img.astype(np.float32)
                             - bl.astype(np.float32)).mean())

    hist       = cv2.calcHist([gray_u8],[0],None,[256],[0,256]).flatten()
    total      = max(1, float(hist.sum()))
    dark_frac  = float(hist[:64].sum()  / total)
    bright_frac= float(hist[192:].sum() / total)
    mid_frac   = float(hist[64:192].sum()/ total)

    prob  = hist / total
    prob  = prob[prob > 0]
    entropy = float(-np.sum(prob * np.log2(prob)))

    auto_lo  = max(10, mean_bright * 0.33)
    auto_hi  = min(255, mean_bright * 1.2)
    edges    = cv2.Canny(gray_u8, auto_lo, auto_hi)
    edge_density = float(edges.mean()) / 255.0

    b_ch, g_ch, r_ch = [img[:,:,c].astype(np.float32).mean() for c in range(3)]
    ch_mean   = (b_ch + g_ch + r_ch) / 3.0
    ch_imbalance = float(max(abs(b_ch - ch_mean), abs(g_ch - ch_mean), abs(r_ch - ch_mean)))

    H, W = gray_u8.shape
    th, tw = max(1, H // 8), max(1, W // 8)
    H_cr = (H // th) * th
    W_cr = (W // tw) * tw
    if H_cr > 0 and W_cr > 0:
        tiles = gray[:H_cr, :W_cr].reshape(H_cr // th, th, W_cr // tw, tw)
        local_std = float(tiles.std(axis=(1, 3)).mean())
    else:
        local_std = std_contrast

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H_ch = hsv_img[:, :, 0].astype(np.float32)
    S_ch = hsv_img[:, :, 1].astype(np.float32)
    V_ch = hsv_img[:, :, 2].astype(np.float32)
    green_mask_pcb = (H_ch >= 17) & (H_ch <= 45) & (S_ch > 40) & (V_ch > 40)
    green_board_frac = float(green_mask_pcb.sum()) / max(1, green_mask_pcb.size)

    non_green_u8 = (~green_mask_pcb).astype(np.uint8) * 255
    contours_all, _ = cv2.findContours(non_green_u8,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comp_blobs = [c for c in contours_all if 100 < cv2.contourArea(c) < 50000]
    component_blob_density = float(len(comp_blobs)) / max(1.0, H * W / 10000.0)

    MAX_FFT_DIM = 256
    fft_gray = gray
    if min(H, W) > MAX_FFT_DIM:
        scale_f  = MAX_FFT_DIM / min(H, W)
        fft_gray = cv2.resize(gray, (max(1, int(W * scale_f)),
                                     max(1, int(H * scale_f))),
                              interpolation=cv2.INTER_AREA)
    Hf, Wf = fft_gray.shape
    fft_mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(fft_gray))))
    cy_f, cx_f = Hf // 2, Wf // 2
    r_max_f = min(cy_f, cx_f)
    Y_g, X_g = np.ogrid[:Hf, :Wf]
    dist_from_centre = np.sqrt((Y_g - cy_f) ** 2 + (X_g - cx_f) ** 2)
    mid_band_mask = (dist_from_centre >= r_max_f * 0.05) & \
                    (dist_from_centre <= r_max_f * 0.40)
    spatial_freq_peak = float(fft_mag[mid_band_mask].mean()) if mid_band_mask.any() else 0.0

    return dict(
        mean=mean_bright, std=std_contrast, lap=lap_var,
        noise=noise_est, dark_frac=dark_frac, bright_frac=bright_frac,
        mid_frac=mid_frac, entropy=entropy, edge_density=edge_density,
        ch_imbalance=ch_imbalance, local_std=local_std,
        green_board_frac=green_board_frac,
        component_blob_density=component_blob_density,
        spatial_freq_peak=spatial_freq_peak,
    )


def _build_auto_pipeline(metrics):
    """High-level intelligent pipeline builder."""
    if not metrics:
        return []

    mean      = metrics.get("mean", 128)
    std       = metrics.get("std", 50)
    lap       = metrics.get("lap", 200)
    noise     = metrics.get("noise", 5)
    dark_f    = metrics.get("dark_frac", 0.2)
    brt_f     = metrics.get("bright_frac", 0.2)
    mid_f     = metrics.get("mid_frac", 0.6)
    entropy   = metrics.get("entropy", 6.0)
    edge_den  = metrics.get("edge_density", 0.1)
    ch_imb    = metrics.get("ch_imbalance", 5.0)
    local_std = metrics.get("local_std", 40.0)

    pipeline = []

    green_board_frac = metrics.get("green_board_frac", 0.0)
    comp_blob_density = metrics.get("component_blob_density", 0.0)
    spatial_freq_peak = metrics.get("spatial_freq_peak", 0.0)
    if green_board_frac > 0.15:
        desat = round(min(0.85, max(0.50, green_board_frac * 1.2)), 2)
        comp_boost = round(min(1.6, max(1.1, 1.3 + max(0.0, 0.5 - comp_blob_density) * 0.4)), 2)
        pipeline.append(PCBColorNormFilter(green_desat=desat, comp_boost=comp_boost))

    if ch_imb > 18.0:
        alpha = round(max(0.85, min(1.15, 128.0 / max(1, mean))), 2)
        pipeline.append(BrightnessContrastFilter(a=alpha, b=0))

    if noise > 14:
        d  = 9 if edge_den < 0.15 else 7
        sc = min(90, int(noise * 5))
        pipeline.append(BilateralDenoiseFilter(d=d, sc=sc, ss=sc))
    elif noise > 8:
        if edge_den < 0.20:
            pipeline.append(MedianBlurFilter(k=3))
        else:
            pipeline.append(BilateralDenoiseFilter(d=5, sc=40, ss=40))
    elif noise > 4 and std < 30:
        pipeline.append(MedianBlurFilter(k=3))

    if dark_f > 0.55 or mean < 55:
        target_mean = 128.0
        gamma = round(min(2.5, max(1.2, np.log(target_mean / 255.0) / np.log(max(1, mean) / 255.0))), 2)
        pipeline.append(GammaFilter(g=gamma))
    elif dark_f > 0.38 or mean < 90:
        gamma = round(min(1.8, max(1.1, 1.0 + (90.0 - mean) / 200.0)), 2)
        pipeline.append(GammaFilter(g=gamma))
    elif brt_f > 0.50 or mean > 210:
        alpha = round(max(0.65, min(0.90, 180.0 / max(1, mean))), 2)
        beta  = int(max(-30, -(mean - 180) * 0.3))
        pipeline.append(BrightnessContrastFilter(a=alpha, b=beta))
    elif brt_f > 0.35 or mean > 185:
        pipeline.append(BrightnessContrastFilter(a=0.88, b=-8))

    if std < 25 or local_std < 18:
        clip = round(min(8.0, max(3.5, 80.0 / max(1, (std + local_std) / 2))), 1)
        tile = 6 if std < 15 else 8
        pipeline.append(CLAHEFilter(clip=clip, tile=tile))
    elif std < 48 or local_std < 35:
        clip_base = round(min(4.0, max(1.5, 50.0 / max(1, std))), 1)
        clip = round(clip_base * (1.0 - max(0.0, (entropy - 6.5) / 10.0)), 1)
        clip = max(1.2, clip)
        pipeline.append(CLAHEFilter(clip=clip, tile=8))
    elif mid_f < 0.45 and entropy < 5.5:
        pipeline.append(CLAHEFilter(clip=2.0, tile=8))

    if lap < 60:
        sigma  = round(min(3.0, max(1.5, 150.0 / max(1, lap ** 0.5))), 1)
        amount = round(min(2.2, max(1.4, 200.0 / max(1, lap))), 1)
        pipeline.append(UnsharpMaskFilter(s=sigma, a=amount))
    elif lap < 150:
        sigma  = round(max(1.0, min(2.0, 80.0 / max(1, lap ** 0.5))), 1)
        amount = round(max(1.0, min(1.6, 120.0 / max(1, lap))), 1)
        pipeline.append(UnsharpMaskFilter(s=sigma, a=amount))
    elif lap < 350 and edge_den > 0.05:
        amt = round(min(1.0, max(0.4, 1.0 - lap / 600.0)), 2)
        pipeline.append(SharpenFilter(a=amt))

    if noise > 12 and lap < 120 and edge_den < 0.12:
        k  = 3 if noise < 18 else 5
        it = 1
        pipeline.append(MorphCloseFilter(k=k, it=it))

    if not pipeline:
        pipeline.append(CLAHEFilter(clip=2.0, tile=8))
        pipeline.append(SharpenFilter(a=0.6))

    return pipeline


# ── AutoCalibrateWorker ──────────────────────────────────────────────────────

class AutoCalibrateWorker(QThread):
    """Closed-loop pipeline optimiser — two-phase model-in-the-loop search."""
    done = Signal(object, str)

    _CONF_COARSE   = 0.15
    _CONF_FINE     = 0.22
    _IMGSZ_COARSE  = 416
    _IMGSZ_FINE    = 640
    _GRID           = 6

    def __init__(self, frame, model_path: str = ""):
        super().__init__()
        self._frame      = frame.copy() if frame is not None else None
        self._model_path = model_path or ""

    def _score(self, model, pipeline, frame, conf, imgsz):
        if not (HAS_YOLO and HAS_CV2 and HAS_NP):
            return 0.0
        try:
            proc  = apply_filters(frame, pipeline) if pipeline else frame
            res   = safe_predict(model, proc, conf=conf, imgsz=imgsz, verbose=False)[0]
            boxes = res.boxes
            n     = len(boxes)
            if n == 0:
                return 0.0
            h, w  = frame.shape[:2]
            confs_np   = boxes.conf.cpu().numpy()
            classes_np = boxes.cls.cpu().numpy().astype(int)
            xyxy_np    = boxes.xyxy.cpu().numpy()
            mean_conf = float(confs_np.mean())
            g = self._GRID
            cx_idx = np.clip((((xyxy_np[:, 0] + xyxy_np[:, 2]) * 0.5) / max(1, w) * g)
                              .astype(int), 0, g - 1)
            cy_idx = np.clip((((xyxy_np[:, 1] + xyxy_np[:, 3]) * 0.5) / max(1, h) * g)
                              .astype(int), 0, g - 1)
            coverage = float(len(set(zip(cx_idx.tolist(), cy_idx.tolist())))) / (g * g)
            quality = float((confs_np > 0.50).sum()) / n
            n_known   = len(model.names) if hasattr(model, "names") else 25
            diversity = float(len(np.unique(classes_np))) / max(1, n_known)
            hi_mask = confs_np > 0.65
            hi_count = int(hi_mask.sum())
            hi_yield = (hi_count * float(confs_np[hi_mask].mean())) if hi_count else 0.0
            return (n * mean_conf
                    * (1.0 + 0.40 * coverage)
                    * max(0.01, quality)
                    * (1.0 + 0.15 * diversity)
                    + 0.50 * hi_yield)
        except Exception:
            return 0.0

    def _candidates(self, metrics):
        base   = _build_auto_pipeline(metrics)
        gbf    = metrics.get("green_board_frac", 0.0)
        mean   = metrics.get("mean", 128.0)
        noise  = metrics.get("noise", 5.0)
        desat  = round(min(0.85, max(0.45, max(gbf, 0.18) * 1.25)), 2)
        try:
            gamma_lift = round(min(2.4, max(1.1,
                math.log(128.0/255.0) / math.log(max(1, mean)/255.0)
            )), 2) if mean < 110 else 1.0
        except (ValueError, ZeroDivisionError):
            gamma_lift = 1.0
        return [
            ("Heuristic", base),
            ("PCBNorm+Heuristic",
             [PCBColorNormFilter(green_desat=desat, comp_boost=1.35)] + list(base)),
            ("PCBNorm-Strong+CLAHE+USM",
             [PCBColorNormFilter(green_desat=0.80, comp_boost=1.55),
              CLAHEFilter(clip=4.5, tile=6),
              UnsharpMaskFilter(s=1.6, a=1.35)]),
            ("Denoise+CLAHE+Sharpen",
             [BilateralDenoiseFilter(d=9, sc=60, ss=60),
              CLAHEFilter(clip=3.0, tile=8),
              SharpenFilter(a=0.90)]),
            ("GammaLift+PCBNorm+CLAHE",
             ([GammaFilter(g=gamma_lift)] if gamma_lift > 1.05 else [])
             + [PCBColorNormFilter(green_desat=0.65, comp_boost=1.30),
                CLAHEFilter(clip=3.0, tile=8),
                SharpenFilter(a=0.70)]),
            ("FullChain",
             [BrightnessContrastFilter(a=1.10, b=5),
              BilateralDenoiseFilter(d=7, sc=45, ss=45),
              CLAHEFilter(clip=3.0, tile=8),
              SharpenFilter(a=0.80)]),
            ("PCBNorm-Mild+CLAHE",
             [PCBColorNormFilter(green_desat=0.45, comp_boost=1.15),
              CLAHEFilter(clip=2.5, tile=8),
              SharpenFilter(a=0.55)]),
            ("Raw", []),
        ]

    @staticmethod
    def _micro_variants(pipeline):
        clahe_caches = []
        for idx, f in enumerate(pipeline):
            if isinstance(f, CLAHEFilter) and f._clahe is not None:
                clahe_caches.append((idx, f._clahe))
                f._clahe = None
        variants = []
        try:
            for fi, f in enumerate(pipeline):
                if isinstance(f, CLAHEFilter):
                    for scale in (0.80, 1.20):
                        new_clip = round(max(0.5, f.params["clip_limit"] * scale), 1)
                        vp = copy.deepcopy(pipeline)
                        vp[fi].params["clip_limit"] = new_clip
                        variants.append(vp)
                if isinstance(f, PCBColorNormFilter):
                    for scale in (0.80, 1.20):
                        new_ds = round(max(0.1, min(0.95,
                                            f.params["green_desat"] * scale)), 2)
                        vp = copy.deepcopy(pipeline)
                        vp[fi].params["green_desat"] = new_ds
                        variants.append(vp)
        finally:
            for idx, cache_obj in clahe_caches:
                pipeline[idx]._clahe = cache_obj
        return variants

    @staticmethod
    def _desc(metrics, winner_label, winner_score, model_used):
        parts = []
        noise = metrics.get("noise", 0)
        mean  = metrics.get("mean", 128)
        std   = metrics.get("std", 50)
        lap   = metrics.get("lap", 200)
        chimb = metrics.get("ch_imbalance", 0.0)
        edged = metrics.get("edge_density", 0.1)
        gbf   = metrics.get("green_board_frac", 0.0)
        if noise > 14:   parts.append(f"heavy noise ({noise:.1f})")
        elif noise > 8:  parts.append(f"moderate noise ({noise:.1f})")
        if mean < 90:    parts.append(f"underexposed ({mean:.0f})")
        elif mean > 185: parts.append(f"overexposed ({mean:.0f})")
        if std < 48:     parts.append(f"low contrast (σ={std:.1f})")
        if lap < 150:    parts.append(f"soft (∇²={lap:.0f})")
        if chimb > 18:   parts.append(f"colour cast ({chimb:.1f})")
        if gbf > 0.15:   parts.append(f"green board ({gbf:.0%})")
        if edged < 0.04: parts.append("sparse edges")
        img_line = "Detected: " + (", ".join(parts) if parts else "good image quality")
        if model_used:
            return f"{img_line}\nWinner: {winner_label} (score={winner_score:.3f})"
        return f"{img_line}\n(heuristic — no model)"

    def run(self):
        try:
            if self._frame is None:
                self.done.emit(None, "No frame loaded.")
                return

            metrics  = _analyse_image(self._frame)
            fallback = _build_auto_pipeline(metrics)

            if not (HAS_YOLO and self._model_path
                    and os.path.exists(self._model_path)):
                self.done.emit(
                    fallback or [CLAHEFilter(clip=2.0), SharpenFilter(a=0.6)],
                    self._desc(metrics, "Heuristic", 0.0, False)
                )
                return

            try:
                model = _YOLO(self._model_path)
                if HAS_NP:
                    safe_predict(model, np.zeros((320, 320, 3), dtype=np.uint8),
                                  verbose=False)
            except Exception as exc:
                self.done.emit(
                    None,
                    self._desc(metrics, "Heuristic", 0.0, False)
                    + f"\n(model load error: {exc})"
                )
                return

            candidates = self._candidates(metrics)
            ranked = []
            for label, pipe in candidates:
                s = self._score(model, pipe, self._frame,
                                self._CONF_COARSE, self._IMGSZ_COARSE)
                ranked.append((s, label, pipe))
            ranked.sort(key=lambda x: x[0], reverse=True)

            best_score    = -1.0
            best_pipeline = ranked[0][2]
            best_label    = ranked[0][1]

            for _, lbl, pipe in ranked[:2]:
                s = self._score(model, pipe, self._frame,
                                self._CONF_FINE, self._IMGSZ_FINE)
                if s > best_score:
                    best_score, best_pipeline, best_label = s, pipe, lbl

                for vp in self._micro_variants(pipe):
                    sv = self._score(model, vp, self._frame,
                                    self._CONF_FINE, self._IMGSZ_FINE)
                    if sv > best_score:
                        best_score    = sv
                        best_pipeline = vp
                        best_label    = f"{lbl}★"

            if best_pipeline is None or (best_score == 0.0 and not ranked):
                best_pipeline = fallback

            self.done.emit(
                best_pipeline,
                self._desc(metrics, best_label, best_score, True)
            )

        except Exception as e:
            self.done.emit(None, f"Calibration error: {e}")


# ── run_roi ──────────────────────────────────────────────────────────────────

def run_roi(img,roi,model=None):
    x,y,rw,rh=roi.rect; ih,iw=img.shape[:2]
    x1,y1=max(0,x),max(0,y); x2,y2=min(iw,x+rw),min(ih,y+rh)
    if x2<=x1 or y2<=y1: return {"passed":False,"info":"bad crop"}
    crop=img[y1:y2,x1:x2]
    if roi.zone_type=="yolo":
        if model is None: return {"passed":False,"info":"no model"}
        try: res=safe_predict(model, crop, conf=0.25, iou=0.35, imgsz=640, verbose=False, device='cpu')[0]; n=len(res.boxes); return {"passed":n>0,"info":f"{n} obj"}

        except Exception as e: return {"passed":False,"info":str(e)[:20]}
    elif roi.zone_type=="ocr":
        if not HAS_OCR: return {"passed":False,"info":"no pytesseract"}
        try: t=pytesseract.image_to_string(crop,config="--psm 7").strip(); return {"passed":bool(t),"info":t[:20] or "empty"}
        except Exception as e: return {"passed":False,"info":str(e)[:20]}
    elif roi.zone_type=="barcode":
        if not HAS_ZBAR: return {"passed":False,"info":"no pyzbar"}
        try:
            codes=pyzbar.decode(crop)
            if codes: return {"passed":True,"info":f"{codes[0].type}:{codes[0].data.decode()[:12]}"}
            return {"passed":False,"info":"no barcode"}
        except Exception as e: return {"passed":False,"info":str(e)[:20]}
    return {"passed":False,"info":"unknown type"}
