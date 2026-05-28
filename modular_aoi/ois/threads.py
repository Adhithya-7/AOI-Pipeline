"""
threads.py — Background QThread workers.

CameraThread, InferenceThread, ThumbThread, TrainingThread, AugThread, AutoLabelThread.
"""
import os, sys, time, glob, shutil, gc
from collections import deque

if __package__ in (None, ""):
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    __package__ = "ois"

from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker, Qt
from PySide6.QtGui import QImage, QPixmap

from .utils import (
    HAS_CV2, HAS_NP, HAS_YOLO,
    _cam_backend, find_best_pt, DATA_ROOT, load_optimized_yolo, safe_predict
)

if HAS_CV2:
    import cv2
if HAS_NP:
    import numpy as np
if HAS_YOLO:
    from ultralytics import YOLO as _YOLO


from .filters import apply_filters

class CameraThread(QThread):
    frame_ready=Signal(QImage); error=Signal(str)
    DISP_W=960; DISP_FPS=30; INFER_FPS=15
    def __init__(self,idx=0):
        super().__init__(); self._idx=idx; self._stop=False
        self._infer_frame=None; self._small_bgr=None; self._mx=QMutex()
        self._filters = []
    def set_filters(self, filters):
        with QMutexLocker(self._mx):
            self._filters = list(filters)
    def get_infer_frame(self):
        with QMutexLocker(self._mx): f=self._infer_frame; self._infer_frame=None; return f
    def get_small_bgr(self):
        with QMutexLocker(self._mx): return self._small_bgr
    def run(self):
        if not HAS_CV2: self.error.emit("OpenCV not installed"); return
        cap=cv2.VideoCapture(self._idx, _cam_backend())
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        if not cap.isOpened(): self.error.emit(f"Camera {self._idx} unavailable"); return
        disp_int=1.0/self.DISP_FPS; inf_int=1.0/self.INFER_FPS
        last_disp=last_inf=0.0
        while not self._stop:
            ret,frame=cap.read()
            if not ret: time.sleep(0.02); continue
            now=time.time()
            if now-last_inf>=inf_int:
                last_inf=now
                with QMutexLocker(self._mx): self._infer_frame=frame
            if now-last_disp>=disp_int:
                last_disp=now
                h,w=frame.shape[:2]; scale=self.DISP_W/w
                small=cv2.resize(frame,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_NEAREST)
                with QMutexLocker(self._mx):
                    self._small_bgr=small
                    if self._filters:
                        small = apply_filters(small, self._filters)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                nw, nh = rgb.shape[1], rgb.shape[0]
                # bytes() creates a Python-owned copy of the buffer so QImage
                # stays valid after `rgb` goes out of scope — no extra .copy() needed.
                qimg = QImage(rgb.tobytes(), nw, nh, nw * 3,
                              QImage.Format.Format_RGB888)
                self.frame_ready.emit(qimg)
            time.sleep(0.004)
        cap.release()
    def stop(self): self._stop=True; self.wait(2000)


class InferenceThread(QThread):
    result_ready=Signal(list,float); log=Signal(str)
    def __init__(self,cfg,cam,filters=None):
        super().__init__(); self._cfg=cfg; self._cam=cam; self._stop=False; self._model=None
        self._filters = filters or []
        self._fps_times: deque = deque(maxlen=20)  # ~2 s window at 8 fps
        self._debug_counter = 0

    def set_filters(self, filters):
        self._filters = list(filters)

    def run(self):
        if not self._load_model(): return
        TARGET=1.0/30.0   # 30fps inference target
        while not self._stop:
            t0=time.time()
            frame=self._cam.get_infer_frame()
            if frame is None: time.sleep(0.020); continue
            
            # Apply filter pipeline (matching AOITab behavior)
            if self._filters:
                proc_img = apply_filters(frame, self._filters)
            else:
                proc_img = frame

            dets,lat=self._infer(proc_img)
            # Track actual FPS
            now=time.time(); self._fps_times.append(now)
            self.result_ready.emit(dets,lat)
            time.sleep(max(0.008,TARGET-(time.time()-t0)))

    def _load_model(self):
        mp = self._cfg.get("model_path")
        if not mp:
            self.log.emit("[AI] Model path is not set in configuration")
            return False
            
        # Resolve relative project paths
        if not os.path.isabs(mp) and self._cfg._path:
            full_mp = os.path.join(self._cfg._path, mp)
            if os.path.exists(full_mp):
                mp = full_mp
        
        # Definitive Purge: If we are still pointing to OpenVINO, force back to PT
        if mp.endswith("_openvino_model"):
            pt = mp.replace("_openvino_model", ".pt")
            if os.path.exists(pt): mp = pt
            
        if not os.path.exists(mp):

            self.log.emit(f"[AI] Model file not found on disk: {mp}")
            return False

            
        if not HAS_YOLO:
            self.log.emit("[AI] Package 'ultralytics' not installed - cannot run detection")
            return False
            
        try:
            self.log.emit(f"[AI] Loading {os.path.basename(mp)}...")
            self._model = load_optimized_yolo(mp)
            
            if self._model is None:
                self.log.emit(f"[AI] Failed to initialize model: {os.path.basename(mp)}")
                return False
                
            self.log.emit(f"[AI] OK {os.path.basename(mp)} [Ready]")
            if HAS_NP:
                safe_predict(self._model, np.zeros((320,320,3),dtype=np.uint8), verbose=False, device='cpu')
            return True
        except Exception as e:
            self.log.emit(f"[AI] Load error: {e}")
            return False

    def _infer(self, frame):
        t0 = time.time(); h, w = frame.shape[:2]
        conf = self._cfg.get("confidence"); mw = self._cfg.get("max_obj_width")
        mh   = self._cfg.get("max_obj_height"); em = self._cfg.get("edge_margin")
        ds   = CameraThread.DISP_W / max(1, w)   # full-res → display-res scale
        dets = []
        try:
            res   = safe_predict(self._model, frame, conf=conf, iou=0.35, imgsz=640, verbose=False, device='cpu')[0]

            boxes = res.boxes
            n     = len(boxes)
            
            # Diagnostic periodic log
            self._debug_counter += 1
            if self._debug_counter % 30 == 0:
                self.log.emit(f"[AI-Debug] Raw model saw {n} boxes (conf={conf:.2f})")

            if n:
                xyxy_np  = boxes.xyxy.cpu().numpy()
                confs_np = boxes.conf.cpu().numpy()
                cls_np   = boxes.cls.cpu().numpy().astype(int)

                bw = xyxy_np[:, 2] - xyxy_np[:, 0]
                bh = xyxy_np[:, 3] - xyxy_np[:, 1]
                
                # Mask components that are too large (filters out the board itself or noise)
                mask = ((bw / w <= mw) & (bh / h <= mh) &
                        (xyxy_np[:, 0] >= em) & (xyxy_np[:, 1] >= em) &
                        (xyxy_np[:, 2] <= w - em) & (xyxy_np[:, 3] <= h - em))
                
                pass_count = int(mask.sum())
                if n > 0 and pass_count == 0 and self._debug_counter % 30 == 0:
                     self.log.emit(f"[AI-Debug] All {n} boxes were filtered by size/edge constraints")

                for i in mask.nonzero()[0]:

                    x1, y1, x2, y2 = xyxy_np[i]
                    cls = int(cls_np[i])
                    dets.append({
                        "class": cls,
                        "name":  self._model.names.get(cls, "?"),
                        "xyxy":  [int(x1*ds), int(y1*ds), int(x2*ds), int(y2*ds)],
                        "conf":  float(confs_np[i]),
                    })
        except Exception as e:
            self.log.emit(f"[AI] {e}")
        return dets, (time.time() - t0) * 1000
    def stop(self): self._stop=True; self.wait(3000)

class ThumbThread(QThread):
    thumb_done=Signal(int,bytes,int,int); all_done=Signal()
    def __init__(self,images,size=96):
        super().__init__(); self._images=list(images); self._size=size; self._abort=False
    def abort(self):
        self._abort = True
        if not self.wait(2000):   # give it 2 s to exit cleanly
            self.terminate()      # force-kill if still stuck (e.g. slow disk read)
            self.wait(500)        # wait for terminate to land before returning
    def run(self):
        sz=self._size
        for i,path in enumerate(self._images):
            if self._abort: return
            try:
                if HAS_CV2:
                    bgr=cv2.imread(path)
                    if bgr is None: time.sleep(0.060); continue
                    h,w=bgr.shape[:2]; scale=min(sz/w,sz/h)
                    nw,nh=max(1,int(w*scale)),max(1,int(h*scale))
                    small=cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_LINEAR)
                    rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
                    self.thumb_done.emit(i,bytes(rgb.tobytes()),nw,nh)
                else:
                    px=QPixmap(path)
                    if not px.isNull():
                        px=px.scaled(sz,sz,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.FastTransformation)
                        img=px.toImage().convertToFormat(QImage.Format.Format_RGB888)
                        ptr=img.bits(); ptr.setsize(img.sizeInBytes())
                        self.thumb_done.emit(i,bytes(ptr),img.width(),img.height())
            except: pass
            # Yield every 4 ms — lets Qt process thumb_done signals between emits
            # so the event queue never floods. Removing this sleep causes full UI freeze
            # on galleries with 50+ images (hundreds of signals pile up instantly).
            time.sleep(0.004)
        self.all_done.emit()

class TrainingThread(QThread):
    progress=Signal(int,str); finished=Signal(str)
    def __init__(self,base,yaml,params,out,name):
        super().__init__(); self.base=base; self.yaml=yaml; self.params=dict(params); self.out=out; self.name=name
    def run(self):
        if not HAS_YOLO: self.progress.emit(0,"[Train] ultralytics missing"); self.finished.emit(""); return
        try:
            self.progress.emit(1,f"[Train] Loading: {os.path.basename(self.base)}")
            model=_YOLO(self.base); p=self.params; t0=time.time()
            def on_epoch(trainer):
                ep=trainer.epoch+1; tot=trainer.epochs; pct=int(ep/tot*100)
                try:
                    bl=cl=m50=pr=re=0
                    if hasattr(trainer,"tloss") and trainer.tloss is not None:
                        tl=trainer.tloss
                        if hasattr(tl,"__len__") and len(tl)>=2: bl,cl=float(tl[0]),float(tl[1])
                        elif hasattr(tl,"item"): bl=float(tl)
                    if hasattr(trainer,"metrics"):
                        mm=trainer.metrics
                        m50=mm.get("metrics/mAP50(B)",0); pr=mm.get("metrics/precision(B)",0)
                        re=mm.get("metrics/recall(B)",0)
                    el=time.time()-t0; avg=el/max(ep,1); rem=(tot-ep)*avg
                    msg=(f"EP {ep}/{tot}  box={bl:.3f} cls={cl:.3f}  mAP50={m50:.3f} P={pr:.3f} R={re:.3f}  "
                         f"{int(el//60)}m{int(el%60)}s/~{int((el+rem)//60)}m{int((el+rem)%60)}s")
                    self.progress.emit(pct,msg)
                except: self.progress.emit(int(ep/tot*100),f"Epoch {ep}/{tot}")
            model.add_callback("on_train_epoch_end",on_epoch)
            model.train(data=self.yaml,epochs=int(p.get("epochs",50)),patience=int(p.get("patience",25)),
                batch=int(p.get("batch",4)),imgsz=int(p.get("imgsz",640)),workers=int(p.get("workers",1)),
                mosaic=float(p.get("mosaic",1.0)),mixup=float(p.get("mixup",0.15)),
                copy_paste=float(p.get("copy_paste",0.3)),hsv_h=float(p.get("hsv_h",0.015)),
                hsv_s=float(p.get("hsv_s",0.6)),hsv_v=float(p.get("hsv_v",0.4)),
                degrees=float(p.get("degrees",12.0)),translate=float(p.get("translate",0.15)),
                scale=float(p.get("scale",0.6)),shear=float(p.get("shear",2.0)),
                perspective=float(p.get("perspective",0.0005)),erasing=float(p.get("erasing",0.3)),
                project=os.path.abspath(self.out),name=self.name,exist_ok=False)
            best=find_best_pt(self.out,self.name)
            if best: self.progress.emit(100,f"Done -> {best}"); self.finished.emit(best)
            else: self.progress.emit(100,"Complete but best.pt not found"); self.finished.emit("")
        except Exception as e: self.progress.emit(0,f"[Train] ERROR: {e}"); self.finished.emit("")

class AugThread(QThread):
    progress=Signal(int,str); finished=Signal(int)
    def __init__(self,src_img,src_lbl,out_img,out_lbl,opts):
        super().__init__(); self.src_img=src_img; self.src_lbl=src_lbl
        self.out_img=out_img; self.out_lbl=out_lbl; self.opts=opts
    def _rb(self,boxes,m):
        res=[]
        for b in boxes:
            c,cx,cy,w,h=b
            if m=="rot90": res.append([c,1-cy,cx,h,w])
            elif m=="rot180": res.append([c,1-cx,1-cy,w,h])
            elif m=="rot270": res.append([c,cy,1-cx,h,w])
            elif m=="flip_h": res.append([c,1-cx,cy,w,h])
            elif m=="flip_v": res.append([c,cx,1-cy,w,h])
            else: res.append(b)
        return res
    def _save(self,name,img,boxes,suf):
        fn=f"{name}_{suf}.jpg"; cv2.imwrite(os.path.join(self.out_img,fn),img)
        with open(os.path.join(self.out_lbl,f"{name}_{suf}.txt"),"w") as f:
            for b in boxes: f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
    def run(self):
        if not (HAS_CV2 and HAS_NP): self.finished.emit(0); return
        for d in [self.out_img,self.out_lbl]:
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d,exist_ok=True)
        files=(glob.glob(os.path.join(self.src_img,"*.jpg"))+
               glob.glob(os.path.join(self.src_img,"*.png"))+
               glob.glob(os.path.join(self.src_img,"*.jpeg")))
        total=len(files); count=0; gen=0; o=self.opts
        for ip in files:
            base=os.path.splitext(os.path.basename(ip))[0]
            txt=os.path.join(self.src_lbl,base+".txt")
            if not os.path.exists(txt): continue
            img=cv2.imread(ip)
            if img is None: continue
            boxes=[list(map(float,l.strip().split())) for l in open(txt) if l.strip()]
            self._save(base,img,boxes,"orig"); gen+=1
            if o.get("brightness"):
                self._save(base,cv2.convertScaleAbs(img,alpha=1.4),boxes,"bright"); gen+=1
                self._save(base,cv2.convertScaleAbs(img,alpha=0.55),boxes,"dark"); gen+=1
            if o.get("noise"):
                n=np.clip(img.astype(float)+np.random.normal(0,15,img.shape),0,255).astype(np.uint8)
                self._save(base,n,boxes,"noise"); gen+=1
            if o.get("flip_h"): self._save(base,cv2.flip(img,1),self._rb(boxes,"flip_h"),"flip_h"); gen+=1
            if o.get("flip_v"): self._save(base,cv2.flip(img,0),self._rb(boxes,"flip_v"),"flip_v"); gen+=1
            if o.get("rot90"):
                self._save(base,cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE),self._rb(boxes,"rot90"),"rot90"); gen+=1
                self._save(base,cv2.rotate(img,cv2.ROTATE_180),self._rb(boxes,"rot180"),"rot180"); gen+=1
                self._save(base,cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE),self._rb(boxes,"rot270"),"rot270"); gen+=1
            if o.get("blur"): self._save(base,cv2.GaussianBlur(img,(5,5),0),boxes,"blur"); gen+=1
            count+=1; self.progress.emit(int(count/total*100) if total else 100,f"[Aug] {gen} generated")
        self.finished.emit(gen)

class AutoLabelThread(QThread):
    progress=Signal(int,str); finished=Signal(int)
    def __init__(self,model_path,inbox,out,conf):
        super().__init__(); self.model_path=model_path; self.inbox=inbox; self.out=out; self.conf=conf
    def run(self):
        if not os.path.exists(self.inbox): self.finished.emit(0); return
        os.makedirs(self.out,exist_ok=True)
        files=[f for f in os.listdir(self.inbox) if f.lower().endswith((".jpg",".png",".jpeg"))]
        total=len(files); count=0
        
        # Pre-resolve the optimized model path to avoid repeated checks in the loop
        best_ov_dir = None
        if os.path.isdir(self.model_path):
            best_ov_dir = self.model_path
        else:
            _stem = os.path.splitext(self.model_path)[0]
            _ov_can = os.path.join(os.path.dirname(self.model_path), f"{_stem}_openvino_model")
            if os.path.isdir(_ov_can): best_ov_dir = _ov_can
            else: best_ov_dir = self.model_path # Fallback to original .pt

        for i,fn in enumerate(files):
            ip=os.path.join(self.inbox,fn)
            model = None
            try:
                # Atomic Inference: Reloading model per image is slightly slower 
                # but it's the ONLY way to prevent the OpenVINO engine from going "blind" on 8K images.
                try:
                    model = _YOLO(best_ov_dir, task="detect")
                except Exception as me:
                    self.progress.emit(int((i+1)/total*100), f"Err {fn}: Load failed: {me}"); continue

                if HAS_CV2:
                    img_raw = cv2.imread(ip)
                    if img_raw is None: self.progress.emit(int((i+1)/total*100), f"Err {fn}: Read failed"); continue
                    h0, w0 = img_raw.shape[:2]
                    
                    # Secure pre-resize for giant images (e.g. 8K) to 1600px "Safe Zone"
                    if max(h0, w0) > 1600:
                        scale = 1600.0 / max(h0, w0)
                        w, h = int(w0 * scale), int(h0 * scale)
                        img_bgr = cv2.resize(img_raw, (w, h), interpolation=cv2.INTER_LINEAR)
                    else:
                        img_bgr = img_raw; w, h = w0, h0
                    
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
                    # Speed optimization: imgsz=512 is ~30% faster than 640 on CPU while maintaining accuracy 
                    res = safe_predict(model, img, conf=self.conf, iou=0.35, imgsz=512, verbose=False, device='cpu')[0]
                else:
                    res = safe_predict(model, ip, conf=self.conf, iou=0.35, imgsz=640, verbose=False, device='cpu')[0]
                    h, w = res.orig_shape

                raw_boxes = res.boxes
                n_raw = len(raw_boxes)
                labels = []
                for box in raw_boxes:
                    cls=int(box.cls[0]); x1,y1,x2,y2=box.xyxy[0].tolist()
                    labels.append([cls,(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h])
                
                if labels and HAS_NP:
                    confs=[float(b.conf[0]) for b in raw_boxes]
                    order=sorted(range(len(labels)),key=lambda _k:-confs[_k])
                    keep=[]; suppressed=set()
                    for _ni in order:
                        if _ni in suppressed: continue
                        keep.append(_ni)
                        l1=labels[_ni]
                        b1=[l1[1]-l1[3]/2,l1[2]-l1[4]/2,l1[1]+l1[3]/2,l1[2]+l1[4]/2]
                        for _nj in order:
                            if _nj in suppressed or _nj==_ni: continue
                            l2=labels[_nj]
                            b2=[l2[1]-l2[3]/2,l2[2]-l2[4]/2,l2[1]+l2[3]/2,l2[2]+l2[4]/2]
                            ix=max(0,min(b1[2],b2[2])-max(b1[0],b2[0]))
                            iy=max(0,min(b1[3],b2[3])-max(b1[1],b2[1]))
                            inter=ix*iy
                            union=(l1[3]*l1[4])+(l2[3]*l2[4])-inter
                            if union>0 and inter/union>0.40: suppressed.add(_nj)
                    labels=[labels[_ni] for _ni in keep]

                if labels:
                    with open(os.path.join(self.out,os.path.splitext(fn)[0]+".txt"),"w") as f:
                        for l in labels: f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
                    count+=1

                self.progress.emit(int((i+1)/total*100), f"{fn} ({w}x{h}): raw={n_raw} nms={len(labels)}")
                
                # Full Cleanup
                del model, res, raw_boxes, labels
                if HAS_CV2: del img_raw, img_bgr, img
                gc.collect() 
                time.sleep(0.04)   # Proper yield to ensure cleanup
                
            except Exception as e:
                self.progress.emit(int((i+1)/total*100), f"Err {fn}: {e}")
        self.finished.emit(count)
