import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_MODULE = 'tflite_runtime'
    logger.info("Using tflite_runtime module")
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    TFLITE_MODULE = 'tensorflow'
    logger.info("Using tensorflow module")
    
from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import math
from ffem_embedder import FFEMTfliteEmbedder
from collections import namedtuple

import mediapipe as mp

logging.getLogger("ultralytics").setLevel(logging.WARNING)

# — appropriate position/size settings —
BAR_WIDTH  =  300    # full bar length (pixels)
BAR_HEIGHT =  10    # bar thickness
MARGIN     =  20    # vertical margin between bars
X0         =  640    # bar start X coordinate
Y0         =  50    # first bar start Y coordinate


def init_yoloface(model_path: str = 'yolov8n-face.pt', device: str = 'cpu') -> YOLO:
    """
    Initialize and return a YOLOv8-based face detector model.

    Args:
        model_path: Path to the yolov8n-face weights file.
        device: 'cpu' or 'gpu' for inference.

    Returns:
        Initialized YOLO model.
    """
    # Load model with specified device
    model = YOLO(model_path)
    model.to(device)
    return model


def detect_yolofaces1(
    img: np.ndarray,
    model: YOLO,
    conf_threshold: float = 0.2,
    max_detections: int = 10
) -> list:
    """
    Run face detection on an image and parse bounding box results.

    Args:
        img: Input image as a NumPy array (BGR).
        model: Initialized YOLO face detection model.
        conf_threshold: Minimum confidence to keep detection.
        max_detections: Maximum number of faces to return.

    Returns:
        List of detections, each a dict with keys:
        - 'class_id': int (always 0 for face)
        - 'confidence': float
        - 'box': [x1, y1, x2, y2] in pixel coordinates
    """
    # Perform inference
    results = model.predict(
        source=img,
        conf=conf_threshold,
        max_det=max_detections
    )
    dets = []
    if len(results) > 0:
        res = results[0]
        # xyxy boxes, confidence, class
        boxes = res.boxes.xyxy.cpu().numpy()  # shape (N,4)
        confs = res.boxes.conf.cpu().numpy()  # shape (N,)
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)  # shape (N,)

        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, cls_ids):
            dets.append({
                'class_id': int(cls),
                'confidence': float(conf),
                'box': [float(x1), float(y1), float(x2), float(y2)]
            })
    return dets


def detect_yolofaces(
    img: np.ndarray,
    model: YOLO,
    conf_threshold: float = 0.2,
    max_detections: int = 10,
    input_size: int = 640
) -> list:
    """
    Run face detection with manual letterbox preprocessing and rescale detected boxes back.

    Args:
        img: Original BGR image as numpy array.
        model: YOLOv8 face detection model.
        conf_threshold: Minimum confidence for detection.
        max_detections: Max number of faces to detect.
        input_size: Square size to letterbox image to (e.g., 640).

    Returns:
        List of detections: dicts with 'class_id', 'confidence', and 'box' [x1,y1,x2,y2] in original image coords.
    """
    # 1) Letterbox preprocess
    h0, w0 = img.shape[:2]
    img_lb, scale, pad_w, pad_h = letterbox(img, (input_size, input_size))
    # 2) Inference on letterboxed image
    results = model.predict(
        source=img_lb,
        conf=conf_threshold,
        max_det=max_detections,
        imgsz=input_size
    )
    detections = []
    if len(results) > 0:
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy()  # (N,4) letterboxed coords
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, cls_ids):
            # 3) Map back to original image
            x1o = (x1 - pad_w) / scale
            y1o = (y1 - pad_h) / scale
            x2o = (x2 - pad_w) / scale
            y2o = (y2 - pad_h) / scale
            # Clamp
            x1o, y1o = max(0, x1o), max(0, y1o)
            x2o, y2o = min(w0, x2o), min(h0, y2o)
            detections.append({
                'class_id': int(cls),
                'confidence': float(conf),
                'box': [x1o, y1o, x2o, y2o]
            })
    return detections



# central warning message function
def draw_warning(frame, text):
  h, w = frame.shape[:2]
  font = cv2.FONT_HERSHEY_SIMPLEX
  scale, thk = 1.0, 2
  (tw, th), _ = cv2.getTextSize(text, font, scale, thk)
  # background rectangle
  x = (w - tw) // 2 - 10
  y = (h - th) // 2 - 10
  cv2.rectangle(frame,
                (x,        y),
                (x + tw+20, y + th+20),
                (255,255,255), -1)
  # text
  cv2.putText(frame, text,
              (x+10, y+th+5),
              font, scale, (0,0,0), thk)

def create_interpreter(model_path, gpu_delegate_path=None):
    """
    Create TFLite Interpreter, fallback to CPU if GPU is not available.
    
    Args:
        model_path (str): TFLite model file path
        gpu_delegate_path (str, optional): GPU delegate library path
    
    Returns:
        Interpreter: TFLite Interpreter object
    """
    if gpu_delegate_path:
        try:
            logger.info("Attempting to load model with GPU delegate")
            interpreter = tflite.Interpreter(
                model_path=model_path,
                experimental_delegates=[tflite.load_delegate(gpu_delegate_path)]
            )
            logger.info("Successfully loaded model with GPU delegate")
            return interpreter
        except Exception as e:
            logger.warning(f"Failed to load GPU delegate: {e}")
            logger.info("Falling back to CPU")

    # CPU fallback
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        logger.info("Successfully loaded model with CPU")
        return interpreter
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# ─────────── Mediapipe FaceDetection ────────────
mp_fd = mp.solutions.face_detection
mp_d  = mp_fd.FaceDetection(model_selection=1,
                             min_detection_confidence=0.5)

# ─────────── SimpleFaceDB ────────────
class SimpleFaceDB:
  def __init__(self, metric="cosine"):
    assert metric in ("cosine","euclid")
    self.db = {}
    self.metric = metric
  def add(self, name, emb):
    e = emb/np.linalg.norm(emb)
    self.db[name]=e
  def search(self, emb, thr):
    e = emb/np.linalg.norm(emb)
    best = (None, float("inf"), 0.0)
    for name, ref in self.db.items():
        if self.metric=="cosine":
            dot  = np.clip(np.dot(ref,e),-1,1)
            dist = math.degrees(math.acos(dot))
            pct  = (1 - dist/180)*100
        else:
            dist = np.linalg.norm(ref-e)
            pct  = (1 - dist/2)*100
        if dist<best[1]:
            best=(name,dist,pct)
    
    nm,d,p = best
    if d<=thr:
        return nm,d,p
    else:
        return f"@{nm}@", d,p

# ─────────── letterbox + normalize ────────────
def letterbox(img, new_size):
    h0,w0 = img.shape[:2]
    w1,h1 = new_size
    scale = min(w1/w0, h1/h0)
    nw,nh = int(w0*scale), int(h0*scale)
    im1 = cv2.resize(img, (nw,nh))
    top = (h1-nh)//2
    left= (w1-nw)//2
    canvas = np.zeros((h1,w1,3), dtype=img.dtype)
    canvas[top:top+nh,left:left+nw] = im1
    return canvas, scale, left, top

# ─────────── network output → pixel coordinate restoration ────────────
def decode_landmark(rx, ry, x1, y1, scale, dx, dy, input_size):
    real_w = input_size  - 2*dx
    real_h = input_size  - 2*dy
    # 1) letterbox space coordinates
    x_l = rx * real_w + dx
    y_l = ry * real_h + dy
    # 2) remove pad → crop space
    x_c = (x_l - dx) / scale
    y_c = (y_l - dy) / scale
    # 3) crop → frame
    return x1 + x_c, y1 + y_c

# ─────────── preprocess_attr ────────────
def preprocess_attr(img, input_size):
    lb, scale, dx, dy = letterbox(img, (input_size,input_size))
    
    lb = lb.astype(np.float32)/255.0
    mean = np.array([0.485,0.456,0.406],dtype=np.float32)
    std  = np.array([0.229,0.224,0.225],dtype=np.float32)
    lb = (lb - mean)/std
    # HWC→batch
    inp = lb[None,...].astype(np.float32)
    return inp, scale, dx, dy

# ─────────── pose & align ────────────
landmark_names = ["eyeL", "eyeR", "nose", "mouthL", "mouthC", "mouthR", "earL", "earR"]
# eyeL, eyeR, nose, mouth, earR, earL
lm_get_idx = {
  "eyeL":0,
  "eyeR":1,
  "nose":2,
  "mouthL":3,
  "mouthC":4,
  "mouthR":5,
  "earL":6,
  "earR":7
  }
lm_idxs = (
  lm_get_idx["nose"],
  lm_get_idx["eyeL"],
  lm_get_idx["eyeR"],
  lm_get_idx["mouthC"],
  lm_get_idx["earL"],
  lm_get_idx["earR"]
  )

_MODEL_POINTS = np.array([
    (  0.0,    0.0,    0.0),   # nose
    (-14.0,  -20.0,  -8.0),   # left eye
    ( 14.0,  -20.0,  -8.0),   # right eye
    (   0.0,   15.0,  -8.0),   # mouth center
    (-30.0,    -15.0,  -40.0),   # left cheek/ear
    ( 30.0,    -15.0,  -40.0),   # right cheek/ear
],dtype=np.float64)


def estimate_pose(frame, img_pts):
    h, w = frame.shape[:2]
    cam = np.array([[w, 0, w/2],
                    [0, w, h/2],
                    [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4,1))
    ok, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS,
        np.array(img_pts, dtype=np.float64),
        cam, dist,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not ok:
        return 0.0, 0.0, 0.0

    R, _ = cv2.Rodrigues(rvec)

    # pitch: x-axis rotation → asin(-R[2,0])
    sinp = R[2, 1]
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.degrees(math.asin(sinp))

    # yaw: y-axis rotation (head left/right) → keep as is
    yaw = math.degrees(math.atan2(-R[2,0], math.hypot(R[2,1], R[2,2])))

    # roll: z-axis rotation (head tilt) → keep as is
    roll = math.degrees(math.atan2(R[1,0], R[0,0]))

    return yaw, pitch, roll

def align_face(img, kps, roll=0, size=(112,112)):
    src = np.array(kps, dtype=np.float32)
    dst = np.array([
        [size[0]*0.3, size[1]*0.4],
        [size[0]*0.7, size[1]*0.4],
        [size[0]*0.5, size[1]*0.6],
    ], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, size)

def rect_from_landmark(lndmk: np.ndarray) -> np.ndarray:
    """
    Calculate BBox based on 5-point landmark, same as Trillion Pairs dataset.
    lndmk: [x0,y0,x1,y1,...,x4,y4] (5 points)
    returns: [min_x, min_y, max_x, max_y] as int32
    """
    xs = lndmk[0::2]
    ys = lndmk[1::2]

    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())

    # add margin (1/1.5)
    width = (max_x - min_x) / 1.5
    height = (max_y - min_y) / 1.5
    min_x -= width
    max_x += width
    min_y -= height
    max_y += height

    # return integer BBox
    return np.array([int(min_x), int(min_y), int(max_x), int(max_y)], dtype=np.int32)


def align_face_ffem_bbox(
    img: np.ndarray,
    pts5: list,
    roll: float,
    output_size: tuple = (112, 112)
) -> np.ndarray:
    """
    Calculate BBox based on Trillion Pairs method + apply roll correction.

    Args:
        img: BGR input image
        pts5: list of 5-point landmarks in order [eyeL, eyeR, nose, mouthL, mouthR]
        roll: face tilt angle (degrees), positive value means clockwise
        output_size: (width, height) output size

    Returns:
        (height, width, 3) rotated and cropped face
    """
    # 1) original landmark array
    # pts5 order: eyeL, eyeR, nose, mouthL, mouthR
    lndmk = np.array([
        pts5[0][0], pts5[0][1],
        pts5[1][0], pts5[1][1],
        pts5[2][0], pts5[2][1],
        pts5[3][0], pts5[3][1],
        pts5[4][0], pts5[4][1]
    ], dtype=np.float32)

    # 2) set nose coordinate as rotation center instead of image center
    nose_x, nose_y = pts5[2]
    M_rot = cv2.getRotationMatrix2D((nose_x, nose_y), -roll, 1.0)

    # 3) calculate rotated image and landmark coordinates
    h, w = img.shape[:2]
    rotated = cv2.warpAffine(img, M_rot, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    # convert lndmk coordinates
    ones = np.ones((5,1), dtype=np.float32)
    pts = lndmk.reshape(5,2)
    pts_hom = np.hstack([pts, ones])  # (5,3)
    rotated_pts = (M_rot @ pts_hom.T).T  # (5,2)

    # 4) calculate BBox based on rotated landmarks
    bbox = rect_from_landmark(rotated_pts.flatten())
    x1, y1, x2, y2 = bbox
    # boundary clipping
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # 5) crop & resize
    if y2-y1 < 5 or x2-x1 < 5:
        return None 
    crop = rotated[y1:y2, x1:x2]
    aligned = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
    return aligned
