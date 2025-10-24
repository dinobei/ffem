import json
import numpy as np
import os, glob
import cv2
from pathlib import Path
from ultralytics import YOLO

def rect_from_landmark(lndmk):
    xs = np.zeros((5))
    ys = np.zeros((5))
    for i in range(5):
        xs[i], ys[i] = lndmk[i*2], lndmk[i*2+1]

    min_x = lndmk[np.argmin(xs)*2]
    min_y = lndmk[np.argmin(ys)*2+1]
    max_x = lndmk[np.argmax(xs)*2]
    max_y = lndmk[np.argmax(ys)*2+1]

    width = (max_x - min_x) / 1.5
    height = (max_y - min_y) / 1.5
    min_x -= width; max_x += width
    min_y -= height; max_y += height

    # yaw correction (center movement based on the ratio of left-right eye and nose)
    left_eye  = np.array([lndmk[0], lndmk[1]])
    right_eye = np.array([lndmk[2], lndmk[3]])
    nose      = np.array([lndmk[4], lndmk[5]])
    left_line  = np.linalg.norm(left_eye - nose)
    right_line = np.linalg.norm(right_eye - nose)
    shift = (-(1 - right_line / left_line) if left_line - right_line > 0
             else (1 - left_line / right_line)) * 1.2
    min_x += width * shift
    max_x += width * shift

    return np.array([min_x, min_y, max_x, max_y], dtype=np.int32)

def save_bbox_to_json(lndmk_file, out_file):
    """
    lndmk_file format (space separated):
      <img_path> <label> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> <x5> <y5>
    """
    results = {}
    label_count = -1
    label_book = {}
    image_count = {}
    with open(lndmk_file, 'r') as f:
        for n, line in enumerate(f):
            data = line.strip().split()
            if len(data) < 12:
                continue
            img_path = data[0]
            label_name = data[1]
            lnd_mark = np.array([float(p) for p in data[2:12]])
            bbox = rect_from_landmark(lnd_mark)

            if label_name not in label_book:
                label_count += 1
                label_book[label_name] = label_count
                image_count[label_name] = 1
            else:
                image_count[label_name] += 1
            label = label_book[label_name]

            results[img_path] = {
                'label': label,
                'x1': int(bbox[0]), 'y1': int(bbox[1]),
                'x2': int(bbox[2]), 'y2': int(bbox[3])
            }
            if n % 10000 == 0:
                print(f'{n} images are processed')

    with open(out_file, 'w') as out_f:
        json.dump(results, out_f, indent=2)

def save_bbox_fixedcrop(root_dir, out_file):
    """
    Fixed crop box using LFW-funneled's center alignment assumption.
    250x250 기준 (x1,y1)=(83,92), (x2,y2)=(166,175) → 비율로 환산 후 각 이미지 크기에 적용.
    """
    RX1, RY1, RX2, RY2 = 83/250.0, 92/250.0, 166/250.0, 175/250.0

    # label is the folder name
    persons = sorted([p.name for p in Path(root_dir).iterdir() if p.is_dir()])
    label_book = {name: i for i, name in enumerate(persons)}

    results = {}
    cnt = 0
    for person in persons:
        lbl = label_book[person]
        # supported image extensions
        image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        
        for ext in image_extensions:
            for img_path in glob.glob(os.path.join(root_dir, person, ext)):
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                x1 = max(0, min(int(RX1 * w), w-1))
                y1 = max(0, min(int(RY1 * h), h-1))
                x2 = max(0, min(int(RX2 * w), w-1))
                y2 = max(0, min(int(RY2 * h), h-1))
                results[img_path] = {'label': lbl, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                cnt += 1
                if cnt % 10000 == 0:
                    print(f'{cnt} images are processed')

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

def save_bbox_detect(root_dir, out_file, cascade_path=None):
    """
    Simple detection based on Haar Cascade (works with only opencv-python installed via pip).
    """
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    persons = sorted([p.name for p in Path(root_dir).iterdir() if p.is_dir()])
    label_book = {name: i for i, name in enumerate(persons)}

    results = {}
    cnt = 0
    for person in persons:
        lbl = label_book[person]
        # supported image extensions
        image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        
        for ext in image_extensions:
            for img_path in glob.glob(os.path.join(root_dir, person, ext)):
                img = cv2.imread(img_path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                      minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
                if len(faces) == 0:
                    # fallback to center approximation or skip
                    h, w = img.shape[:2]
                    x1 = int(0.33*w); y1 = int(0.36*h); x2 = int(0.66*w); y2 = int(0.70*h)
                else:
                    # select the largest box
                    x,y,w_,h_ = max(faces, key=lambda b: b[2]*b[3])
                    x1,y1,x2,y2 = x, y, x+w_, y+h_

                results[img_path] = {'label': lbl, 'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
                cnt += 1
                if cnt % 10000 == 0:
                    print(f'{cnt} images are processed')

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

def save_bbox_yolov8face(root_dir, out_file, model_path=None, conf_threshold=0.1):
    """
    face detection using YOLOv8-face model.
    if model_path is None, automatically download 'yolov8n-face.pt' model.
    
    Args:
        root_dir: LFW dataset root directory
        out_file: Output JSON file path
        model_path: YOLOv8-face model path (if None, automatically download 'yolov8n-face.pt')
        conf_threshold: Confidence threshold (default: 0.5)
    """
    if YOLO is None:
        raise ImportError("ultralytics is not installed. Install it via 'pip install ultralytics'.")
    
    # load YOLOv8-face model
    if model_path is None:
        print("🔄 Downloading YOLOv8-face model...")
        model = YOLO('yolov8n-face.pt')  # automatically download
    else:
        if not os.path.exists(model_path):
            print(f"⚠️ Model file does not exist: {model_path}")
            print("🔄 Downloading YOLOv8-face model...")
            model = YOLO('yolov8n-face.pt')  # automatically download
        else:
            model = YOLO(model_path)
    
    print(f"✅ YOLOv8-face model loaded")
    
    # label is the folder name
    persons = sorted([p.name for p in Path(root_dir).iterdir() if p.is_dir()])
    label_book = {name: i for i, name in enumerate(persons)}
    
    results = {}
    cnt = 0
    
    # supported image extensions
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    
    # calculate total number of images
    total_images = 0
    for person in persons:
        for ext in image_extensions:
            total_images += len(glob.glob(os.path.join(root_dir, person, ext)))
    
    print(f"📊 Total {len(persons)} persons, {total_images} images processing started...")
    
    for person in persons:
        lbl = label_book[person]
        
        # collect all image files with all extensions
        person_images = []
        for ext in image_extensions:
            person_images.extend(glob.glob(os.path.join(root_dir, person, ext)))
        
        for img_path in person_images:
            try:
                # check image size
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Unable to read image: {img_path}")
                    continue
                
                h, w = img.shape[:2]
                
                # try different input sizes based on image size
                img_sizes = [640, 320, 160, 128]
                results_detection = None
                used_size = None
                
                for imgsz in img_sizes:
                    # skip if image is smaller than input size
                    if h < imgsz and w < imgsz:
                        continue
                    
                    results_detection = model(img_path, conf=conf_threshold, verbose=False, imgsz=imgsz)
                    
                    if len(results_detection) > 0 and len(results_detection[0].boxes) > 0:
                        used_size = imgsz
                        break
                
                # debug: print detection results
                if results_detection and len(results_detection) > 0 and len(results_detection[0].boxes) > 0:
                    print(f"✅ {img_path}: {len(results_detection[0].boxes)} faces detected (input size: {used_size})")
                else:
                    print(f"❌ {img_path}: face detection failed (tried sizes: {img_sizes})")
                
                if len(results_detection) > 0 and len(results_detection[0].boxes) > 0:
                    # select the face with the highest confidence
                    boxes = results_detection[0].boxes
                    confidences = boxes.conf.cpu().numpy()
                    best_idx = np.argmax(confidences)
                    
                    # extract bbox coordinates (x1, y1, x2, y2 format)
                    bbox = boxes.xyxy[best_idx].cpu().numpy()
                    x1, y1, x2, y2 = bbox
                    
                    # debug: print detected faces and highest confidence
                    if cnt % 100 == 0:
                        print(f"🔍 {img_path}: {len(boxes)} faces detected, highest confidence: {confidences[best_idx]:.3f}")
                    
                    # clip bbox to image boundaries (using already read image)
                    x1 = max(0, min(int(x1), w-1))
                    y1 = max(0, min(int(y1), h-1))
                    x2 = max(0, min(int(x2), w-1))
                    y2 = max(0, min(int(y2), h-1))
                    
                    # check if bbox is valid (not too small)
                    if (x2 - x1) > 5 and (y2 - y1) > 5:
                        results[img_path] = {
                            'label': lbl, 
                            'x1': int(x1), 'y1': int(y1), 
                            'x2': int(x2), 'y2': int(y2),
                            'confidence': float(confidences[best_idx])
                        }
                    else:
                        print(f"⚠️ Too small face detected: {img_path}")
                        # fallback: center approximation
                        x1 = int(0.33*w); y1 = int(0.36*h)
                        x2 = int(0.66*w); y2 = int(0.70*h)
                        results[img_path] = {
                            'label': lbl, 
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'confidence': 0.0
                        }
                else:
                    # fallback: center approximation
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        x1 = int(0.33*w); y1 = int(0.36*h)
                        x2 = int(0.66*w); y2 = int(0.70*h)
                        results[img_path] = {
                            'label': lbl, 
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'confidence': 0.0
                        }
                        print(f"⚠️ Face detection failed, using center approximation: {img_path}")
                    else:
                        print(f"⚠️ Unable to read image: {img_path}")
                        continue
                
                cnt += 1
                if cnt % 1000 == 0:
                    print(f"📈 {cnt}/{total_images} images processed ({cnt/total_images*100:.1f}%)")
                    
            except Exception as e:
                print(f"❌ Error occurred ({img_path}): {e}")
                # fallback: center approximation
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        x1 = int(0.33*w); y1 = int(0.36*h)
                        x2 = int(0.66*w); y2 = int(0.70*h)
                        results[img_path] = {
                            'label': lbl, 
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'confidence': 0.0
                        }
                        cnt += 1
                except:
                    print(f"❌ Recovery failed: {img_path}")
                    continue
    
    print(f"✅ Total {cnt} images processed")
    
    # save results
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # print statistics
    successful_detections = sum(1 for v in results.values() if v.get('confidence', 0) > 0)
    failed_detections = sum(1 for v in results.values() if v.get('confidence', 0) == 0)
    print(f"📊 Detection success rate: {successful_detections}/{cnt} ({successful_detections/cnt*100:.1f}%)")
    print(f"📊 Detection failed: {failed_detections}/{cnt} ({failed_detections/cnt*100:.1f}%)")
    print(f"💾 Results saved: {out_file}")

def save_bbox_from_landmarks_json(landmarks_json_file, out_file):
    """
    Create bbox from landmarks JSON file.
    
    Args:
        landmarks_json_file: Landmarks JSON file path
        out_file: Output bbox JSON file path
    """
    print(f"🚀 Create bbox from landmarks JSON file")
    print(f"📁 Landmarks file: {landmarks_json_file}")
    print(f"💾 Output file: {out_file}")
    
    # load landmarks JSON file
    with open(landmarks_json_file, 'r', encoding='utf-8') as f:
        landmarks_data = json.load(f)
    
    results = {}
    label_count = -1
    label_book = {}
    image_count = {}
    
    print(f"📊 Total {len(landmarks_data)} images processing started...")
    
    for img_path, landmark_info in landmarks_data.items():
        try:
            # extract landmarks information
            landmarks_5pt = landmark_info['landmarks_5pt']
            person = landmark_info['person']
            confidence = landmark_info.get('confidence', 1.0)
            
            # calculate bbox from landmarks
            bbox = rect_from_landmark(landmarks_5pt)
            
            # map label
            if person not in label_book:
                label_count += 1
                label_book[person] = label_count
                image_count[person] = 1
            else:
                image_count[person] += 1
            
            label = label_book[person]
            
            # save results
            results[img_path] = {
                'label': label,
                'x1': int(bbox[0]), 'y1': int(bbox[1]),
                'x2': int(bbox[2]), 'y2': int(bbox[3]),
                'confidence': confidence,
                'person': person
            }
            
        except Exception as e:
            print(f"❌ Error occurred ({img_path}): {e}")
            continue
    
    # save results
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # print statistics
    print("-" * 50)
    print(f"✅ Create bbox from landmarks JSON file completed!")
    print(f"📊 Total {len(results)} images processed")
    print(f"👥 Total {len(label_book)} persons found")
    print(f"💾 Results saved: {out_file}")


def save_bbox_to_json(root_dir, out_file, method='yolov8face', model_path=None, conf_threshold=0.5, landmarks_json=None):
    """
    Create bbox for LFW dataset.
    
    Args:
        root_dir: LFW dataset root directory
        out_file: Output JSON file path
        method: bbox creation method ('fixedcrop', 'cascade', 'yolov8face', 'landmarks')
        model_path: YOLOv8-face model path (if yolov8face method is used)
        conf_threshold: YOLOv8-face confidence threshold
        landmarks_json: Landmarks JSON file path (if landmarks method is used)
    """
    print(f"🚀 Create bbox for LFW dataset - method: {method}")
    print(f"📁 Input directory: {root_dir}")
    print(f"💾 Output file: {out_file}")
    
    if method == 'landmarks':
        if landmarks_json is None:
            raise ValueError("landmarks method requires landmarks_json parameter.")
        print("🎯 Use landmarks method")
        save_bbox_from_landmarks_json(landmarks_json, out_file)
    elif method == 'fixedcrop':
        print("📐 Use fixed crop ratio method")
        save_bbox_fixedcrop(root_dir, out_file)
    elif method == 'cascade':
        print("🔍 Use Haar Cascade method")
        save_bbox_detect(root_dir, out_file)
    elif method == 'yolov8face':
        print("🤖 Use YOLOv8-face method")
        save_bbox_yolov8face(root_dir, out_file, model_path, conf_threshold)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    print(f"✅ Create bbox for LFW dataset completed: {out_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create bbox for LFW dataset')
    parser.add_argument('--root_dir', required=True, help='LFW dataset root directory')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--method', choices=['fixedcrop', 'cascade', 'yolov8face'], 
                       default='yolov8face', help='bbox creation method')
    parser.add_argument('--model_path', help='YOLOv8-face model path (if automatically downloaded)')
    parser.add_argument('--conf_threshold', type=float, default=0.1, 
                       help='YOLOv8-face confidence threshold (default: 0.1)')
    
    args = parser.parse_args()
    save_bbox_to_json(args.root_dir, args.output, args.method, args.model_path, args.conf_threshold)