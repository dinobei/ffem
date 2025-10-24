# python generate_bbox/main.py lfw --root_dir /notebooks/datasets/face/custom_250826 --output /notebooks/datasets/face/custom_250826.json --method yolov8face --conf_threshold 0.4 --model_path models/yolov12l-face.pt

import argparse
import os
import glob
import cv2

import trillion_pairs
import vggface
import rfw
import lfw


parser = argparse.ArgumentParser()
sub_parser = parser.add_subparsers(dest='cmd')
trillon_parser = sub_parser.add_parser('trillion_pairs')
trillon_parser.add_argument('--lndmk')
trillon_parser.add_argument('--output')
vggface_parser = sub_parser.add_parser('vggface')
vggface_parser.add_argument('--lndmk')
vggface_parser.add_argument('--output')
vggface_parser.add_argument('--method', choices=['bbox', 'landmarks'], 
                           default='bbox', help='bbox creation method (default: bbox)')
rfw_parser = sub_parser.add_parser('rfw')
rfw_parser.add_argument('--lndmk')
rfw_parser.add_argument('--output')
lfw_parser = sub_parser.add_parser('lfw')
lfw_parser.add_argument('--root_dir', required=True, help='LFW dataset root directory')
lfw_parser.add_argument('--output', required=True, help='Output JSON file path')
lfw_parser.add_argument('--method', choices=['fixedcrop', 'cascade', 'yolov8face', 'landmarks'], 
                       default='yolov8face', help='bbox creation method')
lfw_parser.add_argument('--model_path', help='YOLOv8-face model path (if automatically downloaded)')
lfw_parser.add_argument('--conf_threshold', type=float, default=0.5, 
                       help='YOLOv8-face confidence threshold (default: 0.5)')
lfw_parser.add_argument('--landmarks_json', help='Landmarks JSON file path (if landmarks method is used)')

# commands for general ID/image folder structure
crop_parser = sub_parser.add_parser('crop')
crop_parser.add_argument('--root_dir', required=True, help='Person ID folder structure root directory')
crop_parser.add_argument('--output', required=True, help='Output JSON file path')

static_parser = sub_parser.add_parser('static')
static_parser.add_argument('--root_dir', required=True, help='Person ID folder structure root directory')
static_parser.add_argument('--output', required=True, help='Output JSON file path')
static_parser.add_argument('--x1', type=float, required=True, help='Fixed bbox x1 (pixel or ratio[0~1])')
static_parser.add_argument('--y1', type=float, required=True, help='Fixed bbox y1 (pixel or ratio[0~1])')
static_parser.add_argument('--x2', type=float, required=True, help='Fixed bbox x2 (pixel or ratio[0~1])')
static_parser.add_argument('--y2', type=float, required=True, help='Fixed bbox y2 (pixel or ratio[0~1])')


def _iter_persons(root_dir):
    return sorted([p for p in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, p))])


def _iter_images(person_dir):
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    paths = []
    for ext in image_extensions:
        paths.extend(glob.glob(os.path.join(person_dir, ext)))
    return paths


def _build_label_book(root_dir):
    persons = _iter_persons(root_dir)
    return {name: i for i, name in enumerate(persons)}, persons


def generate_bbox_crop(root_dir, out_file):
    label_book, persons = _build_label_book(root_dir)
    results = {}
    cnt = 0
    for person in persons:
        lbl = label_book[person]
        person_dir = os.path.join(root_dir, person)
        for img_path in _iter_images(person_dir):
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            x1, y1, x2, y2 = 0, 0, max(0, w - 1), max(0, h - 1)
            results[img_path] = {'label': lbl, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            cnt += 1
            if cnt % 10000 == 0:
                print('{} images are processed'.format(cnt))

    with open(out_file, 'w') as f:
        import json
        json.dump(results, f, indent=2)


def generate_bbox_static(root_dir, out_file, x1, y1, x2, y2):
    # if all inputs are between 0 and 1, interpret as ratio, otherwise interpret as pixel
    is_ratio = (0.0 <= x1 <= 1.0) and (0.0 <= y1 <= 1.0) and (0.0 <= x2 <= 1.0) and (0.0 <= y2 <= 1.0)
    if not (x1 < x2 and y1 < y2):
        raise ValueError('Fixed bbox coordinates are invalid: (x1,y1) < (x2,y2) must be satisfied.')

    label_book, persons = _build_label_book(root_dir)
    results = {}
    cnt = 0
    skip = 0
    for person in persons:
        lbl = label_book[person]
        person_dir = os.path.join(root_dir, person)
        for img_path in _iter_images(person_dir):
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            if is_ratio:
                # ratio → pixel coordinates conversion
                _x1 = int(round(x1 * (w - 1)))
                _y1 = int(round(y1 * (h - 1)))
                _x2 = int(round(x2 * (w - 1)))
                _y2 = int(round(y2 * (h - 1)))
            else:
                _x1 = int(round(x1))
                _y1 = int(round(y1))
                _x2 = int(round(x2))
                _y2 = int(round(y2))

            # ensure sorting
            if _x1 > _x2:
                _x1, _x2 = _x2, _x1
            if _y1 > _y2:
                _y1, _y2 = _y2, _y1

            # ensure minimum width/height
            if _x1 == _x2:
                if _x2 < w - 1:
                    _x2 += 1
                elif _x1 > 0:
                    _x1 -= 1
            if _y1 == _y2:
                if _y2 < h - 1:
                    _y2 += 1
                elif _y1 > 0:
                    _y1 -= 1

            # boundary clamp
            _x1 = max(0, min(_x1, w - 1))
            _y1 = max(0, min(_y1, h - 1))
            _x2 = max(0, min(_x2, w - 1))
            _y2 = max(0, min(_y2, h - 1))

            if _x1 < 0 or _y1 < 0 or _x2 > w - 1 or _y2 > h - 1 or not (_x1 < _x2 and _y1 < _y2):
                skip += 1
                continue

            results[img_path] = {'label': lbl, 'x1': int(_x1), 'y1': int(_y1), 'x2': int(_x2), 'y2': int(_y2)}
            cnt += 1
            if cnt % 10000 == 0:
                print('{} images are processed'.format(cnt))

    with open(out_file, 'w') as f:
        import json
        json.dump(results, f, indent=2)

    if skip > 0:
        print('WARNING: {} images skipped due to bbox out of image bounds.'.format(skip))


if __name__ == '__main__':
    args = parser.parse_args()

    if args.cmd == 'trillion_pairs':
        if os.path.isfile(args.lndmk):
            trillion_pairs.save_bbox_to_json(args.lndmk, args.output)
        else:
            print('{} not exists.'.format(args.lndmk))
    elif args.cmd == 'vggface':
        if os.path.isfile(args.lndmk):
            vggface.save_bbox_to_json(args.lndmk, args.output, args.method)
        else:
            print('{} not exists.'.format(args.lndmk))
    elif args.cmd == 'rfw':
        if os.path.isfile(args.lndmk):
            rfw.save_bbox_to_json(args.lndmk, args.output)
        else:
            print('{} not exists.'.format(args.lndmk))
    elif args.cmd == 'lfw':
        if os.path.isdir(args.root_dir):
            lfw.save_bbox_to_json(args.root_dir, args.output, args.method, args.model_path, args.conf_threshold, args.landmarks_json)
        else:
            print('{} not exists.'.format(args.root_dir))
    elif args.cmd == 'crop':
        if os.path.isdir(args.root_dir):
            generate_bbox_crop(args.root_dir, args.output)
        else:
            print('{} not exists.'.format(args.root_dir))
    elif args.cmd == 'static':
        if os.path.isdir(args.root_dir):
            generate_bbox_static(args.root_dir, args.output, args.x1, args.y1, args.x2, args.y2)
        else:
            print('{} not exists.'.format(args.root_dir))
