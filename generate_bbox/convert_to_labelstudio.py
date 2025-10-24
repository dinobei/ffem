"""
Convert bbox JSON to Label Studio format

Usage:
  python generate_bbox/convert_to_labelstudio.py \
      --input hanbat_251022_bbox.json \
      --output hanbat_for_labelstudio.json \
      --dataset_name hanbat_251022 \
      [--root_path /notebooks/datasets/face/hanbat_face/output_folder_clean_2] \
      [--strip_prefix /notebooks/datasets/face/]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


def _resolve_image_path(img_path: str, root_path: Optional[str]) -> str:
    p = Path(img_path)
    if p.is_absolute():
        return str(p)
    if root_path:
        return str(Path(root_path) / img_path)
    return str(p)


def _make_relative_path(img_path: str, strip_prefix: Optional[str]) -> str:
    if strip_prefix and img_path.startswith(strip_prefix):
        prefix = strip_prefix.rstrip('/') + '/'
        return img_path[len(prefix):]
    return img_path


def _get_image_size(img_path: str) -> Optional[tuple]:
    # try Pillow first, then OpenCV if failed
    try:
        from PIL import Image  # type: ignore
        with Image.open(img_path) as im:
            width, height = im.size
        return (width, height)
    except Exception:
        try:
            import cv2  # type: ignore
            im = cv2.imread(img_path)
            if im is not None:
                height, width = im.shape[:2]
                return (width, height)
        except Exception:
            pass
    return None


def convert_bbox_to_labelstudio(bbox_json_file: str, output_file: str, dataset_name: str, root_path: Optional[str], strip_prefix: Optional[str], label_name: str) -> None:
    print("🚀 Convert bbox → Label Studio")
    print(f"📁 bbox file: {bbox_json_file}")
    print(f"📁 Dataset name: {dataset_name}")
    print(f"💾 Output file: {output_file}")

    with open(bbox_json_file, 'r', encoding='utf-8') as f:
        bbox_data: Dict[str, Dict[str, Any]] = json.load(f)

    items: List[Dict[str, Any]] = []

    for raw_path, info in bbox_data.items():
        try:
            abs_path = _resolve_image_path(raw_path, root_path)
            rel_path = _make_relative_path(raw_path, strip_prefix)

            img_size = _get_image_size(abs_path)
            if img_size is None:
                print(f"⚠️ Unable to check image size, skipping: {abs_path}")
                continue
            width, height = img_size

            x1 = int(info['x1'])
            y1 = int(info['y1'])
            x2 = int(info['x2'])
            y2 = int(info['y2'])
            score = float(info.get('confidence', 1.0))

            # Label Studio uses x,y,width,height in percentage
            rect_x = (x1 / width) * 100.0
            rect_y = (y1 / height) * 100.0
            rect_w = ((x2 - x1) / width) * 100.0
            rect_h = ((y2 - y1) / height) * 100.0

            ls_item = {
                "data": {
                    "image": f"/data/local-files/?d={dataset_name}/{rel_path}"
                },
                "predictions": [
                    {
                        "model_version": "bbox_converter",
                        "score": score,
                        "result": [
                            {
                                "id": "rect1",
                                "from_name": "bbox",
                                "to_name": "image",
                                "type": "rectanglelabels",
                                "original_width": width,
                                "original_height": height,
                                "image_rotation": 0,
                                "value": {
                                    "x": rect_x,
                                    "y": rect_y,
                                    "width": rect_w,
                                    "height": rect_h,
                                    "rotation": 0,
                                    "rectanglelabels": [label_name]
                                }
                            }
                        ]
                    }
                ]
            }
            items.append(ls_item)
        except Exception as e:
            print(f"❌ Conversion error ({raw_path}): {e}")
            continue

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print("-" * 50)
    print(f"✅ Conversion completed! Total: {len(items)}")
    print(f"💾 Result: {output_file}")


def create_labelstudio_config(output_file: str, label_name: str) -> None:
    config = f"""<View>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="bbox" toName="image">
    <Label value="{label_name}" background="#ff9900"/>
  </RectangleLabels>
</View>"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(config)
    print(f"📋 Label Studio config file created: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert bbox JSON to Label Studio format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python generate_bbox/convert_to_labelstudio.py \
      --input hanbat_251022_bbox.json \
      --output hanbat_for_labelstudio.json \
      --dataset_name hanbat_251022 \
      --root_path /notebooks/datasets/face/hanbat_face/output_folder_clean_2 \
      --strip_prefix /notebooks/datasets/face/
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Input bbox JSON file path')
    parser.add_argument('--output', '-o', required=True, help='Output Label Studio JSON file path')
    parser.add_argument('--dataset_name', '-d', required=True, help='Dataset name')
    parser.add_argument('--root_path', type=str, default=None, help='Image root path (for relative path interpretation)')
    parser.add_argument('--strip_prefix', type=str, default=None, help='Prefix to remove when creating relative path for Label Studio')
    parser.add_argument('--label_name', type=str, default='face', help='Rectangle label name')
    parser.add_argument('--create_config', action='store_true', help='Create Label Studio config file')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    convert_bbox_to_labelstudio(
        bbox_json_file=args.input,
        output_file=args.output,
        dataset_name=args.dataset_name,
        root_path=args.root_path,
        strip_prefix=args.strip_prefix,
        label_name=args.label_name,
    )

    if args.create_config:
        config_file = args.output.replace('.json', '_config.xml')
        create_labelstudio_config(config_file, args.label_name)


if __name__ == '__main__':
    main()


