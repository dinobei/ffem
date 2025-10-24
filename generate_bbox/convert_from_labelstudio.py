"""
Convert Label Studio export(JSON) to bbox JSON format

Usage:
  python generate_bbox/convert_from_labelstudio.py \
      --input hanbat_labelstudio_export.json \
      --output hanbat_bbox_verified.json \
      [--strip_dataset_prefix hanbat_251022]
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List


def extract_relative_path(image_field: str, strip_dataset_prefix: str = "") -> str:
    # "/data/local-files/?d=DATASET/relative/path.jpg" → "relative/path.jpg"
    rel = image_field
    if "/data/local-files/?d=" in image_field:
        rel = image_field.split("/data/local-files/?d=")[1]
    if strip_dataset_prefix and rel.startswith(strip_dataset_prefix + "/"):
        rel = rel[len(strip_dataset_prefix) + 1:]
    return rel


def convert_labelstudio_to_bbox(ls_json_file: str, output_file: str, strip_dataset_prefix: str) -> None:
    print("🚀 Convert Label Studio → bbox")
    print(f"📁 Input file: {ls_json_file}")
    print(f"💾 Output file: {output_file}")

    with open(ls_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out: Dict[str, Dict[str, Any]] = {}

    for item in data:
        try:
            image_field = item['data']['image']
            rel_path = extract_relative_path(image_field, strip_dataset_prefix)

            width = None
            height = None
            x_percent = None
            y_percent = None
            w_percent = None
            h_percent = None
            score = 1.0
            label = None

            # annotation first, then prediction
            if 'annotations' in item and item['annotations']:
                ann = item['annotations'][0]
                # score may be in annotation.prediction.score
                if 'prediction' in ann and ann['prediction'] and 'score' in ann['prediction']:
                    try:
                        score = float(ann['prediction']['score'])
                    except Exception:
                        pass
                for res in ann.get('result', []):
                    if res.get('type') == 'rectanglelabels':
                        value = res.get('value', {})
                        x_percent = float(value.get('x'))
                        y_percent = float(value.get('y'))
                        w_percent = float(value.get('width'))
                        h_percent = float(value.get('height'))
                        width = int(res.get('original_width'))
                        height = int(res.get('original_height'))
                        labels = value.get('rectanglelabels', [])
                        if labels:
                            label = labels[0]
                        break
            # fallback: predictions
            if (x_percent is None or width is None) and 'predictions' in item and item['predictions']:
                pred = item['predictions'][0]
                try:
                    score = float(pred.get('score', score))
                except Exception:
                    pass
                for res in pred.get('result', []):
                    if res.get('type') == 'rectanglelabels':
                        value = res.get('value', {})
                        x_percent = float(value.get('x'))
                        y_percent = float(value.get('y'))
                        w_percent = float(value.get('width'))
                        h_percent = float(value.get('height'))
                        width = int(res.get('original_width'))
                        height = int(res.get('original_height'))
                        labels = value.get('rectanglelabels', [])
                        if labels:
                            label = labels[0]
                        break

            if None in (width, height, x_percent, y_percent, w_percent, h_percent):
                print(f"⚠️ Missing required information, skipping: {rel_path}")
                continue

            # percentage → pixel coordinates
            x1 = int(round((x_percent / 100.0) * width))
            y1 = int(round((y_percent / 100.0) * height))
            x2 = int(round(((x_percent + w_percent) / 100.0) * width))
            y2 = int(round(((y_percent + h_percent) / 100.0) * height))

            out[rel_path] = {
                "label": -1,  # label information is given in a separate mapping step (if needed)
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": float(score)
            }
        except Exception as e:
            print(f"❌ Conversion error: {e}")
            continue

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("-" * 50)
    print(f"✅ Conversion completed! Total: {len(out)}")
    print(f"💾 Result: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Label Studio export(JSON) → bbox JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python generate_bbox/convert_from_labelstudio.py \
      --input hanbat_labelstudio_export.json \
      --output hanbat_bbox_verified.json \
      --strip_dataset_prefix hanbat_251022
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Label Studio export JSON file')
    parser.add_argument('--output', '-o', required=True, help='Output bbox JSON file')
    parser.add_argument('--strip_dataset_prefix', type=str, default='', help='Dataset prefix to remove from image field (e.g. hanbat_251022)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    convert_labelstudio_to_bbox(
        ls_json_file=args.input,
        output_file=args.output,
        strip_dataset_prefix=args.strip_dataset_prefix,
    )


if __name__ == '__main__':
    main()


