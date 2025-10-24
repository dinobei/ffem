#!/usr/bin/env python3
"""
script to convert Label Studio export data to landmarks JSON format

Usage:
    python generate_landmark/convert_from_labelstudio.py \
        --input labelstudio_export.json \
        --output landmarks_verified.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


def extract_image_path_from_labelstudio(image_path: str) -> str:
    """
    extract actual file path from Label Studio image path
    
    Args:
        image_path: Label Studio image path (e.g. "/data/local-files/?d=lfw_funneled/AJ_Cook/AJ_Cook_0001.jpg")
        
    Returns:
        actual file path (e.g. "AJ_Cook/AJ_Cook_0001.jpg")
    """
    # remove "/data/local-files/?d=lfw_funneled/" part
    if "/data/local-files/?d=" in image_path:
        # remove "lfw_funneled/" part to extract relative path
        path = image_path.split("/data/local-files/?d=")[1]
        if path.startswith("lfw_funneled/"):
            path = path[len("lfw_funneled/"):]
        return path
    return image_path


def convert_labelstudio_to_landmarks(labelstudio_file: str, output_file: str) -> None:
    """
    convert Label Studio export data to landmarks JSON format
    
    Args:
        labelstudio_file: Label Studio export JSON file path
        output_file: Output landmarks JSON file path
    """
    print(f"🚀 Convert Label Studio → landmarks")
    print(f"📁 Input file: {labelstudio_file}")
    print(f"💾 Output file: {output_file}")
    
    # load Label Studio export data
    with open(labelstudio_file, 'r', encoding='utf-8') as f:
        labelstudio_data = json.load(f)
    
    # dictionary for storing landmarks data
    landmarks_data = {}
    
    # landmark name mapping (Label Studio → our format)
    landmark_mapping = {
        'left_eye': 0,
        'right_eye': 1,
        'nose': 2,
        'left_mouth': 3,
        'right_mouth': 4
    }
    
    print(f"📊 Total {len(labelstudio_data)} images processing started...")
    
    for item in labelstudio_data:
        try:
            # extract image path
            image_path = item['data']['image']
            relative_path = extract_image_path_from_labelstudio(image_path)
            
            # extract person name (folder name)
            person = relative_path.split('/')[-2]
            
            # extract verified landmarks data
            if 'annotations' in item and len(item['annotations']) > 0:
                annotation = item['annotations'][0]
                if 'result' in annotation:
                    # sort landmarks points in order
                    landmarks_5pt = [0.0] * 10  # 5 points × 2 coordinates
                    image_shape = [250, 250, 3]  # default value
                    
                    for point in annotation['result']:
                        if point['type'] == 'keypointlabels':
                            # extract image size information
                            if 'original_width' in point:
                                image_shape[1] = point['original_width']
                            if 'original_height' in point:
                                image_shape[0] = point['original_height']
                            
                            # extract landmarks label and coordinates
                            label = point['value']['keypointlabels'][0]
                            x_percent = point['value']['x']
                            y_percent = point['value']['y']
                            
                            # convert percentage to pixel coordinates
                            x_pixel = (x_percent / 100.0) * image_shape[1]
                            y_pixel = (y_percent / 100.0) * image_shape[0]
                            
                            # save landmarks points in order
                            if label in landmark_mapping:
                                idx = landmark_mapping[label]
                                landmarks_5pt[idx * 2] = float(x_pixel)
                                landmarks_5pt[idx * 2 + 1] = float(y_pixel)
                    
                    # extract confidence score (from prediction)
                    confidence = 1.0
                    if 'predictions' in item and len(item['predictions']) > 0:
                        # predictions is an array of IDs, so we need to find the actual prediction object
                        prediction_id = item['predictions'][0]
                        # extract actual data from prediction field in annotation
                        if 'prediction' in annotation and annotation['prediction']:
                            prediction = annotation['prediction']
                            if 'score' in prediction:
                                confidence = prediction['score']
                    
                    # save results
                    landmarks_data[relative_path] = {
                        'person': person,
                        'landmarks_5pt': landmarks_5pt,
                        'confidence': confidence,
                        'image_shape': image_shape,
                        'verified': True  # marked as verified
                    }
                    
                    print(f"✅ {relative_path}: verified landmarks saved")
                else:
                    print(f"⚠️  {relative_path}: landmarks data not found")
            else:
                print(f"⚠️  {relative_path}: verification data not found")
                
        except Exception as e:
            import traceback
            print(f"❌ Error occurred while processing image ({image_path}): {e}")
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            continue
    
    # save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(landmarks_data, f, indent=2, ensure_ascii=False)
    
    print("-" * 50)
    print(f"✅ Label Studio → landmarks conversion completed!")
    print(f"📊 Total converted images: {len(landmarks_data)}")
    print(f"💾 Results saved: {output_file}")
    print(f"📋 Now you can use it for bbox creation.")


def main():
    """main function"""
    parser = argparse.ArgumentParser(
        description="Convert Label Studio export data to landmarks JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
    # basic conversion
    python generate_landmark/convert_from_labelstudio.py --input labelstudio_export.json --output landmarks_verified.json
    
    # convert verified data from Label Studio to landmarks format
    python generate_landmark/convert_from_labelstudio.py --input verified_landmarks.json --output final_landmarks.json
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Label Studio export JSON file path')
    parser.add_argument('--output', '-o', required=True, help='Output landmarks JSON file path')
    
    args = parser.parse_args()
    
    # check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return
    
    # create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # run conversion
    convert_labelstudio_to_landmarks(args.input, args.output)


if __name__ == '__main__':
    main()
