#!/usr/bin/env python3
"""
convert landmarks JSON to Label Studio format

Usage:
    python generate_landmark/convert_to_labelstudio.py --input landmarks.json --output labelstudio.json --dataset_name custom_250915
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any


def convert_landmarks_to_labelstudio(landmarks_json_file: str, output_file: str, dataset_name: str) -> None:
    """
    convert landmarks JSON to Label Studio format
    
    Args:
        landmarks_json_file: landmarks JSON file path
        output_file: output Label Studio JSON file path
        dataset_name: dataset name
    """
    print(f"🚀 Convert landmarks JSON → Label Studio")
    print(f"📁 Input file: {landmarks_json_file}")
    print(f"📁 Dataset name: {dataset_name}")
    print(f"💾 Output file: {output_file}")
    
    # load landmarks JSON
    with open(landmarks_json_file, 'r', encoding='utf-8') as f:
        landmarks_data = json.load(f)
    
    # convert to Label Studio format
    labelstudio_data = []
    
    for img_path, landmark_info in landmarks_data.items():
        try:
            # extract landmarks information
            landmarks_5pt = landmark_info['landmarks_5pt']
            person = landmark_info['person']
            confidence = landmark_info.get('confidence', 1.0)
            image_shape = landmark_info.get('image_shape', [250, 250, 3])
            
            # create landmarks points in Label Studio keypoint format
            # 5 points landmarks: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
            # left eye, right eye, nose, left mouth, right mouth
            landmark_points = []
            landmark_names = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
            
            for i in range(5):
                x = landmarks_5pt[i * 2]
                y = landmarks_5pt[i * 2 + 1]
                
                # convert pixel coordinates to percentage
                x_percent = (x / image_shape[1]) * 100
                y_percent = (y / image_shape[0]) * 100
                
                # Label Studio keypoint format
                landmark_points.append({
                    "id": f"point{i+1}",
                    "from_name": "kp",
                    "to_name": "image",
                    "type": "keypointlabels",
                    "original_width": image_shape[1],
                    "original_height": image_shape[0],
                    "image_rotation": 0,
                    "value": {
                        "x": x_percent,
                        "y": y_percent,
                        "width": 1.0,
                        "keypointlabels": [landmark_names[i]]
                    }
                })
            
            # create Label Studio format data
            labelstudio_item = {
                "data": {
                    "image": f"/data/local-files/?d={dataset_name}/{img_path}"  # image path in Label Studio
                },
                "predictions": [
                    {
                        "model_version": "mediapipe_landmarks",
                        "score": confidence,
                        "result": landmark_points
                    }
                ]
            }
            
            labelstudio_data.append(labelstudio_item)
            
        except Exception as e:
            print(f"❌ Error occurred while processing image ({img_path}): {e}")
            continue
    
    # save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labelstudio_data, f, indent=2, ensure_ascii=False)
    
    print("-" * 50)
    print(f"✅ Label Studio format conversion completed!")
    print(f"📊 Total converted images: {len(labelstudio_data)}")
    print(f"💾 Results saved: {output_file}")
    print(f"📋 Now you can use it for import in Label Studio.")


def create_labelstudio_config(output_file: str) -> None:
    """
    create Label Studio config file
    
    Args:
        output_file: output config file path
    """
    config = """<View>
  <Image name="image" value="$image" zoom="true"/>
  <KeyPointLabels name="kp" toName="image">
    <Label value="left_eye" background="red"/>
    <Label value="right_eye" background="blue"/>
    <Label value="nose" background="green"/>
    <Label value="left_mouth" background="yellow"/>
    <Label value="right_mouth" background="purple"/>
  </KeyPointLabels>
</View>"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(config)
    
    print(f"📋 Label Studio config file created: {output_file}")


def main():
    """main function"""
    parser = argparse.ArgumentParser(
        description="Convert landmarks JSON to Label Studio format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
    # basic conversion
    python generate_landmark/convert_to_labelstudio.py --input landmarks.json --output labelstudio.json --dataset_name custom_250915
    
    # create config file also
    python generate_landmark/convert_to_labelstudio.py --input landmarks.json --output labelstudio.json --dataset_name custom_250915 --create_config
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input landmarks JSON file path')
    parser.add_argument('--output', '-o', required=True, help='Output Label Studio JSON file path')
    parser.add_argument('--dataset_name', '-d', required=True, help='Dataset name')
    parser.add_argument('--create_config', action='store_true', help='Create Label Studio config file')
    
    args = parser.parse_args()
    
    # check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return
    
    # run conversion
    convert_landmarks_to_labelstudio(args.input, args.output, args.dataset_name)
    
    # create config file (optional)
    if args.create_config:
        config_file = args.output.replace('.json', '_config.xml')
        create_labelstudio_config(config_file)


if __name__ == '__main__':
    main()
