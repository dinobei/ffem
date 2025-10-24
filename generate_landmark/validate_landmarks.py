#!/usr/bin/env python3
"""
landmark validation script - check face alignment

LFW dataset is aligned faces, so the nose should be in the center of the image.
This script finds faces that are not aligned and prints them.

Usage:
    python generate_landmark/validate_landmarks.py --input landmarks.json --output validation_report.txt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any


def validate_face_alignment(landmarks_data: Dict[str, Any], 
                          image_width: int, 
                          image_height: int,
                          center_threshold: float = 0.3) -> List[Tuple[int, str, Dict[str, float]]]:
    """
    Validate face alignment (5-point landmark average coordinates)
    
    Args:
        landmarks_data: landmarks data
        image_width: image width
        image_height: image height
        center_threshold: center tolerance (0.3 = 30%)
        
    Returns:
        List of failed items [(index, file path, validation info), ...]
    """
    failed_items = []
    
    # image center coordinates
    center_x = image_width / 2
    center_y = image_height / 2
    
    # calculate tolerance range
    x_tolerance = image_width * center_threshold
    y_tolerance = image_height * center_threshold
    
    for idx, (img_path, landmark_info) in enumerate(landmarks_data.items()):
        try:
            person = landmark_info.get('person', '')
            failure_reasons = []
            validation_failed = False

            # validate landmarks existence and count
            landmarks_5pt = landmark_info.get('landmarks_5pt')
            if landmarks_5pt is None:
                validation_failed = True
                failure_reasons.append('Landmarks missing (landmarks_5pt missing)')
                failed_items.append((idx, img_path, {
                    'face_center_x': 0.0,
                    'face_center_y': 0.0,
                    'image_center_x': 0.0,
                    'image_center_y': 0.0,
                    'x_offset_percent': 0.0,
                    'y_offset_percent': 0.0,
                    'nose_x': 0.0,
                    'nose_y': 0.0,
                    'left_eye_x': 0.0,
                    'left_eye_y': 0.0,
                    'right_eye_x': 0.0,
                    'right_eye_y': 0.0,
                    'left_mouth_x': 0.0,
                    'left_mouth_y': 0.0,
                    'right_mouth_x': 0.0,
                    'right_mouth_y': 0.0,
                    'failure_reasons': failure_reasons,
                    'person': person,
                    'confidence': landmark_info.get('confidence', 1.0)
                }))
                continue

            if not isinstance(landmarks_5pt, list) or len(landmarks_5pt) != 10:
                validation_failed = True
                failure_reasons.append(f'Landmarks count error (expected: 5-point/10 values, actual: {0 if not isinstance(landmarks_5pt, list) else len(landmarks_5pt)})')
                # continue even if possible, but report and move on
                # pad with 0.0 to safely fill the missing coordinates
                if isinstance(landmarks_5pt, list):
                    landmarks_5pt = (landmarks_5pt + [0.0]*10)[:10]
                else:
                    landmarks_5pt = [0.0]*10

            # extract 5-point landmarks coordinates
            left_eye_x, left_eye_y = landmarks_5pt[0], landmarks_5pt[1]
            right_eye_x, right_eye_y = landmarks_5pt[2], landmarks_5pt[3]
            nose_x, nose_y = landmarks_5pt[4], landmarks_5pt[5]
            left_mouth_x, left_mouth_y = landmarks_5pt[6], landmarks_5pt[7]
            right_mouth_x, right_mouth_y = landmarks_5pt[8], landmarks_5pt[9]

            # validate value validity (number/None)
            coords = [left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, left_mouth_x, left_mouth_y, right_mouth_x, right_mouth_y]
            if any(v is None for v in coords):
                validation_failed = True
                failure_reasons.append('Landmarks value contains None')

            # calculate average 5-point landmarks coordinates
            face_center_x = (left_eye_x + right_eye_x + nose_x + left_mouth_x + right_mouth_x) / 5
            face_center_y = (left_eye_y + right_eye_y + nose_y + left_mouth_y + right_mouth_y) / 5

            # check image size information
            if 'image_shape' in landmark_info:
                img_h, img_w = landmark_info['image_shape'][:2]
            else:
                img_h, img_w = image_height, image_width

            # actual image center coordinates
            actual_center_x = img_w / 2
            actual_center_y = img_h / 2

            # calculate how far the face center is from the center
            x_offset = abs(face_center_x - actual_center_x)
            y_offset = abs(face_center_y - actual_center_y)

            # calculate how far from the center (percentage)
            x_offset_percent = (x_offset / actual_center_x) * 100
            y_offset_percent = (y_offset / actual_center_y) * 100

            # rule 2: average coordinates deviate from the center
            if x_offset_percent > center_threshold * 100:
                validation_failed = True
                failure_reasons.append(f"X-axis deviation: {x_offset_percent:.1f}% (tolerance: {center_threshold*100:.1f}%)")

            if y_offset_percent > center_threshold * 100:
                validation_failed = True
                failure_reasons.append(f"Y-axis deviation: {y_offset_percent:.1f}% (tolerance: {center_threshold*100:.1f}%)")

            # rule 3: left/right/top/bottom logical consistency
            if not (left_eye_x < right_eye_x):
                validation_failed = True
                failure_reasons.append('Left eye should be left of right eye')

            if not (left_mouth_x < right_mouth_x):
                validation_failed = True
                failure_reasons.append('Left mouth should be left of right mouth')

            eye_avg_y = (left_eye_y + right_eye_y) / 2
            mouth_avg_y = (left_mouth_y + right_mouth_y) / 2
            if not (eye_avg_y < nose_y < mouth_avg_y):
                validation_failed = True
                failure_reasons.append('Nose should be between left eye and right eye')

            # save validation result if failed
            if validation_failed:
                validation_info = {
                    'face_center_x': face_center_x,
                    'face_center_y': face_center_y,
                    'image_center_x': actual_center_x,
                    'image_center_y': actual_center_y,
                    'x_offset_percent': x_offset_percent,
                    'y_offset_percent': y_offset_percent,
                    'nose_x': nose_x,
                    'nose_y': nose_y,
                    'left_eye_x': left_eye_x,
                    'left_eye_y': left_eye_y,
                    'right_eye_x': right_eye_x,
                    'right_eye_y': right_eye_y,
                    'left_mouth_x': left_mouth_x,
                    'left_mouth_y': left_mouth_y,
                    'right_mouth_x': right_mouth_x,
                    'right_mouth_y': right_mouth_y,
                    'failure_reasons': failure_reasons,
                    'person': person,
                    'confidence': landmark_info.get('confidence', 1.0)
                }
                failed_items.append((idx, img_path, validation_info))
                
        except Exception as e:
            print(f"⚠️  Error occurred while processing image ({img_path}): {e}")
            continue
    
    return failed_items


def generate_validation_report(landmarks_file: str, 
                             output_file: str, 
                             center_threshold: float = 0.3) -> None:
    """
    Generate landmark validation report
    
    Args:
        landmarks_file: landmarks JSON file path
        output_file: output report file path
        center_threshold: center tolerance (0.3 = 30%)
    """
    print(f"🔍 Landmark validation started")
    print(f"📁 Input file: {landmarks_file}")
    print(f"💾 Output file: {output_file}")
    print(f"📏 Center tolerance: {center_threshold*100:.1f}%")
    
    # load landmarks data
    with open(landmarks_file, 'r', encoding='utf-8') as f:
        landmarks_data = json.load(f)
    
    print(f"📊 Total {len(landmarks_data)} images to validate...")
    
    # default image size (LFW is usually 250x250)
    default_width, default_height = 250, 250
    
    # run validation
    failed_items = validate_face_alignment(landmarks_data, default_width, default_height, center_threshold)
    
    # generate report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Face alignment validation report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Validation settings:\n")
        f.write(f"- Input file: {landmarks_file}\n")
        f.write(f"- Total images: {len(landmarks_data)}\n")
        f.write(f"- Center tolerance: {center_threshold*100:.1f}%\n")
        f.write(f"- Validation failed: {len(failed_items)} items\n")
        f.write(f"- Validation success rate: {((len(landmarks_data) - len(failed_items)) / len(landmarks_data) * 100):.1f}%\n\n")
        
        if failed_items:
            f.write("Validation failed items:\n")
            f.write("-" * 80 + "\n")
            
            for idx, img_path, validation_info in failed_items:
                f.write(f"\n[{idx+1:4d}] {img_path}\n")
                f.write(f"     Person: {validation_info['person']}\n")
                f.write(f"     Confidence: {validation_info['confidence']:.3f}\n")
                f.write(f"     Face center: ({validation_info['face_center_x']:.1f}, {validation_info['face_center_y']:.1f})\n")
                f.write(f"     Image center: ({validation_info['image_center_x']:.1f}, {validation_info['image_center_y']:.1f})\n")
                f.write(f"     X-axis error: {validation_info['x_offset_percent']:.1f}%\n")
                f.write(f"     Y-axis error: {validation_info['y_offset_percent']:.1f}%\n")
                f.write(f"     Landmarks coordinates:\n")
                f.write(f"       - Left eye: ({validation_info['left_eye_x']:.1f}, {validation_info['left_eye_y']:.1f})\n")
                f.write(f"       - Right eye: ({validation_info['right_eye_x']:.1f}, {validation_info['right_eye_y']:.1f})\n")
                f.write(f"       - Nose: ({validation_info['nose_x']:.1f}, {validation_info['nose_y']:.1f})\n")
                f.write(f"     Failure reasons:\n")
                for reason in validation_info['failure_reasons']:
                    f.write(f"       - {reason}\n")
        else:
            f.write("✅ All images satisfy the center alignment criteria!\n")
    
    # console output
    print("-" * 50)
    print(f"✅ Validation completed!")
    print(f"📊 Total images: {len(landmarks_data)}")
    print(f"❌ Validation failed: {len(failed_items)}")
    print(f"✅ Validation success: {len(landmarks_data) - len(failed_items)}")
    print(f"📈 Success rate: {((len(landmarks_data) - len(failed_items)) / len(landmarks_data) * 100):.1f}%")
    print(f"💾 Report saved: {output_file}")
    
    # print summary of failed items
    if failed_items:
        print(f"\n🔍 Validation failed items summary:")
        for idx, img_path, validation_info in failed_items[:10]:  # print first 10 items
            print(f"  [{idx+1:4d}] {img_path} - {validation_info['person']} (X:{validation_info['x_offset_percent']:.1f}%, Y:{validation_info['y_offset_percent']:.1f}%)")
        
        if len(failed_items) > 10:
            print(f"  ... and {len(failed_items) - 10} more (see report file for details)")


def main():
    """main function"""
    parser = argparse.ArgumentParser(
        description="Landmark validation script - check face alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
    # basic validation (30% tolerance)
    python generate_landmark/validate_landmarks.py --input landmarks.json --output validation_report.txt
    
    # strict validation (20% tolerance)
    python generate_landmark/validate_landmarks.py --input landmarks.json --output strict_report.txt --threshold 0.2
    
    # lenient validation (40% tolerance)
    python generate_landmark/validate_landmarks.py --input landmarks.json --output lenient_report.txt --threshold 0.4
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Landmarks JSON file path')
    parser.add_argument('--output', '-o', required=True, help='Output report file path')
    parser.add_argument('--threshold', '-t', type=float, default=0.3, 
                       help='Center tolerance (0.0-1.0, default: 0.3 = 30%)')
    
    args = parser.parse_args()
    
    # check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return
    
    # create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # run validation
    generate_validation_report(args.input, args.output, args.threshold)


if __name__ == '__main__':
    main()
