#!/usr/bin/env python3
"""
lfw landmark generation module for LFW dataset

Use MediaPipe to detect faces and extract 5-point landmarks.
"""

import json
import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import mediapipe as mp

class FaceLandmarkDetector:
    """Face landmark detector using MediaPipe"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        initialize
        
        Args:
            confidence_threshold: face detection confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        
        # initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # initialize face detector
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0: within 2m, 1: within 5m
            min_detection_confidence=confidence_threshold
        )
        
        # initialize face mesh (for landmark extraction)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=0.5
        )
    
    def detect_face_landmarks(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        detect faces and extract 5-point landmarks from an image
        
        Args:
            image: input image (BGR format)
            
        Returns:
            landmarks information dictionary or None (if detection fails)
        """
        # convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # detect faces
        detection_results = self.face_detection.process(rgb_image)
        
        if not detection_results.detections:
            return None
        
        # select the largest face
        best_detection = max(detection_results.detections, 
                           key=lambda x: x.location_data.relative_bounding_box.width * 
                                       x.location_data.relative_bounding_box.height)
        
        # extract landmarks from face mesh
        mesh_results = self.face_mesh.process(rgb_image)
        
        if not mesh_results.multi_face_landmarks:
            return None
        
        # use the first face's landmarks
        face_landmarks = mesh_results.multi_face_landmarks[0]
        
        # extract 5-point landmarks (left eye, right eye, nose, left mouth, right mouth)
        landmarks_5pt = self._extract_5_point_landmarks(face_landmarks, w, h)
        
        if landmarks_5pt is None:
            return None
        
        # detection confidence
        detection_confidence = best_detection.score[0]
        
        return {
            'landmarks_5pt': landmarks_5pt,
            'confidence': float(detection_confidence),
            'image_shape': [h, w, 3]
        }
    
    def _extract_5_point_landmarks(self, face_landmarks, width: int, height: int) -> Optional[List[float]]:
        """
        extract 5-point landmarks from MediaPipe face mesh
        
        Args:
            face_landmarks: MediaPipe face landmarks object
            width: image width
            height: image height
            
        Returns:
            5-point landmarks list [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5] or None
        """
        try:
            # MediaPipe face mesh indices (5-point landmarks)
            # left eye, right eye, nose, left mouth, right mouth (same format as TrillionPairs)
            landmark_indices = [
                468,  # left eye (left eye center)
                473,  # right eye (right eye center)  
                1,    # nose
                61,   # left mouth (left mouth)
                291   # right mouth (right mouth)
            ]
            
            landmarks_5pt = []
            
            for idx in landmark_indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    # convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    landmarks_5pt.extend([x, y])
                else:
                    return None
            
            return landmarks_5pt
            
        except Exception as e:
            print(f"⚠️  Error occurred while extracting 5-point landmarks: {e}")
            return None
    
    def __del__(self):
        """clean up resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def generate_landmarks(root_dir: str, output_file: str, max_images: Optional[int] = None, 
                      confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    generate landmarks and save as JSON file in LFW dataset
    
    Args:
        root_dir: LFW dataset root directory
        output_file: output JSON file path
        max_images: maximum number of images to process (None means all images)
        confidence_threshold: face detection confidence threshold
        
    Returns:
        processing results statistics
    """
    # create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # initialize landmark detector
    detector = FaceLandmarkDetector(confidence_threshold)
    
    # statistics information
    stats = {
        'total_images': 0,
        'successful_detections': 0,
        'failed_detections': 0,
        'labels_found': set(),
        'confidence_scores': [],
        'image_shapes': []
    }
    
    # dictionary for saving results
    landmarks_data = {}
    
    # LFW dataset structure: root_dir/person_name/image_name.jpg
    persons = sorted([p.name for p in Path(root_dir).iterdir() if p.is_dir()])
    
    print(f"📊 Total {len(persons)} people found")
    
    # supported image extensions
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    
    # calculate total number of images
    total_images = 0
    for person in persons:
        for ext in image_extensions:
            total_images += len(glob.glob(os.path.join(root_dir, person, ext)))
    
    print(f"📊 Total {total_images} images found")
    
    if max_images:
        print(f"🎯 Maximum {max_images} images only")
        total_images = min(total_images, max_images)
    
    processed_count = 0
    
    for person_idx, person in enumerate(persons):
        if max_images and processed_count >= max_images:
            break
            
        print(f"👤 {person_idx + 1}/{len(persons)}: {person} processing...")
        
        # collect all images for the person
        person_images = []
        for ext in image_extensions:
            person_images.extend(glob.glob(os.path.join(root_dir, person, ext)))
        
        for img_path in person_images:
            if max_images and processed_count >= max_images:
                break
                
            try:
                # load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"⚠️  Cannot read image: {img_path}")
                    stats['failed_detections'] += 1
                    continue
                
                stats['total_images'] += 1
                
                # detect landmarks
                result = detector.detect_face_landmarks(image)
                
                if result is not None:
                    # save relative path (based on dataset root)
                    rel_path = os.path.relpath(img_path, root_dir)
                    
                    landmarks_data[rel_path] = {
                        'person': person,
                        'landmarks_5pt': result['landmarks_5pt'],
                        'confidence': result['confidence'],
                        'image_shape': result['image_shape']
                    }
                    
                    stats['successful_detections'] += 1
                    stats['labels_found'].add(person)
                    stats['confidence_scores'].append(result['confidence'])
                    stats['image_shapes'].append(result['image_shape'])
                    
                    if processed_count % 100 == 0:
                        print(f"📈 Progress: {processed_count}/{total_images} ({processed_count/total_images*100:.1f}%) - "
                              f"Confidence: {result['confidence']:.3f}")
                else:
                    stats['failed_detections'] += 1
                    print(f"❌ Face detection failed: {img_path}")
                
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Error occurred while processing image ({img_path}): {e}")
                stats['failed_detections'] += 1
                continue
    
    # save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(landmarks_data, f, indent=2, ensure_ascii=False)
    
    # calculate statistics
    success_rate = (stats['successful_detections'] / stats['total_images'] * 100) if stats['total_images'] > 0 else 0
    avg_confidence = np.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0
    
    print("-" * 50)
    print(f"✅ Landmarks generation completed!")
    print(f"📊 Total processed images: {stats['total_images']}")
    print(f"✅ Successfully detected images: {stats['successful_detections']}")
    print(f"❌ Detected images: {stats['failed_detections']}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    print(f"🎯 Average confidence: {avg_confidence:.3f}")
    print(f"👥 Total people found: {len(stats['labels_found'])}")
    print(f"💾 Results saved: {output_file}")
    
    return stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='LFW dataset landmarks generation')
    parser.add_argument('--root_dir', required=True, help='LFW dataset root directory')
    parser.add_argument('--output', required=True, help='Output landmarks JSON file path')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process (for testing)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, 
                       help='MediaPipe face detection confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    generate_landmarks(args.root_dir, args.output, args.max_images, args.confidence_threshold)
