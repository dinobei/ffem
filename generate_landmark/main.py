#!/usr/bin/env python3
"""
main script for generating landmarks for various datasets

사용법:
    python generate_landmark/main.py lfw --root_dir /path/to/lfw --output /path/to/landmarks.json
"""

import argparse
import os
import sys
from pathlib import Path

import lfw_landmark


def main():
    """main function"""
    parser = argparse.ArgumentParser(
        description="Generate landmarks for various datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
    # LFW dataset landmarks generation
    python generate_landmark/main.py lfw --root_dir /path/to/lfw --output ./lfw_landmarks.json
    
    # VGG-Face2 dataset landmarks generation (future implementation)
    python generate_landmark/main.py vggface2 --root_dir /path/to/vggface2 --output ./vggface2_landmarks.json
        """
    )
    
    sub_parser = parser.add_subparsers(dest='dataset', help='Dataset type')
    
    # LFW sub-parser
    lfw_parser = sub_parser.add_parser('lfw', help='LFW dataset landmarks generation')
    lfw_parser.add_argument('--root_dir', required=True, help='LFW dataset root directory')
    lfw_parser.add_argument('--output', required=True, help='Output landmarks JSON file path')
    lfw_parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process (for testing)')
    lfw_parser.add_argument('--confidence_threshold', type=float, default=0.5, 
                           help='MediaPipe face detection confidence threshold (default: 0.5)')
    
    # VGG-Face2 sub-parser (future implementation)
    vggface2_parser = sub_parser.add_parser('vggface2', help='VGG-Face2 dataset landmarks generation')
    vggface2_parser.add_argument('--root_dir', required=True, help='VGG-Face2 dataset root directory')
    vggface2_parser.add_argument('--output', required=True, help='Output landmarks JSON file path')
    vggface2_parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process (for testing)')
    vggface2_parser.add_argument('--confidence_threshold', type=float, default=0.5, 
                                help='MediaPipe face detection confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    if args.dataset == 'lfw':
        print("🚀 LFW dataset landmarks generation started")
        print(f"📁 Input directory: {args.root_dir}")
        print(f"💾 Output file: {args.output}")
        print(f"🎯 Confidence threshold: {args.confidence_threshold}")
        if args.max_images:
            print(f"📊 Maximum number of images to process: {args.max_images}")
        print("-" * 50)
        
        try:
            lfw_landmark.generate_landmarks(
                args.root_dir, 
                args.output, 
                args.max_images, 
                args.confidence_threshold
            )
            print(f"✅ LFW landmarks generation completed: {args.output}")
        except Exception as e:
            print(f"❌ LFW landmarks generation failed: {e}")
            sys.exit(1)
    
    elif args.dataset == 'vggface2':
        print("⚠️  VGG-Face2 landmarks generation is not implemented yet.")
        sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
