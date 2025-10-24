#!/usr/bin/env python3
"""
Extract images from TFRecord file and draw bbox to visualize

Usage:
    python generate_tfrecord/tfrecord_to_img.py --input /path/to/file.tfrecord --output /path/to/output_dir --num_images 100
"""

import argparse
import os
import sys
from pathlib import Path
import json
from typing import List, Tuple, Optional

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# set Korean font for support
try:
    # find Korean font installed on the system
    import matplotlib.font_manager as fm
    korean_fonts = [f.name for f in fm.fontManager.ttflist if 'Nanum' in f.name or 'Malgun' in f.name or 'Gulim' in f.name]
    if korean_fonts:
        FONT_PATH = fm.findfont(fm.FontProperties(family=korean_fonts[0]))
    else:
        FONT_PATH = None
except:
    FONT_PATH = None


def parse_tfrecord_example(serialized_example) -> Tuple[np.ndarray, int, List[int]]:
    """
    Parse TFRecord example and return image, label, bbox
    
    Args:
        serialized_example: serialized example of TFRecord
        
    Returns:
        image: numpy array of image (H, W, C)
        label: integer label
        bbox: [x1, y1, x2, y2] format of bbox coordinates
    """
    description = {
        'jpeg': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'x1': tf.io.FixedLenFeature((), tf.int64),
        'y1': tf.io.FixedLenFeature((), tf.int64),
        'x2': tf.io.FixedLenFeature((), tf.int64),
        'y2': tf.io.FixedLenFeature((), tf.int64)
    }
    
    example = tf.io.parse_single_example(serialized_example, description)
    
    # decode JPEG image
    image = tf.io.decode_jpeg(example['jpeg'], channels=3)
    image = image.numpy()
    
    # extract label and bbox
    label = int(example['label'].numpy())
    bbox = [
        int(example['x1'].numpy()),
        int(example['y1'].numpy()),
        int(example['x2'].numpy()),
        int(example['y2'].numpy())
    ]
    
    return image, label, bbox


def draw_bbox_on_image(image: np.ndarray, bbox: List[int], label: int, 
                      confidence: Optional[float] = None) -> np.ndarray:
    """
    Draw bbox and label on image
    
    Args:
        image: numpy array of image (H, W, C)
        bbox: [x1, y1, x2, y2] format of bbox coordinates
        label: integer label
        confidence: confidence (optional)
        
    Returns:
        image with bbox drawn
    """
    x1, y1, x2, y2 = bbox
    
    # convert to PIL Image (for Korean font support)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # draw bbox
    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
    
    # prepare label text
    label_text = f"Label: {label}"
    if confidence is not None:
        label_text += f" (Conf: {confidence:.3f})"
    
    # set font
    try:
        if FONT_PATH:
            font = ImageFont.truetype(FONT_PATH, 20)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # draw text background box
    bbox_text = draw.textbbox((x1, y1-30), label_text, font=font)
    draw.rectangle(bbox_text, fill='red')
    
    # draw text
    draw.text((x1, y1-30), label_text, fill='white', font=font)
    
    # convert back to numpy array
    return np.array(pil_image)


def save_image_with_bbox(image: np.ndarray, bbox: List[int], label: int, 
                        output_path: str, confidence: Optional[float] = None):
    """
    Save image with bbox drawn
    
    Args:
        image: numpy array of image (H, W, C)
        bbox: bbox coordinates
        label: label
        output_path: path to save image
        confidence: confidence (optional)
    """
    # draw bbox
    image_with_bbox = draw_bbox_on_image(image, bbox, label, confidence)
    
    # save image
    cv2.imwrite(output_path, cv2.cvtColor(image_with_bbox, cv2.COLOR_RGB2BGR))


def extract_images_from_tfrecord(tfrecord_path: str, output_dir: str, 
                                num_images: int = 100, start_index: int = 0) -> dict:
    """
    Extract images from TFRecord file and draw bbox to save
    
    Args:
        tfrecord_path: TFRecord file path
        output_dir: output directory
        num_images: number of images to extract
        start_index: start index
        
    Returns:
        statistics of processing results
    """
    # create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # load TFRecord dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # statistics information
    stats = {
        'total_processed': 0,
        'successful_saves': 0,
        'failed_saves': 0,
        'labels_found': set(),
        'bbox_info': []
    }
    
    print(f"🚀 Extract images from TFRecord file")
    print(f"📁 Input file: {tfrecord_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Number of images to extract: {num_images}")
    print(f"📍 Start index: {start_index}")
    print("-" * 50)
    
    try:
        # convert dataset to iterator
        iterator = iter(dataset)
        
        # skip to start index
        for i in range(start_index):
            try:
                next(iterator)
            except StopIteration:
                print(f"⚠️  Start index {start_index} exceeds dataset size.")
                return stats
        
        # extract and save images
        for i in range(num_images):
            try:
                # get next example
                serialized_example = next(iterator)
                
                # parse example
                image, label, bbox = parse_tfrecord_example(serialized_example)
                
                # update statistics
                stats['total_processed'] += 1
                stats['labels_found'].add(label)
                stats['bbox_info'].append({
                    'index': start_index + i,
                    'label': label,
                    'bbox': bbox,
                    'image_shape': image.shape
                })
                
                # create output filename
                output_filename = f"image_{start_index + i:06d}_label_{label:04d}.jpg"
                output_filepath = output_path / output_filename
                
                # save image
                save_image_with_bbox(image, bbox, label, str(output_filepath))
                stats['successful_saves'] += 1
                
                # print progress
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"📈 Progress: {i + 1}/{num_images} ({((i + 1)/num_images)*100:.1f}%) - "
                          f"Label: {label}, Bbox: {bbox}, Shape: {image.shape}")
                
            except StopIteration:
                print(f"⚠️  Reached end of dataset. Only {i} images processed.")
                break
            except Exception as e:
                print(f"❌ Error processing image {start_index + i}: {e}")
                stats['failed_saves'] += 1
                continue
    
    except Exception as e:
        print(f"❌ Error reading TFRecord file: {e}")
        return stats
    
    # print final statistics
    print("-" * 50)
    print(f"✅ Image extraction completed!")
    print(f"📊 Total processed images: {stats['total_processed']}")
    print(f"✅ Successfully saved images: {stats['successful_saves']}")
    print(f"❌ Failed to save images: {stats['failed_saves']}")
    print(f"🏷️ Found labels: {len(stats['labels_found'])}")
    print(f"🏷️ Label range: {min(stats['labels_found'])} ~ {max(stats['labels_found'])}")
    
    # save statistics information to JSON file
    stats_file = output_path / "extraction_stats.json"
    stats_for_json = {
        'total_processed': stats['total_processed'],
        'successful_saves': stats['successful_saves'],
        'failed_saves': stats['failed_saves'],
        'labels_found': sorted(list(stats['labels_found'])),
        'num_unique_labels': len(stats['labels_found']),
        'bbox_info': stats['bbox_info']
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_for_json, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Statistics information saved: {stats_file}")
    
    return stats


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract images from TFRecord file and draw bbox to visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
    # basic usage (extract 100 images)
    python generate_tfrecord/tfrecord_to_img.py --input data.tfrecord --output ./output
    
    # extract specific number of images
    python generate_tfrecord/tfrecord_to_img.py --input data.tfrecord --output ./output --num_images 50
    
    # start from specific index
    python generate_tfrecord/tfrecord_to_img.py --input data.tfrecord --output ./output --start_index 1000 --num_images 200
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input TFRecord file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory path')
    parser.add_argument('--num_images', '-n', type=int, default=100,
                       help='Number of images to extract (default: 100)')
    parser.add_argument('--start_index', '-s', type=int, default=0,
                       help='Start index (default: 0)')
    
    args = parser.parse_args()
    
    # check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        sys.exit(1)
    
    # check if input file is a TFRecord file
    if not args.input.endswith('.tfrecord'):
        print(f"⚠️  Input file is not a .tfrecord file: {args.input}")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # create output directory
    try:
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to create output directory: {e}")
        sys.exit(1)
    
    # extract images
    try:
        stats = extract_images_from_tfrecord(
            args.input, 
            args.output, 
            args.num_images, 
            args.start_index
        )
        
        if stats['successful_saves'] > 0:
            print(f"\n🎉 Successfully extracted {stats['successful_saves']} images!")
            print(f"📁 Output location: {args.output}")
        else:
            print(f"\n⚠️  No images extracted.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
