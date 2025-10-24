# python generate_tfrecord/quick_count.py --tfrecord /notebooks/datasets/face/AFD/AFDB_dataset_160_part1_train.tfrecord /notebooks/datasets/face/AFD/AFDB_dataset_160_part1_val.tfrecord /notebooks/datasets/face/AFD/AFDB_dataset_160_part1_test.tfrecord /notebooks/datasets/face/aihub_korean_face/aihub_korean_face_288_aligned_train.tfrecord /notebooks/datasets/face/aihub_korean_face/aihub_korean_face_288_aligned_val.tfrecord /notebooks/datasets/face/aihub_korean_face/aihub_korean_face_288_aligned_test.tfrecord /notebooks/datasets/face/CASIA-WebFace/faces_webface_112x112_extracted_train.tfrecord /notebooks/datasets/face/CASIA-WebFace/faces_webface_112x112_extracted_val.tfrecord /notebooks/datasets/face/CASIA-WebFace/faces_webface_112x112_extracted_test.tfrecord /notebooks/datasets/face/custom/custom_250915_aligned.tfrecord /notebooks/datasets/face/hanbat_face/hanbat_251021_aligned_train.tfrecord /notebooks/datasets/face/hanbat_face/hanbat_251021_aligned_val.tfrecord /notebooks/datasets/face/hanbat_face/hanbat_251021_aligned_test.tfrecord --output generate_tfrecord/quick_count_info
import argparse
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from tqdm import tqdm
import matplotlib.font_manager as fm

# set Korean font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def parse_example(example_proto):
    """parse TFRecord example"""
    feature_description = {
        'jpeg': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'x1': tf.io.FixedLenFeature([], tf.int64),
        'y1': tf.io.FixedLenFeature([], tf.int64),
        'x2': tf.io.FixedLenFeature([], tf.int64),
        'y2': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def analyze_single_tfrecord(tfrecord_path):
    """analyze single TFRecord file and calculate class-wise statistics"""
    
    # load TFRecord dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path).map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    # count class-wise
    class_counts = Counter()
    total_images = 0
    
    for example in tqdm(dataset, desc=f"Processing {os.path.basename(tfrecord_path)}"):
        label = int(example['label'].numpy())
        class_counts[label] += 1
        total_images += 1
    
    # calculate statistics
    class_labels = list(class_counts.keys())
    class_counts_list = list(class_counts.values())
    
    stats = {
        'file_name': os.path.basename(tfrecord_path),
        'total_images': total_images,
        'total_classes': len(class_labels),
        'min_images_per_class': min(class_counts_list) if class_counts_list else 0,
        'max_images_per_class': max(class_counts_list) if class_counts_list else 0,
        'mean_images_per_class': np.mean(class_counts_list) if class_counts_list else 0,
        'std_images_per_class': np.std(class_counts_list) if class_counts_list else 0,
        'median_images_per_class': np.median(class_counts_list) if class_counts_list else 0
    }
    
    return class_counts, stats

def analyze_multiple_tfrecords(tfrecord_paths, output_prefix=None):
    """analyze multiple TFRecord files and calculate combined class-wise statistics"""
    
    all_class_counts = Counter()
    all_stats = []
    total_images_all = 0
    
    print("=== Multiple TFRecord Files Analysis ===")
    
    # analyze each file
    for i, tfrecord_path in enumerate(tfrecord_paths):
        print(f"\n[{i+1}/{len(tfrecord_paths)}] Analyzing: {os.path.basename(tfrecord_path)}")
        
        class_counts, stats = analyze_single_tfrecord(tfrecord_path)
        all_class_counts.update(class_counts)
        all_stats.append(stats)
        total_images_all += stats['total_images']
        
        # print individual file results
        print(f"  - Images: {stats['total_images']:,}")
        print(f"  - Classes: {stats['total_classes']:,}")
        print(f"  - Min/Mean/Max per class: {stats['min_images_per_class']}/{stats['mean_images_per_class']:.1f}/{stats['max_images_per_class']}")
    
    # calculate combined statistics
    class_labels = list(all_class_counts.keys())
    class_counts_list = list(all_class_counts.values())
    
    combined_stats = {
        'total_files': len(tfrecord_paths),
        'total_images': total_images_all,
        'total_classes': len(class_labels),
        'min_images_per_class': min(class_counts_list) if class_counts_list else 0,
        'max_images_per_class': max(class_counts_list) if class_counts_list else 0,
        'mean_images_per_class': np.mean(class_counts_list) if class_counts_list else 0,
        'std_images_per_class': np.std(class_counts_list) if class_counts_list else 0,
        'median_images_per_class': np.median(class_counts_list) if class_counts_list else 0
    }
    
    # print combined results
    print(f"\n=== Combined Analysis Results ===")
    print(f"Total files: {combined_stats['total_files']}")
    print(f"Total images: {combined_stats['total_images']:,}")
    print(f"Total classes: {combined_stats['total_classes']:,}")
    print(f"Min images per class: {combined_stats['min_images_per_class']:,}")
    print(f"Max images per class: {combined_stats['max_images_per_class']:,}")
    print(f"Mean images per class: {combined_stats['mean_images_per_class']:.2f}")
    print(f"Std images per class: {combined_stats['std_images_per_class']:.2f}")
    print(f"Median images per class: {combined_stats['median_images_per_class']:.2f}")
    
    # print detailed information for top 10 classes
    print(f"\n=== Top 10 Classes by Image Count ===")
    sorted_classes = sorted(all_class_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (class_label, count) in enumerate(sorted_classes[:10]):
        print(f"Class {class_label}: {count:,} images")
    
    if len(sorted_classes) > 10:
        print(f"... and {len(sorted_classes) - 10} more classes")
    
    # save CSV files
    if output_prefix:
        # save individual file statistics CSV
        individual_stats_path = f"{output_prefix}_individual_file_stats.csv"
        individual_df = pd.DataFrame(all_stats)
        individual_df.to_csv(individual_stats_path, index=False, encoding='utf-8')
        print(f"\nIndividual file statistics saved: {individual_stats_path}")
        
        # save combined class statistics CSV
        combined_class_path = f"{output_prefix}_combined_class_stats.csv"
        combined_df = pd.DataFrame([
            {
                'class_label': label,
                'image_count': count,
                'percentage': (count / total_images_all) * 100
            }
            for label, count in sorted(all_class_counts.items())
        ])
        combined_df.to_csv(combined_class_path, index=False, encoding='utf-8')
        print(f"Combined class statistics saved: {combined_class_path}")
        
        # save combined summary statistics CSV
        summary_path = f"{output_prefix}_combined_summary_stats.csv"
        summary_df = pd.DataFrame([combined_stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        print(f"Combined summary statistics saved: {summary_path}")
    
    # create and save histogram
    if output_prefix:
        plt.figure(figsize=(16, 12))
        
        # subplot 1: class-wise image count distribution (combined data)
        plt.subplot(2, 2, 1)
        plt.hist(class_counts_list, bins=min(50, len(class_counts_list)), alpha=0.7, edgecolor='black')
        plt.title(f'Combined Class Image Count Distribution\n(All {len(tfrecord_paths)} files: {total_images_all:,} total images)')
        plt.xlabel('Number of Images per Class')
        plt.ylabel('Number of Classes')
        plt.grid(True, alpha=0.3)
        
        # add statistics information as text
        stats_text = f'Total Classes: {len(class_labels):,}\n'
        stats_text += f'Min: {combined_stats["min_images_per_class"]:,}\n'
        stats_text += f'Max: {combined_stats["max_images_per_class"]:,}\n'
        stats_text += f'Mean: {combined_stats["mean_images_per_class"]:.1f}\n'
        stats_text += f'Std: {combined_stats["std_images_per_class"]:.1f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # subplot 2: image count of top classes (combined data)
        plt.subplot(2, 2, 2)
        top_classes = sorted_classes[:min(20, len(sorted_classes))]
        labels, counts = zip(*top_classes)
        bars = plt.bar(range(len(labels)), counts, alpha=0.7, edgecolor='black')
        plt.title(f'Top 20 Classes by Image Count\n(Combined from all {len(tfrecord_paths)} files)')
        plt.xlabel('Class ID (Top 20)')
        plt.ylabel('Number of Images')
        plt.xticks(range(len(labels)), [str(l) for l in labels], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # show values on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                    f'{count:,}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # subplot 3: image count per file
        plt.subplot(2, 2, 3)
        file_names = [stats['file_name'] for stats in all_stats]
        file_image_counts = [stats['total_images'] for stats in all_stats]
        bars = plt.bar(range(len(file_names)), file_image_counts, alpha=0.7, edgecolor='black')
        plt.title('Images per File')
        plt.xlabel('TFRecord File')
        plt.ylabel('Number of Images')
        plt.xticks(range(len(file_names)), [os.path.splitext(name)[0] for name in file_names], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # show values on bars
        for i, (bar, count) in enumerate(zip(bars, file_image_counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(file_image_counts)*0.01, 
                    f'{count:,}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # subplot 4: class count per file
        plt.subplot(2, 2, 4)
        file_class_counts = [stats['total_classes'] for stats in all_stats]
        bars = plt.bar(range(len(file_names)), file_class_counts, alpha=0.7, edgecolor='black')
        plt.title('Classes per File')
        plt.xlabel('TFRecord File')
        plt.ylabel('Number of Classes')
        plt.xticks(range(len(file_names)), [os.path.splitext(name)[0] for name in file_names], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # show values on bars
        for i, (bar, count) in enumerate(zip(bars, file_class_counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(file_class_counts)*0.01, 
                    f'{count:,}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # add overall title
        plt.suptitle(f'TFRecord Analysis Summary - {len(tfrecord_paths)} Files Combined', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # reserve space for title
        
        # save image
        img_path = f"{output_prefix}_combined_distribution.jpg"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        print(f"Combined distribution chart saved: {img_path}")
        plt.close()
    
    return all_class_counts, combined_stats, all_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze class-wise image count in TFRecord files.')
    parser.add_argument('--tfrecord', nargs='+', required=True, 
                       help='Paths to TFRecord files to analyze (at least one)')
    parser.add_argument('--output', type=str, default=None, 
                       help='Prefix for output files (e.g. output -> output_class_stats.csv, output_class_distribution.jpg)')
    
    args = parser.parse_args()
    
    # validate TFRecord file paths
    tfrecord_paths = []
    for path in args.tfrecord:
        if os.path.exists(path):
            tfrecord_paths.append(path)
        else:
            print(f"Warning: File not found: {path}")
    
    if not tfrecord_paths:
        print("Error: No valid TFRecord files found.")
        exit(1)
    
    # set output file prefix
    if args.output is None:
        if len(tfrecord_paths) == 1:
            # use existing method for single file
            base_name = os.path.splitext(os.path.basename(tfrecord_paths[0]))[0]
            output_prefix = f"{base_name}_analysis"
        else:
            # use combined prefix for multiple files
            output_prefix = "combined_analysis"
    else:
        output_prefix = args.output
    
    # run analysis
    if len(tfrecord_paths) == 1:
        # analyze single file (existing method)
        class_counts, stats = analyze_single_tfrecord(tfrecord_paths[0])
        
        # print results for single file
        print(f"\n=== Single File Analysis Results ===")
        print(f"File: {stats['file_name']}")
        print(f"Total images: {stats['total_images']:,}")
        print(f"Total classes: {stats['total_classes']:,}")
        print(f"Min images per class: {stats['min_images_per_class']:,}")
        print(f"Max images per class: {stats['max_images_per_class']:,}")
        print(f"Mean images per class: {stats['mean_images_per_class']:.2f}")
        print(f"Std images per class: {stats['std_images_per_class']:.2f}")
        print(f"Median images per class: {stats['median_images_per_class']:.2f}")
        
        # save CSV file for single file
        csv_path = f"{output_prefix}_class_stats.csv"
        df = pd.DataFrame([
            {
                'class_label': label,
                'image_count': count,
                'percentage': (count / stats['total_images']) * 100
            }
            for label, count in sorted(class_counts.items())
        ])
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\nClass statistics saved: {csv_path}")
        
        # save summary statistics CSV
        summary_path = f"{output_prefix}_summary_stats.csv"
        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        print(f"Summary statistics saved: {summary_path}")
        
        # create and save histogram for single file
        plt.figure(figsize=(14, 10))
        
        class_counts_list = list(class_counts.values())
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        # subplot 1: class-wise image count distribution
        plt.subplot(2, 1, 1)
        plt.hist(class_counts_list, bins=min(50, len(class_counts_list)), alpha=0.7, edgecolor='black')
        plt.title(f'Class Image Count Distribution\nFile: {stats["file_name"]} ({stats["total_images"]:,} total images)')
        plt.xlabel('Number of Images per Class')
        plt.ylabel('Number of Classes')
        plt.grid(True, alpha=0.3)
        
        # add statistics information as text
        stats_text = f'Total Classes: {stats["total_classes"]:,}\n'
        stats_text += f'Min: {stats["min_images_per_class"]:,}\n'
        stats_text += f'Max: {stats["max_images_per_class"]:,}\n'
        stats_text += f'Mean: {stats["mean_images_per_class"]:.1f}\n'
        stats_text += f'Std: {stats["std_images_per_class"]:.1f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # subplot 2: image count of top classes (bar chart)
        plt.subplot(2, 1, 2)
        top_classes = sorted_classes[:min(20, len(sorted_classes))]
        labels, counts = zip(*top_classes)
        bars = plt.bar(range(len(labels)), counts, alpha=0.7, edgecolor='black')
        plt.title(f'Top 20 Classes by Image Count\nFile: {stats["file_name"]}')
        plt.xlabel('Class ID (Top 20)')
        plt.ylabel('Number of Images')
        plt.xticks(range(len(labels)), [str(l) for l in labels], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # show values on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                    f'{count:,}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # add overall title
        plt.suptitle(f'Single TFRecord Analysis - {stats["file_name"]}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # reserve space for title
        
        # save image
        img_path = f"{output_prefix}_class_distribution.jpg"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        print(f"Distribution chart saved: {img_path}")
        plt.close()
        
    else:
        # analyze multiple files
        all_class_counts, combined_stats, all_stats = analyze_multiple_tfrecords(tfrecord_paths, output_prefix)