#!/usr/bin/env python3
"""
1:N face search visualization test script
- additional visualization test functionality based on evaluation.py
- process query and database separately by receiving multiple tfrecord files
- visualize 1:N matching results and save
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from tqdm import tqdm

# add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# import functions from evaluation.py
from evaluation import (
    setup_gpu, load_model_simple, load_test_dataset, extract_embeddings,
    run_tflite_inference, similarity_to_score, is_tflite_model
)

def load_multiple_tfrecord_datasets(tfrecord_files, batch_size=32, input_shape=(112, 112, 3)):
    """load multiple tfrecord files and combine into a single dataset"""
    print(f"📊 loading multiple TFRecord files...")
    print(f"   number of files: {len(tfrecord_files)}")
    
    datasets = []
    for i, tfrecord_file in enumerate(tfrecord_files):
        print(f"   {i+1}. {tfrecord_file}")
        dataset = load_test_dataset(tfrecord_file, batch_size=batch_size, input_shape=input_shape)
        datasets.append(dataset)
    
    # combine all datasets
    combined_dataset = datasets[0]
    for dataset in datasets[1:]:
        combined_dataset = combined_dataset.concatenate(dataset)
    
    print(f"✅ {len(tfrecord_files)} TFRecord files combined")
    return combined_dataset

def extract_embeddings_with_labels(model, dataset, norm=True, progress=False):
    """extract embeddings and labels together (include index information for file separation)"""
    print("🔍 extracting embeddings...")
    
    embeddings = []
    labels = []
    file_indices = []  # track which file each sample came from
    
    # check if TFLite model
    is_tflite = hasattr(model, 'get_input_details')
    
    iterator = tqdm(dataset, desc='extracting embeddings', unit='batch') if progress else dataset
    for batch_x, batch_y in iterator:
        if is_tflite:
            # TFLite model inference
            batch_pred = run_tflite_inference(model, batch_x)
        else:
            # Keras model inference
            if hasattr(model, 'call'):
                # for custom model, call inference mode
                batch_pred = model(batch_x, training=False)
            else:
                # for general Keras model
                batch_pred = model(batch_x)
        
        # apply L2 normalization
        if norm:
            batch_pred = tf.math.l2_normalize(batch_pred, axis=1)
        
        embeddings.append(batch_pred.numpy())
        labels.append(batch_y.numpy())
    
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    
    print(f"✅ embeddings extracted: {embeddings.shape[0]} samples, {embeddings.shape[1]} dimensions")
    return embeddings, labels

def load_images_for_visualization_multiple(tfrecord_files, input_shape=(112, 112, 3), progress=False):
    """load images for visualization from multiple tfrecord files"""
    def parse_tfrecord(example):
        feature_description = {
            'jpeg': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'x1': tf.io.FixedLenFeature((), tf.int64),
            'y1': tf.io.FixedLenFeature((), tf.int64),
            'x2': tf.io.FixedLenFeature((), tf.int64),
            'y2': tf.io.FixedLenFeature((), tf.int64)
        }
        features = tf.io.parse_single_example(example, feature_description)
        image = tf.io.decode_jpeg(features['jpeg'], channels=3)
        label = features['label']
        
        # extract bbox information
        box = [
            tf.cast(features['y1'], tf.float32),
            tf.cast(features['x1'], tf.float32),
            tf.cast(features['y2'], tf.float32),
            tf.cast(features['x2'], tf.float32)
        ]
        return image, label, box

    def load_and_preprocess_image(image, label, box):
        shape = tf.shape(image)
        shape = tf.repeat(shape, [2, 2, 0])
        shape = tf.scatter_nd([[0], [2], [1], [3]], shape, tf.constant([4]))
        box /= tf.cast(shape, tf.float32)
        image = tf.cast(image, tf.float32)
        return image, label, box

    def crop_and_resize(image, label, box):
        cropped = tf.image.crop_and_resize([image], [box], [0], input_shape[:2])[0]
        return cropped, label

    def normalize(image, label):
        return image / 255.0, label

    # combine all tfrecord files and load
    all_datasets = []
    for tfrecord_file in tfrecord_files:
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(crop_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        all_datasets.append(dataset)
    
    # combine all datasets
    combined_dataset = all_datasets[0]
    for dataset in all_datasets[1:]:
        combined_dataset = combined_dataset.concatenate(dataset)
    
    # load all data into memory
    images = []
    labels = []
    iterator = tqdm(combined_dataset, desc='loading images', unit='sample') if progress else combined_dataset
    for image, label in iterator:
        images.append(image.numpy())
        labels.append(label.numpy())
    
    return np.array(images), np.array(labels)

def visualize_1n_face_search(query_images, query_labels, database_images, database_labels, 
                            query_embeddings, database_embeddings, top_k=5, 
                            metric='cos', save_dir='face_search_results', 
                            max_samples=10, score_method='linear'):
    """visualize and save 1:N face search results (same as evaluation.py)"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🎨 1:N face search visualization...")
    print(f"   Query samples: {len(query_images)}")
    print(f"   Database samples: {len(database_images)}")
    print(f"   Top-K: {top_k}")
    print(f"   Save directory: {save_dir}")
    
    # combine query and database to create overall search target
    all_images = np.vstack([query_images, database_images])
    all_labels = np.hstack([query_labels, database_labels])
    all_embeddings = np.vstack([query_embeddings, database_embeddings])
    
    if metric == 'cos':
        # Cosine similarity: inner product of L2 normalized vectors
        def metric_fn(query_emb, all_emb):
            similarities = np.dot(query_emb, all_emb.T)
            return (similarities + 1) / 2  # normalize -1~1 to 0~1
        largest = True
    elif metric == 'l2':
        # calculate L2 distance
        def metric_fn(query_emb, all_emb):
            query_expanded = np.expand_dims(query_emb, axis=1)
            all_expanded = np.expand_dims(all_emb, axis=0)
            dist = np.sqrt(np.sum((query_expanded - all_expanded) ** 2, axis=2))
            return -dist  # convert distance to negative value for better similarity
        largest = True
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # determine number of samples to visualize
    if max_samples <= 0:
        num_samples = len(query_images)
        print(f"   Visualize all Query samples: {num_samples} samples")
    else:
        num_samples = min(max_samples, len(query_images))
        print(f"   Visualize {num_samples} samples")
    
    # visualize 1:N search results for each Query sample
    for i in range(num_samples):
        query_image = query_images[i]
        query_label = query_labels[i]
        query_embedding = query_embeddings[i:i+1]
        
        # calculate similarity with overall search target
        similarities = metric_fn(query_embedding, all_embeddings)[0]
        
        # exclude self (same as evaluation.py)
        similarities[i] = -1.0  # exclude self (i-th sample of Query)
        
        # select Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k] if largest else np.argsort(similarities)[:top_k]
        
        # visualize
        fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 3))
        if top_k == 0:
            axes = [axes]
        
        # Query image
        axes[0].imshow(query_image)
        axes[0].set_title(f'Query\nLabel: {query_label}', fontsize=10, fontweight='bold')
        axes[0].axis('off')
        
        # Top-K matching results
        for j, idx in enumerate(top_indices):
            match_image = all_images[idx]
            match_label = all_labels[idx]
            similarity = similarities[idx]
            score = similarity_to_score(similarity, score_method)
            is_correct = (match_label == query_label)
            
            axes[j+1].imshow(match_image)
            color = 'green' if is_correct else 'red'
            axes[j+1].set_title(f'Rank {j+1}\nLabel: {match_label}\nSim: {similarity:.3f}\nScore: {score:.1f}', 
                               fontsize=10, color=color)
            axes[j+1].axis('off')
            
            # add border if correct
            if is_correct:
                for spine in axes[j+1].spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/query_{i}_label_{query_label}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ 1:N search visualization completed: {save_dir}/ ({num_samples} samples)")

def compute_1n_recall_at_k(query_embeddings, query_labels, database_embeddings, database_labels, 
                          k=1, metric='cos', progress=False):
    """calculate Recall@K in 1:N search (same as evaluation.py)"""
    print(f"📊 calculating 1:N Recall@{k} (metric: {metric})")
    
    # combine query and database to create overall search target
    all_embeddings = np.vstack([query_embeddings, database_embeddings])
    all_labels = np.hstack([query_labels, database_labels])
    
    n_queries = len(query_embeddings)
    correct = 0
    
    iterator = tqdm(range(n_queries), desc=f'1:N Recall@{k}', unit='query') if progress else range(n_queries)
    for i in iterator:
        query_embedding = query_embeddings[i:i+1]
        query_label = query_labels[i]
        
        # calculate distance with overall search target
        if metric == 'cos':
            # cosine similarity (higher is more similar)
            similarities = np.dot(all_embeddings, query_embedding.T).flatten()
        else:
            # Euclidean distance (lower is more similar)
            distances = np.linalg.norm(all_embeddings - query_embedding, axis=1)
            similarities = -distances  # convert distance to similarity
        
        # exclude self (same as evaluation.py)
        similarities[i] = -1.0  # exclude self (i-th sample of Query)
        
        # select top k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # check if correct answer is in top k
        if query_label in all_labels[top_k_indices]:
            correct += 1
    
    recall = correct / n_queries
    print(f"✅ 1:N Recall@{k}: {recall:.4f} ({correct}/{n_queries})")
    return recall

def face_search_visualization_test(model_path, query_files, database_files=None, k_list=[1, 5, 10], 
                                 metric='cos', norm=True, save_dir='face_search_results', 
                                 max_samples=10, score_method='linear', progress=False):
    """main function for 1:N face search visualization test"""
    print(f"\n🎯 1:N face search visualization test started")
    print(f"   Model: {model_path}")
    print(f"   Query files: {query_files}")
    if database_files:
        print(f"   Database files: {database_files}")
    else:
        print(f"   Database files: none (only use Query)")
    print(f"   Metric: {metric}")
    print(f"   L2 normalization: {norm}")
    print(f"   Visualization: {save_dir}/")
    
    # load model
    model = load_model_simple(model_path)
    if model is None:
        return None
    
    # check if TFLite model
    is_tflite = hasattr(model, 'get_input_details')
    batch_size = 1 if is_tflite else 32
    
    # load Query dataset
    print(f"\n📊 loading Query dataset...")
    query_dataset = load_multiple_tfrecord_datasets(query_files, batch_size=batch_size)
    
    # extract Query embeddings
    print(f"\n🔍 extracting Query embeddings...")
    start_time = time.time()
    query_embeddings, query_labels = extract_embeddings_with_labels(model, query_dataset, norm=norm, progress=progress)
    query_extraction_time = time.time() - start_time
    print(f"   Query embeddings extraction time: {query_extraction_time:.2f} seconds")
    
    # process Database (optional)
    database_embeddings = None
    database_labels = None
    database_extraction_time = 0
    
    if database_files:
        # load Database dataset
        print(f"\n📊 loading Database dataset...")
        database_dataset = load_multiple_tfrecord_datasets(database_files, batch_size=batch_size)
        
        # extract Database embeddings
        print(f"\n🔍 extracting Database embeddings...")
        start_time = time.time()
        database_embeddings, database_labels = extract_embeddings_with_labels(model, database_dataset, norm=norm, progress=progress)
        database_extraction_time = time.time() - start_time
        print(f"   Database embeddings extraction time: {database_extraction_time:.2f} seconds")
    else:
        # if Database is not provided, only use Query
        print(f"\n📊 Database not provided - only use Query")
        database_embeddings = query_embeddings
        database_labels = query_labels
    
    # calculate 1:N Recall@K
    results = {}
    for k in k_list:
        start_time = time.time()
        recall = compute_1n_recall_at_k(query_embeddings, query_labels, database_embeddings, database_labels, 
                                      k=k, metric=metric, progress=progress)
        computation_time = time.time() - start_time
        results[f'1n_recall@{k}'] = recall
        print(f"   Calculation time: {computation_time:.2f} seconds")
    
    # load images for visualization
    print(f"\n🎨 loading images for visualization...")
    query_images, _ = load_images_for_visualization_multiple(query_files, progress=progress)
    
    if database_files:
        database_images, _ = load_images_for_visualization_multiple(database_files, progress=progress)
    else:
        database_images = query_images
    
    # visualize 1:N search results
    max_k = max(k_list)
    visualize_1n_face_search(query_images, query_labels, database_images, database_labels,
                            query_embeddings, database_embeddings, top_k=max_k,
                            metric=metric, save_dir=save_dir, max_samples=max_samples, 
                            score_method=score_method)
    
    # summarize results
    print(f"\n📊 1:N search results:")
    for metric_name, value in results.items():
        print(f"   {metric_name}: {value:.4f}")
    print(f"   Query embeddings extraction time: {query_extraction_time:.2f} seconds")
    if database_files:
        print(f"   Database embeddings extraction time: {database_extraction_time:.2f} seconds")
    
    # save results to JSON file
    results['query_extraction_time'] = query_extraction_time
    results['database_extraction_time'] = database_extraction_time
    results['query_files'] = query_files
    results['database_files'] = database_files if database_files else []
    results['model_path'] = model_path
    results['metric'] = metric
    results['norm'] = norm
    
    results_file = os.path.join(save_dir, 'search_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   Results saved to: {results_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='1:N face search visualization test script')
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to the model file (.keras or .tflite)')
    parser.add_argument('--query_files', type=str, nargs='+', required=True,
                        help='Query TFRecord files to search')
    parser.add_argument('--database_files', type=str, nargs='*', default=None,
                        help='Additional search target Database TFRecord files (optional)')
    parser.add_argument('--k_list', type=int, nargs='+', default=[1, 5, 10],
                        help='Recall@K values to calculate (default: [1, 5, 10])')
    parser.add_argument('--metric', type=str, default='cos', choices=['cos', 'euclidean'],
                        help='Distance metric (default: cos)')
    parser.add_argument('--norm', action='store_true', default=True,
                        help='Apply L2 normalization (default: True)')
    parser.add_argument('--save_dir', type=str, default='face_search_results',
                        help='Visualization result save directory (default: face_search_results)')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='Maximum number of samples to visualize (0 or negative means visualize all Query samples, default: 10)')
    parser.add_argument('--score_method', type=str, default='linear', 
                        choices=['linear', 'sqrt', 'log'],
                        help='Method to convert similarity to 0~100 (default: linear)')
    parser.add_argument('--progress', action='store_true',
                        help='Show progress bar with tqdm (recommended for large TFRecord evaluation)')
    
    args = parser.parse_args()
    
    # setup GPU
    setup_gpu()
    
    # 1:N face search visualization test
    results = face_search_visualization_test(
        model_path=args.model_path,
        query_files=args.query_files,
        database_files=args.database_files,
        k_list=args.k_list,
        metric=args.metric,
        norm=args.norm,
        save_dir=args.save_dir,
        max_samples=args.max_samples,
        score_method=args.score_method,
        progress=args.progress
    )
    
    if results is None:
        print("❌ 1:N face search visualization test failed")
        return
    
    print(f"\n✅ 1:N face search visualization test completed!")
    print(f"   결과 저장 위치: {args.save_dir}/")

if __name__ == '__main__':
    main()
