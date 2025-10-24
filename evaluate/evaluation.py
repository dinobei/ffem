#!/usr/bin/env python3
"""
Evaluation script
- Memory-safe recall computation (multi-K in single pass, no NxN matrices)
- Optional accuracy metrics fully skippable or sampled
- EER computation integrated (threshold_mode=eer)
- YAML configuration support
"""

# python evaluate/evaluation.py --model_path checkpoints/ResNet50_adaface_251015_2/best_inference_18.tflite --test_file /notebooks/datasets/face/AFD/AFDB_dataset_160_part1.tfrecord --k_list 1 5 --metric cos --comprehensive True --visualize True --save_dir aaa_251015_2_epoch18_AFD
# python evaluate/evaluation.py --config evaluate/config.yaml

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, roc_curve, auc

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))


def setup_gpu() -> None:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU setup completed: {len(gpus)} GPUs available")
    else:
        print("⚠️ GPU not found. Running on CPU.")


def is_tflite_model(model_path: str) -> bool:
    return model_path.endswith('.tflite')


def load_keras_model(model_path: str):
    custom_objects = {}
    try:
        from train.custom_models.angular_margin_model import AngularMarginModel
        custom_objects['AngularMarginModel'] = AngularMarginModel
        print("✅ AngularMarginModel registered")
    except Exception as e:
        print(f"⚠️ AngularMarginModel registration failed: {e}")
    try:
        from train.custom_models.cosface_model import CosFaceModel
        custom_objects['CosFaceModel'] = CosFaceModel
        print("✅ CosFaceModel registered")
    except Exception as e:
        print(f"⚠️ CosFaceModel registration failed: {e}")
    try:
        from train.custom_models.adaface_model import AdaFaceModel
        custom_objects['AdaFaceModel'] = AdaFaceModel
        print("✅ AdaFaceModel registered")
    except Exception as e:
        print(f"⚠️ AdaFaceModel registration failed: {e}")
    try:
        from train.layers.norm_aware_pooling_layer import NormAwarePoolingLayer
        custom_objects['NormAwarePoolingLayer'] = NormAwarePoolingLayer
        print("✅ NormAwarePoolingLayer registered")
    except Exception as e:
        print(f"⚠️ NormAwarePoolingLayer registration failed: {e}")
    try:
        from train.layers.angular_margin_layer import AngularMarginLayer
        custom_objects['AngularMarginLayer'] = AngularMarginLayer
        print("✅ AngularMarginLayer registered")
    except Exception as e:
        print(f"⚠️ AngularMarginLayer registration failed: {e}")
    try:
        from train.layers.cosface_layer import CosFaceLayer
        custom_objects['CosFaceLayer'] = CosFaceLayer
        print("✅ CosFaceLayer registered")
    except Exception as e:
        print(f"⚠️ CosFaceLayer registration failed: {e}")
    try:
        from train.layers.adaface_layer import AdaFaceLayer
        custom_objects['AdaFaceLayer'] = AdaFaceLayer
        print("✅ AdaFaceLayer registered")
    except Exception as e:
        print(f"⚠️ AdaFaceLayer registration failed: {e}")

    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False, safe_mode=False)
        print("✅ Keras model loading completed (method 1)")
        return model
    except Exception as e1:
        print(f"⚠️ method 1 failed: {e1}")
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False, safe_mode=True)
            print("✅ Keras model loading completed (method 2)")
            return model
        except Exception as e2:
            print(f"⚠️ method 2 failed: {e2}")
            try:
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=True, safe_mode=False)
                print("✅ Keras model loading completed (method 3)")
                return model
            except Exception as e3:
                print(f"❌ all methods failed:\n  method 1: {e1}\n  method 2: {e2}\n  method 3: {e3}")
                return None


def load_tflite_model(model_path: str):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("✅ TFLite model loading completed")
        return interpreter
    except Exception as e:
        print(f"❌ TFLite model loading failed: {e}")
        return None


def load_model_simple(model_path: str):
    print(f"📥 Loading model: {model_path}")
    if is_tflite_model(model_path):
        return load_tflite_model(model_path)
    return load_keras_model(model_path)


def run_tflite_inference(interpreter, input_data: tf.Tensor) -> tf.Tensor:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data.numpy())
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return tf.constant(output_data)


def load_test_dataset(test_file: str, batch_size: int = 32, input_shape: Tuple[int, int, int] = (112, 112, 3)):
    print(f"📊 Test dataset loading: {test_file}")

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

    dataset = tf.data.TFRecordDataset(test_file)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(crop_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def extract_embeddings(model, dataset, norm: bool = True, progress: bool = False, tta_flip: bool = False, tta_avg: bool = True):
    print("🔍 Extracting embeddings...")
    if tta_flip:
        print("🔄 Using Test Time Augmentation (TTA) with horizontal flip")

    embeddings: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    is_tflite = hasattr(model, 'get_input_details')
    iterator = tqdm(dataset, desc='Extracting embeddings', unit='batch') if progress else dataset
    for batch_x, batch_y in iterator:
        batch_embeddings: List[np.ndarray] = []
        if is_tflite:
            batch_pred = run_tflite_inference(model, batch_x)
        else:
            if hasattr(model, 'call'):
                batch_pred = model(batch_x, training=False)
            else:
                batch_pred = model(batch_x)
        if norm:
            batch_pred = tf.math.l2_normalize(batch_pred, axis=1)
        batch_embeddings.append(batch_pred.numpy())

        if tta_flip:
            batch_x_flipped = tf.image.flip_left_right(batch_x)
            if is_tflite:
                batch_pred_flipped = run_tflite_inference(model, batch_x_flipped)
            else:
                if hasattr(model, 'call'):
                    batch_pred_flipped = model(batch_x_flipped, training=False)
                else:
                    batch_pred_flipped = model(batch_x_flipped)
            if norm:
                batch_pred_flipped = tf.math.l2_normalize(batch_pred_flipped, axis=1)
            batch_embeddings.append(batch_pred_flipped.numpy())

        if tta_flip and tta_avg:
            batch_pred_avg = np.mean(batch_embeddings, axis=0)
            embeddings.append(batch_pred_avg)
        else:
            for embedding in batch_embeddings:
                embeddings.append(embedding)
        labels.append(batch_y.numpy())

    embeddings_np = np.vstack(embeddings)
    labels_np = np.hstack(labels)
    print(f"✅ Embedding extraction completed: {embeddings_np.shape[0]} samples, {embeddings_np.shape[1]} dimensions")
    return embeddings_np, labels_np


# ==================== Streaming multi-K Recall (no NxN similarity matrix) ====================

def compute_recall_at_ks_streaming(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: List[int],
    metric: str = 'cos',
    progress: bool = False
) -> Dict[str, float]:
    print(f"📊 Computing Recall@{k_values} (metric: {metric})")
    n_samples = embeddings.shape[0]
    k_values_sorted = sorted(set(int(k) for k in k_values))
    max_k = max(k_values_sorted)

    # Precompute norms if needed
    if metric == 'euclidean':
        emb_sq_norms = np.sum(embeddings * embeddings, axis=1)
    else:
        emb_sq_norms = None

    correct_counts = {k: 0 for k in k_values_sorted}
    iterator = tqdm(range(n_samples), desc='Recall multi-K', unit='sample') if progress else range(n_samples)

    for i in iterator:
        query = embeddings[i]
        true_label = labels[i]

        if metric == 'cos':
            sims = embeddings @ query
            sims[i] = -np.inf  # exclude self
            # top-K via argpartition for efficiency
            if max_k < len(sims) - 1:
                top_idx = np.argpartition(sims, -max_k)[-max_k:]
                # order these top indices
                top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            else:
                top_idx = np.argsort(sims)[::-1]
        else:
            # Euclidean: use ||x||^2 + ||q||^2 - 2 x·q
            if emb_sq_norms is None:
                emb_sq_norms = np.sum(embeddings * embeddings, axis=1)
            dot = embeddings @ query
            q_norm2 = float(np.dot(query, query))
            dists2 = emb_sq_norms + q_norm2 - 2.0 * dot
            dists2[i] = np.inf  # exclude self
            if max_k < len(dists2) - 1:
                top_idx = np.argpartition(dists2, max_k)[:max_k]
                # order these top indices (ascending distance)
                top_idx = top_idx[np.argsort(dists2[top_idx])]
            else:
                top_idx = np.argsort(dists2)

        # Update recalls for all K using the same top list
        for k in k_values_sorted:
            k_idx = top_idx[:k]
            if true_label in labels[k_idx]:
                correct_counts[k] += 1

    recalls = {f'recall@{k}': correct_counts[k] / n_samples for k in k_values_sorted}
    for k in k_values_sorted:
        print(f"✅ Recall@{k}: {recalls[f'recall@{k}']:.4f} ({correct_counts[k]}/{n_samples})")
    return recalls


# ==================== Visualization helpers ====================

def similarity_to_score(similarity: np.ndarray, method: str = 'linear') -> np.ndarray:
    if method == 'linear':
        return similarity * 100.0
    elif method == 'sqrt':
        return (np.sqrt(np.maximum(similarity, 0.0))) * 100.0
    elif method == 'log':
        return (np.log(similarity + 0.01) / np.log(1.01)) * 100.0
    else:
        raise ValueError(f"Unsupported transformation method: {method}")


def load_images_for_visualization_subset(
    test_file: str,
    input_shape: Tuple[int, int, int] = (112, 112, 3),
    max_samples: int = 10,
    progress: bool = False
):
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

    dataset = tf.data.TFRecordDataset(test_file)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(crop_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    iterator = tqdm(dataset, desc='Loading images for visualization', unit='sample') if progress else dataset
    for image, label in iterator:
        images.append(image.numpy())
        labels.append(label.numpy())
        if len(images) >= max_samples:
            break

    return np.array(images), np.array(labels)


def visualize_top_k_matches(
    images: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    top_k: int,
    metric: str = 'cos',
    save_dir: str = 'evaluation_results',
    score_method: str = 'linear'
):
    os.makedirs(save_dir, exist_ok=True)

    if metric == 'cos':
        def metric_fn(X, Y):
            similarities = np.dot(X, Y.T)
            return (similarities + 1.0) / 2.0
        largest = True
    elif metric == 'euclidean':
        X_expanded = np.expand_dims(embeddings, axis=1)
        Y_expanded = np.expand_dims(embeddings, axis=0)
        dist = np.sqrt(np.sum((X_expanded - Y_expanded) ** 2, axis=2))
        metric_fn = lambda X, Y: -dist
        largest = True
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    num_samples = min(len(images), embeddings.shape[0])
    for i in range(num_samples):
        query_image = images[i]
        query_label = labels[i]
        query_embedding = embeddings[i:i+1]

        similarities = metric_fn(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][1:top_k+1] if largest else np.argsort(similarities)[:top_k]

        fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 3))
        if top_k == 0:
            axes = [axes]

        axes[0].imshow(query_image)
        axes[0].set_title(f'Query\nLabel: {query_label}', fontsize=10)
        axes[0].axis('off')

        for j, idx in enumerate(top_indices):
            match_image = images[idx]
            match_label = labels[idx]
            similarity = similarities[idx]
            score = similarity_to_score(np.array([similarity]), score_method)[0]
            is_correct = (match_label == query_label)

            axes[j+1].imshow(match_image)
            color = 'green' if is_correct else 'red'
            axes[j+1].set_title(
                f'Rank {j+1}\nLabel: {match_label}\nSim: {similarity:.3f}\nScore: {score:.1f}',
                fontsize=10, color=color
            )
            axes[j+1].axis('off')

            if is_correct:
                for spine in axes[j+1].spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(3)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/query_{i}_label_{query_label}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✅ Visualization results saved: {save_dir}/ ({num_samples} samples)")


# ==================== Comprehensive metrics (with options to skip accuracy) ====================

def calculate_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    print("  🎯 calculating clustering quality metrics...")
    metrics: Dict[str, float] = {}
    try:
        if len(embeddings) > 10000:
            indices = np.random.choice(len(embeddings), 10000, replace=False)
            sample_embeddings = embeddings[indices]
            sample_labels = labels[indices]
        else:
            sample_embeddings = embeddings
            sample_labels = labels
        silhouette = silhouette_score(sample_embeddings, sample_labels, metric='cosine')
        metrics['silhouette_score'] = float(silhouette)
        print(f"    Silhouette Score: {silhouette:.4f}")
        db_index = davies_bouldin_score(sample_embeddings, sample_labels)
        metrics['davies_bouldin_index'] = float(db_index)
        print(f"    Davies-Bouldin Index: {db_index:.4f}")
        ch_index = calinski_harabasz_score(sample_embeddings, sample_labels)
        metrics['calinski_harabasz_index'] = float(ch_index)
        print(f"    Calinski-Harabasz Index: {ch_index:.2f}")
    except Exception as e:
        print(f"    ⚠️ clustering metrics calculation failed: {e}")
    return metrics


def calculate_embedding_space_metrics(embeddings: np.ndarray, labels: np.ndarray, progress: bool = False) -> Dict[str, float]:
    print("  📊 calculating embedding space analysis metrics...")
    metrics: Dict[str, float] = {}
    unique_labels = np.unique(labels)
    n_samples = len(embeddings)
    try:
        max_samples_per_class = 1000
        use_sampling = n_samples > 5000
        if use_sampling:
            print(f"    large dataset detected ({n_samples} samples) - applying sampling")
            sampled_embeddings = []
            sampled_labels = []
            for label in unique_labels:
                class_indices = np.where(labels == label)[0]
                class_embeddings = embeddings[class_indices]
                if len(class_embeddings) > max_samples_per_class:
                    sample_indices = np.random.choice(len(class_embeddings), max_samples_per_class, replace=False)
                    sampled_embeddings.append(class_embeddings[sample_indices])
                    sampled_labels.extend([label] * max_samples_per_class)
                else:
                    sampled_embeddings.append(class_embeddings)
                    sampled_labels.extend([label] * len(class_embeddings))
            embeddings = np.vstack(sampled_embeddings)
            labels = np.array(sampled_labels)
            print(f"    after sampling: {len(embeddings)} samples")

        class_embeddings: Dict[int, np.ndarray] = {}
        for label in unique_labels:
            class_embeddings[int(label)] = embeddings[labels == label]

        intra_distances: List[float] = []
        for label, class_emb in class_embeddings.items():
            if len(class_emb) > 1:
                if len(class_emb) > 100:
                    sample_size = min(100, len(class_emb))
                    sample_indices = np.random.choice(len(class_emb), sample_size, replace=False)
                    class_emb = class_emb[sample_indices]
                similarity_matrix = np.dot(class_emb, class_emb.T)
                mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
                similarities = similarity_matrix[mask]
                distances = 1 - similarities
                intra_distances.extend(distances.tolist())

        inter_distances: List[float] = []
        class_pairs = [(i, j) for i in range(len(unique_labels)) for j in range(i + 1, len(unique_labels))]
        iterator = tqdm(class_pairs, desc='Inter-class distance calculation', unit='pair') if progress else class_pairs
        for i, j in iterator:
            label1, label2 = unique_labels[i], unique_labels[j]
            emb1 = class_embeddings[int(label1)]
            emb2 = class_embeddings[int(label2)]
            max_samples = 50
            if len(emb1) > max_samples:
                emb1 = emb1[np.random.choice(len(emb1), max_samples, replace=False)]
            if len(emb2) > max_samples:
                emb2 = emb2[np.random.choice(len(emb2), max_samples, replace=False)]
            similarity_matrix = np.dot(emb1, emb2.T)
            distances = 1 - similarity_matrix.flatten()
            inter_distances.extend(distances.tolist())

        if intra_distances and inter_distances:
            intra_mean = float(np.mean(intra_distances))
            intra_std = float(np.std(intra_distances))
            inter_mean = float(np.mean(inter_distances))
            inter_std = float(np.std(inter_distances))
            margin = inter_mean - intra_mean
            metrics['intra_class_distance_mean'] = intra_mean
            metrics['intra_class_distance_std'] = intra_std
            metrics['inter_class_distance_mean'] = inter_mean
            metrics['inter_class_distance_std'] = inter_std
            metrics['margin'] = margin
            print(f"    Intra-class distance: {intra_mean:.4f} ± {intra_std:.4f}")
            print(f"    Inter-class distance: {inter_mean:.4f} ± {inter_std:.4f}")
            print(f"    Margin: {margin:.4f}")
        metrics['embedding_variance'] = float(np.var(embeddings))
        metrics['embedding_std'] = float(np.std(embeddings))
        print(f"    embedding variance: {metrics['embedding_variance']:.4f}")
    except Exception as e:
        print(f"    ⚠️ embedding space analysis failed: {e}")
    return metrics


# ==================== EER computation (sampling-friendly) ====================

def _cos_to_deg(cos_val: np.ndarray) -> np.ndarray:
    cos_clamped = np.clip(cos_val, -1.0, 1.0)
    return np.degrees(np.arccos(cos_clamped))


def compute_eer_threshold_from_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    use_sampling: bool = True,
    sample_ratio: float = 1.0,
    max_genuine_pairs_per_class: int = 5000,
    progress: bool = False
) -> Tuple[float, Dict[str, float]]:
    print("  📐 computing EER (sampling-friendly)")

    # Build class -> indices map
    id_to_indices: Dict[int, List[int]] = {}
    for idx, l in enumerate(labels.tolist()):
        id_to_indices.setdefault(int(l), []).append(idx)

    # Genuine scores (angles)
    genuine_scores: List[float] = []
    for cls, idxs in id_to_indices.items():
        if len(idxs) < 2:
            continue
        # Estimate number of pairs; sample if too many
        num_pairs = len(idxs) * (len(idxs) - 1) // 2
        if use_sampling and num_pairs > max_genuine_pairs_per_class:
            # Random pair sampling (with replacement acceptable)
            num_samples = max_genuine_pairs_per_class
            for _ in (tqdm(range(num_samples), desc=f"genuine@{cls}") if progress else range(num_samples)):
                i, j = np.random.choice(idxs, 2, replace=False)
                cos_val = float(np.dot(embeddings[i], embeddings[j]))
                genuine_scores.append(float(_cos_to_deg(np.array([cos_val]))[0]))
        else:
            # Exhaustive combinations (may be heavy for large classes)
            for a_i in range(len(idxs)):
                va = embeddings[idxs[a_i]]
                for a_j in range(a_i + 1, len(idxs)):
                    vb = embeddings[idxs[a_j]]
                    cos_val = float(np.dot(va, vb))
                    genuine_scores.append(float(_cos_to_deg(np.array([cos_val]))[0]))

    genuine_scores_np = np.array(genuine_scores, dtype=np.float32)

    # Impostor scores (angles)
    keys = list(id_to_indices.keys())
    impostor_scores: List[float] = []
    if use_sampling:
        num_samples = max(1, int(len(genuine_scores_np) * sample_ratio))
        iterator = tqdm(range(num_samples), desc="Impostor sampling") if progress else range(num_samples)
        for _ in iterator:
            a, b = np.random.choice(keys, 2, replace=False)
            i = np.random.choice(id_to_indices[a])
            j = np.random.choice(id_to_indices[b])
            cos_val = float(np.dot(embeddings[i], embeddings[j]))
            impostor_scores.append(float(_cos_to_deg(np.array([cos_val]))[0]))
    else:
        # Exhaustive impostor (dangerous for large datasets)
        for ia, a in enumerate(keys):
            for b in keys[ia + 1:]:
                for i in id_to_indices[a]:
                    va = embeddings[i]
                    vb_idx = id_to_indices[b]
                    dots = va @ embeddings[vb_idx].T
                    impostor_scores.extend(_cos_to_deg(dots).tolist())

    impostor_scores_np = np.array(impostor_scores, dtype=np.float32)

    # ROC/EER on angle domain (lower = more genuine)
    y_true = np.concatenate([
        np.ones_like(genuine_scores_np, dtype=np.int32),
        np.zeros_like(impostor_scores_np, dtype=np.int32)
    ])
    y_score = np.concatenate([-genuine_scores_np, -impostor_scores_np])  # higher better for genuine
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_sc = float(auc(fpr, tpr))
    eer_idx = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    eer_thr_angle = float(-thresholds[eer_idx])  # degrees
    eer_value = float(fpr[eer_idx])

    # Convert angle threshold → cosine similarity threshold
    eer_thr_cos = float(np.cos(np.deg2rad(eer_thr_angle)))
    print(f"    EER={eer_value*100:.2f}%  thr_angle={eer_thr_angle:.4f}°  thr_cos={eer_thr_cos:.5f}  AUC={auc_sc:.4f}")

    stats = {
        'eer': eer_value,
        'eer_threshold_angle_deg': eer_thr_angle,
        'eer_threshold_cos': eer_thr_cos,
        'auc': auc_sc
    }
    return eer_thr_cos, stats


def calculate_security_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.8,
    progress: bool = False,
    max_samples: int = 3000
) -> Dict[str, float]:
    print("  🛡️ calculating security metrics...")
    metrics: Dict[str, float] = {}
    try:
        n_samples = len(embeddings)
        use_sampling = n_samples > max_samples
        if use_sampling:
            print(f"    large dataset detected ({n_samples} samples) - applying sampling")
            sample_indices = np.random.choice(n_samples, max_samples, replace=False)
            embeddings = embeddings[sample_indices]
            labels = labels[sample_indices]
            print(f"    after sampling: {len(embeddings)} samples")

        similarity_matrix = embeddings @ embeddings.T
        np.fill_diagonal(similarity_matrix, -1.0)

        false_positives = 0
        false_negatives = 0
        true_positives = 0
        true_negatives = 0

        iterator = tqdm(range(len(embeddings)), desc='security metrics calculation', unit='sample') if progress else range(len(embeddings))
        for i in iterator:
            sims = similarity_matrix[i]
            true_label = labels[i]
            for j, sim in enumerate(sims):
                if i == j:
                    continue
                is_same = (labels[j] == true_label)
                is_above = (sim >= threshold)
                if is_same and is_above:
                    true_positives += 1
                elif is_same and not is_above:
                    false_negatives += 1
                elif not is_same and is_above:
                    false_positives += 1
                else:
                    true_negatives += 1

        fp = float(false_positives)
        fn = float(false_negatives)
        tp = float(true_positives)
        tn = float(true_negatives)
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['security_score'] = 1 - (metrics['false_positive_rate'] + metrics['false_negative_rate']) / 2
        print(f"    FPR={metrics['false_positive_rate']:.4f}  FNR={metrics['false_negative_rate']:.4f}  TPR={metrics['true_positive_rate']:.4f}  Security={metrics['security_score']:.4f}")
    except Exception as e:
        print(f"    ⚠️ security metrics calculation failed: {e}")
    return metrics


# ==================== Comprehensive driver ====================

def comprehensive_evaluation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    include_security: bool = True,
    progress: bool = False,
    comprehensive_skip_accuracy: bool = True,
    threshold_mode: str = 'fixed',
    threshold: float = 0.8,
    eer_sample_ratio: float = 1.0,
    eer_use_sampling: bool = True
) -> Dict[str, float]:
    print("\n🚀 comprehensive evaluation")
    print("=" * 60)
    all_metrics: Dict[str, float] = {}

    # 1. clustering quality
    all_metrics.update(calculate_clustering_metrics(embeddings, labels))

    # 2. embedding space metrics
    all_metrics.update(calculate_embedding_space_metrics(embeddings, labels, progress=progress))

    # 3. accuracy metrics (optional, default skip for large datasets)
    if not comprehensive_skip_accuracy:
        print("  ⚠️ accuracy metrics are disabled by default in evaluation to avoid NxN memory usage.")
        print("  👉 Use recall-only evaluation, or implement chunked accuracy if strictly needed.")

    # 4. security metrics (with threshold from EER if requested)
    effective_threshold = float(threshold)
    eer_stats: Dict[str, float] = {}
    if include_security:
        if threshold_mode == 'eer':
            effective_threshold, eer_stats = compute_eer_threshold_from_embeddings(
                embeddings, labels, use_sampling=eer_use_sampling, sample_ratio=eer_sample_ratio, progress=progress
            )
            all_metrics.update({f"eer_{k}": v for k, v in eer_stats.items()})
        all_metrics.update(calculate_security_metrics(embeddings, labels, threshold=effective_threshold, progress=progress))

    # Summary (brief)
    print("\n" + "=" * 60)
    print("📊 evaluation results summary")
    print("=" * 60)
    if 'silhouette_score' in all_metrics:
        silhouette = all_metrics['silhouette_score']
        if silhouette > 0.5:
            print(f"🎯 clustering quality: excellent (Silhouette: {silhouette:.3f})")
        elif silhouette > 0.3:
            print(f"🎯 clustering quality: good (Silhouette: {silhouette:.3f})")
        else:
            print(f"🎯 clustering quality: improvement needed (Silhouette: {silhouette:.3f})")
    if 'margin' in all_metrics:
        margin = all_metrics['margin']
        if margin > 0.3:
            print(f"📊 embedding space: excellent separation (margin: {margin:.3f})")
        elif margin > 0.1:
            print(f"📊 embedding space: appropriate separation (margin: {margin:.3f})")
        else:
            print(f"📊 embedding space: improvement needed (margin: {margin:.3f})")
    if 'security_score' in all_metrics:
        security = all_metrics['security_score']
        if security > 0.9:
            print(f"🛡️ security performance: excellent (security score: {security:.3f})")
        elif security > 0.8:
            print(f"🛡️ security performance: good (security score: {security:.3f})")
        else:
            print(f"🛡️ security performance: improvement needed (security score: {security:.3f})")

    return all_metrics


# ==================== YAML config merge ====================

def load_and_merge_config(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, 'config', None) and yaml is not None and os.path.isfile(args.config):
        print(f"📄 Loading config from: {args.config}")
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        print(f"📄 Config loaded: {list(cfg.keys())}")
        
        # Merge simple flat keys; CLI overrides YAML
        for k, v in cfg.items():
            if v is not None:
                old_value = getattr(args, k, None)
                setattr(args, k, v)
                print(f"📄 {k}: {old_value} -> {v}")
    elif getattr(args, 'config', None) and yaml is None:
        print("⚠️ PyYAML not available; --config ignored")
    return args


# ==================== Driver ====================

def evaluate_model(
    model_path: str,
    test_file: str,
    k_list: List[int],
    metric: str = 'cos',
    norm: bool = True,
    visualize: bool = False,
    save_dir: str = 'evaluation_results',
    max_samples: int = 10,
    score_method: str = 'linear',
    comprehensive: bool = False,
    include_security: bool = True,
    threshold: float = 0.8,
    threshold_mode: str = 'fixed',
    progress: bool = False,
    tta_flip: bool = False,
    tta_avg: bool = True,
    comprehensive_skip_accuracy: bool = True,
    eer_sample_ratio: float = 1.0,
    eer_use_sampling: bool = True
):
    print(f"\n🎯 model evaluation started")
    print(f"   model: {model_path}")
    print(f"   test data: {test_file}")
    print(f"   metric: {metric}")
    print(f"   L2 normalization: {norm}")
    if tta_flip:
        print(f"   TTA flip: {tta_flip}, TTA average: {tta_avg}")
    if visualize:
        print(f"   visualization: {save_dir}/")

    model = load_model_simple(model_path)
    if model is None:
        return None

    print(f"\n📋 model information:")
    is_tflite = hasattr(model, 'get_input_details')
    if is_tflite:
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        print(f"   model type: TFLite")
        print(f"   input shape: {input_details[0]['shape']}")
        print(f"   input type: {input_details[0]['dtype']}")
        print(f"   output shape: {output_details[0]['shape']}")
        print(f"   output type: {output_details[0]['dtype']}")
    else:
        print(f"   model type: Keras")
        if hasattr(model, 'input_shape'):
            print(f"   input shape: {model.input_shape}")
        elif hasattr(model, 'backbone') and model.backbone is not None:
            print(f"   input shape: {model.backbone.input_shape}")
        else:
            print(f"   input shape: Unknown")
        try:
            dummy_input = tf.random.normal([1, 112, 112, 3])
            dummy_output = model(dummy_input, training=False)
            print(f"   output shape: {dummy_output.shape}")
        except Exception as e:
            print(f"   output shape: check failed ({e})")
        print(f"   total parameters: {model.count_params():,}")

    batch_size = 1 if is_tflite else 32
    dataset = load_test_dataset(test_file, batch_size=batch_size)

    start_time = time.time()
    embeddings, labels = extract_embeddings(
        model, dataset, norm=norm, progress=progress, tta_flip=tta_flip, tta_avg=tta_avg
    )
    extraction_time = time.time() - start_time

    # Recall multi-K in single pass
    start_time = time.time()
    recall_results = compute_recall_at_ks_streaming(embeddings, labels, k_list, metric=metric, progress=progress)
    recall_time = time.time() - start_time
    print(f"   recall computation time: {recall_time:.2f} seconds")

    results: Dict[str, float] = {}
    results.update(recall_results)

    if comprehensive:
        comp_metrics = comprehensive_evaluation(
            embeddings,
            labels,
            include_security=include_security,
            progress=progress,
            comprehensive_skip_accuracy=comprehensive_skip_accuracy,
            threshold_mode=threshold_mode,
            threshold=threshold,
            eer_sample_ratio=eer_sample_ratio,
            eer_use_sampling=eer_use_sampling,
        )
        results.update(comp_metrics)

    # Visualization (subset-based to avoid OOM on large datasets)
    if visualize:
        try:
            images_subset, labels_subset = load_images_for_visualization_subset(
                test_file, input_shape=(112, 112, 3), max_samples=max_samples, progress=progress
            )
            num_subset = len(images_subset)
            if num_subset > 0:
                embeddings_subset = embeddings[:num_subset]
                labels_for_viz = labels[:num_subset]
                max_k = max(k_list)
                visualize_top_k_matches(
                    images_subset, labels_for_viz, embeddings_subset,
                    top_k=max_k, metric=metric, save_dir=save_dir, score_method=score_method
                )
        except Exception as e:
            print(f"⚠️ visualization failed: {e}")

    print(f"\n📊 evaluation results:")
    for metric_name, value in results.items():
        if isinstance(value, float):
            print(f"   {metric_name}: {value:.4f}")
        else:
            print(f"   {metric_name}: {value}")
    print(f"   embedding extraction time: {extraction_time:.2f} seconds")
    print(f"   recall computation time: {recall_time:.2f} seconds")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='evaluation script')
    p.add_argument('--model_path', type=str, default=None, help='model path (.keras or .tflite file)')
    p.add_argument('--test_file', type=str, default=None, help='test data file (.tfrecord)')
    p.add_argument('--k_list', type=int, nargs='+', default=[1], help='Recall@K values (default: [1])')
    p.add_argument('--metric', type=str, default='cos', choices=['cos', 'euclidean'], help='distance metric (default: cos)')
    p.add_argument('--norm', type=bool, default=True, help='L2 normalization applied (default: True)')
    p.add_argument('--visualize', type=bool, default=False, help='Top-K matching result visualization saved (default: False)')
    p.add_argument('--save_dir', type=str, default='evaluation_results', help='visualization save directory')
    p.add_argument('--max_samples', type=int, default=10, help='maximum number of samples to visualize')
    p.add_argument('--score_method', type=str, default='linear', choices=['linear', 'sqrt', 'log'], help='similarity→score method')    
    p.add_argument('--include_security', type=bool, default=True, help='include security metrics (default: True)')
    p.add_argument('--threshold', type=float, default=0.8, help='security threshold (when threshold_mode=fixed)')
    p.add_argument('--progress', type=bool, default=True, help='tqdm progress bar (default: True)')

    p.add_argument('--config', type=str, default='', help='YAML config path (CLI overrides YAML)')
    
    p.add_argument('--comprehensive', type=bool, default=False, help='comprehensive evaluation (default: False)')
    p.add_argument('--comprehensive_skip_accuracy', type=bool, default=False, help='skip accuracy metrics inside comprehensive (default: False)')
    p.add_argument('--threshold_mode', type=str, default='fixed', choices=['fixed', 'eer'], help='security threshold source')
    p.add_argument('--eer_sample_ratio', type=float, default=1.0, help='impostor sampling ratio relative to genuine count')
    p.add_argument('--eer_use_sampling', type=bool, default=True, help='use sampling for EER computation (default: True)')
    p.add_argument('--tta_flip', type=bool, default=False, help='TTA horizontal flip (default: False)')
    p.add_argument('--tta_avg', type=bool, default=False, help='TTA average when flip is used (default: False)')
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args = load_and_merge_config(args)

    # Check required parameters
    if not args.model_path:
        print("❌ Error: --model_path is required")
        return
    if not args.test_file:
        print("❌ Error: --test_file is required")
        return

    setup_gpu()

    results = evaluate_model(
        model_path=args.model_path,
        test_file=args.test_file,
        k_list=args.k_list,
        metric=args.metric,
        norm=args.norm,
        visualize=args.visualize,
        save_dir=args.save_dir,
        max_samples=args.max_samples,
        score_method=args.score_method,
        comprehensive=args.comprehensive,
        include_security=args.include_security,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        progress=args.progress,
        tta_flip=args.tta_flip,
        tta_avg=args.tta_avg,
        comprehensive_skip_accuracy=args.comprehensive_skip_accuracy,
        eer_sample_ratio=args.eer_sample_ratio,
        eer_use_sampling=args.eer_use_sampling,
    )

    if results is None:
        print("❌ model evaluation failed")
        return
    if is_tflite_model(args.model_path):
        print(f"\n✅ TFLite model evaluation completed!")
    else:
        print(f"\n✅ Keras model evaluation completed!")


if __name__ == '__main__':
    main()


