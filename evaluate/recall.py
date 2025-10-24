import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train.utils import pairwise_distance

import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm


def calc_top_k_label(dist, label, top_k, largest=True):
    if not largest:
        dist = -1 * dist
    
    top_k_label = []
    for x in dist:
        indices = tf.math.top_k(x, top_k)[1]
        top_k_label.append([label[i] for i in indices])

    return top_k_label


def evaluate(model, dataset, metric, top_k: list, batch_size=256, norm=True, dataset_name="dataset"):
    if metric =='cos':
        metric_fn = lambda X, Y: tf.matmul(X, tf.transpose(Y))
        largest = True
    elif metric == 'l2':
        metric_fn = lambda X, Y: pairwise_distance(X, Y)
        largest = False
    else:
        raise 'Unsupported metric.'
    
    # in mixed precision environment, force conversion to float32
    policy = tf.keras.mixed_precision.global_policy()
    
    # first calculate dataset size (for tqdm)
    total_batches = sum(1 for _ in dataset)
    dataset = dataset.repeat()  # make it repeatable

    X = []
    Y = []
    # extract all embeddings in dataset.
    embed_start = time.time()
    batch_count = 0
    
    with tqdm(total=total_batches, desc=f"📊 Extracting embeddings from {dataset_name}", 
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        
        for batch_x, batch_y in dataset:
            if batch_count >= total_batches:
                break
                
            batch_pred = model(batch_x)
            if norm:
                batch_pred = tf.math.l2_normalize(batch_pred, axis=1)
            
            # in mixed precision environment, force conversion to float32
            if policy.name == 'mixed_float16':
                batch_pred = tf.cast(batch_pred, tf.float32)
            
            X.append(batch_pred)
            Y.append(batch_y)
            batch_count += 1
            pbar.update(1)
    
    embed_time = time.time() - embed_start
    X = tf.concat(X, axis=0)
    Y = tf.concat(Y, axis=0)
    ds = tf.data.Dataset.from_tensor_slices(X)
    ds = ds.batch(batch_size)
    P = []
    max_top_k = np.max(top_k)
    
    # calculate number of batches for distance calculation
    total_dist_batches = sum(1 for _ in ds)
    ds = ds.repeat()  # make it repeatable
    
    dist_start = time.time()
    # calculate top_k using batched sample for memory efficiency.
    with tqdm(total=total_dist_batches, desc=f"📊 Computing distances for {dataset_name}", 
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        
        batch_idx = 0
        for batch_x in ds:
            if batch_idx >= total_dist_batches:
                break
                
            dist = metric_fn(batch_x, X)
            # remove self distance.
            max_dist = float('-inf') if largest else float('inf')
            max_dist = tf.fill(batch_x.shape[0], max_dist)
            
            # for mixed precision compatibility, match dtype
            max_dist = tf.cast(max_dist, dist.dtype)
            dist = tf.linalg.set_diag(dist, max_dist, k=batch_idx*batch_x.shape[0])
            # get top_k label
            batch_top_k_label = calc_top_k_label(dist, Y, max_top_k, largest)
            # merge list
            P = P + batch_top_k_label
            batch_idx += 1
            pbar.update(1)
    
    dist_time = time.time() - dist_start

    top_k_results = []
    for k in top_k:
        s = sum([1 for y, p in zip(Y, P) if y in p[:k]])
        top_k_results.append(s / len(Y))

    return top_k_results
