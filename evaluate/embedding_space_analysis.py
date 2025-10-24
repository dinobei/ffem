#!/usr/bin/env python3
"""
Embedding space analysis metrics for face recognition evaluation
- Intra-class and inter-class distances
- Margin calculation
- Embedding variance and consistency
- Spread and distribution analysis
"""

import numpy as np
from tqdm import tqdm


def calculate_all_embedding_metrics(embeddings, labels, progress=False):
    """Calculate all embedding space analysis metrics"""
    print("  📊 calculating embedding space analysis metrics...")
    
    metrics = {}
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    n_samples = len(embeddings)
    
    try:
        # apply sampling for large dataset
        max_samples_per_class = 1000  # maximum samples per class
        use_sampling = n_samples > 5000
        
        if use_sampling:
            print(f"    large dataset detected ({n_samples} samples) - applying sampling")
            sampled_embeddings = []
            sampled_labels = []
            
            for label in unique_labels:
                class_indices = np.where(labels == label)[0]
                class_embeddings = embeddings[class_indices]
                
                if len(class_embeddings) > max_samples_per_class:
                    # random sampling per class
                    sample_indices = np.random.choice(len(class_embeddings), max_samples_per_class, replace=False)
                    sampled_embeddings.append(class_embeddings[sample_indices])
                    sampled_labels.extend([label] * max_samples_per_class)
                else:
                    sampled_embeddings.append(class_embeddings)
                    sampled_labels.extend([label] * len(class_embeddings))
            
            embeddings = np.vstack(sampled_embeddings)
            labels = np.array(sampled_labels)
            print(f"    after sampling: {len(embeddings)} samples")
        
        # group embeddings by class
        class_embeddings = {}
        for label in unique_labels:
            class_embeddings[label] = embeddings[labels == label]
        
        # Intra-class distance (within the same class) - vectorized calculation
        intra_distances = []
        for label, class_emb in class_embeddings.items():
            if len(class_emb) > 1:
                # more efficient vectorized calculation instead of pdist
                if len(class_emb) > 100:
                    # sampling for large classes
                    sample_size = min(100, len(class_emb))
                    sample_indices = np.random.choice(len(class_emb), sample_size, replace=False)
                    class_emb = class_emb[sample_indices]
                
                # calculate cosine distance (1 - cosine similarity)
                similarity_matrix = np.dot(class_emb, class_emb.T)
                # use upper triangular matrix only (avoid duplication)
                mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
                similarities = similarity_matrix[mask]
                distances = 1 - similarities
                intra_distances.extend(distances)
        
        # Inter-class distance (between different classes) - vectorized calculation
        inter_distances = []
        class_pairs = [(i, j) for i in range(n_classes) for j in range(i+1, n_classes)]
        
        iterator = tqdm(class_pairs, desc='Inter-class distance calculation', unit='pair') if progress else class_pairs
        for i, j in iterator:
            label1, label2 = unique_labels[i], unique_labels[j]
            emb1 = class_embeddings[label1]
            emb2 = class_embeddings[label2]
            
            # sampling for large classes
            max_samples = 50
            if len(emb1) > max_samples:
                emb1 = emb1[np.random.choice(len(emb1), max_samples, replace=False)]
            if len(emb2) > max_samples:
                emb2 = emb2[np.random.choice(len(emb2), max_samples, replace=False)]
            
            # vectorized cosine distance calculation
            similarity_matrix = np.dot(emb1, emb2.T)
            distances = 1 - similarity_matrix.flatten()
            inter_distances.extend(distances)
        
        if intra_distances and inter_distances:
            metrics['intra_class_distance_mean'] = np.mean(intra_distances)
            metrics['intra_class_distance_std'] = np.std(intra_distances)
            metrics['inter_class_distance_mean'] = np.mean(inter_distances)
            metrics['inter_class_distance_std'] = np.std(inter_distances)
            
            # calculate margin
            margin = np.mean(inter_distances) - np.mean(intra_distances)
            metrics['margin'] = margin
            
            # calculate margin ratio
            if np.mean(intra_distances) > 0:
                metrics['margin_ratio'] = margin / np.mean(intra_distances)
            else:
                metrics['margin_ratio'] = 0.0
            
            print(f"    Intra-class distance: {np.mean(intra_distances):.4f} ± {np.std(intra_distances):.4f}")
            print(f"    Inter-class distance: {np.mean(inter_distances):.4f} ± {np.std(inter_distances):.4f}")
            print(f"    Margin: {margin:.4f}")
            print(f"    Margin ratio: {metrics['margin_ratio']:.2f}")
        
        # embedding variance
        metrics['embedding_variance'] = np.var(embeddings)
        metrics['embedding_std'] = np.std(embeddings)
        print(f"    embedding variance: {np.var(embeddings):.4f}")
        
        # Additional consistency metrics
        metrics['mean_spread'] = np.mean(intra_distances) if intra_distances else 0.0
        metrics['mean_consistency'] = 1.0 - np.mean(intra_distances) if intra_distances else 0.0
        metrics['norm_consistency'] = np.mean([np.linalg.norm(emb) for emb in embeddings])
        
    except Exception as e:
        print(f"    ⚠️ embedding space analysis failed: {e}")
    
    return metrics
