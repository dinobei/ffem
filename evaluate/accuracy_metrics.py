#!/usr/bin/env python3
"""
Accuracy metrics for face recognition evaluation
- Precision@K
- Recall@K  
- Mean Average Precision (MAP)
"""

import numpy as np
from tqdm import tqdm


def calculate_all_accuracy_metrics(embeddings, labels, k_values=[1, 5, 10, 20], progress=False):
    """Calculate all accuracy metrics"""
    print("  📈 calculating accuracy metrics...")
    
    metrics = {}
    
    try:
        # accuracy metrics are calculated without sampling (maintain consistency with Recall@K)
        n_samples = len(embeddings)
        print(f"    using all data: {n_samples} samples")
        
        # calculate cosine similarity
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        for k in k_values:
            precisions = []
            recalls = []
            
            iterator = tqdm(range(len(embeddings)), desc=f'Precision/Recall@{k}', unit='sample') if progress else range(len(embeddings))
            for i in iterator:
                similarities = similarity_matrix[i]
                similarities[i] = -1  # exclude self
                top_k_indices = np.argsort(similarities)[-k:][::-1]
                
                true_label = labels[i]
                correct_count = np.sum(labels[top_k_indices] == true_label)
                
                # Precision@K
                precision = correct_count / k
                precisions.append(precision)
                
                # Recall@K (same as basic method: 1 if answer is in top K, 0 otherwise)
                recall = 1.0 if correct_count > 0 else 0.0
                recalls.append(recall)
            
            metrics[f'precision@{k}'] = np.mean(precisions)
            metrics[f'recall@{k}'] = np.mean(recalls)
            print(f"    Precision@{k}: {np.mean(precisions):.4f}")
            print(f"    Recall@{k}: {np.mean(recalls):.4f}")
        
        # Mean Average Precision (MAP) - using all data
        map_scores = []
        
        iterator = tqdm(range(len(embeddings)), desc='MAP calculation', unit='sample') if progress else range(len(embeddings))
        for i in iterator:
            similarities = similarity_matrix[i]
            similarities[i] = -1
            sorted_indices = np.argsort(similarities)[::-1]
            
            true_label = labels[i]
            relevant_indices = np.where(labels[sorted_indices] == true_label)[0]
            
            if len(relevant_indices) > 0:
                precisions_at_k = []
                for idx in relevant_indices:
                    k = idx + 1
                    correct_count = np.sum(labels[sorted_indices[:k]] == true_label)
                    precision = correct_count / k
                    precisions_at_k.append(precision)
                map_scores.append(np.mean(precisions_at_k))
        
        metrics['map'] = np.mean(map_scores) if map_scores else 0
        print(f"    MAP: {metrics['map']:.4f}")
        
    except Exception as e:
        print(f"    ⚠️ accuracy metrics calculation failed: {e}")
    
    return metrics
