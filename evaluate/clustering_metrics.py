#!/usr/bin/env python3
"""
Clustering quality metrics for face recognition evaluation
- Silhouette Score
- Davies-Bouldin Index  
- Calinski-Harabasz Index
"""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def calculate_all_clustering_metrics(embeddings, labels):
    """Calculate all clustering quality metrics"""
    print("  🎯 calculating clustering quality metrics...")
    
    metrics = {}
    
    try:
        # Silhouette Score
        if len(embeddings) > 10000:
            # improve calculation efficiency with sampling
            indices = np.random.choice(len(embeddings), 10000, replace=False)
            sample_embeddings = embeddings[indices]
            sample_labels = labels[indices]
        else:
            sample_embeddings = embeddings
            sample_labels = labels
            
        silhouette = silhouette_score(sample_embeddings, sample_labels, metric='cosine')
        metrics['silhouette_score'] = silhouette
        print(f"    Silhouette Score: {silhouette:.4f}")
        
        # Davies-Bouldin Index
        db_index = davies_bouldin_score(sample_embeddings, sample_labels)
        metrics['davies_bouldin_index'] = db_index
        print(f"    Davies-Bouldin Index: {db_index:.4f}")
        
        # Calinski-Harabasz Index
        ch_index = calinski_harabasz_score(sample_embeddings, sample_labels)
        metrics['calinski_harabasz_index'] = ch_index
        print(f"    Calinski-Harabasz Index: {ch_index:.2f}")
        
    except Exception as e:
        print(f"    ⚠️ clustering metrics calculation failed: {e}")
    
    return metrics
