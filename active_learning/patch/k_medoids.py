import torch
import numpy as np
from sklearn_extra.cluster import KMedoids

def k_medoids_query(weight_path, n_query, pool_cases):
    # k-medoids query
    feat = None
    k_medoids = KMedoids(n_clusters=n_query, init='k-medoids++').fit(feat)
    k_medoids_indices = k_medoids.medoid_indices_
    train_cases = np.array(pool_cases)[k_medoids_indices]

    return train_cases.tolist()