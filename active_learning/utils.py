import numpy as np

def kcenter_greedy(dist_mat, n_query):
    """K-center greedy sampling in CoreSet.

    Args:
        dist_mat: distance matrix of the queried embeddings [n_data, n_data]
        n_query: number of querying samples
    """
    n_data = dist_mat.shape[0]
    init_idx = np.random.randint(n_data)
    all_indices = np.arange(n_data)
    labeled_indices = np.zeros((n_data, ), dtype=np.bool_)
    labeled_indices[init_idx] = True

    # sample 
    for _ in range(n_query-1):
        mat = dist_mat[~labeled_indices, :][:, labeled_indices]
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = all_indices[~labeled_indices][q_idx_]
        labeled_indices[q_idx] = True
    
    return all_indices[labeled_indices], all_indices[~labeled_indices]