import numpy as np
from copy import deepcopy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


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


def cover_score(cosine_dist, select_indices, all_indices):
    if not np.any(select_indices):
        return 0.0
    else:
        mat = cosine_dist[select_indices, :][:, all_indices]
        repre = mat.max(axis=0)
        cover_score = repre.sum()
        return cover_score


def max_cover_query_step2(n_data, candidate_thresh, all_indices, cosine_dist):    
    # sample 
    select_indices = np.zeros((n_data, ), dtype=np.bool8)
    pre_cover_score = cover_score(cosine_dist, select_indices, all_indices)

    while (cover_score(cosine_dist, select_indices, all_indices) < candidate_thresh * all_indices.sum()):
        cover_score_list = np.zeros((n_data, ))
        for j in range(n_data):
            if not select_indices[j]:
                select_indices_temp = deepcopy(select_indices)
                select_indices_temp[j] = True
                cover_score_list[j] = cover_score(cosine_dist, select_indices_temp, all_indices)

        cover_score_list -= pre_cover_score
        next_sample_index = cover_score_list.argmax()
        select_indices[next_sample_index] = True
    
    return select_indices


def max_cover_query_step3(n_data, n_query, all_indices, cosine_dist):    
    # sample 
    select_indices = np.zeros((n_data, ), dtype=np.bool8)
    pre_cover_score = cover_score(cosine_dist, select_indices, all_indices)

    for _ in range(n_query):
        cover_score_list = np.zeros((n_data, ))
        for j in range(n_data):
            if not select_indices[j]:
                select_indices_temp = deepcopy(select_indices)
                select_indices_temp[j] = True
                cover_score_list[j] = cover_score(cosine_dist, select_indices_temp, all_indices)

        cover_score_list -= pre_cover_score
        next_sample_index = cover_score_list.argmax()
        select_indices[next_sample_index] = True
    
    return select_indices


def representative_annotation(feature, n_query, n_data, n_clusters=3, candidate_thresh=0.9):
    if int(candidate_thresh * n_data) < n_query:
        raise ValueError(f"candidate_thresh is too small ({candidate_thresh})")
    
    # Step 0: Calculate cosine similarity
    cosine_dist = cosine_similarity(feature)

    # Step 1: Agglomerative Clustering
    agglo_cluster_label = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(feature)

    # Step 2: max-cover to form candidate set
    candidate_indices = np.zeros((n_data, ), dtype=np.bool8)
    for cluster_label in np.unique(agglo_cluster_label):
        cluster_indices = (agglo_cluster_label == cluster_label)

        cluster_sample_indices = max_cover_query_step2(
            n_data=n_data,
            candidate_thresh=candidate_thresh,
            all_indices=cluster_indices,
            cosine_dist=cosine_dist
        )

        candidate_indices = candidate_indices | cluster_sample_indices

    # print(candidate_indices.sum())

    # Step 3: max-cover to final sampling result
    ra_indices = max_cover_query_step3(
        n_data=n_data,
        n_query=n_query,
        all_indices=candidate_indices,
        cosine_dist=cosine_dist
    )
    
    all_indices = np.arange(n_data)
    return all_indices[ra_indices]