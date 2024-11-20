import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances

from .utils import kcenter_greedy

def oracle_patch_selector(num_instance_shot, all_pos_instance_idx, all_neg_instance_idx):
    if num_instance_shot > len(all_pos_instance_idx):
        instance_few_shot_pos_idx = np.random.choice(all_pos_instance_idx, len(all_pos_instance_idx), replace=False).tolist()
    else:
        instance_few_shot_pos_idx = np.random.choice(all_pos_instance_idx, num_instance_shot, replace=False).tolist()
    instance_few_shot_neg_idx = np.random.choice(all_neg_instance_idx, num_instance_shot, replace=False).tolist()

    return instance_few_shot_pos_idx, instance_few_shot_neg_idx


def patch_selector(args, num_instance_shot, all_instance_idx, all_instance_feat):
    patch_al_method = args.patch_active_method.lower()
    if patch_al_method == 'random':
        instance_few_shot_idx = np.random.choice(all_instance_idx, 2 * num_instance_shot, replace=False).tolist()
        instance_few_shot_pos_idx = instance_few_shot_idx[:num_instance_shot]
        instance_few_shot_neg_idx = instance_few_shot_idx[num_instance_shot:]
    elif patch_al_method in ['kmeans', 'k_means', 'k-means', 'kmedoids', 'k_medoids', 'k-medoids']:
        k_medoids = KMedoids(n_clusters=2 * num_instance_shot, init='k-medoids++').fit(all_instance_feat)
        instance_few_shot_pos_idx = k_medoids.medoid_indices_[:num_instance_shot]
        instance_few_shot_neg_idx = k_medoids.medoid_indices_[num_instance_shot:]
    elif patch_al_method in ['coreset', 'core_set', 'k_center_greedy']:
        dist_mat = pairwise_distances(all_instance_feat)
        instance_few_shot_idx, _ = kcenter_greedy(dist_mat, 2 * num_instance_shot)
        instance_few_shot_pos_idx = instance_few_shot_idx[:num_instance_shot]
        instance_few_shot_neg_idx = instance_few_shot_idx[num_instance_shot:]
    else:
        raise NotImplementedError(args.patch_active_method)
    
    return instance_few_shot_pos_idx, instance_few_shot_neg_idx