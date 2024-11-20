import numpy as np
from .patch.k_medoids import k_medoids_query

def oracle_patch_selector(num_instance_shot, all_pos_instance_idx, all_neg_instance_idx):
    if num_instance_shot > len(all_pos_instance_idx):
        instance_few_shot_pos_idx = np.random.choice(all_pos_instance_idx, len(all_pos_instance_idx), replace=False).tolist()
    else:
        instance_few_shot_pos_idx = np.random.choice(all_pos_instance_idx, num_instance_shot, replace=False).tolist()
    instance_few_shot_neg_idx = np.random.choice(all_neg_instance_idx, num_instance_shot, replace=False).tolist()

    return instance_few_shot_pos_idx, instance_few_shot_neg_idx


def patch_selector(args, num_instance_shot, all_instance_idx):
    patch_al_method = args.patch_active_method.lower()
    if patch_al_method == 'random':
        instance_few_shot_idx = np.random.choice(all_instance_idx, 2 * num_instance_shot, replace=False).tolist()
        instance_few_shot_pos_idx = instance_few_shot_idx[:num_instance_shot]
        instance_few_shot_neg_idx = instance_few_shot_idx[num_instance_shot:]
    # elif patch_al_method in ['kmeans', 'k_means', 'k-means', 'kmedoids', 'k_medoids', 'k-medoids']:

    else:
        raise NotImplementedError(args.patch_active_method)
    
    return instance_few_shot_pos_idx, instance_few_shot_neg_idx