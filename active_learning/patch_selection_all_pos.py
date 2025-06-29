import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity

from .utils import kcenter_greedy, representative_annotation

def oracle_patch_selector(num_instance_shot, all_pos_instance_idx):
    if num_instance_shot > len(all_pos_instance_idx):
        instance_few_shot_pos_idx = np.random.choice(all_pos_instance_idx, len(all_pos_instance_idx), replace=False).tolist()
    else:
        instance_few_shot_pos_idx = np.random.choice(all_pos_instance_idx, num_instance_shot, replace=False).tolist()
    # instance_few_shot_neg_idx = np.random.choice(all_neg_instance_idx, num_instance_shot, replace=False).tolist()

    return instance_few_shot_pos_idx #, instance_few_shot_neg_idx


def patch_selector(args, num_instance_shot, all_instance_idx, all_instance_feat):
    patch_al_method = args.patch_active_method.lower()
    
    if patch_al_method == 'random':
        instance_few_shot_idx = np.random.choice(all_instance_idx, num_instance_shot, replace=False).tolist()
        
    elif patch_al_method in ['kmeans', 'k_means', 'k-means', 'kmedoids', 'k_medoids', 'k-medoids']:
        k_medoids = KMedoids(n_clusters=num_instance_shot, init='k-medoids++').fit(all_instance_feat)
        instance_few_shot_idx = k_medoids.medoid_indices_
        
    elif patch_al_method in ['coreset', 'core_set', 'k_center_greedy']:
        dist_mat = pairwise_distances(all_instance_feat)
        instance_few_shot_idx, _ = kcenter_greedy(dist_mat, num_instance_shot)
        
    elif patch_al_method in ['ra', ]:
        instance_few_shot_idx = representative_annotation(
            feature=all_instance_feat,
            n_query=num_instance_shot,
            n_data=all_instance_feat.shape[0])
        
    else:
        raise NotImplementedError(args.patch_active_method)
    
    return np.array(instance_few_shot_idx)


def falst_patch_selector_v1(args, ds, num_instance_shot, pos_slide_feat_train, neg_slide_feat_train, 
                            pos_slide_indexes, density_estimator='gmm'):
    
    # fit density estimators
    if density_estimator == 'gmm':
        print(f'GMM-{args.gmm_components}')
        gmm_kwargs = {
            'n_components': args.gmm_components, 
            'random_state': 0, 
            'verbose': 10, 
            'covariance_type': 'diag',
            'n_init': 1, 
            'max_iter': 100,
            'random_state': args.seed
        } 
        pos_density_estimator = GaussianMixture(**gmm_kwargs).fit(pos_slide_feat_train)
        neg_density_estimator = GaussianMixture(**gmm_kwargs).fit(neg_slide_feat_train)
    
    elif density_estimator == 'kde':
        print('KDE')
        kde_kwargs = {
            'bandwidth': 1.0,
            'kernel': 'gaussian',
            'metric': 'euclidean',
            'algorithm': 'auto',
        }
        pos_density_estimator = KernelDensity(**kde_kwargs).fit(pos_slide_feat_train)
        neg_density_estimator = KernelDensity(**kde_kwargs).fit(neg_slide_feat_train)
    
    else:
        raise NotImplementedError(density_estimator)
    
    bag_instance_shot_indexes_from_pos_slides = []
    for pos_slide_idx in pos_slide_indexes:
        print(pos_slide_idx)
        all_instance_feat = ds.slide_feat_all[pos_slide_idx]
        # all_instance_idx = np.arange(all_instance_feat.shape[0])
        
        pos_likelihood = pos_density_estimator.score_samples(all_instance_feat)
        neg_likelihood = neg_density_estimator.score_samples(all_instance_feat)
    
        # pos patch: 
        if args.pos_density_only:
            # log p_pos(x)
            print('log p_pos(x)')
            pos_patch_ll = pos_likelihood
        elif args.neg_density_only:
            # - log p_neg(x)
            print('- log p_neg(x)')
            pos_patch_ll = -1 * neg_likelihood
        else:
            # log p_pos(x) - log p_neg(x)
            print('log p_pos(x) - log p_neg(x)')
            pos_patch_ll = pos_likelihood - neg_likelihood
        pos_patch_idx_sorted = np.argsort(pos_patch_ll)[::-1]
        pos_patch_idx = pos_patch_idx_sorted[:num_instance_shot]
        assert np.all(pos_patch_ll[pos_patch_idx] == np.sort(pos_patch_ll)[::-1][:num_instance_shot])
        
        # # neg patch: log p_pos(x) + log p_neg(x)
        # if args.neg_density_only:
        #     # log p_neg(x)
        #     print('log p_neg(x)')
        #     neg_patch_ll = neg_likelihood
        # else:
        #     # log p_pos(x) + log p_neg
        #     print('log p_pos(x) + log p_neg(x)')
        #     neg_patch_ll = pos_likelihood + neg_likelihood
        # neg_patch_idx_sorted = np.argsort(neg_patch_ll)[::-1]
        # neg_patch_idx = neg_patch_idx_sorted[:num_instance_shot]
        # assert np.all(neg_patch_ll[neg_patch_idx] == np.sort(neg_patch_ll)[::-1][:num_instance_shot])
        
        # instance_few_shot_idx = np.array(pos_patch_idx.tolist() + neg_patch_idx.tolist())
        instance_few_shot_idx = pos_patch_idx
        instance_few_shot_label = ds.slide_patch_label_all[pos_slide_idx][instance_few_shot_idx]
        instance_few_shot_pos_idx = instance_few_shot_idx[instance_few_shot_label == 1]
        instance_few_shot_neg_idx = instance_few_shot_idx[instance_few_shot_label == 0]
        print(f'num of pos instance: {len(instance_few_shot_pos_idx)}/{num_instance_shot}')
        print(f'num of neg instance: {len(instance_few_shot_neg_idx)}/{num_instance_shot}')
        
        patch_label = ds.slide_patch_label_all[pos_slide_idx]
        actual_pos_ratio = patch_label.sum() / patch_label.shape[0]
        sampling_pos_ratio = len(instance_few_shot_pos_idx) / (num_instance_shot)
        print(f'actual pos num: {patch_label.sum()}')
        print(f'actual pos ratio: {actual_pos_ratio}')
        print(f'sampling pos ratio: {sampling_pos_ratio}')
        
        bag_instance_shot_indexes_from_pos_slides.append((pos_slide_idx, 1, instance_few_shot_pos_idx, instance_few_shot_neg_idx))
    
    return bag_instance_shot_indexes_from_pos_slides


def falst_patch_selector_v2(args, ds, num_instance_shot, pos_slide_feat_train, neg_slide_feat_train, 
                            pos_slide_indexes, density_estimator='gmm'):
    
    # fit density estimators
    all_slide_feat_train = np.concatenate([pos_slide_feat_train, neg_slide_feat_train], axis=0)
    if density_estimator == 'gmm':
        gmm_kwargs = {
            'n_components': args.gmm_components, 
            'random_state': 0, 
            'verbose': 10, 
            'covariance_type': 'diag',
            'n_init': 1, 
            'max_iter': 100,
            'random_state': args.seed
        } 
        pos_density_estimator = GaussianMixture(**gmm_kwargs).fit(pos_slide_feat_train)
        ref_density_estimator = GaussianMixture(**gmm_kwargs).fit(all_slide_feat_train)
    
    elif density_estimator == 'kde':
        kde_kwargs = {
            'bandwidth': 0.5,
            'kernel': 'gaussian',
            'metric': 'euclidean',
            'algorithm': 'auto',
        }
        pos_density_estimator = KernelDensity(**kde_kwargs).fit(pos_slide_feat_train)
        ref_density_estimator = KernelDensity(**gmm_kwargs).fit(all_slide_feat_train)
    
    else:
        raise NotImplementedError(density_estimator)
    
    bag_instance_shot_indexes_from_pos_slides = []
    for pos_slide_idx in pos_slide_indexes:
        all_instance_feat = ds.slide_feat_all[pos_slide_idx]
        # all_instance_idx = np.arange(all_instance_feat.shape[0])
        
        pos_likelihood = pos_density_estimator.score_samples(all_instance_feat)
        ref_likelihood = ref_density_estimator.score_samples(all_instance_feat)
    
        # pos patch: 
        if args.pos_density_only:
            # log p_pos(x)
            print('log p_pos(x)')
            pos_patch_ll = pos_likelihood
        else:
            # log p_pos(x) - log p_neg(x)
            print('log p_pos(x) - log p_ref(x)')
            pos_patch_ll = pos_likelihood - ref_likelihood
        pos_patch_idx_sorted = np.argsort(pos_patch_ll)[::-1]
        pos_patch_idx = pos_patch_idx_sorted[:num_instance_shot]
        assert np.all(pos_patch_ll[pos_patch_idx] == np.sort(pos_patch_ll)[::-1][:num_instance_shot])
        
        # # neg patch: log p_pos(x) + log p_neg(x)
        # if args.neg_density_only:
        #     # log p_neg(x)
        #     print('log p_neg(x)')
        #     neg_patch_ll = neg_likelihood
        # else:
        #     # log p_pos(x) + log p_neg
        #     print('log p_pos(x) + log p_neg(x)')
        #     neg_patch_ll = pos_likelihood + neg_likelihood
        # neg_patch_idx_sorted = np.argsort(neg_patch_ll)[::-1]
        # neg_patch_idx = neg_patch_idx_sorted[:num_instance_shot]
        # assert np.all(neg_patch_ll[neg_patch_idx] == np.sort(neg_patch_ll)[::-1][:num_instance_shot])
        
        # instance_few_shot_idx = np.array(pos_patch_idx.tolist() + neg_patch_idx.tolist())
        instance_few_shot_idx = pos_patch_idx
        instance_few_shot_label = ds.slide_patch_label_all[pos_slide_idx][instance_few_shot_idx]
        instance_few_shot_pos_idx = instance_few_shot_idx[instance_few_shot_label == 1]
        instance_few_shot_neg_idx = instance_few_shot_idx[instance_few_shot_label == 0]
        print(f'num of pos instance: {len(instance_few_shot_pos_idx)}/{num_instance_shot}')
        print(f'num of neg instance: {len(instance_few_shot_neg_idx)}/{num_instance_shot}')
        
        bag_instance_shot_indexes_from_pos_slides.append((pos_slide_idx, 1, instance_few_shot_pos_idx, instance_few_shot_neg_idx))
    
    return bag_instance_shot_indexes_from_pos_slides



def falst_patch_selector_v3(args, ds, num_instance_shot, pos_slide_feat_train, neg_slide_feat_train, 
                            pos_slide_indexes, density_estimator='gmm'):
    
    # fit density estimators
    all_slide_feat_train = np.concatenate([pos_slide_feat_train, neg_slide_feat_train], axis=0)
    if density_estimator == 'gmm':
        gmm_kwargs = {
            'n_components': args.gmm_components, 
            'random_state': 0, 
            'verbose': 10, 
            'covariance_type': 'diag',
            'n_init': 1, 
            'max_iter': 100,
            'random_state': args.seed
        } 
        ref_density_estimator = GaussianMixture(**gmm_kwargs).fit(all_slide_feat_train)
        neg_density_estimator = GaussianMixture(**gmm_kwargs).fit(neg_slide_feat_train)
    
    elif density_estimator == 'kde':
        kde_kwargs = {
            'bandwidth': 0.5,
            'kernel': 'gaussian',
            'metric': 'euclidean',
            'algorithm': 'auto',
        }
        ref_density_estimator = KernelDensity(**kde_kwargs).fit(all_slide_feat_train)
        neg_density_estimator = KernelDensity(**kde_kwargs).fit(neg_slide_feat_train)
    
    else:
        raise NotImplementedError(density_estimator)
    
    bag_instance_shot_indexes_from_pos_slides = []
    for pos_slide_idx in pos_slide_indexes:
        all_instance_feat = ds.slide_feat_all[pos_slide_idx]
        # all_instance_idx = np.arange(all_instance_feat.shape[0])
        
        ref_likelihood = ref_density_estimator.score_samples(all_instance_feat)
        neg_likelihood = neg_density_estimator.score_samples(all_instance_feat)
    
        # pos patch: 
        pos_patch_ll = ref_likelihood - neg_likelihood
        pos_patch_idx_sorted = np.argsort(pos_patch_ll)[::-1]
        pos_patch_idx = pos_patch_idx_sorted[:num_instance_shot]
        assert np.all(pos_patch_ll[pos_patch_idx] == np.sort(pos_patch_ll)[::-1][:num_instance_shot])
        
        # # neg patch: log p_pos(x) + log p_neg(x)
        # if args.neg_density_only:
        #     # log p_neg(x)
        #     print('log p_neg(x)')
        #     neg_patch_ll = neg_likelihood
        # else:
        #     # log p_pos(x) + log p_neg
        #     print('log p_pos(x) + log p_neg(x)')
        #     neg_patch_ll = pos_likelihood + neg_likelihood
        # neg_patch_idx_sorted = np.argsort(neg_patch_ll)[::-1]
        # neg_patch_idx = neg_patch_idx_sorted[:num_instance_shot]
        # assert np.all(neg_patch_ll[neg_patch_idx] == np.sort(neg_patch_ll)[::-1][:num_instance_shot])
        
        # instance_few_shot_idx = np.array(pos_patch_idx.tolist() + neg_patch_idx.tolist())
        instance_few_shot_idx = pos_patch_idx
        instance_few_shot_label = ds.slide_patch_label_all[pos_slide_idx][instance_few_shot_idx]
        instance_few_shot_pos_idx = instance_few_shot_idx[instance_few_shot_label == 1]
        instance_few_shot_neg_idx = instance_few_shot_idx[instance_few_shot_label == 0]
        print(f'num of pos instance: {len(instance_few_shot_pos_idx)}/{num_instance_shot}')
        print(f'num of neg instance: {len(instance_few_shot_neg_idx)}/{num_instance_shot}')
        
        bag_instance_shot_indexes_from_pos_slides.append((pos_slide_idx, 1, instance_few_shot_pos_idx, instance_few_shot_neg_idx))
    
    return bag_instance_shot_indexes_from_pos_slides
