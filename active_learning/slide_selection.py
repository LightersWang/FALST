import numpy as np


def slide_selector(args, num_bag_shot, bag_idx_all):
    if args.slide_active_method.lower() == 'random':
        bag_idx_few_shot = np.random.choice(bag_idx_all, num_bag_shot, replace=False).tolist()
    else:
        raise NotImplementedError(args.slide_active_method)
    
    return bag_idx_few_shot