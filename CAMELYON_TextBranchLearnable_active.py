import argparse

import numpy as np
import torch
import torch.optim
import torch.utils.data
import datetime
import util

import active_learning as al
from prompt_learning import fast, tip_adapter, tip_adapter_f
from Datasets_loader.dataset_CAMELYON16_new import CAMELYON_16_5x_feat


class Active_FewShotBag_FewShotInstance(torch.utils.data.Dataset):
    def __init__(self, args, ds, num_bag_shot=-1, num_instance_shot=-1):
        """
            num_x_shot=-1 --> all shots
        """
        self.ds = ds
        self.args = args
        self.num_bag_shot = num_bag_shot
        self.num_instance_shot = num_instance_shot

        # 1. generate bag few shot idx, compatible with CAMELYON_16_5x_feat
        self.bag_shot_indexes = []
        ds_label = self.ds.slide_label_all
        category = np.unique(ds_label)
        
        if args.slide_active_method.lower() == 'selected':
            pos_slides = [25, 58, 65, 91, 84, 169, 5, 34, 86, 96, 238, 29, 202, 216, 249, 190]
            neg_slides = [243, 214, 135, 33, 181, 229, 117, 71, 134, 186, 142, 158, 197, 0, 146, 170]
            self.bag_shot_indexes += pos_slides[:self.num_bag_shot] + neg_slides[:self.num_bag_shot]
        else:
            for category_i in category:
                idx_category_i_all = np.where(ds_label == category_i)[0]
                if self.num_bag_shot != -1:
                    # actively select slide of each category
                    idx_category_i_few_shot = al.slide_selector(args, self.num_bag_shot, idx_category_i_all)
                else:
                    idx_category_i_few_shot = idx_category_i_all.tolist()
                self.bag_shot_indexes += idx_category_i_few_shot
        print(f'selected slides: {self.bag_shot_indexes}')
        
        # 2. generate corresponding instance few shot idx for each positive bag
        self.bag_instance_shot_indexes = []
        if args.patch_active_method.lower() == 'falst':
            # get pos and neg indexes
            pos_slide_indexes = [bag_shot_idx for bag_shot_idx in self.bag_shot_indexes 
                                 if self.ds.slide_label_all[bag_shot_idx] == 1]
            neg_slide_indexes = [bag_shot_idx for bag_shot_idx in self.bag_shot_indexes 
                                 if self.ds.slide_label_all[bag_shot_idx] == 0]
            
            # get feature from pos and neg slides
            feat_from_pos_bag = [ds.slide_feat_all[bag_shot_idx] for bag_shot_idx in pos_slide_indexes]
            feat_from_neg_bag = [ds.slide_feat_all[bag_shot_idx] for bag_shot_idx in neg_slide_indexes]
            feat_from_pos_bag = np.concatenate(feat_from_pos_bag, 0)
            feat_from_neg_bag = np.concatenate(feat_from_neg_bag, 0)
            
            # append feat from neg slides
            for neg_slide_index in neg_slide_indexes:
                self.bag_instance_shot_indexes.append((neg_slide_index, 0, [], []))
                
            # get few-shot indexes from pos slides
            bag_instance_shot_indexes_from_pos_slides = al.falst_patch_selector_v1(
                args=args, ds=ds,
                num_instance_shot=num_instance_shot,
                pos_slide_feat_train=feat_from_pos_bag, 
                neg_slide_feat_train=feat_from_neg_bag,
                pos_slide_indexes=pos_slide_indexes,
            )
            self.bag_instance_shot_indexes += bag_instance_shot_indexes_from_pos_slides
                
        else:
            for bag_shot_idx in self.bag_shot_indexes:
                # use all patches from negative slides
                if self.ds.slide_label_all[bag_shot_idx] == 0:
                    self.bag_instance_shot_indexes.append((bag_shot_idx, 0, [], []))
                # actively select few-shot patches from positive slides
                else:
                    all_instance_feat = ds.slide_feat_all[bag_shot_idx]
                    all_instance_idx = np.arange(ds.slide_patch_label_all[bag_shot_idx].shape[0])

                    if args.patch_active_method.lower() == 'oracle':
                        all_pos_instance_idx = np.where(ds.slide_patch_label_all[bag_shot_idx] == 1)[0]
                        all_neg_instance_idx = np.where(ds.slide_patch_label_all[bag_shot_idx] == 0)[0]
                        instance_few_shot_pos_idx, instance_few_shot_neg_idx = al.oracle_patch_selector(
                            num_instance_shot, all_pos_instance_idx, all_neg_instance_idx)
                    else:
                        instance_few_shot_idx = al.patch_selector(args, num_instance_shot, all_instance_idx, all_instance_feat)
                        instance_few_shot_label = ds.slide_patch_label_all[bag_shot_idx][instance_few_shot_idx]
                        instance_few_shot_pos_idx = instance_few_shot_idx[instance_few_shot_label == 1]
                        instance_few_shot_neg_idx = instance_few_shot_idx[instance_few_shot_label == 0]
                        print(f'num of pos instance: {len(instance_few_shot_pos_idx)}/{num_instance_shot * 2}')
                        print(f'num of neg instance: {len(instance_few_shot_neg_idx)}/{num_instance_shot * 2}')
                        
                        patch_label = ds.slide_patch_label_all[bag_shot_idx]
                        actual_pos_ratio = patch_label.sum() / patch_label.shape[0]
                        sampling_pos_ratio = len(instance_few_shot_pos_idx) / (num_instance_shot * 2)
                        print(f'actual pos num: {patch_label.sum()}')
                        print(f'actual pos ratio: {actual_pos_ratio}')
                        print(f'sampling pos ratio: {sampling_pos_ratio}')
                        # if len(instance_few_shot_pos_idx) == 0:
                        #     instance_few_shot_pos_idx = None

                    self.bag_instance_shot_indexes.append((bag_shot_idx, 1, instance_few_shot_pos_idx, instance_few_shot_neg_idx))

            print("{}-Shot Bag {}-Shot Instance dataset build".format(num_bag_shot, num_instance_shot))

    def __getitem__(self, index):
        bag_few_shot_idx, bag_label, pos_instance_few_shot_idx, neg_instance_few_shot_idx = self.bag_instance_shot_indexes[index]
        slide_feat, label_list, index_raw = self.ds.__getitem__(bag_few_shot_idx)
        label_list.append(label_list[0])  # append GT instance label in PosBag for Measuring Pseudo-label Acc

        # replace instance labels from pos bag to few-shot labels
        if bag_label == 1:
            instance_few_shot_label = np.ones_like(label_list[0]) * -1
            instance_few_shot_label[pos_instance_few_shot_idx] = 1
            instance_few_shot_label[neg_instance_few_shot_idx] = 0
            label_list[0] = instance_few_shot_label
        return slide_feat, label_list, index

    def __len__(self):
        return len(self.bag_instance_shot_indexes)



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # method
    parser.add_argument('--method', default='fast', type=str, choices=['fast', 'tip_adapter', 'tip_adapter_f'],
                        help='few-shot prompt learning method')
    parser.add_argument('--slide_active_method', default='random', type=str, help='actvie learning method for slide selection')
    parser.add_argument('--patch_active_method', default='random', type=str, help='actvie learning method for patch selection')

    # optimizer
    parser.add_argument('--epochs', default=20000, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=4096, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr_keys', default=0.001, type=float, help='initial learning rate of learnable keys (default: 0.001)')
    parser.add_argument('--lr_values', default=0.01, type=float, help='initial learning rate of learnable values (default: 0.001)')

    # housekeeping
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--workers', default=0, type=int,help='number workers (default: 6)')
    parser.add_argument('--comment', default='FALST', type=str, help='name for tensorboardX')
    parser.add_argument('--save_intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    parser.add_argument('--num_bag_shot', default=1, type=int, help='num of bag few shot')
    parser.add_argument('--num_instance_shot', default=16, type=int, help='num of instance few shot')

    parser.add_argument('--downsample_neg_instances', default=1.0, type=float, help='downsample neg instance when building cache model')
    
    # falst setting
    parser.add_argument('--gmm_components', default=20, type=int, help='number of GMM components')
    parser.add_argument('--pos_density_only', default=False, action='store_true', help='use density difference')
    parser.add_argument('--neg_density_only', default=False, action='store_true', help='use density ratio')

    # MIL_CLIP settings
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()

    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_%s" % args.comment.replace('/', '_') + \
           "_Seed{}_Bs{}_LrKey{}_LrValue{}_{}BagShot{}InstShot_CacheModelDownsample{}".format(
               args.seed, args.batch_size, args.lr_keys, args.lr_values, args.num_bag_shot, 
               args.num_instance_shot, args.downsample_neg_instances
           )
    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    args.model_device = args.device
    util.setup_runtime(seed=args.seed, cuda_dev_id=list(np.unique(args.model_device + args.device)))

    # Setup loaders
    train_ds_return_bag = CAMELYON_16_5x_feat(split='train', return_bag=True, feat="RN50")
    train_ds_return_bag = Active_FewShotBag_FewShotInstance(
        args=args, ds=train_ds_return_bag, 
        num_bag_shot=args.num_bag_shot, num_instance_shot=args.num_instance_shot)
    val_ds_return_bag = CAMELYON_16_5x_feat(split='test', return_bag=True, feat="RN50")

    train_loader_bag = torch.utils.data.DataLoader(train_ds_return_bag, batch_size=1, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader_bag = torch.utils.data.DataLoader(val_ds_return_bag, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False)

    if args.method.lower() == 'fast':
        fast(args, name, train_loader_bag, val_ds_return_bag)
    elif args.method.lower() == 'tip_adapter':
        tip_adapter(args, name, train_loader_bag, val_ds_return_bag)
    elif args.method.lower() == 'tip_adapter_f':
        tip_adapter_f(args, name, train_loader_bag, val_ds_return_bag)
    else:
        raise NotImplementedError(args.method)
