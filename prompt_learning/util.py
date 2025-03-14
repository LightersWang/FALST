import torch
import numpy as np
from tqdm import tqdm

import utliz
from clip import clip
from prompt_learning.select_method import k_means_clustering, mini_batch_k_means


def norm_logit(pred, no_softmax=False):
    pred = pred - pred.min()
    pred = pred / pred.max()
    if no_softmax:
        return pred
    pred = torch.softmax(pred, dim=-1)
    return pred


def cal_matrix_mul(matrix1, matrix2, batch_size=4096):
    # matrix1: [N, D]
    # matrix2: [D, M]
    # return: [N, M]
    if batch_size == -1:
        return matrix1 @ matrix2
    else:
        N, D = matrix1.shape
        D, M = matrix2.shape
        num_batch = matrix1.shape[0] // batch_size + 1
        output = torch.zeros([N, M]).to(matrix1.device).type(matrix1.dtype)
        for i in range(num_batch):
            output[i * batch_size: (i + 1) * batch_size] = matrix1[i * batch_size: (i + 1) * batch_size] @ matrix2
        return output


def load_clip_to_cpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())

    return model.float()


def clip_classifier(prompts, clip_model):
    with torch.no_grad():
        clip_weights = []
        for prompt_i in prompts:
            texts = clip.tokenize(prompt_i).cuda()                              # (2, 77)
            class_embeddings = clip_model.encode_text(texts)                    # (2, 1024)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)     # (2, 1024)
            clip_weights.append(class_embeddings)
        clip_weights = torch.stack(clip_weights)                # (20, 2, 1024)
        clip_weights = clip_weights.mean(dim=0)                 # (2, 1024)
        clip_weights /= clip_weights.norm(dim=-1, keepdim=True) # (2, 1024)
    return clip_weights


### Few-shot dataset

class simple_instance_dataset(torch.utils.data.Dataset):
    def __init__(self, feat, label):
        self.feat = feat
        self.label = label

    def __getitem__(self, index):
        return self.feat[index], self.label[index], index

    def __len__(self):
        return len(self.label)
    

### MIL related func

def gather_instance_prediction_and_pred_bag(instance_pred, instance_corresponding_slide_index, instance_corresponding_slide_label):
    if type(instance_pred) is torch.Tensor:
        instance_pred = instance_pred.detach().cpu().numpy()
    if len(instance_pred.shape) == 2:
        instance_pred = instance_pred[:, 1]

    bag_label_gt = []
    bag_label_pred = []
    for slide_index_i in np.unique(instance_corresponding_slide_index):
        instance_idx_from_slide_i = np.where(instance_corresponding_slide_index == slide_index_i)[0]
        instance_pred_from_slide_i = instance_pred[instance_idx_from_slide_i]
        if instance_corresponding_slide_label[instance_idx_from_slide_i].max() != instance_corresponding_slide_label[instance_idx_from_slide_i].min():
            print("Warning: slide {} contains both positive and negative instances".format(slide_index_i))
            raise
        pred_slide_i = np.max(instance_pred_from_slide_i)
        gt_slide_i = instance_corresponding_slide_label[instance_idx_from_slide_i[0]]
        bag_label_gt.append(gt_slide_i)
        bag_label_pred.append(pred_slide_i)
    bag_label_gt = np.array(bag_label_gt)
    bag_label_pred = np.array(bag_label_pred)
    bag_auc = utliz.cal_auc(bag_label_gt, bag_label_pred)
    return bag_auc


### Text branch optimize

class Optimizer_TextBranch:
    def __init__(self, model, train_loader, test_loader, optimizer,
                 writer=None, num_epoch=100,
                 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer

        self.writer = writer
        self.num_epoch = num_epoch
        self.dev = dev
        self.log_period = 10

    def optimize(self):
        for epoch in range(self.num_epoch):
            self.train_one_epoch(epoch)
            if (epoch % 10 == 0) or (epoch == self.num_epoch-1):
                self.test(epoch)
        return 0

    def train_one_epoch(self, epoch):
        self.model.train()
        loader = self.train_loader

        patch_label = []
        patch_pred = []
        # for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Epoch {} training'.format(epoch))):
        for iter, (data, label, selected) in enumerate(loader):
            data = data.to(self.dev)
            label = label.to(self.dev)

            prediction = self.model(data)
            prediction = torch.softmax(prediction, 1)
            loss = torch.mean(-1. * (label * torch.log(prediction[:, 1]+1e-5) + (1. - label) * torch.log(1. - prediction[:, 1]+1e-5)))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            patch_label.append(label.detach().cpu().numpy())
            patch_pred.append(prediction.detach().cpu().numpy()[:, 1])

            niter = epoch * len(loader) + iter
            self.writer.add_scalar('train_loss', loss.item(), niter)
            # print('\ntrain_loss: {:.4f}'.format(loss.item()))

        patch_label = np.concatenate(patch_label)
        patch_pred = np.concatenate(patch_pred)

        instance_auc = utliz.cal_auc(patch_label.reshape(-1), patch_pred.reshape(-1))
        print("Epoch{} train_instance_AUC: {:.4f}".format(epoch, instance_auc))
        self.writer.add_scalar('train_instance_AUC', instance_auc, epoch)

        return 0

    def test(self, epoch):
        self.model.eval()
        loader = self.test_loader

        patch_label = []
        patch_pred = []
        # for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Epoch {} testing'.format(epoch))):
        for iter, (data, label, selected) in enumerate(loader):
            data = data.to(self.dev)
            label = label.to(self.dev)
            with torch.no_grad():
                prediction = self.model(data)
                prediction = torch.softmax(prediction, 1)

            patch_label.append(label.detach().cpu().numpy())
            patch_pred.append(prediction.detach().cpu().numpy()[:, 1])

        patch_label = np.concatenate(patch_label)
        patch_pred = np.concatenate(patch_pred)

        instance_auc = utliz.cal_auc(patch_label.reshape(-1), patch_pred.reshape(-1))
        print("\nEpoch{} test_instance_AUC: {:.4f}".format(epoch, instance_auc))
        self.writer.add_scalar('test_instance_AUC', instance_auc, epoch)
        return 0


def build_MIL_Adapter_cache_model(train_loader, downsample_neg_instances=1.0):
    # patches have been encoded by CLIP, we can just use them
    # [Attention]: must be iterated in order to build the cache model under few-shot setting
    cache_keys = []
    cache_values = []
    cache_values_unmasked = []
    cache_corresponding_slide_label = []
    cache_corresponding_slide_index = []
    for iter, (data, label, selected) in enumerate(tqdm(train_loader, desc='Building cache model')):
        instance_label = label[0].squeeze()
        instance_label_unmasked = label[3].squeeze()
        data = data.squeeze()
        ## cache all labeled and unlabeled instances
        cache_keys.append(data)
        cache_values.append(instance_label)
        cache_values_unmasked.append(instance_label_unmasked)
        cache_corresponding_slide_label.append(torch.ones_like(instance_label) * label[1].squeeze())
        cache_corresponding_slide_index.append(torch.ones_like(instance_label) * label[2].squeeze())
    cache_keys = torch.cat(cache_keys)
    cache_values = torch.cat(cache_values)
    cache_values_unmasked = torch.cat(cache_values_unmasked)
    cache_corresponding_slide_label = torch.cat(cache_corresponding_slide_label)
    cache_corresponding_slide_index = torch.cat(cache_corresponding_slide_index)

    ## split cache into learnable and static parts
    idx_cache_unlabeled = torch.where(cache_values == -1)[0]
    idx_cache_labeled = torch.where(cache_values != -1)[0]

    cache_keys_unlabeled = cache_keys[idx_cache_unlabeled]
    cache_values_unlabeled = cache_values[idx_cache_unlabeled]
    cache_values_unlabeled_GT = cache_values_unmasked[idx_cache_unlabeled]
    cache_corresponding_slide_label_unlabeled = cache_corresponding_slide_label[idx_cache_unlabeled]
    cache_corresponding_slide_index_unlabeled = cache_corresponding_slide_index[idx_cache_unlabeled]

    cache_keys_labeled = cache_keys[idx_cache_labeled]
    cache_values_labeled = cache_values[idx_cache_labeled]
    cache_corresponding_slide_label_labeled = cache_corresponding_slide_label[idx_cache_labeled]
    cache_corresponding_slide_index_labeled = cache_corresponding_slide_index[idx_cache_labeled]

    if downsample_neg_instances < 0:
        # select representative neg instances
        print("[Select by cluster center] downsample negative instances to {}".format(np.abs(downsample_neg_instances)))
        pos_idx = torch.where(cache_values_labeled == 1)[0]
        neg_idx = torch.where(cache_values_labeled == 0)[0]

        # cluster_centers = k_means_clustering(cache_keys_labeled[neg_idx], num_clusters=np.abs(int(downsample_neg_instances)))
        cluster_centers = mini_batch_k_means(cache_keys_labeled[neg_idx], num_clusters=np.abs(int(downsample_neg_instances)), batch_size=1000, max_iterations=100)

        cache_keys_labeled = torch.cat([cache_keys_labeled[pos_idx], cluster_centers], dim=0)
        cache_values_labeled = torch.cat([cache_values_labeled[pos_idx], torch.zeros([np.abs(int(downsample_neg_instances))]).type(torch.int64)], dim=0)
        cache_corresponding_slide_label_labeled = torch.cat([cache_corresponding_slide_label_labeled[pos_idx], cache_corresponding_slide_label_labeled[neg_idx]], dim=0)
        cache_corresponding_slide_index_labeled = torch.cat([cache_corresponding_slide_index_labeled[pos_idx], cache_corresponding_slide_index_labeled[neg_idx]], dim=0)

    elif downsample_neg_instances > 1.0:
        print("[Random] downsample negative instances to {}".format(downsample_neg_instances))
        pos_idx = torch.where(cache_values_labeled == 1)[0]
        neg_idx = torch.where(cache_values_labeled == 0)[0]
        neg_idx = neg_idx[torch.randperm(neg_idx.shape[0])[:int(downsample_neg_instances)]]
        cache_keys_labeled = torch.cat([cache_keys_labeled[pos_idx], cache_keys_labeled[neg_idx]], dim=0)
        cache_values_labeled = torch.cat([cache_values_labeled[pos_idx], cache_values_labeled[neg_idx]], dim=0)
        cache_corresponding_slide_label_labeled = torch.cat([cache_corresponding_slide_label_labeled[pos_idx], cache_corresponding_slide_label_labeled[neg_idx]], dim=0)
        cache_corresponding_slide_index_labeled = torch.cat([cache_corresponding_slide_index_labeled[pos_idx], cache_corresponding_slide_index_labeled[neg_idx]], dim=0)

    print("Cache Model built (labeled part):\nkey: {}\nvalue: {}".format(cache_keys_labeled.shape, cache_values_labeled.shape))
    print("Cache Model built (unlabeled part):\nkey: {}\nvalue: {}".format(cache_keys_unlabeled.shape, cache_values_unlabeled.shape))
    return (cache_keys_unlabeled, cache_values_unlabeled, cache_corresponding_slide_label_unlabeled, cache_corresponding_slide_index_unlabeled,
            cache_keys_labeled,   cache_values_labeled,   cache_corresponding_slide_label_labeled,   cache_corresponding_slide_index_labeled,
            cache_values_unlabeled_GT)