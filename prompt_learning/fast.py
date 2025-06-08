import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

import utliz
from models.learnable_prompt import Instance_CLIP, PromptLearner
from prompt_learning.prompts import get_patch_level_prompts_forCAMELYON
from prompt_learning.util import (simple_instance_dataset, Optimizer_TextBranch, 
                                  gather_instance_prediction_and_pred_bag, 
                                  norm_logit, cal_matrix_mul, build_MIL_Adapter_cache_model, 
                                  load_clip_to_cpu, clip_classifier)


### FAST model
def search_hp_onlyAlpha(search_scale, search_step, clip_logits, cache_logits, labels):

    alpha_list = [i * (search_scale - 0.1) / search_step + 0.1 for i in range(search_step)]

    best_auc = 0
    best_alpha = 0
    best_logits = None

    for alpha in tqdm(alpha_list, desc='Searching Hyperparameters'):

        tip_logits = clip_logits * (1-alpha) + cache_logits * alpha

        tip_logits_normed = norm_logit(tip_logits[:, 1], no_softmax=True)
        auc = utliz.cal_auc(labels, tip_logits_normed)

        if auc > best_auc:
            best_auc = auc
            best_alpha = alpha
            best_logits = tip_logits_normed

    print("After searching, setting, alpha: {:.2f}".format(best_alpha))
    print("After searching, the best AUC: {:.4f}.".format(best_auc))
    return best_auc, best_logits, best_alpha


def attention_functional_batch(query, key, value, batch_size=4096):
    # query: [N, D]
    # key: [K, D]
    # value: [K, C]
    if key.shape[1] != query.shape[1]:
        key = key.T
    N, D = query.shape
    K, C = value.shape

    if batch_size == -1:
        output = torch.softmax(query @ key.t(), dim=-1) @ value
    else:
        output = torch.zeros([N, C]).to(query.device).type(query.dtype)
        num_batch = N // batch_size + 1
        for i in range(num_batch):
            output[i * batch_size: (i + 1) * batch_size] = torch.softmax(query[i * batch_size: (i + 1) * batch_size] @ key.t(), dim=-1) @ value
    return output


def run_tip_adapter_key_value_learnable(cache_keys_unlabeled, cache_values_unlabeled,
                                        cache_keys_labeled, cache_values_labeled,
                                        test_features, test_labels, clip_weights,
                                        cache_values_unlabeled_GT=None,
                                        epoch=20, lr_value=0.001, lr_key=0.001, writer=None, batch_size=4096):
    cache_keys_unlabeled = cache_keys_unlabeled.to(clip_weights.device)
    cache_values_unlabeled = cache_values_unlabeled.to(clip_weights.device)
    cache_keys_labeled = cache_keys_labeled.to(clip_weights.device)
    cache_values_labeled = cache_values_labeled.to(clip_weights.device)
    test_features = test_features.to(clip_weights.device)
    test_labels = test_labels.to(clip_weights.device)

    if clip_weights.shape[0] != test_features.shape[1]:
        clip_weights = clip_weights.T
    if cache_keys_unlabeled.shape[0] != test_features.shape[1]:
        cache_keys_unlabeled = cache_keys_unlabeled.T
    if cache_keys_labeled.shape[0] != test_features.shape[1]:
        cache_keys_labeled = cache_keys_labeled.T

    # 1. build trainable cache model
    cache_values_learnable = torch.nn.Parameter(torch.ones_like(cache_values_unlabeled).float() * 0)
    cache_keys_learnable = torch.nn.Parameter(cache_keys_unlabeled)
    optimizer_values = torch.optim.Adam([cache_values_learnable], lr=lr_value)
    optimizer_keys = torch.optim.Adam([cache_keys_learnable], lr=lr_key)

    # option: optimize both labeled and unlabeled cache keys
    # cache_keys = torch.cat([cache_keys_unlabeled, cache_keys_labeled], dim=1)
    # cache_keys = torch.nn.Parameter(cache_keys)
    # optimizer_keys = torch.optim.Adam([cache_keys], lr=lr_key)

    # 2. cal clip prediction
    clip_logits_test = 100. * cal_matrix_mul(test_features, clip_weights)
    clip_logits_test = torch.softmax(clip_logits_test, dim=-1)
    clip_auc = utliz.cal_auc(test_labels, clip_logits_test[:, 1])

    # 3. training & validation
    train_features = cache_keys_labeled.T
    train_labels = cache_values_labeled
    best_cache_auc_test = 0
    best_cache_logits_test = None
    best_cache_keys = None
    best_cache_values = None

    for iter in range(epoch):
        cache_values = torch.cat([torch.sigmoid(cache_values_learnable), cache_values_labeled], dim=0)
        cache_values = torch.stack([1 - cache_values, cache_values], dim=1)

        cache_keys = torch.cat([cache_keys_learnable, cache_keys_labeled], dim=1)

        cache_logits_train = attention_functional_batch(train_features, cache_keys, cache_values, batch_size=batch_size)

        tip_train_auc = utliz.cal_auc(train_labels, cache_logits_train[:, 1])

        loss = F.cross_entropy(cache_logits_train, train_labels, weight=torch.Tensor([1, 20]).to(cache_logits_train.device))
        optimizer_values.zero_grad()
        optimizer_keys.zero_grad()
        loss.backward()
        optimizer_values.step()
        optimizer_keys.step()

        if iter % 500 == 0:
            with torch.no_grad():
                cache_logits_test = attention_functional_batch(test_features, cache_keys, cache_values)

                tip_test_auc = utliz.cal_auc(test_labels, cache_logits_test[:, 1])
                pseudo_label_auc = utliz.cal_auc(cache_values_unlabeled_GT, torch.sigmoid(cache_values_learnable).detach().cpu())

                if tip_test_auc > best_cache_auc_test:
                    best_cache_auc_test = tip_test_auc
                    best_cache_logits_test = cache_logits_test
                    best_cache_keys = cache_keys.detach().cpu()
                    best_cache_values = cache_values.detach().cpu()
                    
            print('Iter: {}, Loss: {:.4f}, Train AUC: {:.4f}, Test AUC: {:.4f}'.format(iter, loss, tip_train_auc, tip_test_auc))
            if writer is not None:
                writer.add_scalar('train_loss', loss, iter)
                writer.add_scalar('train_instance_AUC', tip_train_auc, iter)
                writer.add_scalar('test_instance_AUC', tip_test_auc, iter)
                # writer.add_histogram('cache_values_learnable', torch.sigmoid(cache_values_learnable).detach().cpu(), iter)
                # writer.add_histogram('cache_values_learnable_P', torch.sigmoid(cache_values_learnable).detach().cpu()[torch.where(cache_values_unlabeled_GT==1)], iter)
                # writer.add_histogram('cache_values_learnable_N', torch.sigmoid(cache_values_learnable).detach().cpu()[torch.where(cache_values_unlabeled_GT==0)], iter)
                writer.add_scalar('cache_values_learnable_AUC', pseudo_label_auc, iter)

    merged_auc, merged_logits, best_alpha = search_hp_onlyAlpha(1, 100, clip_logits_test, best_cache_logits_test, test_labels)
    return (best_cache_logits_test, clip_logits_test,   merged_logits,
            best_cache_auc_test,    clip_auc,           merged_auc,
            best_cache_keys,        best_cache_values,  best_alpha)


# Final FAST Model

def fast(args, name, train_loader_bag, val_ds_return_bag):
    # Build cache model through iterating over few-shot training set
    (cache_keys_unlabeled, 
    cache_values_unlabeled, 
    cache_corresponding_slide_label_unlabeled, 
    cache_corresponding_slide_index_unlabeled, 
    cache_keys_labeled, 
    cache_values_labeled, 
    cache_corresponding_slide_label_labeled, 
    cache_corresponding_slide_index_labeled, 
    cache_values_unlabeled_GT) = build_MIL_Adapter_cache_model(train_loader_bag, downsample_neg_instances=args.downsample_neg_instances)
    test_features, test_labels = torch.from_numpy(val_ds_return_bag.all_patches).cuda(), torch.from_numpy(val_ds_return_bag.patch_label).cuda()

    # Setup CLIP model and prompts
    clip_model = load_clip_to_cpu(backbone_name='RN50').cuda()
    (prompts_common_templates, 
    prompts_pathology_template, 
    prompts_pathology_template_withDescription) = get_patch_level_prompts_forCAMELYON(tissue_type='simple')
    classifer_common_templates = clip_classifier(prompts_common_templates, clip_model)
    classifer_pathology_template = clip_classifier(prompts_pathology_template, clip_model)
    classifer_pathology_template_withDescription = clip_classifier(prompts_pathology_template_withDescription, clip_model)

    # Build logger
    print(name, flush=True)
    # writer = SummaryWriter(log_dir=os.path.join("./runs_CAMELYON_withTextLearnable", name))
    writer = SummaryWriter(log_dir=os.path.join(args.exp_dir, name))

    ## Build learnable prompt and optimize it
    learnablePrompt_ctx_init = [
        # "A normal image patch with regularly shaped cells and smaller, lighter nuclei. * * * * * * * * * *",
        # "A tumor image patch with irregular cancerous cells and larger, darker nuclei. * * * * * * * * * *",
        "An image of evenly spaced, regular cells, suggesting healthy tissue. * * * * * * * * * *",
        # "Tissue with orderly and uniform cells, indicating no cancer.",
        "An image of disorganized, dense, and irregular cells, suggesting cancer. * * * * * * * * * *"
        # "Tissue with abnormal cells and chaotic structure, indicating malignancy.",
    ]
    instance_prompt_learner = PromptLearner(n_ctx=16, ctx_init=learnablePrompt_ctx_init, all_ctx_trainable=True, csc=True,
                                            classnames=["normal", "tumor"],
                                            clip_model='RN50', p_drop_out=0.0)
    model_text = Instance_CLIP(instance_prompt_learner).to('cuda:0')
    for param in model_text.text_encoder.parameters():
        param.requires_grad = False
    optimizer_text_branch = torch.optim.SGD(model_text.prompt_learner.parameters(), lr=0.001)

    train_ds_fewShotInstance = simple_instance_dataset(feat=cache_keys_labeled, label=cache_values_labeled)
    val_ds_allInstance = simple_instance_dataset(feat=test_features, label=test_labels)
    train_loader_fewShotInstance = torch.utils.data.DataLoader(train_ds_fewShotInstance, batch_size=1024, shuffle=True, num_workers=0, drop_last=False)
    val_loader_allInstance = torch.utils.data.DataLoader(val_ds_allInstance, batch_size=1024, shuffle=False, num_workers=0, drop_last=False)

    optimizer = Optimizer_TextBranch(
        model=model_text, train_loader=train_loader_fewShotInstance, test_loader=val_loader_allInstance,
        optimizer=optimizer_text_branch, writer=writer, num_epoch=50,
        dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer.optimize()
    classifer_pathology_after_optimizing = model_text.get_classifier().detach()
    del model_text, instance_prompt_learner

    # MIL_Adapter: 1. unlabeled instance values learnable; 2. only unlabeled instance keys learnable
    (cache_logits, 
     clip_logits, 
     merged_logits, 
     cache_instanceAUC, 
     clip_instanceAUC, 
     merge_instanceAUC,
     cache_keys,
     cache_values,
     best_alpha) = run_tip_adapter_key_value_learnable(
        cache_keys_unlabeled, 
        cache_values_unlabeled, 
        cache_keys_labeled, 
        cache_values_labeled,
        test_features, 
        test_labels,
        # clip_weights=classifer_pathology_template_withDescription,  # choice 1: use handcraft prompt
        clip_weights=classifer_pathology_after_optimizing,            # choice 2: use learned prompt
        cache_values_unlabeled_GT=cache_values_unlabeled_GT,
        epoch=args.epochs, lr_value=args.lr_values, lr_key=args.lr_keys, writer=writer, batch_size=args.batch_size)
    cache_bagAUC = gather_instance_prediction_and_pred_bag(cache_logits, val_ds_return_bag.patch_corresponding_slide_index, val_ds_return_bag.patch_corresponding_slide_label)
    clip_bagAUC = gather_instance_prediction_and_pred_bag(clip_logits, val_ds_return_bag.patch_corresponding_slide_index, val_ds_return_bag.patch_corresponding_slide_label)
    merged_bagAUC = gather_instance_prediction_and_pred_bag(merged_logits, val_ds_return_bag.patch_corresponding_slide_index, val_ds_return_bag.patch_corresponding_slide_label)
    print("cache_InsAUC: {:.4f}, \nCLIP_InsAUC: {:.4f}, \nmerge_InsAUC: {:.4f}, \n".format(cache_instanceAUC, clip_instanceAUC, merge_instanceAUC))
    print("cache_BagAUC: {:.4f}, \nCLIP_BagAUC: {:.4f}, \nmerge_BagAUC: {:.4f}, \n".format(cache_bagAUC, clip_bagAUC, merged_bagAUC))
    writer.add_scalar('cache_InsAUC', cache_instanceAUC, 0)
    writer.add_scalar('merge_InsAUC', merge_instanceAUC, 0)
    writer.add_scalar('clip_InsAUC', clip_instanceAUC, 0)
    writer.add_scalar('cache_BagAUC', cache_bagAUC, 0)
    writer.add_scalar('merge_BagAUC', merged_bagAUC, 0)
    writer.add_scalar('clip_BagAUC', clip_bagAUC, 0)
    
    # import pdb; pdb.set_trace()
    
    saved_tensor = {
        # "cache_keys": torch.Tensor(cache_keys), 
        # "cache_values": torch.Tensor(cache_values), 
        # "test_features": torch.Tensor(test_features),
        "test_labels": torch.Tensor(test_labels),
        # "best_alpha": torch.tensor(best_alpha),
        # "clip_weights": torch.Tensor(classifer_pathology_after_optimizing),
        
        "cache_logits": torch.Tensor(cache_logits), 
        "clip_logits": torch.Tensor(clip_logits), 
        "merged_logits": torch.Tensor(merged_logits), 
        
        "instance_corresponding_slide_index": torch.Tensor(val_ds_return_bag.patch_corresponding_slide_index),
        "instance_corresponding_slide_label": torch.Tensor(val_ds_return_bag.patch_corresponding_slide_label),
        
        "cache_instanceAUC": torch.tensor(cache_instanceAUC), 
        "clip_instanceAUC": torch.tensor(clip_instanceAUC), 
        "merge_instanceAUC": torch.tensor(merge_instanceAUC),
        
        "cache_bagAUC": torch.tensor(cache_bagAUC), 
        "clip_bagAUC": torch.tensor(clip_bagAUC), 
        "merge_bagAUC": torch.tensor(merged_bagAUC),
    }
    
    torch.save(saved_tensor, os.path.join(args.exp_dir, name, 'save.pt'))
    print('Saved!')
