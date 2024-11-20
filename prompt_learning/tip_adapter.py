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



def search_hp(search_scale, search_step, cache_keys, cache_values, features, labels, clip_weights, batch_size=4096):

    beta_list = [i * (search_scale[0] - 0.1) / search_step[0] + 0.1 for i in range(search_step[0])]
    alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]

    best_auc = 0
    best_beta, best_alpha = 0, 0
    best_logits = None

    clip_logits = 100. * cal_matrix_mul(features, clip_weights)
    for beta in tqdm(beta_list, desc='Searching Hyperparameters'):
        for alpha in alpha_list:
            # affinity = cal_matrix_mul(features, cache_keys)
            # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

            num_batch = features.shape[0] // batch_size + 1
            output_ = torch.zeros((features.shape[0], 2)).cuda()
            for i in range(num_batch):
                affinity = cal_matrix_mul(features[i * batch_size: (i + 1) * batch_size], cache_keys, batch_size=batch_size)
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                output_[i * batch_size: (i + 1) * batch_size] = cache_logits
            cache_logits = output_

            tip_logits = clip_logits + cache_logits * alpha
            # acc = cls_acc(tip_logits, labels)
            tip_logits_normed = norm_logit(tip_logits)[:, 1]
            auc = utliz.cal_auc(labels, tip_logits_normed)

            if auc > best_auc:
                # print("New best setting, beta: {:.2f}, alpha: {:.2f}; AUC: {:.4f}".format(beta, alpha, auc))
                best_auc = auc
                best_beta = beta
                best_alpha = alpha
                best_logits = tip_logits_normed

    print("After searching, setting, beta: {:.2f}, alpha: {:.2f}".format(best_beta, best_alpha))
    print("After searching, the best AUC: {:.4f}.".format(best_auc))

    return best_auc, best_logits


def run_tip_adapter(cache_keys, cache_values, test_features, test_labels, clip_weights, batch_size=4096):
    cache_keys = cache_keys.to(clip_weights.device)
    cache_values = cache_values.to(clip_weights.device)
    test_features = test_features.to(clip_weights.device)
    test_labels = test_labels.to(clip_weights.device)

    # Zero-shot CLIP
    if clip_weights.shape[0] != test_features.shape[1]:
        clip_weights = clip_weights.T
    if cache_keys.shape[0] != test_features.shape[1]:
        cache_keys = cache_keys.T

    clip_logits = 100. * cal_matrix_mul(test_features, clip_weights)
    clip_logits_normed = norm_logit(clip_logits)[:, 1]
    clip_auc = utliz.cal_auc(test_labels, clip_logits_normed)

    # Tip-Adapter
    init_beta = 1
    init_alpha = 1.17
    beta, alpha = init_beta, init_alpha

    num_batch = test_features.shape[0] // batch_size + 1
    output_ = torch.zeros((test_features.shape[0], 2)).cuda()
    for i in range(num_batch):
        affinity = cal_matrix_mul(test_features[i * batch_size: (i + 1) * batch_size], cache_keys, batch_size=batch_size)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        output_[i * batch_size: (i + 1) * batch_size] = cache_logits
    cache_logits = output_

    tip_logits = clip_logits + cache_logits * alpha
    tip_logits_normed = norm_logit(tip_logits)[:, 1]
    tip_auc = utliz.cal_auc(test_labels, tip_logits_normed)

    cache_logits_normed = norm_logit(cache_logits)[:, 1]
    cache_auc = utliz.cal_auc(test_labels, cache_logits_normed)

    # Search Hyperparameters
    search_scale = [7, 3]
    search_step = [100, 10]
    # search_step = [40, 4]
    # search_step = [20, 2]
    print("Seach Hyperparameters: scale: {}, step: {}".format(search_scale, search_step))

    tip_HP_auc, tip_logits_withHP = search_hp(search_scale, search_step, cache_keys, cache_values, test_features, test_labels, clip_weights)
    return cache_logits_normed, tip_logits_normed, tip_logits_withHP, cache_auc, tip_auc, tip_HP_auc


def run_tip_adapter_F(cache_keys, cache_values, test_features, test_labels, clip_weights, batch_size=4096):
    cache_keys = cache_keys.to(clip_weights.device)
    cache_values = cache_values.to(clip_weights.device)
    test_features = test_features.to(clip_weights.device)
    test_labels = test_labels.to(clip_weights.device)

    # Zero-shot CLIP
    if clip_weights.shape[0] != test_features.shape[1]:
        clip_weights = clip_weights.T
    if cache_keys.shape[0] != test_features.shape[1]:
        cache_keys = cache_keys.T

    clip_logits_test = 100. * cal_matrix_mul(test_features, clip_weights)
    clip_logits_normed_test = norm_logit(clip_logits_test)[:, 1]
    clip_auc_test = utliz.cal_auc(test_labels, clip_logits_normed_test)

    # Tip-Adapter
    cache_keys = torch.nn.Parameter(cache_keys)
    optimizer_keys = torch.optim.Adam([cache_keys], lr=0.001, eps=1e-4)
    train_features = cache_keys.T.detach()
    train_labels = cache_values
    clip_logits_train = 100. * cal_matrix_mul(train_features, clip_weights)

    init_beta = 1
    init_alpha = 1.17
    beta, alpha = init_beta, init_alpha

    for iter in range(20):

        num_batch = test_features.shape[0] // batch_size + 1
        output_train = torch.zeros((train_features.shape[0], 2)).cuda()
        for i in range(num_batch):
            affinity = cal_matrix_mul(train_features[i * batch_size: (i + 1) * batch_size], cache_keys,
                                      batch_size=batch_size)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            output_train[i * batch_size: (i + 1) * batch_size] = cache_logits
        cache_logits = output_train

        tip_logits_train = clip_logits_train + cache_logits * alpha
        tip_logits_normed_train = norm_logit(tip_logits_train)[:, 1]
        tip_auc_train = utliz.cal_auc(train_labels[:, 1], tip_logits_normed_train)

        loss = F.cross_entropy(tip_logits_train, train_labels)
        optimizer_keys.zero_grad()
        loss.backward()
        optimizer_keys.step()

        with torch.no_grad():
            num_batch = test_features.shape[0] // batch_size + 1
            output_test = torch.zeros((test_features.shape[0], 2)).cuda()
            for i in range(num_batch):
                affinity = cal_matrix_mul(test_features[i * batch_size: (i + 1) * batch_size], cache_keys, batch_size=batch_size)
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                output_test[i * batch_size: (i + 1) * batch_size] = cache_logits
            cache_logits = output_test

            tip_logits_test = clip_logits_test + cache_logits * alpha
            tip_logits_normed_test = norm_logit(tip_logits_test)[:, 1]
            tip_auc_test = utliz.cal_auc(test_labels, tip_logits_normed_test)

        print('Iter: {}, Loss: {:.4f}, Train AUC: {:.4f}, Test AUC: {:.4f}'.format(iter, loss, tip_auc_train, tip_auc_test))

    # Search Hyperparameters
    search_scale = [7, 3]
    search_step = [100, 10]
    # search_step = [40, 4]
    # search_step = [20, 2]
    print("Seach Hyperparameters: scale: {}, step: {}".format(search_scale, search_step))

    with torch.no_grad():
        tip_HP_auc, tip_logits_withHP = search_hp(search_scale, search_step, cache_keys, cache_values, test_features, test_labels, clip_weights, batch_size=batch_size)
    return tip_logits_withHP, tip_HP_auc


### Final model 

def tip_adapter(args, name, train_loader_bag, val_ds_return_bag):
    # Build cache model through iterating over few-shot training set
    cache_keys_unlabeled, cache_values_unlabeled, cache_corresponding_slide_label_unlabeled, cache_corresponding_slide_index_unlabeled, \
        cache_keys_labeled, cache_values_labeled, cache_corresponding_slide_label_labeled, cache_corresponding_slide_index_labeled, cache_values_unlabeled_GT = \
        build_MIL_Adapter_cache_model(train_loader_bag, downsample_neg_instances=args.downsample_neg_instances)
    test_features, test_labels = torch.from_numpy(val_ds_return_bag.all_patches).cuda(), torch.from_numpy(val_ds_return_bag.patch_label).cuda()

    # Setup CLIP model and prompts
    clip_model = load_clip_to_cpu(backbone_name='RN50').cuda()
    prompts_common_templates, prompts_pathology_template, prompts_pathology_template_withDescription = get_patch_level_prompts_forCAMELYON(tissue_type='simple')
    classifer_common_templates = clip_classifier(prompts_common_templates, clip_model)
    classifer_pathology_template = clip_classifier(prompts_pathology_template, clip_model)
    classifer_pathology_template_withDescription = clip_classifier(prompts_pathology_template_withDescription, clip_model)

    # Build logger
    print(name, flush=True)
    writer = SummaryWriter(log_dir=os.path.join("./runs_CAMELYON_withTextLearnable", name))

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

    optimizer = Optimizer_TextBranch(model=model_text, train_loader=train_loader_fewShotInstance, test_loader=val_loader_allInstance,
                          optimizer=optimizer_text_branch,
                          writer=writer, num_epoch=50,
                          dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer.optimize()
    classifer_pathology_after_optimizing = model_text.get_classifier().detach()
    del model_text, instance_prompt_learner

    # original Tip-Adapter
    cache_logits, tip_logits, tip_logits_withHP, cache_intanceAUC, tip_instanceAUC, tipHP_instanceAUC = run_tip_adapter(
        cache_keys_labeled, torch.nn.functional.one_hot(cache_values_labeled).float(),
        test_features, test_labels, classifer_pathology_template_withDescription, batch_size=args.batch_size)
    cache_bagAUC = gather_instance_prediction_and_pred_bag(cache_logits, val_ds_return_bag.patch_corresponding_slide_index, val_ds_return_bag.patch_corresponding_slide_label)
    tip_bagAUC = gather_instance_prediction_and_pred_bag(tip_logits, val_ds_return_bag.patch_corresponding_slide_index, val_ds_return_bag.patch_corresponding_slide_label)
    tipHP_bagAUC = gather_instance_prediction_and_pred_bag(tip_logits_withHP, val_ds_return_bag.patch_corresponding_slide_index, val_ds_return_bag.patch_corresponding_slide_label)
    print("tip_woCLIP_InsAUC: {:.4f}, \ntip_InsAUC: {:.4f}, \ntipHP_InsAUC: {:.4f}".format(cache_intanceAUC, tip_instanceAUC, tipHP_instanceAUC))
    print("tip_woCLIP_BagAUC: {:.4f}, \ntip_BagAUC: {:.4f}, \ntipHP_BagAUC: {:.4f}".format(cache_bagAUC, tip_bagAUC, tipHP_bagAUC))
    writer.add_scalar('tip_woCLIP_InsAUC', cache_intanceAUC, 0)
    writer.add_scalar('tip_InsAUC', tip_instanceAUC, 0)
    writer.add_scalar('tipHP_InsAUC', tipHP_instanceAUC, 0)
    writer.add_scalar('tip_woCLIP_BagAUC', cache_bagAUC, 0)
    writer.add_scalar('tip_BagAUC', tip_bagAUC, 0)
    writer.add_scalar('tipHP_BagAUC', tipHP_bagAUC, 0)


def tip_adapter_f(args, name, train_loader_bag, val_ds_return_bag):
    # Build cache model through iterating over few-shot training set
    cache_keys_unlabeled, cache_values_unlabeled, cache_corresponding_slide_label_unlabeled, cache_corresponding_slide_index_unlabeled, \
        cache_keys_labeled, cache_values_labeled, cache_corresponding_slide_label_labeled, cache_corresponding_slide_index_labeled, cache_values_unlabeled_GT = \
        build_MIL_Adapter_cache_model(train_loader_bag, downsample_neg_instances=args.downsample_neg_instances)
    test_features, test_labels = torch.from_numpy(val_ds_return_bag.all_patches).cuda(), torch.from_numpy(val_ds_return_bag.patch_label).cuda()

    # Setup CLIP model and prompts
    clip_model = load_clip_to_cpu(backbone_name='RN50').cuda()
    prompts_common_templates, prompts_pathology_template, prompts_pathology_template_withDescription = get_patch_level_prompts_forCAMELYON(tissue_type='simple')
    classifer_common_templates = clip_classifier(prompts_common_templates, clip_model)
    classifer_pathology_template = clip_classifier(prompts_pathology_template, clip_model)
    classifer_pathology_template_withDescription = clip_classifier(prompts_pathology_template_withDescription, clip_model)

    # Build logger
    print(name, flush=True)
    writer = SummaryWriter(log_dir=os.path.join("./runs_CAMELYON_withTextLearnable", name))

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

    optimizer = Optimizer_TextBranch(model=model_text, train_loader=train_loader_fewShotInstance, test_loader=val_loader_allInstance,
                          optimizer=optimizer_text_branch,
                          writer=writer, num_epoch=50,
                          dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer.optimize()
    classifer_pathology_after_optimizing = model_text.get_classifier().detach()
    del model_text, instance_prompt_learner

    # original Tip-Adapter-F
    tipF_logits_withHP, tipHPF_instanceAUC = run_tip_adapter_F(
        cache_keys_labeled, torch.nn.functional.one_hot(cache_values_labeled).float(),
        test_features, test_labels, classifer_pathology_template_withDescription, batch_size=args.batch_size)
    tipHPF_bagAUC = gather_instance_prediction_and_pred_bag(tipF_logits_withHP, val_ds_return_bag.patch_corresponding_slide_index, val_ds_return_bag.patch_corresponding_slide_label)
    print("tipHPF_InsAUC: {:.4f}".format(tipHPF_instanceAUC))
    print("tipHPF_BagAUC: {:.4f}".format(tipHPF_bagAUC))
    writer.add_scalar('tipHPF_InsAUC', tipHPF_instanceAUC, 0)
    writer.add_scalar('tipHPF_BagAUC', tipHPF_bagAUC, 0)