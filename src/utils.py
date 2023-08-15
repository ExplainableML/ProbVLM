import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
from tqdm import tqdm_notebook
import numpy as np
import torch
import torch.nn as nn

import clip
import ds 
from ds import prepare_coco_dataloaders, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders
from tqdm import tqdm
from losses import *

def get_pred_ranks(q_features, g_features, recall_ks=(1,5,10)):
    """
    Args:
        q_features (torch.tensor, size=[#query, embedding dim])
        g_features (torch.tensor, size=[#gallery, embedding dim])
        recall_ks (list[:int] or tuple[:int])
    Returns:
        pred_ranks_all (np.ndarray, size=[#query, max(recall_ks)]):
            data indices of similarity ranking in descending order
    """
    max_k = max(recall_ks)
    n_q_features = len(q_features)

    pred_ranks_all = []
    for idx in range(n_q_features):
        sims = (q_features[idx : idx + 1] @ g_features.t())
        _, pred_ranks = torch.topk(sims, k=max_k, dim=-1)
        pred_ranks_all.append(pred_ranks)
    pred_ranks_all = torch.cat(pred_ranks_all, dim=0).cpu().numpy()

    return pred_ranks_all


def get_recall(pred_ranks_all, recall_ks=(1,5,10), n_gallery_per_query=5):
    """
    Args:
        pred_ranks_all (np.ndarray, size=[#query, max(recall_ks)]): 
            data indices of similarity ranking in descending order
        recall_ks (list[:int] or tuple[:int])
        n_gallery_per_query (float)
    Returns:
        recall_scores (list[:float]): list of recall@k
    """
    existence = lambda arr1, arr2: any([i in arr2 for i in arr1])
    def gt_idxs(query_idx):
        if n_gallery_per_query >= 1:
            return np.arange(query_idx * n_gallery_per_query, 
                             (query_idx + 1) * n_gallery_per_query)
        else:
            return np.array([int(query_idx * n_gallery_per_query)])

    recall_scores = []
    for recall_k in recall_ks:
        score = sum([existence(pred_ranks[:recall_k], gt_idxs(query_idx))
                     for query_idx, pred_ranks in enumerate(pred_ranks_all)]) / len(pred_ranks_all)
        recall_scores.append(score)

    return recall_scores


def get_recall_COCOFLICKR(pred_ranks_all, recall_ks=(1,5,10), n_gallery_per_query=5, q_idx=None):
    """
    Args:
        pred_ranks_all (np.ndarray, size=[#query, max(recall_ks)]): 
            data indices of similarity ranking in descending order
        recall_ks (list[:int] or tuple[:int])
        n_gallery_per_query (float)
    Returns:
        recall_scores (list[:float]): list of recall@k
    """
    existence = lambda arr1, arr2: any([i in arr2 for i in arr1])
    def gt_idxs(query_idx):
        if n_gallery_per_query >= 1:
            return np.arange(query_idx * n_gallery_per_query, 
                             (query_idx + 1) * n_gallery_per_query)
        else:
            return np.array([int(query_idx * n_gallery_per_query)])

    recall_scores = []
    for recall_k in recall_ks:
        score = sum([existence(pred_ranks[:recall_k], q_idx)
                     for query_idx, pred_ranks in enumerate(pred_ranks_all)]) / len(pred_ranks_all)
        recall_scores.append(score)

    return recall_scores


def new_recall(pred_ranks_all,recall_ks=(1,5,10),q_classes_all=None,g_classes_all=None):
    recall_scores = []
    for recall_k in recall_ks:
        corr=0
        total = len(pred_ranks_all)
        for i in range(len(pred_ranks_all)):
            gt_class = q_classes_all[i]
            pred_classes = [g_classes_all[j] for j in pred_ranks_all[i][:recall_k]]
            if gt_class in pred_classes:
                corr+=1
        recall_scores.append(corr/total)

    return recall_scores

def load_data_loader(dataset, data_dir, dataloader_config):
    prepare_loaders = {
        'coco': prepare_coco_dataloaders,
        'flickr': prepare_flickr_dataloaders,
        'CUB':prepare_cub_dataloaders,
        'FLO':prepare_flo_dataloaders
    }[dataset]
    if dataset == 'CUB':
        loaders = prepare_loaders(
            dataloader_config,
            dataset_root=data_dir,
            caption_root=data_dir+'/text_c10',
            vocab_path='ds/vocabs/cub_vocab.pkl')
    elif dataset == 'FLO':
        loaders = prepare_loaders(
            dataloader_config,
            dataset_root=data_dir,
            caption_root=data_dir+'/text_c10',)
    else:
        loaders = prepare_loaders(
            dataloader_config,
            dataset_root=data_dir,
            vocab_path='ds/vocabs/coco_vocab.pkl')
    return loaders

def load_model(device, model_path=None):
    # load zero-shot CLIP model
    model, _ = clip.load(name='ViT-B/32',
                         device=device,
                         loss_type='contrastive')
    if model_path is None:
        # Convert the dtype of parameters from float16 to float32
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    else:
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


### training and evaluation
def emb_mae(x1, x2):
    m = torch.abs(x1-x2).mean()
    return m

def emb_mse(x1, x2):
    m = torch.pow(torch.abs(x1-x2),2).mean()
    return m

def get_GGuncer(x_alpha, x_beta, c1=3, c2=2.8):
    a = 1/(x_alpha + 1e-5)
    a = torch.clip(a, min=1e-4, max=5)
    b = x_beta + 0.1
    b = torch.clip(b, min=0.1, max=5)
    u = (a**2)*torch.exp(torch.lgamma(3/b))/torch.exp(torch.lgamma(2.8/b))
    return u
    
def multi_fwpass_ProbVLM(
    BayesCap_Net,
    xfI, xfT,
    n_fw=15
):
    list_i_mu, list_i_alpha, list_i_beta, list_i_uncer = [], [], [], []
    list_t_mu, list_t_alpha, list_t_beta, list_t_uncer = [], [], [], []
    BayesCap_Net.eval()
    for layer in BayesCap_Net.children():
        for l in layer.modules():
            if(isinstance(l, nn.Dropout)):
                # print(l)
                l.p = 0.3
                l.train()

    for i in range(n_fw):
        (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = BayesCap_Net(xfI, xfT)
        list_i_mu.append(img_mu.unsqueeze(0))
        list_i_alpha.append(img_1alpha.unsqueeze(0))
        list_i_beta.append(img_beta.unsqueeze(0))
        list_i_uncer.append(get_GGuncer(img_1alpha, img_beta))
        ##
        list_t_mu.append(txt_mu.unsqueeze(0))
        list_t_alpha.append(txt_1alpha.unsqueeze(0))
        list_t_beta.append(txt_beta.unsqueeze(0))
        list_t_uncer.append(get_GGuncer(txt_1alpha, txt_beta, c1=3, c2=2))
    ##
    i_mu = torch.cat(list_i_mu, dim=0)
    i_alpha = torch.cat(list_i_alpha, dim=0)
    i_beta = torch.cat(list_i_beta, dim=0)
    i_uncer = torch.cat(list_i_uncer, dim=0)
    #
    t_mu = torch.cat(list_t_mu, dim=0)
    t_alpha = torch.cat(list_t_alpha, dim=0)
    t_beta = torch.cat(list_t_beta, dim=0)
    t_uncer = torch.cat(list_t_uncer, dim=0)
    ##
    i_mu_m, i_mu_v = torch.mean(i_mu, dim=0), torch.var(i_mu, dim=0)
    i_alpha_m, i_alpha_v = torch.mean(i_alpha, dim=0), torch.var(i_alpha, dim=0)
    i_beta_m, i_beta_v = torch.mean(i_beta, dim=0), torch.var(i_beta, dim=0)
    i_uncer_m, i_uncer_v = torch.mean(i_uncer, dim=0), torch.var(i_uncer, dim=0)
    # i_v = i_mu_v + i_alpha_v + 1/i_beta_v
    i_v = (i_uncer_v * i_mu_v)**(1/2)
    ##
    t_mu_m, t_mu_v = torch.mean(t_mu, dim=0), torch.var(t_mu, dim=0)
    t_alpha_m, t_alpha_v = torch.mean(t_alpha, dim=0), torch.var(t_alpha, dim=0)
    t_beta_m, t_beta_v = torch.mean(t_beta, dim=0), torch.var(t_beta, dim=0)
    t_uncer_m, t_uncer_v = torch.mean(t_uncer, dim=0), torch.var(t_uncer, dim=0)
    # t_v = t_mu_v + t_alpha_v + 1/t_beta_v
    t_v = (t_uncer_v * t_mu_v)**(1/2)
    
    return (i_mu_m, i_alpha_m, i_beta_m, i_v), (t_mu_m, t_alpha_m, t_beta_m, t_v)
 
def get_features_uncer_ProbVLM(
    CLIP_Net,
    BayesCap_Net,
    t_loader,
):
    r_dict= {
        'i_f': [],
        't_f': [],
        'ir_f': [],
        'tr_f': [],
        'i_au':[],
        'i_eu':[],
        'i_u': [],
        't_au':[],
        't_eu':[],
        't_u': [],
        'classes': []
    }
    # extract all features
    with torch.no_grad():
        for i_inputs, t_inputs, class_labels, _ in tqdm(t_loader):
            r_dict['classes'].extend(class_labels.cpu().tolist())
            n_batch = i_inputs.shape[0]
            i_inputs, t_inputs = i_inputs.to(device), t_inputs.to(device)
            outputs = CLIP_Net(i_inputs, t_inputs)
            #recons
            outs = multi_fwpass_BayesCap(BayesCap_Net, outputs[0], outputs[1])
            
            for j in range(n_batch):
                r_dict['i_f'].append(outputs[0][j,:])
                r_dict['t_f'].append(outputs[1][j,:])
                r_dict['ir_f'].append(outs[0][0][j,:])
                r_dict['tr_f'].append(outs[1][0][j,:])
                u = get_GGuncer(outs[0][1][j,:], outs[0][2][j,:], c1=3, c2=2.8)
                r_dict['i_au'].append(u)
                r_dict['i_eu'].append(outs[0][3][j,:])
                r_dict['i_u'].append(u+outs[0][3][j,:])
                u = get_GGuncer(outs[1][1][j,:], outs[1][2][j,:], c1=3, c2=2.8)
                r_dict['t_au'].append(u)
                r_dict['t_eu'].append(outs[1][3][j,:])
                r_dict['t_u'].append(u+outs[1][3][j,:])
    
    return r_dict


def sort_wrt_uncer(r_dict):
    orig_v_idx = {}
    for i in range(len(r_dict['i_u'])):
        orig_v_idx[i] = torch.mean(r_dict['i_u'][i]).item()
    sort_v_idx = sorted(orig_v_idx.items(), key=lambda x: x[1], reverse=True)
    
    orig_t_idx = {}
    for i in range(len(r_dict['t_u'])):
        orig_t_idx[i] = 1/torch.mean(r_dict['t_u'][i]).item()
    sort_t_idx = sorted(orig_t_idx.items(), key=lambda x: x[1], reverse=True)
    
    return sort_v_idx, sort_t_idx

def create_uncer_bins_eq_spacing(sort_idx, n_bins=10):
    max_uncer = sort_idx[0][1]
    min_uncer = sort_idx[-1][1]
    
    step_uncer = np.linspace(min_uncer, max_uncer, num=n_bins)
    print('uncer_steps: ', step_uncer)
    
    ret_bins = {'bin{}'.format(i):[] for i in range(n_bins)}
    
    for val in sort_idx:
        idx, uv = val
        for j, step in enumerate(step_uncer):
            if uv<=step:
                ret_bins['bin{}'.format(j)].append(val)
    return ret_bins

def create_uncer_bins_eq_samples(sort_idx, n_bins=10):
    sort_idx = sort_idx[::-1]
    ret_bins = {'bin{}'.format(i):[] for i in range(n_bins)}
    n_len = len(sort_idx)
    z = 0
    for i, val in enumerate(sort_idx):
        if i<=z+(n_len//n_bins):
            ret_bins['bin{}'.format(int(z//(n_len/n_bins)))].append(val)
        else:
            z += n_len//n_bins
    return ret_bins