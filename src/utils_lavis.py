import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
from tqdm import tqdm_notebook
import numpy as np
import torch
import torch.nn as nn

import clip
import ds_lavis
from ds_lavis import prepare_coco_dataloaders, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders
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
    print('lavis!!!')
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
