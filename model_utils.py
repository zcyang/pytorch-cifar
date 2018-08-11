"""Utils for models."""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


def set_bn_to_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def extract_feature(data, model, remove_top=True):
    model.eval()
    if remove_top:
        model = nn.Sequential(*[l for l in model.children()][:-1])

    if isinstance(data, torch.Tensor):
        feature = model(data.to("cuda")).mean(-1).mean(-1).cpu().data
        return feature
    elif isinstance(data, torch.utils.data.DataLoader):
        features = []
        for batch_idx, (inputs, targets) in enumerate(data):
            inputs = inputs.to("cuda")
            features.append(model(inputs).mean(-1).mean(-1).cpu().data)

        return torch.cat(features, dim=0)


def get_nearest_neighbors(features, query, similarity="l2", top_k=100):
    if similarity == "l2":
        distance = torch.norm(features - query, dim=1)
    elif similarity == "cos":
        feature_norm = features.norm(p=2, dim=1, keepdim=True)
        feature_normalized = features.div(feature_norm)
        query_norm = query.norm(p=2, dim=1, keepdim=True)
        query_normalized = query.div(query_norm)
        distance = torch.mm(feature_normalized, query_normalized.permute(1, 0))
        distance = distance.view(distance.numel())

    sorted_distance, indices = torch.sort(distance, descending=True, dim=0)

    return indices[:top_k]

    




