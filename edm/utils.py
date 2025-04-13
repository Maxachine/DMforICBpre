import torch
import torch.nn as nn
import functools
import numpy as np
import pdb
  
device = 'cpu'

sigma_min=0.002
sigma_max=80
sigma_data = 1
P_mean=-1.2
P_std=1.2

def loss_fn(model, x):
    ln_sigma = torch.randn(x.shape[0], device=x.device)*P_std + P_mean
    sigma = ln_sigma.exp()
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    #     y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
#     pdb.set_trace()
    n = torch.randn_like(x) * sigma.unsqueeze(1)
    con = x[:,:-1]
    input = torch.cat([x + n, con], dim = 1)
    
    D_yn = model(input, sigma)
    loss = weight.unsqueeze(1) * ((D_yn - x) ** 2)
    loss = loss.sum()
    # loss6 = (D_yn[:,:-1] - x[:,:-1]) ** 2
    # loss7 = (D_yn[:,-1] - x[:,-1]) ** 2
    # # pdb.set_trace()
    # loss = (weight.unsqueeze(1) * loss6).sum() + 5*(weight.unsqueeze(1) * loss7).sum()   # 增大第七维损失的权重
    return loss
    