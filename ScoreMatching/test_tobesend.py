from likelihood import ode_likelihood
from train import MedicalDataset
import torch
from torch.utils.data import DataLoader
from utils import marginal_prob_std_fn,device,diffusion_coeff_fn
from model import ScoreNet
from sklearn.metrics import roc_auc_score
import os
import csv
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

batch_size = 2000

# file_path_train = 'datatype11_train.txt'
file_path_train = 'train_type123.txt'
# file_path = 'test.txt'
dataset_train = MedicalDataset(file_path_train)

file_path = 'test_type123.txt'
# file_path = 'medicaldata_train4.txt'
# file_path = 'test.txt'
dataset = MedicalDataset(file_path)
dataset.mean_vals = dataset_train.mean_vals
dataset.std_vals = dataset_train.std_vals
dataset.normalize()
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def compute_likelihood_and_save(data_loader, score_model, marginal_prob_std, diffusion_coeff,start_t):
    true_labels = []  
    predicted_labels = [] 
    for batch_idx, (x, true_label) in enumerate(data_loader):
        x = x.to(device)  # Move data to the same device as the model

        x_0 = x.clone()  # Copy the data to modify it without affecting the original
        x_1 = x.clone()
        x_1[..., -1] = (1-dataset.mean_vals[-1])/dataset.std_vals[-1]
        # pdb.set_trace()
        
        # Compute likelihoods (probabilities) for both cases
        prob_0, prior_logp0,delta_logp0 = ode_likelihood(x_0, score_model, marginal_prob_std, diffusion_coeff,start_t=start_t, device=device)
        prob_1, prior_logp1,delta_logp1 = ode_likelihood(x_1, score_model, marginal_prob_std, diffusion_coeff,start_t=start_t, device=device)
        # Compare the likelihoods and output 1 or 0 accordingly
        for i in range(len(x)):
            predicted_labels.append(prob_1[i].item() / (prob_0[i].item()+prob_1[i].item()))
            true_labels.append(true_label[i].item())

    auc_score = roc_auc_score(true_labels, predicted_labels)

    return auc_score

if __name__ == '__main__':
    start_t = 0.7
    model_folder = 'checkpoints'
    auc_dict = {}
    counter = 1
    for file_name in os.listdir(model_folder):
        if file_name.endswith('.pth'):
            score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn,input_dim=7)
            score_model = score_model.to(device)
            pretrained_ckpt = torch.load(f'./checkpoints/{file_name}', map_location=device)
            ckpt = {k.replace('module.', ''): v for k, v in pretrained_ckpt.items()}
            score_model.load_state_dict(ckpt)
            print(f'Evaluating the {counter}th model: {file_name} ...')
            counter = counter + 1 
            auc = compute_likelihood_and_save(data_loader, score_model,marginal_prob_std_fn,diffusion_coeff_fn, start_t=start_t)
            auc_dict[file_name] = auc
    with open("results_of_models.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "AUC"]) 
        for key, value in auc_dict.items():
            writer.writerow([key, value])  
    print("All models evaluated. Done.")
    
