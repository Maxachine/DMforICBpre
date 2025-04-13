from likelihood import ode_likelihood
from train import MedicalDataset
import torch
from torch.utils.data import DataLoader
from utils import marginal_prob_std_fn,device,diffusion_coeff_fn
from model import ScoreNet
from sklearn.metrics import roc_auc_score
import pdb
from sklearn.metrics import average_precision_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

batch_size = 2000

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn,input_dim=7))
score_model = score_model.to(device)
# ckpt = torch.load('type11_ckpt.pth', map_location=device)
ckpt = torch.load('./checkpoints/1110010010001.pth', map_location=device)
score_model.load_state_dict(ckpt)

# score_model1 = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn,input_dim=6))
# score_model1 = score_model1.to(device)
# # ckpt1 = torch.load('type11_ckpt1.pth', map_location=device)
# ckpt1 = torch.load('type123_ckpt1.pth', map_location=device)
# score_model1.load_state_dict(ckpt1)

# file_path_train = 'datatype11_train.txt'
file_path_train = './Selected_train_files/1110010010001.txt'
# file_path = 'test.txt'
dataset_train = MedicalDataset(file_path_train)

file_path = './Selected_test_files/1110010010001.txt'
# file_path = 'medicaldata_train4.txt'
# file_path = 'test.txt'
dataset = MedicalDataset(file_path)
dataset.mean_vals = dataset_train.mean_vals
dataset.std_vals = dataset_train.std_vals
dataset.normalize()
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# vae = VAE(input_dim=6, hidden_dim=256, latent_dim=5)
# vae.load_state_dict(torch.load('vae_model.pth'))
# vae.eval()

def compute_likelihood_and_save(data_loader, score_model, marginal_prob_std, diffusion_coeff,start_t, output_file='likelihood.txt'):
    true_labels = []  
    predicted_labels = [] 
    predicted_labels1 = [] 
    with open(output_file, 'w') as f:
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

            # Optionally, print progress or debug info
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(data_loader)}")

    # pdb.set_trace()
    with open(output_file, 'w') as f:
    # 使用 writelines() 写入，每个元素后加换行符
        for item in predicted_labels:
            f.write(str(item) + '\n')

    auc_score = roc_auc_score(true_labels, predicted_labels)

    return auc_score

if __name__ == '__main__':
    # compute_likelihood_and_save(data_loader, score_model, score_model1, output_file='likelihood.txt')
    best_auc = 0
    best_start_t = 30
    for start_t in [0.72,0.74,0.76,0.78]: #np.arange(1.4, 0.6, -0.1):
        print(f"Evaluating start_t = {start_t} ...")
        output_file = f'./Selected_results/1110010010001_{start_t}.txt'
        auc = compute_likelihood_and_save(data_loader, score_model,marginal_prob_std_fn,diffusion_coeff_fn, start_t=start_t, output_file=output_file)

        if auc > best_auc:
            best_auc = auc
            best_start_t = start_t
        print(f"Start_t {start_t} evaluated. AUC {auc}")

    print(f"Best start_t: {best_start_t} with AUC: {best_auc}")