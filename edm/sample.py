import torch
from torch.utils.data import DataLoader
from model import ScoreNet
from utils import device
from train import MedicalDataset
import itertools
import numpy as np
import pdb 
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from data_refiner import SimpleNN, refiner, RealData

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
 
def edm_sampler(
    net, latents, con, randn_like=torch.randn_like,
    num_steps=72, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=90, S_min=0.05, S_max=2, S_noise=1.002,batchsize=964
):
    # S_churn=0, S_min=0, S_max=float('inf'), S_noise=1
    # sigma_min = max(sigma_min, net.sigma_min)
    # sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        # denoised = net(x_hat, t_hat).to(torch.float64)
        with torch.no_grad():  # Ensure no gradient tracking
#             pdb.set_trace()
            denoised = net(torch.cat([x_hat.to(torch.float32), con],dim=1), t_hat.expand(batchsize).to(torch.float32)).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            with torch.no_grad():
                denoised = net(torch.cat([x_next.to(torch.float32), con],dim=1), t_next.expand(batchsize).to(torch.float32)).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        # Delete temporary variables to free up memory
        del x_cur, t_hat, x_hat, denoised, d_cur
        torch.cuda.empty_cache()  # Clear the cache after each step
    return x_next

def rbf_kernel(X, Y, gamma=1.0):
    """
    计算高斯径向基函数 RBF 核矩阵。
    
    参数：
        X (ndarray): 第一个数据集，形状为 (n_samples_X, n_features)。
        Y (ndarray): 第二个数据集，形状为 (n_samples_Y, n_features)。
        gamma (float): RBF核的参数, 默认为1.0。
    
    返回：
        K (ndarray): 计算得到的核矩阵，形状为 (n_samples_X, n_samples_Y)。
    """
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * sq_dists)

def compute_mmd(X, Y, gamma=1.0):
    """
    计算两个数据集之间的最大均值差异 MMD 。
    
    参数：
        X (ndarray): 第一个数据集，形状为 (n_samples_X, n_features)。
        Y (ndarray): 第二个数据集，形状为 (n_samples_Y, n_features)。
        gamma (float): RBF核的参数, 默认为1.0。
    
    返回：
        mmd (float): 计算得到的MMD值。
    """
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

def evaluate_sampler_fid(params, score_model, real_images, latents, labels):
    model, S_churn, S_min, S_max, S_noise = params
    generated_samples = edm_sampler(score_model, latents, real_images[:,:-1], S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise).numpy()
#     pdb.set_trace()
    
    real_images = np.array(real_images)
    fid = compute_mmd(real_images,generated_samples)
    performance, threshold = evaluate_performance(labels,generated_samples[:,-1])

    return -performance

def evaluate_performance(labels, pred):
    # pdb.set_trace()
    auc = roc_auc_score(labels, pred)
    prauc = average_precision_score(labels, pred)

    performance = []
    thresholds = np.linspace(0, 1, 1000)
    for threshold in thresholds:
        pred01 = [1 if p > threshold else 0 for p in pred]
        f1 = f1_score(labels, pred01)
        acc = accuracy_score(labels, pred01)
        performance_cur = (auc + prauc + f1 + acc) / 4.0
        performance.append(performance_cur)
    
    best_performance = max(performance)
    best_threshold = thresholds[np.argmax(performance)]
    return best_performance, best_threshold

device = 'cpu'
ckpt = torch.load('edm_ckpt_1.2.pth', map_location=device)
ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}

# score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = ScoreNet(con_dim=6, input_dim=7)
score_model = score_model.to(device)
score_model.load_state_dict(ckpt)

batch_size = 515#1 # 

file_path = 'test_type123.txt'
dataset = MedicalDataset(file_path)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
iterator = iter(data_loader)

#----------------------------------------------------------------------
# for i in range(515):
#     print(f"The {i+1}th data...")
#     real_images, labels = next(iterator)
#     latents = torch.randn(batch_size, 7, device=device)
#     error = 5
#     round = 1
#     while(error > 0.2):
#         # print(f"round: {round}")
#         round = round +1
#         if round > 500000:
#             break
#         latents = torch.randn(batch_size, 7, device=device)
#         samples = edm_sampler(score_model,latents,real_images[:,:-1],batchsize=batch_size)
#         samples = samples * dataset.std_vals + dataset.mean_vals
#         real = real_images * dataset.std_vals + dataset.mean_vals
#         error = torch.sum(torch.abs(samples[:,:-1]-real[:,:-1]))
#     if i==0:
#         samples_all = samples
#     else:
#         samples_all = torch.cat((samples_all,samples), dim=0)

#-------------------------------------------------------------------

# Generate samples using the specified sampler.
# error = 5
# round = 1
# while(error > 0.01):
#     print(f"round: {round}")
#     round = round +1
#     latents = torch.randn(batch_size, 7, device=device)
#     samples = edm_sampler(score_model,latents,real_images[:,:-1],batchsize=batch_size)
#     samples = samples * dataset.std_vals + dataset.mean_vals
#     real = real_images * dataset.std_vals + dataset.mean_vals
#     error = torch.sum(torch.abs(samples[:,0]-real[:,0]))
# print(samples)
# print(error)
    # pdb.set_trace()
#--------------------------------------------------------------------
N = 99
real_images, labels = next(iterator)
latents = torch.randn(batch_size, 7, device=device)
samples = edm_sampler(score_model,latents,real_images[:,:-1],batchsize=batch_size)
samples = samples * dataset.std_vals + dataset.mean_vals
for _ in range(N):
    latents = torch.randn(batch_size, 7, device=device)
    samples = samples + edm_sampler(score_model,latents,real_images[:,:-1],batchsize=batch_size) * dataset.std_vals + dataset.mean_vals
samples = samples / (N + 1)
# pdb.set_trace()
#--------------------------------------------------------------------
# with open('generated_data.txt', 'w') as f:
#     # 使用 writelines() 写入，每个元素后加换行符
#     for item in samples:
#         f.write(str(item) + '\n')

# 写入Excel文件
samples_list = samples.tolist()
df = pd.DataFrame(samples_list)
df.to_excel("generated_data_2avg_test.xlsx", index=False, header=False)
#----------------------------------------------------------------------
# # 加入 refiner 修正 samples
# print('Going into refiner part...')
# refiner_ckpt = torch.load('refiner_ckpt.pth', map_location=device)
# refiner_model = SimpleNN()
# refiner_model.to(device)
# refiner_model.load_state_dict(refiner_ckpt)

# samples = samples.to(torch.float32)
# real_dataset = RealData(file_path)
# dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
# real_data = next(iter(dataloader))

# refined_labels = refiner(refiner_model, samples, real_data[:,:-1])

# best_performance, best_threshold = evaluate_performance(labels, refined_labels)
# print('best performance:', best_performance)
# pred01 = [1 if p > best_threshold else 0 for p in refined_labels]
# print('auc:',roc_auc_score(labels, refined_labels))
# print('prauc:',average_precision_score(labels, refined_labels))
# print('F1:',f1_score(labels, pred01))
# print('accuracy:',accuracy_score(labels, pred01))
# print('MCC:',matthews_corrcoef(labels, pred01))
# print('BA:',balanced_accuracy_score(labels, pred01))
#----------------------------------------------------------------------
# 计算各项指标
# best_performance, best_threshold = evaluate_performance(labels, samples[:,-1])
# print('best performance:', best_performance)
# pred01 = [1 if p > best_threshold else 0 for p in samples[:,-1]]
# print('auc:',roc_auc_score(labels, samples[:,-1]))
# print('prauc:',average_precision_score(labels, samples[:,-1]))
# print('F1:',f1_score(labels, pred01))
# print('accuracy:',accuracy_score(labels, pred01))
# print('MCC:',matthews_corrcoef(labels, pred01))
# print('BA:',balanced_accuracy_score(labels, pred01))
#----------------------------------------------------------------------
# param_grid = {
#     'S_churn': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#     'S_min': [0, 0.005, 0.01, 0.02, 0.05, 0.1],
#     'S_max': [0.2, 0.5, 1, 2, 5, 10, 20, 50, 80],
#     'S_noise': [1.000, 1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009, 1.010],
# }

# # Define the grid search combinations
# param_combinations = list(itertools.product(*param_grid.values()))

# best_fid = float('inf')
# best_params_fid = None
# for params in param_combinations:
#     print(f"Evaluating params: {params}")
#     fid_score = evaluate_sampler_fid(params, score_model, real_images, latents) 

#     if fid_score < best_fid:
#         best_fid = fid_score
#         best_params_fid = params

# print(f"Best parameters: {best_params_fid} with MMD: {best_fid}")
#----------------------------------------------------------------------------
# param_grid = {
#     'model': ['edm_ckpt.pth', 'edm_ckpt_1.pth', 'edm_ckpt_2.pth', 'edm_ckpt_5.pth', 'edm_ckpt_20.pth', 'edm_ckpt_50.pth'],
#     'S_churn': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#     'S_min': [0, 0.005, 0.01, 0.02, 0.05, 0.1],
#     'S_max': [0.2, 0.5, 1, 2, 5, 10, 20, 50, 80],
#     'S_noise': [1.000, 1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009, 1.010],
# }

# # Define the grid search combinations
# param_combinations = list(itertools.product(*param_grid.values()))

# best_fid = float('inf')
# best_params_fid = None
# for params in param_combinations:
#     print(f"Evaluating params: {params}")
#     model, S_churn, S_min, S_max, S_noise = params
#     ckpt = torch.load(model, map_location=device, weights_only=True)
#     ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
#     score_model = ScoreNet(con_dim=6, input_dim=7)
#     score_model = score_model.to(device)
#     score_model.load_state_dict(ckpt)

#     fid_score = evaluate_sampler_fid(params, score_model, real_images, latents, labels) 

#     if fid_score < best_fid:
#         best_fid = fid_score
#         best_params_fid = params

# print(f"Best parameters: {best_params_fid} with MMD: {best_fid}")
