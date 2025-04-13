from scipy import integrate
import torch
import numpy as np
from utils import marginal_prob_std_fn,device,diffusion_coeff_fn
from model import ScoreNet
import pdb

## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5 #@param {'type': 'number'}
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                atol=error_tolerance, 
                rtol=error_tolerance, 
                device='cpu', 
                T=1 ,
                z=None,
                eps=1e-3):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that returns the standard deviation 
        of the perturbation kernel.
        diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        atol: Tolerance of absolute errors.
        rtol: Tolerance of relative errors.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        z: The latent code that governs the final sample. If None, we start from p_1;
        otherwise, we start from the given z.
        eps: The smallest time step for numerical stability.
    """
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
        init_x = torch.randn(batch_size,7, device=device) \
        * marginal_prob_std(t)[:, None]
    else:
        init_x = z
        
    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
        with torch.no_grad():    
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def ode_func(t, x):        
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t    
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return  -1.5 * (g**2) * score_eval_wrapper(x, time_steps)
    
    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x

if __name__=='__main__':
    device = 'cpu' #@param ['cuda', 'cpu'] {'type':'string'}
    ckpt = torch.load('Hybrid2Gauss_ckpt.pth', map_location=device)
    # ckpt = torch.load('ckpt.pth', map_location=device)
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn, input_dim=7))
    score_model = score_model.to(device)
    score_model.load_state_dict(ckpt)

    sample_batch_size = 1000 #@param {'type':'integer'}
    sampler = ode_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

    ## Generate samples using the specified sampler.
    samples = sampler(score_model, 
                    marginal_prob_std_fn,
                    diffusion_coeff_fn, 
                    sample_batch_size, 
                    device=device,
                    T=0.5)
    # pdb.set_trace()
    samples_np = samples.cpu().numpy()  # 如果 samples 在 GPU 上，先移到 CPU

    data = []
    file_path='Hybrid2Gauss.txt'
    with open(file_path, 'r') as f:
        for line in f:
            # 去除换行符并按空格或逗号分割
            row = line.strip().replace(',', ' ').split()
            # 将字符串转换为浮点数
            row = [float(x) for x in row]
            if len(row) == 7:  # 确保每行有 7 个数值
                data.append(row)
    traindata = np.array(data)
    pdb.set_trace()
    # 将 samples 写入文件
    # with open('samples.txt', 'w') as f:
    #     for sample in samples_np:
    #         # 将每个样本的 7 个维度值转换为字符串，用空格分隔
    #         sample_str = ' '.join(map(str, sample))
    #         f.write(sample_str + '\n')  # 每行写入一个样本
    # print('Done.')