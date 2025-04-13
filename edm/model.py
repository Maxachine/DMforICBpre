import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pdb 

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

 
class ScoreNet(nn.Module):
    """A simple feedforward network for score matching, with two hidden layers."""

    def __init__(self, con_dim=6, input_dim=7, hidden_dim=256, embed_dim=256):
        """Initialize a simple feedforward network with time embedding for score matching.
        
        Args:
            marginal_prob_std: A function that takes time t and gives the standard deviation of the perturbation kernel.
            input_dim: The dimensionality of the input data (7 in your case).
            hidden_dim: The number of hidden units in each hidden layer.
            embed_dim: The dimensionality of the time embedding.
        """
        super().__init__()
 
        # Gaussian random feature embedding for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Hidden layers
        self.fc1 = nn.Linear(input_dim + con_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, input_dim)  # Output layer

        # Activation function (Swish)
        self.act = lambda x: x * torch.sigmoid(x)

        # self.marginal_prob_std = marginal_prob_std
        
        self.sigma_data = 1
  
    def forward(self, x, sigma):
        # Obtain the Gaussian random feature embedding for time      
        
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
#         pdb.set_trace()
        # First hidden layer
        x = c_in.unsqueeze(1)* x
        embed = self.act(self.embed(c_noise))
        
        h1 = self.fc1(x)
        h1 += embed  # Incorporate time information
        h1 = self.act(h1)

        # Second hidden layer
        h2 = self.fc2(h1)
        h2 += embed  # Incorporate time information
        h2 = self.act(h2)

        # Output layer
        F_x = self.fc3(h2)
        # pdb.set_trace()
        h =  c_skip.unsqueeze(1) * x[:,:-6] + c_out.unsqueeze(1) * F_x.to(torch.float32)
        return h
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
