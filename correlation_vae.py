from __future__ import annotations
from variational_model import VariationalAutoEncoder as VAE
from variational_model import VAEConv as VAEConv
from variational_model import VAERbf as VAErbf
import torch
from torch.nn import Linear

import torch
from torch.utils.data import DataLoader
from models_metrics import correlation, r_squared, multiple_corr
import copy
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
CorVAE are the correlational VAE used in the paper, it create a VAE for the coordinates, a VAE for the target
and it returns the latent presentations of z_x, z_y. z_y will be the input to DeepIDW for interpolation,
and we will make sure throught a correlation loss function that is correlated with z_x.
Three different CorVAE are developped: CorVAE with fully connected layers, CorVaE with Conv layers
and CorVAE with RBF layers 
'''

# CorVAE with Fully connected layers
# returns:
# output_x: reconstruction of x,  output_y: reconstruction of y
# mean_x, var_x : mean and variance of the latent representation of vae_x
# mean_y, var_y : mean and variance of the latent representation of vae_y
# z_y, z_x : latent representation of the target and coordinates
class CorVAE(torch.nn.Module):
    def __init__(self, input_dim_x, input_dim_y, hidden_dim_x, hidden_dim_y, latent_dim_x, latent_dim_y,device=device):
        super(CorVAE, self).__init__()
        self.vae_x = VAE(input_dim_x, hidden_dim_x, latent_dim_x, input_dim_x, device=device).to(device)
        self.vae_y = VAE(input_dim_y, hidden_dim_y, latent_dim_y, input_dim_y, device=device).to(device)

    def forward(self, x, y):

        output_x, z_x, mean_x, var_x, log_var_x = self.vae_x(x)
        output_y, z_y, mean_y, var_y, log_var_y = self.vae_y(y)
        return output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y


# CorVAE with Conv layers

class CorVAEConv(torch.nn.Module):
    def __init__(self, input_dim_x, input_dim_y, latent_dim_x, latent_dim_y, device=device):
        super(CorVAEConv, self).__init__()
        self.vae_conv_x = VAEConv(input_dim=input_dim_x,output_dim=input_dim_x, latent_dim=latent_dim_x,device=device)
        self.vae_conv_y = VAEConv(input_dim=input_dim_y, output_dim=input_dim_y, latent_dim=latent_dim_y, device=device)
    def forward(self, x, y):
        output_x, z_x, mean_x, var_x, log_var_x = self.vae_conv_x(x)
        output_y, z_y, mean_y, var_y, log_var_y = self.vae_conv_y(y)
        return output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y



# CorVAE with RBF layers 

class CorVAERbf(torch.nn.Module):
    def __init__(self, input_dim_x, hidden_dim_x, input_dim_y, hidden_dim_y, latent_dim_x, latent_dim_y, device=device):
        super(CorVAERbf, self).__init__()
        self.vae_rbf_x = VAErbf(input_dim=input_dim_x,output_dim=input_dim_x,hidden_dim=hidden_dim_x, latent_dim=latent_dim_x,device=device)
        self.vae_rbf_y = VAErbf(input_dim=input_dim_y, output_dim=input_dim_y,hidden_dim=hidden_dim_y, latent_dim=latent_dim_y, device=device)
    def forward(self, x, y):
        output_x, z_x, mean_x, var_x, log_var_x = self.vae_rbf_x(x)
        output_y, z_y, mean_y, var_y, log_var_y = self.vae_rbf_y(y)
        return output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y