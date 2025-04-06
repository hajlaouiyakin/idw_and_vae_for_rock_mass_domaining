from __future__ import annotations
from models.variational_model import VariationalAutoEncoder as VAE
from models.variational_model import VAEConv as VAEConv
from models.variational_model import CondVariationalAutoEncoder as CondVAE
from models.variational_model import VAERbf as VAErbf
from models.variational_model import VAELoss
import torch
from torch.nn import Linear

import torch
from torch.utils.data import DataLoader
from models.models_metrics import correlation, r_squared, multiple_corr
import copy
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CorVAE(torch.nn.Module):
    def __init__(self, input_dim_x, input_dim_y, hidden_dim_x, hidden_dim_y, latent_dim_x, latent_dim_y,device=device):
        super(CorVAE, self).__init__()
        self.vae_x = VAE(input_dim_x, hidden_dim_x, latent_dim_x, input_dim_x, device=device).to(device)
        self.vae_y = VAE(input_dim_y, hidden_dim_y, latent_dim_y, input_dim_y, device=device).to(device)

    def forward(self, x, y):

        output_x, z_x, mean_x, var_x, log_var_x = self.vae_x(x)
        output_y, z_y, mean_y, var_y, log_var_y = self.vae_y(y)
        return output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y

    def train_model(self, dataset_train, num_epochs=1000, lr=0.01, batch_size=100, beta_x=1, beta_y=1):
        self.train()
        training_parameters = [{'params': self.parameters()}]
        optimizer = torch.optim.Adamax(training_parameters, lr=lr)
        data_loader_train = DataLoader(dataset_train, batch_size, num_workers=0, shuffle=False)
        for epoch in range(num_epochs):
            overall_loss = 0
            for idx, (x, y) in enumerate(data_loader_train):
                # x = torch.sigmoid_(x)
                optimizer.zero_grad()
                output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self(x, y)
                loss_x = self.vae_x.loss(x, output_x, mean_x, log_var_x, beta_x)
                loss_y = self.vae_y.loss(y, output_y, mean_y, log_var_y, beta_y)
                multi_corr, corr = multiple_corr(z_x, z_y)
                loss_corr = (1-multi_corr) + (corr[0,1]**2 + corr[0,2]**2 + corr[1,2]**2)
                #loss_corr = -r_squared(z_x, z_y, z_y)
                print("correlation", multiple_corr(z_x, z_y))
                #loss_y=0
                #loss_x = 0
                loss = loss_x + 10*loss_y +1000*loss_corr
                loss.backward()
                optimizer.step()
                # rint(loss.retain_grad())

                overall_loss += loss.item()
                print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (idx * batch_size + 1))




class CorCondVAE(torch.nn.Module):       # Conditional correlation VAE
    def __init__(self, input_dim_x, input_dim_y, hidden_dim_x, hidden_dim_y, latent_dim_x, latent_dim_y):
        super(CorCondVAE, self).__init__()
        self.vae_x = VAE(input_dim_x, hidden_dim_x, latent_dim_x, input_dim_x)
        self.vae_y =  CondVAE(input_dim_y, latent_dim_x, hidden_dim_y, latent_dim_y, input_dim_y)

    def forward(self, x, y):
        output_x, z_x, mean_x, var_x, log_var_x = self.vae_x(x)
        output_y, z_y, mean_y, var_y, log_var_y = self.vae_y(y, z_x)
        return output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y

    def train_model(self, dataset_train, num_epochs=1000, lr=0.01, batch_size=100, beta_x=1, beta_y=1):
        self.train()
        training_parameters = [{'params': self.parameters()}]
        optimizer = torch.optim.Adamax(training_parameters, lr=lr)
        data_loader_train = DataLoader(dataset_train, batch_size, num_workers=0, shuffle=False)
        best_model = None
        multi_corr1 = 0
        for epoch in range(num_epochs):
            overall_loss = 0
            for idx, (x, y) in enumerate(data_loader_train):
                # x = torch.sigmoid_(x)
                optimizer.zero_grad()
                output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self(x, y)
                loss_x = self.vae_x.loss(x, output_x, mean_x, log_var_x, beta_x)
                loss_y = self.vae_y.loss(y, output_y, mean_y, log_var_y, beta_y)
                multi_corr, corr = multiple_corr(z_x, z_y)
                loss_corr = (1-multi_corr) + (corr[0,1]**2 + corr[0,2]**2 + corr[1,2]**2)
                if multi_corr> multi_corr1:
                    multi_corr1 = multi_corr
                    best_model = copy.deepcopy(self)

                #loss_corr = -r_squared(z_x, z_y, z_y)
                print("correlation", multiple_corr(z_x, z_y))
                #loss_y=0
                #loss_x = 0
                loss = loss_x + 10*loss_y +20*loss_corr
                loss.backward()
                optimizer.step()
                # rint(loss.retain_grad())

                overall_loss += loss.item()
                print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (idx * batch_size + 1))
        return best_model




class CorVAEConv(torch.nn.Module):
    def __init__(self, input_dim_x, input_dim_y, latent_dim_x, latent_dim_y, device=device):
        super(CorVAEConv, self).__init__()
        self.vae_conv_x = VAEConv(input_dim=input_dim_x,output_dim=input_dim_x, latent_dim=latent_dim_x,device=device)
        self.vae_conv_y = VAEConv(input_dim=input_dim_y, output_dim=input_dim_y, latent_dim=latent_dim_y, device=device)
    def forward(self, x, y):
        output_x, z_x, mean_x, var_x, log_var_x = self.vae_conv_x(x)
        output_y, z_y, mean_y, var_y, log_var_y = self.vae_conv_y(y)
        return output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y



# Correlation AutoEncoder with rbf layers

class CorVAERbf(torch.nn.Module):
    def __init__(self, input_dim_x, hidden_dim_x, input_dim_y, hidden_dim_y, latent_dim_x, latent_dim_y, device=device):
        super(CorVAERbf, self).__init__()
        self.vae_rbf_x = VAErbf(input_dim=input_dim_x,output_dim=input_dim_x,hidden_dim=hidden_dim_x, latent_dim=latent_dim_x,device=device)
        self.vae_rbf_y = VAErbf(input_dim=input_dim_y, output_dim=input_dim_y,hidden_dim=hidden_dim_y, latent_dim=latent_dim_y, device=device)
    def forward(self, x, y):
        output_x, z_x, mean_x, var_x, log_var_x = self.vae_rbf_x(x)
        output_y, z_y, mean_y, var_y, log_var_y = self.vae_rbf_y(y)
        return output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y