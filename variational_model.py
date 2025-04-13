from __future__ import annotations

import math

import torch
from torch.nn import Linear, Conv1d,ConvTranspose1d, Conv2d, ConvTranspose2d

import torch
from torch.utils.data import DataLoader
from losses.variational_ae_loss import VAELoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''
In this file we develope Variational Autoencoders that are used in this project, three types of VAE are used
VAE with fully connected layers VAE with Conv layers and VAE with RBF layers

'''


# Encoder with fully connected layers 
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device=device):
        super(Encoder, self).__init__()
  
        self.fc_input = Linear(input_dim, hidden_dim).to(device)
        self.hidden = Linear(hidden_dim, hidden_dim).to(device)
        self.fc_mean =Linear(hidden_dim, latent_dim).to(device)
        self.fc_log_var = Linear(hidden_dim, latent_dim).to(device)

    def forward(self, x):
 
        h = torch.tanh(self.hidden(torch.relu(self.fc_input(x))))


        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        var = torch.exp(0.5 * log_var)
        z = mean + torch.randn_like(var) * var
        return z, mean, var, log_var


# Decoder with fully connected layes 
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim,device=device):
        super(Decoder, self).__init__()
        self.fc_hidden = Linear(latent_dim, hidden_dim).to(device)
        self.fc_output = Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        h = torch.tanh(self.fc_hidden(x))
        output = self.fc_output(h)

        return output
# VAE with fully connected layes
class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim,device=device):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim,device=device).to(device)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim,device=device).to(device)
        self.loss = VAELoss().to(device)

    def forward(self, x):
        z, mean, var, log_var = self.encoder(x)
        output = self.decoder(z)
        return output, z, mean, var, log_var

    



## CovAutoEncoder

# Encoder with conv layers
class EncoderConv(torch.nn.Module):
    def __init__(self, input_dim=4, latent_dim=1, device=device):
        super(EncoderConv, self).__init__()
        #print(device)
        self.conv_input = Conv1d(in_channels=input_dim, out_channels=2*input_dim, kernel_size=2,stride=1, padding=1, )
        self.conv_hidden = Conv1d(in_channels=2*input_dim, out_channels=4*input_dim, kernel_size=2,stride=1, padding=1)
        #self.hidden = Linear(hidden_dim, hidden_dim).to(device)
        self.fc_mean =Linear(4*input_dim*3, latent_dim).to(device)
        self.fc_log_var = Linear(4*input_dim*3, latent_dim).to(device)

    def forward(self, x):
        #print(x)
        x=x.view(x.size(0), x.size(1), 1)
        h = torch.tanh(self.conv_hidden(torch.relu(self.conv_input(x))))
        #print(h.shape)
        h=h.view(h.size(0),-1)
        #print(h.shape)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        var = torch.exp(0.5 * log_var)
        z = mean + torch.randn_like(var) * var
        return z, mean, var, log_var
    
# Decoder with conv layers
class DecoderConv(torch.nn.Module):
    def __init__(self, latent_dim, output_dim,device=device):
        super(DecoderConv, self).__init__()

        self.fc_hidden = Linear(latent_dim, 4*output_dim).to(device)
        self.deconv_hidden = ConvTranspose1d(in_channels=4*output_dim, out_channels=2*output_dim, kernel_size=1, stride=1, padding=0).to(device)
        self.deconv_output = ConvTranspose1d(in_channels=2 * output_dim, out_channels= output_dim, kernel_size=1,
                                             stride=1, padding=0).to(device)
        self.fc_output = Linear(output_dim,output_dim)


    def forward(self, x):
        h = torch.relu(self.fc_hidden(x))
        h=h.view(h.size(0),-1,1)
        h = self.deconv_hidden(h)
        output = self.deconv_output(h)
        output = output.view(output.size(0),-1)
        output = self.fc_output(torch.tanh(output))
        return output
    
# VAE with conv layers 
class VAEConv(torch.nn.Module):
            def __init__(self, input_dim=4, latent_dim=1, output_dim=4, device=device):
                super(VAEConv, self).__init__()
                self.encoder_conv = EncoderConv(input_dim, latent_dim, device=device).to(device)
                self.decoder_conv = DecoderConv(latent_dim, output_dim, device=device).to(device)
                self.loss = VAELoss().to(device)

            def forward(self, x):
                z, mean, var, log_var = self.encoder_conv(x)
                output = self.decoder_conv(z)
                return output, z, mean, var, log_var


# Variational AutoEncoder with RBF Layers
# Create RBF Layer
class RBFNNLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RBFNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.centers = torch.nn.Parameter(torch.randn(output_dim, input_dim))
        self.widths = torch.nn.Parameter(torch.randn( output_dim))
        self.fun = 'gaussian'
   
    def forward(self, x):
        x = x.unsqueeze(1)

        # Compute the distance between each input point and each center
        distances = torch.norm(x - self.centers, dim=2)
        alpha = distances/((2*self.widths.pow(2)).pow(.5))

        # Compute the activation using radial basis function (Gaussian or inverse_quadratic or spline)
      
        if self.fun == 'gaussian':
            activation = torch.exp(-1*alpha.pow(2))
        elif self.fun == 'inverse_quadratic':
            activation = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
        elif self.fun == 'spline':
            activation = alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha))
        else:
            activation = alpha

        return activation

# Encoder with RBF layers
class EncoderRBF(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device=device):
        super(EncoderRBF, self).__init__()
        #print(device)
        self.rbf_input = RBFNNLayer(input_dim,hidden_dim).to(device)
        #self.fc1= Linear(h, hidden_dim).to(device)
        self.hidden = Linear(hidden_dim, hidden_dim).to(device)
        self.rbf_hidden = RBFNNLayer(hidden_dim, hidden_dim).to(device)
        self.fc_mean =Linear(hidden_dim, latent_dim).to(device)
        self.fc_log_var = Linear(hidden_dim, latent_dim).to(device)

    def forward(self, x):
        h = self.rbf_hidden(torch.tanh(self.hidden(self.rbf_input(x))))


        mean = self.fc_mean(torch.tanh(h))
        log_var = self.fc_log_var(torch.tanh(h))
        var = torch.exp(0.5 * log_var)
        z = mean + torch.randn_like(var) * var

        return z, mean, var, log_var
# Decoder with RBF layers
class DecoderRBF(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim,device=device):
        super(DecoderRBF, self).__init__()
        self.rbf_hidden = RBFNNLayer(latent_dim, hidden_dim).to(device)
        self.fc_hidden = Linear(hidden_dim, hidden_dim).to(device)
        self.rbf_output =  RBFNNLayer(hidden_dim, output_dim).to(device)
        self.fc_output= Linear(output_dim, output_dim)
    def forward(self, x):
        h = self.rbf_output(torch.tanh(self.fc_hidden(self.rbf_hidden(x))))
        print(self.rbf_hidden.fun)
        output = self.fc_output(torch.tanh(h))
        #output = torch.sigmoid_(self.fc_output(h))
        return output
# VAE with RBF layers
class VAERbf(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, latent_dim, output_dim,device=device ):
        super(VAERbf,self).__init__()
        self.encoder_rbf = EncoderRBF(input_dim, hidden_dim, latent_dim, device)
        self.decoder_rbf = DecoderRBF(latent_dim, hidden_dim, output_dim, device)
        self.loss = VAELoss().to(device)
    def forward(self, x):
        z, mean, var, log_var = self.encoder_rbf(x)
        output = self.decoder_rbf(z)
        return output, z, mean, var, log_var
