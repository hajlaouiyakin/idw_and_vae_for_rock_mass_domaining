from __future__ import annotations

import copy
import os

import torch
from torch.nn import Linear
from DeepIDW import Layer, HiddenLayers, DeepNN, NNCoefficients
import tqdm
import gpytorch
from kernels import  InverseDistanceWithParam
from torch import autograd
from losses.variational_ae_loss import VAELoss
from correlation_vae import CorVAE,CorVAEConv, CorVAERbf
from models_metrics import correlation, multiple_corr
device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''
We stack DeepIDW and CorVAE to create DeepIdwAutoEncoder which is the model used in the paper in order
to learn the pseudo-BI. As you could see three type of models are created : models with fully connected layers
model with conv layers, and model with RBF layer
'''


# DeepIdwAutoEncoder with fully connected layers
class DeepIdwAutoEncoderBatch(torch.nn.Module):
    def __init__(self,input_dim_x,input_dim_y,hidden_dim, num_layers, depth, p=None, lengthcale = True, device=device ):
        super(DeepIdwAutoEncoderBatch,self).__init__()
      
        self.cor_vae  = CorVAE(input_dim_x=input_dim_x,input_dim_y=input_dim_y,
                               hidden_dim_x=100, hidden_dim_y=100,latent_dim_x=3,latent_dim_y=1,device=device).to(device)
        self.depth = depth
        self.device = device
        self.nn_coefficients  = NNCoefficients(input_dim_x, 1, hidden_dim,
                                               num_layers, depth).to(device)

        self.nn_bias = NNCoefficients(input_dim_x, 1, hidden_dim,
                                               num_layers, depth).to(device)
       
        self.kernel = InverseDistanceWithParam(ard_num_dims=input_dim_x).to(device)#, lengthscale_constraint=constraints)

    def forward(self,x ,y):
        x=x.to(self.device)
        y_out = y.to(self.device)
        output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self.cor_vae(x, y_out)
       
        z=z_y.flatten().to(self.device)

        for i in range(self.depth):
            z1 = torch.relu(self.nn_coefficients(x,i).flatten()*z + self.nn_bias(x,i).flatten())
            z1.to(device)
            if i < self.depth -1:
                  z =  self.nn_coefficients(x,i+1).flatten()*z1 + self.nn_bias(x,i+1).flatten() + z




    
        covar = self.kernel(x, device=self.device).to(self.device)

      
        z_int = torch.matmul(torch.nn.functional.normalize(covar.evaluate(),p=1, dim=1), z ) #/ torch.matmul(covar.evaluate(), torch.ones_like(y_out) )
    
        y_hat  = self.cor_vae.vae_y.decoder( z_int.reshape(-1,1))

        return  y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y
    def predict(self, x_test,x,  z):
  
        k1 = self.kernel(x_test.to(self.device), x, mode='test',device=self.device).to(self.device)
      

        z_pred = torch.matmul(torch.nn.functional.normalize(k1.evaluate()+1e-10,dim=1,p=1),z) #/ torch.matmul(k1.evaluate(), torch.ones_like(y_out) )
 
        return z_pred
    def train(self, data_loader, n_epochs=200, lr=0.01,beta_x=1, beta_y=1, verbose=True):

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"
    

        #z.to(device)
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss(reduction='mean')
       

        with (tqdm.trange(n_epochs, disable=not verbose) as bar):
            overloss_pred = 0
            for k in bar:

              overall_loss = torch.tensor(0.)

              for batch_idx, (x, y) in enumerate(data_loader):


                x = x.to(self.device)
                y = y.to(self.device)
                print(x)
                print(y)
                optimizer.zero_grad()
                y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self(x, y)
                loss_x = self.cor_vae.vae_x.loss(x, output_x, mean_x, log_var_x, beta_x)
                loss_y = self.cor_vae.vae_y.loss(y, y_hat, mean_y, log_var_y, beta_y)


                multi_corr, corr = multiple_corr(z_x, z_int.reshape(-1,1))

                loss_corr = (1 - multi_corr) + (corr[0, 1] ** 2 + corr[0, 2] ** 2 + corr[1, 2] ** 2)
               
                loss = 1e-6 * loss_x + 3000 * loss_y + 1000 * loss_corr
                
                loss.backward()


                optimizer.step()
                overall_loss =((batch_idx + 1) / (batch_idx + 2)) * overall_loss + (1 / (batch_idx + 2)) *loss

                torch.cuda.empty_cache()

              #overall_loss
              if k == 1:
                  overloss_pred = overall_loss
              if overall_loss < overloss_pred:
                overloss_pred = overall_loss
                model = self





              with autograd.detect_anomaly():

               postfix = dict(Loss=f"{overall_loss.item():.3f}",
                               power=f"{self.kernel.power_param.item():.3f}")

               if (hasattr(self.kernel, 'base_kernel') and hasattr(self.kernel.base_kernel, 'lengthscale')):
                   lengthscale = self.kernel.base_kernel.lengthscale
                   if lengthscale is not None:
                        lengthscale = lengthscale.squeeze(0).detach().cpu().numpy()
               else:
                  lengthscale = self.kernel.lengthscale

              if lengthscale is not None:
                  if len(lengthscale.squeeze(0)) > 1:
                      lengthscale_repr = [f"{l:.3f}" for l in lengthscale.squeeze(0)]
                      postfix['lengthscale'] = f"{lengthscale_repr}"
                  else:
                      
                     postfix['lengthscale'] = f"{lengthscale[0].item():.3f}"

              bar.set_postfix(postfix)
        return (model,y_hat,z_int,covar, overall_loss.item())

# DeepIdwAutoEncoder with Conv layers 

class DeepIdwAutoEncoderConv(torch.nn.Module):
    def __init__(self,input_dim_x,input_dim_y,hidden_dim, num_layers, depth, p=None, lengthcale = True, device=device ):
        super(DeepIdwAutoEncoderConv,self).__init__()
        
        self.cor_vae_conv  = CorVAEConv(input_dim_x=input_dim_x,input_dim_y=input_dim_y,latent_dim_x=1,latent_dim_y=1,device=device).to(device)
        self.depth = depth
        self.device = device
        self.nn_coefficients  = NNCoefficients(input_dim_x, 1, hidden_dim,
                                               num_layers, depth).to(device)

        self.nn_bias = NNCoefficients(input_dim_x, 1, hidden_dim,
                                               num_layers, depth).to(device)
        
        self.kernel = InverseDistanceWithParam(ard_num_dims=input_dim_x).to(device)#, lengthscale_constraint=constraints)
    
    def forward(self,x ,y):
        x=x.to(self.device)
        y_out = y.to(self.device)
        output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self.cor_vae_conv(x, y_out)
        
        z=z_y.flatten().to(self.device)
        for i in range(self.depth):
            z1 = torch.relu(self.nn_coefficients(x,i).flatten()*z + self.nn_bias(x,i).flatten())
            z1.to(device)
            if i < self.depth -1:
                  z =  self.nn_coefficients(x,i+1).flatten()*z1 + self.nn_bias(x,i+1).flatten() + z

        covar = self.kernel(x, device=self.device).to(self.device)

        z_int = torch.matmul(torch.nn.functional.normalize(covar.evaluate(),p=1, dim=1), z ) #/ torch.matmul(covar.evaluate(), torch.ones_like(y_out) )
    
        y_hat  = self.cor_vae_conv.vae_conv_y.decoder_conv( z_int.reshape(-1,1))
       
        return  y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y
    def predict(self, x_test,x,  z):
  
        k1 = self.kernel(x_test.to(self.device), x, mode='test',device=self.device).to(self.device)

        z_pred = torch.matmul(torch.nn.functional.normalize(k1.evaluate()+1e-10,dim=1,p=1),z) #/ torch.matmul(k1.evaluate(), torch.ones_like(y_out) )
      
        return z_pred
    def train(self, data_loader, n_epochs=200, lr=0.01,beta_x=1, beta_y=1, verbose=True):

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"
        
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss(reduction='mean')
        with (tqdm.trange(n_epochs, disable=not verbose) as bar):
            overloss_pred = 0
            for k in bar:

              overall_loss = torch.tensor(0.)
              for batch_idx, (x, y) in enumerate(data_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self(x, y)
                loss_x = self.cor_vae_conv.vae_conv_x.loss(x, output_x, mean_x, log_var_x, beta_x)
                loss_y = self.cor_vae_conv.vae_conv_y.loss(y, y_hat, mean_y, log_var_y, beta_y)
                loss_corr = (1 -correlation(z_x, z_int.reshape(-1,1)))
                loss = 1e-6 * loss_x + 1000 * loss_y + 1000 * loss_corr
                loss.backward()
                optimizer.step()
                overall_loss =((batch_idx + 1) / (batch_idx + 2)) * overall_loss + (1 / (batch_idx + 2)) *loss

                torch.cuda.empty_cache()
              if k == 1:
                  overloss_pred = overall_loss
              if overall_loss < overloss_pred:
                overloss_pred = overall_loss
                model = self
              with autograd.detect_anomaly():
               postfix = dict(Loss=f"{overall_loss.item():.3f}",
                               power=f"{self.kernel.power_param.item():.3f}")

               if (hasattr(self.kernel, 'base_kernel') and hasattr(self.kernel.base_kernel, 'lengthscale')):
                   lengthscale = self.kernel.base_kernel.lengthscale
                   if lengthscale is not None:
                        lengthscale = lengthscale.squeeze(0).detach().cpu().numpy()
               else:
                  lengthscale = self.kernel.lengthscale

              if lengthscale is not None:
                  if len(lengthscale.squeeze(0)) > 1:
                      lengthscale_repr = [f"{l:.3f}" for l in lengthscale.squeeze(0)]
                      postfix['lengthscale'] = f"{lengthscale_repr}"
                  else:
                 
                     postfix['lengthscale'] = f"{lengthscale[0].item():.3f}"

              bar.set_postfix(postfix)
        return (model,y_hat,z_int,covar, overall_loss.item())




## DeepIdwAutoEncoder with  rbf layers
class DeepIdwAutoEncoderRbf(torch.nn.Module):
    def __init__(self,input_dim_x,input_dim_y,hidden_dim, num_layers, depth, p=None, lengthcale = True, device=device ):
        super(DeepIdwAutoEncoderRbf,self).__init__()
        self.cor_vae_rbf  = CorVAERbf(input_dim_x=input_dim_x, hidden_dim_x=3,input_dim_y=input_dim_y, hidden_dim_y=4,latent_dim_x=1,latent_dim_y=1,device=device).to(device)
        self.depth = depth
        self.device = device
        self.nn_coefficients  = NNCoefficients(input_dim_x, 1, hidden_dim,
                                               num_layers, depth).to(device)

        self.nn_bias = NNCoefficients(input_dim_x, 1, hidden_dim,
                                               num_layers, depth).to(device)
     
        self.kernel = InverseDistanceWithParam(ard_num_dims=input_dim_x).to(device)#, lengthscale_constraint=constraints)
       
    def forward(self,x ,y):
        x=x.to(self.device)
        y_out = y.to(self.device)
        output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self.cor_vae_rbf(x, y_out)
        z=z_y.flatten().to(self.device)
        for i in range(self.depth):
            z1 = torch.relu(self.nn_coefficients(x,i).flatten()*z + self.nn_bias(x,i).flatten())
            z1.to(device)
            if i < self.depth -1:
                  z =  self.nn_coefficients(x,i+1).flatten()*z1 + self.nn_bias(x,i+1).flatten() + z

        covar = self.kernel(x, device=self.device).to(self.device)
        z_int = torch.matmul(torch.nn.functional.normalize(covar.evaluate(),p=1, dim=1), z ) #/ torch.matmul(covar.evaluate(), torch.ones_like(y_out) )
    
        y_hat  = self.cor_vae_rbf.vae_rbf_y.decoder_rbf( z_int.reshape(-1,1))
     
        return  y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y
    def predict(self, x_test,x,  z):
  
        k1 = self.kernel(x_test.to(self.device), x, mode='test',device=self.device).to(self.device)
    
        z_pred = torch.matmul(torch.nn.functional.normalize(k1.evaluate()+1e-10,dim=1,p=1),z) #/ torch.matmul(k1.evaluate(), torch.ones_like(y_out) )
      
        return z_pred
    def train(self, data_loader, n_epochs=200, lr=0.01,beta_x=1, beta_y=1, verbose=True):

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"
     
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss(reduction='mean')
    
        with (tqdm.trange(n_epochs, disable=not verbose) as bar):
            overloss_pred = 0
            for k in bar:

              overall_loss = torch.tensor(0.)
              for batch_idx, (x, y) in enumerate(data_loader):

                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self(x, y)
                loss_x = self.cor_vae_rbf.vae_rbf_x.loss(x, output_x, mean_x, log_var_x, beta_x)
                loss_y = self.cor_vae_rbf.vae_rbf_y.loss(y, y_hat, mean_y, log_var_y, beta_y)
                multi_corr, corr = multiple_corr(z_x, z_int.reshape(-1,1))
                loss_corr = (1 -correlation(z_x, z_int.reshape(-1,1)))
                loss = 1e-20 * loss_x + 1000 * loss_y + 1000*loss_corr
                overall_loss =((batch_idx + 1) / (batch_idx + 2)) * overall_loss + (1 / (batch_idx + 2)) *loss
                print('corr_loss', 1000*loss_corr)
                print('y_rec_loss', 1000*loss_y)
                torch.cuda.empty_cache()
              if k == 1:
                  overloss_pred = overall_loss
              if overall_loss < overloss_pred:
                overloss_pred = overall_loss
                model = self
              with autograd.detect_anomaly():
               loss.backward()
               optimizer.step()
               postfix = dict(Loss=f"{overall_loss.item():.3f}",
                               power=f"{self.kernel.power_param.item():.3f}")

               if (hasattr(self.kernel, 'base_kernel') and hasattr(self.kernel.base_kernel, 'lengthscale')):
                   lengthscale = self.kernel.base_kernel.lengthscale
                   if lengthscale is not None:
                        lengthscale = lengthscale.squeeze(0).detach().cpu().numpy()
               else:
                  lengthscale = self.kernel.lengthscale

              if lengthscale is not None:
                  if len(lengthscale.squeeze(0)) > 1:
                      lengthscale_repr = [f"{l:.3f}" for l in lengthscale.squeeze(0)]
                      postfix['lengthscale'] = f"{lengthscale_repr}"
                  else:
                
                     postfix['lengthscale'] = f"{lengthscale[0].item():.3f}"

              bar.set_postfix(postfix)
        return (model,y_hat,z_int,covar, overall_loss.item())


