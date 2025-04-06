from __future__ import annotations

import copy
import os

import torch
from torch.nn import Linear
from models.DeepIDW import Layer, HiddenLayers, DeepNN, NNCoefficients
import tqdm
import gpytorch
#from models.models_utils import FeatureExtractor, Perturbation
from models.kernels import ExponentialKernel, SimpleSincKernel,  InverseDistance, InverseDistanceWithParam, KnnKernel
from torch import autograd
from losses.variational_ae_loss import VAELoss
from models.correlation_vae import CorVAE,CorVAEConv, CorVAERbf
from models.models_metrics import correlation, multiple_corr
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Creating encoder to reduce dimension
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.deepnne = DeepNN(input_dim,latent_dim,hidden_dim,100).to(device)
        #self.fc_input = Linear(input_dim, hidden_dim, bias=True).to(device)
        #self.hidden = Linear(hidden_dim, hidden_dim,  bias=True).to(device)
        self.fc_mean =Linear(latent_dim, latent_dim,  bias=True).to(device)
        self.fc_log_var = Linear(latent_dim, latent_dim,  bias=True).to(device)

    def forward(self, x):
        #print(x)
        #h = torch.tanh(self.hidden(torch.relu(self.fc_input(x.to(device)))))
        h = self.deepnne(x.to(device))


        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        var = torch.exp(0.5 * log_var)
        z = mean + torch.randn_like(var) * var
        return z, mean, var, log_var
#Decoder to decode the latent space
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.deepnnd = DeepNN(latent_dim, hidden_dim, hidden_dim, 100).to(device)
        #self.fc_hidden = Linear(latent_dim, hidden_dim,  bias=True).to(device)
        #self.fc_hidden2 =  Linear(hidden_dim, hidden_dim,  bias=True).to(device)
        self.fc_output = Linear(hidden_dim, output_dim,  bias=True).to(device)

    def forward(self, x):
        #h = torch.relu(self.fc_hidden2(torch.relu(self.fc_hidden(x))))
        h= self.deepnnd(x)
        output = self.fc_output(h)
        #output = torch.sigmoid_(self.fc_output(h))
        return output

class DeepIdwAutoEncoder(torch.nn.Module):
    def __init__(self, x, y, hidden_dim, num_layers, depth, p=None, lengthcale = True ):
        super(DeepIdwAutoEncoder,self).__init__()
        self.x = x.to(device)
        self.y = y.to(device)
        #self.encoder = Encoder(self.y.shape[1],100,1).to(device)
        #self.decoder = Decoder(1,100, self.y.shape[1]).to(device)
        self.cor_vae  = CorVAE(input_dim_x=self.x.shape[1],input_dim_y=self.y.shape[1],
                               hidden_dim_x=100, hidden_dim_y=100,latent_dim_x=3,latent_dim_y=1,device=device).to(device)
        self.depth = depth
        self.nn_coefficients  = NNCoefficients(self.x.shape[1], 1, hidden_dim,
                                               num_layers, depth).to(device)
        self.nn_bias = NNCoefficients(self.x.shape[1], 1, hidden_dim,
                                               num_layers, depth).to(device)
        #self.nn_coefficients = DeepNN(input_dim=self.x.shape[1], output_dim=depth, hidden_dim=hidden_dim, num_layers=num_layers)
        #constraints = gpytorch.constraints.Interval(torch.tensor([5., 5., 5.]), torch.tensor([30., 30., 15.]))
        self.kernel = InverseDistanceWithParam(ard_num_dims=self.x.shape[1]).to(device)#, lengthscale_constraint=constraints)
        #self.loss = VAELoss().to(device)
    def forward(self,x ,y):
        x=x.to(device)
        y_out = y.to(device)
        output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self.cor_vae(x, y_out)
        #z, m, var, log_var = self.encoder(y_out)
        z=z_y.flatten().to(device)

        #print(y.shape)
        #print(self.nn_coefficients(x,0).flatten().shape)
        for i in range(self.depth):
            z1 = torch.relu(self.nn_coefficients(x,i).flatten()*z + self.nn_bias(x,i).flatten())
            z1.to(device)
            if i < self.depth -1:
                  z =  self.nn_coefficients(x,i+1).flatten()*z1 + self.nn_bias(x,i+1).flatten() + z




            #else:
             #   y_out = y1


        #print('diff',y_out-y)

        covar = self.kernel(x).to(device)

        #print(torch.matmul(torch.softmax(covar.evaluate(), dim=1),torch.ones_like(y_out)))
        z_int = torch.matmul(torch.nn.functional.normalize(covar.evaluate(),p=1, dim=1), z ) #/ torch.matmul(covar.evaluate(), torch.ones_like(y_out) )
        #print(z_int.shape)
        print('reshaped z', z.reshape(-1,1))
        y_hat  = self.cor_vae.vae_y.decoder( z_int.reshape(-1,1))
        return  y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y
    def predict(self, x_test, z):
        # out_test = self.nn(x_test)
        k1 = self.kernel(x_test.to(device), self.x, mode='test').to(device)
        #c = torch.nn.functional.softmax(self.raw_c)

        z_pred = torch.matmul(torch.nn.functional.normalize(k1.evaluate()+1e-10,dim=1,p=1),z) #/ torch.matmul(k1.evaluate(), torch.ones_like(y_out) )
       # y_pred = self.cor_vae.vae_y.decoder(z_pred)
        return z_pred
    def train(self, x, y, n_epochs=200, lr=0.01,beta_x=1, beta_y=1, verbose=True):
        # print(self.para)
        # initialize the coefficients
        #c = torch.ones_like(y)
        x.to(device)
        y.to(device)

        #z.to(device)
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss(reduction='mean')
        #criterion = torch.nn.BCELoss()

        #print(self)
        with (tqdm.trange(n_epochs, disable=not verbose) as bar):
            for _ in bar:
                optimizer.zero_grad()
                y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self(x.to(device), y.to(device))
                loss_x = self.cor_vae.vae_x.loss(x, output_x, mean_x, log_var_x, beta_x)
                loss_y = self.cor_vae.vae_y.loss(y, y_hat, mean_y, log_var_y, beta_y)

                multi_corr, corr = multiple_corr(z_x, z_int.reshape(-1,1))
                loss_corr = (1 - multi_corr) + (corr[0, 1] ** 2 + corr[0, 2] ** 2 + corr[1, 2] ** 2)
                a = copy.copy(z_y)
                a.requires_grad_(requires_grad=False)
                loss_idw = criterion(z_int.flatten(), a.flatten().to(device))
                print("correlation", multiple_corr(z_x, z_int.reshape(-1,1)))
                y.to(device)
                print(y_hat)
                print('z', z)
                print('z_int', z_int)

                #e = torch.tensor((out - y.to(device)) ** 2, requires_grad=False).to(device)
                #print(y_out-y)
                #print(self.nn_coefficients.forward(x,2))
                #c= self.coefficient
                #print(self.coefficient)
                #i = torch.argmin(self.coefficient)
                #print(i)
                #print(self.coefficient[3955])
                #print(out[3955])
                #print(y[3955])
                #for name, param in self.kernel.named_parameters():
                 #   if param.requires_grad:
                  #      print(param.grad)
                   #     print(name, param.data)

                #print(self.nn_coefficients(x, self.depth-1))




                # print(covar.evaluate())
                # print(out)
                #print("y_out", y_out)
                #print("out", out)
                #print("z",z)
                #print("y",y)

                #print("ecartt",out.flatten()-z.flatten())
                with autograd.detect_anomaly():
                   # loss =  self.loss(y.to(device), y_hat, m, log_var,0.1)
                    #loss = 1e-6*loss_x + 10 * loss_y + 1000 * loss_corr + loss_idw
                    loss = 1e-6 * loss_x + 10 * loss_y + 1000 * loss_corr
                    loss.backward()


                #for name, param in self.parameters():
                 #  if param.requires_grad:
                  #    print(param.grad)


                    optimizer.step()


                # print(self.kernel.power_param.item())
                # print(loss.item())
                postfix = dict(Loss=f"{loss.item():.3f}",
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
                        #print(len(lengthscale))

                        #print(lengthscale)
                        postfix['lengthscale'] = f"{lengthscale[0].item():.3f}"

                bar.set_postfix(postfix)
            return (self,y_hat,z_int,covar, loss.item())


class DeepIdwAutoEncoderBatch(torch.nn.Module):
    def __init__(self,input_dim_x,input_dim_y,hidden_dim, num_layers, depth, p=None, lengthcale = True, device=device ):
        super(DeepIdwAutoEncoderBatch,self).__init__()
        #self.device=device
        #self.encoder = Encoder(self.y.shape[1],100,1).to(device)
        #self.decoder = Decoder(1,100, self.y.shape[1]).to(device)
        self.cor_vae  = CorVAE(input_dim_x=input_dim_x,input_dim_y=input_dim_y,
                               hidden_dim_x=100, hidden_dim_y=100,latent_dim_x=3,latent_dim_y=1,device=device).to(device)
        self.depth = depth
        self.device = device
        self.nn_coefficients  = NNCoefficients(input_dim_x, 1, hidden_dim,
                                               num_layers, depth).to(device)

        self.nn_bias = NNCoefficients(input_dim_x, 1, hidden_dim,
                                               num_layers, depth).to(device)
        #self.nn_coefficients = DeepNN(input_dim=self.x.shape[1], output_dim=depth, hidden_dim=hidden_dim, num_layers=num_layers)
        #constraints = gpytorch.constraints.Interval(torch.tensor([5., 5., 5.]), torch.tensor([30., 30., 15.]))
        self.kernel = InverseDistanceWithParam(ard_num_dims=input_dim_x).to(device)#, lengthscale_constraint=constraints)
        #self.loss = VAELoss().to(device)
    def forward(self,x ,y):
        x=x.to(self.device)
        y_out = y.to(self.device)
        output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self.cor_vae(x, y_out)
        #z, m, var, log_var = self.encoder(y_out)
        z=z_y.flatten().to(self.device)

        #print(y.shape)
        #print(self.nn_coefficients(x,0).flatten().shape)
        for i in range(self.depth):
            z1 = torch.relu(self.nn_coefficients(x,i).flatten()*z + self.nn_bias(x,i).flatten())
            z1.to(device)
            if i < self.depth -1:
                  z =  self.nn_coefficients(x,i+1).flatten()*z1 + self.nn_bias(x,i+1).flatten() + z




            #else:
             #   y_out = y1


        #print('diff',y_out-y)

        covar = self.kernel(x, device=self.device).to(self.device)

        #print(torch.matmul(torch.softmax(covar.evaluate(), dim=1),torch.ones_like(y_out)))
        z_int = torch.matmul(torch.nn.functional.normalize(covar.evaluate(),p=1, dim=1), z ) #/ torch.matmul(covar.evaluate(), torch.ones_like(y_out) )
        #print(z_int.shape)
        #print('reshaped z', z.reshape(-1,1))
        y_hat  = self.cor_vae.vae_y.decoder( z_int.reshape(-1,1))
        #print('y_hat', y_hat)
        return  y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y
    def predict(self, x_test,x,  z):
        # out_test = self.nn(x_test)
        k1 = self.kernel(x_test.to(self.device), x, mode='test',device=self.device).to(self.device)
        #c = torch.nn.functional.softmax(self.raw_c)

        z_pred = torch.matmul(torch.nn.functional.normalize(k1.evaluate()+1e-10,dim=1,p=1),z) #/ torch.matmul(k1.evaluate(), torch.ones_like(y_out) )
       # y_pred = self.cor_vae.vae_y.decoder(z_pred)
        return z_pred
    def train(self, data_loader, n_epochs=200, lr=0.01,beta_x=1, beta_y=1, verbose=True):

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"
        # print(self.para)
        # initialize the coefficients
        #c = torch.ones_like(y)


        #z.to(device)
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss(reduction='mean')
        #criterion = torch.nn.BCELoss()

        #print(self)

        with (tqdm.trange(n_epochs, disable=not verbose) as bar):
            overloss_pred = 0
            for k in bar:

              overall_loss = torch.tensor(0.)

              #model = self
              for batch_idx, (x, y) in enumerate(data_loader):


                #print(batch_idx)
                #print(x.shape)
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
                #print('loss_x',1e-6 * loss_x)
                #print('loss_y',10*loss_y)
                #print('loss_corr', 1000 * loss_corr)
                #print('loss_x1',   loss_x)
                #print('loss_y1',  loss_y)
                #print('loss_corr1',  loss_corr)

                #a = copy.copy(z_y)
                #a.requires_grad_(requires_grad=False)
                #loss_idw = criterion(z_int.flatten(), a.flatten().to(device))

                #loss = ((batch_idx + 1) / (batch_idx + 2)) * loss + (1 / (batch_idx + 2)) * (
                 #           1e-6 * loss_x + 10 * loss_y + 1000 * loss_corr)
                loss = 1e-6 * loss_x + 1000 * loss_y + 1000 * loss_corr
                #overall_loss0 = ((batch_idx + 1) / (batch_idx + 2)) * overall_loss + (1 / (batch_idx + 2)) * loss
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


                    #loss.backward()


                #for name, param in self.parameters():
                 #  if param.requires_grad:
                  #    print(param.grad)


                    #optimizer.step()


                # print(self.kernel.power_param.item())
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
                        #print(len(lengthscale))

                        #print(lengthscale)
                     postfix['lengthscale'] = f"{lengthscale[0].item():.3f}"

              bar.set_postfix(postfix)
        return (model,y_hat,z_int,covar, overall_loss.item())



class DeepIdwAutoEncoderConv(torch.nn.Module):
    def __init__(self,input_dim_x,input_dim_y,hidden_dim, num_layers, depth, p=None, lengthcale = True, device=device ):
        super(DeepIdwAutoEncoderConv,self).__init__()
        #self.device=device
        #self.encoder = Encoder(self.y.shape[1],100,1).to(device)
        #self.decoder = Decoder(1,100, self.y.shape[1]).to(device)
        self.cor_vae_conv  = CorVAEConv(input_dim_x=input_dim_x,input_dim_y=input_dim_y,latent_dim_x=1,latent_dim_y=1,device=device).to(device)
        self.depth = depth
        self.device = device
        self.nn_coefficients  = NNCoefficients(input_dim_x, 1, hidden_dim,
                                               num_layers, depth).to(device)

        self.nn_bias = NNCoefficients(input_dim_x, 1, hidden_dim,
                                               num_layers, depth).to(device)
        #self.nn_coefficients = DeepNN(input_dim=self.x.shape[1], output_dim=depth, hidden_dim=hidden_dim, num_layers=num_layers)
        #constraints = gpytorch.constraints.Interval(torch.tensor([5., 5., 5.]), torch.tensor([30., 30., 15.]))
        self.kernel = InverseDistanceWithParam(ard_num_dims=input_dim_x).to(device)#, lengthscale_constraint=constraints)
        #self.loss = VAELoss().to(device)
    def forward(self,x ,y):
        x=x.to(self.device)
        y_out = y.to(self.device)
        output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self.cor_vae_conv(x, y_out)
        #z, m, var, log_var = self.encoder(y_out)
        z=z_y.flatten().to(self.device)

        #print(y.shape)
        #print(self.nn_coefficients(x,0).flatten().shape)
        for i in range(self.depth):
            z1 = torch.relu(self.nn_coefficients(x,i).flatten()*z + self.nn_bias(x,i).flatten())
            z1.to(device)
            if i < self.depth -1:
                  z =  self.nn_coefficients(x,i+1).flatten()*z1 + self.nn_bias(x,i+1).flatten() + z




            #else:
             #   y_out = y1


        #print('diff',y_out-y)

        covar = self.kernel(x, device=self.device).to(self.device)

        #print(torch.matmul(torch.softmax(covar.evaluate(), dim=1),torch.ones_like(y_out)))
        z_int = torch.matmul(torch.nn.functional.normalize(covar.evaluate(),p=1, dim=1), z ) #/ torch.matmul(covar.evaluate(), torch.ones_like(y_out) )
        #print(z_int.shape)
        #print('reshaped z', z.reshape(-1,1))
        y_hat  = self.cor_vae_conv.vae_conv_y.decoder_conv( z_int.reshape(-1,1))
        #print('y_hat', y_hat)
        return  y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y
    def predict(self, x_test,x,  z):
        # out_test = self.nn(x_test)
        k1 = self.kernel(x_test.to(self.device), x, mode='test',device=self.device).to(self.device)
        #c = torch.nn.functional.softmax(self.raw_c)

        z_pred = torch.matmul(torch.nn.functional.normalize(k1.evaluate()+1e-10,dim=1,p=1),z) #/ torch.matmul(k1.evaluate(), torch.ones_like(y_out) )
       # y_pred = self.cor_vae.vae_y.decoder(z_pred)
        return z_pred
    def train(self, data_loader, n_epochs=200, lr=0.01,beta_x=1, beta_y=1, verbose=True):

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"
        # print(self.para)
        # initialize the coefficients
        #c = torch.ones_like(y)


        #z.to(device)
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss(reduction='mean')
        #criterion = torch.nn.BCELoss()

        #print(self)

        with (tqdm.trange(n_epochs, disable=not verbose) as bar):
            overloss_pred = 0
            for k in bar:

              overall_loss = torch.tensor(0.)

              #model = self
              for batch_idx, (x, y) in enumerate(data_loader):


                #print(batch_idx)
                #print(x.shape)
                x = x.to(self.device)
                y = y.to(self.device)
                #print(x)
                #print(y)
                optimizer.zero_grad()
                y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self(x, y)
                loss_x = self.cor_vae_conv.vae_conv_x.loss(x, output_x, mean_x, log_var_x, beta_x)
                loss_y = self.cor_vae_conv.vae_conv_y.loss(y, y_hat, mean_y, log_var_y, beta_y)


                #multi_corr, corr = multiple_corr(z_x, z_int.reshape(-1,1))

                #loss_corr = (1 - multi_corr) + (corr[0, 1] ** 2 + corr[0, 2] ** 2 + corr[1, 2] ** 2)
                loss_corr = (1 -correlation(z_x, z_int.reshape(-1,1)))
                #print('loss_y',10*loss_y)
                #print('loss_corr', 1000 * loss_corr)
                #print('loss_x1',   loss_x)
                #print('loss_y1',  loss_y)
                #print('loss_corr1',  loss_corr)

                #a = copy.copy(z_y)
                #a.requires_grad_(requires_grad=False)
                #loss_idw = criterion(z_int.flatten(), a.flatten().to(device))

                #loss = ((batch_idx + 1) / (batch_idx + 2)) * loss + (1 / (batch_idx + 2)) * (
                 #           1e-6 * loss_x + 10 * loss_y + 1000 * loss_corr)
                loss = 1e-6 * loss_x + 1000 * loss_y + 1000 * loss_corr
                #overall_loss0 = ((batch_idx + 1) / (batch_idx + 2)) * overall_loss + (1 / (batch_idx + 2)) * loss
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


                    #loss.backward()


                #for name, param in self.parameters():
                 #  if param.requires_grad:
                  #    print(param.grad)


                    #optimizer.step()


                # print(self.kernel.power_param.item())
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
                        #print(len(lengthscale))

                        #print(lengthscale)
                     postfix['lengthscale'] = f"{lengthscale[0].item():.3f}"

              bar.set_postfix(postfix)
        return (model,y_hat,z_int,covar, overall_loss.item())




# model with rbf layers
class DeepIdwAutoEncoderRbf(torch.nn.Module):
    def __init__(self,input_dim_x,input_dim_y,hidden_dim, num_layers, depth, p=None, lengthcale = True, device=device ):
        super(DeepIdwAutoEncoderRbf,self).__init__()
        #self.device=device
        #self.encoder = Encoder(self.y.shape[1],100,1).to(device)
        #self.decoder = Decoder(1,100, self.y.shape[1]).to(device)
        self.cor_vae_rbf  = CorVAERbf(input_dim_x=input_dim_x, hidden_dim_x=3,input_dim_y=input_dim_y, hidden_dim_y=4,latent_dim_x=1,latent_dim_y=1,device=device).to(device)
        self.depth = depth
        self.device = device
        self.nn_coefficients  = NNCoefficients(input_dim_x, 1, hidden_dim,
                                               num_layers, depth).to(device)

        self.nn_bias = NNCoefficients(input_dim_x, 1, hidden_dim,
                                               num_layers, depth).to(device)
        #self.nn_coefficients = DeepNN(input_dim=self.x.shape[1], output_dim=depth, hidden_dim=hidden_dim, num_layers=num_layers)
        #constraints = gpytorch.constraints.Interval(torch.tensor([5., 5., 5.]), torch.tensor([30., 30., 15.]))
        self.kernel = InverseDistanceWithParam(ard_num_dims=input_dim_x).to(device)#, lengthscale_constraint=constraints)
        #self.loss = VAELoss().to(device)
    def forward(self,x ,y):
        x=x.to(self.device)
        y_out = y.to(self.device)
        output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self.cor_vae_rbf(x, y_out)
        #z, m, var, log_var = self.encoder(y_out)
        z=z_y.flatten().to(self.device)

        #print(y.shape)
        #print(self.nn_coefficients(x,0).flatten().shape)
        for i in range(self.depth):
            z1 = torch.relu(self.nn_coefficients(x,i).flatten()*z + self.nn_bias(x,i).flatten())
            z1.to(device)
            if i < self.depth -1:
                  z =  self.nn_coefficients(x,i+1).flatten()*z1 + self.nn_bias(x,i+1).flatten() + z




            #else:
             #   y_out = y1


        #print('diff',y_out-y)

        covar = self.kernel(x, device=self.device).to(self.device)

        #print(torch.matmul(torch.softmax(covar.evaluate(), dim=1),torch.ones_like(y_out)))
        z_int = torch.matmul(torch.nn.functional.normalize(covar.evaluate(),p=1, dim=1), z ) #/ torch.matmul(covar.evaluate(), torch.ones_like(y_out) )
        print('z_int',z_int)
        print('covar',covar.evaluate())
        #print('reshaped z', z.reshape(-1,1))
        y_hat  = self.cor_vae_rbf.vae_rbf_y.decoder_rbf( z_int.reshape(-1,1))
        #print('y_hat', y_hat)
        return  y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y
    def predict(self, x_test,x,  z):
        # out_test = self.nn(x_test)
        k1 = self.kernel(x_test.to(self.device), x, mode='test',device=self.device).to(self.device)
        #c = torch.nn.functional.softmax(self.raw_c)

        z_pred = torch.matmul(torch.nn.functional.normalize(k1.evaluate()+1e-10,dim=1,p=1),z) #/ torch.matmul(k1.evaluate(), torch.ones_like(y_out) )
       # y_pred = self.cor_vae.vae_y.decoder(z_pred)
        return z_pred
    def train(self, data_loader, n_epochs=200, lr=0.01,beta_x=1, beta_y=1, verbose=True):

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"
        # print(self.para)
        # initialize the coefficients
        #c = torch.ones_like(y)


        #z.to(device)
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss(reduction='mean')
        #criterion = torch.nn.BCELoss()

        #print(self)

        with (tqdm.trange(n_epochs, disable=not verbose) as bar):
            overloss_pred = 0
            for k in bar:

              overall_loss = torch.tensor(0.)

              #model = self
              for batch_idx, (x, y) in enumerate(data_loader):


                #print(batch_idx)
                #print(x.shape)
                x = x.to(self.device)
                y = y.to(self.device)
                #print(x)
                #print(y)
                optimizer.zero_grad()
                y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = self(x, y)
                loss_x = self.cor_vae_rbf.vae_rbf_x.loss(x, output_x, mean_x, log_var_x, beta_x)
                loss_y = self.cor_vae_rbf.vae_rbf_y.loss(y, y_hat, mean_y, log_var_y, beta_y)


                multi_corr, corr = multiple_corr(z_x, z_int.reshape(-1,1))

                #loss_corr = (1 - multi_corr) + (corr[0, 1] ** 2 + corr[0, 2] ** 2 + corr[1, 2] ** 2)
                loss_corr = (1 -correlation(z_x, z_int.reshape(-1,1)))
                print('lossx',loss_x)
                print('lossy', loss_y)
                print('z_x',z_x)
                print(loss_corr)
                #print('loss_y',10*loss_y)
                #print('loss_corr', 1000 * loss_corr)
                #print('loss_x1',   loss_x)
                #print('loss_y1',  loss_y)
                #print('loss_corr1',  loss_corr)

                #a = copy.copy(z_y)
                #a.requires_grad_(requires_grad=False)
                #loss_idw = criterion(z_int.flatten(), a.flatten().to(device))

                #loss = ((batch_idx + 1) / (batch_idx + 2)) * loss + (1 / (batch_idx + 2)) * (
                 #           1e-6 * loss_x + 10 * loss_y + 1000 * loss_corr)
                loss = 1e-6 * loss_x + 1000 * loss_y + 1000*loss_corr
                #overall_loss0 = ((batch_idx + 1) / (batch_idx + 2)) * overall_loss + (1 / (batch_idx + 2)) * loss

                overall_loss =((batch_idx + 1) / (batch_idx + 2)) * overall_loss + (1 / (batch_idx + 2)) *loss

                torch.cuda.empty_cache()

              #overall_loss
              if k == 1:
                  overloss_pred = overall_loss
              if overall_loss < overloss_pred:
                overloss_pred = overall_loss
                model = self





              with autograd.detect_anomaly():
               print(loss)
               loss.backward()
               #torch.nn.utils.clip_grad_norm_(self.parameters(), .5)
               optimizer.step()


                    #loss.backward()


                    #optimizer.step()


                # print(self.kernel.power_param.item())
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
                        #print(len(lengthscale))

                        #print(lengthscale)
                     postfix['lengthscale'] = f"{lengthscale[0].item():.3f}"

              bar.set_postfix(postfix)
        return (model,y_hat,z_int,covar, overall_loss.item())


