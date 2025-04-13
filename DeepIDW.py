from __future__ import annotations
import torch
import tqdm
import gpytorch
from kernels import  InverseDistanceWithParam
from torch import autograd
device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''
DeepIDW is a spatial Interpolater that is compatible with neural network based on applying backpropagation 
inference on IDW and adding nonlinear layers (inspired from deep neural network)  
it will be stacked with Variational Auroencoders as described in the paper.
 For details about deep DeepIDW please see our papers:
https://www.sciencedirect.com/science/article/pii/S0098300424002395
https://publications.waset.org/10013861/enhancing-spatial-interpolation-a-multi-layer-inverse-distance-weighting-model-for-complex-regression-and-classification-tasks-in-spatial-data-analysis

'''


#  one neural network Layer
class Layer(torch.nn.Module):
    def __init__(self, layer_dim):
        super(Layer, self).__init__()
        self.fc = torch.nn.Linear(layer_dim, layer_dim,bias=True)
    def forward(self, x):
        out = torch.relu(self.fc(x))
        return out
# Building  Hidden NN layers 
class HiddenLayers(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(HiddenLayers, self).__init__()
        self.hidden_layers = torch.nn.ModuleList([Layer(hidden_dim) for i in range(num_layers)])
    def forward(self,x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x
# Modeling deepNN based on a number of hidden layers 
class DeepNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(DeepNN, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim,bias=True)
        self.hidden_layers = HiddenLayers(hidden_dim,num_layers)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim,bias=True)
        self.sequential = torch.nn.Sequential(
            self.input_layer,
            self.hidden_layers,
            self.output_layer,
            #torch.nn.Softmax()

      )
    def forward(self, x):
        out = self.sequential(x)

        return out
    
# Coefficient for DeepIDW model used for modeling biases and coefficients to to make linear transformation to 
# the observation so that we apply interpolation on the tranformed values
class NNCoefficients(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, depth ):
        super(NNCoefficients,self).__init__()
        self.module_list = torch.nn.ModuleList([DeepNN(input_dim, output_dim, hidden_dim, num_layers) for i in range(depth)])
    def forward(self,x , i):
        out =self.module_list[i](x)
        return(out)
# The  DeepIDW model  
class DeepIDW(torch.nn.Module):
    def __init__(self, x, y, hidden_dim, num_layers, depth, p=None, lengthcale = True ):
        super(DeepIDW,self).__init__()
        
        self.x = x.to(device)   # Location (coordinates
        self.y = y.to(device)   # target
        self.depth = depth      # depth of the deepidw model
        self.nn_coefficients  = NNCoefficients(self.x.shape[1], 1, hidden_dim,
                                               num_layers, depth) # coefficients
        self.nn_bias = NNCoefficients(self.x.shape[1], 1, hidden_dim,
                                               num_layers, depth)  # bias
        self.kernel = InverseDistanceWithParam(ard_num_dims=self.x.shape[1]).to(device) # kernel matrix 
    def forward(self,x ,y):
        x=x.to(device)
        y_out = y.flatten().to(device)
        # create layers of nonlinear transformaion of depth self.depth
        for i in range(self.depth):
            y1 = torch.relu(self.nn_coefficients(x,i).flatten()*y_out + self.nn_bias(x,i).flatten())
            y1.to(device)
            if i < self.depth -1:
                  y_out =  self.nn_coefficients(x,i+1).flatten()*y1 + self.nn_bias(x,i+1).flatten() + y_out
        # compute the kernel
        covar = self.kernel(x).to(device)
        # interplate on the transformed values
        y_int = torch.matmul(torch.nn.functional.normalize(covar.evaluate(),p=1, dim=1), y_out ) #/ torch.matmul(covar.evaluate(), torch.ones_like(y_out) )
        return y_int, covar, y_out
    def predict(self, x_test, y_out):
        k1 = self.kernel(x_test.to(device), self.x, mode='test').to(device)
        y_pred = torch.matmul(torch.nn.functional.normalize(k1.evaluate()+1e-10,dim=1,p=1), y_out ) #/ torch.matmul(k1.evaluate(), torch.ones_like(y_out) )
        return y_pred
    def train(self, x, y,z, n_epochs=200, lr=0.01, verbose=True):
        x.to(device)

        z.to(device)
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        #print(self)
        with tqdm.trange(n_epochs, disable=not verbose) as bar:
            for _ in bar:
                optimizer.zero_grad()
                out, covar, y_out = self(x, y)
                y.to(device)

                e = torch.tensor((out - y.to(device)) ** 2, requires_grad=False).to(device)
        
                with autograd.detect_anomaly():
                    loss = criterion(out.flatten(), z.flatten().to(device))
                    loss.backward()
                    optimizer.step()
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
                        postfix['lengthscale'] = f"{lengthscale[0].item():.3f}"

                bar.set_postfix(postfix)
            return (self,y_out, e, loss.item())
        

# Deep IDW for classification (in case we have categorical target)

class DeepIDWClassifier(torch.nn.Module):
    def __init__(self, x, y, hidden_dim, num_layers,num_labels, depth, p=None, lengthcale=True):
        super(DeepIDWClassifier, self).__init__()
        self.x = x
        self.y = y
        self.depth = depth
        self.nn_coefficients = NNCoefficients(self.x.shape[1], 1, hidden_dim,
                                              num_layers, depth)
        self.nn_bias = NNCoefficients(self.x.shape[1], 1, hidden_dim,
                                      num_layers, depth)
        self.nn_post_inter = DeepNN(input_dim=1, output_dim=num_labels,hidden_dim=hidden_dim, num_layers=num_layers)
        self.kernel = InverseDistanceWithParam(ard_num_dims=self.x.shape[1])#, power_param_constraint=pwr_constraints)#, lengthscale_constraint=constraints)

    def forward(self, x, y):
        y_out = y.flatten()

        for i in range(self.depth):
            y1 = torch.relu(self.nn_coefficients(x, i).flatten() * y_out + self.nn_bias(x, i).flatten())
            if i < self.depth - 1:
                y_out = self.nn_coefficients(x, i + 1).flatten() * y1 + self.nn_bias(x, i + 1).flatten() + y_out


        covar = self.kernel(x)
        y_int = torch.matmul(torch.softmax(torch.log(covar.evaluate() + 1e-10), dim=1),
                             y_out)  

        prob = torch.softmax(self.nn_post_inter(y_int.reshape(-1,1)), dim=1)
        return y_int, covar, y_out, prob

    def predict(self, x_test, y_out):
    
        k1 = self.kernel(x_test, self.x, mode='test')
        y_pred = torch.matmul(torch.softmax(torch.log(k1.evaluate() + 1e-10), dim=1),
                              y_out)  
        prob_pred = torch.softmax(self.nn_post_inter(y_pred.reshape(-1,1)), dim=1)
        return y_pred, prob_pred
    def train(self, x, y,z, n_epochs=200, lr=0.01, verbose=True):
   
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
     
        criterion = torch.nn.CrossEntropyLoss()

        with tqdm.trange(n_epochs, disable=not verbose) as bar:
            for _ in bar:
                optimizer.zero_grad()
                out, covar, y_out, prob = self(x, y)
                if  not torch.isnan(prob).max():
                    prob1 = prob
                else:
                    return (self, y_out, prob1, loss.item())
                    break
                
                loss = criterion(prob, z.flatten())
                loss.backward()
               

                optimizer.step()

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
                    
                        postfix['lengthscale'] = f"{lengthscale[0].item():.3f}"

                bar.set_postfix(postfix)
            return (self,y_out,prob1,loss.item())















