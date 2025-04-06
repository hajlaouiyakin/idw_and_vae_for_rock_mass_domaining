"""
This module is set in order to implement a custom exponential kernel in GPytorch
"""
from __future__ import annotations
import torch
import copy
from math import pi
import gpytorch
from gpytorch.constraints import Positive, Interval
from models.softsort import SoftSort, KnnWeights
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class ExponentialKernel(gpytorch.kernels.Kernel):
    # The exponential kernel is stationary
    #is_stationary = True
    has_lengthscale = True
    def forward(self, x1, x2, diag=False, **params):
        def postprocess_fun(dist):
            return dist.div_(-3).exp_()
        x1_  = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        dist  = self.covar_dist(x1_, x2_,**params)
        dist.where(dist == 0, torch.as_tensor(1e-20))
        #a = self.covar_dist(x1[:, 0:2].view(-1, 2), x2[:, 0:2].view(-1, 2), **params)
        #a[a > 1e-15] = 1

        return torch.exp(-3*dist)

class SimpleSincKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # apply lengthscale
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        # calculate the distance between inputs
        diff = self.covar_dist(x1_, x2_, **params)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        # return sinc(diff) = sin(diff) / diff
        return torch.sin(diff).div(diff)


class InverseDistance(gpytorch.kernels.Kernel):
    is_stationary = True

    def forward(self, x1, x2, mode = 'train', **params):
        diff = self.covar_dist(x1, x2, **params)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        k= 1 / diff
        if mode == 'train':
            k.fill_diagonal_(0)
        return k
class InverseDistanceWithParam(gpytorch.kernels.Kernel):
    has_lengthscale = True
    is_stationary = True
    def __init__(self, power_param_prior= None, power_param_constraint = None, device=device, **kwargs):

        super(InverseDistanceWithParam, self).__init__(**kwargs)

        #self.is_stationary = is_stationary
        #self.has_lengthscale = has_legthscale
        if power_param_constraint == None:
            power_param_constraint = Positive()
        self.register_parameter(name = 'raw_power_param', parameter=torch.nn.Parameter(torch.ones(*self.batch_shape,1,1)))
        if power_param_prior is not None:
            self.register_prior('power_param_prior', power_param_prior, lambda m: m.power_param, lambda m,v: m._set_power_param(v))
        self.register_constraint("raw_power_param", power_param_constraint)

    @property
    def power_param(self):
        return self.raw_power_param_constraint.transform(self.raw_power_param)
    @power_param.setter
    def power_param(self, value):
        self._set_power_param(value)
    def _set_power_param(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_power_param)
        self.initialize(raw_power_param = self.raw_power_param_constraint.inverse_transform(value))
    def forward(self, x1, x2, mode = 'train',device=device, **params):
        if self.has_lengthscale:
            #print(self.lengthscale)
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
        else:
            x1_ = x1
            x2_ = x2
        if mode == 'train':
            diff = self.covar_dist(x1_, x2_, **params) + torch.eye(x1_.shape[0]).to(device)
        else:
            diff = self.covar_dist(x1_, x2_, **params)

        diff.where(diff == 0, torch.as_tensor(1e-20)).to(device)
        a = self.covar_dist(x1[:, 0:2].view(-1, 2), x2[:, 0:2].view(-1, 2), **params).to(device)
        a[a > 1e-15] = 1
        if mode == 'train':

            diff.fill_diagonal_(1.)
            k = torch.exp(self.power_param*torch.log(1 / diff + 1e-7  )) - torch.eye(diff.shape[0]).to(device)


            return k*torch.ones_like(k).fill_diagonal_(1e-40).to(device)
        else:
            k = torch.exp(self.power_param*torch.log(1 / (diff+1e-3) + 1e-7  ))
            #print('grad',torch.log1p(1 / diff  ))
            return k
        #a = self.covar_dist(x1[:,0:2].view(-1,2), x2[:,0:2].view(-1,2), **params)
        #print('z1', x1[:,2])
        #print('z2',x2[:,2] )
        #print('a avant',a )
        #print('k', k)
        #a[a>1e-15] = 1
        #print('a apres',a)
        #if mode == 'train':
         #  return k*torch.ones_like(k).fill_diagonal_(1e-40)*a
        #else:
         #  return k*a





        #return 1/diff

class KnnKernel(gpytorch.kernels.Kernel):
    is_stationary = True
    has_lengthscale = True
    def __init__(self, k, **kwargs):
        super(KnnKernel, self).__init__(**kwargs)
        #self.softsort = SoftSort(hard=True)
        self.knnweight = KnnWeights(k=k)
        self.k  = k
    def forward(self, x1, x2, mode = 'train', **params ):
        if self.has_lengthscale:
            # print(self.lengthscale)
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
        else:
            x1_ = x1
            x2_ = x2
        diff = self.covar_dist(x1_, x2_, **params)
        #k1 = self.softsort(diff)
        P = self.knnweight(diff)
        #P = k1[:,:self.k,:].sum(1)
        a = self.covar_dist(x1[:, 0:2].view(-1, 2), x2[:, 0:2].view(-1, 2), **params)
        a[a > 1e-15] = 1
        if mode == 'train':
            return P * torch.ones_like(P).fill_diagonal_(1e-40)*a
        else:
            return P*a

