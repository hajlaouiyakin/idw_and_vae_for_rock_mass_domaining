"""
This module is set in order to implement a parametric IDW spatial wight matrix in gpytorch 

"""
from __future__ import annotations
import torch
import copy
from math import pi
import gpytorch
from gpytorch.constraints import Positive
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class InverseDistanceWithParam(gpytorch.kernels.Kernel):
    has_lengthscale = True
    is_stationary = True
    def __init__(self, power_param_prior= None, power_param_constraint = None, device=device, **kwargs):

        super(InverseDistanceWithParam, self).__init__(**kwargs)
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
            return k
    




 