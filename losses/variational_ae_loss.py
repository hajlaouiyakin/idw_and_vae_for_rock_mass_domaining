import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
In this file we compute Kullback-Leibler divergence and the Variational AutoEncoder loss with beta variation
'''

class KLD(torch.nn.Module):
    def __init__(self):
        super(KLD, self).__init__()

    def forward(self, mean, log_var):
        return - 0.5*torch.sum(1 + log_var - mean**2- log_var.exp())


class VAELoss(torch.nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='mean').to(device)
        self.kld = KLD().to(device)

    def forward(self, x, x_hat, mean, log_var,beta=.1):

        return  self.mse(x_hat.to(device), x.to(device)) + beta*self.kld(mean.to(device), log_var.to(device))




