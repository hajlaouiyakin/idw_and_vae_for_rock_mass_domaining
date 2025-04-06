import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class KLD(torch.nn.Module):
    def __init__(self):
        super(KLD, self).__init__()

    def forward(self, mean, log_var):
        print('meanpow',mean**2)
        print(- 0.5*torch.sum(1 + log_var - mean**2- log_var.exp()))
        return - 0.5*torch.sum(1 + log_var - mean**2- log_var.exp())


class VAELoss(torch.nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        #self.bce = torch.nn.BCELoss(reduction='mean')
        self.mse = torch.nn.MSELoss(reduction='mean').to(device)
        self.kld = KLD().to(device)

    def forward(self, x, x_hat, mean, log_var,beta=.1):
        #print(self.mse(x_hat.to(device), x.to(device)))
        #print(beta*self.kld(mean.to(device), log_var.to(device)))
        return  self.mse(x_hat.to(device), x.to(device)) + beta*self.kld(mean.to(device), log_var.to(device))

class DiscriminantVAELoss(torch.nn.Module):
    def __init__(self):
        super(DiscriminantVAELoss, self).__init__()
        # self.bce = torch.nn.BCELoss(reduction='mean')
        #self.mse = torch.nn.MSELoss(reduction='mean')
        self.kld = KLD()
    def forward(self, z, y, mean, log_var,beta):
        C= y.shape[1]
        N = y.sum(0, keepdim=True)      # vector containing (N1, N2, ...., Nc)
        z_c = mean[:, :, None]*y[:,None,:]        # matrix  (batch, p, C)
       # print(z_c.shape)
        m_c =1/N* z_c.sum(0,keepdim =True )  # dim = (1,p, C)
       # print(m_c.shape)
        m =1/C* m_c.sum(2, keepdim = True ) # dim = (1,p, 1)
        # compute between classes
        #dm = torch.cdist(torch.transpose(m_c,-1,-2).squeeze(0), torch.transpose(m,-1,-2).squeeze(0)).sum()
        # compute the distances within classes
        #dw =torch.norm(torch.transpose(z_c-m_c,-1,-2), dim =2).sum()

        Sb = 1/C*torch.matmul(m_c - m, torch.transpose(m_c-m, -1,-2)).squeeze(0)  # dim = (1,p,p)
        #print(Sb)
        Sw =(1/C)*(1/y.shape[0])*(torch.matmul(z_c - m_c, torch.transpose(z_c - m_c, -1, -2))).sum(0, keepdim=True).squeeze(0)

        #print(Sb.shape)
        M = torch.matmul(torch.cholesky_inverse(Sb), Sw)
        #print(M.shape)
        J = torch.trace(M)
        print("metric J", J)
        return 100*J + beta*self.kld(mean, log_var)


        #print(self.mse(x_hat, x))
        #print(beta*self.kld(mean, log_var))
        return    beta*self.kld(mean, log_var)




