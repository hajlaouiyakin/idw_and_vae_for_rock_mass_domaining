from __future__ import annotations

import math

import torch
from torch.nn import Linear, Conv1d,ConvTranspose1d, Conv2d, ConvTranspose2d

import torch
from torch.utils.data import DataLoader
from losses.variational_ae_loss import VAELoss, DiscriminantVAELoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Encoder
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device=device):
        super(Encoder, self).__init__()
        #print(device)
        self.fc_input = Linear(input_dim, hidden_dim).to(device)
        self.hidden = Linear(hidden_dim, hidden_dim).to(device)
        self.fc_mean =Linear(hidden_dim, latent_dim).to(device)
        self.fc_log_var = Linear(hidden_dim, latent_dim).to(device)

    def forward(self, x):
        #print(x)
        h = torch.tanh(self.hidden(torch.relu(self.fc_input(x))))


        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        var = torch.exp(0.5 * log_var)
        z = mean + torch.randn_like(var) * var
        return z, mean, var, log_var


# Decoder
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim,device=device):
        super(Decoder, self).__init__()
        self.fc_hidden = Linear(latent_dim, hidden_dim).to(device)
        self.fc_output = Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        h = torch.tanh(self.fc_hidden(x))
        output = self.fc_output(h)
        #output = torch.sigmoid_(self.fc_output(h))
        return output


# Variational auto encoder model

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

    def train_model(self, dataset_train, num_epochs=1000, lr=0.01, batch_size=100, beta = 1):
        training_parameters = [{'params': self.parameters()}]
        optimizer = torch.optim.Adamax(training_parameters, lr=lr)
        data_loader_train = DataLoader(dataset_train, batch_size, num_workers=0, shuffle=False)
        for epoch in range(num_epochs):
            overall_loss = 0
            for idx, (x, _) in enumerate(data_loader_train):
                #x = torch.sigmoid_(x)
                optimizer.zero_grad()
                output, z, mean, _, log_var = self(x)
                print(output)
                loss = self.loss(x, output, mean, log_var,beta)
                loss.backward()
                optimizer.step()
                #rint(loss.retain_grad())


                overall_loss += loss.item()
                print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (idx * batch_size+1))

# conditional encoder
class CondEncoder(torch.nn.Module):
    def __init__(self, input_dim,dim_c, hidden_dim, latent_dim):
        super(CondEncoder, self).__init__()
        dim = input_dim + dim_c
        self.fc_input = Linear(dim, hidden_dim)
        self.fc_mean = Linear(hidden_dim, latent_dim)
        self.fc_log_var = Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        x = torch.cat((x, c), dim=1)
        h = torch.relu(self.fc_input(x))
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        var = torch.exp(0.5 * log_var)
        z = mean + torch.randn_like(var) * var
        return z, mean, var, log_var



class CondDecoder(torch.nn.Module):
    def __init__(self, latent_dim,dim_c, hidden_dim, output_dim):
        super(CondDecoder, self).__init__()
        dim = latent_dim +dim_c
        self.fc_hidden = Linear(dim, hidden_dim)
        self.fc_output = Linear(hidden_dim, output_dim)

    def forward(self, x,c):
        #print(c.shape)
        #print(x.shape)
        x= torch.cat((x,c), dim=1)
        h = torch.tanh(self.fc_hidden(x))
        output = torch.relu(self.fc_output(h))

        #output = torch.sigmoid_(self.fc_output(h))
        return output



class CondVariationalAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, dim_c, hidden_dim, latent_dim, output_dim):
        super(CondVariationalAutoEncoder, self).__init__()
        self.encoder = CondEncoder(input_dim,dim_c, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
        self.loss = VAELoss()

    def forward(self, x, c):
        z, mean, var, log_var = self.encoder(x, c)
        output = self.decoder(z)
        return output, z, mean, var, log_var

    def train_model(self, dataset_train, num_epochs=1000, lr=0.01, batch_size=100, beta = 1):
        training_parameters = [{'params': self.parameters()}]
        optimizer = torch.optim.Adamax(training_parameters, lr=lr)
        data_loader_train = DataLoader(dataset_train, batch_size, num_workers=0, shuffle=False)
        for epoch in range(num_epochs):
            overall_loss = 0
            for idx, (x, y) in enumerate(data_loader_train):
                #x = torch.sigmoid_(x)
                optimizer.zero_grad()
                output, z, mean, _, log_var = self(x,y)
                print( output)
                loss = self.loss(x, output, mean, log_var,beta)
                loss.backward()
                optimizer.step()
                #rint(loss.retain_grad())


                overall_loss += loss.item()
                print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (idx * batch_size+1))

def loss_whole_dataset(data_loader, model, criterion, beta):
    overall_loss = 0
    i = 0
    for idx, (x, y) in enumerate(data_loader):
        z, mean, var, log_var = model(x)
        loss = criterion(z, y, mean, log_var,beta)

        overall_loss += loss.item()
        i = i + 1
    return 1/i*overall_loss





class Variational_discriminant(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Variational_discriminant, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.loss = DiscriminantVAELoss()

    def forward(self, x):
        z, mean, var, log_var = self.encoder(x)
        return   z, mean, var, log_var

    def train_model(self, dataset_train,dataset_val,  num_epochs=1000, lr=0.01, batch_size=100, beta=1):
        training_parameters = [{'params': self.parameters()}]
        optimizer = torch.optim.Adamax(training_parameters, lr=lr)
        data_loader_train = DataLoader(dataset_train, batch_size, num_workers=0, shuffle=False)
        data_loader_val = DataLoader(dataset_val, batch_size, num_workers=0, shuffle=False)
        best_val_loss = 0
        best_model = None
        for epoch in range(num_epochs):
            overall_loss = 0
            for idx, (x, y) in enumerate(data_loader_train):
                # x = torch.sigmoid_(x)
                optimizer.zero_grad()
                z, mean, var, log_var = self(x)
                loss = self.loss(z, y, mean, log_var,beta)
                loss.backward()
                optimizer.step()
                val_loss = loss_whole_dataset(data_loader_val,self, self.loss, beta)
                train_loss = loss_whole_dataset(data_loader_train, self, self.loss, beta)
                if epoch % 1 == 0:
                    print(f"Epoch {epoch:2d}, \
                                 Train:loss={train_loss:.3f}\
                                 Val: loss={val_loss:.3f}"  ) #, accuracy={accuracy_val.item() * 100:.1f}%",

                if -1*val_loss > -1*best_val_loss:
                    best_model = self
                    best_val_loss = val_loss
        return best_model, best_val_loss


                # rint(loss.retain_grad())

                #overall_loss += loss.item()
                #print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (idx * batch_size + 1))

## CovAutoEncoder
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
        #print('in',h.shape)
        h = self.deconv_hidden(h)
        #print(h)
        #h=h.view(h.size(0),h.size(1),1)
        #print('in convhidd',h.shape)
        output = self.deconv_output(h)
        output = output.view(output.size(0),-1)
        output = self.fc_output(torch.tanh(output))
        #print(output.shape)
        return output
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
       # self.init_param()
    #def init_param(self):
     #   torch.nn.init.normal_(self.centers,0,1)
      #  torch.nn.init.constant_(self.log_sigmas,0)
    def forward(self, x):
        x = x.unsqueeze(1)

        # Compute the distance between each input point and each center
        distances = torch.norm(x - self.centers, dim=2)
        alpha = distances/((2*self.widths.pow(2)).pow(.5))

        # Compute the activation using radial basis function (Gaussian)
        #activation = torch.exp(-distances.pow(2) / (2 * self.widths.pow(2)))
        if self.fun == 'gaussian':
            activation = torch.exp(-1*alpha.pow(2))
        elif self.fun == 'inverse_quadratic':
            activation = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
        elif self.fun == 'spline':
            activation = alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha))
        else:
            activation = alpha

        return activation


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
        #print(x)
        h = self.rbf_hidden(torch.tanh(self.hidden(self.rbf_input(x))))


        mean = self.fc_mean(torch.tanh(h))
        log_var = self.fc_log_var(torch.tanh(h))
        var = torch.exp(0.5 * log_var)
        z = mean + torch.randn_like(var) * var
        #print(z.shape)

        return z, mean, var, log_var

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
