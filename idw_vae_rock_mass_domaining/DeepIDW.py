from __future__ import annotations
import torch
import tqdm
import gpytorch
#from models.models_utils import FeatureExtractor, Perturbation
from models.kernels import ExponentialKernel, SimpleSincKernel,  InverseDistance, InverseDistanceWithParam, KnnKernel
from torch import autograd
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# A deep neural network with customized size
class Layer(torch.nn.Module):
    def __init__(self, layer_dim):
        super(Layer, self).__init__()
        self.fc = torch.nn.Linear(layer_dim, layer_dim,bias=True)
    def forward(self, x):
        out = torch.relu(self.fc(x))
        return out

class HiddenLayers(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(HiddenLayers, self).__init__()
        self.hidden_layers = torch.nn.ModuleList([Layer(hidden_dim) for i in range(num_layers)])
    def forward(self,x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x

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
        #print('seq_out',out)
        #print(torch.sigmoid((out)))

        return out





# a deep neural network structure that provide coefficients for each layer
# inp: available features, Output: N coefficients for N target vectors
class NNCoefficients(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, depth ):
        super(NNCoefficients,self).__init__()
        self.module_list = torch.nn.ModuleList([DeepNN(input_dim, output_dim, hidden_dim, num_layers) for i in range(depth)])
    def forward(self,x , i):
        out =self.module_list[i](x)
        return(out)

class DeepIDW(torch.nn.Module):
    def __init__(self, x, y, hidden_dim, num_layers, depth, p=None, lengthcale = True ):
        super(DeepIDW,self).__init__()
        self.x = x.to(device)
        self.y = y.to(device)
        self.depth = depth
        self.nn_coefficients  = NNCoefficients(self.x.shape[1], 1, hidden_dim,
                                               num_layers, depth)
        self.nn_bias = NNCoefficients(self.x.shape[1], 1, hidden_dim,
                                               num_layers, depth)
        #self.nn_coefficients = DeepNN(input_dim=self.x.shape[1], output_dim=depth, hidden_dim=hidden_dim, num_layers=num_layers)
        #constraints = gpytorch.constraints.Interval(torch.tensor([5., 5., 5.]), torch.tensor([30., 30., 15.]))
        self.kernel = InverseDistanceWithParam(ard_num_dims=self.x.shape[1]).to(device)#, lengthscale_constraint=constraints)
    def forward(self,x ,y):
        x=x.to(device)
        y_out = y.flatten().to(device)
        #print(y.shape)
        #print(self.nn_coefficients(x,0).flatten().shape)
        for i in range(self.depth):
            y1 = torch.relu(self.nn_coefficients(x,i).flatten()*y_out + self.nn_bias(x,i).flatten())
            y1.to(device)
            if i < self.depth -1:
                  y_out =  self.nn_coefficients(x,i+1).flatten()*y1 + self.nn_bias(x,i+1).flatten() + y_out


            #else:
             #   y_out = y1


        #print('diff',y_out-y)

        covar = self.kernel(x).to(device)

        #print(torch.matmul(torch.softmax(covar.evaluate(), dim=1),torch.ones_like(y_out)))
        y_int = torch.matmul(torch.nn.functional.normalize(covar.evaluate(),p=1, dim=1), y_out ) #/ torch.matmul(covar.evaluate(), torch.ones_like(y_out) )
        return y_int, covar, y_out
    def predict(self, x_test, y_out):
        # out_test = self.nn(x_test)
        k1 = self.kernel(x_test.to(device), self.x, mode='test').to(device)
        #c = torch.nn.functional.softmax(self.raw_c)

        y_pred = torch.matmul(torch.nn.functional.normalize(k1.evaluate()+1e-10,dim=1,p=1), y_out ) #/ torch.matmul(k1.evaluate(), torch.ones_like(y_out) )
        return y_pred
    def train(self, x, y,z, n_epochs=200, lr=0.01, verbose=True):
        # print(self.para)
        # initialize the coefficients
        #c = torch.ones_like(y)
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
                    loss = criterion(out.flatten(), z.flatten().to(device))
                    loss.backward()


                #for name, param in self.:
                   #if param.requires_grad:
                    #  print(param.grad)
                     # print(name, param.data)

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
            return (self,y_out, e, loss.item())

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
        # self.nn_coefficients = DeepNN(input_dim=self.x.shape[1], output_dim=depth, hidden_dim=hidden_dim, num_layers=num_layers)
        #constraints = gpytorch.constraints.Interval(torch.tensor([5., 5., 5.]), torch.tensor([30., 30., 15.]))
        #pwr_constraints= gpytorch.constraints.GreaterThan(2.)
        self.kernel = InverseDistanceWithParam(ard_num_dims=self.x.shape[1])#, power_param_constraint=pwr_constraints)#, lengthscale_constraint=constraints)

    def forward(self, x, y):
        y_out = y.flatten()
        print(y_out.shape)
        #print(y.shape)
        #print(self.nn_coefficients(x, 0).flatten().shape)
        for i in range(self.depth):
            y1 = torch.relu(self.nn_coefficients(x, i).flatten() * y_out + self.nn_bias(x, i).flatten())
            if i < self.depth - 1:
                y_out = self.nn_coefficients(x, i + 1).flatten() * y1 + self.nn_bias(x, i + 1).flatten() + y_out
            # else:
            #   y_out = y1

        #print('diff', y_out - y)

        covar = self.kernel(x)


        # print(torch.matmul(torch.softmax(covar.evaluate(), dim=1),torch.ones_like(y_out)))
        y_int = torch.matmul(torch.softmax(torch.log(covar.evaluate() + 1e-10), dim=1),
                             y_out)  # / torch.matmul(covar.evaluate(), torch.ones_like(y_out) )

        prob = torch.softmax(self.nn_post_inter(y_int.reshape(-1,1)), dim=1)
        return y_int, covar, y_out, prob

    def predict(self, x_test, y_out):
        # out_test = self.nn(x_test)
        k1 = self.kernel(x_test, self.x, mode='test')
        # c = torch.nn.functional.softmax(self.raw_c)

        y_pred = torch.matmul(torch.softmax(torch.log(k1.evaluate() + 1e-10), dim=1),
                              y_out)  # / torch.matmul(k1.evaluate(), torch.ones_like(y_out) )
        prob_pred = torch.softmax(self.nn_post_inter(y_pred.reshape(-1,1)), dim=1)
        return y_pred, prob_pred
    def train(self, x, y,z, n_epochs=200, lr=0.01, verbose=True):
        # print(self.para)
        # initialize the coefficients
        #c = torch.ones_like(y)
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        #criterion = torch.nn.MSELoss()
        criterion = torch.nn.CrossEntropyLoss()
       # print(self)
        with tqdm.trange(n_epochs, disable=not verbose) as bar:
            for _ in bar:
                optimizer.zero_grad()
                out, covar, y_out, prob = self(x, y)
                if  not torch.isnan(prob).max():
                    prob1 = prob
                else:
                    return (self, y_out, prob1, loss.item())
                    break
                print(z.shape)
                print(prob.shape)
                loss = criterion(prob, z.flatten())
                loss.backward()
                #for name, param in self.nn_coefficients.named_parameters():
                   #if param.requires_grad:
                      #print(param.grad)
                      #print(name, param.data)

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
            return (self,y_out,prob1,loss.item())










"""
src= "D:/PHD math appliquées/Research/python_project/data/Data.csv"
#get the sequential hole data_sets
np.random.seed(3)
df = pd.read_csv(src, sep=",", encoding='cp1252')
targets = ["BI"]
predictors = ["CoordX", "CoordY", "CoordZ","WOB"]#, "RPM", "WOB", "TRQ", "ROP"]
hole_names = [i for i in range(1, 74) if i != 7]
sequential_data = SequentialData(df, predictors, targets, hole_names)
dataset_seq = sequential_data.forward()

np.random.seed(0)
hole_names_test = [20,37, 50, 62, 72 ]
#hole_names_test = [18]
#hole_names_test =[69,70,71,72,73]
#hole_names_test =[64,65,66,67,68,69,70,71,72,73]# np.random.choice(hole_names, 30, replace = False)
#hole_names_test = np.random.choice(hole_names,5 , replace = False)
#hole_names_test = [21, 38, 51, 63, 73]
hole_names_train = [i for i in hole_names if i not in hole_names_test]
#hole_names_train = [41,39, 51, 52, 42, 63]

# the original training predictors and target variables

x_train = dataset_seq[hole_names_train[0]].x[:,0:3]
y_train = dataset_seq[hole_names_train[0]].x[:,3].flatten()
z_train = dataset_seq[hole_names_train[0]].y
for i in range(1,len(hole_names_train)+1):
    if i < len(hole_names_train):
        x_train = torch.cat((x_train, dataset_seq[hole_names_train[i]].x[:,0:3]), dim=0)
        y_train =torch.cat((y_train, dataset_seq[hole_names_train[i]].x[:,3].flatten()), dim=0)
        z_train = torch.cat((z_train, dataset_seq[hole_names_train[i]].y), dim=0)

# the original training predictors and target variables
x_test = dataset_seq[hole_names_test[0]].x[:,0:3]

y_test = dataset_seq[hole_names_test[0]].x[:,3].flatten()
z_test = dataset_seq[hole_names_test[0]].y
for i in range(1,len(hole_names_test)+1):
    if i < len(hole_names_test):
        x_test = torch.cat((x_test, dataset_seq[hole_names_test[i]].x[:,0:3]), dim=0)
        z_test = torch.cat((z_test, dataset_seq[hole_names_test[i]].y), dim=0)
        y_test = torch.cat((y_test, dataset_seq[hole_names_test[i]].x[:, 3]), dim=0)
model = DeepIDW(x_train, y_train, hidden_dim=50, depth=10, num_layers=10)
torch.autograd.set_detect_anomaly(True)
_,y_out, e, train_loss = model.train(x_train, y_train,z_train, n_epochs=100, lr=.001)
y_pred = model.predict(x_test, y_out)
print(r_squared(y_pred, z_test, z_train))
for i in range(len(hole_names_test)):
    x_test1 = dataset_seq[hole_names_test[i]].x[:,0:3].reshape(-1,3)
    z_test1 = dataset_seq[hole_names_test[i]].y
    y_test1 = dataset_seq[hole_names_test[i]].x[:,3].flatten()
    y_pred = model.predict(x_test1, y_out)
    print(r_squared(y_pred.flatten(),z_test1.flatten(), z_train.flatten() ))
    print("z_test1", z_test1)
    print("y_test1", y_test1)
    print("pred", y_pred)
    print(y_out)
"""






