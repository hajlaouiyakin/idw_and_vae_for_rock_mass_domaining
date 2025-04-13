
import pandas as pd
import idw_cluster

from torch import torch
import gc
from utils.load_data import CustomDataLoader, reduce_bi_data
#import pycuda.driver as cuda
#import pycuda.autoinit
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from utils.load_data import  ScaledBIDataSet
#device = 'cpu'


# read data

src = 'drilling_data.csv'
   
df = pd.read_csv(src)    
# reduce the resolution of data 
df = reduce_bi_data(df, resol=0.3)

predictors = ["CoordX", "CoordY", "Depth"]
targets = ["ROP", "WOB", "RPM", "TRQ"]
BI_dataset = ScaledBIDataSet(df=df, predictors=predictors, targets=targets, scale_predictors=True,
                             scale_targets=True, mode='train', transform=None) # scaled data set 
#FC
#model = idw_cluster.DeepIdwAutoEncoderBatch(3, 4, hidden_dim=3, num_layers=1, depth=1, device=device)
# Conv
#model = idw_cluster.DeepIdwAutoEncoderConv(3, 4, hidden_dim=3, num_layers=1, depth=1,device=device)
#RBF
model = idw_cluster.DeepIdwAutoEncoderRbf(3, 4, hidden_dim=3, num_layers=1, depth=1)
data_loader = CustomDataLoader(BI_dataset, df, 1)
model, _, _, _, _ = model.train(data_loader, n_epochs=200, lr=0.01, beta_x=1e-6, beta_y=1e-6,
                                               verbose=True)

torch.save(model.state_dict(), 'rbf_model_weights.pth')
#model = model.to(torch.device("cpu"), non_blocking=True)

 

