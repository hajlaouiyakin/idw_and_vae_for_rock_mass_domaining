import os
import tkinter
from tkinter import filedialog
import copy
import math

import pandas as pd
import idw_cluster
from models_metrics import misfit_percentage, silouhette_score, calinski_harabasz_score, davies_bouldin_score, \
    average_moran_local, average_std, accuracy_score_clusters

import matplotlib.pyplot as plt
from utils.plots import plot3d_colormap, scatter_with_dist, box_plot, box_plot_stat
from torch import torch
import matplotlib.colors as mcolors
import pickle
import gc
from utils.load_data import CustomDataLoader, reduce_bi_data
from utils.remove_far_holes import remove_holes
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from sklearn.cluster import Birch, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from utils.load_data import ScaledBIDataSet

src = 'drilling_data.csv'
df = df = pd.read_csv(src) 
df = reduce_bi_data(df, resol=0.3)
predictors = ["CoordX", "CoordY", "Depth"]
targets = ["ROP", "WOB", "RPM", "TRQ"]

BI_dataset = ScaledBIDataSet(df=df, predictors=predictors, targets=targets, scale_predictors=True,
                                 scale_targets=True, mode='train', transform=None)

data_loader = CustomDataLoader(BI_dataset, df, 1)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Sample every 5th point for better visualization
plot_df = df.iloc[::5]

# Scatter plot with color mapping
sc = ax.scatter(
    BI_dataset.x[:, 0],
    BI_dataset.x[:, 1],
    BI_dataset.x[:, 2],
    c=df['SED'],
    cmap='viridis',
    alpha=0.6,
    s=5
)

# Labels and title
ax.set_xlabel('Coord X (m)')
ax.set_ylabel('Coord Y (m)')
ax.set_zlabel('Coord Z (m)')
ax.set_title('3D Distribution of SED (J/m³)')

# Color bar
cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('SED (J/m³)')

plt.tight_layout()
plt.savefig('3d_sed_distribution.png', dpi=300)
plt.show()
 # restore model trained with collar
# FC
#model = idw_cluster.DeepIdwAutoEncoderBatch(3, 4, hidden_dim=3, num_layers=1, depth=1)
 # Conv
#model = idw_cluster.DeepIdwAutoEncoderConv(3, 4, hidden_dim=3, num_layers=1, depth=1, device=device)
# RBF
model = idw_cluster.DeepIdwAutoEncoderRbf(3, 4, hidden_dim=3, num_layers=1, depth=1)
#PATH1 = 'fc_model_weights.pth' 
model.load_state_dict(torch.load('rbf_model_weights.pth', weights_only=True))
print('model loaded')
# getting pseudo BI

   # df2.
    # df2['CoordZ']=-1*df2['CoordZ']
#BI_dataset2 = ScaledBIDataSet(df=df, predictors=predictors, targets=targets, scale_predictors=True,
                                #  scale_targets=True, mode='train', transform=None)
#BI_dataseto2 = ScaledBIDataSet(df=df, predictors=predictors, targets=targets, scale_predictors=False,
                #                   scale_targets=False, mode='train', transform=None)
#data_loader2 = CustomDataLoader(BI_dataseto2, df, 1)
# load model trained on post collar
# FC
#model = idw_cluster.DeepIdwAutoEncoderBatch(3, 4, hidden_dim=3, num_layers=1, depth=1)
# Conv
#model2 = idw_cluster.DeepIdwAutoEncoderConv(3, 4, hidden_dim=3, num_layers=1, depth=1, device=device)
# RBF
model = idw_cluster.DeepIdwAutoEncoderRbf(3, 4, hidden_dim=3, num_layers=1, depth=1)


with torch.no_grad():
     
           
            
            y_hat, z_int, covar, z, output_x, z_x, mean_x, var_x, log_var_x, output_y, z_y, mean_y, var_y, log_var_y = model.forward(
                BI_dataset.x, BI_dataset.y)
            
y_h = y_hat
z_i = z_int
yt =BI_dataset.y
y_bi = BI_dataset.scaler2.inverse_transform(yt)
lat = BI_dataset.scaler2.inverse_transform(y_h.cpu().detach().numpy())

pseudoBi = z_i
#bit_area = 0.311 * 0.311 * math.pi / 4
#recSED = np.abs((2 * math.pi * lat[:, 3] * lat[:, 2] / (lat[:, 0] * bit_area) + lat[:, 1] / bit_area) / (1000))
#recBI = 100*recSED/205

#df['rec-SED']=recSED
#s2 = torch.cat(
  #      (BI_dataset.x + 0.2 * torch.randn_like(BI_dataset.x), BI_dataset.x + 0.2 * torch.randn_like(BI_dataset.x)),
 #       dim=0)
#torch.cuda.empty_cache()
#with torch.no_grad():
 #       z_pred2 = model.predict(s2.to(device), BI_dataset.x.to(device), z_y.flatten())
#coord2 = torch.cat((s2, BI_dataset.x), dim=0)
#pseudoBi2 = torch.cat((z_pred2, z_i), dim=0)

    # results with a given numnber of clusters
   # num_clusters=3
X = pseudoBi.reshape(-1, 1).cpu()
birch = Birch(n_clusters=None, threshold=.001, branching_factor=50)
birch.fit(X)
centroids = birch.subcluster_centers_
num_clusters = 3
    # num_clusters = len(centroids)
print(num_clusters)
agg = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
agg.fit(centroids)
cluster_ids_x2 = agg.labels_[birch.predict(z_i.reshape(-1, 1).cpu())]
    # cluster_ids_xp2, cluster_centersp2 = kmeans(X, num_clusters=num_clusters, distance='euclidean', device=device)
    # cluster_ids_x2 = kmeans_predict(z_i.reshape(-1,1).cpu() , cluster_centersp2, 'euclidean', device=device)
alpha_list = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
bi_clustering_list = []

for alpha in alpha_list:
        print(torch.tensor(df['SED']).reshape(-1, 1).size())
        print(BI_dataset.x.size())
        X = torch.cat((alpha * BI_dataset.x, (1 - alpha) * torch.tensor(df['SED']).reshape(-1, 1)), dim=1)

        birch.fit(X)
        centroids = birch.subcluster_centers_
        num_clusters = len(centroids)
        agg = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
        agg.fit(centroids)
        cluster_ids_x_bi2 = agg.labels_[birch.predict(X)]
        # cluster_ids_x_bi2, cluster_centers_bi2 = kmeans_pytorch.kmeans(
        #   X=torch.cat((alpha*BI_dataset2.x, (1-alpha)*torch.tensor(df2['BI']).reshape(-1,1)),dim=1), num_clusters=num_clusters,cluster_centers= cluster_centersp2, distance='euclidean', device=device)

        bi_clustering_list.append((cluster_ids_x_bi2, centroids))


    # saving some figures and boxplots

    # Box plots
    # Pseudo BI clusters with mec param and Pseudo_mec
    # set file and path
    # #generate_box_plots()








def plot3dClusters(clusters, str):
        fig = plt.figure(figsize=(8, 6))
        fig_name = '3d_cluster.png'
        cmap = plt.cm.get_cmap('rainbow', len(np.unique(clusters)))
        
        title = '3d clusters' 
        label = 'clusters' + str
       
      

        plot3d_colormap(x= BI_dataset.x[:, 0], y= BI_dataset.x[:, 1], z= BI_dataset.x[:, 2], c=clusters,
                        fig=fig,
                        title=title, label=label, xlabel='CoordX', ylabel='CoordY',
                        zlabel='Depth', cmap=cmap, cat=True)
        fig.savefig(fig_name)
        pickle.dump(fig, open(fig_name + '.fig.pickle', 'wb'))


plot3dClusters(cluster_ids_x2, 'PseudoBI')
    # plot3dClusters(bi_clustering_list[0][0], 'BI')






def generate3Dplots():
        # Pseudo BI mapping
        fig_name ='3d_map_PseudoBI.png'
        fig = plt.figure(figsize=(8, 6))

        c = pseudoBi.detach().cpu().numpy().flatten()
        coord2 = BI_dataset.scaler1.inverse_transform(coord)
        plot3d_colormap(x=coord2[:, 0], y=coord2[:, 1], z=coord2[:, 2], c=pseudoBi, fig=fig,
                        title='3d mapping of PseudoBI', label='PseudoBI', xlabel='CoordX', ylabel='CoordY',
                        zlabel='Depth', tag=None, cmap='rainbow', vmax=np.quantile(c, .75), vmin=np.quantile(c, .25))
        # Cluster BI vs Cluster PseudoBI
        fig.savefig(fig_name)
        fig = plt.figure(figsize=(8, 6))
        fig_name =  '/3d_clusters_PseudoBI_vs_BI.png'
        cmap = plt.cm.get_cmap('rainbow', len(np.unique(cluster_ids_x2)))

        plot3d_colormap(x=BI_dataset.x[:, 0], y=BI_dataset.x[:, 1], z=BI_dataset.x[:, 2], c=cluster_ids_x2,
                        fig=fig,
                        title='3d clusters PseudoBI vs BI', label='clusters PseudoBI', xlabel='CoordX', ylabel='CoordY',
                        zlabel='Depth', tag=121, cmap=cmap, cat=True)

        plot3d_colormap(x=BI_dataset.x[:, 0], y=BI_dataset.x[:, 1], z=BI_dataset.x[:, 2],
                        c=bi_clustering_list[0][0],
                        fig=fig,
                        title='3d clusters PseudoBI vs BI', label='clusters BI', xlabel='CoordX', ylabel='CoordY',
                        zlabel='Depth',
                        tag=122, cmap=cmap, cat=True)
        # plt.figure(figsize=(8, 8))
        fig.savefig(fig_name)
        fig = plt.figure(figsize=(8, 6))

        fig_name = 'PseudoBI_vs_BI.png'
        plot3d_colormap(x=BI_dataset.x[:, 0], y=BI_dataset.x[:, 1], z=BI_dataset.x[:, 2], c=z_i,
                        fig=fig,
                        title=' PseudoBI vs BI', label='PseudoBI', xlabel='CoordX', ylabel='CoordY',
                        zlabel='Depth', tag=121, cmap='rainbow')

        plot3d_colormap(x=BI_dataset.x[:, 0], y=BI_dataset.x[:, 1], z=BI_dataset.x[:, 2],
                        c=df['SED'],
                        fig=fig,
                        title='PseudoBI vs BI', label='SED', xlabel='CoordX', ylabel='CoordY',
                        zlabel='Depth',
                        tag=122, cmap='rainbow')
        fig.savefig(fig_name)


    # generate3Dplots()
def generate_scatter_plots():
        k = 0
        for str in targets:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            scatter_name = 'scatter.png'
            plt = scatter_with_dist(plt, z_i, df[str], lat[:, k], xlabel='PseudoBI', ylabel=str,
                                    title=str + ' vs ' + 'PseudoBI',
                                    color='orange')

            legend = plt.legend([str, 'Pseudo' + str])
            for text in legend.texts:
                if text.get_text() == str:
                    text.set_visible(False)
            legend.legendHandles[0].set_alpha(alpha=0)

            fig.savefig(scatter_name)
            k = k + 1


def generate_scatter_plot_matrix():
        l = [(0, 0), (0, 1), (1, 0), (1, 1)]
        k = 0
        scatter_name = 'scatter_pseudoBI_.png'
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        for str in targets:

            print(l[k][0], l[k][1])
            axes[l[k][0], l[k][1]] = scatter_with_dist(axes[l[k][0], l[k][1]], z_i, df[str], lat[:, k],
                                                       xlabel='PseudoBI', ylabel=str, title=str + ' vs ' + 'PseudoBI',
                                                       color='orange')
            legend = axes[l[k][0], l[k][1]].legend([str, 'Pseudo' + str])
            for text in legend.texts:
                if text.get_text() == str:
                    text.set_visible(False)
            #legend.legendHandles[0].set_alpha(alpha=0)
            k = k + 1
        fig.savefig(scatter_name)


generate_scatter_plot_matrix()