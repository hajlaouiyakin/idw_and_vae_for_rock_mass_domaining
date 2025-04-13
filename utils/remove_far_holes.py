from sklearn.cluster import Birch, AgglomerativeClustering, KMeans
from utils.load_data import BIDataSet, ScaledBIDataSet, Bi2DDataFrame, MyDataSet
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
def remove_holes(df):
    #print(df['HoleNames'].values)
    hole_names = list(dict.fromkeys(df['HoleNames'].values))

    hole_names1 = [int(elm) for elm in hole_names]
    dataset = Bi2DDataFrame(df, ['SED'], ["CoordX", "CoordY"], hole_names)
    mydata = BIDataSet(dataset.df2, predictors=['CoordX', 'CoordY'], targets=['Average_SED'], transform=None)
    # birch clustering
    #num_clusters = 2
    X = mydata.x
    #birch = Birch(n_clusters=None, threshold=.5, branching_factor=5)
    #birch.fit(X)
    #centroids = birch.subcluster_centers_
    #agg = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    #agg.fit(centroids)
    #cluster_ids = agg.labels_[birch.predict(X)]
    #kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    #cluster_ids = kmeans.labels_
   # iso_forest = IsolationForest(contamination=0.015,bootstrap=False)  # Adjust contamination based on your dataset
    clf = LocalOutlierFactor(n_neighbors=2, contamination=0.005)
    outliers = clf.fit_predict(X)
    # Fit the model to your data
    #iso_forest.fit(X)
    #outliers =  iso_forest.predict(X)
    outlier_indices = np.where(outliers== -1)[0]
    #print(outlier_indices )
    #idx1 = np.where(cluster_ids == 0)[0].tolist()
    #idx2 = np.where(cluster_ids == 1)[0].tolist()

    # compute the min distance between clusters
    #print(len(idx1))
    #print(len(idx2))
    #dist = distance.cdist(X[idx1,:], X[idx2,:])
    #min_dist  = np.min(dist)
    # chose the clusters with min hole numbers
    #min_cluster = idx1 > idx2
    #ids  = np.where(cluster_ids == min_cluster)[0].tolist()
    #compute the max distance with that cluster
    #dist2 = distance.cdist(X[idx1,:], X[idx1 ,:])
    #print(np.max(dist2).item())

    #print(np.max(distance.cdist(X[idx2,:], X[idx2,:])).item())

    #max_hole_dist =max(np.max(dist2), np.max(distance.cdist(X[idx2,:], X[idx2,:])))
    #print(max_hole_dist)
    #print(min_dist)
    #check if you should remove the cluster:
    #if max_hole_dist < min_dist and len(ids) < 4:

    #else:
        #hole_name_drop=[]
    ids= [int(elm) for elm in  outlier_indices]
    hole_name_drop = [hole_names1[i] for i in ids]
    idx_drop = df[df['HoleNames'].isin(hole_name_drop)].index.tolist()
    df2 = df.drop(index= idx_drop )
    df2.reset_index(inplace=True)
    return df2

"""""
    # main
src= "C:/Users/yakin/Documents/PhD-Yakin/python_project/Validation/3D_domaining_validation_dataset_(feb2024)/grp4_BW_620_634_648/data/pattern_BW-634-037/ pattern_BW-634-037.csv"
df = pd.read_csv(src, sep =',')
df2 = remove_holes(df)
hole_names = list(dict.fromkeys(df['HoleNames'].values))
hole_names1 = [int(elm) for elm in hole_names]
#print(len(hole_names1))
dataset = Bi2DDataFrame(df2, ['SED'], ["CoordX", "CoordY"], hole_names)
mydata = BIDataSet(dataset.df2, predictors=['CoordX', 'CoordY'], targets=['Average_SED'], transform=None)
#%matplotlib qt
fig = plt.figure()

#fig = pl
ax = fig.add_subplot()
scat = ax.scatter(mydata.x[:,0],mydata.x[:,1],c=mydata.y,alpha=0.5,cmap='nipy_spectral',s=100)
fig.colorbar(scat, shrink=1, label='Average SED')
#for i, txt in enumerate(hole_names1):
 #   ax.annotate(txt, (mydata.x[i,0], mydata.x[i,1]))

ax.set_xlabel('log(coordX)', fontweight='bold')
ax.set_ylabel('log(coordY)', fontweight='bold')

#plt.show()

# birch clustering


    # get the 2d df
"""
