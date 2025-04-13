import matplotlib.pyplot  as plt
from matplotlib import colors
import numpy as np
import torch
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd
from matplotlib.ticker import ScalarFormatter
formatter=ScalarFormatter(useOffset=False, useMathText=True)
def plot3d_colormap(x,y ,z, c, fig, title,label, xlabel, ylabel, zlabel,tag=None, cmap = 'rainbow' , vmax = None, vmin=None,cat=False):
    fig.suptitle(title, size=15)
    l= [x, y, z, c]
    for i in range(len(l)):

        if torch.is_tensor(l[i]):
            if i==3:
                l[i].flatten()
            if l[i].is_cuda:
                l[i] = l[i].cpu().detach().numpy()
            else:
                l[i] = l[i].detach().numpy()


    # fig = pl
    # cmap = plt.cm.rainbow
    # norm = colors.BoundaryNorm(np.array([100,101, 102, 103,104,105,106,107]), cmap.N)
    if tag==None:
        ax = fig.add_subplot(tag, projection='3d')
    else:
        ax = fig.add_subplot(tag, projection='3d')
    # scat = ax.scatter(s2[:,0].cpu().detach().numpy(),s2[:,1].cpu().detach().numpy(),-1*s2[:,2].cpu().detach().numpy(),c=z_pred2.cpu().flatten().detach().numpy(),vmax=z_int.max(), vmin=z_int.min(),alpha=0.5,cmap='nipy_spectral')#, norm=norm)

    scat = ax.scatter(l[0], l[1],
                       l[2], c=l[3],
                      cmap=cmap, vmax=vmax,
                      vmin=vmin)  # ,vmax=z_int.max(), vmin=z_int.min(),alpha=0.5,cmap='nipy_spectral')
    #  scat = ax.scatter(s2[:, 0].cpu().detach().numpy(), s2[:, 1].cpu().detach().numpy(),
    # -1 * s2[:, 2].cpu().detach().numpy(), c=z_pred2.cpu().flatten().detach().numpy(),
    # cmap='nipy_spectral')  # ,vmax=z_int.max(), vmin=z_int.min(),alpha=0.5,cmap='nipy_spectral')#, norm=norm)
    cbar = fig.colorbar(scat, shrink=0.5, aspect=15, pad = 0.2, label=label)
    cbar.set_label(label=label, size=10)
    if cat == True:
        cbar.set_ticks(ticks=[i for i in range(len(np.unique(c)))], labels=list(np.unique(c)))

    ax.tick_params(axis='both', labelsize=12)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(xlabel, size=8, weight='bold')
    ax.set_ylabel(ylabel, size=8, weight='bold')
    ax.set_zlabel(zlabel, size=8, weight='bold')
    return fig, cbar
def scatter_with_dist(plt, x, y,b, title, xlabel, ylabel, color = None):
    plt.set_title(title, fontsize=14)
    plt.set_xlabel(xlabel, fontsize=14)
    plt.set_ylabel(ylabel, fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    l = [x, y, b]
    for i in range(len(l)):

        if torch.is_tensor(l[i]):
            if l[i].is_cuda:
                l[i]=l[i].cpu().detach().numpy().flatten()
            else:
               l[i]=   l[i].detach().numpy().flatten()

    xy = np.vstack([l[0], l[1]])
    z = gaussian_kde(xy)(xy)
    plt.scatter(l[0], l[1], c=z, cmap='rainbow')
    plt.scatter(l[0], l[2], color=color)
    return plt


def box_plot(title1:str, var1, title2:str, var2, lim, ax=None, title = None, palette=None):
    df = pd.DataFrame({title1:var1, title2: var2})
    fig = plt.figure()
    if ax==None:
        ax = fig.add_subplot()
    else:
        ax=ax
    ax=sns.boxplot(x=title1, y=title2, data=df, ax=ax, palette=palette)
    min = df[title2].quantile(.5)
    max= df[title2].quantile(.5)
    for cluster in np.unique(df[title1]):
        q1 = df[df[title1]==cluster][title2].quantile(.25)
        q3 = df[df[title1] == cluster][title2].quantile(.75)
        ax.axhline(y=q1, color = 'black', linestyle='--')
        ax.axhline(y=q3, color='black', linestyle='--')

        iqr = q3-q1
        oulierf =  q1-3*iqr
        oulieru = q3 + 3*iqr
        if min >  oulierf:
            min =  oulierf
        if max <  oulieru :
             max =  oulieru
        ax.set_ylim(min, max)



    if lim is not None :
        ax.set_ylim([lim[0],lim[1]])
    ax.set_title(title)
    return fig, ax

def box_plot_stat(var):

    Q3 = np.quantile(var, .75)
    Q1 = np.quantile(var, .25)
    IQR = Q3 - Q1
    lf = Q1 - 1.5 * IQR
    uf = Q3 + 1.5 * IQR
    return lf, uf, Q1, Q3








