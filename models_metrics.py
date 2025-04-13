from __future__ import annotations
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd


'''
In metrics.py we compute the different metrics used in the  project

'''

def correlation(x, y):
    x=x.to(device)
    y=y.to(device)
    a = torch.zeros(x.shape[-1]).to(device)
    b = torch.zeros(x.shape[-1]).to(device)
    corr = torch.zeros(x.shape[-1]).to(device)
    for i in range(x.shape[-1]):
        s = x[:, i].flatten()
        r = y.flatten()
        mean_x = torch.mean(s).to(device)
        mean_y = torch.mean(r).to(device)
        a[i] = torch.sum((s - mean_x) * (r - mean_y))
        b[i] = torch.sqrt(torch.sum((s - mean_x) ** 2) * torch.sum((r - mean_y) ** 2))
    corr[i] = (a[i] / b[i]).to(device)
    return corr


def corrcoef(x):
    # print(x.shape)
    x= x.to(device)
    mean_x = torch.mean(x, 1).reshape(-1, 1).to(device)
    # print(mean_x)
    xm = x.sub(mean_x.expand_as(x)).to(device)
    c = xm.mm(xm.t()).to(device)
    c = c / (x.size(1) - 1)
    d = torch.diag(c).to(device)
    stddev = torch.pow(d, 0.5).to(device)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0).to(device)
    return c


def multiple_corr(x, y):
    x=x.to(device)
    y=y.to(device)
    corr_xx = corrcoef(x.transpose(-1, -2)).to(device)
    # print(corr_xx)

    corr_xx_inv = torch.inverse(corr_xx)
    # print(corr_xx_inv)

    corr = correlation(x, y).reshape(-1, 1).to(device)
    print(corr)
    multi_corr = torch.matmul(torch.matmul(corr.transpose(-1, -2), corr_xx_inv), corr)
    return multi_corr, corr_xx


def r_squared(predicted, target, y_train):
    mean_y_train = torch.mean(y_train)
    # print(mean_y_train)
    r2 = 1 - torch.sum((predicted - target) ** 2) / torch.sum((mean_y_train - target) ** 2)
    return r2


def relative_r2(predicted1, predicted2, target ):
    #mean_y_train = torch.mean(y_train)
    #r2 = 1 - torch.sum(torch.abs(predicted - target) / torch.abs(target)) / torch.sum(
     #   torch.abs(mean_y_train - target) / torch.abs(target))
    r2 = 1 - sym_mape(predicted1, target)/sym_mape(predicted2, target)
    return r2


def mse(predicted, target):
    return torch.sum((predicted - target) ** 2)


def nmse(predicted, target):
    return torch.mean((predicted - target) ** 2)


def mape(predicted, target):  # Mean Absolute Percentage
    MAPE = torch.mean(torch.abs((predicted - target) / target + 0.001))
    return MAPE


def sym_mape(predicted, target):
    MAPE = torch.mean(torch.abs((predicted - target) / (target + predicted) / 2))
    return MAPE


def rmse(predicted, target):
    return torch.sqrt(mse(predicted, target))


def rnmse(predicted, target):
    return torch.sqrt(nmse(predicted, target))
def mae (predicted, target):
   return torch.sum(torch.abs(predicted - target))
def nmae(predicted, target):
    return torch.mean(torch.abs(predicted-target))

def compute_misfit_percentage(y_true, y_pred, class_label):
    class_indices = (y_true == class_label)
    print(y_true)
    print(y_pred)
    misfit_percentage = (1.0 - np.mean(y_true[class_indices] == y_pred[class_indices]))
    return misfit_percentage
def misfit_percentage(x, clusters):
    #svm_classifier = SVC(kernel='linear')
    lda = LinearDiscriminantAnalysis()
    #lr= LogisticRegression()
    #svm_classifier.fit(x, clusters)
    lda.fit(x, clusters)
    #y_pred = svm_classifier.predict(x)
    y_pred = lda.predict(x)
    misfit_percentage_per_class = {}
    for c in np.unique(clusters):
        #print(c)
        misfit_percentage = compute_misfit_percentage(clusters, y_pred, c)
        print('misfit', misfit_percentage)
        misfit_percentage_per_class[c] = misfit_percentage
    average_misfit_percentage = np.mean(list(misfit_percentage_per_class.values()))
    return  average_misfit_percentage,  misfit_percentage_per_class
def accuracy_score_clusters(x, clusters):
    # svm_classifier = SVC(kernel='linear')
    lda = LinearDiscriminantAnalysis()
    # lr= LogisticRegression()
    # svm_classifier.fit(x, clusters)
    lda.fit(x, clusters)
    # y_pred = svm_classifier.predict(x)
    y_pred = lda.predict(x)
    accuracy= accuracy_score(clusters,y_pred)
    return accuracy, y_pred
def silouhette_score(x, clusters):
    return metrics.silhouette_score(x, clusters)
def calinski_harabasz_score(x, clusters):
    return metrics.calinski_harabasz_score(x, clusters)
def davies_bouldin_score(x, clusters):
    return metrics.davies_bouldin_score(x, clusters)

def average_moran_local(model, x,target):
    torch.cuda.empty_cache()
    t= torch.tensor(target,dtype=float).to(device)-torch.mean(torch.tensor(target,dtype=float)).to(device)
    m2 = (torch.std(torch.tensor(target,dtype=float))**2).to(device)
    covar = model.kernel(x.to(device))
    print(m2)
    print(t)
    I=  torch.matmul(torch.nn.functional.normalize(covar.evaluate(),p=1, dim=1).float(), t.float() )
    W=torch.sum(torch.nn.functional.normalize(covar.evaluate(),p=1, dim=1).float())
    averge_local_i = (1/W)*(1/m2)*torch.matmul(I,t.float())
    if m2==0:
        averge_local_i = torch.tensor(1.)
    print(averge_local_i.item())
    return averge_local_i.item()


def average_std(df, domains,str):
    c = pd.factorize(domains, sort=True)
    std_list = []
    for i in np.unique(c[0]):
        #print(domains)
        #print(len(df[domains==i]['rec-BI']))
        print(df[domains==i][str])
        print(np.nanvar(df[domains==i][str]))
        if len(df[domains == i][str]) != 0:
            #std_list.append(len(df[domains==i]['rec-BI'])*np.var(df[domains==i]['rec-BI'],ddof=0))
            std_list.append(np.var(df[domains == i][str], ddof=0))


    return np.sqrt(np.mean(std_list))





