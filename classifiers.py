import numpy as np
import scipy as sc
import pandas as pd
from sklearn.decomposition import PCA
import cornsoydata as csd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def before(date1, date2):
    #print(date1, date2)
    y1 = int(date1[0:4])
    y2 = int(date2[0:4])
    m1 = int(date1[5:7])
    m2 = int(date2[5:7])
    d1 = int(date1[8:10])
    d2 = int(date2[8:10])
    #print(y1, m1, d1)
    if y1<y2:
        return True
    elif y1>y2:
        return False
    if m1<m2:
        return True
    elif m1>m2:
        return False
    if d1<d2:
        return True
    else:
        return False

def make_labels(full, full_dates, mkt, mkt_dates):
    days_ahead = 5
    lendates = len(mkt_dates)
    labels = np.zeros(full_dates.shape)
    mkt_index = 0
    labeled = 0
    unavailable = 0
    for i in range(len(full_dates)-1):
        today = full_dates[i]
        today_i = 0
        exp_i = 0
        #print(date)
        #try dates
        
        while before(mkt_dates[today_i], today):
            today_i += 1
            if today_i >= lendates:
                break
            
        if today_i+days_ahead < lendates:
            #today = mkt_dates[mkt_index]
            #tomorrow = mkt_dates[mkt_index+1]
            #print("yay")
            labeled += 1
            labels[i] = mkt[today_i+days_ahead][3] / mkt[today_i][3]
        else:
            unavailable += 1
    #print(labels)
    labels = [i>1. for i in labels]
    print("unlabeled = ", unavailable)
    print("labeled = ", labeled)
    return pd.DataFrame(np.array(labels))

def make_targets(full, full_dates, mkt, mkt_dates):
    days_ahead = 5
    lendates = len(mkt_dates)
    labels = np.zeros(full_dates.shape)
    mkt_index = 0
    labeled = 0
    unavailable = 0
    for i in range(len(full_dates)-1):
        today = full_dates[i]
        today_i = 0
        exp_i = 0
        #print(date)
        #try dates
        
        while before(mkt_dates[today_i], today):
            today_i += 1
            if today_i >= lendates:
                break
            
        if today_i+days_ahead < lendates:
            #today = mkt_dates[mkt_index]
            #tomorrow = mkt_dates[mkt_index+1]
            #print("yay")
            labeled += 1
            labels[i] = float(mkt[today_i+days_ahead][3])# - mkt[today_i][3]
        else:
            unavailable += 1
    #print(labels)
    print("unlabeled = ", unavailable)
    print("labeled = ", labeled)
    return pd.DataFrame(np.array(labels))

def make_feat_labels(mkt, days_back=5):
    features = []
    labels = []
    lenmkt = len(mkt)
    for i in range(days_back, lenmkt-1):
        feature = mkt[i-days_back:i+1].flatten(order='C')
        label = mkt[i][3]<mkt[i+1][3]
        features.append(feature)
        labels.append(label)
    return np.array(features), np.array(labels)


def make_feat_targets(mkt, days_back=5):
    features = []
    labels = []
    lenmkt = len(mkt)
    for i in range(days_back, lenmkt-1):
        feature = mkt[i-days_back:i+1]#.flatten(order='C')
        label = float(mkt[i+1][3])#/mkt[i][3]
        features.append(feature)
        labels.append(label)
    return np.array(features), np.array(labels)



def prep_data(labels='binary', days_back=5):
    (cornmkt, soymkt, fullcorn, fullsoy, corn_mkt_dates, soy_mkt_dates, 
        fullcorn_dates, fullsoy_dates) = csd.openfiles()
    #regularize all features to a 0-1 scale
    cornmkt = csd.regularize(np.array(cornmkt))
    soymkt = csd.regularize(np.array(soymkt))
    fullcorn = csd.regularize(np.array(fullcorn))
    fullsoy = csd.regularize(np.array(fullsoy))

    if labels=='binary':
        cornlabels = make_labels(fullcorn, fullcorn_dates, cornmkt, corn_mkt_dates)
        soylabels = make_labels(fullsoy, fullsoy_dates, soymkt, soy_mkt_dates)
        cornfeat, cornlab = make_feat_labels(cornmkt, days_back)
        soyfeat, soylab = make_feat_labels(soymkt, days_back)
    elif labels=='real':
        cornlabels = make_targets(fullcorn, fullcorn_dates, cornmkt, corn_mkt_dates)
        soylabels = make_targets(fullsoy, fullsoy_dates, soymkt, soy_mkt_dates)
        cornfeat, cornlab = make_feat_targets(cornmkt, days_back)
        soyfeat, soylab = make_feat_targets(soymkt, days_back)
    #use mkt to get a target for the following week
    return (fullcorn, fullsoy, cornlabels, soylabels, 
        cornfeat, cornlab, soyfeat, soylab)
    



def pca3(data, labels, name):

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = principalComponents, 
        columns = ['principal component 1', 'principal component 2',
        'principal component 3'])

    fig = plt.figure(figsize = (8,8))

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)

    
    ax.set_title('3 Component PCA', fontsize = 20)

    finalDf = pd.concat([principalDf, labels], axis = 1)
    #ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'])
    targets = [True, False]
    colors = ['r', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf[0] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], 
            finalDf.loc[indicesToKeep, 'principal component 2'], 
            finalDf.loc[indicesToKeep, 'principal component 3'],
            c = color, s = 50)
    ax.legend(["Price Increase", "Price Decrease"])
    ax.grid()
    plt.savefig(name)
    #plt.show()
    plt.close()
    print("done 3")


def pca2(data, labels, name):

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = principalComponents, 
        columns = ['principal component 1', 'principal component 2'])

    fig = plt.figure(figsize = (8,8))

    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)

    
    ax.set_title('2 Component PCA', fontsize = 20)

    finalDf = pd.concat([principalDf, labels], axis = 1)
    #ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'])
    targets = [True, False]
    colors = ['r', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf[0] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], 
            finalDf.loc[indicesToKeep, 'principal component 2'], 
            c = color, s = 50)
    ax.legend(["Price Increase", "Price Decrease"])
    ax.grid()
    plt.savefig(name)
    #plt.show()
    plt.close()
    print("done 2")



# (fullcorn, fullsoy, cornlabels, soylabels, 
#         cornfeat, cornlab, soyfeat, soylab) = prep_data()

# components = 3

# if components == 2:
#     pca2(cornfeat, cornlab, "cornmkt5.png")
#     pca2(soyfeat, soylab, "soymkt5.png")
#     #if float('inf') not in fullcorn:
#     pca2(fullcorn, cornlabels, "fullcorn.png")
#     #if float('inf') not in fullsoy:
#     pca2(fullsoy, soylabels, "fullsoy.png")
# elif components == 3:
#     pca3(cornfeat, cornlab, "cornmkt5,3d.png")
#     pca3(soyfeat, soylab, "soymkt5,3d.png")
#     #if float('inf') not in fullcorn:
#     pca3(fullcorn, cornlabels, "fullcorn3d.png")
#     #if float('inf') not in fullsoy:
#     pca3(fullsoy, soylabels, "fullsoy3d.png")











	