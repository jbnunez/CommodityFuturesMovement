import numpy as np
import scipy as sc
import pandas as pd
from sklearn.decomposition import PCA
import cornsoydata as csd
import classifiers as clf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier




def pca_rnn(comp_list, k_list, data, labels):
    clen = len(comp_list)
    klen = len(k_list)
    acc_list = np.zeros((clen, klen))
    for j in range(clen):
        pca = PCA(n_components=comp_list[j])
        principalComponents = pca.fit_transform(data)
        #print(type(principalComponents))
        #principalDf = pd.DataFrame(data = principalComponents, 
        #    columns = range(cnum))
        for i in range(klen):
            nbrs = KNeighborsClassifier(n_neighbors=k_list[i])
            nbrs.fit(principalComponents, labels) 
            acc_list[j][i] = nbrs.score(principalComponents, labels)

    return acc_list

def plot_pca_rnn(acc_list, k_list, comp_list, name):
    #acc_df = pd.DataFrame(data=acc_list,
    #    indices=[str(c)+" Components" for c in comp_list],
    #    columns=['K = '+str(k) for k in k_list])
    fig = plt.figure(figsize = (8,8))

    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('K (Number of Means)', fontsize = 15)
    ax.set_ylabel('KNN Classifier Accuracy', fontsize = 15)

    for row in range(len(comp_list)):
        ax.set_title('K-Means Accuracy', fontsize = 20)
        ax.plot(k_list, acc_list[row], label="Components = "+str(comp_list[row]))
        print(comp_list[row], acc_list[row])
    ax.legend()
    plt.savefig(name)
    #plt.show()
    plt.close()


(fullcorn, fullsoy, cornlabels, soylabels, 
        cornfeat, cornlab, soyfeat, soylab) = clf.prep_data()

#cornlabels, soylabels = cornlabels.T, soylabels.T
        
k_list = np.array(range(2,9))
#print(k_list)
comp_lists = []
for i in range(5):
    c = np.array(range(2+6*i, 2+6*(i+1)))
    comp_lists.append(c)
#print(comp_list)
for comp_list in comp_lists:
    end = comp_list[-1]
    start = comp_list[0]
    # corn_acc_list = pca_rnn(comp_list, k_list, fullcorn, cornlabels)
    # print("corn done")
    # soy_acc_list = pca_rnn(comp_list, k_list, fullsoy, soylabels)
    # print("soy done")

    # plot_pca_rnn(corn_acc_list, k_list, comp_list, 
    #     'cornacc'+str(start)+','+str(end)+',w.png')
    # plot_pca_rnn(soy_acc_list, k_list, comp_list, 
    #     'soyacc'+str(start)+','+str(end)+',w.png')

    # corn_acc_list = pca_rnn(comp_list, k_list, cornfeat, cornlab)
    # print("corn done")
    # soy_acc_list = pca_rnn(comp_list, k_list, soyfeat, soylab)
    # print("soy done")

    # plot_pca_rnn(corn_acc_list, k_list, comp_list, 
    #     'cornacc'+str(start)+','+str(end)+',m.png')
    # plot_pca_rnn(soy_acc_list, k_list, comp_list, 
    #     'soyacc'+str(start)+','+str(end)+',m.png')








