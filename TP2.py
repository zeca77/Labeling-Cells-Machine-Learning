# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:31:54 2021

@author: bruno
"""

import numpy as np
import matplotlib.pyplot as plt
from os import path
from tp2_aux import images_as_matrix
from tp2_aux import report_clusters
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import f_classif
from sklearn.manifold import Isomap
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation

def plot_dist(features):
    kn = KNeighborsClassifier()
    kn.fit(globals()['X_new'+features], np.ones(globals()['X_new'+features].shape[0]))
    globals()['dist'+features] = kn.kneighbors(globals()['X_new'+features])[0][:,-1]
    globals()['dist'+features] = np.sort(globals()['dist'+features])
    globals()['dist'+features] = globals()['dist'+features][::-1]
    #points = np.arange(0,globals()['X_new'+features].shape[0])
    #plt.plot(points, globals()['dist'+features], "o"+colors[int(features)-2])
    
def test_db_performance(features):
    globals()['ssdb'+features] = sm.silhouette_score(globals()['X_new'+features], globals()['labelsdb'+features])
    globals()['randscoredb'+features] = sm.rand_score(y_red, globals()['labels_testdb'+features])
    globals()['supdb'+features] = sm.precision_recall_fscore_support(y_red, globals()['labels_testdb'+features], average='macro', zero_division=0)
    
def test_kmeans_performance(features):
    globals()['sskm'+features] = sm.silhouette_score(globals()['X_new'+features], globals()['labelskm'+features])
    globals()['randscorekm'+features] = sm.rand_score(y_red, globals()['labels_testkm'+features])
    globals()['supkm'+features] = sm.precision_recall_fscore_support(y_red, globals()['labels_testkm'+features], average='macro', zero_division=0)
    
def print_results(features, alg, best_parameter):
    print('features =',features,'------------------------------')
    print('best_parameter =',best_parameter)
    print('silhouette score:',globals()['ss'+alg+features])
    print('rand score:', globals()['randscore'+alg+features])
    print('recall:', globals()['sup'+alg+features][1])
    print('precision:',globals()['sup'+alg+features][0])
    print('fscore:',globals()['sup'+alg+features][2])
    print('adjusted rand score:', globals()['adjrandscore'+alg+features])

def k_means():
    plt.clf()
    clusters_idx=np.arange(2,9)
    plt.xlabel('Clusters')
    plt.ylabel('Score')
    for feats in range(2,5):
        features = str(feats)
        globals()['best_clusters'+features] = 0
        globals()['adjrandscorekm'+features] = 0
        scores = []
        for clusters in range(2,9):
            kmeans = KMeans(n_clusters=clusters).fit(globals()['X_new'+features])
            kmeans_labels = kmeans.labels_
            kmeans_labels_test = kmeans_labels[y!=0]
            adj =  sm.adjusted_rand_score(y_red, kmeans_labels_test)
            scores.append(adj)
            if (adj > globals()['adjrandscorekm'+features]):
                globals()['adjrandscorekm'+features] = adj
                globals()['best_clusters'+features] = clusters
                globals()['labelskm'+features] = kmeans_labels
                globals()['labels_testkm'+features] = kmeans_labels_test
            test_kmeans_performance(features)
        plt.plot(clusters_idx, scores, colors[feats-2], label = features+' labels')
        print('----------KMeans-------------------')
        print_results(features,'km', globals()['best_clusters'+features])
    
    leg = plt.legend(loc='lower right')
    plt.savefig('kmeans_clusters.png')
    
    kmeans_final = KMeans(n_clusters=4).fit(X_new4)
    predict_final_km = kmeans_final.predict(X_new4)
    report_clusters(mat[:,0], predict_final_km, 'kmeans.html')
        
#DBSCAN
def db():
    plt.clf()
    plt.xlabel('Epsilon')
    plt.ylabel('Score')
    for feats in range(2,5):
        features = str(feats)
        plot_dist(features)
        globals()['best_eps'+features] = 0
        globals()['adjrandscoredb'+features] = 0
        scores=[]
        for epsilon in globals()['dist'+features]:
            db = DBSCAN(eps=epsilon)
            db_labels = db.fit_predict(globals()['X_new'+features])
            db_labels_test = db_labels[y!=0]
            adj = sm.adjusted_rand_score(y_red, db_labels_test)
            scores.append(adj)
            if(adj > globals()['adjrandscoredb'+features]):
                globals()['adjrandscoredb'+features] = adj
                globals()['best_eps'+features] = epsilon
                globals()['labelsdb'+features] = db_labels
                globals()['labels_testdb'+features] = db_labels_test
        test_db_performance(features)
        plt.plot(globals()['dist'+features], scores, colors[feats-2], label = features+' labels')
        print('----------DBScan-------------------')
        print_results(features,'db', globals()['best_eps'+features])
        
    leg = plt.legend(loc='lower right')
    plt.savefig('db_epsilons.png')
    
    #plt.show()
    
    predict_final_db = DBSCAN(eps=globals()['best_eps3']).fit_predict(X_new3)
    report_clusters(mat[:,0], predict_final_db, 'db.html')
    
def aff_prop():
    for feats in range(2,5):
        features = str(feats)
        globals()['best_damp'+features] = 0
        globals()['adjrandscoreaf'+features] = 0
        for damp in np.arange(0.5, 1.0, 0.05):
            afin = AffinityPropagation(damping=damp, random_state=None)
            af_labels = afin.fit_predict(globals()['X_new'+features])
            af_labels_test = af_labels[y!=0]
            adj = sm.adjusted_rand_score(y_red, af_labels_test)
            if(adj > globals()['best_damp'+features]):
                globals()['best_damp'+features] = damp
                globals()['adjrandscoreaf'+features] = adj
    
    predict_final_af = AffinityPropagation(damping=globals()['best_damp4'], random_state=None).fit_predict(X_new4)
    report_clusters(mat[:,0], predict_final_af, 'af.html')

mat = np.loadtxt('labels.txt', delimiter = ',')
y = mat[:,-1]
colors=["b", "r", "g"]

if(not path.isfile('best_features.npz')):
    data = images_as_matrix()
    data = (data - np.mean(data))/np.std(data)
    pca = PCA(n_components=6)
    pca_data = pca.fit_transform(data)
    
    tsne = TSNE(n_components=6, method='exact')
    tsne_data = tsne.fit_transform(data)
    
    isomap = Isomap(n_components=6)
    isomap_data = isomap.fit_transform(data)
    
    np.savez('best_features.npz', pca_data=pca_data, tsne_data=tsne_data, isomap_data=isomap_data)
else:
    best_features = np.load('best_features.npz')
    lst = best_features.files
    pca_data = best_features[lst[0]]
    tsne_data = best_features[lst[1]]
    isomap_data = best_features[lst[2]]

result = np.hstack((pca_data, tsne_data,isomap_data))

result = (result - np.mean(result))/np.std(result)

y_red = y[np.where(y!=0)]
X = result[y!=0]

f, prob = f_classif(X, y_red)

kbest2 = SelectKBest(f_classif, k=2).fit(X, y_red)
kbest3 = SelectKBest(f_classif, k=3).fit(X, y_red)
kbest4 = SelectKBest(f_classif, k=4).fit(X, y_red)

X_new2 = kbest2.transform(result)
X_new3 = kbest3.transform(result)
X_new4 = kbest4.transform(result)

k_means()
db()
#aff_prop()