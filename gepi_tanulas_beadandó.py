# -*- coding: utf-8 -*-
"""
Created on Fri May 21 23:50:21 2021

@author: Maródi Bence TT9PVO
"""

import numpy as np;  # Numerical Python library
from matplotlib import pyplot as plt;  # Matlab-like Python module
from urllib.request import urlopen;  # importing url handling
from sklearn.cluster import KMeans;  # importing clustering algorithms
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression;



#dataset beolvasás -----------------------------------------
url = 'https://raw.githubusercontent.com/bencem01/gepitanulas/main/data2.tsv';
raw_data = urlopen(url);  # url megnyitás 
data = np.loadtxt(raw_data, delimiter="\t");  # dataset betöltés
X = data[:,0:2];  #  input attributumok
y = data[:,2];

#pca illesztés --------------------------------------------
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)

#főkomponensek fontossága oszlop diagramm---------------------
plt.title('Főkomponensek varianciája');
var_ratio = pca.explained_variance_ratio_;
x_pos = np.arange(len(var_ratio))+1;
plt.xlabel('Főkomponensek');
plt.ylabel('Variancia');
plt.bar(x_pos,var_ratio, align='center', alpha=0.5);
plt.show(); 


# 2D pontdiagramm--------------------------------------------
fig = plt.figure(1);
plt.title('2D pontdiagramm');
plt.xlabel('X');
plt.ylabel('Y');
plt.scatter(X[:,0],X[:,1],s=50,c=y);
plt.show();

#Klaszter innen kezdődik--------------------------------------
# Alap klaszter szám 
K = 5;

# Konzolbeli klaszter input megadás 
user_input = input('Klaszterek száma [alap:5]: ');
if len(user_input) != 0 :
    K = np.int8(user_input);

# klaszterezés meghatározott K - val
kmeans_cluster = KMeans(n_clusters=K, random_state=2020);
kmeans_cluster.fit(X);   #  klaszter modell illesztés X re
y_pred = kmeans_cluster.predict(X);   #  predicting cluster label
sse = kmeans_cluster.inertia_;   # sum of squares of error (within sum of squares)
centers = kmeans_cluster.cluster_centers_;  # centroid of clusters

DB = davies_bouldin_score(X,y_pred);  

# Eredmények kiiratása
print(f'Klaszterek száma: {K}');
print(f'Within SSE: {sse}');
print(f'Davies-Bouldin index: {DB}');

# Klaszteres pontdiagramm-------------------------------------
fig = plt.figure(2);
plt.title('Pontdiagramm a klaszterek számával');
plt.xlabel('X');
plt.ylabel('Y');
plt.scatter(X[:,0],X[:,1],s=50,c=y_pred);   #  dataponts with cluster label
plt.scatter(centers[:,0],centers[:,1],s=50,c='orange');  #  center pontok
plt.show();

# Optimális klaszter szám megkeresése 
Max_K = 30;  # maximális klaszter szám
SSE = np.zeros((Max_K-2));  #  array for sum of squares errors
DB = np.zeros((Max_K-2));  # array for Davies Bouldin indeces
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(X);
    y_pred = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(X,y_pred);
    
    

    
