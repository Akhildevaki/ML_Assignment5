#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



# In[42]:


dataset_cc = pd.read_csv('C:\\Users\\Rams\Downloads\\CC.csv')
dataset_cc.info()


# In[43]:


dataset_cc.describe()


# In[44]:


dataset_cc[:]


# In[45]:


dataset_cc.isna().sum()


# In[46]:


dataset_cc['MINIMUM_PAYMENTS'].fillna(dataset_cc['MINIMUM_PAYMENTS'].mean(), inplace=True)
dataset_cc.dropna(axis=1,inplace=True)


# In[47]:


dataset_cc.drop(['CUST_ID'], axis=1, inplace=True)


# In[48]:


### Applying K-means with 3 clusters
nclusters = 3
km = KMeans(n_clusters=nclusters)
km.fit(dataset_cc)
print(km)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(dataset_cc)
score = metrics.silhouette_score(dataset_cc, y_cluster_kmeans)
print(score)


# In[74]:


### Using Principal Component Analysis
pca = PCA(2)
x_pca = pca.fit_transform(dataset_cc)
df2 = pd.DataFrame(data=x_pca)
df2.head()


# In[50]:


# K means with 3 clusters with PCA
nclusters = 3
km = KMeans(n_clusters=nclusters)
km.fit(df2)
print(km)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(df2)
score = metrics.silhouette_score(df2, y_cluster_kmeans)
print(score)


# In[76]:


### Scaling of Features
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df2)


# In[52]:


### K means with 3 clusters with PCA and Scaled features
nclusters = 3
km = KMeans(n_clusters=nclusters)
km.fit(scaled_df)
print(km)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(scaled_df)
score = metrics.silhouette_score(scaled_df, y_cluster_kmeans)
print(score)


# silhoutee scores (KMeans clusters =3):
# 
# KMeans                              = 0.55
# KMeans + PCA(k=2)                   = 0.60
# KMeans + PCA(k=2) + feature scaling = 0.63

# In[53]:


speech_df = pd.read_csv('C:\\Users\\Rams\Downloads\\pd_speech_features.csv')


# In[54]:


speech_df.info()


# In[55]:


speech_df['class'].value_counts()


# In[56]:


X = speech_df.drop(['class'],axis=1)
y = speech_df['class']
X.columns


# In[57]:


pca = PCA(3)
x_pca = pca.fit_transform(X)
X2 = pd.DataFrame(data=x_pca)


# In[58]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25)


# In[60]:


### SVM with PCA and Scaled features
clf = svm.SVC()
clf.fit(X_train, y_train)
clf.score(X_train,y_train)


# In[61]:


y_pred = clf.predict(X_test)


# In[62]:


### Accuracy score using SVM with PCA and Scaled features
metrics.accuracy_score(y_test, y_pred)


# In[63]:


#3.


# In[70]:


iris_data = pd.read_csv('C:\\Users\\Rams\Downloads\\Iris.csv')


# In[71]:


X = iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[72]:


### Using Linear Discriminant Analysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# In[73]:


### classification using Random Forest Classifier with LDA
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)


# In[68]:


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()
print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))


# # PCA vs LDA
# 
# Both PCA and LDA are linear transformation techniques.
# However, PCA is an unsupervised while LDA is a supervised dimensionality reduction technique.
# 
# ## Principal Component Analysis
# PCA summarizes the feature set without relying on the output.
# PCA tries to find the directions of the maximum variance in the dataset.
# In a large feature set, there are many features that are merely duplicate of the other features or have a high correlation with the other features.
# Such features are basically redundant and can be ignored. The role of PCA is to find such highly correlated or duplicate features and to come up with a new feature set where there is minimum correlation between the features or in other words feature set with maximum variance between the features.
# Since the variance between the features doesn't depend upon the output, therefore PCA doesn't take the output labels into account.
# 
# ## Linear Discriminant Analysis
# LDA tries to reduce dimensions of the feature set while retaining the information that discriminates output classes.
# LDA tries to find a decision boundary around each cluster of a class.
# It then projects the data points to new dimensions in a way that the clusters are as separate from each other as possible and the individual elements within a cluster are as close to the centroid of the cluster as possible.
# The new dimensions are ranked on the basis of their ability to maximize the distance between the clusters and minimize the distance between the data points within a cluster and their centroids.
# These new dimensions form the linear discriminants of the feature set.
