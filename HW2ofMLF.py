# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:12:49 2021

@author: Administrator
"""

import math 
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as skpre
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

#DATA PROCESSING

data, meta = arff.loadarff('C:/Users/Administrator/Desktop/4year.arff')
df=pd.DataFrame(data)
df['bankruptcy'] = (df['class']==b'1')
df.drop(columns=['class'], inplace=True)
df.columns = ['X{0:02d}'.format(k) for k in range(1,65)] + ['bankruptcy']
df.fillna(df.mean(), inplace=True)
X_imp = df.values
X, y = X_imp[:, :-1], X_imp[:, -1]
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
scaler = skpre.StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

y_train=y_train*1
y_test=y_test*1
y_train=y_train.astype(int)
y_test=y_test.astype(int)
lr = LogisticRegression(penalty='l1',C=0.01, solver='liblinear')
lr.fit(X_train_std, y_train.astype(int))
lr.coef_[lr.coef_!=0].shape      
X_train_std=X_train_std[:,lr.coef_[0]!=0]
X_test_std=X_test_std[:,lr.coef_[0]!=0]

#lR MODEL
lr = LogisticRegression(penalty='l1')
lr.fit(X_train_std, y_train.astype(int))
print('LR Training accuracy:', lr.score(X_train_std, y_train.astype(int)))
print('LR Test accuracy:', lr.score(X_test_std, y_test.astype(int)))

#models = (svm.SVC(kernel='linear', C=C),
#          svm.LinearSVC(C=C, max_iter=10000),
#          svm.SVC(kernel='rbf', gamma=0.7, C=C),
#          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
##for clf in models:

clf=svm.SVC(C=1,kernel='rbf',gamma=10)
clf.fit(X_train_std,y_train)
print('SVM Training accuracy:', clf.score(X_train_std, y_train))
print('SVM Test accuracy:', clf.score(X_test_std, y_test))


tree= DecisionTreeClassifier(criterion='gini', 
                                    max_depth=4, 
                                    random_state=1)
tree.fit(X_train, y_train)

#############Codes below can't run, but I can't find the reason
#############Codes below can't run, but I can't find the reason
#############Codes below can't run, but I can't find the reason
#############print('DecTree Training accuracy:', tree.score(X_train_std,y_train))
#############print('DecTree Test accuracy:', tree.score(X_test_std,y_test))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')
        
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=lr, test_idx=range(0, 50))
plt.xlabel('ratio1 [standardized]')
plt.ylabel('ratio2[standardized]')
plt.legend(loc='upper left')

plt.tight_layout()

plt.show()

#############Codes below can't run, but I can't find the reason
#############Codes below can't run, but I can't find the reason
#############Codes below can't run, but I can't find the reason

#plot_decision_regions(X_combined_std, y_combined,
#                      classifier=svm.SVC, test_idx=range(105, 150))
#plt.xlabel('petal length [standardized]')
#plt.ylabel('petal width [standardized]')
#plt.legend(loc='upper left')
#plt.tight_layout()
##plt.savefig('images/03_15.png', dpi=300)
#plt.show()
#
#plt.tight_layout()
##plt.savefig('images/03_01.png', dpi=300)
#plt.show()
#
#X_combined = np.vstack((X_train_std, X_test_std))
#y_combined = np.hstack((y_train, y_test))
#plot_decision_regions(X_combined, y_combined, 
#                      classifier=tree_model,
#                      test_idx=range(105, 150))
#
#plt.xlabel('ratio1')
#plt.ylabel('ratio2')
#plt.legend(loc='upper left')
#plt.tight_layout()
##plt.savefig('images/03_20.png', dpi=300)
#plt.show()
##