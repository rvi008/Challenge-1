
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn import feature_selection
from numpy import dot, zeros
from numpy.linalg import matrix_rank, norm
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn import cross_validation
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sys


log_file = open("message.log","w")
sys.stdout = log_file

# In[2]:

# Critere de performance
def compute_pred_score(y_true, y_pred):
    y_comp = y_true * y_pred
    score = float(10*np.sum(y_comp == -1) + np.sum(y_comp == 0))
    score /= y_comp.shape[0]
    return score


# In[3]:

X_train_fname = 'training_templates.csv'
y_train_fname = 'training_labels.txt'
X_test_fname  = 'testing_templates.csv'
X_train = pd.DataFrame(pd.read_csv(X_train_fname, sep=',', header=None))
X_test  = pd.DataFrame(pd.read_csv(X_test_fname,  sep=',', header=None).values)
y_train = np.loadtxt(y_train_fname, dtype=np.int)


# Tout d'abord regardons si les colonnes sont *correlées* entre elles, auquel cas on poura enlever celles qui le sont trop.

# In[4]:

sns.set(context="paper", font="monospace")
corrmat = X_test.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, xticklabels=False, yticklabels=False);


# Visiblement, aucune corrélation évidente n'apparait.

# Etudions la distribution des variables

# In[5]:

X_train.hist(figsize=(50,50));


# Les variables suivent toutes des lois normales, il n'y a pas de problèmes évidents sur ces densités (distribution anormale, valeurs manquantes ...)

# On va se donner une référence avec un modèle de base.

# In[6]:

clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)


# In[7]:

# Prediction
y_pred_train =  clf.predict(X_train)

# Compute the score
score = compute_pred_score(y_train, y_pred_train)
print('Score sur le train : %s' % score)


# On va maintenant étudier les différentes *features*

# In[8]:

estimator = linear_model.LogisticRegression()
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X_train, y_train)


# In[9]:

print(selector.support_) 
print(selector.ranking_)


# In[10]:

print(np.where(selector.support_ == False))


# D'aprés cette sélection on voit que les colonnes #0, #10, #17, #18, #28, #72, #80, #104 et #117 ne sont pas significatives pour la régression logistique

# In[11]:

X_test.drop([  0,  10,  17,  18,  28,  72,  80, 104, 117], axis=1, inplace= True)
X_train.drop([  0,  10,  17,  18,  28,  72,  80, 104, 117],axis=1, inplace= True)


# In[12]:

print(X_test.shape, X_train.shape)


# Est-ce que le retrait de ces colonnes améliore le score ?

# In[13]:

clf.fit(X_train, y_train)
# Prediction
y_pred_train =  clf.predict(X_train)

# Compute the score
score = compute_pred_score(y_train, y_pred_train)
print('Score sur le train : %s' % score)


# Il y'a une légère amélioration du score.
# Prennons ce classifieur comme notre base de référence.
# Sur le leaderboard le score n'est pas interessant.
# La classification concerne le traitement d'image, nous savons que sur ce type de problématique, les réseaux de neurones sont performants. Nous allons dorénavant exploiter la classe MLP de sklearn et faire notre sélection de variables en ce sens. Nous allons reprendre le dataset d'origine puisque la feature selection a été faite pour une régression logistique mais n'est pas possible pour un réseau de neurones.

# In[14]:

X_train = pd.DataFrame(pd.read_csv(X_train_fname, sep=',', header=None))
X_test  = pd.DataFrame(pd.read_csv(X_test_fname,  sep=',', header=None).values)
clf2 = MLPClassifier()
clf2.fit(X_train, y_train)


# In[15]:

# Prediction
y_pred_train =  clf2.predict(X_train)

# Compute the score
score = compute_pred_score(y_train, y_pred_train)
print('Score sur le train : %s' % score)


# Le score est déjà plus intéressant

# In[16]:

y_pred = clf2.predict(X_test)
np.savetxt('y_pred.txt', y_pred, fmt='%d')


# Sur le leaderboard nous obtenons un score de **0.35**
# Est-ce que la standardization améliore le score ?

# In[17]:

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  


# In[18]:

# Prediction
y_pred_train =  clf2.predict(X_train)

# Compute the score
score = compute_pred_score(y_train, y_pred_train)
print('Score sur le train : %s' % score)


# Sur les données d'entrainement le score se dégrade.

# In[19]:

y_pred = clf2.predict(X_test)
np.savetxt('y_pred.txt', y_pred, fmt='%d')


# La même dégradation se constate sur le leaderboard avec un score de **0.40**

# Maintenant nous allons rechercher les hyper paramètres optimaux via un grid search

# In[20]:

# Rechargeons nos données sans Standardization
X_train = pd.DataFrame(pd.read_csv(X_train_fname, sep=',', header=None))
X_test  = pd.DataFrame(pd.read_csv(X_test_fname,  sep=',', header=None).values)


# In[24]:

clf2 = MLPClassifier(max_iter=100, solver='adam', hidden_layer_sizes=14, activation='tanh', alpha = 0.0002)
#param_grid = { 'max_iter' : [100, 300, 500, 1000]}
#grid_search = GridSearchCV(clf2, param_grid=param_grid)
#start = time()
#grid_search.fit(X_train, y_train)

#print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#      % (time() - start, len(grid_search.cv_results_['params'])))

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#report(grid_search.cv_results_)


# Maintenant que l'on a optimisé les hyper-paramètres on va prédire sur les données de test pour voir si la performance c'est améliorée.
# Le score obtenu est moins bon que le précédent **0.3438**

# In[25]:

clf2 = MLPClassifier(solver='adam', hidden_layer_sizes=14, activation='tanh', alpha = 0.0002, max_iter=300)
# Prediction
clf2.fit(X_train, y_train)
y_pred_train =  clf2.predict(X_train)
# Compute the score
score = compute_pred_score(y_train, y_pred_train)
print('Score sur le train : %s' % score)
y_pred = clf2.predict(X_test)
np.savetxt('y_pred.txt', y_pred, fmt='%d')


# Revenons au MLP standard sans modifier les hyper-paramêtres.
# Nous allons maintenant affiner notre décision sur le critère de la probabilité de prédiction

# In[32]:

clf3 = MLPClassifier()
clf3.fit(X_train, y_train)
# Compute the score
y_pred_train =  clf3.predict(X_train)
score = compute_pred_score(y_train, y_pred_train)
print('Score sur le train : %s' % score)
y_pred = clf3.predict(X_test)


# In[33]:

proba_class_neg = np.transpose(clf3.predict_proba(X_test))[0]
n, bins, patches = plt.hist(proba_class_neg, 10, facecolor = 'red')


# In[31]:

proba_class_pos = np.transpose(clf3.predict_proba(X_test))[1]
npos, binspos, patchespos = plt.hist(proba_class_pos, 10, facecolor = 'blue')


# In[34]:

print(clf3.classes_)
print(clf3.predict_proba(X_test))


# In[35]:

undefined = np.transpose(np.where(np.logical_and(0.0001 < proba_class_neg, proba_class_neg < 0.999)))
y_pred = clf3.predict(X_test)


# In[36]:

y_pred[undefined] = 0
print(len(y_pred[undefined]))


# In[37]:

np.savetxt('y_pred.txt', y_pred, fmt='%d')


# In[4]:

knn = KNeighborsClassifier(weights='distance', algorithm='kd_tree', p=1 )
knn.fit(X_train, y_train)
# Compute the score
y_pred_train =  knn.predict(X_train)
score = compute_pred_score(y_train, y_pred_train)
print('Score sur le train : %s' % score)
y_pred = knn.predict(X_test)


# In[5]:

np.savetxt('y_pred.txt', y_pred, fmt='%d')


# In[ ]:

param_grid = { 'n_neighbors=' : [3,5], 'p':[1,2,3] }
grid_search = GridSearchCV(knn, param_grid=param_grid)
start = time()
grid_search.fit(X_train, y_train)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


sys.stdout = old_stdout
log_file.close()
# In[ ]:



