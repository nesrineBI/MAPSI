# -*- coding: utf-8 -*-
"""



"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import random

"""
---------------------------Classification de lettres manuscrites-------------------------------
"""

"""
-----------------------Format des données----------------------------
"""
# old version = python 2
# data = pkl.load(file("ressources/lettres.pkl","rb"))
# new : 
with open('TME6_lettres.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1') 
X = np.array(data.get('letters')) # récupération des données sur les lettres
Y = np.array(data.get('labels')) # récupération des étiquettes associées 


# affichage d'une lettre
def tracerLettre(let):
    a = -let*np.pi/180; # conversion en rad
    coord = np.array([[0, 0]]); # point initial
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.figure()
    plt.plot(coord[:,0],coord[:,1])
    plt.savefig("exlettre.png")
    return

"""
--------------------Apprentissage d'un modèle CM (max de vraisemblance)------------------------
"""


""" ------------1.Discrétisation--------------
"""


def  discretise(X,d):
    y=[]
    for x in X:
        y.append(np.floor([i/(360/d) for i in x]))
    return np.array(y)

test=discretise(X,3)
#print(test)
"""-------2.Regrouper les indices des signaux par classe---------------
"""


def groupByLabel( y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index

"""--------3. Apprendre les modèles CM-----------------
"""

label=groupByLabel(Y)
#print(label)

def learnMarkovModel(Xc, d):
      A = np.zeros((d,d))
      Pi = np.zeros(d)
      for x in Xc:
          Pi[int(x[0])]+=1
          for i in range(len(x)):
              if i<len(x)-1:
                  A[int(x[i])][int(x[i+1])]+=1
      A = A/np.maximum(A.sum(1).reshape(d,1),1) # normalisation
      Pi = Pi/Pi.sum()
      return A,Pi

A_test,P_test=learnMarkovModel(discretise(X[label[0]],3),3)
print("validation pour une discretisation sur 3 états:")
print(P_test)
print(A_test)
print("\n\n")



"""----------4. Stocker les modèles dans une liste-------------
"""
d=20     # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
index = groupByLabel(Y)  # groupement des signaux par classe
models = []
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[index[cl]], d)) 

"""
--------------------Test (affectation dans les classes sur critère MV)------------
"""

"""-------------1. (log)Probabilité d'une séquence dans un modèle------
"""

def probaSequence(s,Pi,A):
    p=np.log(Pi[int(s[0])])           #début
    for i in range(len(s)-1):
        p=p+np.log(A[int(s[i])][int(s[i+1])])
    return p

test1=[]
for cl in models:
    test1.append(probaSequence(Xd[0],cl[1],cl[0]))
print("VALIDATION probabilité du premier signal dans les 26 modèles avec une discrétisation sur 20 états:")
print(test1)

"""
Q1:Ce signal est-il bien classé?
    Non on voit qu'il existe beaucoup de -Inf et il ne nous sert à rien pour déduire la classe qu'il appartient.
Q2:D'où viennent tous les -inf? 
    Il vient de la log(0). Et ça veut dire qu'il existe beaucoup de case qu'il n'a pas mis à jour dans la matrice A.
"""

"""
-----------2. Application de la méthode précédente pour tous les signaux et tous les modèles de lettres------
"""
    
proba = np.array([[probaSequence(Xd[i], models[cl][1], models[cl][0]) for i in range(len(Xd))]for cl in range(len(np.unique(Y)))])


"""
-----------3. Evaluation des performances------------------------------
"""
Ynum = np.zeros(Y.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num
pred = proba.argmax(0) # max colonne par colonne
print("\n taux de bonne classification d=20:")
print (np.where(pred != Ynum, 0.,1.).mean())



"""
--------------------------------------Biais d'évaluation, notion de sur-apprentissage----------------
"""
# separation app/test, pc=ratio de points en apprentissage
def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:int(np.floor(pc*n))])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest
# exemple d'utilisation
itrain,itest = separeTrainTest(Y,0.8)


    
"""-----------------------Lutter contre le sur-apprentissage--------------------------------------
"""
"""modele avec données apprentissage"""
d=20     # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
models = []
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[itrain[cl]], d)) 

"""données test et label test"""
X_test=[]
for cl in range(len(np.unique(Y))):
    for i in range(len(itest[cl])):
        X_test.append(Xd[itest[cl]][i])
X_test=np.array(X_test)

Y_test=[]
for cl in range(len(np.unique(Y))):
    for i in range(len(itest[cl])):
        Y_test.append(Y[itest[cl]][i])
Y_test=np.array(Y_test)
    
proba = np.array([[probaSequence(X_test[i], models[cl][1], models[cl][0]) for i in range(len(X_test))]for cl in range(len(np.unique(Y)))])


Ynum = np.zeros(Y_test.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y_test==char] = num
pred = proba.argmax(0) # max colonne par colonne
print("\n taux de bonne classification avec les données séparé d=20:")
print (np.where(pred != Ynum, 0.,1.).mean())

"""-----------------------------initialisation ones au lieu de zeros--------------------------------"""

def learnMarkovModel_ones(Xc, d):
      A = np.ones((d,d))
      Pi = np.ones(d)
      for x in Xc:
          Pi[int(x[0])]+=1
          for i in range(len(x)):
              if i<len(x)-1:
                  A[int(x[i])][int(x[i+1])]+=1
      A = A/np.maximum(A.sum(1).reshape(d,1),1) # normalisation
      Pi = Pi/Pi.sum()
      return A,Pi

d=20     # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
models = []
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel_ones(Xd[itrain[cl]], d)) 

"""données test et label test"""
X_test=[]
for cl in range(len(np.unique(Y))):
    for i in range(len(itest[cl])):
        X_test.append(Xd[itest[cl]][i])
X_test=np.array(X_test)

Y_test=[]
for cl in range(len(np.unique(Y))):
    for i in range(len(itest[cl])):
        Y_test.append(Y[itest[cl]][i])
Y_test=np.array(Y_test)
    
proba = np.array([[probaSequence(X_test[i], models[cl][1], models[cl][0]) for i in range(len(X_test))]for cl in range(len(np.unique(Y)))])


Ynum1 = np.zeros(Y_test.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum1[Y_test==char] = num
pred = proba.argmax(0) # max colonne par colonne
print("\n taux de bonne classification avec les données séparé avec ones initialisé au lieu de zeros d=20:")
print (np.where(pred != Ynum1, 0.,1.).mean())



"""
-----------------------------------Evaluation qualitative------------------------------------------------
"""

conf = np.zeros((26,26))
for i in range(len(Y_test)):
    conf[pred[i]][int(Ynum[i])]+=1
        
plt.figure()
plt.imshow(conf, interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(26),np.unique(Y))
plt.yticks(np.arange(26),np.unique(Y))
plt.xlabel(u'Vérité terrain')
plt.ylabel(u'Prédiction')
plt.savefig("mat_conf_lettres.png")
    




"""
    -------------------------------------Modèle génératif------------------------------------------------------
    """
"""--------Tirage selon une loi de probabilité discrète----------------"""
def tirage_etat(V):
    V=np.cumsum(V)
    val_random=random.random()
    cl=0
    for i in range(len(V)):
        if val_random<= V[i]:
            return cl
        else:
            cl+=1

    

"""----------Génération d'une séquence de longueur N----------------------"""
    

d=20
etats=[360/d*i for i in range(d)]


def generate(A,Pi,n):
    s=[etats[tirage_etat(Pi)]]
    i=1
    while i<=n:
        e=tirage_etat(A[int(s[-1]/d)])
        s.append(int(etats[e]))
        i+=1
        
    return s
        
            

newa = generate(models[8][0], models[8][1], 25) # generation d'une séquence d'états
intervalle = 360./d # pour passer des états => valeur d'angles
newa_continu = np.array([i*intervalle for i in newa]) # conv int => double
tracerLettre(newa_continu)







































