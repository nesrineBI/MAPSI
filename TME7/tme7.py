# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:03:52 2018

@author: nesrineBI
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import random

# truc pour un affichage plus convivial des matrices numpy
np.set_printoptions(precision=2, linewidth=320)
plt.close('all')

with open('TME6_lettres.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1') 
    X = np.array(data.get('letters'))
    Y = np.array(data.get('labels'))

nCl = 26

"""------------------------------------Passage au MMC-----------------------------------------

""------Apprentissage d'un modèle connaissant les états-------"""

def initGD(X,N):
    S=[]
    for x in X:
        s=np.floor(np.linspace(0,N-.00000001,len(x)))
        S.append(s)
    return np.array(S)

def  discretise(X,d):
    y=[]
    for x in X:
        y.append(np.floor([i/(360/d) for i in x]))
    return np.array(y)

K = 10 # discrétisation (=10 observations possibles)
N = 5  # 5 états possibles (de 0 à 4 en python) 
Xd=discretise(X,K) # Xd = angles observés discrétisés
Q=initGD(Xd,N)


def learnHMM(allx, allq, N, K, initTo0=False):
    if initTo0:
        A = np.zeros((N,N))
        B = np.zeros((N,K))
        Pi = np.zeros(N)
    else:
        eps = 1e-8
        A = np.ones((N,N))*eps
        B = np.ones((N,K))*eps
        Pi = np.ones(N)*eps
    for q,x in zip(allq,allx):
        Pi[int(q[0])]+=1
        for t in range(len(q)):
            if t<len(q)-1:
                A[int(q[t])][int(q[t+1])]+=1
                B[int(q[t])][int(x[t])]+=1
            if t==len(q)-1:
                B[int(q[t])][int(x[t])]+=1
    A = A/np.maximum(A.sum(1).reshape(N,1),1) # normalisation
    B = B/np.maximum(B.sum(1).reshape(N,1),1)
    Pi = Pi/Pi.sum()
    return Pi,A,B
                
            
Pi, A, B = learnHMM(Xd[Y=='a'],Q[Y=='a'],N,K)

print("Validation sur les séquences de la classe A:")
print("Pi=\n",np.around(Pi,decimals=2))
print("A=\n",np.around(A,decimals=2))
print("B=\n",np.around(B,decimals=2))        
            
"""---------------------------------------Viterbi (en log)----------------------------------------------"""       
            
def viterbi(x,Pi,A,B):
    #déclaration:
    N=len(Pi)
    T=len(x)
    deta=np.zeros((T,N))
    phi=np.zeros((T,N))
    
    #initialisation:
    for i in range(N):
        deta[0][i]=np.log(Pi[i])+np.log(B[i][int(x[0])])
        phi[0][i]=-1
        
    #récursion:
    for t in range(1,T):
        for j in range(N):
            deta[t][j]=max(deta[t-1][i]+np.log(A[i][j]) for i in range(N))+np.log(B[j][int(x[t])])
            phi[t][j]=np.argmax([deta[t-1][i]+np.log(A[i][j]) for i in range(N)])
    
    #terminaison:
    P=max(deta[T-1][i] for i in range(N))

    #chemin:
    S=[]
    S.append(np.argmax([deta[T-1][i] for i in range(N)]))
    for t in reversed(range(1,T)):
        S.append(int(phi[t][int(S[-1])]))       
        
    return S[::-1],P
          
s_est, p_est = viterbi(Xd[0], Pi, A, B)  
   
print("validation en utilisant le modèle précédent λA (mêmes valeurs de N et K):")    
print("s_est=\n",s_est)
print("p_est=\n",p_est)     
            
            
"""-------------------------------[OPT] Probabilité d'une séquence d'observation---------------------------"""     

def calc_log_pobs_v2(x,Pi, A, B):
    #déclaration:
    N=len(Pi)
    T=len(x)
    alpha=np.zeros((T,N))
    
    #initialisation:
    for i in range(N):
        alpha[0][i]=Pi[i]*B[i][int(x[0])]
    
    #récursion:
    for t in range(1,T):
        for j in range(N):
            alpha[t][j]=np.sum([alpha[t-1][i]*A[i][j] for i in range(N)])*B[j][int(x[t])]
            
    #terminaison:
    P=np.log(np.sum(alpha[T-1]))
    
    return P
p =  calc_log_pobs_v2(Xd[0],Pi, A, B)
print("validation pour alpha:")
print("p=\n",p) 
print("la version alpha est plus élevé ")           
            
"""-----------------------------Apprentissage complet (Baum-Welch simplifié)---------------------------------"""




def Baum_Welch_simplifie(X_cl,N,K):
    #initialisation:
    Q_cl=initGD(X_cl,N)
    eps = 1e-4   
    k,L_old,L_new,L_tmp1=0,0,1,0,
    liste_k=[]
    L=[]
    Pi, A, B = learnHMM(X_cl,Q_cl,N,K)

        
    #démarche:
    while(k==0 or k==1 or abs((L_old-L_new)/L_old) > eps):
        L_old=L_new
        for i in range(len(X_cl)):
            s_est, p_new = viterbi(X_cl[i], Pi, A, B)   #estimation
            Q_cl[i]=s_est
            L_tmp1+=p_new
            
        Pi, A, B = learnHMM(X_cl,Q_cl,N,K)
        
        L_new=L_tmp1
        k+=1
        liste_k.append(k)
        L.append(L_new)
        plt.plot(liste_k,L)
        L_tmp1=0
    return Pi,A,B

#apprendre les paramètres pour tous les classes:   
#model=[]
#for i,char in zip(range(nCl),np.unique(Y)):  
#    model.append(Baum_Welch_simplifie(Xd[Y==char],N,K)) 

          
"""----------------------------Evaluation des performances--------------------------"""
            
            
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
            
""" données pour apprendre"""           
X_train=[]
for cl in range(len(np.unique(Y))):
    for i in range(len(itrain[cl])):
        X_train.append(Xd[itrain[cl]][i])
X_train=np.array(X_train)   

Y_train=[]
for cl in range(len(np.unique(Y))):
    for i in range(len(itrain[cl])):
        Y_train.append(Y[itrain[cl]][i])
Y_train=np.array(Y_train) 

"""données pour test"""
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
            

""" apprendre pour des données train"""
model=[]
for i,char in zip(range(nCl),np.unique(Y)):
    print("apprendre les paramètre pour classe %d"%i)
    model.append(Baum_Welch_simplifie(X_train[Y_train==char],N,K))   

""" tester pour des données test"""

print("test...")
proba = np.array([[viterbi(X_test[i], model[cl][0], model[cl][1],model[cl][2])[1] for i in range(len(X_test))]for cl in range(len(np.unique(Y)))])
Ynum = np.zeros(Y_test.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y_test==char] = num
pred = proba.argmax(0) # max colonne par colonne
print("\n taux de bonne classification avec les données séparé:")
print (np.where(pred != Ynum, 0.,1.).mean())     

            
            
"""---------------------------------Génération de lettres-------------------------------------"""   

etats=[i for i in range(N)]
obs=[i for i in range(K)]
def tirage_etat(V):
    V=np.cumsum(V)
    val_random=random.random()
    cl=0
    for i in range(len(V)):
        if val_random<= V[i]:
            return cl
        else:
            cl+=1
            
def generateHMM(Pi,A,B,n):
    s=[etats[tirage_etat(Pi)]]
    x=[obs[tirage_etat(B[int(s[-1])])]]
    i=1
    while i<=n:
        e_t=tirage_etat(A[int(s[-1])])
        x_t=tirage_etat(B[int(etats[e_t])])
        s.append(int(etats[e_t]))
        x.append(int(obs[x_t]))
        i+=1
    return s,x
        
    
# affichage d'une lettre (= vérification bon chargement)
def tracerLettre(let):
    a = -let*np.pi/180;
    coord = np.array([[0, 0]]);
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.plot(coord[:,0],coord[:,1])
    return

#Trois lettres générées pour 5 classes (A -> E)
n = 3          # nb d'échantillon par classe
nClred = 5   # nb de classes à considérer
fig = plt.figure()
for cl in range(nClred):
    Pic = model[cl][0].cumsum() # calcul des sommes cumulées pour gagner du temps
    Ac = model[cl][1].cumsum(1)
    Bc = model[cl][2].cumsum(1)
    long = np.floor(np.array([len(x) for x in Xd[itrain[cl]]]).mean()) # longueur de seq. à générer = moyenne des observations
    for im in range(n):
        s,x = generateHMM(Pic, Ac, Bc, int(long))
        intervalle = 360./K  # pour passer des états => angles
        newa_continu = np.array([i*intervalle for i in x]) # conv int => double
        sfig = plt.subplot(nClred,n,im+n*cl+1)
        sfig.axes.get_xaxis().set_visible(False)
        sfig.axes.get_yaxis().set_visible(False)
        tracerLettre(newa_continu)
plt.savefig("lettres_hmm.png")         
            
            
            
