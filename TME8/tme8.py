# -*- coding: utf-8 -*-
"""
    
    
@author: nesrineBI
    
    
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
#from sklearn import svm

""" rapporter des fonction pratique de la TME précédent"""

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
        if(t%10000==1):
            print("classifing...%.2f " %(t/1000000))
    #terminaison:
    P=max(deta[T-1][i] for i in range(N))

    #chemin:
    
    S=[]
    S.append(np.argmax([deta[T-1][i] for i in range(N)]))
    for t in reversed(range(1,T)):
        S.append(int(phi[t][int(S[-1])]))       

        
    return S[::-1],P
          


"""----------------------------------TME______8--------------------------------------"""


#data = pkl.load(file("genome_genes.pkl","rb"))
with open('genome_genes.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1') 

    
Xgenes  = data.get("genes") #Les genes, une array de arrays

Genome = data.get("genome") #le premier million de bp de Coli

Annotation = data.get("annotation") ##l'annotation sur le genome
##0 = non codant, 1 = gene sur le brin positif

### Quelques constantes
DNA = ["A", "C", "G", "T"]
stop_codons = ["TAA", "TAG", "TGA"] 



"""-------------------------------------Question 2 : Un modèle séparant les codons et l'intergénique-----------------
"""
a=1/200
"""Q1:
    comme L suit la loi Ge(a), sa moyenne est de 200 alors on estime a=1/200 car E[X]=1/a=200
"""

b=1/(np.sum([len(x) for x in Xgenes])/len(Xgenes))


"""
    Q2: moyenne de longueur est 941.54 alors b=1/941.54
"""

""" Q3: calcul de distribution d'observation"""

Binter=[]
for i in range(4):
    g_i=np.where(Genome==i)
    Binter.append(len(g_i[0])/len(Genome))
    
print("validation de résultats avec Binter:\n",Binter)

Binter=np.array([Binter])

""" Q4: estimer la distribution chacun des positions des condons"""

Bgene=np.zeros((3,4))
nb_total=0
for x in Xgenes:
    for i in range(3,len(x)-3):
        if i%3==0:
            Bgene[0][x[i]]+=1
        elif i%3==1:
            Bgene[1][x[i]]+=1
        else:
            Bgene[2][x[i]]+=1
    nb_total+=len(x)-6
Bgene/=nb_total/3
print("validation de résultats avec Bgene:\n",Bgene)

Bgene=np.array(Bgene)

""" Q5:initialisation de la matrice de transition"""

Pi = np.array([1, 0, 0, 0]) 
A =  np.array([[1-a, a  , 0, 0], 
              [0  , 0  , 1, 0],
              [0  , 0  , 0, 1],
              [b  , 1-b, 0, 0 ]])
    
B = np.vstack((Binter, Bgene))



pred,proba  = viterbi(Genome,Pi,A,B)
#vsbce contient la log vsbce
#pred contient la sequence des etats predits (valeurs entieres entre 0 et 3)

#on peut regarder la proportion de positions bien predites
#en passant les etats codant a 1
sp1 = np.array(pred)
sp1[np.where(sp1>=1)] = 1
percpred1 = float(np.sum(sp1 == Annotation) )/ len(Annotation)
print("validation de résultats de prédiction des position des gènes:\n\n",percpred1)

""" Q6: affichage de graphe """
def show(x,a,L=50):
    n=len(x)
    img=np.zeros((L,n))
    for i in range(n):
        img[0:L,i]=x[i]
        img[L:i]=a[i]
    plt.imshow(img,cmap='Reds')
    plt.axis('off')
    
    
show(sp1[0:6000],Annotation[0:6000])


"""-----------------------------------------------Question 3-----------------------------------------------"""

Pi_q3 = np.array([1,0,0,0,0,0,0,0,0,0,0,0])
A_q3= np.array([[1-a,a,0,0,0,0,0,0,0,0,0,0], 
             [0,0,1,0,0,0,0,0,0,0,0,0],
             [1,0,0,1,0,0,0,0,0,0,0,0],
             [0,0,0,0,1,0,0,0,0,0,0,0],
             [0,0,0,0,0,1,0,0,0,0,0,0],
             [0,0,0,0,0,0,1,0,0,0,0,0],
             [0,0,0,0,1-b,0,b,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0.5,0.5,0,0],
             [0,0,0,0,0,0,0,0,0,0,1,0],
             [0,0,0,0,0,0,0,0,0,0,0.5,0.5],
             [0,0,0,0,0,0,0,0,0,0,0,1],
             [0,0,0,0,0,0,0,0,0,0,0,1]])
B_q3=np.zeros((12,4))
B_q3[0]=B[0]
B_q3[1]=[0.83,0,0.14,0.03]
B_q3[2]=[0,0,0,1]
B_q3[3]=[0,0,1,0]
B_q3[4]=B[1]
B_q3[5]=B[2]
B_q3[6]=B[3]
B_q3[7]=[0,0,0,1]
B_q3[8]=[0,0,1,0]
B_q3[9]=[1,0,0,0]
B_q3[10]=[1,0,0,0]
B_q3[11]=[0,0,1,0]


pred,proba  = viterbi(Genome,Pi_q3,A_q3,B_q3)
#vsbce contient la log vsbce
#pred contient la sequence des etats predits (valeurs entieres entre 0 et 3)

#on peut regarder la proportion de positions bien predites
#en passant les etats codant a 1
sp2 = np.array(pred)
sp2[np.where(sp2 ==4) ] = 1
sp2[np.where(sp2 ==5) ] = 1
sp2[np.where(sp2 ==6) ] = 1
percpred2 = float(np.sum(sp2 == Annotation) )/ len(Annotation)
print("validation de résultats de prédiction des position des gènes:\n\n",percpred2)

""" Q6: affichage de graphe """
show(sp2[0:6000],Annotation[0:6000])





















    


