# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:04:35 2017

@author: 3410370
"""

import random as rnm
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import pyAgrum as gum
import pyAgrum.lib.ipython as gnb


# creation de P(X,Y,Z,T)
P_XYZT = np.array([[[[ 0.0192,  0.1728],
                     [ 0.0384,  0.0096]],

                    [[ 0.0768,  0.0512],
                     [ 0.016 ,  0.016 ]]],

                   [[[ 0.0144,  0.1296],
                     [ 0.0288,  0.0072]],

                    [[ 0.2016,  0.1344],
                     [ 0.042 ,  0.042 ]]]])
                     
                     
P_YZ=np.sum(P_XYZT,(0,3))
print(P_YZ)

P_XTcondYZ=np.zeros((2,2,2,2))
print(P_XYZT[0][1])

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                    P_XTcondYZ[i][j][k][l]=P_XYZT[i][j][k][l]/P_YZ[j][k]
                    
print(P_XTcondYZ)

P_XcondYZ=np.sum(P_XTcondYZ,1)
P_TcondYZ=np.sum(P_XTcondYZ,0)

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                print(P_XTcondYZ[i][j][k][l]==P_XcondYZ[j][k][l]*P_TcondYZ[l][k][l])
                
                
                
print("------------------------------------------------")

P_XYZ=np.sum(P_XYZT,3)
P_X=np.sum(P_XYZ,(1,2))
P_YZ=np.sum(P_XYZ,0)
print(P_X)

for i in range(2):
    for j in range(2):
        for k in range(2):
            print(P_XYZ[i][j][k]==P_X[i]*P_YZ[j][k])
            
            
print("------------------------------------------------")            
# creation de deux variables booléennes A et B :
A = gum.LabelizedVariable( 'A', 'A', 2 )
B = gum.LabelizedVariable( 'B', 'B', 2 )

# creation d'une distribution de probabilité P(A,B) :
proba = gum.Potential ()
proba.add ( A )
proba.add ( B )

# affichage de la probabilité
proba


def read_file ( filename ):
    """
    Renvoie les variables aléatoires et la probabilité contenues dans le
    fichier dont le nom est passé en argument.
    """
    Pjointe = gum.Potential ()
    variables = []

    fic = open ( filename, 'r' )
    # on rajoute les variables dans le potentiel
    nb_vars = int ( fic.readline () )
    for i in range ( nb_vars ):
        name, domsize = fic.readline ().split ()
        variable = gum.LabelizedVariable(name,name,int (domsize))
        variables.append ( variable )
        Pjointe.add(variable)

    # on rajoute les valeurs de proba dans le potentiel
    cpt = []
    for line in fic:
        cpt.append ( float(line) )
    Pjointe.fillWith(np.array ( cpt ) )

    fic.close ()
    return np.array ( variables ), Pjointe
    
    

var,p=read_file("2017_tme2_asia.txt")
#print(p.variableSequence().name())    



def conditional_indep(P_jointe,X,Y,Z,eps): 
    #P=P_jointe.margSumIn([i.name() for i in Z])
    P_Z=P_jointe.margSumIn([X.name(),Y.name()])
    P_XcondZ=P_jointe.margSumIn([Y.name()])/P_Z
    P_YcondZ=P_jointe.margSumIn([X.name()])/P_Z
    Q=P_jointe-(P_XcondZ*P_YcondZ)
    e=Q.abs().max()
    print(e)
    if e<= eps:
        return True
    else:
        return False


print(conditional_indep(p,var[0],var[1],np.delete(var,[0,1]),0.5))

def compact_conditional_proba(P_jointe,X,eps):
    var,p=read_file("2017_tme2_asia.txt")
    K=np.copy(var)
    K=np.delete(K,np.where(K==X))
    for i in len(K):
        if(conditional_indep(p,X,K[i],np.delete(K,[i]),eps)):
            K=np.delete(K,[i])
    return P_jointe




































