# -*- coding: utf-8 -*-
"""
    
    
@author: nesrineBI
    
    
"""
import random as rnm
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import pyAgrum as gum
import pyAgrum.lib.ipython as gnb


#----------------------------fonction Bernoulli-------------------
def bernoulli(p):
    if(p<0 or p>1):
        raise ValueError("le probabilite doit etre entre 0 et 1")
    x=rnm.random()
    if x<=p:
        return 1
    else:
        return 0
    
#--------------------------fonction binomiale------------------------
def binomiale(n,p):
    k=0
    for i in range(0,n):
        k+=bernoulli(p)
    return k
#----------------------------gestion des sechemas---------------------------

plt.figure(figsize=(10,10))
plt.subplot(3,1,1)    
#-------------------------planche de Galton----------------------


tableau_1000_cases=np.zeros(1000)
for i in range(0,1000):
    tableau_1000_cases[i]=binomiale(1000,0.5)

res=plt.hist(tableau_1000_cases,100,color='b')
plt.xlabel("nombre de bulles")
plt.ylabel("nombre d'occurences")
plt.title("Planche de Galton")

#-----------------------------Visulisation d'indépendance------------------
#-----------------------------Loi normale centrée-------------------------

def normal(k,sigma):
    if k%2==0:
        raise ValueError("le nombre k doit etre impair")
    x=np.linspace(-2*sigma,2*sigma,k)
    y=np.array([(1/math.sqrt(2*math.pi))*math.exp(-(1/2)*math.pow(xi/sigma,2)) for xi in x])
    return y
#------test k=31,sigma=1------------------------------------------

plt.subplot(3,1,2)
plt.plot(np.linspace(-2*1,2*1,31),normal(31,1))
plt.legend(["Loi normale centrée k=31,sigma=1"])
    
#----------------------------Distribution de probabilite affine----------------

def proba_affine ( k, slope ):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    if abs ( slope  ) > 2. / ( k * k ):
        raise ValueError ( 'la pente est trop raide : pente max = ' + 
        str ( 2. / ( k * k ) ) )
    if slope < 0:
        raise ValueError("le slope ne peut pas etre negative")
    x=np.linspace(0,20,k)
    y=np.array([1/k+(i-(k-1)/2)*slope for i in x])
    return y
#-------test k=21, slope=0.004------------------
plt.subplot(3,1,3)
plt.plot(np.linspace(0,20,21),proba_affine(21,0.004))
plt.xlim(0,20)
plt.ylim(0,0.1)
plt.xlabel("x")
plt.ylabel("proba_affine")
plt.legend(["Distribution affine k=21,slope=0.004"])
plt.show()
    
#----------------------------Distribution jointe--------------------------------

def Pxy(x,y):
    pxy=np.array([])
    for i in x:
        pxy=np.append(pxy,(np.array([i*j for j in y])))
    pxy=np.reshape(pxy,(len(x),len(y)))

    return pxy
#--------------test P_AB---------------------------
print("PA=[0.2,0.7,0.1],PB=[0.4,0.4,0.2]")
print("PAB=")
print(Pxy(np.array([0.2,0.7,0.1]),np.array([0.4,0.4,0.2])))

#----------------------------Affichage de la distribution jointe--------------------
def dessine ( P_jointe ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace ( -3, 3, P_jointe.shape[0] )
    y = np.linspace ( -3, 3, P_jointe.shape[1] )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, P_jointe, rstride=1, cstride=1 )
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B)')
    plt.show ()

#----------------test PAB---------------------------------

dessine(Pxy(proba_affine(21,0.004),normal(21,1)))




#---------------------Independances conditionnelles----------------------------------

# creation de P(X,Y,Z,T)
P_XYZT = np.array([[[[ 0.0192,  0.1728],
                     [ 0.0384,  0.0096]],

                    [[ 0.0768,  0.0512],
                     [ 0.016 ,  0.016 ]]],

                   [[[ 0.0144,  0.1296],
                     [ 0.0288,  0.0072]],

                    [[ 0.2016,  0.1344],
                     [ 0.042 ,  0.042 ]]]])


#----------------------independance de X et T conditionnellement à (Y,Z)----------------

P_YZ=np.zeros((2,2))
P_YZ=np.sum(P_XYZT,(0,3))
print("calcul de P_YZ:")
print(P_YZ)

P_XT_YZ=np.zeros(16).reshape(2,2,2,2)
P_XT_YZ[0][0][0][0]=P_XYZT[0][0][0][0]/P_YZ[0][0]
P_XT_YZ[1][0][0][0]=P_XYZT[1][0][0][0]/P_YZ[0][0]
P_XT_YZ[0][1][0][0]=P_XYZT[0][0][0][1]/P_YZ[0][0]
P_XT_YZ[1][1][0][0]=P_XYZT[1][0][0][1]/P_YZ[0][0]
P_XT_YZ[0][0][1][0]=P_XYZT[0][1][0][0]/P_YZ[1][0]
P_XT_YZ[0][0][0][1]=P_XYZT[0][0][1][0]/P_YZ[0][1]
P_XT_YZ[0][0][1][1]=P_XYZT[0][1][1][0]/P_YZ[1][1]
P_XT_YZ[1][0][1][0]=P_XYZT[1][1][0][0]/P_YZ[1][0]
P_XT_YZ[1][0][0][1]=P_XYZT[1][0][1][0]/P_YZ[0][1]
P_XT_YZ[1][1][1][0]=P_XYZT[1][1][0][1]/P_YZ[1][0]
P_XT_YZ[1][1][0][1]=P_XYZT[1][0][1][1]/P_YZ[0][1]
P_XT_YZ[1][1][1][1]=P_XYZT[1][1][1][1]/P_YZ[1][1]
P_XT_YZ[0][1][1][0]=P_XYZT[0][1][0][1]/P_YZ[1][0]
P_XT_YZ[0][1][0][1]=P_XYZT[0][0][1][1]/P_YZ[0][1]
P_XT_YZ[0][1][1][1]=P_XYZT[0][1][1][1]/P_YZ[1][1]
P_XT_YZ[1][0][1][1]=P_XYZT[1][1][1][0]/P_YZ[1][1]

print("calcul de P_XT|YZ")
print(P_XT_YZ)


#------------------------------------Proba de P_X|YZ---------------------
P_X_YZ=np.zeros((2,2,2))
P_X_YZ=np.sum(P_XT_YZ,1)
print("calcul de P_X|YZ")
print(P_X_YZ)

#------------------------------------Proba de P_T|YZ---------------------

P_T_YZ=np.zeros((2,2,2))
P_T_YZ=np.sum(P_XT_YZ,0)
print("calcul de P_T|YZ")
print(P_T_YZ)


#------------------------------Verification P(XT|YZ) = P(X|YZ)  *  P(T|YZ) ---------------------

print(P_XT_YZ[0][0][0][0]==P_X_YZ[0][0][0]*P_T_YZ[0][0][0])
print("X|YZ et T|YZ ils n'est pas independant")



#-----------------------------independance X et (Y,Z)----------------------------

#----------------------------------calcul de P(XYZ)------------------------------
P_XYZ=np.zeros((2,2,2))

# np.sum(np.random.rand(5,6, 7), (0,2)).shape
P_XYZ=np.sum(P_XYZT,3)


#-------------------------------calcul de P(X)-------------------------------------
P_X=np.zeros(2)
P_X=np.sum(P_XYZ,(1,2))


#------------------------------verification de P(XYZ)=P(X) * P(YZ)----------------------

print("\n\ntester P_XYZ[0][0][0]==P_X[0]*P_YZ[0][0] si ils sont égaux")
print(P_XYZ[0][0][0]==P_X[0]*P_YZ[0][0])

print("X et YZ n'est pas independants")



#----------------------------------Independances conditionnelles et consommation mémoire--------



# creation de deux variables booléennes A et B :
A = gum.LabelizedVariable( 'A', 'A', 2 )
B = gum.LabelizedVariable( 'B', 'B', 2 )

# creation d'une distribution de probabilité P(A,B) :
proba = gum.Potential ()
proba.add ( A )
proba.add ( B )

# affichage de la probabilité

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


#------------------------------------------Test d'indépendance conditionnelle------------------
var,proba=read_file('2017_tme2_asia.txt')

def conditional_indep(proba,X,Y,Z,eps):
    return true


















