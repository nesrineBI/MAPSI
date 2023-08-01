#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D


def read_file ( filename ):
    """
    Lit un fichier USPS et renvoie un tableau de tableaux d'images.
    Chaque image est un tableau de nombres réels.
    Chaque tableau d'images contient des images de la même classe.
    Ainsi, T = read_file ( "fichier" ) est tel que T[0] est le tableau
    des images de la classe 0, T[1] contient celui des images de la classe 1,
    et ainsi de suite.
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )    
    nb_classes, nb_features = [ int( x ) for x in infile.readline().split() ]

    # creation de la structure de données pour sauver les images :
    # c'est un tableau de listes (1 par classe)
    data = np.empty ( 10, dtype=object )   
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    filler( data, data )

    # lecture des images du fichier et tri, classe par classe
    for ligne in infile:
        champs = ligne.split ()
        if len ( champs ) == nb_features + 1:
            classe = int ( champs.pop ( 0 ) )
            data[classe].append ( list ( map ( lambda x: float(x), champs ) ) )
    infile.close ()

    # transformation des list en array
    output  = np.empty ( 10, dtype=object )
    filler2 = np.frompyfunc(lambda x: np.asarray (x), 1, 1)
    filler2 ( data, output )

    return output

def display_image ( X ):
    """
    Etant donné un tableau X de 256 flotants représentant une image de 16x16
    pixels, la fonction affiche cette image dans une fenêtre.
    """
    # on teste que le tableau contient bien 256 valeurs
    if X.size != 256:
        raise ValueError ( "Les images doivent être de 16x16 pixels" )

    # on crée une image pour imshow: chaque pixel est un tableau à 3 valeurs
    # (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X

    # on indique que toutes les images sont de 16x16 pixels
    img.shape = (16,16,3)

    # affichage de l'image
    plt.imshow( img )
    plt.show ()
    
training_data = read_file ( "2015_tme3_usps_train.txt" )

# affichage du 1er chiffre "2" de la base:
display_image ( training_data[2][0] )

# affichage du 5ème chiffre "3" de la base:
display_image ( training_data[3][4] )

#----------------------------------Maximum de vraisemblance pour une classe------------------------
display_image ( training_data[0][0] )

def learnML_class_parameters(train_class):
    nb_pic=len(train_class)
    param_u=np.sum(train_class,0)/nb_pic
    param_v=np.zeros(256)
    for i in range(0,256):
        for j in range(0,nb_pic):
            param_v[i]+=math.pow(train_class[j][i]-param_u[i],2)
    param_v/=nb_pic
    return param_u,param_v
        
param_u0,param_v0=learnML_class_parameters(training_data[0])
print("classe[0] a parametre u0 et v0:")
print(param_u0)
print(param_v0)



#--------------------------------Maximum de vraisemblance pour toutes les classes---------------------

def learnML_all_parameters(train_class):
    list_class=[]
    for i in range(0,10):
        m,v=learnML_class_parameters(train_class[i])
        list_class.append([m,v])
    return list_class





#--------------------------------Log-vraisemblance d'une image-------------------------------------

def log_likelihood(image,param):
    val=0
    for i in range(0,256):
        if param[1][i]!=0:
            val+=(-1/2)*math.log(2*math.pi*param[1][i])-(1/2)*math.pow(image[i]-param[0][i],2)/(param[1][i])
    return val


parameters = learnML_all_parameters ( training_data )
test_data = read_file ( "2015_tme3_usps_test.txt" )
test1=log_likelihood ( test_data[2][3], parameters[1] )
test2=[ log_likelihood ( test_data[0][0], parameters[i] ) for i in range ( 10 ) ]
print("\n2 tests dans LOG-VRAISEMBLANCE D'UNE IMAGE:")
print(test1)
print(test2)

#-------------------------------Log-vraisemblance d'une image(bis)---------------------------------

def log_likelihoods(image,list_param):
    tab=[]
    for param_classe in list_param:
        tab.append(log_likelihood(image,param_classe))
    return np.array(tab)
    
print("\nun test dans LOG-VRAISEMBLANCE D'UNE IMAGE (BIS) :")
test3=log_likelihoods ( test_data[1][5], parameters )
print(test3)




#------------------------------Classification d'une image---------------------------------------------

def classify_image(image,list_param):
    tab=log_likelihoods(image,list_param)
    val_max=np.max(tab)
    return np.where(tab==val_max)[0]
    
print("\ntest pour classifier d'une image:")
print(classify_image( test_data[1][5], parameters ))

print(classify_image( test_data[4][1], parameters ))


#------------------------------Partie optionelle------------------------------------------------------

def classify_all_images(test_data,list_param):
    T=np.zeros((10,10))
    for i in range(0,10):
        for test_image in test_data[i]:
            T[i][classify_image(test_image,list_param)]+=1
        T[i]/=len(test_data[i])
    return T

T=classify_all_images(test_data,parameters)
print("\nverification de pourcentage de classification")
print("T[0,0]=")
print(T[0,0])
print("T[2,3]=")
print(T[2,3])

#----------------------------Affichage du resultat des classfications------------------------------------

def dessine ( classified_matrix ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.linspace ( 0, 9, 10 )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, classified_matrix, rstride = 1, cstride=1 )



dessine(T)






