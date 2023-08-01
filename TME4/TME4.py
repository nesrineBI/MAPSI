# -*- coding: utf-8 -*-

import numpy as np
from math import *
from pylab import *
import matplotlib.pyplot as plt

#plt.ioff()

def read_file ( filename ):
    """
    Lit le fichier contenant les données du geyser Old Faithful
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )
    for ligne in infile:
        if ligne.find ( "eruptions waiting" ) != -1:
            break

    # ici, on a la liste des temps d'éruption et des délais d'irruptions
    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [ float (x) for x in ligne.split () ]
        data.append ( eruption )
        data.append ( waiting )
    infile.close ()

    # transformation de la liste en tableau 2D
    data = np.asarray ( data )
    data.shape = ( int ( data.size / 2 ), 2 )

    return data

data = read_file ( "2015_tme4_faithful.txt" )

#-------------------------------------Loi normal bidimensionnelle-----------------------------

print("param=[ux,uz,sigx,sigz,r]")
def normale_bidim(x,z,param):
    a=1/(2*pi*param[2]*param[3]*sqrt(1-(param[4]*param[4])))
    b=-1/(2*(1-param[4]*param[4]))
    c=pow((x-param[0])/param[2],2)
    d=2*param[4]*(x-param[0])*(z-param[1])/(param[2]*param[3])
    e=pow((z-param[1])/param[3],2)
    return a*exp(b*(c-d+e))

print(normale_bidim(1,2,(1,2,3,4,0)))
print(normale_bidim(1,0,(1,2,1,2,0.7)))

#-----------------------------Visualisation de loi normale bidimensionnelle-----------------

def dessine_1_normale ( params ):
    # récupération des paramètres
    mu_x, mu_z, sigma_x, sigma_z, rho = params

    # on détermine les coordonnées des coins de la figure
    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z

    # création de la grille
    x = np.linspace ( x_min, x_max, 100 )
    z = np.linspace ( z_min, z_max, 100 )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm = X.copy ()
    for i in range ( x.shape[0] ):
        for j in range ( z.shape[0] ):
            norm[i,j] = normale_bidim ( x[i], z[j], params )

    # affichage
    fig = plt.figure ()
    plt.contour ( X, Z, norm, cmap=plt.cm.autumn )
    plt.show ()
    
#dessine_1_normale ( (-3.0,-5.0,3.0,2.0,0.7) )
#dessine_1_normale ( (-3.0,-5.0,3.0,2.0,0.2) )

#---------------------------------Visualisation des données du Old Faithful---------------
    
def dessine_normales ( data, params, weights, bounds, ax ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # on détermine les coordonnées des coins de la figure
    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]

    # création de la grille
    nb_x = nb_z = 100
    x = np.linspace ( x_min, x_max, nb_x )
    z = np.linspace ( z_min, z_max, nb_z )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm0 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
            norm0[j,i] = normale_bidim ( x[i], z[j], params[0] )# * weights[0]
    norm1 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
             norm1[j,i] = normale_bidim ( x[i], z[j], params[1] )# * weights[1]

    # affichages des normales et des points du dataset
    ax.contour ( X, Z, norm0, cmap=plt.cm.winter, alpha = 0.5 )
    ax.contour ( X, Z, norm1, cmap=plt.cm.autumn, alpha = 0.5 )
    for point in data:
        ax.plot ( point[0], point[1], 'k+' )


def find_bounds ( data, params ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # calcul des coins
    x_min = min ( mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:,0].min() )
    x_max = max ( mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:,0].max() )
    z_min = min ( mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:,1].min() )
    z_max = max ( mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:,1].max() )

    return ( x_min, x_max, z_min, z_max )


# affichage des données : calcul des moyennes et variances des 2 colonnes
mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()

# les paramètres des 2 normales sont autour de ces moyennes
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [0.4, 0.6] )
bounds = find_bounds ( data, params )

# affichage de la figure
fig = plt.figure ()
ax = fig.add_subplot(111)
dessine_normales ( data, params, weights, bounds, ax )
plt.show ()

#------------------------------------EM:l'etape E------------------------------

def Q_i(datas,current_param,current_weight):
    T=np.array([])
    for data in datas:
        alpha_0=current_weight[0]*normale_bidim(data[0],data[1],current_param[0])
        alpha_1=current_weight[1]*normale_bidim(data[0],data[1],current_param[1])
        T=np.append(T,[alpha_0/(alpha_0+alpha_1),alpha_1/(alpha_0+alpha_1)])
    return np.reshape(T,(272,2))
        

#----------------------les 2 tests ici:-------------------------------------

#current_params = np.array ( [(mu_x, mu_z, sigma_x, sigma_z, rho),   # params 1ère loi normale
#                             (mu_x, mu_z, sigma_x, sigma_z, rho)] ) # params 2ème loi normale
current_params = np.array([[ 3.28778309, 69.89705882, 1.13927121, 13.56996002, 0. ],
                           [ 3.68778309, 71.89705882, 1.13927121, 13.56996002, 0. ]])

# current_weights = np.array ( [ pi_0, pi_1 ] )
current_weights = np.array ( [ 0.5, 0.5 ] )

T = Q_i ( data, current_params, current_weights )

current_params = np.array([[ 3.2194684, 67.83748075, 1.16527301, 13.9245876,  0.9070348 ],
                           [ 3.75499261, 73.9440348, 1.04650191, 12.48307362, 0.88083712]])
current_weights = np.array ( [ 0.49896815, 0.50103185] )
T = Q_i ( data, current_params, current_weights )


#------------------------------EM: l'etape M--------------------------------

def M_step(data,T,param,poids):
    sum_T_1=np.sum(T,0)[0]
    sum_T_2=np.sum(T,0)[1]
    #pi_0:
    
    pi_0=sum_T_1/(sum_T_1+sum_T_2)
    
    #u_x0 et u_z0:
    
    x=0
    z=0
    for i in range(len(data)):
       x+=T[i][0]*data[i][0]
       z+=T[i][0]*data[i][1]
    u_x0=x/sum_T_1
    u_z0=z/sum_T_1
    
    #sig_x0 et sig_z0:
    x=0
    z=0
    for i in range(len(data)):
        x+=T[i][0]*pow(data[i][0]-u_x0,2)
        z+=T[i][0]*pow(data[i][1]-u_z0,2)
    sig_x0=sqrt(x/sum_T_1)
    sig_z0=sqrt(z/sum_T_1)
    
    #r_0:
    
    r=0
    for i in range(len(data)):
        r+=T[i][0]*(data[i][0]-u_x0)*(data[i][1]-u_z0)/(sig_x0*sig_z0)
    r_0=r/sum_T_1


    #pi_1:
    
    pi_1=sum_T_2/(sum_T_1+sum_T_2)
    
    #u_x1 et u_z1:
    
    x=0
    z=0
    for i in range(len(data)):
       x+=T[i][1]*data[i][0]
       z+=T[i][1]*data[i][1]
    u_x1=x/sum_T_2
    u_z1=z/sum_T_2
    
    #sig_x0 et sig_z0:
    
    x=0
    z=0
    for i in range(len(data)):
        x+=T[i][1]*pow(data[i][0]-u_x1,2)
        z+=T[i][1]*pow(data[i][1]-u_z1,2)
    sig_x1=sqrt(x/sum_T_2)
    sig_z1=sqrt(z/sum_T_2)
    
    #r_0:
    
    r=0
    for i in range(len(data)):
        r+=T[i][1]*(data[i][0]-u_x1)*(data[i][1]-u_z1)/(sig_x1*sig_z1)
    r_1=r/sum_T_2


    return np.array([(u_x0,u_z0,sig_x0,sig_z0,r_0),(u_x1,u_z1,sig_x1,sig_z1,r_1)]),np.array([pi_0,pi_1])



current_params = array([(2.51460515, 60.12832316, 0.90428702, 11.66108819, 0.86533355),
                        (4.2893485,  79.76680985, 0.52047055,  7.04450242, 0.58358284)])
current_weights = array([ 0.45165145,  0.54834855])
Q = Q_i ( data, current_params, current_weights )
X,Y=M_step ( data, Q, current_params, current_weights )
print(X,Y)

#-----------------------------------------Algorithme EM : mise au point------------------------------

mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()

params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [ 0.5, 0.5 ] )

for i in range(20):
    fig = plt.figure ()
    ax = fig.add_subplot(111)
    dessine_normales(data, params, weights, bounds, ax)
    plt.show()
    E_pas = Q_i( data, params, weights )
    params, weights = M_step( data, E_pas, params, weights)
    print(params)
    print(weights)





#--------------------------------------Algorithme EM: version finale et animation---------------------
    
    
mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()


params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [ 0.5, 0.5 ] )


res_EM=[]
for i in range(20):
    res_EM.append([params,weights])
    E_pas = Q_i( data, params, weights )
    params, weights = M_step( data, E_pas, params, weights)

    
    
# calcul des bornes pour contenir toutes les lois normales calculées
def find_video_bounds ( data, res_EM ):
    bounds = np.asarray ( find_bounds ( data, res_EM[0][0] ) )
    for param in res_EM:
        new_bound = find_bounds ( data, param[0] )
        for i in [0,2]:
            bounds[i] = min ( bounds[i], new_bound[i] )
        for i in [1,3]:
            bounds[i] = max ( bounds[i], new_bound[i] )
    return bounds

bounds = find_video_bounds ( data, res_EM )


import matplotlib.animation as animation

# création de l'animation : tout d'abord on crée la figure qui sera animée
fig = plt.figure ()
ax = fig.gca (xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]))

# la fonction appelée à chaque pas de temps pour créer l'animation
def animate ( i ):
    ax.cla ()
    dessine_normales (data, res_EM[i][0], res_EM[i][1], bounds, ax)
    ax.text(5, 40, 'step = ' + str ( i ))
    print ("step animate = %d" % ( i ))

# exécution de l'animation
anim = animation.FuncAnimation(fig, animate, 
                               frames = len ( res_EM ), interval=500 )
plt.show ()
anim.save("em.gif", writer="imagemagick")
# éventuellement, sauver l'animation dans une vidéo
# anim.save('old_faithful.avi', bitrate=4000)
    
    
    

















