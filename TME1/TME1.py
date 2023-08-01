# -*- coding: utf-8 -*-
"""


@author: nesrineBI


"""
import requests
import pickle as pkl
import time
import matplotlib.pyplot as plt
import numpy as np

#plt.ioff()

fname = "dataVelib.pkl"
f= open(fname,'rb')
data = pkl.load(f)
f.close()
#------------------------------recuperation des donnees--------------------

list_station=[]
matrice_station=[]      # matrice attendu
for station in data:
    number_arr=station['number']//1000
    if(number_arr<=20 and number_arr>=1):
        list_station.append(station)
        lat=station['position'][u'lat']
        lng=station['position'][u'lng']
        alt=station['alt']
        arr=number_arr
        place_totale=station['bike_stands']
        place_dispo=station['available_bike_stands']
        ligne=[lat,lng,alt,arr,place_totale,place_dispo]
        matrice_station.append(ligne)

#print(matrice_station)
#print(len(list_station))

#----------------------------P(Ar)--------------------------------------
        
        
matrice_Ar=[0]*20
for s in matrice_station:
    for ar in range(1,21):
        if(s[3]==ar):
            matrice_Ar[ar-1]+=1
    matrice_P_Ar=[i/len(matrice_station) for i in matrice_Ar]  # matrice attendu

#print(matrice_Ar)
#print(matrice_P_Ar)
#---------------------------P(Al)-------------------------------------------


list_alt=[i[2] for i in matrice_station]
nIntervalles = 30
res = np.histogram(list_alt, nIntervalles)
matrice_Al=res[0] # effectif dans les intervalles
matrice_P_Al=[i/len(matrice_station) for i in matrice_Al]
#print(matrice_P_Al)
#print (res[1]) # definition des intervalles (ATTENTION: 31 valeurs)

#----------------------------P(Sp|Al)---------------------------------------

matrice_Sp_Al=[[0]*30]
matrice_Sp_Al.append([0]*30)  # matrice de taille 2 * 30

for s in matrice_station:
    for i in range(0,30):
        if (s[2]>=res[1][i] and s[2]<res[1][i+1]):
            if(s[5]==0):
                matrice_Sp_Al[0][i]+=1
            else:
                matrice_Sp_Al[1][i]+=1
                
ligne1=[matrice_Sp_Al[0][i]/(matrice_Sp_Al[0][i]+matrice_Sp_Al[1][i]) for i in range(0,30)]
ligne2=[matrice_Sp_Al[1][i]/(matrice_Sp_Al[1][i]+matrice_Sp_Al[0][i]) for i in range(0,30)]
matrice_P_Sp_Al=[]                            ## matrice attendu
matrice_P_Sp_Al.append(ligne1)
matrice_P_Sp_Al.append(ligne2)

#print(matrice_P_Sp_Al)
        
#----------------------------P(Vd|Al)-------------------------------------------
        
matrice_Vd_Al=[[0]*30]
matrice_Vd_Al.append([0]*30)

for s in matrice_station:
    for i in range(0,30):
        if (s[2]>=res[1][i] and s[2]<res[1][i+1]):
            if(s[5]>=2):
                matrice_Vd_Al[0][i]+=1
            else:
                matrice_Vd_Al[1][i]+=1
                
ligne1=[matrice_Vd_Al[0][i]/(matrice_Vd_Al[0][i]+matrice_Vd_Al[1][i]) for i in range(0,30)]
ligne2=[matrice_Vd_Al[1][i]/(matrice_Vd_Al[1][i]+matrice_Vd_Al[0][i]) for i in range(0,30)]
matrice_P_Vd_Al=[]                            ## matrice attendu
matrice_P_Vd_Al.append(ligne1)
matrice_P_Vd_Al.append(ligne2)

#print(matrice_P_Vd_Al)

#-----------------------------P(Vd_Ar)--------------------------------------------


matrice_Vd_Ar=[[0]*20]
matrice_Vd_Ar.append([0]*20)

for s in matrice_station:
    for ar in range(1,21):
        if (s[3]==ar):
            if(s[5]>=2):
                matrice_Vd_Ar[0][ar-1]+=1
            else:
                matrice_Vd_Ar[1][ar-1]+=1
                
ligne1=[matrice_Vd_Ar[0][i]/(matrice_Vd_Ar[0][i]+matrice_Vd_Ar[1][i]) for i in range(0,20)]
ligne2=[matrice_Vd_Ar[1][i]/(matrice_Vd_Ar[1][i]+matrice_Vd_Ar[0][i]) for i in range(0,20)]
matrice_P_Vd_Ar=[]                            ## matrice attendu
matrice_P_Vd_Ar.append(ligne1)
matrice_P_Vd_Ar.append(ligne2)

#print(matrice_P_Vd_Ar)

#-----------------------------trace P(Al)---------------------------------------------

alt=res[1]
intervalle=alt[1]-alt[0]
plt.bar((alt[1:]+alt[:-1])/2,matrice_P_Al,intervalle)
#plt.show()

#---------------------------E(P(Vd|Al))------------------------------------------------

matrice_E_Vd_Al=[0]*30

for i in range(0,30):
    matrice_E_Vd_Al[i]=matrice_P_Vd_Al[0][i]*1+matrice_P_Vd_Al[1][i]*0
plt.bar((alt[1:]+alt[:-1])/2,matrice_E_Vd_Al,alt[1]-alt[0])
plt.title("E(P(Vd|Al))")
plt.xlabel("Al")
plt.ylabel("E(P(Vd|Al))")
#print(matrice_E_Vd_Al)
#plt.show()

stations=np.array(matrice_station)
#----------------------------population-------------------------------------------------
plt.figure(figsize=(10,10))
plt.subplot(3,1,1)

x1 = stations[:,1] # recuperation des coordonnées 
x2 = stations[:,0]
# définition de tous les styles (pour distinguer les arrondissements)
style = [(s,c) for s in "o^**" for c in "byrmck" ] 

# tracé de la figure
for i in range(1,21):
    ind, = np.where(stations[:,3]==i)
    # scatter c'est plus joli pour ce type d'affichage
    plt.scatter(x1[ind],x2[ind],marker=style[i-1][0],c=style[i-1][1],linewidths=0)

plt.axis('equal') # astuce pour que les axes aient les mêmes espacements
plt.legend(range(1,21), fontsize=10)
plt.savefig("carteArrondissements.pdf")
#plt.imshow(plt_p)

#----------------------------rouge les stations pleines-------------------------------

plt.subplot(3,1,2)
x=stations[:,5]
total=stations[:,4]
ind,=np.where(x==0)
ind1=np.where(x==total)
tout=range(0,len(stations))
ind_=np.delete(tout,ind)
ind2=np.delete(ind_,ind1)

plt.scatter(x1[ind1],x2[ind1],marker=('o'),c=('r'),linewidths=0)
plt.scatter(x1[ind],x2[ind],marker=('o'),c=('y'),linewidths=0)
plt.scatter(x1[ind2],x2[ind2],marker=('o'),c=('g'),linewidths=0)
plt.axis('equal')
plt.savefig("ProjeteDesStations.pdf")
plt.legend(["stations pleines","stations vides","stations d'autres"])
#plt.imshow(plt_s)

#---------------------------moyenne--------------------------------------------------
plt.subplot(3,1,3)

altt=stations[:,2]
moy_alt=np.mean(altt)
med_alt=np.median(altt)
ind=np.where(altt <= moy_alt)    
ind1=np.where(altt >= med_alt)
plt.scatter(x1[ind],x2[ind],marker=('o'),c=('b'),linewidths=0)
plt.scatter(x1[ind1],x2[ind1],marker=('o'),c=('c'),linewidths=0)
plt.axis('equal')
plt.savefig("stations_moy_med.pdf")
plt.legend(["station_alt_inf_moy","station_alt_sup_med"])
plt.show()

#---------------------------correlation------------------------------------------------

corr_Alt_Velo=np.corrcoef(altt,x)[0][1]
corr_Ar_Velo=np.corrcoef(stations[:,3],x)[0][1]
print("la correlation entre Altitude et Velo dispo est %s "%corr_Alt_Velo)
print("la correlation entre Arrondissement et Velo dispo est %s "%corr_Ar_Velo)
print("On a trouvé que la corrélation entre altitude et velos disponibles est inférieur à celui entre arrondissement et velos disponibles, Alors on déduit que l'arrondissement est plus lié.")
#On a trouvé que la corrélation entre altitude et velos disponibles est inférieur à celui entre arrondissement 
# et velos disponibles, Alors on déduit que l'arrondissement est plus lié.






















