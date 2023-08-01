# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

"""--------------------------------------------Estimation de π par Monte Carlo---------------------------------"""

def tirage(m):
    x=np.random.rand()*m*2-m
    y=np.random.rand()*m*2-m
    return x,y

def monteCarlo(n):
    x=np.array([])
    y=np.array([])
    nb=0
    for i in range(n):
        t_x,t_y=tirage(1)
        x=np.append(x,t_x)
        y=np.append(y,t_y)
        if t_x*t_x+t_y*t_y<=1:
            nb+=1
    return 4*nb/n,x,y

plt.figure()

# trace le carré
plt.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')

# trace le cercle
x = np.linspace(-1, 1, 100)
y = np.sqrt(1- x*x)
plt.plot(x, y, 'b')
plt.plot(x, -y, 'b')

# estimation par Monte Carlo
pi, x, y = monteCarlo(int(1e4))

# trace les points dans le cercle et hors du cercle
dist = x*x + y*y 
plt.plot(x[dist <=1], y[dist <=1], "go")
plt.plot(x[dist>1], y[dist>1], "ro")
plt.show()

"""----------------------------------------Décodage par la méthode de Metropolis-Hastings---------------------------------"""



with open('countWar.pkl', 'rb') as f:
    (count, mu, A) = pkl.load(f,encoding='latin1') 
    
secret2 = (open("secret2.txt", "r")).read()[0:-1] # -1 pour supprimer le saut de ligne

tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }

def swapF(tau_old):
    val1=int(np.random.rand()*len(list(tau_old.keys())))
    val2=int(np.random.rand()*len(list(tau_old.keys())))
    c1=list(tau_old.keys())[val1]
    c2=list(tau_old.keys())[val2]
    tau_new=dict()
    for key,value in tau_old.items():
        if key==c2:
            tau_new[c1]=tau_old[c2]
        elif key==c1:
            tau_new[c2]=tau_old[c1]
        else:
            tau_new[key]=tau_old[key]
    return tau_new
print("validation de fonction swapF\n",tau)
tau_new=swapF(tau)
print("tau_new:\n",tau_new)

def decrypt(mess,tau):
    mess_decrypt=''
    for i in mess:
        mess_decrypt+=tau[i]
    return mess_decrypt

print("validation de fonction decrypt\nT = aabcd")
mess_dcrypt=decrypt ( "aabcd", tau )
print("T'=",mess_dcrypt)

chars2index = dict(zip(np.array(list(count.keys())), np.arange(len(count.keys()))))


def logLikelihood(mess,mu,A,chars2index):
    p=np.log(mu[chars2index[mess[0]]])
    for i in range(len(mess)-1):
        p+=np.log(A[chars2index[mess[i]]][chars2index[mess[i+1]]])
    return p

print("validation de fonction logLikelihood\n T=abcd")
p_abcd=logLikelihood("abcd",mu,A,chars2index)
print(p_abcd)
print("validation de fonction logLikelihood\n T=dcba")
p_dcba=logLikelihood("dcba",mu,A,chars2index)
print(p_dcba)


def MetropolisHastings(mess, mu, A, tau, N, chars2index):

    p_plus_possible=-np.Inf
    m_plus_possible=list()
    mess_old=decrypt(mess,tau)
    p_tau=logLikelihood(mess_old,mu,A,chars2index)
        
    for i in range(N):
        if i%100==0:
            print("learning...  pourcent: %.2f " %(i/100))
        tau_new=swapF(tau)
        mess_new=decrypt(mess,tau_new)
        p_tau_new=logLikelihood(mess_new,mu,A,chars2index)
        
        alpha=min(1,np.exp(p_tau_new-p_tau))
        val_ran=np.random.rand()
        if(val_ran<=alpha):
            tau=tau_new
            p_tau=p_tau_new
            if p_tau_new>p_plus_possible:
                p_plus_possible=p_tau_new
                m_plus_possible=mess_new
                #print(m_plus_possible)


    return m_plus_possible
    
    
def identityTau (count):
    tau = {} 
    for k in list(count.keys ()):
        tau[k] = k
    return tau


#m=MetropolisHastings( secret2, mu, A, identityTau (count), 50000, chars2index )
#print(m)

# ATTENTION: mu = proba des caractere init, pas la proba stationnaire
# => trouver les caractères fréquents = sort (count) !!
# distribution stationnaire des caracteres
freqKeys = np.array(list(count.keys()))
freqVal  = np.array(list(count.values()))
# indice des caracteres: +freq => - freq dans la references
rankFreq = (-freqVal).argsort()

# analyse mess. secret: indice les + freq => - freq
cles = np.array(list(set(secret2))) # tous les caracteres de secret
rankSecret = np.argsort(-np.array([secret2.count(c) for c in cles]))
# ATTENTION: 37 cles dans secret, 77 en général... On ne code que les caractères les plus frequents de mu, tant pis pour les autres
# alignement des + freq dans mu VS + freq dans secret
tau_init = dict([(cles[rankSecret[i]], freqKeys[rankFreq[i]]) for i in range(len(rankSecret))])

m=MetropolisHastings(secret2, mu, A, tau_init, 10000,chars2index )
print("validation pour le texte secret\n")
print(m)



"""
-------------------------------Estimation de π par MCMC------------------------------
"""
def transition(x,y,pas):
    val_ran=np.random.rand()
    if val_ran<0.25:
        return x-pas,y
    elif val_ran >=0.25 and val_ran < 0.5:
        return x+pas,y
    elif val_ran >=0.5 and val_ran <0.75:
        return x,y+pas
    else:
        return x,y-pas
    
    
def MCMC(n,p):
    pas=1/p
    depart_x,depart_y=tirage(1)
    nb=0
    X=np.array([depart_x])
    Y=np.array([depart_y])
    for i in range(n):
        x,y=transition(depart_x,depart_y,pas)
        if x>-1 and x<1 and y>-1 and y<1:
            depart_x,depart_y=x,y
            X=np.append(X,x)
            Y=np.append(Y,y)
            if x*x+y*y<1:
                nb+=1
    return 4*nb/n,X,Y
        
#pi_est,x,y=MCMC(50000,100)
#print(pi_est)


plt.figure()

# trace le carré
plt.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')

# trace le cercle
x = np.linspace(-1, 1, 100)
y = np.sqrt(1- x*x)
plt.plot(x, y, 'b')
plt.plot(x, -y, 'b')

# estimation par MCMC:
pi, x, y = MCMC(100000,100)

# trace les points dans le cercle et hors du cercle
dist = x*x + y*y 
plt.plot(x[dist <=1], y[dist <=1], "go")
plt.plot(x[dist>1], y[dist>1], "ro")
plt.show()
print(pi)





        
