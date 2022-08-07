# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:12:41 2022

@author: yfyya
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

############################
def dec_2(x,N):
    y = bin(x)
    y = y[2::]
    yn = np.array([eval(ni) for ni in y])
    NR = N-len(yn)
    yr = np.zeros(NR)
    yt = np.append(yr,yn)
    yt = np.array(yt,dtype=int)
    return yt

def dec_10(x):
    N = len(x)
    y = np.arange(N-1,-0.1,-1)
    c = int(np.sum(2**y*x))
    return c
###########################

N = 18  #### number of atoms
aa = np.zeros(N,dtype = int)
for ni in range(N):
    if np.mod(ni,2)==0:
        aa[ni] = 1
    else:
        aa[ni] = 0
Psi = np.array([],dtype = int)
Psi = np.append(Psi,dec_10(aa))
V = np.array([1])

H = np.zeros((N+1,N+1))
n_H = 1
for ni in range(N):
    Psi1 = np.array([],dtype = int)
    V1 = []
    for nj in range(len(Psi)):
        aa = dec_2(Psi[nj],N)
        for nk in range(N):
            bb = np.copy(aa)
            if nk==0:
                n_l = N-1
                n_r = 1
            elif nk==N-1:
                n_l = N-2
                n_r = 0
            else:
                n_l = nk-1
                n_r = nk+1
            n_proj = np.mod(nk,2)
            if aa[n_l]==0 and aa[n_r]==0 and n_proj^aa[nk]:
                bb[nk] = 1-bb[nk]
                numb = dec_10(bb)
                if numb in Psi1:
                    n_o = np.where(Psi1==numb)
                    V1[n_o] += V[nj]
                else:
                    Psi1 = np.append(Psi1,numb)
                    V1 = np.append(V1,V[nj])
    beta = np.linalg.norm(V1)
    V1 = V1/beta
    H[ni+1,ni] = beta
    Psi = np.copy(Psi1)
    V = np.copy(V1)
    n_H += len(Psi1)
        
H = H+H.T

HE = linalg.eigh(H)
E = HE[0]
W = HE[1]

plt.figure(1)
plt.plot(E,np.log(np.abs(W[0,:])**2),'rx')

        
        