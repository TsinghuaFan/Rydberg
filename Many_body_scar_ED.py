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

N = 18 #### number of atoms

n_1 = 1
n_2 = 1
for nn in range(N-1):
    n_0 = n_2*1
    n_2 = n_1*1
    n_1 = n_1 + n_0
n_H = n_1 + n_0

aa = np.zeros(N,dtype = int)
for ni in range(N):
    if np.mod(ni,2)==0:
        aa[ni] = 1
    else:
        aa[ni] = 0
Psi = np.array([],dtype = int)
Psi = np.append(Psi,dec_10(aa))

n_pre = 0
H = np.zeros((n_H,n_H))
Psi1 = np.copy(Psi)
Row = np.array([],dtype = int)
Row1 = np.array([],dtype = int)
Col = np.array([],dtype = int)
Col1 = np.array([],dtype = int)
for ni in range(N+1):
    for nj in range(len(Psi)-n_pre):
        aa = dec_2(Psi[nj+n_pre],N)
        row = aa[0:round(N/2)]
        col = aa[round(N/2)::]
        row1 = dec_10(row)
        col1 = dec_10(col)
        if row1 in Row1:
            n_row = np.where(Row1==row1)
            Row = np.append(Row,n_row)
        else:
            Row1 = np.append(Row1,row1)
            Row = np.append(Row,len(Row1)-1)
        if col1 in Col1:
            n_col = np.where(Col1==col1)
            Col = np.append(Col,n_col)
        else:
            Col1 = np.append(Col1,col1)
            Col = np.append(Col,len(Col1)-1)
            
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
                    H[n_o,nj+n_pre] = 1
#                    print('no')
                else:
                    Psi1 = np.append(Psi1,numb)
                    H[len(Psi1)-1,nj+n_pre] = 1
#                    print('yes')
    n_pre = len(Psi)
    Psi = np.copy(Psi1)

H = H+H.T

HE = linalg.eigh(H)
E = HE[0]
W = HE[1]

plt.figure(1)
plt.plot(E,np.log(np.abs(W[0,:])**2),'s')
plt.ylim([-16,0])

Psi_ab = np.zeros((len(Row1),len(Col1)))
SE = E*0
for ne in range(n_H):
    for nn in range(n_H):
        Psi_ab[Row[nn],Col[nn]] = W[nn,ne]
    u,ss,vt = np.linalg.svd(Psi_ab)
    Namb = ss**2
    se = -np.sum(Namb*np.log(Namb))
    SE[ne] = se

plt.figure(2)
plt.plot(E,SE,'o')





