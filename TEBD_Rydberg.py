# -*- coding: utf-8 -*-
"""
Created on Sat Aug 6 16:54:18 2022

@author: FanYang
"""

import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

###### Physical parameters ####
initial_state = 0  #### starting with the paramagnetic state |ggggg...>
#initial_state = 1  #### starting with the anti-ferromagnetic state |rgrgr...>

chi_max = 200 #### bond dimension of MPS
d = 2 #### dimension of the local spin
Omega = 2 #### Rabi-frequency
V = 10 #### setting Rydberg interaction strength, should be larger than Rabi-frequency to satisfy the blockade condition

t_end = 20 #### total evolution time
delta_t = 0.01 #### time step

N_tot = round(t_end/delta_t)
E_ground = np.zeros(N_tot)
MM = round(N_tot/10)

Mag1 = np.zeros_like(E_ground,dtype = complex) #### local Rydberg density <sigma_z+1>/2
Mag2 = np.copy(Mag1) #### local Rydberg density <sigma_z+1>/2
CSZ = np.copy(Mag1) #### correlation function <S_zS_{z+1}>
SE1 = np.copy(Mag1) #### entanglement entropy
SE2 = np.copy(Mag1) #### entanglement entropy

#### Definition of the Hamiltonian #####
Sz = np.array([[1,0],[0,-1]])
Sx = np.array([[0,1],[1,0]])
Sy = np.array([[0,-1j],[1j,0]])
II = np.eye(2)

H = V*np.kron((Sz+II)/2,(Sz+II)/2) + Omega/2*np.kron(Sx,II)

#### Time-evolution operator ####

HE = linalg.eig(H)
E = HE[0]
W = HE[1]
W_ = linalg.inv(W)
#U0 = np.diag(np.exp(-1j*E*delta_t))
U0 = np.diag(np.exp(-1j*E*delta_t))
U = W.dot(U0.dot(W_))

##### Definition of the tensors #####

lambda_a = np.zeros(1,dtype = complex)
lambda_a[0] = 1
Lambda_A = np.diag(lambda_a)
Lambda_B = np.copy(Lambda_A)

Gamma_A = np.zeros((d,1,1),dtype = complex)
Gamma_B = np.copy(Gamma_A)

Gamma_A[1,:,:] = 1
if initial_state==0:
    Gamma_B[1,:,:] = 1
else:
    Gamma_B[0,:,:] = 1

#### Time-evolution Updating Function TUF #####
def TUF(Lambda_A,Lambda_B,Gamma_A,Gamma_B,chi_max,U,d):
    
    Theta = np.tensordot(Lambda_B,Gamma_A,axes=(1,1))
    Theta = np.tensordot(Theta,Lambda_A,axes=(2,0))
    Theta = np.tensordot(Theta,Gamma_B,axes=(2,1))
    Theta = np.tensordot(Theta,Lambda_B,axes=(3,0))
    chi1,chi2 = len(Gamma_A[0,:,0]),len(Gamma_B[0,0,:])
    #### Updating Theta #####
    U = np.reshape(U,(d,d,d,d))
    Theta = np.tensordot(U,Theta,axes=([2,3],[1,2]))
    Theta = np.transpose(Theta,(0,2,1,3))
    Theta_re = np.reshape(Theta,(d*chi1,d*chi2))
    u,ss,vt = linalg.svd(Theta_re)
    s = np.diag(ss)
    chi_c = np.min([np.sum(ss>1e-10),chi_max])
    s = s[0:chi_c,0:chi_c]
    u = u[:,0:chi_c]
    vt = vt[0:chi_c,:]
    Theta_re = u.dot(s).dot(vt)
    N_re = np.sum(np.abs(Theta_re)**2)
    Lambda_A = s/np.sqrt(N_re)
    u = np.reshape(u,(d,chi1,chi_c))
    vt = np.reshape(vt,(chi_c,d,chi2))
    Gamma_A = np.tensordot(linalg.inv(Lambda_B),u,axes=(1,1))
    Gamma_A = np.transpose(Gamma_A,(1,0,2))
    Gamma_B = np.tensordot(vt,linalg.inv(Lambda_B),axes=(2,0))
    Gamma_B = np.transpose(Gamma_B,(1,0,2))
    return Lambda_A,Gamma_A,Gamma_B,N_re

#### step-by-step imaginary time evolution main function #####
for nt in range(N_tot):
    ER = TUF(Lambda_A,Lambda_B,Gamma_A,Gamma_B,chi_max,U,d)
    Lambda_A,Gamma_A,Gamma_B = ER[0],ER[1],ER[2]
    ER = TUF(Lambda_B,Lambda_A,Gamma_B,Gamma_A,chi_max,U,d)
    Lambda_B,Gamma_B,Gamma_A = ER[0],ER[1],ER[2]

    E_ground[nt] = -np.log(ER[-1])/2/delta_t
    
    AA = np.tensordot(Lambda_B,Gamma_A,axes=(1,1))
    AA = np.tensordot(AA,Lambda_A,axes=(2,0))
    AA = np.transpose(AA,(1,0,2))
    A2 = np.tensordot(np.conj(AA),AA,axes=([1,2],[1,2]))
    Mag1[nt] = np.tensordot((Sz+II)/2,A2,axes=([0,1],[0,1]))
    
    AA = np.tensordot(Lambda_A,Gamma_B,axes=(1,1))
    AA = np.tensordot(AA,Lambda_B,axes=(2,0))
    AA = np.transpose(AA,(1,0,2))
    A2 = np.tensordot(np.conj(AA),AA,axes=([1,2],[1,2]))
    Mag2[nt] = np.tensordot((Sz+II)/2,A2,axes=([0,1],[0,1]))
    
    
    Theta = np.tensordot(Lambda_B,Gamma_A,axes=(1,1))
    Theta = np.tensordot(Theta,Lambda_A,axes=(2,0))
    Theta = np.tensordot(Theta,Gamma_B,axes=(2,1))
    Theta = np.tensordot(Theta,Lambda_B,axes=(3,0))
    
    Sz_2 = np.kron(Sz,Sz)
    Sz_2 = np.reshape(Sz_2,(d,d,d,d))
    
    Theta_1 = np.tensordot(Sz_2,Theta,axes=([2,3],[1,2]))
    CSZ[nt] = np.tensordot(np.conj(Theta),Theta_1,axes=([0,1,2,3],[2,0,1,3]))
    
    if np.mod(nt,MM)==0:
        print(str(nt/N_tot*100)+'% complete')
        
    Lamb = np.diag(Lambda_A**2)
    SE1[nt] = -np.sum(Lamb*np.log(Lamb))
    Lamb = np.diag(Lambda_B**2)
    SE2[nt] = -np.sum(Lamb*np.log(Lamb))

T = np.linspace(0,t_end,N_tot)

plt.figure(1)
plt.plot(T,np.real(Mag1))

plt.figure(2)
plt.plot(T,np.real(Mag2))

plt.figure(4)
plt.plot(T,SE1)

plt.figure(5)
plt.plot(T,SE2)

plt.figure(6)
plt.plot(T,np.real(CSZ))
plt.ylim([-1.1,1.1])



