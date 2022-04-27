#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import copy
import scipy.fftpack as fftpack
def VMD_for_1D_signal(f,alpha,tau,K,tol,Niter):
    ltemp = len(f)//2 
    fs=1./len(f)
    fMirr =  np.append(np.flip(f[:ltemp],axis = 0),f)  
    fMirr = np.append(fMirr,np.flip(f[-ltemp:],axis = 0))
    T = len(fMirr)
    t = np.arange(1,T+1)/T  
    freqs = t-0.5-(1/T)
    f_hat=np.zeros([1, K])
    f_hat = fftpack.fftshift((fftpack.fft(fMirr)))
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[:T//2] = 0
    omega_plus = np.zeros([1, K])
    lambda_hat = np.zeros([1, T], dtype = complex)
    uDiff = tol+np.spacing(1) 
    n = 0 
    sum_uk = 0
    u_hat_plus = np.zeros([1, T, K],dtype=complex)  
    u_hat_plus_cur=np.zeros([1, T, K],dtype=complex)  
    lambda_hat_cur=np.zeros([1, T], dtype = complex)
    omega_plus_cur=np.zeros([1, K])
    
    while ( uDiff > tol and  n < Niter-1 ): # not converged and below iterations limit
        k = 0
        sum_uk = u_hat_plus[:,:,K-1] + sum_uk - u_hat_plus[:,:,0]
        a=(f_hat_plus - sum_uk - lambda_hat[:]/2)
        b=(1.+alpha*(freqs - omega_plus[:,k])**2)
        u_hat_plus_cur[:,:,k]= a/b
        c1=abs(u_hat_plus_cur[:,T//2:T,k])**2
        d1=freqs[T//2:T]
        e1=np.dot(c1,d1.T)
        f1=np.sum(abs(u_hat_plus_cur[:,T//2:T,k])**2)
        omega_plus_cur[:,k] = e1/f1
        for k in np.arange(1,K):
            sum_uk = u_hat_plus_cur[:,:,k-1] + sum_uk - u_hat_plus[:,:,k]
            u_hat_plus_cur[:,:,k] = (f_hat_plus - sum_uk - lambda_hat[:]/2)/(1+alpha*(freqs - omega_plus[:,k])**2)
            c2=abs(u_hat_plus_cur[:,T//2:T,k])**2
            d2=freqs[T//2:T]
            e2=np.dot(c2,d2.T)
            f2=np.sum(abs(u_hat_plus_cur[:,T//2:T,k])**2)
            omega_plus_cur[:,k] = e2/f2
        lambda_hat_cur[:,:] = lambda_hat[:,:] + tau*(np.sum(u_hat_plus_cur[:,:,:],axis = 2) - f_hat_plus)
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1/T)*np.dot((u_hat_plus_cur[:,:,i]-u_hat_plus[:,:,i]),np.transpose(np.conj((u_hat_plus_cur[:,:,i]-u_hat_plus[:,:,i]))))

        uDiff = np.abs(uDiff)   
        #n=n+1
        u_hat_plus_pre=copy.deepcopy(u_hat_plus)
        lambda_hat_pre=copy.deepcopy(lambda_hat)
        omega_plus_pre=copy.deepcopy(omega_plus)
        u_hat_plus=copy.deepcopy(u_hat_plus_cur)
        lambda_hat=copy.deepcopy(lambda_hat_cur)
        omega_plus=copy.deepcopy(omega_plus_cur)
        n=n+1
    omega = omega_plus_pre
    idxs = np.flip(np.arange(1,T//2+1),axis = 0)
    u_hat = np.zeros([1, T, K],dtype=complex)
    u_hat[:,T//2:T,:] = u_hat_plus_pre[:,T//2:T,:]
    u_hat[:,idxs,:] = np.conj(u_hat_plus_pre[:,T//2:T,:])
    u_hat[:,0,:] = np.conj(u_hat_plus_pre[:,-1,:])   
    u = np.zeros([1,K,T])
    for k in range(K):
        u[:,k,:] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,:,k])))
    u = u[:,:,T//4:3*T//4]
    u_hat = np.zeros([f.shape[0],u.shape[2], K],dtype=complex)
    for k in range(K):
        u_hat[:,:,k] = fftpack.fftshift(fftpack.fft(u[:,k,:]), axes=(1,))
    return u, u_hat, omega

