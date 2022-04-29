#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import copy
import torch
def VMD_for_nD_signal_torch(f,alpha,tau,K,tol,Niter):
    ltemp = f.shape[1]//2 
    fs=1./f.shape[1]
    fMirr =  torch.cat((torch.flip(f[:,:ltemp],dims = [1]),f),1)
    fMirr =  torch.cat((fMirr,torch.flip(f[:,-ltemp:],dims = [1])),1)
    T=fMirr.shape[1]
    t1 = torch.arange(1,T+1)/T  
    t=torch.tile(t1,(f.shape[0],1))
    freqs = t-0.5-(1/T)
    freqs=freqs.to("cuda")
    f_hat = torch.fft.fftshift((torch.fft.fft(fMirr)),dim=[1])
    f_hat_plus = torch.clone (f_hat)
    f_hat_plus[:,:T//2] = 0
    omega_plus = torch.zeros([f.shape[0], K]).cuda()
    lambda_hat_real = torch.zeros([f.shape[0], T]).cuda()
    lambda_hat_imag = torch.zeros([f.shape[0], T]).cuda()
    lambda_hat=torch.complex(lambda_hat_real,lambda_hat_imag)
    uDiff = tol+np.spacing(1)
    uDiff_max=uDiff
    n = 0 
    sum_uk = 0#
    u_hat_plus_real = torch.zeros([f.shape[0], T, K]).cuda()  
    u_hat_plus_imag = torch.zeros([f.shape[0], T, K]).cuda() 
    u_hat_plus=torch.complex(u_hat_plus_real,u_hat_plus_imag)
    u_hat_plus_cur=torch.clone(u_hat_plus)
    lambda_hat_cur=torch.clone(lambda_hat)
    omega_plus_cur=torch.clone(omega_plus)
    while ( uDiff_max > tol and  n < Niter-1 ):
        k = 0
        sum_uk = u_hat_plus[:,:,K-1] + sum_uk - u_hat_plus[:,:,0]
        a=(f_hat_plus - sum_uk - lambda_hat[:]/2)
        b=(1.+alpha*(freqs - torch.tile(omega_plus[:,k].reshape(f.shape[0],1),(1,T)))**2)
        u_hat_plus_cur[:,:,k]= a/b
        c1=abs(u_hat_plus_cur[:,T//2:T,k])**2
        d1=freqs[:,T//2:T]
        e1=torch.diagonal(torch.matmul(c1,d1.t()))
        f1=torch.sum(abs(u_hat_plus_cur[:,T//2:T,k])**2,1)
        omega_plus_cur[:,k] = e1/f1
        for k in np.arange(1,K):
            sum_uk = u_hat_plus_cur[:,:,k-1] + sum_uk - u_hat_plus[:,:,k]
            a=(f_hat_plus - sum_uk - lambda_hat[:]/2)
            b=(1.+alpha*(freqs - torch.tile(omega_plus[:,k].reshape(f.shape[0],1),(1,T)))**2)
            u_hat_plus_cur[:,:,k]= a/b
            c2=abs(u_hat_plus_cur[:,T//2:T,k])**2
            d2=freqs[:,T//2:T]
            e2=torch.diagonal(torch.matmul(c2,d2.t()))
            f2=torch.sum(abs(u_hat_plus_cur[:,T//2:T,k])**2,1)
            omega_plus_cur[:,k] = e2/f2
        lambda_hat_cur[:,:] = lambda_hat[:,:] + tau*(torch.sum(u_hat_plus_cur[:,:,:],dim=[2]) - f_hat_plus)
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1/T)*torch.matmul((u_hat_plus_cur[:,:,i]-u_hat_plus[:,:,i]),(torch.conj((u_hat_plus_cur[:,:,i]-u_hat_plus[:,:,i])).t()))
        uDiff =torch.abs(uDiff)
        uDiff_max=min(torch.diagonal(uDiff))
        u_hat_plus_pre=torch.clone(u_hat_plus)
        lambda_hat_pre=torch.clone(lambda_hat)
        omega_plus_pre=torch.clone(omega_plus)
        u_hat_plus=torch.clone(u_hat_plus_cur)
        lambda_hat=torch.clone(lambda_hat_cur)
        omega_plus=torch.clone(omega_plus_cur)
        n=n+1
    omega = omega_plus_pre
    idxs = torch.flip(torch.arange(1,T//2+1),dims = [0])
    idxs=idxs.to("cuda")
    u_hat_real = torch.zeros([f.shape[0], T, K]).cuda()
    u_hat_imag = torch.zeros([f.shape[0], T, K]).cuda()
    u_hat=torch.complex(u_hat_real,u_hat_imag)

    u_hat[:,T//2:T,:] = u_hat_plus_pre[:,T//2:T,:]
    u_hat[:,idxs,:] = torch.conj(u_hat_plus_pre[:,T//2:T,:])
    u_hat[:,0,:] = torch.conj(u_hat_plus_pre[:,-1,:])   
    u = torch.zeros([f.shape[0],K,T]).cuda()
    for k in range(K):
        u[:,k,:] = torch.real(torch.fft.ifft(torch.fft.ifftshift(u_hat[:,:,k],dim=1)))
    u = u[:,:,T//4:3*T//4]

    u_hat_real = torch.zeros([f.shape[0],u.shape[2], K]).cuda()
    u_hat_imag = torch.zeros([f.shape[0],u.shape[2], K]).cuda()
    u_hat=torch.complex(u_hat_real,u_hat_imag)

    for k in range(K):
        u_hat[:,:,k] = torch.fft.fftshift(torch.fft.fft(u[:,k,:]),dim=1)
    return u, u_hat, omega