# vmdpy for nD signal: Variational-Modal-Decomposition-for-1D-and-nD-signal in Python

Function for decomposing a signal according to the Variational Mode Decomposition ([Dragomiretskiy and Zosso, 2014](https://doi.org/10.1109/TSP.2013.2288675)) method.  

This demo is a improvement of https://github.com/vrcarva/vmdpy

I noticed that some matrixes in the original vmdpy.py(https://github.com/vrcarva/vmdpy) file used redundant dimension
like 
omega_plus = np.zeros([Niter, K])
lambda_hat = np.zeros([Niter, len(freqs)], dtype = complex)
u_hat_plus = np.zeros([Niter, len(freqs), K],dtype=complex)   
Actually for the final results we want ,the iteration matrix is not needed,so I modify the code and remove the Niter dimension,which reduce the memory useage.

Like the same thought in 1D signal
Simplify the dimension and discard the iteration matrix dimension
But there are some varieties for multiple-dimension array usage in nD situation 
