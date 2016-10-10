
# -*- coding: utf-8 -*-
"""
This module contains different state estimation approaches related to the Kalman filter.


@author: Sascha Eichstaedt
original MATLAB code by Wiebke Heins (TU Clausthal)

"""

import numpy as np
from scipy.sparse import issparse

if __name__=="NLO.nodal_load_observer": # module is imported from within package
	from tools.data_tools import process_admittance, separate_Yslack, makeYbus
else:
	from tools.data_tools import process_admittance, separate_Yslack, makeYbus


def get_system_matrices(pmeas,qmeas,vmeas):
	"""Construction of matrices Cm, Dm and Dnm which map all power/voltage values to the actual
	measured/non-measured ones.

	:param pmeas: (n_K,) shaped array of booleans indicating measurement positions of active power
	:param qmeas: (n_K,) shaped array of booleans indicating measurement positions of reactive power
	:param vmeas: (n_K,) shaped array of booleans indicating measurement positions of voltage
	"""
	assert(len(pmeas)==len(qmeas))
	# assert(len(qmeas)==len(vmeas))

	n_K = len(pmeas)

# construction of matrix Cm
	m = vmeas.nonzero()[0]
	r = len(m)
	Cm = np.zeros((2*r,2*n_K))
	for i in range(r):
		Cm[i,m[i]] = 1
		Cm[r+i,n_K+m[i]] = 1

# construction of matrix Dm
	pm = len(pmeas.nonzero()[0])
	qm = len(qmeas.nonzero()[0])
	Dm = np.zeros((2*n_K, pm+qm),dtype=int)  # (Eq.4.78) in WH thesis
	for k,ind in enumerate(pmeas.nonzero()[0]):
		Dm[ind,k] = 1
	for k,ind in enumerate(qmeas.nonzero()[0]):
		Dm[n_K+ind,pm+k] = 1

# construction of matrix Dnm
	pnmeas = -pmeas[:]
	qnmeas = -qmeas[:]
	pnm = len(pnmeas.nonzero()[0])
	qnm = len(qnmeas.nonzero()[0])
	Dnm = np.zeros((2*n_K, pnm+qnm))  # (Eq.4.78) in WH thesis
	for k,ind in enumerate(pnmeas.nonzero()[0]):
		Dnm[ind,k] = 1
	for k,ind in enumerate(qnmeas.nonzero()[0]):
		Dnm[n_K+ind,pnm+k] = 1
	return Cm, Dnm, Dm




def jacobian(Y00,Ys,V,Vs):
	"""
	Calculation of jacobian matrix for the extended Kalman filter
	"""
	n_K = Y00.shape[0]/2
	G = Y00[:n_K,:n_K]
	B = Y00[n_K:,:n_K]
	Gs= Ys[:n_K,0]
	Bs= Ys[n_K:,0]

	n = V.shape[0]/2

	VsRe = Vs[0]
	VsIm = Vs[1]

	VRe = V[:n]
	VIm = V[n:]

	# Dh = [[H,N],[M,L]]
	H = np.zeros((n,n))
	N = np.zeros((n,n))

	# outer diagonal elements of H
	for i in range(n):
		for j in range(n):
			H[i,j] = 3*VRe[i]*G[i,j] + 3*VIm[i]*B[i,j]
	# diagonal elements
	for i in range(n):
		H[i,i] = 3*np.dot(G[i,:],VRe)- 3*np.dot(B[i,:],VIm) + 3*G[i,i]*VRe[i] + 3*B[i,i]*VIm[i] + 3*Gs[i]*VsRe - 3*Bs[i]*VsIm

	# outer diagonal elements of N
	for i in range(n):
		for j in range(n):
			N[i,j] = -3*VRe[i]*B[i,j] + 3*VIm[i]*G[i,j]
	# diagonal elements
	for i in range(n):
		N[i,i] = 3*np.dot(B[i,:],VRe) + 3*np.dot(G[i,:],VIm) - 3*B[i,i]*VRe[i] + 3*G[i,i]*VIm[i] + 3*Bs[i]*VsRe + 3*Gs[i]*VsIm

	# outer diagonal elements of M
	M = N.copy()
	# diagonal elements of M
	for i in range(n):
		M[i,i] = -3*np.dot(B[i,:],VRe) - 3*np.dot(G[i,:],VIm) -3*B[i,i]*VRe[i] + 3*G[i,i]*VIm[i] - 3*Bs[i]*VsRe - 3*Gs[i]*VsIm

	# outer diagonal elements of L
	L = -H.copy()
	# diagonal elements of L
	for i in range(n):
		L[i,i] = 3*np.dot(G[i,:],VRe) -3*np.dot(B[i,:],VIm) - 3*B[i,i]*VIm[i] - 3*G[i,i]*VRe[i] + 3*Gs[i]*VsRe - 3*Bs[i]*VsIm

	Dh = np.vstack((np.hstack((H,N)),
					np.hstack((M,L))))
	return Dh






def LinearKalmanFilter(topology, meas, meas_unc, meas_idx, pseudo_meas, model, V0,
						   Vs, slack_idx=0, Y=None):
	"""
	Quasi-Linear Kalman filter for the nodal load observer
	This version of the NLO state estimation method ignores the nonlinearity for the calculation of the
	Kalman gain and utilizes the fix point equation property for the calculation of voltages from powers.
	Estimated nodal voltages are returned as real and imaginary parts

	Real-valued matrices of complex-valued quantities are assumed to be structured as [ [real part], [imag part] ]
	:param topology: dict containing information on bus, branch, ... in PyPower format
	:param meas: dict containing measurements "Pk", "Qk", "Vm" and "Va"
	:param meas_unc: dict containing associated uncertainties
	:param meas_idx: dict containing the corresponding indices
	:param pseudo_meas: dict containing the corresponding pseudo-measurements
	:param model: DynamicModel object as defined in dynamic_models.py
	:param V0: initial estimate of nodal voltages (magnitude and phase)
	:param Vs: voltages at slack node (magnitude and phase)
	:param slack_idx: index of slack node
	:param Y: (optional) user defined admittance matrix

	"""
	if issparse(Y):
		Y = Y.toarray()

	if not isinstance(Y,np.ndarray):
		Y = makeYbus(topology["baseMVA"],topology["bus"],topology["branch"])
	Yadm, Y_slack = separate_Yslack(Y,slack_idx)

	nK = len(V0)/2
	pmeas = np.zeros(nK,dtype = bool); pmeas[meas_idx["Pk"]] = True
	qmeas = np.zeros(nK,dtype = bool); qmeas[meas_idx["Qk"]] = True
	vmeas = np.zeros(nK,dtype = bool); vmeas[meas_idx["Vm"]] = True
	Cm,Dnm,Dm = get_system_matrices(pmeas,qmeas,vmeas)

	# transform voltages at slack node to real and imaginary parts
	Vs_ri = np.vstack((Vs[0,:]*np.cos(Vs[1,:]),
					   Vs[0,:]*np.sin(Vs[1,:])))
	Slack = np.linalg.solve(Yadm,np.dot(Y_slack,Vs_ri))
	# y = np.r_[meas["Vm"], meas["Va"]] + np.dot(Cm,Slack)

	S = np.dot(Dm, np.r_[meas["Pk"], meas["Qk"]]) + np.dot(Dnm, np.r_[pseudo_meas["Pk"], pseudo_meas["Qk"]])

	# adjust uncertainties in case that their dimension is wrong
	if isinstance(meas_unc["Vm"],float):
		meas_unc["Vm"] = meas_unc["Vm"]*np.ones_like(meas["Vm"])
	elif len(meas_unc["Vm"].shape)==1:
		meas_unc["Vm"] = np.tile(meas_unc["Vm"],(meas["Vm"].shape[1],1)).T

	if isinstance(meas_unc["Va"],float):
		meas_unc["Va"] = meas_unc["Va"]*np.ones_like(meas["Va"])
	elif len(meas_unc["Va"].shape)==1:
		meas_unc["Va"] = np.tile(meas_unc["Va"],(meas["Va"].shape[1],1)).T

	nx = model.dim
	nm = 2*meas["Vm"].shape[0]
	t_f= meas["Vm"].shape[1]

	def calcM(mu):
		divisor = 3*(mu[:nK]**2 + mu[nK:]**2)
		return np.r_[np.c_[np.diag(mu[:nK]/divisor), np.diag(mu[nK:]/divisor)],
				     np.c_[np.diag(mu[nK:]/divisor), -np.diag(mu[:nK]/divisor)]]

	def calcKs(V,Sh,k,accuracy=1e-12):
		# According to W. Heins' Thesis calculation of V using Ks is a fix point equation
		# We take that into account by doing a fixed number of iterations of the corresponding
		# fix point iterations
		Vn = V.copy()
		fp_diff = 1.0
		maxiter = 20
		count = 0
		while fp_diff>accuracy and count < maxiter:
			tmp = Vn
			MU = calcM(Vn)
			Ks = np.linalg.solve(Yadm,MU)
			Vn = np.dot(Ks, np.dot(Dnm,Sh) + S[:,k]) - Slack[:,k]
			fp_diff = np.linalg.norm(tmp-Vn)
			count += 1
		return Ks


	P = model.forecast_unc()

# initialize solution matrices
	V_est = np.zeros((2*nK,t_f+1))     # Estimated voltages
	V_est[:nK,0] = V0[:nK]*np.cos(V0[nK:])
	x_est = np.zeros((nx,t_f+1))          # Estimated active and reactive powers
	x_est[:,0] = model.forecast_state()
	DeltaS_est = np.zeros((nx,t_f))     # Estimated power deviation
	UncDeltaS = np.zeros_like(DeltaS_est)
	Dnm = model.adjust_Dnm(Dnm)

#%% ########################### KALMAN ########################################
	print '.',
	for k in range(1,t_f+1):
		# transform voltages to real and imaginary parts
		yRe,yIm,R = amph_phase_to_real_imag(meas["Vm"][:,k-1],np.radians(meas["Va"][:,k-1]),meas_unc["Vm"][:,k-1]**2,meas_unc["Va"][:,k-1]**2)
		y = np.r_[yRe,yIm] + np.dot(Cm,Slack[:,k-1])
	# preparation of state space system matrices
		Ks = calcKs(V_est[:,k-1],x_est[:,k-1],k-1)
		C = np.dot(Cm,np.dot(Ks,Dnm))
		D = np.dot(Cm,Ks)
#========================== actual Kalman filter part =========================
	#  Kalman filter forecast step
		xf = model.forecast_state(x_est[:,k-1])[:,np.newaxis]
		Pf = model.forecast_unc(P)
	# Kalman gain matrix K
		K =  np.linalg.solve(np.dot(C,np.dot(Pf,C.T)) + R, np.dot(C,P)).T
	# corrected state estimate
		x_est[:,k][:,np.newaxis] = xf + np.dot(K, y.reshape(nm,1)
									  - (np.dot(C,xf) + np.dot(D,S[:,k-1].reshape(2*nK,1))) )
	# corrected error covariance matrix
		P = np.dot(np.eye(nx) - np.dot(K,C),Pf)
#==============================================================================
	# calculate voltage from estimated power
		V_est[:,k] = np.dot(Ks, np.dot(Dnm,x_est[:,k-1]) + S[:,k-1]) - Slack[:,k-1]
		DeltaS_est[:,k-1] = x_est[:,k]
		UncDeltaS[:,k-1] = np.sqrt(np.diag(P))
		print '.',
	print '.'
#%%

  # results
	S_est = S + np.dot(Dnm,DeltaS_est) # S_est = S + D_ng * DeltaS_est
	UncS  = np.zeros_like(S) + np.dot(Dnm,UncDeltaS)
	return S_est, V_est[:,1:], UncS, DeltaS_est,UncDeltaS



def IteratedExtendedKalman(topology, meas, meas_unc, meas_idx, pseudo_meas, model, V0,
						   Vs,slack_idx=0, Y=None, accuracy=1e-9, maxiter=50):
	"""
	Iterated Extended Kalman filter for the nodal load observer
	Real-valued matrices of complex-valued quantities are assumed to be structured as [ [real part], [imag part] ]

	:param topology: dict containing information on bus, branch, ... in PyPower format
	:param meas: dict containing measurements "Pk", "Qk", "Vm" and "Va"
	:param meas_unc: dict containing uncertainties with the values in `meas'
	:param meas_idx: dict containing the corresponding indices
	:param pseudo_meas: dict containing the corresponding pseudo-measurements
	:param model: DynamicModel object as defined in dynamic_models.py
	:param V0: initial estimate of nodal voltages at all nodes except slack
	:param Vs: voltage amplitude and phase at slack node as (2,nT)-dimensional vector
	:param slack_idx: index of slack node
	:param Y: (optional) admittance matrix
	:param accuracy: threshold for inner iteration of the iterated EKF
	:param maxiter: maximum number of inner iterations of the iterated EKF

	:return: Shat, Vhat, uShat, DeltaS, uDeltaS
	"""
	if issparse(Y):
		Y = Y.toarray()
	n = model.dim
	n_K = len(V0)/2
	pmeas = np.zeros(n_K,dtype = bool); pmeas[meas_idx["Pk"]] = True
	qmeas = np.zeros(n_K,dtype = bool); qmeas[meas_idx["Qk"]] = True
	vmeas = np.zeros(n_K,dtype = bool); vmeas[meas_idx["Vm"]] = True
	Cm,Dnm,Dm = get_system_matrices(pmeas,qmeas,vmeas)
	if not isinstance(Y,np.ndarray):
		Y = makeYbus(topology["baseMVA"],topology["bus"],topology["branch"])
	Y00, Ys = separate_Yslack(Y,slack_idx)

	# transform voltages at slack node to real and imaginary parts
	Vs_ri = np.vstack((Vs[0,:]*np.cos(np.radians(Vs[1,:])),
					   Vs[0,:]*np.sin(np.radians(Vs[1,:]))))
	Slack = np.linalg.solve(Y00,np.dot(Ys,Vs_ri))
	Sm = np.r_[meas["Pk"], meas["Qk"]]
	Sfc = np.r_[pseudo_meas["Pk"], pseudo_meas["Qk"]]
	u = np.dot(Dm,Sm) + np.dot(Dnm,Sfc)
 	#np.savetxt("/Users/makara01/Documents/oioi/Sm.out", u)

	# adjust uncertainties in case that their dimension is wrong
	if isinstance(meas_unc["Vm"],float):
		meas_unc["Vm"] = meas_unc["Vm"]*np.ones_like(meas["Vm"])
	elif len(meas_unc["Vm"].shape)==1:
		meas_unc["Vm"] = np.tile(meas_unc["Vm"],(meas["Vm"].shape[1],1)).T

	if isinstance(meas_unc["Va"],float):
		meas_unc["Va"] = meas_unc["Va"]*np.ones_like(meas["Va"])
	elif len(meas_unc["Va"].shape)==1:
		meas_unc["Va"] = np.tile(meas_unc["Va"],(meas["Va"].shape[1],1)).T

	def calcM(mu):
		divisor = 3*(mu[:n_K]**2 + mu[n_K:]**2)
		return np.r_[np.c_[np.diag(mu[:n_K]/divisor), np.diag(mu[n_K:]/divisor)],
				     np.c_[np.diag(mu[n_K:]/divisor), -np.diag(mu[:n_K]/divisor)]]
         
	nT = Vs.shape[1]
	xhat = np.zeros((n,nT))
	Vhat = np.zeros((2*n_K,nT))
	Shat = np.zeros((2*n_K,nT))
	DeltaS = np.zeros_like(xhat)
	uDeltaS= np.zeros_like(xhat)
	Pfilter = model.forecast_unc()
	Dnm = model.adjust_Dnm(Dnm)
	phi0=np.tile(np.array([model.parameters["phi"][0],model.parameters["phi"][1]]), (n/2,1))
	Beta=np.zeros((n/2,2,nT+1))
	sigma = np.zeros((n/2,nT+1))
	sigma_0 = model.parameters["noise"]* np.ones((n/2))
   	la=np.zeros((nT+1))     
	Psi = np.zeros((n/2,nT,2))
	e=np.zeros((n/2,nT))
	gamma =  np.zeros((n/2,nT+1))
	F=np.eye(2) 
	sigma[:,0]=sigma_0[:] # already squared
	la[0] = 1.0
	gamma[:,0] = np.ones((n/2))  
     
     
	for k in range(nT):
		# transform voltages to real and imaginary parts
		yRe,yIm,R = amph_phase_to_real_imag(meas["Vm"][:,k],np.radians(meas["Va"][:,k]),meas_unc["Vm"][:,k]**2,meas_unc["Va"][:,k]**2)
		
		y = np.r_[yRe,yIm]
		if k==0:
			xhatfc = model.forecast_state()
			Pfilterfc = model.forecast_unc()
			mu = np.hstack((V0[:n_K]*np.cos(V0[n_K:]),V0[:n_K]*np.sin(V0[n_K:])))
			Beta[:,:,k]=np.hstack((phi0[:,0].reshape(n/2,1),phi0[:,1].reshape(n/2,1)))
			
   
		else:


			A11 = Beta[:n/2,0,k]*np.eye(2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]))
			A12 = Beta[:n/2,1,k]*np.eye(2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]))
			A21 = np.ones((n/2))*np.eye(2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]))
			A22 = np.zeros((n/2))*np.eye(2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]))
			A=np.vstack((np.hstack((A11,A12)),np.hstack((A21,A22)))) 
    
			Q11 = sigma[:n/2,k]*np.eye(2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]))
			Q12 = np.zeros((2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]),2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"])))
			Q21 = np.zeros((2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]),2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"])))
			Q22 = np.zeros((2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]),2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"])))
			Q=np.vstack((np.hstack((Q11,Q12)),np.hstack((Q21,Q22))))
			xhatfc = np.dot(A,xhat[:,k-1])
			Pfilterfc = np.dot(A,np.dot(Pfilter,A.T)) + Q
			mu = Vhat[:,k-1]

		eta = xhatfc
		varstop1 = 1
		j1 = 1
		while (varstop1 > accuracy) and (j1 < maxiter):
			M_U = calcM(mu)
			muiter = np.dot(np.dot(np.linalg.inv(Y00),M_U),u[:,k] + np.dot(Dnm,eta)) - Slack[:,k] 
           
			varstop2 = 1
			j2 = 1
			while (varstop2 > accuracy) and (j2 < maxiter):
				M_U = calcM(muiter)
				temp2 = muiter
				muiter = np.dot(np.dot(np.linalg.inv(Y00),M_U),u[:,k] + np.dot(Dnm,eta)) - Slack[:,k]
				varstop2 = np.linalg.norm(muiter-temp2)
				j2 += 1
			mu = muiter
			Dh = jacobian(Y00,Ys,mu,Vs_ri[:,k])
			H = np.dot(Cm, np.dot(np.linalg.inv(Dh), Dnm))
			K = np.dot(np.dot(Pfilterfc,H.T), np.linalg.pinv(np.dot(H,np.dot(Pfilterfc,H.T))+R))
			temp1 = eta
			eta = xhatfc + np.dot(K, y - np.dot(Cm,mu) - np.dot(H, xhatfc-eta))
			varstop1 = np.linalg.norm(temp1-eta)
			j1 += 1
		# Data assimilation step
		Pfilter = np.dot( np.eye(n) - np.dot(K,H), Pfilterfc)
		xhat[:,k] = eta
		Shat[:,k] = u[:,k] + np.dot(Dnm,xhat[:,k])
		Vhat[:,k] = mu
		DeltaS[:,k-1] = xhat[:,k]
		uDeltaS[:,k-1] = np.sqrt(np.diag(Pfilter))
 
		la[k+1]=0.99*la[k]+1-0.99 
		gamma[:,k+1] = gamma[:,k]/(la[k+1]+gamma[:,k])

		Psi[:,k,:] = np.dstack((xhat[::2,k-1],xhat[::2,k-2]))     
		psi = Psi[:,k,:].squeeze()
		beta = Beta[:,:,k].squeeze()      
		e = xhat[::2,k]-np.diag(np.dot(beta, psi.T))
		beta += np.dot(np.diag(gamma[:,k+1]/sigma[:,k] *e), (np.dot(F, psi.T)).T ) 
		sigma[:,k+1]=sigma[:,k]+gamma[:,k+1]*(e**2-sigma[:,k])  
		Beta[:,:,k+1] = beta                         
		A11 = Beta[:n/2,0,k+1]*np.eye(2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]))
		A12 = Beta[:n/2,1,k+1]*np.eye(2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]))
		A21 = np.ones((n/2))*np.eye(2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]))
		A22 = np.zeros((n/2))*np.eye(2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]))
		A=np.vstack((np.hstack((A11,A12)),np.hstack((A21,A22)))) 
    
		Q11 = sigma[:n/2,k+1]*np.eye(2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]))
		Q12 = np.zeros((2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]),2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"])))
		Q21 = np.zeros((2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]),2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"])))
		Q22 = np.zeros((2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"]),2*n_K-np.size(meas_idx["Pk"])-np.size(meas_idx["Qk"])))
		Q=np.vstack((np.hstack((Q11,Q12)),np.hstack((Q21,Q22))))
	uS  = np.dot(Dnm,uDeltaS)
	return Shat, Vhat, uS, DeltaS, uDeltaS
 

 
 
 
 
#def IteratedExtendedKalman_moi(y,Sm,R,Yws,Vs,iterstop=50,accuracy=1e-9):
#        
#        p0=1.0
#        P = p0*np.eye(nx) 
#        Y00,Ys = separate_Yslack(Yws,slack_idx,dcomplex=False,prec=roundY)
#        Slack = np.linalg.solve(Y00,np.dot(Ys,Vs)).__array__()
#
#    
#        muiter=np.zeros((nm))
#        temp1 = np.zeros((nx))
#        temp2 = np.zeros((nm))
#        V_est = np.zeros((2*n_K,t_f+1)) 
#        v0 = 11/(np.sqrt(3))*np.hstack((np.ones(n_K), np.zeros(n_K)))
#        
#        if isinstance(v0,np.ndarray):
#            V_est[:,0] = v0
#        else:
#            V_est[:n_K,0] = v0
#        x_est=np.zeros((nx,t_f+1))         
#            
#        mu = np.hstack((v0[:n_K]*np.cos(v0[n_K:]),v0[:n_K]*np.sin(v0[n_K:])))
#        Pf = P
#        
#        A11 = Beta[:nx/2,0,0]*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
#        A12 = Beta[:nx/2,1,0]*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
#        A21 = np.ones((nx/2))*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
#        A22 = np.zeros((nx/2))*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
#        A=np.vstack((np.hstack((A11,A12)),np.hstack((A21,A22)))) # (2*(n_k-1),2*(n_k-1))
#                
#        Q11 = sigma[:nx/2,0]*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
#        Q12 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
#        Q21 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
#        Q22 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
#        Q=np.vstack((np.hstack((Q11,Q12)),np.hstack((Q21,Q22)))) # (2*(n_k-1),2*(n_k-1))
#        for k in range(1,t_f+1):                    
#                    xf = np.dot(A,x_est[:,k-1])
#                    Pf = np.dot(A,np.dot(P,A.T)) + Q
#                    mu = V_est[:,k-1]
#                    eta = xf
#                    est_error1 = 1
#                    counter1 = 1   
#                    while(est_error1 > accuracy) and (counter1 < iterstop):
#                            MU= calcM(mu)
#                            muiter =np.dot(np.dot(np.linalg.inv(Y00),MU),Sm[:,k-1]+np.dot(Dnm,eta)) - Slack[:,k-1]
#                            est_error2 = 1
#                            counter2 = 1
#                            while  (est_error2 > accuracy) and (counter2 < iterstop):
#                                    MU= calcM(muiter)
#                                    temp2 = muiter
#                                    muiter =np.dot(np.dot(np.linalg.inv(Y00),MU),Sm[:,k-1]+np.dot(Dnm,eta)) - Slack[:,k-1]
#                                    est_error2 = np.linalg.norm(muiter-temp2)
#                                    counter2 += 1
#                                   
#                            mu = muiter     
#                            Dh = jacobian(Y00,Ys,mu,Vs[:,k-1])
#                            H = np.dot(Cm,np.dot(np.linalg.inv(Dh),Dnm))
#                            K=np.dot(np.dot(Pf,H.T),np.linalg.pinv(np.dot(H,np.dot(Pf,H.T))+R))
#                            temp1 = eta
#                            eta = xf + np.dot(K,y[:,k-1] - np.dot(Cm,mu) - np.dot(H,xf-eta))
#                            est_error1 = np.linalg.norm(temp1-eta)
#                            counter1 += 1
#        
#                    x_est[:,k]= eta
#                    P = np.dot(np.eye(nx)-np.dot(K,H),Pf)
#                    V_est[:,k] = mu
#                    DeltaS_est[:,k-1] = x_est[:,k]
#                    UncDeltaS[:,k-1] = np.sqrt(np.diag(P))
#                    S_est[:,k-1] = Sm[:,k-1] + np.dot(Dnm,DeltaS_est[:,k-1]) # S_est = S + D_ng * DeltaS_est
#                    
#                    la[k]=0.99*la[k-1]+1-0.99
#                    gamma[:,k] = gamma[:,k-1]/(la[k]+gamma[:,k-1])
#                    #gamma[:,k] = 1.0/(k+0.1)   
#                    Psi[:,k-1,:] = np.dstack((x_est[::2,k-2],x_est[::2,k-3]))     
#                    psi = Psi[:,k-1,:].squeeze()
#                    beta = Beta[:,:,k-1].squeeze()      
#                    e = x_est[::2,k-1]-np.diag(np.dot(beta, psi.T))
#                    beta += np.dot(np.diag(gamma[:,k]/sigma[:,k-1] *e), (np.dot(F, psi.T)).T ) 
#                    sigma[:,k]=sigma[:,k-1]+gamma[:,k]*(e**2-sigma[:,k-1])  
#                    Beta[:,:,k] = beta                         
#                    
#                    A11 = Beta[:nx/2,0,k]*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
#                    A12 = Beta[:nx/2,1,k]*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
#                    A21 = np.ones((nx/2))*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
#                    A22 = np.zeros((nx/2))*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
#                    A=np.vstack((np.hstack((A11,A12)),np.hstack((A21,A22)))) 
#
#                    Q11 = sigma[:nx/2,k]*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
#                    Q12 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
#                    Q21 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
#                    Q22 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
#                    Q=np.vstack((np.hstack((Q11,Q12)),np.hstack((Q21,Q22))))
#        return S_est, x_est, Beta, sigma  
 
 

def amph_phase_to_real_imag(A,P,Ua,Up):
	"""
	Transforms uncertainties associated with amplitude A and phase P to the corresponding real and imaginary parts
	including the evaluation of associated standard uncertainties (ignoring correlations).

	For more details see
		Eichstädt, S. and Wilkens, V. "GUM2DFT – A software tool for uncertainty evaluation of transient signals in the frequency domain"
		Metrologia 2016

	:param A: ndarray of amplitude values; shape (N,)
	:param P: ndarray of phase values; shape (N,)
	:param UAP: ndarray of uncertainties associated with A and P; shape (2xN,)
	:return: Re, Im, URI
	"""
	from scipy import sparse
	assert(len(A.shape)==1)
	assert(A.shape==P.shape)
	assert(Ua.shape==A.shape)
	assert(Up.shape==P.shape)
	# calculation of F
	Re = A*np.cos(P)
	Im = A*np.sin(P)
	# calculation of sensitivities
	CRA = np.cos(P)
	CRP = -A*np.sin(P)
	CIA = np.sin(P)
	CIP = A*np.cos(P)

	# assignment of uncertainty blocks in UAP
	N = len(A)

	U11 = CRA*Ua*CRA + CRP*Up*CRP
	U12 = CRA*Ua*CIA + CRP*Up*CIP
	U22 = CIA*Ua*CIA + CIP*Up*CIP
	URI = sparse.diags([np.r_[U11,U22],U12,U12],[0,N,-N]).toarray()
	return Re, Im, URI

