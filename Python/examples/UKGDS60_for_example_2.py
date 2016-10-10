# -*- coding: utf-8 -*-
"""

# Remark
The network is such that although the actual slack node is 301, also the subsequent node 1100 has to be
taken out from the estimation, because 301 is on the upper network level.

"""

from pypower.ext2int import ext2int
from pypower.idx_brch import F_BUS
from pypower.idx_brch import T_BUS
from pypower.idx_bus import BUS_I
from pypower.ppoption import ppoption
from pypower.api import runpf

import numpy as np
from tools.load import convert_to_python_indices
from scipy.io import loadmat

def get_topology():
	from numpy import asarray
	mpc=dict()
	mpc["version"]='2'
	mpc["baseMVA"]=100
	mpc["bus"]=asarray(
	[  [  3.010e+02,   3.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   1.000e+00,   3.060e+00,   0.000e+00,   3.300e+01,   1.000e+00,   1.030e+00,   9.700e-01],
	   [  1.100e+03,   1.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
	   [  1.115e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
	   [  1.116e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
	   [  1.117e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
	   [  1.118e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
	   [  1.119e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
	   [  1.120e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
	   [  1.121e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
	   [  1.122e+03,   1.000e+00,   2.120e-01,   4.240e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
	   [  1.123e+03,   1.000e+00,   2.120e-01,   4.240e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
	   [  1.124e+03,   1.000e+00,   2.140e-01,   4.280e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
	   [  1.125e+03,   1.000e+00,   2.140e-01,   4.280e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01]])
	mpc["gen"]=asarray(
	[[  3.010e+02,   0.000e+00,   0.000e+00,   6.000e+01,  -6.000e+01,   3.060e+00,   1.000e+02,   1.000e+00,   6.000e+01,  -6.000e+01,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00],
	   [  1.119e+03,   1.730e+00,   8.100e-01,   1.100e+00,  -1.000e+00,   1.000e+00,   0.000e+00,   1.000e+00,   1.000e+01,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00]])
	mpc["branch"]=asarray(
	[  [  3.010e+02,   1.100e+03,   4.707e-02,   6.541e-01,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   3.000e+00,  -3.000e+01,   1.000e+00,  -3.600e+02,   3.600e+02],
	   [  1.100e+03,   1.115e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
	   [  1.115e+03,   1.116e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
	   [  1.116e+03,   1.117e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
	   [  1.117e+03,   1.118e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
	   [  1.118e+03,   1.119e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
	   [  1.119e+03,   1.120e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
	   [  1.120e+03,   1.121e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
	   [  1.116e+03,   1.122e+03,   5.420e-02,   1.470e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
	   [  1.118e+03,   1.123e+03,   5.420e-02,   1.470e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
	   [  1.119e+03,   1.124e+03,   5.420e-02,   1.470e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
	   [  1.121e+03,   1.125e+03,   5.420e-02,   1.470e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02]])
	mpc["busnames"]=['SUPPLY_D','DG1_SWBD','DG2_SWBD','DG1_SWBX','DG2_SWBX']
	convert_to_python_indices(mpc)
	return mpc

def draw_network(fignr=754):
	from matplotlib.pyplot import figure, show
	import networkx as nx

	casedata = get_topology()
	ppc = casedata
	ppopt = ppoption(PF_ALG=2)
	ppc = ext2int(ppc)
	figure(fignr)
	g = nx.Graph()
	i = ppc['bus'][:, BUS_I].astype(int)
	g.add_nodes_from(i, bgcolor='green')
	#nx.draw_networkx_nodes(g,pos=nx.spring_layout(g))
	fr = ppc['branch'][:, F_BUS].astype(int)
	to = ppc['branch'][:, T_BUS].astype(int)
	g.add_edges_from(zip(fr, to), color='magenta')
	nx.draw(g, with_labels=True, node_size=1000,node_color='skyblue',width=0.5)
	show()

def measured_data():
	"""
	Load data generated in the EMRP SmartGrid project.
	"""
	data = loadmat("UKGDS60_data/matpowerPQ.mat")
	loadsP15to25 = data['loadsP15to25']  # Active power
	loadsQ15to25 = data['loadsQ15to25']  # Reactive power
	S = np.vstack((loadsP15to25,loadsQ15to25))

	return S

def simulate_data(Sm, gen=None, PVdata=None, PV_idx=None, verbose = 0):
	"""
	Simulate data using power flow analysis
	:param PVgen: data from PV generation injected at bus 6
	:return: dict
	"""
	from scipy.io import loadmat
 
	S=Sm.copy()

	t_ges = 1440
	delta_t = 15
	time= np.arange(delta_t, t_ges+delta_t,delta_t)
	t_f = len(time)
	bus_var = np.arange(2,13,1)  # buses that are varied
	v_mag = np.zeros((13,t_f))
	v_ang = np.zeros((13,t_f))
	P = np.zeros((13,t_f))
	Q = np.zeros((13,t_f))
	loadP = np.zeros((13,t_f))
	loadQ = np.zeros((13,t_f))
	genP_all = np.zeros((2,t_f))
	genQ_all = np.zeros((2,t_f))
	P_into_00 = np.zeros(t_f)
	Q_into_00 = np.zeros(t_f)

	if (len(bus_var)==S.shape[0]):     
		Pdata = np.real(S)
		Qdata = np.imag(S)
	elif (2*len(bus_var)==S.shape[0]):
		Pdata = S[:S.shape[0]/2,:]
		Qdata = S[S.shape[0]/2:,:]
	else:
		raise ValueError("Powers have wrong dimension.")
      
	if isinstance(PVdata,np.ndarray):
		if len(PVdata.shape)==2:   # P and Q for PVgen
			if not PVdata.shape[0]==2:
				PVdata = PVdata.T
			Pdata[PV_idx,:] = -PVdata[0,:]+Pdata[PV_idx,:]
			Qdata[PV_idx,:] = -PVdata[1,:]+Qdata[PV_idx,:]
		else:
			Pdata[PV_idx,:] = -PVdata[:] +Pdata[PV_idx,:]# with MVA
			Qdata[PV_idx,:] = np.zeros_like(PVdata)

        
	casedata = get_topology()
	for n in range(len(time)):
		casedata['bus'][bus_var,2] = Pdata[:,n]  #Changing the values for the active power
		casedata['bus'][bus_var,3] = Qdata[:,n]  #Changing the values for the reactive power
		if isinstance(gen,np.ndarray):
			casedata['gen'][1,1] = gen[n]
			casedata['gen'][1,2] = 0
		ppopt = ppoption(PF_ALG=2)
		ppopt["VERBOSE"] = verbose
		resultPF, success = runpf(casedata, ppopt)

		if success == 0:
			print ('ERROR in step %d', n)

		slack_ang = resultPF['bus'][1,8]
		v_mag[:,n] = resultPF['bus'][:,7]               # Voltage, magnitude
		v_ang[:,n] = resultPF['bus'][:,8] - slack_ang   # Voltage, angle
		loadP[:,n] = resultPF['bus'][:,2]
		loadQ[:,n] = resultPF['bus'][:,3]
		genP_all[:,n] = resultPF['gen'][:,1]
		genQ_all[:,n] = resultPF['gen'][:,2]
		P_into_00[n]=-resultPF['branch'][0,15]
		Q_into_00[n]=-resultPF['branch'][0,16]

	loadP[6,:] = loadP[6,:]+genP_all[1,:]
	loadQ[6,:] = loadQ[6,:]+genQ_all[1,:] 
	simdata = dict([])
	simdata["Vm"] = (11/(np.sqrt(3)))*v_mag
	simdata["Va"] = v_ang
	simdata["Pk"] = -loadP[2:,:]
	simdata["Qk"] = -loadQ[2:,:]
	simdata["P_00"]=P_into_00
 	simdata["Q_00"]=Q_into_00
	return simdata
 



def create_pseudomeas(simdata,meas_idx,PVdataSim=None,PV_idx=None):
	"""Using the power injected at bus 0 (301) and some PV data (simulated),
	pseudo-measurements are created for all nodes at which no measurement
	is assumed available
	"""
	from numpy import matlib
	n_K = simdata["Pk"].shape[0]
	Pfc = -simdata["P_00"]/11 #- simdata["Pk"][0,0]/11
	Qfc = -simdata["Q_00"]/11# - simdata["Qk"][0,0]/11
	full_Pf = matlib.repmat(Pfc,n_K,1)
	full_Qf = matlib.repmat(Qfc,n_K,1)
	if isinstance(PVdataSim,np.ndarray):
		if len(PVdataSim.shape)==2:   # P and Q for PVgen
			if not PVdataSim.shape[0]==2:
				PVdataSim = PVdataSim.T
			full_Pf[PV_idx,:] =PVdataSim[0,:]
			full_Qf[PV_idx,:] =PVdataSim[1,:]
		else:
			full_Pf[PV_idx,:] =PVdataSim[:]
			full_Qf[PV_idx,:] = np.zeros_like(PVdataSim)
	pseudo_meas = dict([])
	pseudo_meas["Pk"] = np.delete(full_Pf,meas_idx["Pk"],0)
	pseudo_meas["Qk"] = np.delete(full_Qf,meas_idx["Qk"],0)
	return pseudo_meas



def full_network():
	from numpy import asarray
	mpc=dict()
	mpc["version"]='2'
	mpc["baseMVA"]=100
	mpc["bus"]=asarray(
	[[  3.010e+02,   3.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   1.000e+00,   3.060e+00,   0.000e+00,   3.300e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.100e+03,   1.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.101e+03,   1.000e+00,   3.920e-01,   7.840e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.102e+03,   1.000e+00,   3.920e-01,   7.840e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.103e+03,   1.000e+00,   1.160e-01,   2.320e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.104e+03,   1.000e+00,   3.920e-01,   7.840e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.105e+03,   1.000e+00,   3.920e-01,   7.840e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.106e+03,   1.000e+00,   1.160e-01,   2.320e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.107e+03,   1.000e+00,   3.920e-01,   7.840e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.108e+03,   1.000e+00,   3.920e-01,   7.840e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.109e+03,   1.000e+00,   1.160e-01,   2.320e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.110e+03,   1.000e+00,   3.940e-01,   7.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.111e+03,   1.000e+00,   3.940e-01,   7.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.112e+03,   1.000e+00,   3.960e-01,   7.920e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.113e+03,   1.000e+00,   1.000e-01,   2.000e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.114e+03,   1.000e+00,   1.020e-01,   2.040e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.115e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.116e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.117e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.118e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.119e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.120e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.121e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.122e+03,   1.000e+00,   2.120e-01,   4.240e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.123e+03,   1.000e+00,   2.120e-01,   4.240e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.124e+03,   1.000e+00,   2.140e-01,   4.280e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.125e+03,   1.000e+00,   2.140e-01,   4.280e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.126e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.127e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.128e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.129e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.130e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.131e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.132e+03,   1.000e+00,   4.260e-01,   8.520e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.133e+03,   1.000e+00,   2.120e-01,   4.240e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.134e+03,   1.000e+00,   2.120e-01,   4.240e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.135e+03,   1.000e+00,   2.140e-01,   4.280e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.136e+03,   1.000e+00,   2.140e-01,   4.280e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.137e+03,   1.000e+00,   4.340e-01,   8.680e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.138e+03,   1.000e+00,   4.340e-01,   8.680e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.139e+03,   1.000e+00,   4.360e-01,   8.720e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.140e+03,   1.000e+00,   4.360e-01,   8.720e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.141e+03,   1.000e+00,   4.360e-01,   8.720e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.142e+03,   1.000e+00,   4.360e-01,   8.720e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.143e+03,   1.000e+00,   4.360e-01,   8.720e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.144e+03,   1.000e+00,   4.360e-01,   8.720e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.145e+03,   1.000e+00,   4.360e-01,   8.720e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.146e+03,   1.000e+00,   2.160e-01,   4.320e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.147e+03,   1.000e+00,   2.180e-01,   4.360e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.148e+03,   1.000e+00,   2.180e-01,   4.360e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.149e+03,   1.000e+00,   2.180e-01,   4.360e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.150e+03,   1.000e+00,   2.180e-01,   4.360e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.151e+03,   1.000e+00,   3.420e-01,   6.840e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.152e+03,   1.000e+00,   3.420e-01,   6.840e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.153e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.154e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.155e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.156e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.157e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.158e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.159e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.160e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.161e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.162e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.163e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.164e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.165e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.166e+03,   1.000e+00,   3.440e-01,   6.880e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.167e+03,   1.000e+00,   2.220e-01,   4.440e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.168e+03,   1.000e+00,   2.220e-01,   4.440e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.169e+03,   1.000e+00,   2.240e-01,   4.480e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.170e+03,   1.000e+00,   2.240e-01,   4.480e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.171e+03,   1.000e+00,   2.240e-01,   4.480e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.172e+03,   1.000e+00,   2.240e-01,   4.480e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.173e+03,   1.000e+00,   2.240e-01,   4.480e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.174e+03,   1.000e+00,   2.240e-01,   4.480e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01],
       [  1.175e+03,   1.000e+00,   2.240e-01,   4.480e-02,   0.000e+00,   0.000e+00,   1.000e+00,   1.000e+00,   0.000e+00,   1.100e+01,   1.000e+00,   1.030e+00,   9.700e-01]])
	mpc["gen"]=asarray(
	[[  3.010e+02,   3.000e+01,   1.000e+01,   6.000e+01,  -6.000e+01,   3.060e+00,   1.000e+02,   1.000e+00,   6.000e+01,  -6.000e+01,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00],
       [  1.119e+03,   1.730e+00,   8.100e-01,   1.100e+00,  -1.000e+00,   1.000e+00,   0.000e+00,   1.000e+00,   1.000e+01,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00]])
	mpc["branch"]=asarray(
	[[  3.010e+02,   1.100e+03,   4.707e-02,   6.541e-01,   0.000e+00,   0.000e+00,   0.000e+00,   0.000e+00,   3.000e+00,  -3.000e+01,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.100e+03,   1.101e+03,   2.038e-01,   1.056e-01,   0.000e+00,   6.820e+00,   6.820e+00,   6.820e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.101e+03,   1.102e+03,   2.038e-01,   1.056e-01,   0.000e+00,   6.820e+00,   6.820e+00,   6.820e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.102e+03,   1.103e+03,   6.240e-02,   1.700e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.100e+03,   1.104e+03,   2.038e-01,   1.056e-01,   0.000e+00,   6.820e+00,   6.820e+00,   6.820e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.104e+03,   1.105e+03,   2.038e-01,   1.056e-01,   0.000e+00,   6.820e+00,   6.820e+00,   6.820e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.105e+03,   1.106e+03,   6.240e-02,   1.700e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.100e+03,   1.107e+03,   2.038e-01,   1.056e-01,   0.000e+00,   6.820e+00,   6.820e+00,   6.820e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.107e+03,   1.108e+03,   2.038e-01,   1.056e-01,   0.000e+00,   6.820e+00,   6.820e+00,   6.820e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.108e+03,   1.109e+03,   6.240e-02,   1.700e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.100e+03,   1.110e+03,   2.660e-01,   1.378e-01,   0.000e+00,   6.820e+00,   6.820e+00,   6.820e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.110e+03,   1.111e+03,   2.660e-01,   1.378e-01,   0.000e+00,   6.820e+00,   6.820e+00,   6.820e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.111e+03,   1.112e+03,   2.660e-01,   1.378e-01,   0.000e+00,   6.820e+00,   6.820e+00,   6.820e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.111e+03,   1.113e+03,   6.630e-02,   1.800e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.112e+03,   1.114e+03,   6.630e-02,   1.800e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.100e+03,   1.115e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.115e+03,   1.116e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.116e+03,   1.117e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.117e+03,   1.118e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.118e+03,   1.119e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.119e+03,   1.120e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.120e+03,   1.121e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.116e+03,   1.122e+03,   5.420e-02,   1.470e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.118e+03,   1.123e+03,   5.420e-02,   1.470e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.119e+03,   1.124e+03,   5.420e-02,   1.470e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.121e+03,   1.125e+03,   5.420e-02,   1.470e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.100e+03,   1.126e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.126e+03,   1.127e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.127e+03,   1.128e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.128e+03,   1.129e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.129e+03,   1.130e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.130e+03,   1.131e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.131e+03,   1.132e+03,   7.450e-02,   5.740e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.127e+03,   1.133e+03,   5.420e-02,   1.470e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.129e+03,   1.134e+03,   5.420e-02,   1.470e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.130e+03,   1.135e+03,   5.420e-02,   1.470e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.132e+03,   1.136e+03,   5.420e-02,   1.470e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.100e+03,   1.137e+03,   9.170e-02,   7.060e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.137e+03,   1.138e+03,   9.170e-02,   7.060e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.138e+03,   1.139e+03,   9.170e-02,   7.060e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.139e+03,   1.140e+03,   9.170e-02,   7.060e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.140e+03,   1.141e+03,   9.170e-02,   7.060e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.141e+03,   1.142e+03,   9.170e-02,   7.060e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.142e+03,   1.143e+03,   9.170e-02,   7.060e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.143e+03,   1.144e+03,   9.170e-02,   7.060e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.144e+03,   1.145e+03,   9.170e-02,   7.060e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.138e+03,   1.146e+03,   5.710e-02,   1.550e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.140e+03,   1.147e+03,   5.710e-02,   1.550e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.141e+03,   1.148e+03,   5.710e-02,   1.550e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.143e+03,   1.149e+03,   5.710e-02,   1.550e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.145e+03,   1.150e+03,   5.710e-02,   1.550e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.100e+03,   1.151e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.151e+03,   1.152e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.152e+03,   1.153e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.153e+03,   1.154e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.154e+03,   1.155e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.155e+03,   1.156e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.156e+03,   1.157e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.157e+03,   1.158e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.158e+03,   1.159e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.159e+03,   1.160e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.160e+03,   1.161e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.161e+03,   1.162e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.162e+03,   1.163e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.163e+03,   1.164e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.164e+03,   1.165e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.165e+03,   1.166e+03,   6.650e-02,   5.120e-02,   0.000e+00,   8.860e+00,   8.860e+00,   8.860e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.152e+03,   1.167e+03,   7.290e-02,   1.980e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.154e+03,   1.168e+03,   7.290e-02,   1.980e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.155e+03,   1.169e+03,   7.290e-02,   1.980e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.157e+03,   1.170e+03,   7.290e-02,   1.980e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.159e+03,   1.171e+03,   7.290e-02,   1.980e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.161e+03,   1.172e+03,   7.290e-02,   1.980e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.162e+03,   1.173e+03,   7.290e-02,   1.980e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.164e+03,   1.174e+03,   7.290e-02,   1.980e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02],
       [  1.166e+03,   1.175e+03,   7.290e-02,   1.980e-02,   0.000e+00,   4.840e+00,   4.840e+00,   4.840e+00,   1.000e+00,   0.000e+00,   1.000e+00,  -3.600e+02,   3.600e+02]])
	mpc["busnames"]=['SUPPLY_D','DG1_SWBD','DG2_SWBD','DG1_SWBX','DG2_SWBX']
	convert_to_python_indices(mpc)
	return mpc
