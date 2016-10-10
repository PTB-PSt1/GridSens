# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 12:41:39 2016

@author: makara01
"""
if __name__ == '__main__' and __package__ is None:
	from os import sys, path
	sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
 
import UKGDS60_for_example_2 as network
from scipy.io import loadmat                                                                
import numpy as np
from matplotlib.pyplot import * 
from NLO.nodal_load_observer import IteratedExtendedKalman
import tools.profilesPV as ProfilePV
from numpy import matlib
from NLO.dynamic_models import AR2Model_single, SimpleModel

# indices of measured active power
PMeasIdx = []
nPMeas = len(PMeasIdx)

# indices of measured reactive power
QMeasIdx = []
nQMeas = len(QMeasIdx)

# indices of measured bus voltage
VMeasIdx = [0,1,2,3,4,5,6,7,8,9,10]
nVMeas = len(VMeasIdx)

mat = loadmat('UKGDS60_data/MatpowerPQ.mat')
loadsP15to25 = mat['loadsP15to25']  # Active power  
loadsQ15to25 = mat['loadsQ15to25']  # Reactive power
genP19 = mat['genP19']
genQ19 = mat['genQ19']

topology = network.full_network()
Ybus, Yf, Yt = makeYbus(topology['baseMVA'], topology['bus'], topology['branch']) # nodal admittance matrix Ybus
baseMVA= topology['baseMVA']
condition = np.hstack((1,np.arange(16,27,1))) 
Yws = Ybus[np.ix_(condition, condition)]

t_f = 96
# time variation
t_ges = 1440  # all time in min
delta_t = 15  # time intervals in min
time= np.arange(delta_t, t_ges+delta_t,delta_t)

filename1 = 'dataMonthKlant55.dat'
PVdata = np.loadtxt(filename1)

location = 'Normal' # ['High', 'Normal', 'Low']
Season = 'Summer' # ['Winter', 'Spring', 'Summer', 'Autumn']
    
Ampl, mu, sig = ProfilePV.ProfilePower(location, Season)  # paramater identification according your choise of location and season
FittedPV = ProfilePV.gaussian(np.linspace(0, t_f-1, t_f), Ampl, baseMVA, mu, sig, location) # creating PV pseudo-data according your choise of location and season
LoadP = -PVdata[:,19] # in kVA 

################# Changing Loads ####################
loadsP15to25[6,:] = -PVdata[:,14] /1000 +loadsP15to25[6,:]# in kVA ***************
bus_var = np.arange(2,13,1)  # buses that are varied 

casedata = network.get_topology()
v_mag = np.zeros((13,t_f)) 
v_ang = np.zeros((13,t_f))
loadP_all = np.zeros((13,t_f))
loadQ_all = np.zeros((13,t_f))
genP_all = np.zeros((2,t_f))
genQ_all = np.zeros((2,t_f))
P_into_00 = np.zeros((1,t_f))
Q_into_00 = np.zeros((1,t_f))


for n in range(len(time)):
    casedata['bus'][bus_var,2] = loadsP15to25[:,n]  #Changing the values for the active power
    casedata['bus'][bus_var,3] = loadsQ15to25[:,n]  #Changing the values for the reactive power
    casedata['gen'][1,1] = LoadP[n]/(baseMVA*1000)  #Changing the values for the gen-active power
    casedata['gen'][1,2] = 0                        #Changing the values for the gen-reactive power
    ppopt = ppoption(PF_ALG=2)
    resultPF, success = runpf(casedata, ppopt)
    
    if success == 0:
        print ('ERROR in step %d', n) 
            
    
    slack_ang = resultPF['bus'][1,8]
    v_mag[:,n] = resultPF['bus'][:,7]               # Voltage, magnitude
    v_ang[:,n] = resultPF['bus'][:,8] - slack_ang   # Voltage, angle
    loadP_all[:,n] = resultPF['bus'][:,2]
    loadQ_all[:,n] = resultPF['bus'][:,3]
    genP_all[:,n] = resultPF['gen'][:,1]
    genQ_all[:,n] = resultPF['gen'][:,2]
    P_into_00[:,n]=-resultPF['branch'][0,15]
    Q_into_00[:,n]=-resultPF['branch'][0,16]

    
Pn = -loadP_all[2:,:]
Qn = -loadQ_all[2:,:]
Pn[4,:] = Pn[4,:]+genP_all[1,:]
Qn[4,:] = Qn[4,:]+genQ_all[1,:]
S= np.vstack((Pn,Qn))                       
S0 = np.vstack((P_into_00, Q_into_00))

ReV = (11/(np.sqrt(3)))*v_mag*np.cos(np.radians(v_ang))
ImV = (11/(np.sqrt(3)))*v_mag*np.sin(np.radians(v_ang))
V = np.vstack((ReV[2:,:],ImV[2:,:])) 
ReVs = 11/(np.sqrt(3))*v_mag[1,:]*np.cos(np.radians(v_ang[1,:]))
ImVs = 11/(np.sqrt(3))*v_mag[1,:]*np.sin(np.radians(v_ang[1,:]))
Vs = np.vstack((ReVs, ImVs))

nNodes = S.shape[0]/2
nT = S.shape[1]
t = np.arange(15,15*(nT+1),15)

basekV = 11
Vhat0 = np.r_[basekV/(np.sqrt(3)*np.ones(nNodes)), np.zeros(nNodes)]


Sfctemp = -S0/11
Pfc = Sfctemp[0,:]
Qfc = Sfctemp[1,:]
Sfc = np.vstack((matlib.repmat(Pfc,len(Pn[:,0]),1), matlib.repmat(Qfc,len(Qn[:,0]),1)))
Sfc[6,:] = FittedPV[:]

n = 2*nNodes-nPMeas-nQMeas
model = AR2Model_single(n, phi1=1.661,phi2=-0.664,noise=0.001) # noise = float
simdata = dict([])
simdata['Pk']=Pn
simdata['Qk']=Qn
simdata['Vm']=(11/(np.sqrt(3)))*v_mag
simdata['Va']=v_ang
meas = { "Pk": simdata['Pk'][PMeasIdx,:], "Qk": simdata['Qk'][QMeasIdx,:], "Vm": simdata['Vm'][2:,:][VMeasIdx,:], "Va": simdata['Va'][2:,:][VMeasIdx,:]}
meas_unc = { "Vm": 1e-4*np.ones(nVMeas), "Va": 1e-4*np.ones(nVMeas) }
meas_idx = { "Pk": PMeasIdx, "Qk":  QMeasIdx, "Vm": VMeasIdx, "Va": VMeasIdx}
pseudo_meas = dict([])
pseudo_meas["Pk"] = Sfc[:nNodes,:]
pseudo_meas["Qk"] = Sfc[nNodes:,:]
Shat, Vhat, uShat, DeltaS, uDeltaS = IteratedExtendedKalman(topology, meas, meas_unc, meas_idx, pseudo_meas, model,Vhat0,Vs,Y=Yws)

model = SimpleModel(n, alpha = 0.95, q=10)
meas_idx = { "Pk": PMeasIdx, "Qk":  QMeasIdx, "Vm": VMeasIdx, "Va": VMeasIdx}
simdata = network.simulate_data(S, gen=LoadP/(baseMVA*1000), PVdata=PVdata[:,14]/1000,PV_idx=6, verbose = 0)
meas = { "Pk": simdata['Pk'][PMeasIdx,:], "Qk": simdata['Qk'][QMeasIdx,:], "Vm": simdata['Vm'][2:,:][VMeasIdx,:], "Va": simdata['Va'][2:,:][VMeasIdx,:]}
Vs = np.vstack((simdata['Vm'][1,:],simdata['Va'][1,:]))
pseudo_meas=network.create_pseudomeas(simdata,meas_idx,PVdataSim=FittedPV[:],PV_idx=6)
meas_unc = { "Vm": 1e-4*np.ones(nVMeas), "Va": 1e-4*np.ones(nVMeas) }
Shat1, Vhat1, uShat1, DeltaS1, uDeltaS1 = IteratedExtendedKalman(topology, meas, meas_unc, meas_idx, pseudo_meas, model,Vhat0,Vs,Y=Yws)

figure(4); clf()    # active power
ylabel('Active power [MW]', fontsize=20, fontweight='bold')
plot(time,-S[8,:],'b-D',linewidth=2.5,label="true value")
plot(time,-Shat[8,:],'c-o',linewidth=2.5,label="AR model")
plot(time,Shat1[8,:],'k-v',linewidth=2.5,label="simple model",ms=5)
plot(time,-Sfc[8,:],'m-',linewidth=4.5,label="pseudo-measurements",alpha=0.5)   
legend(loc="upper right")
xticks(np.linspace(0,t_f*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 12, fontweight='bold')
xticks(rotation=45)
yticks(fontweight='bold')
grid()


