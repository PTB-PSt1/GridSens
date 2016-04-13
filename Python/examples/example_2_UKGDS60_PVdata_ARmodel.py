# -*- coding: utf-8 -*-
"""

Application of the NLO-IEKF to the subgrid of UKGDS60, published SmartGIFT 2016

"""
# if run as script, add parent path for relative importing
if __name__ == '__main__' and __package__ is None:
	from os import sys, path
	sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


# indices of measured active power
PMeasIdx = []
nPMeas = len(PMeasIdx)

# indices of measured reactive power
QMeasIdx = []
nQMeas = len(QMeasIdx)

# indices of measured bus voltage
VMeasIdx = [0,1,2,3,4,5,6,7,8,9,10]
nVMeas = len(VMeasIdx)
##########################################
import UKGDS60_for_example_2 as network
from matplotlib.pyplot import *
from scipy.io import loadmat
from NLO.dynamic_models import AR2Model_single, SimpleModel
from NLO.nodal_load_observer import IteratedExtendedKalman, LinearKalmanFilter
import tools.profilesPV as ProfilePV
from pypower.api import ppoption, runpf, makeYbus 
from tools.load import convert_mcase, convert_to_python_indices
from pypower.ext2int import ext2int
from tools.data_tools import process_admittance, separate_Yslack

# from NLO.nodal_load_observer import LinearKalmanFilter

# load data
topology = network.full_network()
S = network.measured_data()

Ybus, Yf, Yt = makeYbus(topology['baseMVA'], topology['bus'], topology['branch']) # nodal admittance matrix Ybus
baseMVA= topology['baseMVA']
condition = np.hstack((1,np.arange(16,27,1))) 
Yws = Ybus[np.ix_(condition, condition)]

nNodes = S.shape[0]/2
nT = S.shape[1]
t = np.arange(15,15*(nT+1),15)
#################################################### PV data
filename1 = 'dataMonthKlant55.dat'
PVdata = np.loadtxt(filename1)
location = 'Normal' # ['High', 'Normal', 'Low']
Season = 'Summer' # ['Winter', 'Spring', 'Summer', 'Autumn']
Ampl, mu, sig = ProfilePV.ProfilePower(location, Season)  # paramater identification according your choise of location and season
FittedPV = ProfilePV.gaussian(np.linspace(0, nT-1, nT), Ampl, baseMVA, mu, sig, location) # creating PV pseudo-data according your choise of location and season
LoadP = PVdata[:,19] # in kVA 

############################################### IEKF
# set initial values
basekV = 11
Vhat0 = np.r_[basekV/(np.sqrt(3)*np.ones(nNodes)), np.zeros(nNodes)]

#--------------------------------------
n = 2*nNodes-nPMeas-nQMeas
model = AR2Model_single(n, phi1=1.661,phi2=-0.664,noise=0.001) # noise = float
meas_idx = { "Pk": PMeasIdx, "Qk":  QMeasIdx, "Vm": VMeasIdx, "Va": VMeasIdx}
simdata = network.simulate_data(S, gen=LoadP/(baseMVA*1000), PVdata=PVdata[:,14]/1000,PV_idx=6, verbose = 0)
meas = { "Pk": simdata['Pk'][PMeasIdx,:], "Qk": simdata['Qk'][QMeasIdx,:], "Vm": simdata['Vm'][2:,:][VMeasIdx,:], "Va": simdata['Va'][2:,:][VMeasIdx,:]}
Vs = np.vstack((simdata['Vm'][1,:],simdata['Va'][1,:]))
pseudo_meas=network.create_pseudomeas(simdata,meas_idx,PVdataSim=FittedPV[:],PV_idx=6)
meas_unc = { "Vm": 1e-4*np.ones(nVMeas), "Va": 1e-4*np.ones(nVMeas) }
Shat, Vhat, uShat, DeltaS, uDeltaS = IteratedExtendedKalman(topology, meas, meas_unc, meas_idx, pseudo_meas, model,Vhat0,Vs,Y=Yws)

model = SimpleModel(n, alpha = 0.95, q=10)
meas_idx = { "Pk": PMeasIdx, "Qk":  QMeasIdx, "Vm": VMeasIdx, "Va": VMeasIdx}
simdata = network.simulate_data(S, gen=LoadP/(baseMVA*1000), PVdata=PVdata[:,14]/1000,PV_idx=6, verbose = 0)
meas = { "Pk": simdata['Pk'][PMeasIdx,:], "Qk": simdata['Qk'][QMeasIdx,:], "Vm": simdata['Vm'][2:,:][VMeasIdx,:], "Va": simdata['Va'][2:,:][VMeasIdx,:]}
Vs = np.vstack((simdata['Vm'][1,:],simdata['Va'][1,:]))
pseudo_meas=network.create_pseudomeas(simdata,meas_idx,PVdataSim=FittedPV[:],PV_idx=6)
meas_unc = { "Vm": 1e-4*np.ones(nVMeas), "Va": 1e-4*np.ones(nVMeas) }
Shat1, Vhat1, uShat1, DeltaS1, uDeltaS1 = IteratedExtendedKalman(topology, meas, meas_unc, meas_idx, pseudo_meas, model,Vhat0,Vs,Y=Yws)
##---------------------------------



NonMeasIdx = list(set(range(nNodes)) - set(PMeasIdx))
figure(4); clf()    # active power
ylabel('Active power [MW]', fontsize=20, fontweight='bold')
plot(t, S[8,:],'b-D',linewidth=2.5,label="true value")
plot(t, -Shat[8,:],'c-o',linewidth=2.5,label="AR model")
plot(t, -Shat1[8,:],'k-v',linewidth=2.5,label="simple model",ms=5)
plot(t, -pseudo_meas['Pk'][NonMeasIdx.index(8),:],'m-',linewidth=4.5,label="pseudo-measurements",alpha=0.5)   
legend(loc="upper right")
xticks(np.linspace(0,nT*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 12, fontweight='bold')
xticks(rotation=45)
yticks(fontweight='bold')
grid()

