# -*- coding: utf-8 -*-
"""

Application of the NLO-IEKF to the subgrid of UKGDS60 with data from previous SmartGrid project with PV data from TUE

"""
# if run as script, add parent path for relative importing
if __name__ == '__main__' and __package__ is None:
	from os import sys, path
	sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


# indices of measured active power
PMeasIdx = [2,3]
nPMeas = len(PMeasIdx)

# indices of measured reactive power
QMeasIdx = [2,3]
nQMeas = len(QMeasIdx)

# indices of measured bus voltage
VMeasIdx = [0,6,8,9]
nVMeas = len(VMeasIdx)
##########################################
import UKGDS60 as network
from matplotlib.pyplot import *
from scipy.io import loadmat
from NLO.dynamic_models import SimpleModel
from NLO.nodal_load_observer import IteratedExtendedKalman
import tools.profilesPV as ProfilePV
from NLO.nodal_load_observer import LinearKalmanFilter

# load data
topology = network.get_topology()
S, Vs, V, Y, Sfc = network.measured_data()
baseMVA = topology['baseMVA']
nNodes = S.shape[0]/2
nT = S.shape[1]
t = np.arange(15,15*(nT+1),15)
################################################### PV data
filename1 = 'dataMonthKlant55.dat'
PVdata = np.loadtxt(filename1)
location = 'Normal' # ['High', 'Normal', 'Low']
Season = 'Summer' # ['Winter', 'Spring', 'Summer', 'Autumn']
Ampl, mu, sig = ProfilePV.ProfilePower(location, Season)  # paramater identification according your choise of location and season
FittedPV = ProfilePV.gaussian(np.linspace(0, nT-1, nT), Ampl, baseMVA, mu, sig, location) # creating PV pseudo-data according your choise of location and season
LoadP = -PVdata[:,19] # in kVA 
Sfc[6,:] = FittedPV[:]

# Forecasts
if nPMeas == 0:
	SNMeasFc = Sfc
else:
	idx = np.zeros(2*nNodes,dtype=bool)
	idx[:nNodes][PMeasIdx] = 1
	idx[nNodes:][QMeasIdx] = 1
	notidx = ~idx
	SNMeasFc = Sfc[notidx,:]
 


############################################## IEKF
# set initial values
basekV = 11
Vhat0 = np.r_[basekV/(np.sqrt(3)*np.ones(nNodes)), np.zeros(nNodes)]

#--------------------------------------
n = 2*nNodes-nPMeas-nQMeas
model = SimpleModel(n, alpha = 0.95, q=10)
meas_idx = { "Pk": PMeasIdx, "Qk":  QMeasIdx, "Vm": VMeasIdx, "Va": VMeasIdx}
simdata = network.simulate_data(S, gen=LoadP/(baseMVA*1000), PVdata=PVdata[:,14]/1000,PV_idx=6, verbose = 0)
meas = { "Pk": simdata['Pk'][PMeasIdx,:], "Qk": simdata['Qk'][QMeasIdx,:], "V": V,"Vm": simdata['V'][VMeasIdx,:], "Va": simdata['Va'][VMeasIdx,:]}
pseudo_meas=network.create_pseudomeas(simdata,meas_idx,PVdata=PVdata[:,14]/1000,PV_idx=6)
meas_unc = { "Vm": 1e-2*np.ones(nVMeas), "Va": 1e-2*np.ones(nVMeas) }
Shat, Vhat, uShat, DeltaS, uDeltaS = IteratedExtendedKalman(topology, meas, meas_unc, meas_idx, pseudo_meas, model,Vhat0,Vs,Y=Y)

#--------------------------------------

Veff = np.sqrt( V[:nNodes,:]**2 + V[nNodes:,:]**2)
Vhateff = np.sqrt(Vhat[:nNodes,:]**2 + Vhat[nNodes:,:]**2)
Delta_Vhateff = Veff - Vhateff

figure(1,figsize = (18,10));clf()
suptitle("Active power", fontsize=20, fontweight='bold')
for i in range(nNodes):
	subplot(3,4,i+1)
	plot(t, -1000*simdata['Pk'][i,:],'o-c',linewidth=2.5,label="reference")
	plot(t, -1000*Shat[i,:],'D-m',linewidth=2.5,label="estimate")
	plot(t, -1000*Sfc[i,:],'x-k',linewidth=2.5,label="forecast")
	if i in PMeasIdx:
		title("bus %d (measured)"%i, fontsize=20, fontweight='bold')
	else:
		title("bus %d"%i, fontsize=20, fontweight='bold')
	xticks(np.linspace(0,nT*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 12, fontweight='bold')
	xticks(rotation=45)
	yticks(fontweight='bold')
	grid()
legend(loc="upper left",bbox_to_anchor=(1.2,0.7))
subplots_adjust(left=0.05,right=0.98,top=0.95,bottom=0.05)


