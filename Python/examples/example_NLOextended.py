# -*- coding: utf-8 -*-
# if run as script, add parent path for relative importing
from __future__ import division

if __name__ == '__main__' and __package__ is None:
	from os import sys, path
	sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

from NLO.dynamic_models import SimpleModel
from NLO.nodal_load_observer import NLOextended

base_voltage = 6000
nK = 11

base_voltage = 6000
nK = 11

folder = "NLOextended_data"

def loadDaten():
	dfile = pd.ExcelFile(join(folder,"Messdaten.xlsx"))
	names = {}
	inds = {}
	names["Sk"] = dfile.parse(0, header = None, parse_cols = "B").dropna()
	inds["Sk"] = dfile.parse(0, header = None, parse_cols = "A").dropna().values.flatten().astype(int) - 1
	Sk = dfile.parse(0, header = None, parse_cols = "C:CT", index_col = 0).T.values

	names["V"] = dfile.parse(1, header = None, parse_cols = "B")
	inds["V"] = dfile.parse(1, header = None, parse_cols = "A").dropna().values.flatten().astype(int) - 1
	Vk = dfile.parse(1, header = None, parse_cols = "C:CT", index_col = 0).T.values

	names["Sl"] = dfile.parse(2, header = None, parse_cols = "C").dropna()
	inds["Sl"] = dfile.parse(2, header = None, parse_cols = "A:B").dropna().values.astype(int) - 1
	Sl = dfile.parse(2, header = None, parse_cols = "D:CU", index_col = 0).T.values

	names["Sfc"] = dfile.parse(3, header = None, parse_cols = "A")
	inds["Sfc"] = dfile.parse(3, header = None, parse_cols = "A").dropna().astype(int) - 1
	Sfc = dfile.parse(3, header = None, parse_cols = "B:CS", index_col = 0).T.values

	nT = Sk.shape[0]

	Pk = np.zeros((nT, nK))
	Qk = np.zeros((nT, nK))
	names["Pk"], names["Qk"] = names["Sk"], names["Sk"]
	inds["Pk"], inds["Qk"] = inds["Sk"], inds["Sk"]
	Pk[:, inds["Sk"]] = -Sk[:, ::2] / base_voltage ** 2
	Qk[:, inds["Sk"]] = -Sk[:, 1::2] / base_voltage ** 2

	Pl = Sl[:, ::2] / base_voltage ** 2
	Ql = Sl[:, 1::2] / base_voltage ** 2
	names["Pl"], names["Ql"] = names["Sl"], names["Sl"]
	inds["Pl"], inds["Ql"] = inds["Sl"], inds["Sl"]

	Vk /= base_voltage

	Pkfc = Sfc[:, ::2] / base_voltage ** 2
	Qkfc = Sfc[:, 1::2] / base_voltage ** 2
	names["Pfc"], names["Qfc"] = names["Sfc"], names["Sfc"]
	inds["Pfc"], inds["Qfc"] = inds["Sfc"], inds["Sfc"]

	return nT, names, inds, Pk.T, Qk.T, Pl.T, Ql.T, Vk.T, Pkfc.T, Qkfc.T


def netdata():
	dfile = pd.ExcelFile(join(folder,"Netzdaten.xlsx"))
	Ndleitung = dfile.parse(0, header = 0)
	dimension_leitung_daten = Ndleitung.shape
	num_leitung = dimension_leitung_daten[0]
	r = np.zeros((nK, nK))
	x = np.zeros_like(r)
	cap = np.zeros_like(r)  # Kapazität zwischen Leitung und Erde

	for i in range(num_leitung):
		k_start = Ndleitung.ix[i, 0] - 1
		k_end = Ndleitung.ix[i, 1] - 1
		r[k_start, k_end] = Ndleitung.ix[i, 2]
		x[k_start, k_end] = Ndleitung.ix[i, 3]
		cap[k_start, k_end] = Ndleitung.ix[i, 4]

	r = r + r.T
	x = x + x.T
	cap = cap + cap.T
	z = r + 1j * x
	y = np.zeros_like(z)
	for i in range(nK):
		for j in range(nK):
			if z[i, j] != 0:
				y[i, j] = 1 / z[i, j]

	Y = -y
	for i in range(nK):
		Y[i, i] = np.sum(y[i, :]) + 1j * np.sum(cap[i, :] / 2)
	return Y, y, cap


def get_topology():
	topology = {}
	dfile = pd.ExcelFile(join(folder,"Netzdaten.xlsx"))
	topology["branch"] = dfile.parse(0, header = 0).values
	topology["bus"] = np.c_[range(nK), np.r_[3, np.ones(nK - 1)]]
	return topology




nT, names, inds, Pk, Qk, Pl, Ql, Vk, Pkfc, Qkfc = loadDaten()
idx = np.zeros(2*nK,dtype=bool)
# Indeces of measured active power, nPMeas x 1
PMeasIdx = [0, 2, 3, 6, 9]
nPMeas = len(PMeasIdx)

# Indeces of measured reactive power, nQMeas x 1
QMeasIdx = [0, 2, 3, 6, 9]
nQMeas = len(QMeasIdx)

nVMeas = len(inds["V"])

# derive indices for forecasts
idx[:nK][PMeasIdx] = 1
idx[nK:][QMeasIdx] = 1
notidx = ~idx

n = 2 * nK - nPMeas - nQMeas
model = SimpleModel(n, alpha = 0.95, q = 10)
model.P0 = np.eye(n)
meas_idx = {"Pk": PMeasIdx, "Qk": QMeasIdx, "Vm": inds["V"], "Pl": inds["Pl"], "Ql": inds["Ql"]}
meas = {"Pk": Pk[PMeasIdx,:], "Qk": Qk[QMeasIdx,:], "Vm": Vk, "Pl": Pl, "Ql": Ql}
pseudo_meas = {"Pk": Pkfc[notidx[:nK],:], "Qk": Qkfc[notidx[nK:],:]}
meas_unc = {"Pk": 1e-2 * np.ones(nPMeas), "Qk": 1e-2 * np.ones(nQMeas),
			"Pl": 1e-2 * np.ones(inds["Pl"].shape[0]), "Ql": 1e-2 * np.ones(inds["Ql"].shape[0]),
			"Vm": 1e-2 * np.ones(nVMeas)}
Vhat0 = np.r_[np.ones(nK), np.zeros(nK)]
if inds.has_key("V"):
	Vhat0[inds["V"]] = Vk[:, 0]
Vs = Vk[0,:]
Y, y, cap = netdata()
topology = get_topology()

Shat, Vhat, uShat, DeltaS, uDeltaS = \
	NLOextended(topology, meas, meas_unc, meas_idx, pseudo_meas, model, Vhat0, Vs, Y = Y)

time = np.linspace(0, 15 * 96, 95)

for k in range(nK):
	plt.figure(k+1)
	plt.clf()
	plt.plot(time,Shat[k,:],label='estimation')
	plt.plot(time,Pk[k,:],label='measurement')
	plt.legend(loc='best')
	plt.title('bus number %d'%k)

plt.show()
