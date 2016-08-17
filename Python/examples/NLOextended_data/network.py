# -*- coding: utf-8 -*-

from __future__ import division
from scipy.io import loadmat
import numpy as np
import pandas as pd
from os.path import join

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


