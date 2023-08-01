#!/usr/bin/env python3
import numpy as np
import h5py as hp
import matplotlib.pyplot as plt
import pandas
import pickle
from functions import read_sim_data, plotter, param_calc_ens, smoothing, spec_from_ens

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
n_runs = 8

plots_folder = '/test/new_paper_plots/'



Nfiles = 23
mode = 1
Lambda = 3 * (2 * np.pi)
Lambda_int = int(Lambda / (2*np.pi))
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

leg = True
H0 = 100
n_use = n_runs-1
nbins_x, nbins_y, npars = 10, 10, 3

flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

file = open("./data/new_trunc_alpha_c_{}.p".format(kind), "rb")
a_list, alpha_c_true, alpha_c_naive, alpha_c_naive2, alpha_c_naive3, alpha_c_naive4 = np.array(pickle.load(file))
file.close()

file = open("./data/spectra_2l_k{}_{}.p".format(mode, kind), "rb")
read_file = pickle.load(file)
xaxis, yaxes = np.array(read_file)
P_nb, P_1l, P_2l = yaxes[0], yaxes[1], yaxes[2]

file = open("spt_spectra_{}".format(kind), 'rb')
read_file = pickle.load(file)
P_11, P_12, P_13, P_22 = np.array(read_file)
file.close()

P_nb = P_nb / 1e4
P_1l = P_1l / 1e4
P_11 = P_11 / xaxis**2
P_12 = P_12 / xaxis**2
P_13 = P_13 / xaxis**2
P_22 = P_22 / xaxis**2

P_eft = P_1l + (2 * alpha_c_naive * P_11)
P_eft2 = P_1l + (2 * alpha_c_naive2 * P_11)
P_eft3 = P_1l + (2 * alpha_c_naive3 * P_11)
P_eft4 = P_1l + (2 * alpha_c_naive4 * P_11)
P_true = P_1l + (2 * alpha_c_true * P_11)

ratio_true = (2 * alpha_c_true * P_11 / (2*np.pi)**2) / (P_13) /3
ratio = (2 * alpha_c_naive * P_11 / (2*np.pi)**2) / (P_13)
ratio2 = (2 * alpha_c_naive2 * P_11 / (2*np.pi)**2) / (P_13)
ratio3 = (2 * alpha_c_naive3 * P_11 / (2*np.pi)**2) / (P_13)
ratio4 = (2 * alpha_c_naive4 * P_11 / (2*np.pi)**2) / (P_13)

plt.plot(xaxis[3:], ratio[3:])
plt.plot(xaxis[3:], ratio2[3:])
plt.plot(xaxis[3:], ratio3[3:])
plt.plot(xaxis[3:], ratio4[3:])
plt.plot(xaxis[3:], ratio_true[3:])
plt.yscale('log')
plt.show()

# yaxes = [P_nb, P_1l, P_eft, P_eft2, P_eft3, P_eft4, P_true]#, P_13+P_11]
# for axis in yaxes:
#     axis *= (2*np.pi)**2
#
# a_sc = 0# 1 / np.max(initial_density(x, A, 1))
# colours = ['b', 'brown', 'k', 'cyan', 'orange', 'xkcd:dried blood', 'r', 'seagreen']
# labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle[\tau]_{\Lambda}\rangle$',  r'EFT: M\&W', 'EFT: $B^{+12}$', r'EFT: DDE', r'EFT: from matching $P_{N-\mathrm{body}}$', r'$a_{\mathrm{sc}}$']
# # labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $[\tau]_{\Lambda}$',  r'EFT: M\&W', 'EFT: $B^{+12}$', r'EFT: DDE', r'EFT: from matching $P_{N-\mathrm{body}}$']#, 'Zel']
# linestyles = ['solid', 'dashdot', 'dashed', 'dashed', 'dashed', 'dashed', 'dotted', 'dotted']
#
# # fac = (P_true / P_nb)[0]
# # P_nb *= fac
# # m = 5
# # yaxes.pop(m)
# # colours.pop(m)
# # labels.pop(m)
# # linestyles.pop(m)
#
#
# savename = 'eft_spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
# xlabel = r'$a$'
# ylabel = r'$a^{-2}k^{2}P(k, a)$'
#
#
# # ylabel = r'$|\tilde{\delta}(k=1, a)|^{2}\; / a^{2}$'
# title = r'$k = k_{{\mathrm{{f}}}},\; \Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt)
# # title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, Lambda_int, kind_txt)
# save = True
# err_Int = []
# plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc=a_sc, title_str=title, terr=[], zel=False, save=save, leg=leg, flags=flags)
