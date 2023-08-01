#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import pandas
import pickle
import numpy as np
from functions import plotter, initial_density, SPT_real_tr, smoothing, alpha_to_corr, alpha_c_finder, EFT_sm_kern, dc_in_finder, dn, read_density
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
path = 'cosmo_sim_1d/sim_k_1_11/run1/'
plots_folder =  '../plots/test/sim_k_1_11/real_space/{}/'.format(kind)
Nfiles = 50
mode = 1
Lambda = 3 * (2 * np.pi)
H0 = 100
A = [-0.05, 1, -0.5, 11, 0]


flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

j = 14

a_all, P_nb, P_spt = [], [], []
for j in range(1):
    j = 23
    L = 1.0
    # moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    # moments_file = np.genfromtxt(path + moments_filename)
    # a = moments_file[:,-1][0]
    # x_cell = moments_file[:,0]
    # M0_nbody = moments_file[:,2]-1
    # M0_nbody -= np.mean(M0_nbody)
    # k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x_cell.size/2, x_cell.size/2)) / (2*np.pi)
    # den_nbody = smoothing(M0_nbody, k, Lambda, kind)
    # dc_in, k_in = dc_in_finder(path, x_cell, interp=True)

    dk_par, a, dx = read_density(path, j)
    x_grid = np.arange(0, L, dx)
    k = np.fft.ifftshift(2.0 * np.pi * np.arange(-dk_par.size/2, dk_par.size/2)) / (2*np.pi)
    dk_par /= 125
    den_nbody = smoothing((np.real(np.fft.ifft(dk_par)))-1, k, Lambda, kind)
    x = np.arange(0, 1, 1/dk_par.size)
    dc_in, k_in = dc_in_finder(path, x, interp=True)

    den_spt_tr = SPT_real_tr(dc_in, k, L, Lambda, a, kind)

    err = (den_spt_tr - den_nbody) * 100 /(1+ den_nbody)
    # plt.plot(x, den_spt_tr, ls='dashed')
    plt.plot(x, err)
    plt.show()
    # P_nb.append(np.abs(np.fft.fft(den_nbody)[1])**2 / den_nbody.size)
    # P_spt.append(np.abs(np.fft.fft(den_spt_tr)[1])**2 / den_spt_tr.size)
    # a_all.append(a)
    # print('a = ', a)

# a_list = np.array(a_all)
# P_nb = np.array(P_nb) #/ a_list**2
# P_spt = np.array(P_spt) #/ a_list**2
#
# err = (P_spt - P_nb) * 100 / P_nb
#
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=False, gridspec_kw={'width_ratios': [1], 'height_ratios': [3, 1]})
# ax[1].axhline(0, c='grey', lw=1)
# ax[0].set_title(r'$k = k_{{\mathrm{{f}}}}, \Lambda = {}\;k_{{\mathrm{{f}}}}$ ({})'.format(int(Lambda / (2 * np.pi)), kind_txt), fontsize=18, y=1.01)
# ax[1].set_xlabel(r'$a$', fontsize=20)
# ax[0].set_ylabel(r'$P(k=1)$', fontsize=20)
# ax[1].set_ylabel(r'\% err', fontsize=20)
# ax[0].plot(a_list, P_nb, label=r'$N-$body', lw=2, c='b')
# ax[0].plot(a_list, P_spt, label=r'tSPT-4', lw=2, c='brown', ls='dashdot')
# ax[0].legend(fontsize=12)
# ax[1].plot(a_list, err, c='brown', lw=2, ls='dashdot')
# for j in range(2):
#     ax[j].minorticks_on()
#     ax[j].tick_params(axis='both', which='both', direction='in', labelsize=15)
#     ax[j].yaxis.set_ticks_position('both')
#
# # plt.show()
# plt.savefig('../plots/test/new_paper_plots/test_power_par.png', bbox_inches='tight', dpi=150)
# plt.close()
