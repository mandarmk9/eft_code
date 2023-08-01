#!/usr/bin/env python3
import numpy as np
import h5py as hp
import matplotlib.pyplot as plt
import pandas
import pickle
from functions import read_sim_data, plotter, param_calc_ens, smoothing, spec_from_ens, dc_in_finder, dn, read_hier
from zel import *

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# path = 'cosmo_sim_1d/final_phase_run1/'
# path = 'cosmo_sim_1d/sim_k_1_11/run1/'

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
n_runs = 8

# path = 'cosmo_sim_1d/another_sim_k_1_11/run1/'
# n_runs = 24

# path = 'cosmo_sim_1d/sim_k_1/run1/'


# path = 'cosmo_sim_1d/multi_k_sim/run1/'
# n_runs = 8

# A = [-0.05, 1, -0.5, 11]
# A = [-0.05, 1, 0, 11]
A = [-0.05, 1, -0.5, 11]

# plots_folder = '/sim_k_1_11/'
# plots_folder = '/sim_k_1/'
plots_folder = '/test/new_paper_plots/'
# plots_folder = '/new_sim_k_1_11/'


Nfiles = 23
mode = 2
Lambda = 3 * (2 * np.pi)
Lambda_int = int(Lambda / (2*np.pi))
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

leg = True#False
H0 = 100
n_use = n_runs-1
zel = False
modes = True
folder_name = '' # '/new_data_{}/L{}'.format('sharp', Lambda_int)
save = True
# fitting_method = 'WLS'
# fitting_method = 'lmfit'
fitting_method = 'curve_fit'
# fitting_method = ''
nbins_x, nbins_y, npars = 10, 10, 3

# def P_finder(path, Nfiles, Lambda, kind, mode):
#     Nx = 2048
#     L = 1.0
#     dx = L/Nx
#     x = np.arange(0, L, dx)
#     k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
#     a_list, P13_list, P22_list, P11_list, P12_list = [], [], [], [], []
#     for j in range(Nfiles):
#         a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
#         dc_in, k = dc_in_finder(path, x, interp=True) #[0]
#         dc_in = smoothing(dc_in, k, Lambda, kind)
#         Nx = dc_in.size
#         F = dn(3, L, dc_in)
#         d1k = (np.fft.fft(F[0]) / Nx)
#         d2k = (np.fft.fft(F[1]) / Nx)
#         d3k = (np.fft.fft(F[2]) / Nx)
#         P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
#         P11 = (d1k * np.conj(d1k)) * (a**2)
#         P22 = (d2k * np.conj(d2k)) * (a**4)
#         P12 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k))) * (a**3)
#
#         P13_list.append(np.real(P13)[mode])
#         P11_list.append(np.real(P11)[mode])
#         P12_list.append(np.real(P12)[mode])
#         P22_list.append(np.real(P22)[mode])
#
#         a_list.append(a)
#         print('a = ', a)
#     return np.array(a_list), np.array(P13_list), np.array(P11_list), np.array(P22_list), np.array(P12_list)
#


# def SPT(dc_in, L, a):
#     """Returns the SPT PS upto 2-loop order"""
#     F = dn(5, L, dc_in)
#     Nx = dc_in.size
#     d1k = (np.fft.fft(F[0]) / Nx)
#     d2k = (np.fft.fft(F[1]) / Nx)
#     d3k = (np.fft.fft(F[2]) / Nx)
#     d4k = (np.fft.fft(F[3]) / Nx)
#     d5k = (np.fft.fft(F[4]) / Nx)
#
#     P11 = (d1k * np.conj(d1k)) * (a**2)
#     P12 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3)
#     P22 = (d2k * np.conj(d2k)) * (a**4)
#     P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
#     P14 = ((d1k * np.conj(d4k)) + (d4k * np.conj(d1k))) * (a**5)
#     P23 = ((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a**5)
#     P33 = (d3k * np.conj(d3k)) * (a**6)
#     P15 = ((d1k * np.conj(d5k)) + (d5k * np.conj(d1k))) * (a**6)
#     P24 = ((d2k * np.conj(d4k)) + (d4k * np.conj(d2k))) * (a**6)
#
#     P_lin = P11
#     P_1l = P_lin + P12 + P13 + P22
#     P_2l = P_1l + P14 + P15 + P23 + P24 + P33
#     return np.real(P11), np.real(P12), np.real(P13), np.real(P22), np.real(P14), np.real(P23), np.real(P33), np.real(P15), np.real(P24)
#
# sm = True
# moments_filename = 'output_hierarchy_{0:04d}.txt'.format(0)
# moments_file = np.genfromtxt(path + moments_filename)
# a0 = moments_file[:,-1][0]
# folder_name = 'hierarchy'
# P11 = np.zeros(Nfiles)
# P12 = np.zeros(Nfiles)
# P13 = np.zeros(Nfiles)
# P22 = np.zeros(Nfiles)
# P14 = np.zeros(Nfiles)
# P23 = np.zeros(Nfiles)
# P33 = np.zeros(Nfiles)
# P15 = np.zeros(Nfiles)
# P24 = np.zeros(Nfiles)
#
# a_list = np.zeros(Nfiles)
#
# for j in range(0, Nfiles):
#     a, dx, M0_par = read_hier(path, j, folder_name)[:3]
#     M0_k = np.fft.fft(M0_par) / M0_par.size
#     x = np.arange(0, 1, dx)
#     k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))
#     dc_in, k_in = dc_in_finder(path, x)
#     if sm == True:
#         M0_par = smoothing(M0_par, k, Lambda, kind)
#         dc_in = smoothing(dc_in, k_in, Lambda, kind)
#     else:
#         pass
#     P_nb_a = np.real(M0_k * np.conj(M0_k))
#     # P_lin_a, P_1l_a, P_2l_a = SPT(dc_in, 1, a)
#     P_11, P_12, P_13, P_22, P_14, P_23, P_33, P_15, P_24 = SPT(dc_in, 1, a)
#
#     #we now extract the solutions for a specific mode
#     P11[j] = P_11[mode]
#     P12[j] = P_12[mode]
#     P13[j] = P_13[mode]
#     P22[j] = P_22[mode]
#     P14[j] = P_14[mode]
#     P23[j] = P_23[mode]
#     P33[j] = P_33[mode]
#     P15[j] = P_14[mode]
#     P24[j] = P_24[mode]
#
#     a_list[j] = a
#     print('a = ', a, '\n')
#
# # a_list, P_13, P_11, P_22, P_12 = P_finder(path, Nfiles, Lambda, kind, mode)
# #
#
# file = open("spt_2l_spectra_k{}_{}".format(mode, kind), 'wb')
# df =  pandas.DataFrame(data=[P11, P12, P13, P22, P14, P23, P33, P15, P24])
# pickle.dump(df, file)
# file.close()

file = open("spt_2l_spectra_k{}_{}".format(mode, kind), 'rb')
read_file = pickle.load(file)
P11, P12, P13, P22, P14, P23, P33, P15, P24 = np.array(read_file)
file.close()

a_list = np.zeros(23)
for j in range(23):
    a_list[j] = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))


file = open("./data/spectra_2l_k{}_{}.p".format(mode, kind), "rb")
read_file = pickle.load(file)
xaxis, yaxes = np.array(read_file)
P_nb, P_1l, P_2l = yaxes[0], yaxes[1], yaxes[2]

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()

P_nb /= 1e4
# power = []
# Ps = [P11, P12, P13, P22, P14, P23, P33, P15, P24]
# for P in Ps:
#     P = ((P/a_list**2))#-P_nb)*100 / P_nb
#     power.append(P)
#
# power = np.array(power)
# P_1l = power[0] + power[1] + power[2] + power[3]
# P_2l = power[4] + power[5] + power[6] + power[7] + power[8]

# P_1l = P11 + P12 + P13 + P22
# P_2l = P14 + P33 + P15 + P24 + P23


ax.set_xlabel(r'$a$', fontsize=16)
# ax.set_ylabel(ylabel, fontsize=16)

# ax.plot(a_list, P11/a_list**2, c='r', lw=1.5, label=r'$P_{11}$')
ax.plot(a_list, P_nb*1e4/a_list**2, c='b', lw=2, label=r'$P_{N\mathrm{-body}}$')
ax.plot(a_list, P_1l/a_list**2, c='seagreen', ls='dashdot', lw=2, label=r'$P_{\mathrm{SPT-4}}$')
ax.plot(a_list, (P_2l)/a_list**2, c='k', ls='dashed', lw=2, label=r'$P_{\mathrm{SPT-6}}$')


ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.ticklabel_format(scilimits=(-2, 3))
ax.yaxis.set_ticks_position('both')
ax.legend(fontsize=12)#, loc=2, bbox_to_anchor=(1,1))

plt.show()
# plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
# plt.close()
