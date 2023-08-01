#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import pandas
import pickle
from functions import plotter, dn, read_density, EFT_sm_kern, smoothing, dc_in_finder, read_hier
from scipy.interpolate import interp1d
# from EFT_nbody_solver import *
from tqdm import tqdm
from SPT import SPT_final
from zel import *
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#run5: k2 = 7; Nfiles = 101
#run2: k2 = 11; Nfiles = 81
#run6: only k1; Nfiles = 51
#./cosmo_sim_1d  -a 0.5 -A 6.0 -s 60000 -n 250000 -l 1000 -m


def SPT(dc_in, L, a):
    """Returns the SPT PS upto 2-loop order"""
    F = dn(5, L, dc_in)
    Nx = dc_in.size
    d1k = (np.fft.fft(F[0]) / Nx)
    d2k = (np.fft.fft(F[1]) / Nx)
    d3k = (np.fft.fft(F[2]) / Nx)
    d4k = (np.fft.fft(F[3]) / Nx)
    d5k = (np.fft.fft(F[4]) / Nx)

    P11 = (d1k * np.conj(d1k)) * (a**2)
    P12 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3)
    P22 = (d2k * np.conj(d2k)) * (a**4)
    P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
    P14 = ((d1k * np.conj(d4k)) + (d4k * np.conj(d1k))) * (a**5)
    P23 = ((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a**5)
    P33 = (d3k * np.conj(d3k)) * (a**6)
    P15 = ((d1k * np.conj(d5k)) + (d5k * np.conj(d1k))) * (a**6)
    P24 = ((d2k * np.conj(d4k)) + (d4k * np.conj(d2k))) * (a**6)

    P_lin = P11
    P_1l = P_lin + P12 + P13 + P22
    P_2l = P_1l + P14 + P15 + P23 + P24 + P33
    # print(np.real(P_lin[1]), np.real(P12[1]), np.real(P13[1]), np.real(P22[1]))
    # print(np.real(d1k[1]), np.real(d1k[1]**2), np.real(d2k[2]), np.real(d3k[1]))

    return np.real(P_lin), np.real(P_1l), np.real(P_2l)

# path = 'cosmo_sim_1d/nbody_new_run2/'
# path = 'cosmo_sim_1d/sim_k_1_15/run1/'
# A = [-0.05, 1, -0.5, 15]
#
# path = 'cosmo_sim_1d/amp_ratio_test/run1/'
# A = [-0.1, 1, -0.5, 11]

# path = 'cosmo_sim_1d/new_sim_k_1_11/run1/'
# A = [-0.1, 1, -0.5, 11]

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
A = [-0.05, 1, -0.5, 11]

# # path = 'cosmo_sim_1d/sim_3_15/run1/'
# path = 'cosmo_sim_1d/multi_sim_3_15_33/run1/'

A = []


# path = 'cosmo_sim_1d/test_run2/'
# A = [-0.05, 1, -0.5, 11]


Nfiles = 50
mode = 1
sm = True
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'

# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'


#define lists to store the data
a_list = np.zeros(Nfiles)

#the densitites
P_nb = np.zeros(Nfiles)
P_lin = np.zeros(Nfiles)
P_1l = np.zeros(Nfiles)
P_2l = np.zeros(Nfiles)
P_zel = np.zeros(Nfiles)


moments_filename = 'output_hierarchy_{0:04d}.txt'.format(0)
moments_file = np.genfromtxt(path + moments_filename)
a0 = moments_file[:,-1][0]
q = np.genfromtxt(path + 'output_{0:04d}.txt'.format(0))[:,0]
q_zel = q[::50] #np.arange(0, 1, 0.001)
k_zel = np.fft.ifftshift(2.0 * np.pi * np.arange(-q_zel.size/2, q_zel.size/2))
P_zel, a_zel = [], []
folder_name = 'hierarchy'
# folder_name = '/new_hier/data_{}/L{}'.format(kind, int(Lambda/(2*np.pi)))

# for j in (range(0, Nfiles)):
#     # moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
#     # moments_file = np.genfromtxt(path + moments_filename)
#     # a = moments_file[:,-1][0]
#     # x_cell = moments_file[:,0]
#     # dk_par, a_, dx = read_density(path, j)
#     # L = 1.0
#     # x = np.arange(0, L, dx)
#     # Nx = x.size
#     # k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
#     # dc_in, k_in = dc_in_finder(path, x_cell)
#     #
#     # # M0_par = moments_file[:,2]
#     # M0_par = np.real(np.fft.ifft(dk_par))
#     # M0_par = (M0_par / np.mean(M0_par)) - 1
#     # M0_k = np.fft.fft(M0_par) / M0_par.size
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
#     P_lin_a, P_1l_a, P_2l_a = SPT(dc_in, 1, a)
#     # print((P_nb_a[1] - P_1l_a[1]) * 100 / P_nb_a[1])

#     #we now extract the solutions for a specific mode
#     P_nb[j] = P_nb_a[mode]
#     P_lin[j] = P_lin_a[mode]
#     P_1l[j] = P_1l_a[mode]
#     P_2l[j] = P_2l_a[mode]
#     a_list[j] = a
#     # print('a = ', a, '\n')
#     # if a < 1.6:
#     #     dc_zel = eulerian_sampling(q_zel, a, A, L)[1]
#     #     if sm == True:
#     #         dc_zel = smoothing(dc_zel, k_zel, Lambda, kind)
#     #     dk_zel = np.fft.fft(dc_zel) / dc_zel.size
#     #     P_zel_a = np.real(dk_zel * np.conj(dk_zel))[mode]
#     #     P_zel.append(P_zel_a)
#     #     a_zel.append(a)

# # # P_zel = np.array(P_zel)
# # # a_zel = np.array(a_zel)
# #
# # # print((P_nb - P_1l) * 100 / P_nb)
# # # print(P_zel.size)
# # # #for plotting the spectra
# xaxis = a_list
# yaxes = [P_nb * 1e4 / a_list**2, P_1l * 1e4 / a_list**2, P_2l * 1e4 / a_list**2]#, P_zel * 1e4 / a_zel**2]#, P_2l * 1e4 / a_list**2]#, P_zel * 1e4 / a_list**2]# / a_list**2]
# # yaxes = [P_nb * 1e4, P_1l * 1e4]#, P_2l * 1e4 / a_list**2]#, P_zel * 1e4 / a_zel**2]#, P_2l * 1e4 / a_list**2]#, P_zel * 1e4 / a_list**2]# / a_list**2]

# df = pandas.DataFrame(data=[xaxis, yaxes])
# file = open("./data/spectra_sm_{}.p".format(sm, kind), "wb")
# # file = open("./data/spectra_2l_k{}_{}.p".format(mode, kind), "wb")
# pickle.dump(df, file)
# file.close()

# # # file = open("./data/spectra_2l_k{}_{}.p".format(mode, kind), "rb")
# # # xaxis, yaxes = np.array(pickle.load(file))
# # # file.close()

sm = False
file = open("./data/spectra_sm_{}.p".format(sm, kind), "rb")
xaxis, yaxes = np.array(pickle.load(file))
file.close()
spt_4 = yaxes[1]


sm = True
file = open("./data/spectra_sm_{}.p".format(sm, kind), "rb")
xaxis, yaxes = np.array(pickle.load(file))
file.close()

file = open(f"./data/renorms_{kind}.p", "rb")
alpha_ctr, alpha_c, alpha_ren, P11 = np.array(pickle.load(file))
file.close()



yaxes = [yaxes[0], spt_4, yaxes[1]]
# for j in range(len(yaxes)):
#     yaxes[j] *= (2*np.pi / 10) 


# df = pandas.DataFrame(data=[xaxis, yaxes])
# file = open(rf"./{path}/spt_spec_plot_{kind}.p", "wb")
# pickle.dump(df, file)
# file.close()


# fac = 200 * np.pi / (1e4)
# # yaxes = [yaxes[0]/xaxis**2, yaxes[1]/xaxis**2, yaxes[2]/xaxis**2]
# yaxes = [yaxes[0][:Nfiles]*fac, spt_4[:Nfiles]*fac, yaxes[1][:Nfiles]*fac]
# xaxis = xaxis[:Nfiles]

# colours = ['b', 'brown', 'r', 'k', 'magenta']#, 'k']
# colours = ['b', 'brown', 'k', 'r', 'magenta']#, 'k']
colours = ['b', 'k', 'brown', 'r', 'magenta']#, 'k']

linestyles = ['solid', 'dashdot', 'dashdot', 'dashed']#, 'dashed', 'dashdot']

x = np.arange(0, 1, 1/1000)
a_sc = 0#1 / np.max(initial_density(x, A, 1))

# plots_folder = 'test/multi_sim_3_15_33/'#/paper_plots' #'phase_full_run3' #_k7_L3'
plots_folder = 'paper_plots_final/'#/paper_plots' #'phase_full_run3' #_k7_L3'
save = True

# sm = True
if sm == True:
    # labels = [r'$N$-body', 'tSPT-4', 'tSPT-6']#'SPT: 2-loop',]
    # labels = [r'$N$-body', 'tSPT-4', 'tSPT-6']#'SPT: 2-loop',]
    labels = [r'$N$-body', 'SPT', 'tSPT', 'EFT: ren']#'SPT: 2-loop',]

    # savename = 'spt_spec_{}_1'.format(kind)
    # savename = 'early_k{}'.format(mode)
    savename = 'spt_spec_{}'.format(kind)

else:
    labels = [r'$N$-body', 'SPT-4', 'SPT-6']
    savename = 'spt_spec_ns'

leg = True
xlabel = r'$a$'
# ylabel = r'$a^{-2}P(k, a) \times 10^{4}\;\;[h^{-2}\mathrm{Mpc}^{2}]$'
# ylabel = r'$P(k, a) \times 10^{4}\;\;[h^{-2}\mathrm{Mpc}^{2}]$'
# ylabel = r'$a^{-2}P(k, a) \; [10^{-4}L^{2}]$'
# ylabel = r'$a^{-2}kP(k, a) \times 10^{3}$'
ylabel = r'$a^{-2}L^{-1}P(k, a) \times 10^{4}$'


errors = [(yaxis - yaxes[0]) * 100 / yaxes[0] for yaxis in yaxes[1:]]
print(errors[0][-4:], errors[1][-4:])
# err_zel = (P_zel - P_nb[:int(P_zel.size)]) * 100 / P_nb[:int(P_zel.size)]
# errors.append(err_zel)

# print('a = {}, err = {}'.format(errors[1]))
if sm == True:
    # title = r'$k = {}\,k_{{\mathrm{{f}}}},\; \Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)
    title = r'$k = k_{{\mathrm{{f}}}},\; \Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt)

else:
    # title = r'$k = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(mode)
    title = r'$k = {}\,k_{{\mathrm{{f}}}}$'.format(mode)

save = True
dashes = [None]*len(yaxes)
plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc=a_sc, which='', title_str=title, error_plotting=True, zel=False, save=save, leg=leg, dashes=dashes)
