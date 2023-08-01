#!/usr/bin/env python3
import numpy as np
import h5py as hp
import matplotlib.pyplot as plt
import pandas
import pickle
from functions import read_sim_data, plotter, param_calc_ens, smoothing, spec_from_ens
from zel import *

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# path = 'cosmo_sim_1d/final_phase_run1/'
path = 'cosmo_sim_1d/sim_k_1_11/run1/'

# path = 'cosmo_sim_1d/sim_k_1_15/run1/'
# path = 'cosmo_sim_1d/sim_3_15/run1/'
# path = 'cosmo_sim_1d/multi_sim_3_15_33/run1/'
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
plots_folder = '/paper_plots_final/'

# plots_folder = '/sim_k_1/'
# plots_folder = '/test/multi_sim_3_15_33/'
# plots_folder = '/new_sim_k_1_11/'
sim_num = '1_11'

Nfiles = 51
mode = 1
Lambda = 3 * (2 * np.pi)
Lambda_int = int(Lambda / (2*np.pi))
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

leg = True#False
H0 = 100
n_use = 8
zel = False
modes = True
folder_name = '' # '/new_data_{}/L{}'.format('sharp', Lambda_int)
save = True
# fitting_method = 'WLS'
# fitting_method = 'lmfit'
fitting_method = 'curve_fit'            
# fitting_method = ''
nbins_x, nbins_y, npars = 10, 10, 3

folder_name = '/new_hier/data_{}/L{}'.format(kind, int(Lambda/(2*np.pi)))
# folder_name = '/data_even_coarser/'
flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

#for plotting the spectra
# a_list, x, P_nb, P_1l_tr, P_eft_tr, P_eft2_tr, P_eft3_tr, P_eft4_tr, P_eft_fit = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, folder_name=folder_name)
# yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2, P_eft2_tr / a_list**2, P_eft3_tr / a_list**2, P_eft4_tr / a_list**2, P_eft_fit / a_list**2]

a_list, x, P_nb, P_1l_tr, P_eft_F3P, P_eft_F6P, P_eft_MW, P_eft_SC, P_eft_SCD, P_eft_true = spec_from_ens(Nfiles, Lambda, path, mode, kind, n_runs, n_use, folder_name=folder_name)
yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_F3P / a_list**2, P_eft_F6P / a_list**2, P_eft_MW / a_list**2, P_eft_SC / a_list**2, P_eft_SCD / a_list**2, P_eft_true / a_list**2]
# yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_F3P / a_list**2, P_eft_SC / a_list**2, P_eft_true / a_list**2]

# print(((P_eft_SC - P_nb) * 100 / P_nb)[-1])
# print(((P_eft_SC - P_nb) * 100 / P_nb)[-1])
# print(((P_1l_tr - P_nb) * 100 / P_nb)[-1])

xaxis = a_list
# # for spec in yaxes:
#     # spec /= ((1e-4))
#     # spec *= (2*np.pi)**2
# err_Int /= (1e-4 * a_list**2)


# alpha_c_fit_recon = (P_eft_tr[-1] - P_1l_tr[-1]) / (2 * (6*np.pi)**2) / (2.34e-5)
# alpha_c_mw_recon = (P_eft2_tr[-1] - P_1l_tr[-1]) / (2 * (6*np.pi)**2) / (2.34e-5)
# alpha_c_true_recon = (P_eft_fit[-1] - P_1l_tr[-1]) / (2 * (6*np.pi)**2) / (2.34e-5)
#
# print(alpha_c_fit_recon, alpha_c_mw_recon, alpha_c_true_recon)

file = open(rf"./{path}/spec_plot_{kind}.p", 'wb')
df = pandas.DataFrame(data=[xaxis, yaxes])
pickle.dump(df, file)
file.close()


#
# file = open("new_spec_plot_{}_L{}_k{}.p".format(kind, int(Lambda/(2*np.pi)), mode), 'rb')
# read_file = pickle.load(file)
# mode, Lambda, xaxis, yaxes, err_Int = np.array(read_file[0])
# file.close()
# # yaxes = [np.array(axis) for axis in yaxes]

# yaxes_ = yaxes
# yaxes = []
# for axis in yaxes_:
#     axis = np.delete(np.array(axis), 22)
#     yaxes.append(axis)
#
# xaxis = np.delete(xaxis, 22)
# err_Int = np.delete(err_Int, 22)

new_yaxes = []
for axes in yaxes:
    new_yaxes.append(axes[:Nfiles])
yaxes = new_yaxes
xaxis = xaxis[:Nfiles]

for spec in yaxes:
    spec *= 1e4

# file = open("./data/spectra_2l_k{}_{}.p".format(mode, kind), "rb")
# read_file = pickle.load(file)
# xaxis, yaxes = np.array(read_file)
# yaxes = yaxes[0], yaxes[1], yaxes[2]
# print(yaxes)

# yaxes.pop(5)

a_sc = 0# 1 / np.max(initial_density(x, A, 1))

print(kind)
    # colours = ['b', 'brown', 'k', 'cyan', 'orange', 'xkcd:dried blood', 'r', 'seagreen']
    # colours = ['b', 'brown', 'k', 'cyan', 'orange', 'lightseagreen', 'r', 'seagreen']
# colours = ['b', 'brown', 'k', 'cyan', 'magenta', 'orange', 'g', 'seagreen']

# labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle\tau\rangle$',  r'EFT: M\&W', 'EFT: $B^{+12}$', r'EFT: from matching $P_{N-\mathrm{body}}$', 'Zel']
# labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle\tau\rangle$',  r'EFT: M\&W', 'EFT: $B^{+12}$', r'EFT: DDE', r'EFT: from matching $P_{N\mathrm{-body}}$']#, 'Zel']
# labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle\tau\rangle$',  r'EFT: M\&W', 'EFT: Spatial Corr', r'EFT: Spatial Corr from $\delta_{\ell}$', r'EFT: from matching $P_{N\mathrm{-body}}$']#, 'Zel']

if kind == 'sharp':
    colours = ['b', 'brown', 'k', 'seagreen', 'midnightblue', 'magenta', 'orange', 'r']
    labels = [r'$N$-body', 'tSPT', r'EFT: F3P',  r'EFT: F6P', r'EFT: M\&W', 'EFT: SC', r'EFT: SC$\delta$', r'EFT: matching $P_{N\mathrm{-body}}$']#, 'Zel']
    linestyles = ['solid', 'dashdot', 'dashed', 'dashed', 'dashed', 'dashed', 'dashed', 'dotted']
    dashes = [None, None, None, [1, 2, 1], [2, 1, 2], [2, 2, 1], [1, 1, 2], None]

elif kind == 'gaussian':
    yaxes.pop(3)
    colours = ['b', 'brown', 'k', 'midnightblue', 'magenta', 'orange', 'r']
    labels = [r'$N$-body', 'tSPT-4', r'EFT: F3P',  r'EFT: M\&W', 'EFT: SC', r'EFT: SC$\delta$', r'EFT: matching $P_{N\mathrm{-body}}$']#, 'Zel']
    linestyles = ['solid', 'dashdot', 'dashed', 'dashed', 'dashed', 'dashed', 'dotted']
    dashes = [None, None, None, [2, 1, 2], [2, 2, 1], [1, 1, 2], None]

# if kind == 'sharp':
#     colours = ['b', 'brown', 'k', 'magenta', 'g']
#     labels = [r'$N$-body', 'tSPT-4', r'EFT: F3P',  r'EFT: SC', r'EFT: matching $P_{N\mathrm{-body}}$']#, 'Zel']
#     linestyles = ['solid', 'dashdot', 'dashed', 'dashed', 'dotted']
#     dashes = [None, None, None, [2, 2, 1], None]

# elif kind == 'gaussian':
#     # colours = ['b', 'brown', 'k', 'cyan', 'orange', 'r', 'seagreen']
#     # labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle\tau\rangle$',  r'EFT: M\&W', 'EFT: $B^{+12}$', r'EFT: from matching $P_{N-\mathrm{body}}$', 'Zel']
#     labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle\tau\rangle$',  r'EFT: M\&W', 'EFT: $B^{+12}$', r'EFT: from matching $P_{N\mathrm{-body}}$']#, 'Zel']
#     linestyles = ['solid', 'dashdot', 'dashed', 'dashed', 'dashed', 'dotted', 'dotted']
#     yaxes.pop(5)

# else:
#     pass

# labels = [r'$N$-body', 'tSPT-4', r'tSPT-6']#, 'Zel']

# yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2, P_eft2_tr / a_list**2, P_eft_fit / a_list**2]


# P_1l_corr = P_eft_tr - P_1l_tr

# df = pandas.DataFrame(data=[P_1l_corr, a_list])
# pickle.dump(df, open('./data/1l_corr.p', 'wb'))


# colours = ['b', 'brown', 'k', 'cyan', 'r', 'seagreen']
# labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle[\tau]_{\Lambda}\rangle$',  r'EFT: M\&W', r'EFT: from matching $P_{N\mathrm{-body}}$']#, 'Zel']
# linestyles = ['solid', 'dashdot', 'dashed', 'dashed', 'dotted', 'dotted']

# yaxes = [axis*1e4 for axis in yaxes]

# yaxes = yaxes[:2]
# colours = colours[:2]
# labels = labels[:2]
# linstyles = linestyles[:2]

# #without fourth estimator
# # yaxes = [P_nb * 1e4 / a_list**2, P_1l_tr * 1e4  / a_list**2, P_eft_tr * 1e4  / a_list**2, P_eft2_tr * 1e4  / a_list**2, P_eft3_tr * 1e4  / a_list**2, P_eft_fit * 1e4  / a_list**2]
# colours = ['b', 'brown', 'k', 'cyan', 'orange', 'r', 'seagreen']
# labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle[\tau]_{\Lambda}\rangle$',  r'EFT: M\&W', 'EFT: $B^{+12}$', r'EFT: from matching $P_{\mathrm{N-body}}$', 'Zel']
# linestyles = ['solid', 'dashdot', 'dashed', 'dashed', 'dashed', 'dotted', 'dotted']


# savename = '{}_eft_spectra_k{}_L{}_{}'.format(sim_num, mode, int(Lambda/(2*np.pi)), kind)
savename = 'eft_spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
# savename = 'eft_spectra_k{}_L{}_{}_talk'.format(mode, int(Lambda/(2*np.pi)), kind)

# savename = 'test_eft_spectrum'

xlabel = r'$a$'
ylabel = r'$a^{-2}L^{-1}P(k, a) \times 10^{4}$'
# ylabel = r'$a^{-2}kP(k, a) \times 10^{3}$'

# ylabel = r'$a^{-2}P(k, a) \times 10^{4}\;\;[h^{-2}\mathrm{Mpc}^{2}]$'
# ylabel = r'$a^{-2}P(k, a) \; [10^{-4}L^{2}]$'
# ylabel = r'$a^{-2}k^{2}P(k, a)$'

# ylabel = r'$|\tilde{\delta}(k=1, a)|^{2}\; / a^{2}$'
# title = r'$k = {}\,k_{{\mathrm{{f}}}},\; \Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)
# title = r'$k = {}\,k_{{\mathrm{{f}}}},\; \Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)
# title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, Lambda_int, kind_txt)
title = r'$k = k_{{\mathrm{{f}}}},\; \Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt)

save = True

if kind == 'sharp':
    yaxes.pop(7)
    labels.pop(7)

    # for j in [6, 5, 3]:
    #     yaxes.pop(j)
    #     labels.pop(j)
    #     colours.pop(j)
    #     linestyles.pop(j)


elif kind == 'gaussian':
    yaxes.pop(6)
    labels.pop(6)


err_Int = []
plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc=a_sc, title_str=title, terr=err_Int, zel=zel, save=save, leg=leg, flags=flags, dashes=dashes)


# # f1 = 'curve_fit'
# # # f1 = 'weighted_ls_fit'
# #
# # #for plotting the spectra
# # a_list, x, P_nb, P_1l_tr, P_eft_tr, P_eft2_tr, P_eft3_tr, P_eft_fit, err_Int = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, H0, zel, folder_name, modes, fitting_method=f1)
# # yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2]
# # for spec in yaxes:
# #     spec /= 1e-4
# # err_Int /= (1e-4 * a_list**2)
# #
# # xaxis = a_list
# # a_sc = 0# 1 / np.max(initial_density(x, A, 1))
# #
# # colours = ['b', 'brown', 'k']
# # labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle[\tau]_{\Lambda}\rangle$']
# # linestyles = ['solid', 'dashdot', 'dashed']
# # # savename = 'eft_spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
# # # savename = 'sim_old'
# # savename = 'sim_new'
# #
# # xlabel = r'$a$'
# # # ylabel = r'$a^{-2}P(k=1, a) \times 10^{4}$'
# # ylabel = r'$a^{-2}P(k, a) \times 10^{4}\;\;[h^{-2}\mathrm{Mpc}^{2}]$'
# #
# # # ylabel = r'$|\tilde{\delta}(k=1, a)|^{2}\; / a^{2}$'
# # title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, Lambda_int, kind_txt)
# # #             # err_str = r'$C_{{0}} = {}$'.format(np.round(C_list[i][j][0], 3)) + '\n' + r'$C_{{1}} = {}$'.format(np.round(C_list[i][j][1], 3)) + '\n' + r'$C_{{2}} = {}$'.format(np.round(C_list[i][j][2], 3))
# #
# # text_str = r'$N_{{\mathrm{{runs}}}} = {}$'.format(n_runs) + '\n' + r'$N_{{\mathrm{{points}}}} = {}$'.format(n_use)
# # text_loc = (0.35, 0.05)
# #
# # texts = [text_str, text_loc]
# # plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc=a_sc, title_str=title, terr=err_Int, zel=zel, save=save, leg=leg, texts=texts)

# # #for plotting the spectra
# # a_list, x, P_nb, P_1l_tr, P_eft_tr, P_eft2_tr, P_eft3_tr, P_eft_fit, err_Int = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, H0, zel, folder_name, modes, fitting_method)
# # yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2, P_eft2_tr / a_list**2, P_eft_fit / a_list**2]
# # for spec in yaxes:
# #     spec /= 1e-4
# # err_Int /= (1e-4 * a_list**2)
# #
# # xaxis = a_list
# # a_sc = 0# 1 / np.max(initial_density(x, A, 1))
# #
# # colours = ['b', 'brown', 'k', 'cyan', 'g']
# # labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle[\tau]_{\Lambda}\rangle$',  r'EFT: M\&W', r'EFT: from matching $P_{\mathrm{N-body}}$', 'Zel']
# # linestyles = ['solid', 'dashdot', 'dashed',  'dashed', 'dotted']
# # savename = 'eft_spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
# # xlabel = r'$a$'
# # # ylabel = r'$a^{-2}P(k=1, a) \times 10^{4}$'
# # ylabel = r'$a^{-2}P(k, a) \times 10^{4}\;\;[h^{-2}\mathrm{Mpc}^{2}]$'
# #
# # # ylabel = r'$|\tilde{\delta}(k=1, a)|^{2}\; / a^{2}$'
# # title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, Lambda_int, kind_txt)
# #
# # plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc=a_sc, title_str=title, terr=err_Int, zel=zel, save=save, leg=leg)
# #
# # df = pandas.DataFrame(data=[mode, Lambda, xaxis, yaxes, errors, err_Int, a_sc])
# # pickle.dump(df, open("spec_plot_{}_L{}.p".format(kind, int(Lambda/(2*np.pi))), "wb"))
# # for Lambda in range(2, 7):
# #    Lambda *= (2*np.pi)
# # [mode, Lambda, xaxis, yaxes, errors, err_Int, a_sc] = pickle.load(open("spec_plot_{}_L{}.p".format(kind, int(Lambda/(2*np.pi))), "rb" ))[0]
