#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import pandas
import pickle
import numpy as np
from functions import plotter, initial_density, SPT_real_tr, smoothing, alpha_c_finder, dc_in_finder, dn, read_sim_data
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# path = 'cosmo_sim_1d/multi_k_sim/run1/'

Nfiles = 23
mode = 1
Lambda = 3 * (2 * np.pi)
H0 = 100
A = []
# folder_name = '/test_hier/'

folder_name = '/new_hier/data_{}/L{}/'.format(kind, int(Lambda/(2*np.pi)))
N_bins = 10000

# folder_name = '/data_even_coarser/'
# N_bins = 1000

# folder_name = '/data_coarse/'
# N_bins = 100

fde = 'percentile'

npars = 3
a_list, x, alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, alpha_c_list4, err_Int = alpha_c_finder(Nfiles, Lambda, path, A, mode, kind, n_runs=8, n_use=10, H0=100, fde_method=fde, folder_name=folder_name, npars=npars)
# # df = pandas.DataFrame(data=[a_list, x, alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, alpha_c_list4, err_Int])
# # file = open("./data/new_trunc_alpha_c_{}.p".format(kind), "wb")
# # pickle.dump(df, file)
# # file.close()
# alpha_c_6par = alpha_c_list
#
# file = open("./data/new_trunc_alpha_c_{}.p".format(kind), "rb")
# read_file = pickle.load(file)
# a_list, x, alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, alpha_c_list4, err_Int = np.array(read_file)
# file.close()

c0 = 0.39729
# c0 = 1.5894816698814516
print(a_list[0])
Q1 = (0.5**(9/2) / a_list[:Nfiles]**(5/2)) * (2/9)
Q2 = (0.5**2) / 2
c1 = (2*c0 / (5*H0**2)) * ((Q1 - Q2))

flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

Nx = x.size
L = 1.0
k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
dc_in, k_in = dc_in_finder(path, x, interp=True)
dc_in = smoothing(dc_in, k, Lambda, kind) #truncating the initial overdensity
F = dn(3, L, dc_in)
d1k = np.real(np.fft.fft(F[0])) / k.size

a_list = a_list[:Nfiles]
alpha_c_true_list = alpha_c_true_list[:Nfiles] #* 2 * k[mode]**2 * (d1k[mode] * a_list)**2
alpha_c_list = alpha_c_list[:Nfiles] + c1 #* 2 * k[mode]**2 * (d1k[mode] * a_list)**2
alpha_c_list2 = alpha_c_list2[:Nfiles] +c1 #* 2 * k[mode]**2 * (d1k[mode] * a_list)**2
alpha_c_list3 = alpha_c_list3[:Nfiles] + c1 #* 2 * k[mode]**2 * (d1k[mode] * a_list)**2

err_Int = err_Int[:Nfiles] #* 2 * k[mode]**2 * (d1k[mode] * a_list)**2

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(9,6))
ax.set_title(r'$k = k_{{\mathrm{{f}}}}, \Lambda = {}\,k_{{\mathrm{{f}}}}$ ({}), $N_{{\mathrm{{bins}}}} = {}$'.format(int(Lambda/(2*np.pi)), kind_txt, N_bins), fontsize=20, y=1.01)
ax.set_xlabel(r'$a$', fontsize=20)
# ax.set_ylabel(r'$2 (k/k_{f})^{2}\alpha_{c} P_{\mathrm{lin}} \; [10^{4}\;L^{2}]$', fontsize=20)
ax.set_ylabel(r'$\alpha_{c} \;[10^{-4}L^{2}]$', fontsize=20)

# for j in range(Nfiles):
#     if flags[j] == 1:
#         sc_line = ax.axvline(a_list[j], c='teal', lw=0.5)
#     else:
#         pass

yaxes = [alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, alpha_c_list4]

yaxes = [(yaxis * 1e4) for yaxis in yaxes]

errors = [(100 * (yaxes[j] - yaxes[0]) / yaxes[0]) for j in range(len(yaxes))]

alpha_c_list4 = alpha_c_list4[:Nfiles] * 1e4 #* 2 * k[mode]**2 * (d1k[mode] * a_list)**2
err_Int *= 1e4

xaxis = a_list
colors = ['g', 'k', 'cyan', 'orange', 'xkcd:dried blood']
linestyles = ['solid', 'dashed', 'dashed', 'dashed', 'dashed']
# handles = [sc_line]
# labels=[r'$a_\mathrm{sc}$', r'from matching $P_{N-\mathrm{body}}$', r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$']
handles = []
labels=[r'from matching $P_{N-\mathrm{body}}$', r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'DDE']


plots_folder = '../plots/test/new_paper_plots/'
savename = 'alpha_c_{}'.format(kind)

for i in range(len(yaxes)):
    line, = ax.plot(xaxis, yaxes[i], c=colors[i], ls=linestyles[i], lw=2)
    handles.append(line)

# err_line, = ax.plot(a_list, alpha_c_list4, c='xkcd:dried blood', ls='dashed', lw=2)
# ctot2_4_err = ax.fill_between(a_list, alpha_c_list4-err_Int, alpha_c_list4+err_Int, color='darkslategray', alpha=0.55)
# handles.append((err_line, ctot2_4_err))

# c = (alpha_c_list[5] - alpha_c_list[4]) / (a_list[5] - a_list[4]) * 1e4
# b = -c*a_list[0]**2

# from scipy.optimize import curve_fit

# def fitting_function(X, a0, a1):
#     return a0*X[0] + a1*(X[1]**2)
# X = (np.ones(a_list[2:].size), a_list[2:])

# guesses = -1, 1
# C, cov = curve_fit(fitting_function, X, alpha_c_list2[2:], method='lm', absolute_sigma=True)
# C0, C1 = C

# new_a_list = a_list[2:]#np.arange(0.5, 4, 0.25)
# # X = (np.ones(new_a_list.size), new_a_list)

# # print(C0, C1)
# # fit = fitting_function(X, C0, C1) * 1e4
# # pred_line, = ax.plot(new_a_list, (C0 + C1*a_list[2:]**2)*1e4, ls='dashdot', c='magenta', lw=1.5, zorder=0) #DDE
# # handles.append(pred_line)
# # labels[-1] = r'$\alpha_{c} \propto a^{2}$'

plt.legend(handles, labels, fontsize=14, framealpha=1, loc=3) #, bbox_to_anchor=(1,1))
# ax.set_xlim(0.4, 3.2)
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.yaxis.set_ticks_position('both')
# plt.show()

# plt.savefig('../plots/test/new_paper_plots/alpha_c_{}_{}.png'.format(kind, N_bins), bbox_inches='tight', dpi=150)
plt.savefig('../plots/test/new_paper_plots/alpha_c_{}.pdf'.format(kind), bbox_inches='tight', dpi=150)
plt.close()
