#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import pandas
import pickle
import numpy as np
from functions import plotter, initial_density, SPT_real_tr, smoothing, alpha_to_corr, alpha_c_finder, dc_in_finder, dn, read_sim_data, read_density
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


kind = 'sharp'
kind_txt = 'sharp cutoff'
kind = 'gaussian'
kind_txt = 'Gaussian smoothing'
path = 'cosmo_sim_1d/sim_k_1_11/run1/'
plots_folder =  '../plots/test/sim_k_1_11/real_space/{}/'.format(kind)
Nfiles = 50
mode = 1
Lambda = 3 * (2 * np.pi)
H0 = 100
A = [-0.05, 1, -0.5, 11, 0]

# a_list, x, alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, err_Int = alpha_c_finder(Nfiles, Lambda, path, A, mode, kind, n_runs=8, n_use=6, H0=100, fde_method='percentile')
# df = pandas.DataFrame(data=[a_list, x, alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, err_Int])
# file = open("./data/alpha_c_{}.p".format(kind), "wb")
# pickle.dump(df, file)
# file.close()

file = open("./data/alpha_c_{}.p".format(kind), "rb")
read_file = pickle.load(file)
a_list, x, alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, err_Int = np.array(read_file)
file.close()

flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

Nx = x.size
L = 1.0
k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
dc_in, k_in = dc_in_finder(path, x, interp=True)
x = np.arange(0, 1, 1/k.size)
a_list = a_list[:Nfiles]
alpha_c_true_list = alpha_c_true_list[:Nfiles]
alpha_c_list = alpha_c_list[:Nfiles]
alpha_c_list2 = alpha_c_list2[:Nfiles]
alpha_c_list3 = alpha_c_list3[:Nfiles]

j = 22
a = a_list[j]
alpha_c = alpha_c_list[j]

dc_in = smoothing(dc_in, k, Lambda, kind) #truncating the initial overdensity
F = dn(3, L, dc_in)
d1k = (np.fft.fft(F[0]))
d2k = (np.fft.fft(F[1]))
d3k = (np.fft.fft(F[2]))
d3k_corr = alpha_c * (k**2) * d1k * a
dc_k3 = a*d1k + (a**2)*d2k + (a**3)*(d3k)
den_k = dc_k3 + d3k_corr
den_eft = np.real(np.fft.ifft(den_k))

den_spt_tr = SPT_real_tr(dc_in, k, L, Lambda, a, kind)

dk_par, a, dx = read_density(path, j)
k_par = np.fft.ifftshift(2.0 * np.pi * np.arange(-dk_par.size/2, dk_par.size/2))
M0_par = np.real(np.fft.ifft(dk_par))
M0_par = (M0_par / np.mean(M0_par)) - 1
M0_par = smoothing(M0_par, k_par, Lambda, kind)
M0_k = np.real(np.fft.fft(M0_par) / M0_par.size)
den_nbody = M0_par + 1
# P_nb, P_1l = read_sim_data(path, Lambda, kind, j)[-2:]
#
#
# k_par /= 2*np.pi
# k_nb = np.fft.ifftshift(np.arange(-P_nb.size/2, P_nb.size/2))
#
# print(P_nb.size, k.size)
# plt.scatter(k_par, M0_k, c='b', s=50)
# plt.scatter(k_nb, P_nb, c='k', s=20)
# plt.show()





plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [3, 1]})
ax[0].set_title(r'$a = {}$'.format(np.round(a, 3)), fontsize=18, y=1.01)
ax[1].set_xlabel(r'$x/L$', fontsize=20)
ax[0].set_ylabel(r'$1 + \delta_{l}(x)$', fontsize=20)
ax[1].set_ylabel(r'% err', fontsize=20)

yaxes = [1+den_nbody, 1+den_spt_tr, 1+den_eft]
# yaxes = [smoothing(axis, k, (2*np.pi), 'sharp') for axis in yaxes]
xaxis = x
colors = ['b', 'brown', 'k']
linestyles = ['solid', 'dashdot', 'dashed']
handles = []
labels=[r'$N-\mathrm{body}$', r'tSPT-4', r'EFT: from fit to $\langle[\tau]_{\Lambda}\rangle$']

plots_folder = '../plots/test/new_paper_plots/'
savename = 'alpha_c_{}'.format(kind)

err = [((yaxis - yaxes[0]) * 100 / yaxes[0]) for yaxis in yaxes]


for i in range(len(yaxes)):
    line, = ax[0].plot(xaxis, yaxes[i], c=colors[i], ls=linestyles[i], lw=2)
    ax[1].axhline(0, c=colors[0], ls=linestyles[0], lw=2)
    if i > 0:
        err_line, = ax[1].plot(xaxis, err[i], c=colors[i], ls=linestyles[i], lw=2)

    handles.append(line)

ax[0].legend(handles, labels, fontsize=13, framealpha=1, bbox_to_anchor=(1,1))

for j in range(2):
    ax[j].minorticks_on()
    ax[j].tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax[j].yaxis.set_ticks_position('both')
plt.show()
