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
path = 'cosmo_sim_1d/sim_k_1_7/run1/'
# path = 'cosmo_sim_1d/multi_k_sim/run1/'

plots_folder =  '../plots/test/sim_k_1_11/real_space/{}/'.format(kind)
Nfiles = 50
mode = 1
Lambda = 3 * (2 * np.pi)
H0 = 100
A = [-0.05, 1, -0.5, 11, 0]

folder_name = '/new_hier/data_{}/L{}/'.format(kind, int(Lambda/(2*np.pi)))

a_list, x, alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, alpha_c_list4, err_Int = alpha_c_finder(Nfiles, Lambda, path, A, mode, kind, n_runs=8, n_use=6, H0=100, fde_method='percentile', folder_name=folder_name)
df = pandas.DataFrame(data=[a_list, x, alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, err_Int])
file = open("./data/alpha_c_{}.p".format(kind), "wb")
pickle.dump(df, file)
file.close()

file = open("./data/alpha_c_{}.p".format(kind), "rb")
read_file = pickle.load(file)
a_list, x, alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, err_Int = np.array(read_file)
file.close()

# flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

j = 15
Nx = x.size
L = 1.0
k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
dk_par, a, dx = read_density(path, j)
x_grid = np.arange(0, L, dx)
k = np.fft.ifftshift(2.0 * np.pi * np.arange(-dk_par.size/2, dk_par.size/2)) / (2*np.pi)

dk_par /= 125
den_nbody = smoothing((np.real(np.fft.ifft(dk_par)))-1, k, Lambda, kind)
x = np.arange(0, 1, 1/dk_par.size)
dc_in, k_in = dc_in_finder(path, x, interp=True)

a_list = a_list[:Nfiles]
alpha_c_true_list = alpha_c_true_list[:Nfiles]
alpha_c_list = alpha_c_list[:Nfiles]
alpha_c_list2 = alpha_c_list2[:Nfiles] #M&W
alpha_c_list3 = alpha_c_list3[:Nfiles] #B12

a = a_list[j]
alpha_c = alpha_c_list2[j]


moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
moments_file = np.genfromtxt(path + moments_filename)
a = moments_file[:,-1][0]
x_cell = moments_file[:,0]
# M0_nbody = moments_file[:,2]-1
# M0_nbody -= np.mean(M0_nbody)
# den_eft = alpha_to_corr(alpha_c, a, x, k, L, dc_in, Lambda, kind)
# den_nbody = smoothing(M0_nbody, k, Lambda, kind)
den_spt_tr = SPT_real_tr(dc_in, k, L, Lambda, a, kind)

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=False, gridspec_kw={'width_ratios': [1], 'height_ratios': [3, 1]})
# ax[0].axhline(1, c='grey', lw=1)
ax[0].set_title(r'a = {}, $\Lambda = {}\;k_{{\mathrm{{f}}}}$ ({})'.format(a, int(Lambda / (2 * np.pi)), kind_txt), fontsize=18, y=1.01)
ax[1].set_xlabel(r'$x/L$', fontsize=20)
ax[0].set_ylabel(r'$1 + \delta_{l}$', fontsize=20)
ax[1].set_ylabel(r'\% err', fontsize=20)


# yaxes = [1+den_nbody, 1+den_spt_tr, 1+den_eft]
# # yaxes = [smoothing(axis, k, (2*np.pi), 'sharp') for axis in yaxes]
# xaxis = x
# colors = ['b', 'brown', 'k']
# linestyles = ['solid', 'dashdot', 'dashed']
# handles = []
# labels=[r'$N-\mathrm{body}$', r'tSPT-4', r'EFT: from fit to $[\tau]_{\Lambda}$']


def mode_sep(arr, m):
    arr = np.fft.fft(arr)
    for j in range(arr.size):
        if j == m:# or j == arr.size-m:
            pass
        else:
            arr[j] = 0
    return np.real(np.fft.ifft(arr))

den_nbody1 = mode_sep(den_nbody, 1)
den_nbody2 = mode_sep(den_nbody, 2)
den_nbody3 = mode_sep(den_nbody, 3)

den_spt_tr1 = mode_sep(den_spt_tr, 1)
den_spt_tr2 = mode_sep(den_spt_tr, 2)
den_spt_tr3 = mode_sep(den_spt_tr, 3)

# den_eft1 = mode_sep(den_eft, 1)
# den_eft2 = mode_sep(den_eft, 2)
# den_eft3 = mode_sep(den_eft, 3)

# yaxes = [1+den_nbody, 1+den_nbody1, 1+den_nbody2, 1+den_nbody3, 1+den_spt_tr, 1+den_eft]
# # yaxes = [smoothing(axis, k, (2*np.pi), 'sharp') for axis in yaxes]
# xaxis = x
# colors = ['b', 'r', 'cyan', 'seagreen', 'brown', 'k']
# linestyles = ['solid', 'dashed', 'dashed', 'dashed', 'dashdot', 'dotted']
# handles = []
# labels=[r'$N-\mathrm{body}$: full', r'$N-\mathrm{body}: k=1$', r'$N-\mathrm{body}: k=2$', r'$N-\mathrm{body}: k=3$', r'tSPT-4', r'EFT: from fit to $[\tau]_{\Lambda}$']

yaxes = [1+den_nbody1, 1+den_spt_tr1]#, 1+den_eft3]
xaxis = x
colors = ['b', 'brown', 'k']
linestyles = ['solid', 'dashdot', 'dotted']
handles = []
labels=[r'$N-\mathrm{body}$', r'tSPT-4', r'EFT']

# labels=[r'$N-\mathrm{body}: k=1$', r'tSPT-4: $k=1$', r'EFT: $k=1$']
# labels=[r'$N-\mathrm{body}: k=2$', r'tSPT-4: $k=2$', r'EFT: $k=2$']


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

err_den = (np.real(np.fft.fft(den_spt_tr1)[1]) - np.real(np.fft.fft(den_nbody1))[1]) * 100 / np.real(np.fft.fft(1+den_nbody1)[1])
print(err_den)

err_pow = (np.real(np.fft.fft(den_spt_tr1)[1]**2) - np.real(np.fft.fft(den_nbody1)[1])**2) * 100 / np.real(np.fft.fft(1+den_nbody1)**2)[1]
print(err_pow)

plt.show()

# plt.savefig('../plots/test/new_paper_plots/real_den_comp_{}_k3.png'.format(kind), bbox_inches='tight', dpi=150)
# # # plt.savefig('../plots/test/new_paper_plots/real_den_comp_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# plt.close()



# W_EFT = EFT_sm_kern(k, Lambda)
#
# xlabel = r'$x\;[h^{-1}\mathrm{Mpc}]$'
# ylabel = r'$1+\delta_{l}(x)$'
# colours = ['b', 'brown', 'k']
# labels = [r'$N$-body', r'tSPT-4', r'EFT: from fit to $[\tau]_{\Lambda}']
# linestyles = ['solid', 'dashdot', 'dashed']
# save = True
# leg = True
#
# a_sc = 0
#
# # j = 15
# for j in range(0, 50):
#     a = a_list[j]
#     print('a = ', a)
#     alpha_c = alpha_c_list[j]
#     moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
#     moments_file = np.genfromtxt(path + moments_filename)
#     a = moments_file[:,-1][0]
#     x_cell = moments_file[:,0]
#     M0_nbody = moments_file[:,2]
#     den_eft, err_eft = alpha_to_corr(alpha_c, a, x, k, L, dc_in, Lambda, kind, err_Int)
#     den_spt_tr = SPT_real_tr(dc_in, k, L, Lambda, a, kind)
#     den_nbody = smoothing(M0_nbody-1, k, Lambda, kind)
#
#     xax[0]is = x
#     yax[0]es = [den_nbody+1, den_spt_tr+1, den_eft+1]
#     title = 'a = {}, $\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(a, int(Lambda / (2 * np.pi)), kind_txt)
#     savename = 'eft_real_{}_{}'.format(kind, j)
#     print(plots_folder + savename)
#     plotter(mode, Lambda, xax[0]is, yax[0]es, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc=a_sc, title_str=title, save=save, leg=leg)
