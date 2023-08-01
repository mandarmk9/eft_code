#!/usr/bin/env python3
import h5py
import pickle
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from functions import dc_in_finder, smoothing, dn, param_calc_ens, alpha_c_finder
from tqdm import tqdm

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
mode = 1
A = [-0.05, 1, -0.5, 11]
kind = 'sharp'
kind_txt = 'sharp cutoff'
kind = 'gaussian'
kind_txt = 'Gaussian smoothing'
n_runs = 8
n_use = 8
Lambda_list = np.arange(3, 8)
# nums = [10, 15, 23]
nums = [0, 14, 50]

nbins_x, nbins_y, npars = 10, 10, 3
fm = 'curve_fit'

ctot2_0_list, ctot2_1_list, ctot2_2_list, ctot2_3_list, ctot2_4_list = [[], [], []],  [[], [], []], [[], [], []], [[], [], []], [[], [], []]
a_list = []

for j in range(len(nums)):
    c0, c1, c2, c3, c4 = [], [], [], [], []
    file_num = nums[j]
    for i in range(Lambda_list.size):
        Lambda = Lambda_list[i] * (2*np.pi)
        folder_name = '/new_hier/data_{}/L{}/'.format(kind, int(Lambda/(2*np.pi)))
        sol = param_calc_ens(file_num, Lambda, path, A, mode, kind, n_runs, n_use, fitting_method=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars, folder_name=folder_name, per=46.6)
        a = float(sol[0])
        c0.append(float(sol[2])) #Fit
        c1.append(float(sol[3])) #M&W
        c2.append(float(sol[4])) #B+12
        c3.append(float(sol[-2])) #spatial corr
        c4.append(float(sol[-1])) #spatial corr from \delta

        # err.append(float(sol[-1]))
    print('a = ', a)

    a_list.append(a)
    ctot2_0_list[j] = c0 #Fit
    ctot2_1_list[j] = c1 #M&W
    ctot2_2_list[j] = c2 #B+12
    ctot2_3_list[j] = c3 #spatial corr with \delta
    ctot2_4_list[j] = c4 #spatial corr with \delta

    # error_list[j] = err

# error_list = np.array(error_list)

file = open("./{}/ctot2_lambda_dep_{}.p".format(path, kind), 'wb')
df = pandas.DataFrame(data=[a_list, ctot2_0_list, ctot2_1_list, ctot2_2_list, ctot2_3_list, ctot2_4_list])
pickle.dump(df, file)
file.close()

file = open("./{}/ctot2_lambda_dep_{}.p".format(path, kind), 'rb')
read_file = pickle.load(file)
a_list, ctot2_0_list, ctot2_1_list, ctot2_2_list, ctot2_3_list, ctot2_4_list = np.array(read_file)
file.close()

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1]})

# ax[0].set_ylabel(r'$c^{2}_{\mathrm{tot}}\;[\mathrm{km^{2}\,s}^{-2}]$', fontsize=20)
# ax[1].set_xlabel(r'$\Lambda\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(kind_txt), fontsize=20)
# ax[2].set_ylabel(r'$c^{2}_{\mathrm{tot}}\;[\mathrm{km^{2}\,s}^{-2}]$', fontsize=20)

ax[0].set_ylabel('$c_{\mathrm{tot}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=24)
ax[1].set_xlabel(r'$\Lambda\;[k_{{\mathrm{{f}}}}]$ ({})'.format(kind_txt), fontsize=24)
ax[2].set_ylabel('$c_{\mathrm{tot}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=24)

# titles = [r'M\&W', r'$\mathrm{B^{+12}}$', r'from fit to $[\tau]_{\Lambda}$']
# titles = [r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$']
titles = [r'from fit to $\langle\tau\rangle$', r'M\&W', r'Spatial Corr']

linestyles = ['solid', 'dashdot', 'dashed']
ax[2].yaxis.set_label_position('right')
for i in range(3):

    # ax[i].set_title(r'$a = {}$'.format(np.round(a_list[i], 3)), fontsize=20)
    ax[i].set_title(titles[i], fontsize=24)

    ctot2_line, = ax[1].plot(Lambda_list, ctot2_1_list[i], c='c', lw=1.5, ls=linestyles[i], marker='*')#, label=r'M\&W')
    ctot2_2_line, = ax[2].plot(Lambda_list, ctot2_3_list[i], c='magenta', lw=1.5, ls=linestyles[i],  marker='v')#, label=r'$B^{+12}$')
    ctot2_3_line, = ax[0].plot(Lambda_list, ctot2_0_list[i], c='k', lw=1.5, ls=linestyles[i],  marker='o')#, label=r'fit to $[\tau]_{\Lambda}$')
    # ctot2_4_line, = ax[0].plot(Lambda_list, ctot2_3_list[i], c='xkcd:dried blood', lw=1.5, marker='+', ls=linestyles[i])#, label=r'FDE')
    # ctot2_4_err = ax[i].fill_between(Lambda_list, ctot2_3_list[i]-error_list[i], ctot2_3_list[i]+error_list[i], color='darkslategray', alpha=0.55, rasterized=True)

    # ax[i].errorbar(Lambda_list, ctot2_0_list[i], yerr=error_list[i], c='k', lw=1.5, marker='o', label=r'fit to $[\tau]_{\Lambda}$')

    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=22)
    ax[i].yaxis.set_ticks_position('both')
    #
    if kind == 'sharp':
        # ax[i].set_ylim(0.125, 0.875)
        anchor_x = 1.1
    #     ax[0,0].set_ylim(0.201, 0.25)
    #     ax[1,0].set_ylim(-0.25, 1.8)
    #     ax[0,1].set_ylim(0.3, 2.5)
    #     ax[1,1].set_ylim(0, 1.3)
    #
    elif kind == 'gaussian':
        anchor_x = 1.52
    #     # ax[1,0].set_ylim(0, 1.7)
    #     ax[0,1].set_ylim(-0.35, 1.5)
    #
    #
    # else:
    #     pass
    # ax[2].set_ylim(0.456, 0.595)

# print(type(ctot2_line), type(ctot2_4_err))
# plt.legend(handles=[ctot2_3_line, ctot2_line, ctot2_2_line, (ctot2_4_line, ctot2_4_err)], labels=[r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'FDE'], fontsize=14, loc=1, framealpha=1)
# plt.legend(handles=[ctot2_3_line, ctot2_line, ctot2_2_line, ctot2_4_line], labels=[r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'FDE'], fontsize=14, loc=3, framealpha=1)

# plt.legend(handles=[ctot2_3_line, ctot2_line, ctot2_2_line], labels=[r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, framealpha=1)

ax[1].tick_params(labelleft=False)
ax[2].tick_params(labelleft=False, labelright=True)

fig.align_labels()
# plt.subplots_adjust(hspace=0)
plt.subplots_adjust(wspace=0.05)

line1 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='solid')
line2 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashdot')
line3 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashed')
labels = ['a = {}'.format(a_list[0]), 'a = {}'.format(a_list[1]), 'a = {}'.format(a_list[2])]
handles = [line1, line2, line3]
plt.legend(handles=handles, labels=labels, fontsize=22, bbox_to_anchor=(1.67,1.05))
# plt.savefig('../plots/test/new_paper_plots/new_ctot2_lambda_dep_{}.png'.format(kind), bbox_inches='tight', dpi=150)
# plt.savefig('../plots/test/new_paper_plots/ctot2_lambda_dep_{}.png'.format(kind), bbox_inches='tight', dpi=150)
plt.savefig('../plots/paper_plots_final/ctot2_lambda_dep_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
plt.close()
# plt.show()