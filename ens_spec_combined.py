#!/usr/bin/env python3

#import libraries
import pickle
import pandas
import h5py

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from functions import plotter2, read_density, dn, smoothing, spec_nbody, spec_from_ens
from SPT import SPT_final
from zel import initial_density
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


mode = 1
H0 = 100
n_runs = 8
n_use = 10
zel = False
sm = True

Lambda_int = 3
Lambda = Lambda_int * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

file = open("./data/spec_comp_plot_y_{}_L{}.p".format(kind, int(Lambda/(2*np.pi))), "rb" )
data = pickle.load(file)
file.close()
# xaxes, yaxes = data[:,0], data[:,1]
xaxes = [data[j][0] for j in range(data.shape[1])]
yaxes = [data[j][1] for j in range(data.shape[1])]

# for j in range(len(yaxes)):
#     if j not in [2, 5, 8]:
#         yaxes[j] = np.delete(yaxes[j], 22)
#         xaxes[j] = np.delete(xaxes[j], 22)


plots_folder = '/paper_plots_final/' #test/new_paper_plots/'
a_sc = 1 #/ np.max(initial_density(x, A, 1))

if sm == True:
    title = r'$k = k_{{\mathrm{{f}}}},\; \Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt)
else:
    title = r'$k = k_{\mathrm{f}}$'

colours = ['b', 'r', 'k', 'magenta', 'r', 'k', 'magenta', 'r', 'k', 'magenta']


labels = []
patch1 = mpatches.Patch(color='b', label=r'\texttt{sim\_k\_1}')
patch2 = mpatches.Patch(color='r', label=r'\texttt{sim\_k\_1\_7}')
patch3 = mpatches.Patch(color='k', label=r'\texttt{sim\_k\_1\_11}')
patch4 = mpatches.Patch(color='magenta', label=r'\texttt{sim\_k\_1\_15}')
line1 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='solid', label='$N$-body')
line2 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashed', label='tSPT')
# line3 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashed', label=r'EFT: from fit to $\langle\tau\rangle$')
# line3 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dotted', label=r'EFT: from fit to $\langle\tau\rangle$')
line3 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dotted', label=r'EFT: SC')

handles = [patch1, patch2, patch3, patch4]#[patch1, patch2, patch3, patch4]
handles2 = [line1, line2, line3]

# linestyles = ['solid', 'solid', 'solid', 'dashdot', 'dashdot', 'dashed', 'dashed']#, 'solid']
linestyles = ['solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dotted', 'dotted', 'dotted']#, 'solid']

savename = 'spec_comp_k{}_L{}_{}'.format(mode, int(Lambda / (2*np.pi)), kind)
ylabel = r'$a^{-2}L^{-1}P(k, a) \times 10^{4}$'
# ylabel = r'$a^{-2}P(k, a) \; [10^{-4}L^{2}]$'
# ylabel = r'$a^{-2}k^{2}P(k, a)$'
# ylabel = r'$a^{-2}P(k, a)$'

err_x, err_y, err_c, err_ls = [], [], [], []
for j in range(len(yaxes)):
    # yaxes[j] *= 1e4 / xaxes[j]**2
    # yaxes[j] *= (2*np.pi)**2 / xaxes[j]**2
    yaxes[j] /= xaxes[j]**2

    if 3<j<7:
        err_y.append((yaxes[j] - yaxes[j-3]) * 100 / yaxes[j-2])
        err_x.append(xaxes[j])
        err_c.append(colours[j])
        err_ls.append(linestyles[j])

    elif 6<j<10:
        err_y.append((yaxes[j] - yaxes[j-6]) * 100 / yaxes[j-4])
        err_x.append(xaxes[j])
        err_c.append(colours[j])
        err_ls.append(linestyles[j])

errors = [err_x, err_y, err_c, err_ls]

err_x, err_y, err_c, err_ls = errors
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})

fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [3, 1]})
ax[0, 0].set_title(r'$k = k_{\mathrm{f}}$', fontsize=22)
ax[0, 1].set_title(r'$k = k_{{\mathrm{{f}}}},\, \Lambda = {}\;k_{{\mathrm{{f}}}}$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=22)



for i in range(2):
    ax[0, i].set_ylabel(ylabel, fontsize=22)
    ax[1, i].set_ylabel(r'$\% \, |\mathrm{err}|$', fontsize=22)
    # ax[1, i].set_ylim(-0.25, 6)
    ax[1, i].set_ylim(0.0035, 9)

    ax[1, i].axhline(0, c='cyan', lw=2.5)
    ax[1, i].set_xlabel(r'$a$', fontsize=22)

ax[0, 1].yaxis.set_label_position('right')
ax[1, 1].yaxis.set_label_position('right')

print(err_y[-1][-1])

for i in range(len(yaxes)):
    ax[0, 1].plot(xaxes[i], yaxes[i]*1e4, c=colours[i], ls=linestyles[i], lw=2.5, zorder=i)#, label=labels[i])
    if i < len(err_x):
        ax[1, 1].plot(err_x[i], err_y[i], c=err_c[i], ls=err_ls[i], lw=2.5)
    else:
        pass

ax[0, 1].legend(handles=handles2, fontsize=16, loc=3)


for i in range(2):
    ax[i,1].tick_params(labelleft=False, labelright=True)
    for j in range(2):
        ax[i, j].minorticks_on()
        ax[i, j].tick_params(axis='both', which='both', direction='in', labelsize=16)
        # ax[i].axvline(a_sc, c='g', lw=1, label=r'$a_{\mathrm{sc}}$')
        ax[i, j].yaxis.set_ticks_position('both')


fig.align_labels()
# ax[0, 0].set_ylim(3.46, 3.95)
# ax[0, 1].set_ylim(3.46, 3.95)

for j in range(2):
    ax[0, j].set_ylim(5.475, 6.275)


# ax[0, 0].set_ylim(0.0216, 0.0249)
# ax[0, 1].set_ylim(0.0216, 0.0249)



mode = 1
data = pickle.load(open("./data/unsm_spec_comp.p", "rb"))
xaxes = [data[j][0] for j in range(data.shape[1])]
yaxes = [data[j][1] for j in range(data.shape[1])]


# plots_folder = 'test/new_paper_plots/'
plots_folder = '/paper_plots_final/'

a_sc = 1

# title = r'$k = {}[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(mode)
title = r'$k = k_{\mathrm{f}}$'

colours = ['b', 'r', 'k', 'magenta', 'b', 'r', 'k', 'magenta']

labels = []
patch1 = mpatches.Patch(color='b', label=r'\texttt{sim\_1}')
patch2 = mpatches.Patch(color='r', label=r'\texttt{sim\_1\_7}')
patch3 = mpatches.Patch(color='k', label=r'\texttt{sim\_1\_11}')
patch4 = mpatches.Patch(color='magenta', label=r'\texttt{sim\_1\_15}')
line1 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='solid', label='$N$-body')
line2 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashed', label='SPT')

handles = [patch1, patch2, patch3, patch4]
handles2 = [line1, line2]
ax[0, 0].legend(handles=handles2, fontsize=16, loc=3)
linestyles = ['solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dashed']

savename = 'spec_comp_unsmoothed'
ylabel = r'$a^{-2}kP(k, a) \times 10^{3}$'
# ylabel = r'$a^{-2}k^{2}P(k, a)$'
# ylabel = r'$a^{-2}P(k, a)$'


#[P_nb_k1, P_nb_k7, P_nb_k11, P_nb_k15, P_1l_k1, P_1l_k7, P_1l_k11, P_1l_k15]
for j in range(len(yaxes)):
    if yaxes[j].size == 51:
        yaxes[j] = np.delete(yaxes[j], 23)
        xaxes[j] = np.delete(xaxes[j], 23)


err_x, err_y, err_c, err_ls = [], [], [], []
for j in range(len(yaxes)):
    yaxes[j] *= 1e4 / xaxes[j]**2
    # yaxes[j] *= (2*np.pi)**2 / xaxes[j]**2
    # yaxes[j] *= 1 / xaxes[j]**2

    if 3<j<8:
        err_y.append(np.abs(yaxes[j] - yaxes[j-4]) * 100 / yaxes[j-2])
        err_x.append(xaxes[j])
        err_c.append(colours[j])
        err_ls.append(linestyles[j])

errors = [err_x, err_y, err_c, err_ls]

for i in range(len(yaxes)):
    ax[0, 0].plot(xaxes[i], yaxes[i], c=colours[i], ls=linestyles[i], lw=2.5, zorder=i)#, label=labels[i])
    if i < len(err_x):
        ax[1, 0].plot(err_x[i], err_y[i], c=err_c[i], ls=err_ls[i], lw=2.5)
    else:
        pass

for j in range(2):
    ax[1, j].set_yscale('log')

# ax[0, 1].get_shared_y_axes().join(ax[0, 0], ax[0, 1])

# leg1 = plt.legend(handles=handles, fontsize=14, loc=3, bbox_to_anchor=(-0.7,4.1), ncol=2)
leg1 = plt.legend(handles=handles, fontsize=14, loc=3, bbox_to_anchor=(-0.635,4.25), ncol=4)
plt.gca().add_artist(leg1)

# handles = np.concatenate((handles[::2],handles[1::2]))
# labels = np.concatenate((labels[::2],labels[1::2]))
# handles, labels = plt.gca().get_legend_handles_labels()
# print(labels)
plt.subplots_adjust(hspace=0, wspace=0.065)

save = True
savename = 'ens_spec_combined'
if save == False:
    plt.show()
else:
    plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
    # plt.savefig('../plots/{}/{}.png'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
    plt.close()
