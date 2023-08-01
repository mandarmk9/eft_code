#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import pandas
import pickle
# from EFT_nbody_solver import *
from tqdm import tqdm
from zel import *
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
plots_folder = 'paper_plots_final/'
savename = 'spectra_all_sharp'

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [3, 1]})

# left panel prep
linestyles = ['solid', 'dotted', 'dashdot']
labels = [r'$N$-body', 'SPT', 'tSPT']
colours = ['b', 'hotpink', 'brown']
xlabel = r'$a$'
ylabel = r'$a^{-2}L^{-1}P(k, a) \times 10^{4}$'

file = open(rf"./{path}/spt_spec_plot_sharp.p", "rb")
xaxis, yaxes = np.array(pickle.load(file))
file.close()

errors = [(yaxis - yaxes[0]) * 100 / yaxes[0] for yaxis in yaxes[1:]]
title = r'$k = k_{{\mathrm{{f}}}},\; \Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(3, 'sharp cutoff')
handles = []
fig.suptitle(title, fontsize=24)#, y=0.95)
for j in range(3):
    line, = ax[0, 0].plot(xaxis, yaxes[j] * 10 / (2*np.pi), ls=linestyles[j], c=colours[j], lw=2.5)
    if j < 2:
        ax[1, 0].plot(xaxis, errors[j], ls=linestyles[j+1], c=colours[j+1], lw=2.5)
    handles.append(line)
for j in range(2):
    ax[1, j].set_xlabel(r'$a$', fontsize=20)
    ax[1, j].axhline(0, c=colours[0])
    for l in range(2):
        ax[j, l].minorticks_on()
        ax[j, l].tick_params(axis='both', which='both', direction='in', labelsize=18)
        ax[j, l].yaxis.set_ticks_position('both')

sc_line = ax[0, 0].axvline(1.81818, c='teal', ls='dashed', lw=0.75, zorder=1)#, label=r'$a_{\mathrm{shell}}$')
ax[1, 0].axvline(1.8181, c='teal', ls='dashed', lw=0.75, zorder=1)
handles.append(sc_line)
labels.append(r'$a_{\mathrm{shell}}$')

knl_line = ax[0, 0].axvline(2.26, c='magenta', ls='dotted', lw=1, zorder=1)#, label=r'$k_{\mathrm{NL}} / k_{\mathrm{f}} = 11$')
ax[1, 0].axvline(2.26, c='magenta', ls='dotted', lw=1, zorder=1)

handles.append(knl_line)
labels.append(r'$k_{\mathrm{NL}} / k_{\mathrm{f}} = 11$')

ax[0, 0].set_ylabel(ylabel, fontsize=22)
ax[0, 0].legend(handles, labels, fontsize=14, framealpha=1, loc=3)
ax[0, 1].set_ylabel(ylabel, fontsize=22)

# right panel
file = open(rf"./{path}/spec_plot_sharp.p", 'rb')
xaxis, yaxes = np.array(pickle.load(file))
file.close()

handles=[]
colours = ['b', 'brown', 'k', 'seagreen', 'midnightblue', 'magenta', 'orange', 'r']
labels = [r'$N$-body', 'tSPT', r'EFT: F3P',  r'EFT: F6P', r'EFT: M\&W', 'EFT: SC', r'EFT: SC$\delta$', r'EFT: matching $P_{N\mathrm{-body}}$']#, 'Zel']
linestyles = ['solid', 'dashdot', 'dashed', 'dashed', 'dashed', 'dashed', 'dashed', 'dotted']
dashes = [None, None, None, [1, 2, 1], [2, 1, 2], [2, 2, 1], [1, 1, 2], None]
errors = [(yaxis - yaxes[0]) * 100 / yaxes[0] for yaxis in yaxes[1:]]

# print(yaxes[0][23], yaxes[0][-1])

colours.pop(7)
labels.pop(7)
linestyles.pop(7)

for j in range(len(colours)):
    if dashes[j] == None:
        line, = ax[0, 1].plot(xaxis, yaxes[j]*1e4, c=colours[j], ls=linestyles[j], lw=2.5)
        handles.append(line)
    else:
        line, = ax[0, 1].plot(xaxis, yaxes[j]*1e4, c=colours[j], ls=linestyles[j], lw=2.5, dashes=dashes[j])
        handles.append(line)
    if 0 < j < len(colours)-1:
        if dashes[j+1] == None:
            ax[1, 1].plot(xaxis, errors[j], ls=linestyles[j+1], c=colours[j+1], lw=2.5)
        else:
            ax[1, 1].plot(xaxis, errors[j], ls=linestyles[j+1], c=colours[j+1], lw=2.5, dashes=dashes[j+1])


flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')
Nfiles = 51
labels.append(r'$a_{\mathrm{shell}}$')
for j in range(Nfiles):
    if flags[j] == 1:
        sc_line = ax[0, 1].axvline(xaxis[j], c='teal', ls='dashed', lw=0.75, zorder=1)
        ax[1, 1].axvline(xaxis[j], c='teal', ls='dashed', lw=0.75, zorder=1)
else:
    pass
handles.append(sc_line)
# handles.pop(7)

# ax[1, 0].set_ylabel(r'$\%\,|\mathrm{err}|$', fontsize=22) #16 for pdf
ax[1, 0].set_ylabel(r'$\%\,\mathrm{err}$', fontsize=22) #16 for pdf
ax[1, 1].set_ylabel(r'$\%\,\mathrm{err}$', fontsize=22) #16 for pdf

for j in range(2):
    # ax[0, j].set_ylim(3.69, 3.94)
    ax[0, j].set_ylim(5.855, 6.275)


# handles = handles[2:]
# labels = labels[2:]

ax[0, 1].legend(handles, labels, fontsize=14, framealpha=1, loc=3)
# ax[1, 1].yaxis.set_label_position('right')
ax[0, 1].tick_params(axis='y', which='both', labelleft=False, labelright=True)
ax[1, 1].tick_params(axis='y', labelleft=False, labelright=True)
# ax[1, 1].set_yscale('log')
# ax[1, 0].set_yscale('log')
for j in range(2):
    ax[j, 1].minorticks_on()

ax[0, 1].yaxis.set_label_position('right')
ax[1, 1].yaxis.set_label_position('right')
ax[1, 0].minorticks_on()
ax[1, 1].set_ylim(-0.12, 0.12)
ax[1, 0].set_ylim(-2.5, 2.3)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
fig.align_labels()
plt.savefig(rf'../plots/{plots_folder}/{savename}.pdf', bbox_inches='tight', dpi=300)
plt.close()