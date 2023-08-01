#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker
from functions import read_sim_data
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
folder_name = '/new_hier/data_{}/L{}'.format(kind, int(Lambda/(2*np.pi)))

##plotting
for file_num in range(1):
    file_num = 24

    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\delta_{l}$', fontsize=18)
    ax.set_ylabel(r'$\theta_{l}$', fontsize=18)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=13.5)
    cm = plt.cm.get_cmap('RdYlBu')
    colors = iter(['brown', 'darkcyan', 'dimgray', 'violet', 'orange', 'cyan', 'b', 'r', 'k'])

    for run in range(1,2):
        path = 'cosmo_sim_1d/sim_k_1_11/run{}/'.format(run)
        a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, file_num, folder_name)
        # print(a)
        dv_l *= np.sqrt(a) / 100
        tau_l -= np.mean(tau_l)
        obj = ax.scatter(dc_l, dv_l, c=tau_l, s=5, cmap='rainbow', rasterized=True)#, norm=colors.Normalize(vmin=del_tau.min(), vmax=del_tau.max()))
        # obj = ax.scatter(dc_l, dv_l, color=next(colors), s=5, rasterized=True)#, norm=colors.Normalize(vmin=del_tau.min(), vmax=del_tau.max()))
        # inds = np.argsort(dv_l)
        # obj = ax.scatter(dc_l[inds], dv_l[inds], color=next(colors), s=1, rasterized=True)#, norm=colors.Normalize(vmin=del_tau.min(), vmax=del_tau.max()))
        # ax.plot(x, tau_l, lw=2, c='b')

    # ax.plot(x, dc_l[inds])
    # ax.plot(x, dv_l[inds], c='r')

    cbar = fig.colorbar(obj, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=10)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.set_ylabel(r'$[\tau]_{\Lambda}\; [\mathrm{M}_{\mathrm{p}}H_{0}^{2}L^{-1}]$', fontsize=18)
    cbar.ax.tick_params(labelsize=12.5)
    ax.set_title(r'$a = {}, \Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(np.round(a, 3), int(Lambda/(2*np.pi)), kind_txt), fontsize=16)
    # plt.savefig('../plots/test/new_paper_plots/dc_dv_nsim/tau_{}.png'.format(file_num), bbox_inches='tight', dpi=150)
    # plt.savefig('../plots/test/new_paper_plots/dc_dv_run2/tau_{}.png'.format(file_num), bbox_inches='tight', dpi=150)

    # plt.close()
    plt.show()
