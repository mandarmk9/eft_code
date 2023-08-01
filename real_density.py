#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import pandas
import numpy as np
from functions import read_hier, smoothing, SPT_real_tr, dc_in_finder
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
Nfiles = 50
mode = 1
Lambda = 3 * (2 * np.pi)
H0 = 100
def corr(x, field):
    N = x.size
    corr_0 = np.zeros(field.size)
    sep = []
    for j in range(N):
        if j <= N//2:
            y = np.roll(x, -j)
            sep.append(x[0]+y[0])
            corr_0 += field * np.roll(field, -j)
        elif j > N//2:
            y = np.roll(x, j)
            sep.append(x[0]-y[0])
            corr_0 += field * np.roll(field, j)

    sep = np.roll(np.array(sep), N//2-1)
    corr_0 /= field.size

    field_k = np.fft.fft(field)
    corr_1 = np.real(np.fft.ifft((field*np.conj(field_k))))
    return sep, corr_0, corr_1


for j in range(1):
    j = 15
    path = 'cosmo_sim_1d/sim_k_1_11/run1/'
    # path = 'cosmo_sim_1d/multi_k_sim/run1/'

    folder_name = '/hierarchy/'
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x_cell = moments_file[:,0]
    M0_nbody = moments_file[:,2]
    k_nb = np.fft.ifftshift(2.0 * np.pi * np.arange(-x_cell.size/2, x_cell.size/2))
    M0_nbody = smoothing(M0_nbody, k_nb, Lambda, kind)-1

    a, dx, M0, M1, M2, C0, C1, C2 = read_hier(path, j, folder_name)
    x = np.arange(0, 1, dx)
    dc_in, k_in = dc_in_finder(path, x, interp=True)
    M0_spt = SPT_real_tr(dc_in, k_in, 1.0, Lambda, a, kind)+1
    k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))
    M0_spt = smoothing(M0_spt-1, k, Lambda, kind)
    M0 = smoothing(M0-1, k, Lambda, kind)




    print('a = {}'.format(np.round(a, 3)))

    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots()
    ax.set_title(r'a = {}, $\Lambda = {}\;k_{{\mathrm{{f}}}}$ ({})'.format(a, int(Lambda / (2 * np.pi)), kind_txt), fontsize=18, y=1.01)

    # ax.plot(x_cell, M0_nbody, label='Old', lw=2, ls='solid', c='k')
    # ax.plot(x, M0, label='Old', lw=2, ls='dashdot', c='r')
    # ax.plot(x, M0_spt, label='New', lw=2, ls='dashed', c='b')

    err = (M0_spt - M0) * 100 / (1+M0)
    ax.plot(x, err, c='cyan', lw=2, ls='dotted')
    ax.set_ylabel(r'$1 + \delta$', fontsize=18)
    ax.set_xlabel(r'$x/L$', fontsize=18)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=14)
    ax.legend(fontsize=12, bbox_to_anchor=(1,1))
    ax.yaxis.set_ticks_position('both')
    plt.show()
    plt.tight_layout()
    # plt.savefig('../plots/test/new_paper_plots/sanity/spectra_{}.png'.format(j), bbox_inches='tight', dpi=150)
    # plt.close()
