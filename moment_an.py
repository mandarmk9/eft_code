#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from functions import read_hier

path = 'cosmo_sim_1d/gaussian_ICs/'
folder_name = '/hierarchy/'
plots_folder = 'spectra'

j = 0

for j in range(11):
    # j = 0
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x = moments_file[:,0]
    M0_nbody = moments_file[:,2]-1
    M1_nbody = moments_file[:,4]
    M2_nbody = moments_file[:,6]
    C1_nbody = moments_file[:,5]
    C2_nbody = moments_file[:,7]

    M0_nbody -= np.mean(M0_nbody)
    M0_k = np.fft.fft(M0_nbody) / M0_nbody.size
    P_k = np.abs(M0_k * np.conj(M0_k))
    k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))


    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots()
    ax.set_title(rf'a = {np.round(a, 3)}', fontsize=14)
    # ax.set_xlabel()
    # ax.set_ylabel()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.scatter(k, P_k, c='b', s=15)
    # ax.set_xlim(-0.5, 1000.5)
    # ax.plot(x, C2_nbody)
    plt.savefig(f'../plots/gauss/{plots_folder}/Pk_{j}.png', bbox_inches='tight', dpi=150)
    plt.close()
    # plt.show()
