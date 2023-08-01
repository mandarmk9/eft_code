#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from functions import read_hier
from tqdm import tqdm
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


path = 'cosmo_sim_1d/sim_3_15/run1/'
folder_name = '/hierarchy/'


file_num = 0
for file_num in tqdm(range(51)):
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(file_num)
    moments_file = np.genfromtxt(path + moments_filename)
    x = moments_file[:,0]
    a = moments_file[:,-1][0]
    # den = moments_file[:,2]
    v = moments_file[:,5]



    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots()

    ax.set_title('a = {}'.format(np.round(a, 3)), fontsize=18)

    # ax.plot(x, np.log10(den), c='b', lw=1.5)
    # ax.set_ylabel(r'$\mathrm{log}_{10}\;(1+\delta)$', fontsize=16)

    ax.plot(x, v, c='b', lw=1.5)
    ax.set_ylabel(r'$\bar{v}$', fontsize=16)


    ax.set_xlabel(r'$x/L$', fontsize=16)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    plt.savefig('../plots/test/velocities/vel_{0:03d}.png'.format(file_num), bbox_inches='tight', dpi=300, pad_inches=0.3)
    plt.close()
