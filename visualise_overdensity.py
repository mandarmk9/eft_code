#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

L = 1.0
x = np.arange(0, L, 0.001)

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
plots_folder = '/paper_plots_final/vis/'


def extract_fields(path, file_num):
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(file_num)
    moments_file = np.genfromtxt(path + moments_filename)
    x = moments_file[:,0]
    a = moments_file[:,-1][0]
    dc = moments_file[:,2]
    return a, x, dc

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
j = 0
for j in range(51):
    a, x, dc = extract_fields(path, j)
    fig, ax = plt.subplots()
    ax.set_title(rf'a = {a}', fontsize=20, color='w')
    plt.axis('off')
    fig.set_facecolor('#1E1E1E')
    ax.plot(x, np.log10(1+dc), c='w', lw=1.5)
    plt.savefig(rf'../plots/{plots_folder}/dc_{j:03d}.png', dpi=300)
    plt.close()