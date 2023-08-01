#!/usr/bin/env python3
"""A script for reading and plotting snapshots from cosmo_sim_1d"""

import os
import numpy as np
import matplotlib.pyplot as plt
from functions import smoothing, spectral_calc, SPT_real_tr, read_hier
from scipy.interpolate import interp1d
from zel import initial_density
from tqdm import tqdm
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_1_11_long/run1/'
Nfiles = 0
H0 = 100
Lambda = 3 * (2* np.pi)
kind = 'sharp'

q_list, a_list = [], []
j = 12

for j in tqdm(range(12)):
    nbody_filename = 'output_{0:04d}.txt'.format(j)
    nbody_file = np.genfromtxt(path + nbody_filename)
    x_nbody = nbody_file[:,-1]
    v_nbody = nbody_file[:,2]

    a, dx, M0, M1, M2, C0, C1, C2 = read_hier(path, j, folder_name='')

    # moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    # moments_file = np.genfromtxt(path + moments_filename)
    # a = moments_file[:,-1][0]
    # x = moments_file[:,0]
    # M0 = moments_file[:,2]
    # M1 = moments_file[:,4]
    # M2 = moments_file[:,6]
    # C1 = moments_file[:,8]
    # C2 = moments_file[:,7]

    L = 1.0
    x = np.arange(0, L, dx)
    k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))

    #solve Poisson to get the potential \phi
    rhs = (3 * H0**2 / (2 * a)) * (M0-1) #using the hierarchy δ here
    phi = spectral_calc(rhs, L, o=2, d=1)
    grad_phi = spectral_calc(phi, L, o=1, d=0) #this is the gradient of the unsmoothed potential ∇ϕ

    threshold = 1
    den = M0 #- 1
    halo_ind = np.where(den >= threshold)[0]
    halo_regions = np.split(halo_ind, np.where(np.diff(halo_ind) != 1)[0] + 1)

    # print(len(halo_regions))

    # plt.plot(den, c='b')
    # for region in halo_regions:
    #     plt.plot(region, den[region], 'ro', label='Halo', markersize=1)

    # plt.axhline(y=threshold, c='r', ls='dashed')
    # plt.show()
    # # plt.savefig('../plots/test/new_paper_plots/halos.png', dpi=150)
    # # plt.close()


    # m = x.size // 2
    # g = 200 #set to 6000 for the innermost halo


    # #define the kinetic energy 'scalar' inside the defined region
    def virial(halo_inds, C2, phi):
        T = sum((C2)[halo_inds]) / 2
        U = (sum(phi[halo_inds]))
        q = 2*T / np.abs(U)
        return T, U, q

    # # plt.scatter(k, np.fft.fft(M0)/M0.size, s=10, c='b')
    # # plt.show()

    h = halo_regions[5]
    T, U, q = virial(h, C2, phi)
    a_list.append(a)
    q_list.append(q)

    # print(T, U, q)

    # plt.rcParams.update({"text.usetex": True})
    # fig, ax = plt.subplots()
    # ax.plot(a_list, q_list, lw=1.5, c='b', label=r'$q$')
    # ax.set_xlabel(r'$x/L$', fontsize=16)
    # ax.set_ylabel(r'$1 + \delta$', fontsize=16)
    # ax.plot(x[h], M0[h], c='b')
    # ax.text(0.8, 0.9, rf'$q = {np.round(q, 3)}$', fontsize=14, transform=ax.transAxes)
    # ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
    # ax.grid(lw=0.2, ls='dashed', color='grey')
    # ax.yaxis.set_ticks_position('both')
    # ax.minorticks_on()

    # # plt.legend()
    # # plt.savefig('../plots/test/new_paper_plots/vir_ratio.png', dpi=300)
    # plt.show()


plt.rcParams.update({"text.usetex": True})
fig, ax = plt.subplots()
ax.plot(a_list, q_list, lw=1.5, c='b', label=r'$q$')
ax.set_xlabel(r'$a$', fontsize=16)
ax.set_ylabel(r'$q$', fontsize=16)
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('both')
ax.minorticks_on()

# plt.legend()
# plt.savefig('../plots/test/new_paper_plots/vir_ratio.png', dpi=300)
plt.show()