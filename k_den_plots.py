#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import pandas
import pickle
import numpy as np
from functions import plotter, initial_density, SPT_real_tr, smoothing, alpha_to_corr, alpha_c_finder, EFT_sm_kern, dc_in_finder, dn, read_density
from scipy.interpolate import interp1d
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
path = 'cosmo_sim_1d/sim_k_1_11/run1/'
plots_folder = 'test/new_paper_plots/k_space_den/'
Nfiles = 50
mode = 1
Lambda = 3 * (2 * np.pi)
H0 = 100
A = [-0.05, 1, -0.5, 11, 0]

# a_list, x, alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, err_Int = alpha_c_finder(Nfiles, Lambda, path, A, mode, kind, n_runs=8, n_use=6, H0=100, fde_method='percentile')
# df = pandas.DataFrame(data=[a_list, x, alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, err_Int])
# file = open("./data/alpha_c_{}.p".format(kind), "wb")
# pickle.dump(df, file)
# file.close()

file = open("./data/alpha_c_{}.p".format(kind), "rb")
read_file = pickle.load(file)
a_list, x, alpha_c_true_list, alpha_c_list, alpha_c_list2, alpha_c_list3, err_Int = np.array(read_file)
file.close()

flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

Nx = x.size
L = 1.0
k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
dc_in, k_in = dc_in_finder(path, x, interp=True)
x = np.arange(0, 1, 1/k.size)
a_list = a_list[:Nfiles]
alpha_c_true_list = alpha_c_true_list[:Nfiles]
alpha_c_list = alpha_c_list[:Nfiles]
alpha_c_list2 = alpha_c_list2[:Nfiles] #M&W
alpha_c_list3 = alpha_c_list3[:Nfiles] #B12

k /= (2*np.pi)
for j in range(1):
    # j = 11
    a = a_list[j]
    alpha_c = alpha_c_list2[j]


    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x_cell = moments_file[:,0]
    M0_nbody = moments_file[:,2]-1
    M0_nbody -= np.mean(M0_nbody)
    den_eft = alpha_to_corr(alpha_c, a, x, k, L, dc_in, Lambda, kind)
    den_spt_tr = SPT_real_tr(dc_in, k, L, Lambda, a, kind)
    den_nbody = M0_nbody#@smoothing(M0_nbody, k, Lambda, kind)

    dk_par, a, dx = read_density(path, j)
    L = 1.0
    x_grid = np.arange(0, L, dx)
    k_par = np.fft.ifftshift(2.0 * np.pi * np.arange(-dk_par.size/2, dk_par.size/2)) / (2*np.pi)

    dk_par /= 125
    # dk_par = (dk_par - 1000) / 1000 #/ 125000
    dx_par = (np.real(np.fft.ifft(dk_par)))-1

    # dk_par /= dk_par.size
    # print(dk_par[0])
    x_par = np.arange(0, 1, 1/dk_par.size)
    plt.plot(x_par, dx_par)
    plt.plot(x, M0_nbody, ls='dashed')
    plt.show()


    # M0_par = np.real(np.fft.ifft(dk_par))
    # M0_par /= np.mean(M0_par)
    # f_M0 = interp1d(x_grid, M0_par, fill_value='extrapolate')
    # M0_par = f_M0(x)
    #
    # M0_k = np.real(np.fft.fft(M0_par - 1) / M0_par.size)

    #
    # # real_dk_nbody = np.real(np.fft.fft(den_nbody) / den_nbody.size)
    # # im_dk_nbody = np.imag(np.fft.fft(den_nbody) / den_nbody.size)
    # # amp = (real_dk_nbody + im_dk_nbody)
    #
    # dk_spt = np.real(np.fft.fft(den_spt_tr) / den_spt_tr.size)
    #
    # plt.rcParams.update({"text.usetex": True})
    # plt.rcParams.update({"font.family": "serif"})
    # fig, ax = plt.subplots()
    # ax.set_title(r'a = {}, $\Lambda = {}\;k_{{\mathrm{{f}}}}$ ({})'.format(a, int(Lambda / (2 * np.pi)), kind_txt), fontsize=18, y=1.01)
    # ax.set_xlabel(r'$k\;[k_{\mathrm{f}}]$', fontsize=20)
    # ax.set_ylabel(r'Re($\widetilde{\delta}_{l}(k)$)', fontsize=20)
    # # ax.set_ylabel(r'$|\widetilde{\delta}_{l}(k)|^{2}$', fontsize=20)
    #
    # # ax.scatter(k, M0_k, s=50, c='b', label=r'$N-$body')
    # ax.scatter(k_par, np.real(dk_par), s=50, c='b', label=r'$N-$body')
    # # ax.scatter(k, dk_spt, s=30, c='brown', label=r'SPT-4')
    #
    # ax.set_xlim(-0.5, 13.5)
    # # ax.set_ylim(-1e-6, 2e-5)
    # ax.set_ylim(-1.005, -0.9925)
    #
    # ax.minorticks_on()
    # ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
    # ax.legend(fontsize=12)
    # ax.yaxis.set_ticks_position('both')
    # # ax.set_yscale('lo g')
    # # plt.show()
    # plt.savefig('../plots/{}/dk_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=150)
    # plt.close()
