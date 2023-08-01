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
    j = 23
    path = 'cosmo_sim_1d/sim_k_1_11/run1/'
    # path = 'cosmo_sim_1d/multi_k_sim/run1/'

    folder_name = '/hierarchy/'
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x_cell = moments_file[:,0]
    M0_nbody = moments_file[:,2]


    a, dx, M0, M1, M2, C0, C1, C2 = read_hier(path, j, folder_name)
    x = np.arange(0, 1, dx)
    dc_in, k_in = dc_in_finder(path, x, interp=True)
    M0_spt = SPT_real_tr(dc_in, k_in, 1.0, Lambda, a, kind)+1

    # path = 'cosmo_sim_1d/sim_k_1/run1/'
    # folder_name = '/hierarchy/'
    #
    # a, dx, M0_k1, M1_k1, M2_k1, C0, C1, C2 = read_hier(path, j, folder_name)
    # x_k1 = np.arange(0, 1, dx)
    #
    # moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    # moments_file = np.genfromtxt(path + moments_filename)
    # a = moments_file[:,-1][0]
    # x_cell_k1 = moments_file[:,0]
    # M0_nbody_k1 = moments_file[:,2]


    print('a = {}'.format(np.round(a, 3)))

    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots()
    ax.set_title(r'a = {}, $\Lambda = {}\;k_{{\mathrm{{f}}}}$ ({})'.format(a, int(Lambda / (2 * np.pi)), kind_txt), fontsize=18, y=1.01)

    # fields = [M0_nbody, M0_nbody_k1, M0, M0_k1, M0_spt]
    # labels = ['Old', r'Old, $k_{1}$', 'New', r'New, $k_{1}$', r'SPT-4']
    # pos = [x_cell, x_cell_k1, x, x_k1, x]

    # fields = [M0_nbody, M0, M0_spt]
    # labels = ['Old', 'New', r'SPT-4']
    # pos = [x_cell, x, x]

    fields = [M0, M0_spt]
    labels = [r'$N-$body', r'SPT-4']
    pos = [x, x]


    colours = ['b', 'k', 'r', 'seagreen', 'orange', 'cyan', 'magenta', 'brown']
    ls = ['solid', 'dashdot', 'dashed', 'dotted', 'solid', 'dashdot', 'dashed', 'dotted']
    s = [60, 50, 40, 30, 20, 10, 5]

    Lambda_int = int(Lambda/(2*np.pi))
    f = []
    for m in range(0, len(fields)):
        wavevector = np.fft.ifftshift(2.0 * np.pi * np.arange(-pos[m].size/2, pos[m].size/2))
        field_k = np.fft.fft(fields[m]-1)
        power = np.abs(field_k * np.conj(field_k) / (field_k.size**2) * (field_k**2))
        ax.scatter((wavevector/(2*np.pi)), power/a**2, s=s[m]+5, c=colours[m+1], label=labels[m])
        fields[m] = (smoothing(fields[m]-1, wavevector, Lambda, kind))
        field_k = np.fft.fft(fields[m])
        ntrunc = int(field_k.size-Lambda_int)
        field_k[Lambda_int+1:ntrunc] = 0
        # field_k[1] = 0
        # field_k[-1] = 0
        field_k[3:-2] = 0
        # fields[m] = np.real(np.fft.ifft(field_k))

        print(power[1]/a**2)
        # power = np.abs(field_k * np.conj(field_k)) / (field_k.size**2)
        # ax.scatter((wavevector/(2*np.pi)), power, s=s[m]-5, c=colours[m+3], label=labels[m])
        # # fields[m] = fields[m]-1
        # # field_k = np.abs(np.fft.fft(fields[m]) / fields[m].size)**2
        # sep, corr_, corr__ = corr(x, fields[m])
        # # ax.plot(sep, corr_, lw=2, c=colours[m], label='real', ls='solid')
        # # ax.plot(x, corr__, lw=2, c=colours[m], label='Fourier', ls='dashed')
        # f.append(corr__)
    # # err = (f[1] - f[0]) * 100 / (1+f[0])
    # # print(sum(err))
    # # # ax.plot(x, f[0], c='b', lw=2)
    # # # ax.plot(x, f[1], c='r', lw=2, ls='dashed')
    # err = (fields[1]-fields[0])*100 / (1+fields[0])
    # for m in range(2):
    #     ax.plot(pos[m], fields[m], lw=2, c=colours[m], label=labels[m], ls=ls[m])
    # # ax.plot(x, err, c='r', lw=2)
    # # print(sum(err))


        # ax.scatter(wavevector/(2*np.pi), field_k, s=s[m], c=colours[m], label=labels[m])

    # resid = sum((fields[1]-fields[0])**2)
    # print(resid)
    #
    # resid_k = sum(((np.fft.fft(fields[1])) - (np.fft.fft(fields[0])))**2)
    # print(np.abs(resid_k))

    ax.set_xlim(-0.5, 5.5)
    # ax.set_ylabel(r'$|\widetilde{\delta}_{l}(k)|^{2}$', fontsize=18)
    # ax.set_xlabel(r'$k\;[k_{\mathrm{f}}]$', fontsize=18)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1e3)
    ylabel = r'$a^{-2}k^{2}P(k, a) \; [10^{-4}L^{2}]$'

    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(r'$x/L$', fontsize=18)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=14)
    ax.legend(fontsize=12, bbox_to_anchor=(1,1))
    ax.yaxis.set_ticks_position('both')
    plt.show()
    # plt.savefig('../plots/test/new_paper_plots/sanity/spectra_{}.png'.format(j), bbox_inches='tight', dpi=150)
    # plt.close()
