#!/usr/bin/env python3

#import libraries
# import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle
from functions import read_density, SPT, dc_in_finder

# path = 'cosmo_sim_1d/multi_sim_3_33/run1/'
path = 'cosmo_sim_1d/sim_k_1_11/run1/'
path_k1 = 'cosmo_sim_1d/sim_k_1/run1/'
file_num = 50

def P_nb_calc(path, j):
   dk_par, a, dx = read_density(path, j)
   x = np.arange(0, 1.0, dx)
   M0_par = np.real(np.fft.ifft(dk_par))
   M0_par = (M0_par / np.mean(M0_par)) - 1
   M0_k = np.fft.fft(M0_par) / M0_par.size
   P_nb_a = np.real(M0_k * np.conj(M0_k))
   k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2)) / (2*np.pi)
   return a, P_nb_a, k, np.real(M0_k[1])


for file_num in range(1):
    file_num = 50
    a, P_nb, k, dc_kf = P_nb_calc(path, file_num)
    x = np.arange(0, 1, 1/k.size)
    dc_in, k_1l = dc_in_finder(path, x, interp=True)
    d1_k, _, P_1l, P_2l = SPT(dc_in, 1.0, a)
    a, P_nb_k1, k, dc_kf_k1 = P_nb_calc(path_k1, file_num)
    dc_in, k_1l = dc_in_finder(path_k1, x, interp=True)
    d1_k1, _, P_1l_k1, P_2l_k1 = SPT(dc_in, 1.0, a)

    # print(np.abs(dc_kf - dc_kf_k1) / dc_kf)
    k_1l /= (2*np.pi)
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots()
    ax.set_title(r'$a = {}$'.format(a))
    ax.set_ylabel(r'$\log_{10}P(k)$', fontsize=14)
    # ax.set_xlabel(r'$k\,[k_{\mathrm{f}}]$', fontsize=14)
    ax.set_xlabel(r'$k/k_{\mathrm{f}}$', fontsize=14)

    diff_nb =(P_nb_k1 - P_nb) / P_nb
    print(diff_nb[1]*100)
    P = np.log10(np.abs(P_nb))
    P_k1 = np.log10(np.abs(P_nb_k1))
    P_lin_k1 = np.abs(d1_k1**2) * a**2
    print((np.abs(P_lin_k1 - P_nb_k1) * 100 / P_nb_k1)[1])

    # P = (P_nb)
    # P_k1 = (P_nb_k1)

    diff_spt = np.abs(np.abs(P_1l[2:10]) - P_nb[2:10]) / P_nb[2:10]
    k_diff_spt = k[2:10]
    diff_spt_k1 = np.abs(np.abs(P_1l_k1[:7]) - P_nb[:7]) / P_nb[:7]

    # print((P_1l_k1[:14]))

    # ax.scatter(k, P, label=r'\texttt{sim\_3\_15}; $N$-body', s=30, c='b', marker=MarkerStyle('o', fillstyle='full'))
    ax.scatter(k, P, label=r'\texttt{sim\_1\_11}; $N$-body', s=30, c='b', marker=MarkerStyle('o', fillstyle='full'))
    ax.scatter(k, P_k1, label=r'\texttt{sim\_1}; $N$-body', s=25, c='k', marker=MarkerStyle('o', fillstyle='none'))
    # ax.scatter(k_1l[:40], np.log10(np.abs(P_1l[:40])), label=r'\texttt{sim\_3\_33}; SPT-4', s=25, c='seagreen', marker=MarkerStyle('d', fillstyle='full'))
    ax.scatter(k_1l[:16], np.log10(np.abs(P_1l_k1[:16])), label=r'\texttt{sim\_1}; SPT-4', s=25, c='cyan', marker=MarkerStyle('d', fillstyle='none'))

    # ax.scatter(k_1l, lP_1l, label=r'\texttt{sim\_1\_11}; SPT-4', s=25, c='seagreen', marker='v')
    # ax.scatter(k_1l, P_1l_k1, label=r'\texttt{sim\_1}; SPT-4', s=25, c='cyan', marker='+')

    ax.tick_params(axis='both', which='both', direction='in')

    # ax.ticklabel_format(scilimits=(-2, 3))
    # ax.grid(lw=0.2, ls='dashed', color='grey')
    ax.yaxis.set_ticks_position('both')
    ax.set_ylim(-15, 1)
    ax.legend(fontsize=11, loc='lower left')#, loc=2, bbox_to_anchor=(1,1))
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.set_xlim(0.5, 16.5)
    # plt.show()

    # text_str = r'$\Delta P(k) = \frac{\left|P_{\texttt{\footnotesize sim\_1}} - P_{\texttt{\footnotesize sim\_1\_11}}\right|}{P_{\texttt{\footnotesize sim\_1\_11}}}$'

    # # ax.text(1.01, 0.75, text_str, transform=ax.transAxes, fontsize=10)
    # diff[0] = 0


    left, bottom, width, height = [0.65, 0.25, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.set_xlim(0.5, 5.5)
    ax2.yaxis.set_ticks_position('both')
    ax2.tick_params(axis='both', which='both', direction='in')
    ax2.tick_params(axis='y', which='minor', bottom=False)
    ax2.set_ylim(0, 0.025)
    ax2.set_xlabel(r'$k/k_{\mathrm{f}}$', fontsize=12)
    ax2.set_ylabel(r'$\frac{\Delta P}{P}$', fontsize=14)
    ax2.minorticks_on()
    ax2.tick_params(axis='x', which='minor', bottom=False, top=False)

    # ax2.scatter(k, diff_nb, c='k', marker=MarkerStyle('o', fillstyle='none'), s=25)
    # ax2.scatter(k_diff_spt, diff_spt, c='seagreen', marker=MarkerStyle('d', fillstyle='full'), s=25)
    ax2.scatter(k[:7], diff_spt_k1, c='cyan', marker=MarkerStyle('d', fillstyle='none'), s=25)

    # ax2.scatter(k, diff_nb, c='k', marker=MarkerStyle('o', fillstyle='none'), s=25)
    # ax2.scatter(k[:7], diff_spt, c='seagreen', marker=MarkerStyle('d', fillstyle='full'), s=25)
    # ax2.scatter(k[:7], diff_spt_k1, c='cyan', marker=MarkerStyle('d', fillstyle='none'), s=25)

    # plt.savefig('../plots/test/multi_sim_3_33/PS/PS_{0:03d}.png'.format(file_num), bbox_inches='tight', dpi=300)
    # plt.close()
# plt.show()

# print('c = ', c)
# print('n = ', n)
