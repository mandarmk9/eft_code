#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt

from functions import *
from zel import eulerian_sampling

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
path_k1 = 'cosmo_sim_1d/sim_k_1/run1/'
A = [-0.05, 1, -0.0, 11]
Lambda_int = 3
Lambda = Lambda_int * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

# a0 = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(0))
# q = np.genfromtxt(path + 'output_{0:04d}.txt'.format(0))[:,0]
# q_zel = q[::50]
# P_zel, a_zel = [], []
# k_zel = np.fft.ifftshift(2.0 * np.pi * np.arange(-q_zel.size/2, q_zel.size/2))
# dc_zel = eulerian_sampling(q_zel, a0, A, 1.0)[1]+1 #smoothing(eulerian_sampling(q_zel, a, A, 1.0)[1], k_zel, Lambda, kind)
# dc_zel = smoothing(dc_zel, k_zel, 2*np.pi, kind)

# dk_zel = np.fft.fft(dc_zel) / dc_zel.size
# P_zel_a = np.real(dk_zel * np.conj(dk_zel))
# P_zel.append(P_zel_a[mode])
# a_zel.append(a)

# def extract_sm_fields(path, file_num, Lambda, kind, sm=True):
#     moments_filename = 'output_hierarchy_{0:04d}.txt'.format(file_num)
#     moments_file = np.genfromtxt(path + moments_filename)
#
#     x = moments_file[:,0]
#     a = moments_file[:,-1][0]
#     dc = moments_file[:,2]
#     v = moments_file[:,5]
#     # v = -spectral_calc(dc, 1, o=1, d=0) / (a*100)
#     k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))
#     if sm == True:
#         dc = smoothing(dc, k, Lambda, kind)
#         v = smoothing(v, k, Lambda, kind)
#     # dc_k = np.fft.fft(dc)
#     # dc_k[2:-1] = 0
#     # dc = np.real(np.fft.ifft(dc_k))
#     return a, x, dc, v

# folder_name = '/new_hier/data_{}/L{}'.format(kind, Lambda_int)
folder_name = '/hierarchy/'
def extract_sm_fields(path, file_num, Lambda, kind, sm=True):
    a, dx, M0_nbody, M1_nbody, M2_nbody, C0_nbody, C1_nbody, C2_nbody = read_hier(path, file_num, folder_name)
    x = np.arange(0, 1, dx)
    dc = M0_nbody #dc is 1+\delta
    v = C1_nbody
    k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))

    if sm == True:
        dc = smoothing(dc, k, Lambda, kind)
        v = smoothing(v, k, Lambda, kind)

    # dc_k = np.fft.fft(dc)
    # dc_k[2:-1] = 0
    # dc = np.real(np.fft.ifft(dc_k))
    return a, x, dc, v

# j = 10
a_0, x, dc_0, v_0 = extract_sm_fields(path, 0, Lambda, kind)
a_1, x, dc_1, v_1 = extract_sm_fields(path, 14, Lambda, kind)
a_2, x, dc_2, v_2 = extract_sm_fields(path, 23, Lambda, kind)
# a_3, x, dc_3, v_3 = extract_sm_fields(path, 35, Lambda, kind)

a_0, x_k1, dc_k1_0, v_k1_0 = extract_sm_fields(path_k1, 0, Lambda, kind)#, sm=False)
a_1, x_k1, dc_k1_1, v_k1_1 = extract_sm_fields(path_k1, 14, Lambda, kind)#, sm=False)
a_2, x_k1, dc_k1_2, v_k1_2 = extract_sm_fields(path_k1, 23, Lambda, kind)#, sm=False)
# a_3, x_k1, dc_k1_3, v_k1_3 = extract_sm_fields(path_k1, 35, Lambda, kind)#, sm=False)


# a_1 = 1.93
a_list = [a_0, a_1, a_2]# , a_3]
den_list = [dc_0, dc_1, dc_2]#, dc_3]
vel_list = [v_0, v_1, v_2]#, v_3]
den_k1_list = [dc_k1_0, dc_k1_1, dc_k1_2]#, dc_k1_3]
vel_k1_list = [v_k1_0, v_k1_1, v_k1_2]#, v_k1_3]

# print(a_list)

def P_nb_calc(path, j):
   dk_par, a, dx = read_density(path, j)
   x = np.arange(0, 1.0, dx)
   M0_par = np.real(np.fft.ifft(dk_par))
   M0_par = (M0_par / np.mean(M0_par)) - 1
   M0_k = np.fft.fft(M0_par) / M0_par.size
   P_nb_a = np.real(M0_k * np.conj(M0_k))
   k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2)) / (2*np.pi)
   M0_par = smoothing(M0_par, k, 1*2*np.pi, 'sharp')  
   return a, P_nb_a, k, x, M0_par

# def P_nb_calc(path, j):
#     a, dx, M0_nbody, M1_nbody, M2_nbody, C0_nbody, C1_nbody, C2_nbody = read_hier(path, j, '')
#     x = np.arange(0, 1.0, dx)
#     M0_par = (M0_nbody / np.mean(M0_nbody)) - 1
#     M0_k = np.fft.fft(M0_par) / M0_par.size
#     P_nb_a = np.real(M0_k * np.conj(M0_k))
#     k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2)) / (2*np.pi)
#     M0_par = smoothing(M0_par, k, 3*2*np.pi, 'sharp')
#     return a, P_nb_a, k, x, M0_par


a, P_nb, k, x, M0_par = P_nb_calc(path, 23)
a, P_nb_k1, k, x_k1, M0_par_k1 = P_nb_calc(path_k1, 23)
P = np.log10(P_nb)
P_k1 = np.log10(P_nb_k1)

# P_1_11 = np.log10(np.abs(np.fft.fft(1+dc_2))**2)
# P_1 = np.log10(np.abs(np.fft.fft(1+dc_k1_2))**2)
# k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2)) / (2*np.pi)
# k_k1 = np.fft.ifftshift(2.0 * np.pi * np.arange(-x_k1.size/2, x_k1.size/2)) / (2*np.pi)

# diff_dc = np.abs(M0_par - M0_par_k1) / (1+M0_par)
# P_ = np.abs(np.fft.fft(M0_par) / M0_par.size)**2
# P_k1_ = np.abs(np.fft.fft(M0_par_k1) / M0_par_k1.size)**2
# diff_dk2 = np.abs(P_ - P_k1_) / P_
# diff_dk = np.abs(np.fft.fft(M0_par)[1:4] - np.fft.fft(M0_par_k1)[1:4]) / np.fft.fft(M0_par)[1:4]
# sum_dk = np.abs(np.fft.fft(M0_par)[1:4] + np.fft.fft(M0_par_k1)[1:4]) / np.fft.fft(M0_par)[1:4]

# print(diff_dc.max(), diff_dk2[1], np.real(diff_dk[1]))

# M0_par = 1.1 * np.sin(2*np.pi*x)
# M0_par_k1 = np.sin(2*np.pi*x)

diff__ = np.abs(M0_par.max() - M0_par_k1.max()) / (M0_par.max())
print(diff__)

# print(M0_par.max() - M0_par_k1.max())

#density plot
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(1,2, figsize =(12,6))
fig.suptitle('a = {}'.format(a))
ax[0].plot(x, M0_par, c='b', lw=1.5)
ax[0].plot(x, M0_par_k1, c='k', lw=1.5, ls='dashed')
ax[0].axhline(M0_par.max(), c='b', ls='dotted', lw=1)

strA = str(np.round(float(M0_par.max()), 4))
strB = str(np.round(float(M0_par_k1.max()), 4))
ax[0].text(0, 1.025*(M0_par.max()), strA, color='b')
ax[0].text(0, 0.95*(M0_par_k1.max()), strB, color='k')
ax[0].axhline(M0_par_k1.max(), c='k', ls='dotted', lw=1)

diff_real = np.abs(M0_par_k1 - M0_par)/(1+M0_par)
# ax[0].plot(x, diff_real, c='r', lw=1.5, ls='dashdot')
ax[0].set_xlabel(r'$x$', fontsize=14)
ax[0].set_ylabel(r'$\delta_{l}$', fontsize=14)

P = (np.real(np.fft.fft(M0_par)*np.conj(np.fft.fft(M0_par)) / M0_par.size**2))
P_k1 = (np.real(np.fft.fft(M0_par_k1)*np.conj(np.fft.fft(M0_par_k1)) / M0_par_k1.size**2))
diff = np.abs((P_k1 - P) / P)
# fac = diff_real.max()/diff[1]
# print(1/fac, fac)
ax[1].scatter(k[1], diff[1], label=r'\texttt{sim\_1\_11}', s=25, c='r', marker='*')
print(diff[1])
ax[1].set_xlim(0.5, 1.5)
ax[1].yaxis.set_ticks_position('both')
ax[1].tick_params(axis='both', which='both', direction='in')
ax[1].tick_params(axis='y', which='minor', bottom=False)
# ax[1].set_ylim(-0.01, 0.2)
ax[1].set_xlabel(r'$k\,[k_{\mathrm{f}}]$', fontsize=14)
ax[1].set_ylabel(r'$\frac{\Delta P}{P}$', fontsize=14)
ax[1].minorticks_on()
ax[1].tick_params(axis='x', which='minor', bottom=False, top=False)
# # ax.scatter(k, np.log10(P), label=r'\texttt{sim\_1\_11}', s=30, c='b')
# # ax.scatter(k, np.log10(P_k1), label=r'\texttt{sim\_1}', s=25, c='k', marker='^')

# ax.tick_params(axis='both', which='both', direction='in')
# ax.yaxis.set_ticks_position('both')
# ax.legend(fontsize=11)
# ax.minorticks_on()
# ax.tick_params(axis='x', which='minor', bottom=False, top=False)
# ax.set_xlim(0.5, 16.5)
plt.savefig('../plots/test/new_paper_plots/test_spec.png', dpi=300)
plt.close()

# # fig, ax = plt.subplots(3, 1, figsize=(6, 9), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1]})

# # # fig.suptitle(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(Lambda_int, kind_txt), fontsize=18, y=0.91)
# # fig.suptitle(r'$\Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(Lambda_int, kind_txt), fontsize=18, y=0.91)

# # # fig.suptitle(r'$k = 1 \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$', fontsize=16, y=0.91)

# # for i in range(3):
# #     ax[i].set_title('a = {}'.format(np.round(a_list[i], 3)), x=0.88, y=0.85, fontsize=18)
# #     ax[i].plot(x, den_list[i], c='b', lw=1.5, label=r'\texttt{sim\_1\_11}')
# #     # if i == 0:
# #     #     ax[i].plot(q_zel, dc_zel, c='r', ls='dotted', lw=1.5, label='Zel')

# #     ax[i].plot(x_k1, den_k1_list[i], c='k', lw=1.5, ls='dashed', label=r'\texttt{sim\_1}')
# #     ax[i].set_ylabel(r'$1+\delta_{l}$', fontsize=20)
# #     ax[i].minorticks_on()
# #     ax[i].tick_params(axis='both', which='both', direction='in', labelsize=15)
# #     ax[i].yaxis.set_ticks_position('both')
# # fig.align_labels()
# # plt.rcParams.update({"text.usetex": True})
# # ax[0].legend(fontsize=13, loc=8)
# # plt.rcParams.update({"text.usetex": False})
# # # ax[2].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
# # ax[2].set_xlabel(r'$x/L$', fontsize=20)

# # plt.subplots_adjust(hspace=0)
# # # plt.show()
# # plt.savefig('../plots/test/new_paper_plots/den_ev_sm.pdf', bbox_inches='tight', dpi=300)#, pad_inches=0.3)
# # # # plt.savefig('../plots/test/den_ev.png', bbox_inches='tight', dpi=300)
# # plt.close()


# # #velocity plot
# # plt.rcParams.update({"text.usetex": True})
# # plt.rcParams.update({"font.family": "serif"})

# # fig, ax = plt.subplots(3, 1, figsize=(6, 9), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1]})
# # # fig.suptitle(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(Lambda_int, kind_txt), fontsize=18, y=0.91)
# # fig.suptitle(r'$\Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(Lambda_int, kind_txt), fontsize=18, y=0.91)

# # # fig.suptitle(r'$k = 1 \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$', fontsize=16, y=0.91)

# # for i in range(3):
# #     ax[i].set_title('a = {}'.format(np.round(a_list[i], 3)), x=0.88, y=0.85, fontsize=18)
# #     ax[i].plot(x, vel_list[i], c='b', lw=1.5, label=r'\texttt{sim\_1\_11}')
# #     ax[i].plot(x_k1, vel_k1_list[i], c='k', lw=1.5, ls='dashed', label=r'\texttt{sim\_1}')

# #     # ax[i].set_ylabel(r'$\bar{v}_{l}\;[\mathrm{km\,s}^{-1}]$', fontsize=20)
# #     ax[i].set_ylabel(r'$\bar{v}_{l}\;[H_{0}L]$', fontsize=20)

# #     ax[i].minorticks_on()
# #     ax[i].tick_params(axis='both', which='both', direction='in', labelsize=15)
# #     ax[i].yaxis.set_ticks_position('both')
# # fig.align_labels()
# # plt.rcParams.update({"text.usetex": True})
# # ax[0].legend(fontsize=13, loc=3)

# # plt.rcParams.update({"text.usetex": False})
# # # ax[2].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
# # ax[2].set_xlabel(r'$x/L$', fontsize=20)

# # plt.subplots_adjust(hspace=0)
# # # plt.show()
# # plt.savefig('../plots/test/new_paper_plots/vel_ev_sm.pdf', bbox_inches='tight', dpi=300)#, pad_inches=0.3)
# # # plt.savefig('../plots/test/vel_ev.png', bbox_inches='tight', dpi=300)
# # plt.close()
