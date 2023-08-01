#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt

from functions import *

path = 'cosmo_sim_1d/sim_k_1_11/run1/'

# path = 'cosmo_sim_1d/multi_sim_3_15_33/run1/'
# plots_folder = 'test/multi_sim_3_15_33/'
plots_folder = 'paper_plots_final/'


def extract_fields(path, file_num):
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(file_num)
    moments_file = np.genfromtxt(path + moments_filename)
    # dk_par, a, dx = read_density(path, file_num)
    # x0 = 0.0
    # xn = 1.0 #+ dx
    # x_grid = np.arange(x0, xn, (xn-x0)/dk_par.size)
    # M0_par = np.real(np.fft.ifft(dk_par))
    # M0_par /= np.mean(M0_par)
    # f_M0 = interp1d(x_grid, M0_par, fill_value='extrapolate')

    x = moments_file[:,0]
    a = moments_file[:,-1][0]
    # dc = f_M0(x)
    dc = moments_file[:,2]
    v = moments_file[:,5]
    return a, x, dc, v


# j = 10
a_0, x, dc_0, v_0 = extract_fields(path, 0)
a_1, x, dc_1, v_1 = extract_fields(path, 12)
a_2, x, dc_2, v_2 = extract_fields(path, 35)
a_3, x, dc_3, v_3 = extract_fields(path, 50)

folder_name = '/hierarchy_coarse/'
def extract_fields(path, file_num):
    a, dx, M0_nbody, M1_nbody, M2_nbody, C0_nbody, C1_nbody, C2_nbody = read_hier(path, file_num, folder_name)
    x = np.arange(0, 1, dx)
    dc = M0_nbody #dc is 1+\delta
    v = C1_nbody
    return a, x, dc, v

# a_2, x, dc_2_prime, v_2 = extract_fields(path, 22)

# for j in range(0, 34):
#     fig, ax = plt.subplots()
#     ax.set_title(r'$a = {}$'.format(a_0), fontsize=14)
#     ax.set_ylabel(r'$1+\delta$', fontsize=14)
#     ax.plot(x, dc_0, lw=2, c='b')
#     ax.set_xlim(0.4, 0.45)
#     ax.set_ylim(-5, 100)
#
#     ax.minorticks_on()
#     ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=14)
#     ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
#     ax.ticklabel_format(scilimits=(-2, 3))
#     ax.yaxis.set_ticks_position('both')
#     # ax.grid(lw=0.2, ls='dashed', color='grey')
#     # plots_folder = 'test/paper_plots'
#     # savename = 'den_IC'
#     plt.savefig('../plots/test/dc_{0:03d}.png'.format(j), bbox_inches='tight', dpi=300)
#     plt.close()


# a_1 = 1.93
a_list = [a_0, a_1, a_2 , a_3]
den_list = [dc_0, dc_1, dc_2, dc_3]
vel_list = [v_0, v_1, v_2, v_3]

# print(a_list)
#
# k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2)) / (2*np.pi)
#
# dc_k = np.fft.fft(dc_2) / dc_2.size
# P = np.real(dc_k * np.conj(dc_k))
#
# dc_k_prime = np.fft.fft(dc_2_prime) / dc_2.size
# P_prime = np.real(dc_k_prime * np.conj(dc_k_prime))
#
# #density plot
# plt.rcParams.update({"text.usetex": True})
# fig, ax = plt.subplots()
# ax.scatter(k[1:16], np.log10(P[1:16]))
# ax.scatter(k[1:16], np.log10(P_prime[1:16]))
# plt.show()
#
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})


fig, ax = plt.subplots(4, 1, figsize=(6, 12), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1, 1]})
fig.suptitle(r'\texttt{sim\_1\_11}', fontsize=24, y=0.91, usetex=True)

for i in range(4):
    ax[i].plot(x, np.log10(den_list[i]), c='b', lw=1.5, label=r'$a = {}$'.format(np.round(a_list[i], 3)))
    ax[i].set_title('a = {}'.format(np.round(a_list[i], 3)), x=0.86, y=0.845, fontsize=24)
    # ax[i].legend(fontsize=14, loc=1)
    ax[i].set_ylabel(r'$\mathrm{log}_{10}\;(1+\delta)$', fontsize=26)
    # ax[i].set_ylabel(r'$(1+\delta)$', fontsize=20)

    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=22)
    ax[i].yaxis.set_ticks_position('both')
fig.align_labels()
# ax[3].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
ax[3].set_xlabel(r'$x/L$', fontsize=26)
ax[1].set_ylim(-0.5, 2.2)
# ax[0].set_ylim(-0.14, 0.16)
plt.subplots_adjust(hspace=0)
# plt.show()

plt.savefig(f'../plots/{plots_folder}/den_ev.pdf', bbox_inches='tight', dpi=300, pad_inches=0.3)
# plt.savefig('../plots/test/new_paper_plots/den_ev.png', bbox_inches='tight', dpi=300)
plt.close()

#velocity plot
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(4, 1, figsize=(6, 12), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1, 1]})
fig.suptitle(r'\texttt{sim\_1\_11}', fontsize=24, y=0.91, usetex=True)


for i in range(4):
    ax[i].plot(x, vel_list[i], c='b', lw=1.5, label=r'$a = {}$'.format(np.round(a_list[i], 3)))
    # ax[i].legend(fontsize=14, loc=1)
    ax[i].set_title('a = {}'.format(np.round(a_list[i], 3)), x=0.86, y=0.845, fontsize=24)

    # ax[i].set_ylabel(r'$\bar{v}\;[\mathrm{km\,s}^{-1}]$', fontsize=20)
    ax[i].set_ylabel(r'$\bar{v}\;[H_{0}L]$', fontsize=26)

    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=22)
    ax[i].yaxis.set_ticks_position('both')
fig.align_labels()
# ax[3].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
ax[3].set_xlabel(r'$x/L $', fontsize=26)

plt.subplots_adjust(hspace=0)
# plt.show()
plt.savefig(f'../plots/{plots_folder}/vel_ev.pdf', bbox_inches='tight', dpi=300, pad_inches=0.3)
# plt.savefig('../plots/test/vel_ev.png', bbox_inches='tight', dpi=300)
plt.close()



def kappa_calc(path, j):
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x_cell = moments_file[:,0]
    M0_nbody = moments_file[:,2]
    M1_nbody = moments_file[:,4]
    M2_nbody = moments_file[:,6]
    C1_nbody = moments_file[:,5]
    C2_nbody = moments_file[:,7]
    rho_b = 27.755 / a**3
    rho = M0_nbody * (rho_b)
    path = 'cosmo_sim_1d/sim_k_1_11/run1/shell_crossed_hier/'
    filename = 'hier_{0:04d}.hdf5'.format(j)
    file = h5py.File(path+filename, mode='r')
    header = file['/Header']
    count = header.attrs.get('frac')
    return a, x_cell, np.sqrt(np.abs(C2_nbody / rho)), 1-count

a0, x, kappa_0, c0 = kappa_calc(path, 0)
a1, x, kappa_1, c1 = kappa_calc(path, 12)
a2, x, kappa_2, c2 = kappa_calc(path, 35)
a3, x, kappa_3, c3 = kappa_calc(path, 50)
a_list = [a0, a1, a2, a3]
kappa_list = [kappa_0, kappa_1, kappa_2, kappa_3]
# clist = [c0, c1, c2, c3]

count = np.loadtxt(fname=path+'/count_fraction.txt', delimiter='\n')
clist = [count[0], count[12], count[35], count[50]]

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(4, 1, figsize=(6, 12), sharex=True, sharey=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1, 1]})
fig.suptitle(r'\texttt{sim\_1\_11}', fontsize=24, y=0.91, usetex=True)

for i in range(4):
    ax[i].plot(x, kappa_list[i], c='b', lw=1.5, label=r'$a = {}$'.format(np.round(a_list[i], 3)))
    ax[i].set_title('a = {}'.format(np.round(a_list[i], 3)), x=0.86, y=0.845, fontsize=24)
    # ax[i].set_ylabel(r'$\kappa\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=20)
    # ax[i].set_ylabel(r'$\rho^{-1}\Upsilon\;[H_{0}^{2}L^{2}]$', fontsize=20)
    ax[i].set_ylabel(r'$\sigma_{\mathrm{v}}\;[H_{0}L]$', fontsize=26)

    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=22)
    # ax[i].set_ylim(-0.5, 12)
    ax[i].yaxis.set_ticks_position('both')
    ax[i].text(0.05, 0.875, rf'$\nu = {np.round(clist[i], 3)}$', transform=ax[i].transAxes, fontsize=22)

fig.align_labels()
# ax[3].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
ax[3].set_xlabel(r'$x/L$', fontsize=26)
plt.subplots_adjust(hspace=0)
# plt.show()

plt.savefig(f'../plots/{plots_folder}/kappa_ev.pdf', bbox_inches='tight', dpi=300, pad_inches=0.3)
# plt.savefig(f'../plots/{plots_folder}/kappa_ev.pdf', bbox_inches='tight', dpi=300, pad_inches=0.3)

plt.close()

# #density plot
# fig, ax = plt.subplots(3, 1, figsize=(8, 12), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1]})
# for i in range(3):
#     ax[i].plot(x, np.log(den_list[i]+1), c='b', lw=1.5, label=r'$a = {}$'.format(np.round(a_list[i], 3)))
#     ax[i].legend(fontsize=14, loc=1)
#     ax[i].set_ylabel(r'$\mathrm{log}\;(1+\delta)$', fontsize=18)
#     ax[i].minorticks_on()
#     ax[i].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#     ax[i].yaxis.set_ticks_position('both')
# fig.align_labels()
# ax[2].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=18)
# plt.subplots_adjust(hspace=0)
# # plt.show()
# plt.savefig('../plots/test/new_paper_plots/den_ev.pdf', bbox_inches='tight', dpi=300)
# # plt.savefig('../plots/test/den_ev.png', bbox_inches='tight', dpi=300)
# plt.close()
#
#
# #velocity plot
# fig, ax = plt.subplots(3, 1, figsize=(8, 12), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1]})
# for i in range(3):
#     ax[i].plot(x, vel_list[i], c='b', lw=1.5, label=r'$a = {}$'.format(np.round(a_list[i], 3)))
#     ax[i].legend(fontsize=14)
#     ax[i].set_ylabel(r'$\bar{v}\;[\mathrm{km\,s}^{-1}]$', fontsize=18)
#     ax[i].minorticks_on()
#     ax[i].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#     ax[i].yaxis.set_ticks_position('both')
#
# fig.align_labels()
# ax[2].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=18)
# plt.subplots_adjust(hspace=0)
# # plt.show()
# plt.savefig('../plots/test/new_paper_plots/vel_ev.pdf', bbox_inches='tight', dpi=300)
# # plt.savefig('../plots/test/vel_ev.png', bbox_inches='tight', dpi=300)
# plt.close()
# fig, ax = plt.subplots(2, 2, figsize=(12,10), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
#
# ax[0,0].plot(x, dc_in, c='b', lw=2, label=r'$\delta_{\mathrm{in}}$')
# ax[0,0].set_ylabel(r'$1+\delta(x)$', fontsize=20)
#
# ax[0,1].plot(x, dc_fi, c='b', lw=2, label=r'$1+\delta_{\mathrm{fi}}$')
# # ax[0,1].set_ylabel(r'$1+\delta(x)$', fontsize=20)
#
# ax[1,0].plot(x, v_in, c='b', lw=2, label=r'$v_{\mathrm{in}}$')
# ax[1,0].set_ylabel(r'$v(x)$', fontsize=20, )
#
# ax[1,1].plot(x, v_fi, c='b', lw=2, label=r'$v_{\mathrm{fi}}$')
# # ax[1,1].set_ylabel(r'$v(x)$', fontsize=20)
#
# for i in range(2):
#     ax[i,0].set_title(r'$a = {}$'.format(a_in), fontsize=20)
#     ax[i,1].set_title(r'$a = {}$'.format(a_fi), fontsize=20)
#     ax[i,1].yaxis.set_label_position('right')
#
#     for j in range(2):
#         ax[i,j].set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=20)
#         ax[i,j].minorticks_on()
#         ax[i,j].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#         ax[i,j].yaxis.set_ticks_position('both')
#
# # ax.ticklabel_format(scilimits=(-2, 3))
# # # ax.grid(lw=0.2, ls='dashed', color='grey')
# # plt.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
#
# plots_folder = 'test/paper_plots'
# savename = 'den_v_plots'
# plt.tight_layout()
# # plt.savefig('../plots/test/in.png', bbox_inches='tight', dpi=300)
# plt.subplots_adjust(hspace=0.25)
# plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
# plt.close()
#


# fig, ax = plt.subplots()
#
# ax.set_title(r'$a = {}$'.format(a), fontsize=20)
# ax.set_ylabel(r'$v(x)[\mathrm{km}\;\mathrm{s}^{-1}]$', fontsize=20)
# ax.plot(x, C1_nbody, lw=2, c='b')
# ax.minorticks_on()
# ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
# ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
# ax.ticklabel_format(scilimits=(-2, 3))
# # ax.grid(lw=0.2, ls='dashed', color='grey')
# ax.yaxis.set_ticks_position('both')
# plots_folder = 'test/paper_plots'
# savename = 'vel_IC'
# plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
# plt.close()
