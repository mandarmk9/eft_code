#!/usr/bin/env python3
import h5py
import pickle
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from tqdm import tqdm
from functions import dc_in_finder, smoothing, dn, param_calc_ens
from alpha_c_function import *

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
A = []
mode = 1
Nfiles, N = 50, 50
Lambda_int = 3
Lambda = Lambda_int * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

n_runs = 8
n_use = n_runs-1
fm = '' #fitting method
nbins_x, nbins_y, npars = 20, 20, 3

def P13_finder(path, Nfiles, Lambda, kind, mode):
    print('P13 finder')
    Nx = 2048
    L = 1.0
    dx = L/Nx
    x = np.arange(0, L, dx)
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    a_list, P13_list, P11_list = [], [], []
    for j in tqdm(range(Nfiles)):
        a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
        dc_in, k = dc_in_finder(path, x, interp=True) #[0]
        dc_in = smoothing(dc_in, k, Lambda, kind)
        Nx = dc_in.size
        F = dn(3, L, dc_in)
        d1k = (np.fft.fft(F[0]) / Nx)
        d2k = (np.fft.fft(F[1]) / Nx)
        d3k = (np.fft.fft(F[2]) / Nx)
        P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
        P11 = (d1k * np.conj(d1k)) * (a**2)
        P13_list.append(np.real(P13)[mode])
        P11_list.append(np.real(P11)[mode])
        a_list.append(a)
        # print('a = ', a)
    return np.array(a_list), np.array(P13_list), np.array(P11_list) #, np.array(P)

def ctot_finder(Lambda, path, A, mode, kind, n_runs, n_use):
    print('ctot2 finder')
    a_list, ctot2_list, ctot2_2_list, ctot2_3_list = [], [], [], []
    for j in tqdm(range(Nfiles)):
        # sol = param_calc_ens(j, Lambda, path, A, mode, kind, n_runs, n_use)
        sol = param_calc_ens(j, Lambda, path, A, mode, kind, fitting_method=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)

        a_list.append(sol[0])
        ctot2_list.append(sol[2])
        ctot2_2_list.append(sol[3])
        ctot2_3_list.append(sol[4])
    return np.array(a_list), np.array(ctot2_list), np.array(ctot2_2_list), np.array(ctot2_3_list)

# a_list, P13, P11 = P13_finder(path, Nfiles, Lambda, kind, mode)

# df = pandas.DataFrame(data=[P13, P11])
# file = open("./{}/P13_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), "wb")
# pickle.dump(df, file)
# file.close()

# fde_method = 'percentile'
# folder_name = '/new_hier/data_{}/L{}'.format(kind, int(Lambda/(2*np.pi)))
# a_list, x, ac_true, ac, ac2, ac3, ac4 = alpha_c_finder(Nfiles, Lambda, path, A, mode, kind, fm=fm, npars=npars, fde_method=fde_method, folder_name='')
# df = pandas.DataFrame(data=[a_list, ac_true, ac, ac2, ac3, ac4])
# file = open("./{}/alpha_c_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), "wb")
# pickle.dump(df, file)
# file.close()


file = open("./{}/P13_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), "rb")
P13, P11 = np.array(pickle.load(file))
file.close()

file = open(f"./{path}/alpha_c_{kind}_{Lambda_int}.p", "rb")
read_file = pickle.load(file)
a_list, ac_true, ac, ac2, ac3, ac4, ac5, _, _, _, ac_pred = np.array(read_file)
file.close()


# file = open(f"./{path}/alpha_c_{kind}_{Lambda_int}.p", "rb")
# read_file = pickle.load(file)
# # a_list, alpha_c_true, alpha_c_naive, alpha_c_naive2, alpha_c_naive3, alpha_c_naive4, alpha_c_pred = np.array(read_file)
# a_list, ac_true, alpha_c_F3P, alpha_c_F6P, alpha_c_MW, alpha_c_SC, alpha_c_SCD, _, _, _, alpha_c_pred = np.array(read_file)
# file.close()

# axes = [a_list, x, ac_true, ac, ac2, ac3, ac4, err]
# for axis in axes:
#     axis = axis[:23]

e = 0.1

# for e in np.arange(0.01, 0.15, 0.01):
# e = np.round(e, 3)
err0_list, errp_list, errm_list = [], [], []
for j in range(N):
    a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, f'new_hier/data_{kind}/L{Lambda_int}/')
    P_lin = np.abs(np.conj(d1k) * d1k) * a**2
    # acp = (P_nb[mode] * (1 + 0.01*e)) - P_1l[mode] / (2 * (2*np.pi)**2 * P_lin[mode])
    # acm = (P_nb[mode] * (1 - 0.01*e)) - P_1l[mode] / (2 * (2*np.pi)**2 * P_lin[mode])
    
    acp = ac_true[j] + (P_nb[mode] * e / (2 * 100 * P_lin[mode]))
    acm = ac_true[j] - (P_nb[mode] * e / (2 * 100 * P_lin[mode]))
    errp_list.append(acp)
    errm_list.append(acm)

# print(errp_list, ac)

# print(np.mean(err0_list), np.mean(errp_list), np.mean(errm_list))

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})


# from scipy.optimize import curve_fit
#
# def fitting_function(X, a0, a1):
#     return a0*X[0] + a1*(X[1]**2)
# X = (np.ones(a_list[3:].size), a_list[3:])
#
# guesses = -1, 1
# C, cov = curve_fit(fitting_function, X, ac2[3:], method='lm', absolute_sigma=True)
# C0, C1 = C
#
# print(C0, C1)
# # plt.plot(a_list, ac)#/a_list**2)
# # plt.plot(a_list, P11/P13*a_list**2)
#
# # plt.plot(a_list[3:], ac2[3:])
# # plt.plot(a_list[3:], C0 + C1*X[1]**2, ls='dashed')
# plt.plot(a_list[3:], (ac2[3:]-C0) / C1 / a_list[3:]**2)
#
# # plt.ylim(0, 1)
# # plt.plot(a_list, a_list**4)
#
# plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_title(r'$\Lambda = {} \;$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=20)
ax.set_title(r'$\Lambda = {} \,k_{{\mathrm{{f}}}}$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=24, y=1.01)
ax.set_xlabel(r'$a$', fontsize=24)
# ax.set_ylabel('$c_{\mathrm{tot}}^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=18)
# ax.set_ylabel(r'$\alpha_{c} P_{11} / P_{13} \;[10^{-4}L^{-2}]$', fontsize=22)

# ax.set_ylabel(r'$\eta \;[10^{-4}L^{2}]$', fontsize=22)
ax.set_ylabel(r'$\eta(a, k=k_{\mathrm{f}})$', fontsize=24)

# ax.plot(a_list, ctot2, c='k', lw=1.5, zorder=4, marker='o', label=labels[0])
# ax.plot(a_list, ctot2_2, c='cyan', lw=1.5, marker='*', label=labels[1])
# ax.plot(a_list, ctot2_3, c='orange', lw=1.5, marker='v', label=labels[2])
# ax.plot(a_list, P13, c='blue', lw=1.5, marker='x', label=labels[3])


# ratio_true = P11[:N] / P13[:N] / 8000000 #2 * ac_true[:N] * P11[:N] / P13[:N] * (1 * (2*np.pi)**2)
# ratio = P11[:N] / P13[:N] / 8000000 #2 * ac[:N] * P11[:N]/ P13[:N] * (1 * (2*np.pi)**2)
# ratio2 = ac_true[:N] ##* P11[:N] / P13[:N]#2 * ac2[:N] * P11[:N] / P13[:N] * (1 * (2*np.pi)**2)
# ratio3 = ac[:N] #* P11[:N] / P13[:N]#2 * ac3[:N] * P11[:N] / P13[:N] * (1 * (2*np.pi)**2)
# # ratio4 = ac[:N] #* P11[:N] / P13[:N]#2 * ac4[:N] * P11[:N] / P13[:N] * (1 * (2*np.pi)**2)
# # ratio2 = -0.000133* a_list**2 * P11[:23] / P13[:23] * (1 * (2*np.pi)**2)

# # print(ac_true[:5], (P11/P13)[:5], ratio_true[:5])
# # print(ac[:5], (P11/P13)[:5], ratio[:5])

N = 20
ratio_true = ac_true[:N] * P11[:N] / P13[:N] #* (1 * (2*np.pi)**2)
# ratio_true = P11[:N] / P13[:N] #* (1 * (2*np.pi)**2)
ratio = ac[:N] * P11[:N] / P13[:N] #* P11[:N] / P13[:N] #* (1 * (2*np.pi)**2)
# ratio2 = ac2[:N] * P11[:N] / P13[:N] #* (1 * (2*np.pi)**2)
# ratio3 = ac3[:N] * P11[:N] / P13[:N] #* (1 * (2*np.pi)**2)
# ratio4 = ac4[:N] * P11[:N] / P13[:N] #* (1 * (2*np.pi)**2)
# ratio5 = ac5[:N] * P11[:N] / P13[:N] #* (1 * (2*np.pi)**2)



# flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')
# for j in range(N):
#     if flags[j] == 1:
#         sc_line = ax.axvline(a_list[j], c='teal', ls='dashed', lw=0.5, zorder=1)


# # err = err[:23]
# # err *= 1 * P11[:23] / P13[:23] * (1 * (2*np.pi)**2)
# # line6 = ax.axhline(np.mean(ratio_true[2:12]), lw=0.5, c='magenta')


# # ax.plot(a_list[:N], ratio_true[:N]/fac, c='b', lw=2, label=r'$P_{11} / P_{13}$') #, marker='o') #fit
# # ax.plot(a_list[:N], ratio2[:N]/fac, c='r', lw=2, ls='dashed', label=r'$\alpha^{\mathrm{true}}_{\mathrm{c}}$')#, marker='*') #M&W
# # ax.plot(a_list[:N], ratio3[:N]/fac, c='k', lw=2, ls='dashed', label=r'$\alpha_{\mathrm{c}}$ from fit')#, marker='v') #B^{+12}
# # plt.legend(fontsize=16)
# print(ratio[:N])
# print(ac[:N], P11[:N], P13[:N])
line0, = ax.plot(a_list[:N], ratio_true[:N], c='g', lw=2.5) #, marker='o')
line1, = ax.plot(a_list[:N], ratio[:N], c='k', lw=2.5, ls='dashed') #, marker='o') #F3P
# line2, = ax.plot(a_list[:N], ratio2[:N], c='seagreen', lw=2.5, ls='dashed', dashes=[1,2,1])#, marker='*') #F6P
# line3, = ax.plot(a_list[:N], ratio3[:N], c='midnightblue', lw=2.5, ls='dashed', dashes=[2,1,2])#, marker='v') #M&W
# line4, = ax.plot(a_list[:N], ratio4[:N], c='magenta', lw=2.5, ls='dashed', dashes=[2,2,1])#, marker='v') #SC
# line5, = ax.plot(a_list[:N], ratio5[:N], c='orange', lw=2.5, ls='dashed', dashes=[1,1,2])#, marker='v') #SC\delta



# ratio_p = np.array(errp_list)[:N] * P11[:N] / P13[:N] #* (1 * (2*np.pi)**2)
# ratio_m = np.array(errm_list)[:N] * P11[:N] / P13[:N] #* (1 * (2*np.pi)**2)


# # band = ax.fill_between(a_list[:N], ratio_m[:N], ratio_p[:N], color='darkslategray', alpha=0.5)



# # ax.axhline(np.mean(ratio_true[:N]), c='grey', lw=1, ls='dotted')
# # ax.axhline(np.mean(ratio[:N]/, c='grey', lw=1, ls='dotted')


# # handles = [line1, line2, line3, line4, line5, (line0, band), sc_line]#, line6]
# handles = [line1, line2, line3, line4, line5, line0, sc_line]#, line6]


# # labels=[r'from fit to $\langle\tau\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$', r'DDE', r'from matching $P_{N-}$body', r'$a_{\mathrm{sc}}$']
# # labels=[r'from fit to $\langle\tau\rangle$', r'M\&W', r'Spatial Corr', r'Spatial Corr from $\delta_{\ell}$', r'from matching $P_{N-}$body', r'$a_{\mathrm{sc}}$']
# labels = [r'F3P',  r'F6P', r'M\&W', 'SC', r'SC$\delta$', r'matching $P_{N\mathrm{-body}}$', r'$a_{\mathrm{shell}}$']

# ax.legend(handles=handles, labels=labels, fontsize=14, loc='upper right', framealpha=1)#, bbox_to_anchor=(0.43, 0.425))
# # ax.set_ylim(7/fac*2, 11.4/fac*2)

# # # err_line = ax.fill_between(a_list, ratio4-err, ratio4+err, color='darkslategray', alpha=0.55, rasterized=True)
# # # handles = [sc_line, line1, line2, line3, (line4, err_line)]
# # from scipy.optimize import curve_fit
# # def fitting_function(X, a0, a1):
# #     return a0*X[0] + a1*(X[1]**(0))
# # X = (np.ones(a_list.size), a_list)
# #
# # guesses = 1, 1
# # C, cov = curve_fit(fitting_function, X, ratio2, method='lm', absolute_sigma=True)
# # C0, C1 = C
# #
# # fit = fitting_function(X, C0, C1)
# # pred_line, = ax.plot(a_list, fit, ls='dashdot', c='magenta', lw=1.5, zorder=0) #DDE
# # handles.append(pred_line)
# # labels.append(r'$\eta \propto a^{4}$')

ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=16)
ax.yaxis.set_ticks_position('both')
# ax.set_ylim(0.2, 1)

plt.show()
# plt.savefig('../plots/paper_plots_final/loops_with_time_{}.pdf'.format(kind), bbox_inches='tight', dpi=300) #ctot2/err/lined/
# # # plt.savefig(f'../plots/paper_plots_final/loops_band/e_{e}.png', bbox_inches='tight', dpi=300) #ctot2/err/lined/
# plt.close()

# #
# # fig, ax = plt.subplots(figsize=(10, 6))
# # labels=[r'from fit to $\langle[\tau]_{\Lambda}\rangle$', 'M&W', r'$\mathrm{B^{+12}}$', r'$P_{13}$']
# #
# # ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=18)
# # ax.set_xlabel(r'$a$', fontsize=18)
# # # ax.set_ylabel('$c_{\mathrm{tot}}^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=18)
# # ax.set_ylabel('$c_{\mathrm{tot}}^{2} / P_{13}\;[\mathrm{km}^{2}s^{-2}]$', fontsize=18)
# #
# #
# # # ax.plot(a_list, ctot2, c='k', lw=1.5, zorder=4, marker='o', label=labels[0])
# # # ax.plot(a_list, ctot2_2, c='cyan', lw=1.5, marker='*', label=labels[1])
# # # ax.plot(a_list, ctot2_3, c='orange', lw=1.5, marker='v', label=labels[2])
# # # ax.plot(a_list, P13, c='blue', lw=1.5, marker='x', label=labels[3])
# #
# # ratio = ctot2 / P13
# # ratio2 = ctot2_2 / P13
# # ratio3 = ctot2_3 / P13
# #
# # ax.plot(a_list, ratio, c='k', lw=1.5, zorder=4, marker='o', label=labels[0])
# # ax.plot(a_list, ratio2, c='cyan', lw=1.5, marker='*', label=labels[1])
# # ax.plot(a_list, ratio3, c='orange', lw=1.5, marker='v', label=labels[2])
# #
# # ax.minorticks_on()
# # ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
# # ax.yaxis.set_ticks_position('both')
# # ax.legend(fontsize=11)
# # plt.show()
# # # plt.savefig('../plots/sim_k_1_11/ctot2_ev_{}.png'.format(kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# # # plt.close()

# npars = 3
# fde = 'percentile'
# folder_name = '/new_hier/data_{}/L{}/'.format(kind, int(Lambda/(2*np.pi)))

# a_list, x, ac_true, ac, ac2, ac3, ac4 = alpha_c_finder(Nfiles, Lambda, path, A, mode, kind, n_runs=8, n_use=10, H0=100, fde_method=fde, folder_name=folder_name, npars=npars)
# df = pandas.DataFrame(data=[a_list, ac_true, ac, ac2, ac3, ac4, err])
# file = open("./{}/alpha_c_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), "wb")
# pickle.dump(df, file)
# file.close()