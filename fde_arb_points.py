#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from functions import read_sim_data, param_calc_ens
from scipy.interpolate import interp1d, interp2d, griddata
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

j = 20
# folder_name = '/hierarchy/'
folder_name = '/new_hier/data_{}/L{}'.format(kind, int(Lambda/(2*np.pi)))

a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)
dv_l *= -np.sqrt(a) / 100
tau_l -= np.mean(tau_l)

def new_param_calc(dc_l, dv_l, tau_l, dist, ind):
    def dir_der_o1(X, tau_l, ind):
        """Calculates the first-order directional derivative of tau_l along the vector X."""
        ind_right = ind + 2
        if ind_right >= tau_l.size:
            ind_right = ind_right - tau_l.size
            print(ind, ind_right)

        x1 = np.array([X[0][ind], X[1][ind]])
        x2 = np.array([X[0][ind_right], X[1][ind_right]])
        v = (x2 - x1)
        D_v_tau = (tau_l[ind_right] - tau_l[ind]) / v[0]
        return v, D_v_tau

    def dir_der_o2(X, tau_l, ind):
        """Calculates the second-order directional derivative of tau_l along the vector X."""
        ind_right = ind + 4
        if ind_right >= tau_l.size:
            ind_right = ind_right - tau_l.size
            print(ind, ind_right)
        x1 = np.array([X[0][ind], X[1][ind]])
        x2 = np.array([X[0][ind_right], X[1][ind_right]])
        v1, D_v_tau1 = dir_der_o1(X, tau_l, ind)
        v2, D_v_tau2 = dir_der_o1(X, tau_l, ind_right)
        v = (x2 - x1)
        D2_v_tau = (D_v_tau2 - D_v_tau1) / v[0]
        return v, D2_v_tau



    X = np.array([dc_l, dv_l])
    params_list = []
    for j in range(-dist//2, dist//2 + 1):
        v1, dtau1 = dir_der_o1(X, tau_l, ind+j)
        v1_o2, dtau1_o2 = dir_der_o2(X, tau_l, ind+j)
        dc_0, dv_0 = dc_l[ind], dv_l[ind]
        C_ = [((tau_l[ind])-(dtau1*dc_0)+((dtau1_o2*dc_0**2)/2)), dtau1-(dtau1_o2*dc_0), dtau1_o2/2]
        # C_ = [(tau_l[ind]), dtau1-(dtau1_o2*dc_0), dtau1_o2/2]

        params_list.append(C_)

    params_list = np.array(params_list)
    dist = params_list.shape[0]
    if dist != 0:
        C0_ = np.mean(np.array([params_list[j][0] for j in range(dist)]))
        C1_ = np.mean(np.array([params_list[j][1] for j in range(dist)]))
        C2_ = np.mean(np.array([params_list[j][2] for j in range(dist)]))
        C_ = [C0_, C1_, C2_]
    else:
        C_ = [0, 0, 0]
    return C_

dist = 1
# ind = np.argmin(dc_l**2 + dv_l**2)
distance = np.sqrt(dc_l**2 + dv_l**2)
per_ = 50
per = np.percentile(distance, per_)
# print(per)
print(distance - per)
indices = np.where(distance < per)[0]
print(indices)

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
ax.set_ylabel(r'Distance from $(0,0)$', fontsize=18)
ax.set_xlabel(r'$x/L$', fontsize=18)
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.yaxis.set_ticks_position('both')
ax.plot(x, distance, c='k', lw=2)
ax.plot(x[indices], distance[indices], c='r', ls='dashed', lw=2)
ax.axhline(per, c='gray', label=r'${}^{{\mathrm{{th}}}}$ percentile'.format(per_))
plt.legend(fontsize=12)
plt.show()

# weights = 1/(np.abs(distance[indices]))

# # plt.plot(dv_l, tau_l)
# # plt.show()
# #
# # per = np.percentile(np.sqrt((dc_l-dc_l[dc_l.size//2])**2 + (dv_l-dc_l[dv_l.size//2])**2), 5)
# # indices2 = np.where(distance < per)[0]
# #
# # indices2 = np.argmax(tau_l)
#
# # indices = indices[100]#np.union1d(indices[:1], indices[-100:])
#
# # M = 1000
# # max_ind = np.argmax(tau_l)
# # thresh = 50
# # indices2 = np.union1d(np.arange(max_ind-M, max_ind-M/2, 1, dtype='int'), np.arange(max_ind+M/2, max_ind+M, 1, dtype='int'))
# # indices = np.union1d(indices, indices2)
# #
# #
# #
# # M = 100
# # max_ind = np.argmin(tau_l)
# # thresh = 0
# # indices3 = np.union1d(np.arange(max_ind-M, max_ind-M/2, 1, dtype='int'), np.arange(max_ind+M/2, max_ind+M, 1, dtype='int'))
# # indices = np.union1d(indices, indices3)
#
#
# # indices = np.union1d(indices, np.array([5000-100]))
#
# # indices = [ind, ind-10000]
# # indices = [np.argmax(tau_l), 0]
# params_list = []
# for ind in indices:
#     C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
#     params_list.append(C_)
#     # print(C_)
#
# # morm = 'mean'
# morm = 'mean'
#
# if morm == 'median':
#     C0_ = np.median([params_list[j][0] for j in range(len(params_list))])
#     C1_ = np.median([params_list[j][1] for j in range(len(params_list))])
#     C2_ = np.median([params_list[j][2] for j in range(len(params_list))])
# elif morm == 'mean':
#     C0_ = np.average([params_list[j][0] for j in range(len(params_list))], weights=weights)
#     C1_ = np.average([params_list[j][1] for j in range(len(params_list))], weights=weights)
#     C2_ = np.average([params_list[j][2] for j in range(len(params_list))], weights=weights)
#     # C0_ = np.mean([params_list[j][0] for j in range(len(params_list))])
#     # C1_ = np.mean([params_list[j][1] for j in range(len(params_list))])
#     # C2_ = np.mean([params_list[j][2] for j in range(len(params_list))])
#
#
# C_ = [C0_, C1_, C2_]
#
# guesses = 1, 1, 1
#
# def fitting_function(X, a0, a1, a2):
#     x1 = X
#     return a0 + a1*x1 + a2*(x1**2)
# C, cov = curve_fit(fitting_function, (dc_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
#
# fit = C[0] + C[1]*dc_l + C[2]*dc_l**2
# fit_fde = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2
#
# print('C0_deriv = ', C_[0], 'C0_fit = ', C[0])
# print('C1_deriv = ', C_[1]/(27.755/a**3), 'C1_fit = ', C[1])
# print('C2_deriv = ', C_[2], 'C2_fit = ', C[2])
#
#
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# fig, ax = plt.subplots()
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
# ax.yaxis.set_ticks_position('both')
# # ax.set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
# ax.set_ylabel(r'$[\tau]_{\Lambda}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
#
# ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
# ax.set_title(r'$a ={}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(np.round(a,3), int(Lambda/(2*np.pi)), kind_txt), fontsize=16, y=1.01)
#
# plt.plot(x, tau_l, c='b', label=r'measured')
# plt.plot(x, fit, c='r', ls='dashed', label='fit')
# plt.plot(x, fit_fde, c='k', ls='dashed', label='FDE: {}, {}\%'.format(morm, per_))
# plt.scatter(x[indices], tau_l[indices], s=20, c='seagreen')
# # plt.plot(x, fit3, c='cyan', ls='dotted', label='using derivatives 2')
#
#
# # plt.plot(x, est, c='k', ls='dashed', label='using derivatives')
# plt.legend(fontsize=14)
# plt.show()
# # plt.savefig('../plots/test/new_paper_plots/fde_{}_{}.png'.format(per_, morm), bbox_inches='tight', dpi=150)
# # plt.close()
