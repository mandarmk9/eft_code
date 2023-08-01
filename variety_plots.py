#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from functions import read_sim_data
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

j = 20
folder_name = '/new_hier/data_{}/L{}'.format(kind, int(Lambda/(2*np.pi)))


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


a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)
dv_l *= -np.sqrt(a) / 100
tau_l -= np.mean(tau_l)

distance = np.sqrt(dc_l**2 + dv_l**2)
per = np.percentile(distance, 5)
indices = np.where(distance < per)[0]

params_list = []
for ind in indices:
    params_list.append(new_param_calc(dc_l, dv_l, tau_l, 1, ind))

weights = None
C0_ = np.average([params_list[j][0] for j in range(len(params_list))], weights=weights)
C1_ = np.average([params_list[j][1] for j in range(len(params_list))], weights=weights)
C2_ = np.average([params_list[j][2] for j in range(len(params_list))], weights=weights)
C_ = [C0_, C1_, C2_]
fit = C0_ + C1_*dc_l + C2_*dv_l

##plotting
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
ax.set_xlabel(r'$\delta_{l}$', fontsize=18)
ax.set_ylabel(r'$\theta_{l}$', fontsize=18)
# ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(np.round(a, 3), int(Lambda/(2*np.pi)), kind_txt), fontsize=16)
ax.set_title(r'$a = {}, \Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(np.round(a, 3), int(Lambda/(2*np.pi)), kind_txt), fontsize=16)

ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=13.5)

cm = plt.cm.get_cmap('RdYlBu')

from matplotlib import ticker


fit = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2
del_tau = fit - tau_l
obj = ax.scatter(dc_l, dv_l, c=del_tau, s=20, cmap='rainbow', rasterized=True)#, norm=colors.Normalize(vmin=del_tau.min(), vmax=del_tau.max()))


cbar = fig.colorbar(obj, ax=ax)
# cbar.ax.set_ylabel(r'$[\tau]_{\Lambda}$', fontsize=18)
# cbar.ax.set_ylabel(r'$\Delta[\tau]_{\Lambda}\; [\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=18)
cbar.ax.set_ylabel(r'$\Delta[\tau]_{\Lambda}\; [\mathrm{M}_{\mathrm{p}}H_{0}^{2}L^{-1}]$', fontsize=18)

cbar.ax.tick_params(labelsize=12.5)

tick_locator = ticker.MaxNLocator(nbins=10)
cbar.locator = tick_locator
cbar.update_ticks()

# plt.savefig('../plots/test/new_paper_plots/tau_diff.png', bbox_inches='tight', dpi=150)
# plt.savefig('../plots/test/new_paper_plots/tau_diff_pre_sc_gauss.pdf', bbox_inches='tight', dpi=300)
# plt.savefig('../plots/test/new_paper_plots/tau_diff_post_sc_gauss.pdf', bbox_inches='tight', dpi=300)

plt.show()
