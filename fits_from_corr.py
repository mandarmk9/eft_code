#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from functions import read_sim_data, AIC, BIC, spectral_calc
from scipy.optimize import curve_fit
from tau_fits import tau_calc

def calc(j, Lambda, path, mode, kind, n_runs, n_use, folder_name):
    a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)
    rho_0 = 27.755
    rho_b = rho_0 / a**3

    # spatial correlations
    H = a**(-1/2)*100
    dv_l = dv_l / (H)
    tD = np.mean(tau_l*dc_l) / rho_b
    tT = np.mean(tau_l*dv_l) / rho_b
    DT = np.mean(dc_l*dv_l)
    TT = np.mean(dv_l*dv_l)
    DD = np.mean(dc_l*dc_l)
    rhs = (tD / DT) - (tT / TT)
    lhs = (DD / DT) - (DT / TT)
    cs2 = rhs / lhs
    cv2 = (DD*cs2 - tD) / DT
    ctot2 = (cs2+cv2)
    ctot2_del = (tD/ DD)

    fit = np.mean(tau_l) + rho_b*cs2*dc_l - rho_b*cv2*dv_l
    return a, x, tau_l, fit, ctot2, ctot2_del

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
n_runs = 8
n_use = 8
mode = 1
Lambda = (2*np.pi) * 3
kinds = ['sharp', 'gaussian']
kinds_txt = ['sharp cutoff', 'Gaussian smoothing']

which = 0
kind = kinds[which]
kind_txt = kinds_txt[which]

# j = 14
folder_name = '/new_hier/data_{}/L{}/'.format(kind, int(Lambda/(2*np.pi)))


file_nums = [0, 14, 50]
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})


fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=False, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1]})
fig.suptitle(r'$\Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=24)

ax[0].set_ylabel(r'$\langle\tau\rangle\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=22)
ax[2].set_ylabel(r'$\langle\tau\rangle\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=22)

ax[2].yaxis.set_label_position('right')
ax[1].set_xlabel(r'$x/L$', fontsize=20)

for j in range(3):
    file_num = file_nums[j]
    a, x, tau_l, fit, ctot2, ctot2_del = calc(file_num, Lambda, path, mode, kind, n_runs, n_use, folder_name)
    ax[j].set_title(r'$a = {}$'.format(np.round(a, 3)), x=0.15, y=0.9, fontsize=20)
    ax[j].plot(x, tau_l, c='b', lw=1.5, label=r'$\langle \tau\rangle$')
    ax[j].plot(x, fit, c='k', lw=1.5, ls='dashed', label=r'$\tau$ from Spatial Corr')
    ax[j].minorticks_on()
    ax[j].tick_params(axis='both', which='both', direction='in', labelsize=18)
    ax[j].yaxis.set_ticks_position('both')

plt.legend(fontsize=18, bbox_to_anchor=(1, 1.325))
fig.align_labels()
plt.subplots_adjust(wspace=0.17)
# plt.show()
plt.savefig('../plots/paper_plots_final/tau_fits_corr_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
plt.close()
