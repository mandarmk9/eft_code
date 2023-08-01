#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from functions import read_sim_data, AIC, BIC, spectral_calc
from scipy.optimize import curve_fit
from tau_fits import tau_calc

def calc(j, Lambda, path, mode, kind, n_runs, n_use, folder_name):
    a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)

    H = a**(-1/2)*100
    dv_l = -dv_l / (H)
    taus = []
    taus.append(tau_l_0)
    for run in range(1, n_runs+1):
        path = path[:-2] + '{}/'.format(run)
        sol = read_sim_data(path, Lambda, kind, j, folder_name)
        taus.append(sol[-3])

    Nt = len(taus)

    tau_l = sum(np.array(taus)) / Nt

    rho_0 = 27.755
    rho_b = rho_0 / a**3
    H0 = 100

    diff = np.array([(taus[i] - tau_l)**2 for i in range(1, Nt)])
    yerr = np.sqrt(sum(diff) / (Nt*(Nt-1)))

    del_dc = spectral_calc(dc_l, 1, o=1, d=0)
    del_v = spectral_calc(dv_l, 1, o=1, d=0)

    n_ev = x.size // n_use
    dc_l_sp = dc_l[0::n_ev]
    dv_l_sp = dv_l[0::n_ev]
    del_v_sp = del_v[0::n_ev]
    del_dc_sp = del_dc[0::n_ev]
    tau_l_sp = tau_l[0::n_ev]
    x_sp = x[0::n_ev]
    yerr_sp = yerr[0::n_ev]

    # def fitting_function(X, a0, a1, a2):
    #     x1, x2 = X
    #     return a0 + a1*(x1+x2) + a2*(x1-x2)

    def fitting_function(X, a0, a1, a2):
        x1, x2 = X
        return a0 + a1*(x1) + a2*(x2)

    guesses = 1, 1, 1
    C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    fit = fitting_function((dc_l, dv_l), C[0], C[1], C[2])

    return a, x, tau_l, fit, C, cov

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
n_runs = 8
n_use = 8
mode = 1
Lambda = (2*np.pi) * 3
kinds = ['sharp', 'gaussian']
kinds_txt = ['sharp cutoff', 'Gaussian smoothing']

which = 1
kind = kinds[which]
kind_txt = kinds_txt[which]

# j = 14
folder_name = '/new_hier/data_{}/L{}/'.format(kind, int(Lambda/(2*np.pi)))

file_nums = [0, 14, 22]
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})




# print(corr)
fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1]})
fig.suptitle(r'$\Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=24)

ax[0].set_ylabel(r'$\langle\tau\rangle\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=22)
ax[2].set_ylabel(r'$\langle\tau\rangle\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=22)

ax[2].yaxis.set_label_position('right')
ax[1].set_xlabel(r'$x/L$', fontsize=20)

for j in range(3):
    file_num = file_nums[j]
    a, x, tau_l, fit, C, cov = calc(file_num, Lambda, path, mode, kind, n_runs, n_use, folder_name)
    corr = np.zeros(cov.shape)
    for i in range(3):
        corr[i,:] = [cov[i,j] / np.sqrt(cov[i,i]*cov[j,j]) for j in range(3)]

    print(corr)
    ax[j].set_title(r'$a = {}$'.format(np.round(a, 3)), x=0.15, y=0.9, fontsize=20)
    ax[j].plot(x, tau_l, c='b', lw=1.5, label=r'$\langle \tau\rangle$')
    ax[j].plot(x, fit, c='k', lw=1.5, ls='dashed', label=r'fit to $\langle \tau\rangle, N_{{\mathrm{{p}}}} = 3$')
    ax[j].minorticks_on()
    ax[j].tick_params(axis='both', which='both', direction='in', labelsize=18)
    ax[j].yaxis.set_ticks_position('both')


plt.legend(fontsize=18, bbox_to_anchor=(1, 1.325))
fig.align_labels()
plt.subplots_adjust(wspace=0.17)
plt.show()
# plt.savefig('../plots/paper_plots_final/tau_fits_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# plt.close()
