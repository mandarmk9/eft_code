#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pickle
from scipy.optimize import curve_fit
from functions import read_sim_data, param_calc_ens, spectral_calc, AIC, BIC, binning
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
j = 3
nbins_x, nbins_y = 10, 10

a_list, ctot2, dc_list, dv_list, tau_list, mean_tau, mean_dc = [], [], [], [], [], [], []
for j in range(0, 50):
    a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
    bin_sol = binning(j, path, Lambda, kind, nbins_x, nbins_y, 3)
    taus = np.array(bin_sol[5])
    dels = np.array(bin_sol[6])
    lin_dels = np.array(bin_sol[-1])


    a_list.append(a)
    dc_list.append(np.real((np.fft.fft(dc_l)/dc_l.size)[1]))
    dv_list.append(np.real((np.fft.fft(dv_l)/dv_l.size)[1]))
    tau_list.append(np.real((np.fft.fft(tau_l)/tau_l.size)[1]))
    mean_tau.append(np.real((np.fft.fft(taus)/taus.size)[1]))
    mean_dc.append(np.real((np.fft.fft(dels)/dels.size)[1]))

    # M&W Estimator
    Lambda_int = int(Lambda / (2*np.pi))
    rho_b = 27.755 / a**3
    tau_l_k = np.fft.fft(taus) / x.size
    d1k = np.fft.fft(lin_dels) / lin_dels.size
    num = (np.conj(a * d1k) * ((np.fft.fft(taus)) / x.size))
    denom = ((d1k * np.conj(d1k)) * (a**2))
    ntrunc = int(num.size-Lambda_int)
    num[Lambda_int+1:ntrunc] = 0
    denom[Lambda_int+1:ntrunc] = 0
    ctot2_a = np.real(sum(num) / sum(denom)) / rho_b

    ctot2.append(ctot2_a)
    print('a = ', a, 'ctot2 = ', ctot2_a)

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=18, y=1.01)
ax.set_xlabel(r'$a$', fontsize=20)
# ax.set_ylabel('$$', fontsize=20)
# plt.legend(handles=[ctot2_line, ctot2_2_line, ctot2_3_line, (ctot2_4_line, ctot2_4_err)], labels=[r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'binned fit'], fontsize=14, loc=1, bbox_to_anchor=(1,1), framealpha=0.75)
# ax.plot(a_list, tau_list, c='b', lw=2, label=r'$\widetilde{\tau}_{l}(k=1)$')
# ax.plot(a_list, dc_list, c='r', lw=2, label=r'$\widetilde{\delta}_{l}(k=1)$')

# ax.plot(a_list, mean_tau, c='cyan', ls='dashed', lw=2, label=r'binned $\widetilde{\tau}_{l}(k=1)$')
# ax.plot(a_list, mean_dc, c='k', ls='dashed', lw=2, label=r'binned $\widetilde{\delta}_{l}(k=1)$')

ax.plot(a_list, ctot2, c='k', lw=2, label=r'binned $c_{\mathrm{tot}}^{2}$')

ax.axvline(1.8, lw=1, ls='dashed', c='grey', label=r'$a_{\mathrm{sc}}$')
ax.axvline(4.35, lw=1, ls='dashed', c='grey')

plt.legend(fontsize=14)
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.yaxis.set_ticks_position('both')
plt.savefig('../plots/test/new_paper_plots/ctot2_mw_binned.png'.format(kind), bbox_inches='tight', dpi=150)
plt.close()
# plt.show()
