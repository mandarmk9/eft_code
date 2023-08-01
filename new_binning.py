#!/usr/bin/env python3

import os
import pandas
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from functions import read_sim_data, param_calc_ens, percentile_fde, spectral_calc, AIC, BIC
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def bin_arr(arr, bin_inds, method='mean'):
    binned_arr, std_dev = [], []
    for j in range(bin_inds.size):
        if method == 'mean':
            if j >= bin_inds.size - 1:
                arr_slice = arr[bin_inds[j]:-1]
                # dev = np.std(arr_slice)
                dev = np.sqrt(sum((arr_slice - np.mean(arr_slice))**2) / ((arr_slice.size)*(arr_slice.size-1)))
                binned_arr.append(np.mean(arr_slice))
                std_dev.append(dev)

            else:
                arr_slice = arr[bin_inds[j]:bin_inds[j+1]]
                # dev = np.std(arr_slice)
                dev = np.sqrt(sum((arr_slice - np.mean(arr_slice))**2) / ((arr_slice.size)*(arr_slice.size-1)))
                binned_arr.append(np.mean(arr_slice))
                std_dev.append(dev)

        elif method == 'median':
            if j >= bin_inds.size - 1:
                arr_slice = arr[bin_inds[j]:-1]
                dev = np.sqrt(sum((arr_slice - np.mean(arr_slice))**2) / ((arr_slice.size)*(arr_slice.size-1)))
                binned_arr.append(np.median(arr_slice))
                std_dev.append(dev)

            else:
                arr_slice = arr[bin_inds[j]:bin_inds[j+1]]
                dev = np.sqrt(sum((arr_slice - np.mean(arr_slice))**2) / ((arr_slice.size)*(arr_slice.size-1)))
                binned_arr.append(np.median(arr_slice))
                std_dev.append(dev)

        else:
            raise Exception('Method must be "mean" or "median"!')
    return np.array(binned_arr), np.array(std_dev)

def Power_fou(f1, f2):
    f1_k = np.fft.fft(f1)
    f2_k = np.fft.fft(f2)
    corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
    return corr[1]


path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
nbins_x, nbins_y, npars = 10, 10, 3
bin_method = 'median'

# j = 15
ctot2_list, ctot2_6_list, a_list, a_3list, a_6list, ctot2_3_list, err_list = [], [], [], [], [], [], []
for j in range(24):
    # j =10
    a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
    rho_b = 27.755 / a**3

    dv_l = -dv_l / (100 / (a**(1/2)))
    tau_l -= np.mean(tau_l)

    old_inds = np.arange(0, x.size)
    new_inds = np.argsort(dv_l)
    ind_map = np.array([(new_inds[j], old_inds[j]) for j in range(x.size)])

    dv_l = dv_l[new_inds]
    dc_l = dc_l[new_inds]
    tau_l = tau_l[new_inds]
    x = x[new_inds]

    n = x.size // nbins_x
    dv_bins = dv_l[::n]
    ind_bins = ind_map[::n]
    dc_bins = dc_l[::n]
    bin_inds = old_inds[::n]

    dels, dev = bin_arr(dc_l, bin_inds, bin_method)
    thes, dev = bin_arr(dv_l, bin_inds, bin_method)
    delsq, dev = bin_arr(dc_l**2, bin_inds, bin_method)
    thesq, dev = bin_arr(dv_l**2, bin_inds, bin_method)
    delthe, dev = bin_arr(dc_l*dv_l, bin_inds, bin_method)
    taus, yerr = bin_arr(tau_l, bin_inds, bin_method)
    ctot2_3 = np.real(Power_fou(tau_l/rho_b, dc_l) / Power_fou(dc_l, dv_l))
    try:
        guesses = 1, 1, 1
        def fitting_function(X, a0, a1, a2):
            x1, x2 = X
            return a0 + a1*x1 + a2*x2
        C, cov = curve_fit(fitting_function, (dels, thes), taus, guesses, sigma=yerr, method='lm', absolute_sigma=True)

        corr = np.zeros(cov.shape)
        cov[1,1] = cov[1,1] / rho_b
        cov[2,2] = cov[2,2] / rho_b
        corr[1,2] = cov[1,2] / np.sqrt(cov[1,1]*cov[2,2])

        err0, err1, err2 = np.sqrt(np.diag(cov))
        terr = (err1**2 + err2**2 + corr[2,1]*err1*err2 + corr[2,1]*err2*err1)**(0.5)
        fit = C[0] + C[1]*dels + C[2]*thes
        ctot2 = (C[1]+C[2])/rho_b
        ctot2_list.append(ctot2)
        a_3list.append(a)
        err_list.append(terr)

        # guesses = 1, 1, 1, 1, 1, 1
        # def fitting_function(X, a0, a1, a2, a3, a4, a5):
        #     x1, x2, x3, x4, x5 = X
        #     return a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5
        # C_, cov = curve_fit(fitting_function, (dels, thes, delsq, thesq, delthe), taus, guesses, sigma=None, method='lm', absolute_sigma=True)
        #
        # fit_6par = C_[0] + C_[1]*dels + C_[2]*thes + C_[3]*delsq + C_[4]*thesq + C_[5]*delthe
        # ctot2_6par = (C_[1]+C_[2])/rho_b
        # ctot2_6_list.append(ctot2_6par)
        # a_6list.append(a)
        print('a = {}, ctot2 = {}, ctot2_6par = {}'.format(a, ctot2, ctot2))

    except Exception as e:
        print(e)
        pass

    ctot2_3_list.append(ctot2_3)
    a_list.append(a)



a_list = np.array(a_list)
a_3list = np.array(a_3list)
a_6list = np.array(a_6list)

ctot2_list = np.array(ctot2_list)
err_list = np.array(err_list)
ctot2_6_list = np.array(ctot2_6_list)

df = pandas.DataFrame(data=[a_3list, ctot2_list, err_list])
file = open("./{}/new_binning_ctot2_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'wb')
pickle.dump(df, file)
file.close()

# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# fig, ax = plt.subplots()
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
# ax.yaxis.set_ticks_position('both')
#
# # ax.set_ylabel(r'$[\tau]_{\Lambda}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
# # ax.set_ylabel(r'$\delta_{l}$', fontsize=20)
# # ax.set_ylabel(r'$\delta_{l}$', fontsize=20)
#
# # ax.set_xlabel(r'$\theta_{l}$', fontsize=20)
# # ax.set_title(r'$a ={}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(np.round(a,3), int(Lambda/(2*np.pi)), kind_txt), fontsize=16, y=1.01)
# #
# # ax.plot(dv_l, tau_l, c='b', label='data', lw=2)
# # ax.plot(thes, taus, c='k', label='binned', lw=2, ls='dashdot')
# # ax.plot(thes, fit, c='seagreen', label='fit', lw=2, marker='o', ls='dashed')
# # ax.plot(thes, fit_6par, c='r', label='fit, six par', lw=2, marker='o', ls='dashed')
#
# err_list = np.array(err_list)
# ctot2_list = np.array(ctot2_list)
#
# ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=16, y=1.01)
# ax.plot(a_3list, ctot2_list, lw=2, c='seagreen', marker='o', label='binned fit, 1st order')
# # ax.fill_between(a_3list, ctot2_list-err_list, ctot2_list+err_list, color='darkslategray', alpha=0.55)
#
# # ax.plot(a_6list, ctot2_6_list, lw=2, c='k', marker='+', label='binned fit, 2nd order')
# ax.plot(a_list, ctot2_3_list, lw=2, c='orange', marker='x', label=r'$B^{+12}$')
#
# ax.set_xlabel(r'$a$', fontsize=14)
# ax.set_ylabel(r'$c_{\mathrm{tot}}^{2}(a)$', fontsize=14)
#
# # ax.plot(x, tau_l, lw=2, c='b')
# # ax.plot(x, fit, lw=2, c='k', ls='dasdot')
# # ax.plot(x, fit_6par, lw=2, c='r', ls='dashed')
#
#
# # ax.scatter(dc_bins, dv_bins, c='seagreen', label='bins', marker='o')
# # ax.plot(dc_bins, dv_bins, c='seagreen', label='bins', lw=2, ls='dashed')
#
# # print(dv_bins)
#
# plt.legend(fontsize=14)#, bbox_to_anchor=(1, 1))
#
# # plt.show()
# plt.savefig('../plots/test/new_paper_plots/test_ctot2.png', bbox_inches='tight', dpi=150)
# plt.close()
