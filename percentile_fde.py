#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from functions import read_sim_data, param_calc_ens, percentile_fde, spectral_calc, AIC, BIC
from scipy.interpolate import interp1d, interp2d, griddata
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
nbins_x, nbins_y, npars = 10, 10, 3

def binning(j, path, Lambda, kind, nbins_x, nbins_y, npars):
    a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)

    dv_l = -dv_l / (100 / (a**(1/2)))
    tau_l -= np.mean(tau_l)
    d_dcl = spectral_calc(dc_l, 1.0, o=1, d=0)
    d_dvl = spectral_calc(dv_l, 1.0, o=1, d=0)

    nvlines, nhlines = nbins_x, nbins_y
    min_dc, max_dc = dc_l.min(), dc_l.max()
    dc_bins = np.linspace(min_dc, max_dc, nvlines)

    min_dv, max_dv = dv_l.min(), dv_l.max()
    dv_bins = np.linspace(min_dv, max_dv, nhlines)

    mns, meds, counts, inds, yerr, taus, dels, thes, delsq, thesq, delthe, dx_del, dx_the, delcu, thecu, delsqthe, thesqdel, x_binned  = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    count = 0
    for i in range(nvlines-1):
        for j in range(nhlines-1):
            count += 1
            start_coos = (i,j)
            m,n = start_coos
            mns.append([m,n])

            indices = [l for l in range(x.size) if dc_bins[m] <= dc_l[l] <= dc_bins[m+1] and dv_bins[n] <= dv_l[l] <= dv_bins[n+1]]

            try:
                left = indices[0]
                right = len(indices)
                inds_ = list(np.arange(left, left+right+1, 1))

                tau_mean = sum(tau_l[inds_]) / len(inds_)
                delta_mean = sum(dc_l[inds_]) / len(inds_)
                theta_mean = sum(dv_l[inds_]) / len(inds_)
                del_sq_mean = sum((dc_l**2)[inds_]) / len(inds_)
                the_sq_mean = sum((dv_l**2)[inds_]) / len(inds_)
                del_the_mean = sum((dc_l*dv_l)[inds_]) / len(inds_)
                d_dcl_mean = sum((d_dcl)[inds_]) / len(inds_)
                d_dvl_mean = sum((d_dvl)[inds_]) / len(inds_)
                delcu_mean = sum((dc_l**3)[inds_]) / len(inds_)
                thecu_mean = sum((dv_l**3)[inds_]) / len(inds_)
                delsqthe_mean = sum((dv_l*dc_l**2)[inds_]) / len(inds_)
                thesqdel_mean = sum((dc_l*dv_l**2)[inds_]) / len(inds_)
                x_bin = sum(x[inds_]) / len(inds_)
                # print(tau_mean, delta_mean, theta_mean, x_bin)
                taus.append(tau_mean)
                dels.append(delta_mean)
                thes.append(theta_mean)
                delsq.append(del_sq_mean)
                thesq.append(the_sq_mean)
                delthe.append(del_the_mean)
                dx_del.append(d_dcl_mean)
                dx_the.append(d_dvl_mean)
                delcu.append(delcu_mean)
                thecu.append(thecu_mean)
                delsqthe.append(delsqthe_mean)
                thesqdel.append(thesqdel_mean)
                x_binned.append(x_bin)
                yerr_ = np.sqrt(sum((tau_l[inds_] - tau_mean)**2) / (len(inds_) - 1))

                yerr.append(yerr_)

                medians = np.mean(inds_)
                meds.append(medians)
                counts.append(count)
            except:
                left = None

    meds, counts = (list(t) for t in zip(*sorted(zip(meds, counts))))
    meds, x_binned = (list(t) for t in zip(*sorted(zip(meds, x_binned))))
    meds, taus = (list(t) for t in zip(*sorted(zip(meds, taus))))
    meds, dels = (list(t) for t in zip(*sorted(zip(meds, dels))))
    meds, thes = (list(t) for t in zip(*sorted(zip(meds, thes))))
    meds, delsq = (list(t) for t in zip(*sorted(zip(meds, delsq))))
    meds, thesq = (list(t) for t in zip(*sorted(zip(meds, thesq))))
    meds, delthe = (list(t) for t in zip(*sorted(zip(meds, delthe))))
    meds, dx_del = (list(t) for t in zip(*sorted(zip(meds, dx_del))))
    meds, dx_the = (list(t) for t in zip(*sorted(zip(meds, dx_the))))
    meds, dx_the = (list(t) for t in zip(*sorted(zip(meds, dx_the))))
    meds, thecu = (list(t) for t in zip(*sorted(zip(meds, dx_the))))
    meds, delsqthe = (list(t) for t in zip(*sorted(zip(meds, delsqthe))))
    meds, thesqdel = (list(t) for t in zip(*sorted(zip(meds, thesqdel))))

    if npars == 1: #constant
        def fitting_function(X, a0):
            return a0 * X[0]
        X = (np.ones(len(dels)))
        X_ = (np.ones(len(x)))
        guesses = 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0 = C
        fit_sp = C0 * X
        fit = C0 * X_

    elif npars == 3: #first-order
        def fitting_function(X, a0, a1, a2):
            return a0*X[0] + a1*X[1] + a2*X[2]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes))
        X_ = (np.ones(len(x)), dc_l, dv_l)
        guesses = 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2 = C
        fit_sp = fitting_function(X, C0, C1, C2)
        fit = fitting_function(X_, C0, C1, C2)

    elif npars == 32: #first-order
        def fitting_function(X, a0, a1, a2):
            return a0*X[0] + a1*X[1] + a2*X[2]
        X = (np.ones(len(dels)), np.array(dels), np.array(dels)**2)
        X_ = (np.ones(len(x)), dc_l, dc_l**2)
        guesses = 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2 = C
        fit_sp = fitting_function(X, C0, C1, C2)
        fit = fitting_function(X_, C0, C1, C2)
        npars = 3


    elif npars == 5: #derivative terms
        def fitting_function(X, a0, a1, a2, a3, a4):
            return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes), np.array(dx_del), np.array(dx_the))
        X_ = (np.ones(len(x)), dc_l, dv_l, d_dcl, d_dvl)
        guesses = 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4 = C
        fit_sp = fitting_function(X, C0, C1, C2, C3, C4)
        fit = fitting_function(X_, C0, C1, C2, C3, C4)


    elif npars == 6: #second-order
        def fitting_function(X, a0, a1, a2, a3, a4, a5):
            return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4] + a5*X[5]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes), np.array(delsq), np.array(thesq), np.array(delthe))
        X_ = (np.ones(len(x)), dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l)
        guesses = 1, 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4, C5 = C
        fit_sp = fitting_function(X, C0, C1, C2, C3, C4, C5)
        fit = fitting_function(X_, C0, C1, C2, C3, C4, C5)

    else:
        pass

    resid = fit_sp - taus
    chisq = sum((resid / yerr)**2)
    red_chi = chisq / (len(dels) - npars)
    # print('chisq = ', chisq, 'dels size = ', len(dels))
    # aic = AIC(npars, red_chi, n=1)
    aic = AIC(npars, chisq, n=1)
    bic = BIC(npars, len(taus), chisq)
    # print(a, chisq, red_chi)
    # print(C)
    return a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, fit, cov, C, x_binned, red_chi

for j in range(1):
    j = 15
    a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
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

    # dist = 1
    # ind = np.argmin(dc_l**2 + dv_l**2)
    per = 20
    # ind_ord = np.argsort(dv_l)
    # dc_l = dc_l[ind_ord]
    # dv_l = dv_l[ind_ord]
    # tau_l = tau_l[ind_ord]
    C_, err_, C_first, indices, ind_00 = percentile_fde(dc_l, dv_l, tau_l, per)
    # print('ind out', dc_l[ind_00], dv_l[ind_00], tau_l[ind_00])

    distance = np.sqrt(dc_l**2 + dv_l**2)
    per_inds = np.percentile(distance, per)
    indices = np.where(distance < per_inds)[0]

    # params_list = []
    # for ind in indices:
    #     C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
    #     params_list.append(C_)
    #
    #
    # C0_ = np.median([params_list[j][0] for j in range(len(params_list))])
    # C1_ = np.median([params_list[j][1] for j in range(len(params_list))])
    # C2_ = np.median([params_list[j][2] for j in range(len(params_list))])
    # C_ = [C0_, C1_, C2_]

    guesses = 1, 1, 1
    rho_b = 27.755 / (a**3)


    def fitting_function(X, a0, a1, a2):
        x1, x2 = X
        return a0 + a1*x1 + a2*x2
    C, cov = curve_fit(fitting_function, (dc_l, dv_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
    fit = C[0] + C[1]*dc_l + C[2]*dv_l
    resid = sum((tau_l - fit)**2)
    print('fit: ', (C[1]+C[2])/rho_b, 'resid: ', resid)


    def fitting_function(X, a0, a1, a2):
        x1 = X
        return a0 + a1*x1 + a2*(x1**2)
    C, cov = curve_fit(fitting_function, (dc_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
    fit_2 = C[0] + C[1]*dc_l + C[2]*dc_l**2
    resid = sum((tau_l - fit_2)**2)
    print('fit 2nd: ', (C[1])/rho_b, 'resid: ', resid)


    def fitting_function(X, a0, a1, a2, a3, a4, a5):
        return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4] + a5*X[5]
    X = (np.ones(len(x)), dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l)
    guesses = 1, 1, 1, 1, 1, 1
    C, cov = curve_fit(fitting_function, X, tau_l, sigma=None, method='lm', absolute_sigma=True)
    C0, C1, C2, C3, C4, C5 = C
    fit_2_6par = fitting_function(X, C0, C1, C2, C3, C4, C5)
    resid = sum((tau_l - fit_2_6par)**2)

    print('fit 2nd, six: ', (C[1]+C[2])/rho_b, 'resid: ', resid)

    # fit = fit[ind_ord]
    # fit_fde = fit_fde[ind_ord]
    # fit_fde_first = fit_fde_first[ind_ord]

    # print('C0_deriv = ', C_[0], 'C0_fit = ', C[0])#, 'C0_first_deriv = ', C_first[0])
    # print('C1_deriv = ', C_[1], 'C1_fit = ', C[1])
    # print('C2_deriv = ', C_[2], 'C2_fit = ', C[2])

    binned_sol = binning(j, path, Lambda, kind, nbins_x, nbins_y, 3)
    taus = binned_sol[5]
    thes = binned_sol[7]
    binned_fit = binned_sol[-5]
    C_sol = binned_sol[-3]
    resid = sum((tau_l - binned_fit)**2)
    red_chi = binned_sol[-1]
    print('binned: ', (C_sol[1]+C_sol[2])/rho_b, 'resid: ', resid)


    binned_sol_second_order = binning(j, path, Lambda, kind, nbins_x, nbins_y, 32)
    taus_2 = binned_sol_second_order[5]
    thes_2 = binned_sol_second_order[7]
    binned_fit_2 = binned_sol_second_order[-5]
    C_sol_2 = binned_sol_second_order[-3]
    resid = sum((tau_l - binned_fit_2)**2)
    red_chi_2 = binned_sol_second_order[-1]
    print('red chi-sq = ', red_chi_2)
    print('binned 2nd: ', C_sol_2[1]/rho_b, 'resid: ', resid)

    binned_sol_second_order_6 = binning(j, path, Lambda, kind, nbins_x, nbins_y, 6)
    taus_3 = binned_sol_second_order_6[5]
    thes_3 = binned_sol_second_order_6[7]
    binned_fit_3 = binned_sol_second_order_6[-5]
    C_sol_3 = binned_sol_second_order_6[-3]
    red_chi_3 = binned_sol_second_order_6[-1]
    resid = sum((tau_l - binned_fit_3)**2)
    print('binned 2nd, six: ', (C_sol_3[1]+C_sol_3[2])/rho_b, 'resid: ', resid)

    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax.yaxis.set_ticks_position('both')
    ax.set_ylabel(r'$[\tau]_{\Lambda}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
    # ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
    ax.set_xlabel(r'$\theta_{l}$', fontsize=20)

    ax.set_title(r'$a ={}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(np.round(a,3), int(Lambda/(2*np.pi)), kind_txt), fontsize=16, y=1.01)
    # plt.plot(x, tau_l, c='b', label=r'measured')
    # plt.plot(x, fit, c='r', ls='dashed', label='fit')
    # plt.plot(x, fit_fde, c='k', ls='dashed', label='FDE')
    # plt.plot(x, fit_fde_first, c='magenta', ls='dotted', label='FDE 1st order')

    plt.axhline(0, c='grey', ls='dashed', lw=0.3)
    plt.axvline(0, c='grey', ls='dashed', lw=0.3)

    plt.plot(dv_l, tau_l, c='b', label=r'measured')
    plt.plot(dv_l, fit, c='cyan', ls='dashdot', label='fit 1st, 3')
    # # plt.plot(dv_l, fit_2, c='orange', ls='dashdot', label='fit 2nd, 6')
    plt.plot(dv_l, fit_2_6par, c='olive', ls='dashdot', label='fit 2nd, 6')

    # plt.plot(dv_l, fit_fde, c='xkcd:dried blood', ls='dashdot', label='FDE')

    plt.plot(dv_l, binned_fit, c='k', ls='dashed', label=r'binned 1st, red. $\chi^{{2}} = {}$'.format(np.round(red_chi, 3)))
    # plt.plot(dv_l, binned_fit_2, c='r', ls='dashed', label=r'binned 2nd (3 par), red. $\chi^{{2}} = {}$'.format(np.round(red_chi_2, 3)))
    plt.plot(dv_l, binned_fit_3, c='magenta', ls='dotted', label=r'binned 2nd, red. $\chi^{{2}} = {}$'.format(np.round(red_chi_3, 3)))

    plt.scatter(thes, taus, c='seagreen', label='bins')
    # plt.scatter(dv_l[indices], tau_l[indices], s=20, c='seagreen', label='FDE: used points')

    # plt.plot(dv_l, fit_fde, c='cyan', ls='dotted', label='FDE')

    # plt.plot(dv_l, fit_fde_first, c='magenta', ls='dotted', label='FDE 1st order')

    # plt.scatter(dv_l[indices], tau_l[indices], s=20, c='seagreen', label='used points')

    # plt.scatter(dv_l[ind_00], tau_l[ind_00], s=20, c='orange')
    # plt.axhline(tau_l[ind_00], c='cyan')
    plt.legend(fontsize=14)#, bbox_to_anchor=(1, 1))

    plt.show()
    # plt.savefig('../plots/test/new_paper_plots/fit_tau_theta/tau_theta_{}.png'.format(j), bbox_inches='tight', dpi=150)
    # # plt.savefig('../plots/test/new_paper_plots/tau_theta_{}.png'.format(j), bbox_inches='tight', dpi=150)
    #
    # plt.close()
