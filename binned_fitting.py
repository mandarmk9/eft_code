#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pickle
from scipy.optimize import curve_fit
from functions import read_sim_data, param_calc_ens, spectral_calc, AIC, BIC
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
kind = 'gaussian'
kind_txt = 'Gaussian smoothing'
j = 3

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
                yerr_ = np.sqrt(sum((tau_l[inds_] - tau_mean)**2) / (1*(len(inds_) - 1)))
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

    elif npars == 8: #second-order + derivative terms
        def fitting_function(X, a0, a1, a2, a3, a4, a5, a6, a7):
            return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4] + a5*X[5] + a6*X[6] + a7*X[7]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes), np.array(dx_del), np.array(dx_the), np.array(delsq), np.array(thesq), np.array(delthe))
        X_ = (np.ones(len(x)), dc_l, dv_l, d_dcl, d_dvl, dc_l**2, dv_l**2, dc_l*dv_l)
        guesses = 1, 1, 1, 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4, C5, C6, C7 = C
        fit_sp = fitting_function(X, C0, C1, C2, C3, C4, C5, C6, C7)
        fit = fitting_function(X_, C0, C1, C2, C3, C4, C5, C6, C7)

    elif npars == 10: #second-order + derivative terms
        def fitting_function(X, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
            return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4] + a5*X[5] + a6*X[6] + a7*X[7] + a8*X[8] + a9*X[9]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes), np.array(delsq), np.array(thesq), np.array(delthe), np.array(delcu), np.array(thecu), np.array(delsqthe), np.array(thesqdel))
        X_ = (np.ones(len(x)), dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l, dc_l**3, dv_l**3, dv_l*(dc_l**2), dc_l*(dv_l**2))

        guesses = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4, C5, C6, C7, C8, C9 = C
        fit_sp = fitting_function(X, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9)
        fit = fitting_function(X_, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9)

    elif npars == 12: #second-order + derivative terms
        def fitting_function(X, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11):
            return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4] + a5*X[5] + a6*X[6] + a7*X[7] + a8*X[8] + a9*X[9] + a10*X[10] + a11*X[11]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes), np.array(dx_del), np.array(dx_the), np.array(dx_del)*np.array(dels), np.array(dx_the)*np.array(thes), np.array(dx_del)*np.array(thes), np.array(dx_the)*np.array(dels), np.array(delsq), np.array(thesq), np.array(delthe))
        X_ = (np.ones(len(x)), dc_l, dv_l, d_dcl, d_dvl, d_dcl*dc_l, d_dvl*dv_l, d_dcl*dv_l, d_dvl*dc_l, dc_l**2, dv_l**2, dc_l*dv_l)

        guesses = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11 = C
        fit_sp = fitting_function(X, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11)
        fit = fitting_function(X_, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11)

    else:
        pass

    resid = fit_sp - taus
    chisq = sum((resid / yerr)**2)
    red_chi = chisq / (len(dels) - npars)
    aic = AIC(npars, red_chi, n=1)
    # aic = AIC(npars, chisq, n=1)
    bic = BIC(npars, len(taus), chisq)
    print(a, chisq, red_chi, aic, bic)
    return a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, fit, cov, C, x_binned



a_list, ctot2_list, err_list = [], [], []
Nfiles = 50
H0 = 100
nbins_x, nbins_y, npars = 10, 10, 6

for j in range(3, Nfiles):
    a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, binned_fit, cov, C_, x_binned = binning(j, path, Lambda, kind, nbins_x, nbins_y, npars)
    a_list.append(a)
    rho_b = 27.755 / a**3
    f1 = (1 / rho_b)
    ctot2 = C_[1]*f1 + C_[2]*f1
    cov[1,1] *= f1**2
    cov[2,2] *= f1**2
    cov[1,2] *= f1**2
    corr = cov[1,2] / np.sqrt(cov[1,1]*cov[2,2])
    err0, err1, err2 = np.sqrt(np.diag(cov))[:3]
    terr = np.sqrt(err1**2 + err2**2 + corr*err1*err2 + corr*err2*err1)
    ctot2_list.append(ctot2)
    err_list.append(terr)
    print('a = ', a, 'ctot2 = ', ctot2, 'err = ', terr)

df = pandas.DataFrame(data=[ctot2_list, a_list, err_list])
file = open("./{}/6_par_binned_ctot2_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'wb')
pickle.dump(df, file)
file.close()

file = open("./{}/6_par_binned_ctot2_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'rb')
read_file = pickle.load(file)
binned_ctot2_list, binned_a_list, binned_err_list = np.array(read_file)
file.close()

# print(binned_err_list)

file = open("./{}/ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'rb')
read_file = pickle.load(file)
a_list, ctot2_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, err4_list = np.array(read_file)
file.close()

# fit_a_list, ctot2_list = [], []
# guesses = 1, 1, 1, 1, 1, 1
# def fitting_function(X, a0, a1, a2, a3, a4, a5):
#     return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4] + a5*X[5]

# guesses = 1, 1, 1
# def fitting_function(X, a0, a1, a2):
#     return a0*X[0] + a1*X[1] + a2*X[2]

# for j in range(3, 50):
#     # a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, binned_fit, cov, C_, x_binned = binning(j, path, Lambda, kind, nbins_x, nbins_y, npars)
#     a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
#     dv_l = -dv_l / (100 / (a**(1/2)))
#     tau_l -= np.mean(tau_l)
#     X = (np.ones(len(x)), dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l)
#     C, cov = curve_fit(fitting_function, X, tau_l, sigma=None, method='lm', absolute_sigma=True)
#     C0, C1, C2, C3, C4, C5 = C
#     # X = (np.ones(dc_l.size), dc_l, dv_l)
#     # C, cov = curve_fit(fitting_function, X, tau_l, sigma=None, method='lm', absolute_sigma=True)
#     # C0, C1, C2 = C
#
#     rho_b = 27.755 / a**3
#     f1 = (1 / rho_b)
#     # f2 = (-100 / (rho_b * np.sqrt(a)))
#     ctot2 = C[1]*f1 + C[2]*f1
#     print(C[1], C[2], ctot2)
#     ctot2_list.append(ctot2)
#     # print('a = ', a, 'ctot2 = ', ctot2)
#     fit_a_list.append(a)
#     print('a = ', a)
#
# ctot2_list = np.array(ctot2_list)
# fit_a_list = np.array(fit_a_list)

N = 50
N2 = 47
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=18, y=1.01)
ax.set_xlabel(r'$a$', fontsize=20)
ax.set_ylabel('$c_{\mathrm{tot}}^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=20)
ctot2_line, = ax.plot(a_list[:N], ctot2_list[:N], c='k', lw=1.5, zorder=1, marker='o') #from tau fit
# ctot2_line, = ax.plot(fit_a_list[:N], ctot2_list[:N], c='k', lw=1.5, zorder=1, marker='o') #from tau fit

ctot2_2_line, = ax.plot(a_list[:N], ctot2_2_list[:N], c='cyan', lw=1.5, marker='*', zorder=2) #M&W
ctot2_3_line, = ax.plot(a_list[:N], ctot2_3_list[:N], c='orange', lw=1.5, marker='v', zorder=3) #B+12
ctot2_4_line, = ax.plot(binned_a_list[:N2], binned_ctot2_list[:N2], c='xkcd:dried blood', lw=1.5, marker='+', zorder=4) #FDE
ctot2_4_err = ax.fill_between(binned_a_list[:N2], binned_ctot2_list[:N2]-binned_err_list[:N2], binned_ctot2_list[:N2]+binned_err_list[:N2], color='darkslategray', alpha=0.5)
plt.legend(handles=[ctot2_line, ctot2_2_line, ctot2_3_line, (ctot2_4_line, ctot2_4_err)], labels=[r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'binned fit'], fontsize=14, bbox_to_anchor=(1,1), framealpha=0.75)

ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.yaxis.set_ticks_position('both')
plt.savefig('../plots/test/new_paper_plots/binned_ctot2/6_par_{}.png'.format(kind), bbox_inches='tight', dpi=150)
plt.close()
# plt.show()


# # a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, binned_fit, cov, C_, x_binned = binning(j, path, Lambda, kind, nbins_x, nbins_y, npars)
#
#
# # guesses = 1, 1, 1
# # def fitting_function(X, a0, a1, a2):
# #     x1, x2 = Xdv_l
# #     return a0 + a1*x1 + a2*x2
# #
# # C, cov = curve_fit(fitting_function, (dc_l, dv_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
# # C0, C1, C2 = C
# # fit = fitting_function((dc_l, dv_l), C0, C1, C2)
#
# # guesses = 1, 1, 1, 1, 1, 1
# # guesses = 0, 0, 0, 0, 0, 0
# #
# # def fitting_function(X, a0, a1, a2, a3, a4, a5):
# #     x1, x2 = X
# #     return a0 + a1*x1 + a2*x2 + a3*(x1**2) + a4*(x2**2) + a5*(x1*x2)
# #
# # C_calc, cov = curve_fit(fitting_function, (dc_l, dv_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
# # C = np.array(C)
#
# # guesses = 1, 1, 1
# #
# # def fitting_function(X, a0, a1, a2):
# #     x1 = X
# #     return a0 + a1*x1 + a2*(x1**2)
# # C, cov = curve_fit(fitting_function, (dc_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
# #
# # fit = C[0] + C[1]*dc_l + C[2]*dc_l**2
#
# guesses = 1, 1, 1
#
# def fitting_function(X, a0, a1, a2):
#     x1, x2 = X
#     return a0 + a1*x1 + a2*(x2)
# C, cov = curve_fit(fitting_function, (dc_l, dv_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
#
# fit = C[0] + C[1]*dc_l + C[2]*dv_l
#
#
# print('C0_binned = ', C_[0], 'C0_fit = ', C[0])
# print('C1_binned = ', C_[1], 'C1_fit = ', C[1])
# print('C2_binned = ', C_[2], 'C2_fit = ', C[2])
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
# plt.plot(x, binned_fit, c='k', ls='dashed', label='binned fit')
# # plt.scatter(x[indices], tau_l[indices], s=20, c='seagreen')
# # plt.scatter(x_binned, tau_l[x_binned], s=20, c='seagreen')
#
# # plt.plot(x, fit3, c='cyan', ls='dotted', label='using derivatives 2')
#
#
# # plt.plot(x, est, c='k', ls='dashed', label='using derivatives')
# plt.legend(fontsize=14, bbox_to_anchor=(1, 1))
# plt.show()
# # plt.savefig('../plots/test/new_paper_plots/fde_vs_fit_one_point/{}_tau_{}.png'.format(kind, j), bbox_inches='tight', dpi=150)
# # plt.close()
