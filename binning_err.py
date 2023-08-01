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
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
j = 41
nbins_x, nbins_y, npars = 25, 25, 6


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

    if npars == 3: #first-order
        def fitting_function(X, a0, a1, a2):
            return a0*X[0] + a1*X[1] + a2*X[2]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes))
        X_ = (np.ones(len(x)), dc_l, dv_l)
        guesses = 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2 = C
        fit_sp = fitting_function(X, C0, C1, C2)
        fit = fitting_function(X_, C0, C1, C2)

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
    # aic = AIC(npars, red_chi, n=1)
    aic = AIC(npars, chisq, n=1)
    bic = BIC(npars, len(taus), chisq)
    return a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, fit, cov, C, x_binned, red_chi

a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, binned_fit, cov, C_, x_binned, red_chi = binning(j, path, Lambda, kind, nbins_x, nbins_y, npars)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=18, y=1.01)
ax.set_xlabel(r'$\theta_{l}$', fontsize=20)
ax.set_ylabel(r'$\tau_{l}$', fontsize=20)


taus = np.array(taus)
yerr = np.array(yerr)

measured, = ax.plot(dv_l, tau_l, c='b', lw=2)
binned_fit, = ax.plot(dv_l, binned_fit, c='k', lw=2, ls='dashdot')

# binned_fit = ax.scatter(thes, taus, c='r', lw=2, s=20, label='binned fit')
bins = ax.errorbar(thes, fit_sp, c='r', yerr=yerr, fmt='o')
import matplotlib.patches as mpatches
r = mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor='None', visible=False)
# err = ax.fill_between(thes, taus-yerr, taus+yerr, color='darkslategray', alpha=0.5)
plt.legend(handles=[measured, binned_fit, bins, r], labels=[r'measured', r'binned fit', r'bins', r'reduced $\chi^{{2}}$ = {}'.format(np.round(red_chi, 3))], fontsize=14, loc=1, bbox_to_anchor=(1,1), framealpha=0.75)

ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.yaxis.set_ticks_position('both')
# plt.savefig('../plots/test/new_paper_plots/test_{}.png'.format(kind), bbox_inches='tight', dpi=150)
# plt.close()
plt.show()
