#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from functions import read_sim_data, AIC, BIC, spectral_calc
from scipy.optimize import curve_fit


def calc(j, Lambda, path, mode, kind, n_runs, n_use, folder_name, npars=3):
    a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)

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

    if npars == 3:

        def fitting_function(X, a0, a1, a2):
            x1, x2 = X
            return a0 + a1*x1 + a2*x2

        guesses = 1, 1, 1
        C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
        fit = fitting_function((dc_l, dv_l), C[0], C[1], C[2])

    elif npars == 5:
        def fitting_function(X, a0, a1, a2, a3, a4):
            x1, x2, x3, x4 = X
            return a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4

        guesses = 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp, del_dc_sp, del_v_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
        fit = fitting_function((dc_l, dv_l, del_dc, del_v), C[0], C[1], C[2], C[3], C[4])

    elif npars == 6:
        def fitting_function(X, a0, a1, a2, a3, a4, a5):
            x1, x2 = X
            return a0 + a1*x1 + a2*x2 + a3*x1**2 + a4*x2**2 + a5*x1*x2

        guesses = 1, 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
        fit = fitting_function((dc_l, dv_l), C[0], C[1], C[2], C[3], C[4], C[5])

    else:
        pass

    resid = fit[0::n_ev] - tau_l_sp
    red_chi = sum((resid / yerr_sp)**2) / (n_use - npars)
    AIC_ = AIC(npars, red_chi, n_use)
    BIC_ = BIC(npars, n_use, red_chi)


    return a, x, tau_l, tau_l_0, fit


path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
mode = 1
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
plots_folder = 'test/new_paper_plots/'

j = 23
n_runs = 8
n_use = 10
folder_name = '/new_hier/data_{}/L{}/'.format(kind, int(Lambda/(2*np.pi)))
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
handles = []
fig, ax = plt.subplots()
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=14)
ax.yaxis.set_ticks_position('both')
ax.set_ylabel(r'$J / \langle[\tau]_{\Lambda}\rangle $', fontsize=20)
ax.set_xlabel(r'$x/L$', fontsize=20)
# ax.set_ylim(-0.007, 0.0085)
colours = ['k', 'b']
times = [0, 22]
for num in range(len(times)):
    j = times[num]
    a, x, taus, tau_l, fit = calc(j, Lambda, path, mode, kind, n_runs, n_use, folder_name, npars=3)
    ax.set_title(r'$\Lambda = {}\;k_{{\mathrm{{f}}}}$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=20)
    frac = (tau_l - taus) / taus
    print(np.real(np.fft.fft(frac))[1:5] / frac.size)
    ax.plot(x, frac, c=colours[num], lw=1.5, label='$a={}$'.format(np.round(a,3)))


plt.legend(fontsize=12)
fig.align_labels()
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('../plots/{}/stoch.pdf'.format(plots_folder, j), bbox_inches='tight', dpi=300)
plt.close()
# plt.show()
