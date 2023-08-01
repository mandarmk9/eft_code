#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from functions import read_sim_data, AIC, BIC, spectral_calc
from scipy.optimize import curve_fit
from tau_fits import tau_calc

def calc(j, Lambda, path, mode, kind, n_runs, n_use, folder_name):
    a, x, d1k, dc_l_0, dv_l_0, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)

    # H = a**(-1/2)*100
    # dv_l = -dv_l / (H)
    # taus = []
    # taus.append(tau_l_0)
    # for run in range(1, n_runs+1):
    #     path = path[:-2] + '{}/'.format(run)
    #     sol = read_sim_data(path, Lambda, kind, j, folder_name)
    #     taus.append(sol[-3])

    # Nt = len(taus)

    # tau_l = sum(np.array(taus)) / Nt

    H0 = 100
    H = a**(-1/2)*H0
    dv_l_0 = -dv_l_0 / (H)

    taus, dels, thes = [], [], []
    taus.append(tau_l_0)
    dels.append(dc_l_0)
    thes.append(dv_l_0)

    for run in range(1, n_runs+1):
        path = path[:-2] + '{}/'.format(run)
        sol = read_sim_data(path, Lambda, kind, j, folder_name)
        dels.append(sol[3])
        thes.append(-sol[4] / H)
        taus.append(sol[5])


    Nt = len(taus)

    tau_l = sum(np.array(taus)) / Nt
    dc_l = sum(np.array(dels)) / Nt
    dv_l = sum(np.array(thes)) / Nt


    rho_0 = 27.755
    rho_b = rho_0 / a**3

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

    def fitting_function(X, a0, a1, a2):
        x1, x2 = X
        return a0 + a1*x1 + a2*x2

    guesses = 1, 1, 1
    C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    fit_3par = fitting_function((dc_l, dv_l), C[0], C[1], C[2])
    resid = fit_3par[0::n_ev] - tau_l_sp
    red_chi_3par = sum((resid / yerr_sp)**2) / (n_use - 3)
    AIC_3 = AIC(3, red_chi_3par, 1)
    BIC_3 = BIC(3, n_use, red_chi_3par)

    def fitting_function(X, a0, a1, a2, a3, a4):
        x1, x2, x3, x4 = X
        return a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4

    C_3par = C #/ rho_b
    cov_3par = cov
    print(C[1], C[2])

    guesses = 1, 1, 1, 1, 1
    C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp, del_dc_sp, del_v_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    fit_5par = fitting_function((dc_l, dv_l, del_dc, del_v), C[0], C[1], C[2], C[3], C[4])
    resid = fit_5par[0::n_ev] - tau_l_sp
    red_chi_5par = sum((resid / yerr_sp)**2) / (n_use - 5)
    AIC_5 = AIC(5, red_chi_5par, 1)
    BIC_5 = BIC(5, n_use, red_chi_3par)

    def fitting_function(X, a0, a1, a2, a3, a4, a5):
        x1, x2 = X
        return a0 + a1*x1 + a2*x2 + a3*x1**2 + a4*x2**2 + a5*x1*x2


    guesses = 1, 1, 1, 1, 1, 1
    C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    fit_6par = fitting_function((dc_l, dv_l), C[0], C[1], C[2], C[3], C[4], C[5])
    resid = fit_6par[0::n_ev] - tau_l_sp
    red_chi_6par = sum((resid / yerr_sp)**2) / (n_use - 6)
    AIC_6 = AIC(6, red_chi_6par, 1)
    BIC_6 = BIC(6, n_use, red_chi_3par)
    C_6par = C #/ rho_b

    return a, x, tau_l, tau_l_0, fit_3par, fit_5par, fit_6par, red_chi_3par, red_chi_5par, red_chi_6par, AIC_3, AIC_5, AIC_6, BIC_3, BIC_5, BIC_6, dc_l, dv_l, C_3par, C_6par, cov_3par

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


file_nums = [11, 23, 50]
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})

# a, x, tau_l, tau_l_0, fit_3par, fit_5par, fit_6par, red_chi_3par, red_chi_5par, red_chi_6par, AIC_3, AIC_5, AIC_6, BIC_3, BIC_5, BIC_6, dc_l, dv_l, C, C6, cov = calc(22, Lambda, path, mode, kind, n_runs, n_use, folder_name)

# print(cov)
# corr = np.zeros(cov.shape)
# for i in range(3):
#     corr[i,:] = [cov[i,j] / np.sqrt(cov[i,i]*cov[j,j]) for j in range(3)]


# print(corr)
fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=False, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1]})

# fig.suptitle(r'$\Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=20)
fig.suptitle(r'$\Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=24)

ax[0].set_ylabel(r'$\langle\tau\rangle\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=22)
# ax[2].set_ylabel(r'$\langle\tau\rangle\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=22)

# ax[0].set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
# ax[2].set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
# ax[2].yaxis.set_label_position('right')
# ax[2].tick_params(labelleft=False, labelright=True)
# ax[1].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
ax[1].set_xlabel(r'$x/L$', fontsize=20)

for j in range(3):
    file_num = file_nums[j]
    a, x, tau_l, tau_l_0, fit_3par, fit_5par, fit_6par, red_chi_3par, red_chi_5par, red_chi_6par, AIC_3, AIC_5, AIC_6, BIC_3, BIC_5, BIC_6, dc_l, dv_l, C, C6, cov = calc(file_num, Lambda, path, mode, kind, n_runs, n_use, folder_name)
    # print(AIC_3, AIC_5, AIC_6)
    AIC_min = min(AIC_3, AIC_5, AIC_6)
    AIC_3 = np.exp((AIC_min - AIC_3) / 2)
    AIC_5 = np.exp((AIC_min - AIC_5) / 2)
    AIC_6 = np.exp((AIC_min - AIC_6) / 2)
    ax[j].set_title(r'$a = {}$'.format(np.round(a, 3)), x=0.15, y=0.9, fontsize=20)

    ax[j].plot(x, tau_l, c='b', lw=1.5, label=r'$\langle \tau\rangle$')
    # ax[j].plot(x, fit_3par, c='k', lw=1.5, ls='dashed', label=r'fit to $\langle \tau\rangle, N_{{\mathrm{{p}}}} = 3$')
    # ax[j].plot(x, fit_6par, c='r', lw=1.5, ls='dashed', label=r'fit to $\langle \tau\rangle, N_{{\mathrm{{p}}}} = 6$')
    ax[j].plot(x, fit_3par, c='k', lw=1.5, ls='dashed', label=r'$\tau_{\mathrm{fit}}$ (F3P)')
    ax[j].plot(x, fit_6par, c='r', lw=1.5, ls='dashed', label=r'$\tau_{\mathrm{fit}}$ (F6P)')

    ax[j].minorticks_on()
    ax[j].tick_params(axis='both', which='both', direction='in', labelsize=18)
    ax[j].yaxis.set_ticks_position('both')
    # ax[j].plot(x, dc_l, c='g', ls='dashed', lw=2)
    # ax[j].plot(x, dv_l, c='cyan', ls='dashed', lw=2)
    # ax[j].plot(x, dc_l - dv_l*0.99, c='g', ls='dashed', lw=2)
    # print(C[1], C[2])
    # ax[j].plot(x, C6[0] + dc_l*C6[1] + C6[2]*dv_l + C6[3]*dc_l**2 + C6[4]*dv_l**2 + C6[5]*dc_l*dc_l, c='g', ls='dashed', lw=2)


plt.legend(fontsize=17, bbox_to_anchor=(1.025, 1.17), ncol=3, columnspacing=0.65)
fig.align_labels()
plt.subplots_adjust(wspace=0.20)
# plt.show()
plt.savefig('../plots/paper_plots_final/tau_fits_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
plt.close()
