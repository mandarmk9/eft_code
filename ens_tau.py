#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from functions import read_sim_data, AIC, BIC, spectral_calc
from scipy.optimize import curve_fit
from tau_fits import tau_calc
from tqdm import tqdm

def calc(j, Lambda, path, mode, kind, n_runs, n_use, folder_name):
    a, x, d1k, dc_l_0, dv_l_0, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)

    H = a**(-1/2)*100
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

    def fitting_function(X, a0, a1, a2):
        x1, x2 = X
        return a0 + a1*x1 + a2*x2

    guesses = 1, 1, 1
    C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    fit = fitting_function((dc_l, dv_l), C[0], C[1], C[2])

    return a, x, dc_l, dv_l, tau_l, dc_l_0, dv_l_0, tau_l_0, fit, C

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
n_runs = 8
n_use = 8
mode = 1
Lambda_int = 3
Lambda = (2*np.pi) * Lambda_int
kinds = ['sharp', 'gaussian']
kinds_txt = ['sharp cutoff', 'Gaussian smoothing']

which = 0
kind = kinds[which]
kind_txt = kinds_txt[which]

# j = 14
folder_name = '/new_hier/data_{}/L{}/'.format(kind, int(Lambda/(2*np.pi)))

for j in tqdm(range(51)):
    a, x, dc_l, dv_l, tau_l, dc_l_0, dv_l_0, tau_l_0, fit, C = calc(j, Lambda, path, mode, kind, n_runs, n_use, folder_name)

    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots()
    ax.set_title(rf'$a = {a}, \Lambda = {Lambda_int}\,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=24)
    ax.set_xlabel(r'$x/L$', fontsize=20)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=18)
    ax.yaxis.set_ticks_position('both')

    ## tau plots
    # ax.set_ylabel(r'$\langle\tau\rangle\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=22)
    # ax.plot(x, tau_l, c='b', lw=1.5, label=r'$\langle \tau\rangle$')
    # ax.plot(x, tau_l_0, c='k', ls='dashed', lw=1.5, label=r'$\tau$: \texttt{sim\_1\_11}')
    # plt.tight_layout()
    # fig.align_labels()
    # plt.savefig(f'../plots/paper_plots_final/tau_{kind}/ens_tau_{j}.png', bbox_inches='tight', dpi=300)
    # plt.close()

    # dc_l plots
    ax.set_ylabel(r'$\delta_{\ell}$', fontsize=22)
    ax.plot(x, dc_l, c='b', lw=1.5, label=r'averaged')
    ax.plot(x, dc_l_0, c='k', ls='dashed', lw=1.5, label=r'\texttt{sim\_1\_11}')
    plt.tight_layout()
    fig.align_labels()
    plt.legend(fontsize=18, bbox_to_anchor=(1, 1.025))
    plt.savefig(f'../plots/paper_plots_final/dc_{kind}/ens_dc_{j}.png', bbox_inches='tight', dpi=300)
    plt.close()

    ## dv_l plots
    # ax.set_ylabel(r'$\theta_{\ell}$', fontsize=22)
    # ax.plot(x, dv_l, c='b', lw=1.5, label=r'averaged')
    # ax.plot(x, dv_l_0, c='k', ls='dashed', lw=1.5, label=r'\texttt{sim\_1\_11}')
    # plt.tight_layout()
    # fig.align_labels()
    # plt.savefig(f'../plots/paper_plots_final/dv_{kind}/ens_dv_{j}.png', bbox_inches='tight', dpi=300)
    # plt.close()


    # plt.show()
