#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from functions import read_sim_data, spectral_calc
from scipy.optimize import curve_fit
from tqdm import tqdm

def J_calc(file_num, Lambda, path, mode, kind, folder_name, n_use=10):
    a, x, d1k, dc_l_0, dv_l_0, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, file_num, folder_name)
    H = a**(-1/2)*100
    dv_l_0 = -dv_l_0 / (H)

    taus, dels, thes = [], [], []
    taus.append(tau_l_0)
    dels.append(dc_l_0)
    thes.append(dv_l_0)

    for run in range(1, n_runs+1):
        path = path[:-2] + '{}/'.format(run)
        sol = read_sim_data(path, Lambda, kind, file_num, folder_name)
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
    n_ev = x.size // n_use
    dc_l_sp = dc_l[0::n_ev]
    dv_l_sp = dv_l[0::n_ev]
    tau_l_sp = tau_l[0::n_ev]
    yerr_sp = yerr[0::n_ev]

    guesses = 1, 1, 1
    def fitting_function(X, a0, a1, a2):
        x1, x2 = X
        return a0 + a1*x1 + a2*x2
    C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    fit = fitting_function((dc_l, dv_l), C[0], C[1], C[2])

    J_fit = tau_l_0 - fit


    # spatial correlations
    tD = np.mean(tau_l_0*dc_l_0) / rho_b
    tT = np.mean(-tau_l_0*dv_l_0) / rho_b
    DT = np.mean(-dc_l_0*dv_l_0)
    TT = np.mean(-dv_l_0*-dv_l_0)
    DD = np.mean(dc_l_0*dc_l_0)
    rhs = (tD / DT) - (tT / TT)
    lhs = (DD / DT) - (DT / TT)
    cs2 = rhs / lhs
    cv2 = (DD*cs2 - tD) / DT
    ctot2_5 = (cs2+cv2)

    J_corr = tau_l_0 - (np.mean(tau_l_0) + cs2*rho_b*dc_l_0 + cv2*rho_b*dv_l_0)
    return a, x, tau_l_0, tau_l, J_fit, J_corr


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

folder_name = '/new_hier/data_{}/L{}/'.format(kind, Lambda_int)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})

# j = 50
for j in tqdm(range(0, 51)):
    a, x, tau_l_0, tau_l, J, J_corr = J_calc(j, Lambda, path, mode, kind, folder_name)
    fig, ax = plt.subplots()
    ax.set_title(rf"$a = {a},\,\Lambda = {Lambda_int}\,k_{{\mathrm{{f}}}}$ ({kind_txt})", fontsize=22)
    ax.plot(x, J / tau_l_0 , c='k', lw=1.5, label=r'F3P')
    ax.plot(x, J_corr / tau_l_0, c='r', ls='dashed', lw=1.5, label=r'SC')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=16)
    ax.yaxis.set_ticks_position('both')
    ax.set_xlabel(r'$x/L$', fontsize=20)
    ax.set_ylabel(r'$J / \langle\tau\rangle$', fontsize=20)
    ax.legend(fontsize=16)
    # ax.set_ylim(-0.82, 0.5) #, bbox_to_anchor=(1.35, 1.025))
    plt.savefig(f'../plots/paper_plots_final/stochastic_terms/{kind}_{j}.png', bbox_inches='tight', dpi=300)
    plt.close()




# plt.show()




# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# fig.suptitle(rf"$\Lambda = {Lambda_int}\,k_{{\mathrm{{f}}}}$ ({kind_txt})", fontsize=22)

# # # a = 0.5
# # a, x, tau_l, J = stoch_calc(0, Lambda, path, mode, kind, n_runs, n_use, folder_name)
# # ax.plot(x, J / tau_l, c='k', ls='solid', lw=1.5, label=rf'$a = {a}$')

# # # a = 2.04
# # a, x, tau_l, J = stoch_calc(14, Lambda, path, mode, kind, n_runs, n_use, folder_name)
# # ax.plot(x, J / tau_l, c='r', ls='dashdot', lw=1.5, label=rf'$a = {a}$')

# # a = 0.5
# a, x, tau_l_0, tau_l, J, J_corr = J_calc(0, Lambda, path, mode, kind, folder_name)
# ax[0].set_title(rf'$a = {a}$', fontsize=22)
# ax[0].plot(x, J / tau_l, c='k', lw=1.5, label=r'from fit to $\langle\tau\rangle$')
# ax[0].plot(x, J_corr / tau_l_0, c='r', ls='dashed', lw=1.5, label=r'Spatial Corr')

# # a = 3.03
# a, x, tau_l_0, tau_l, J, J_corr = J_calc(22, Lambda, path, mode, kind, folder_name)
# ax[1].set_title(rf'$a = {a}$', fontsize=22)
# ax[1].plot(x, J / tau_l, c='k', lw=1.5, label=r'from fit to $\langle\tau\rangle$')
# ax[1].plot(x, J_corr / tau_l, c='r', ls='dashed', lw=1.5, label=r'Spatial Corr')


# plt.subplots_adjust(wspace=0.05)
# ax[1].tick_params(labelleft=False, labelright=True)

# for i in range(2):
#     ax[i].minorticks_on()
#     ax[i].tick_params(axis='both', which='both', direction='in', labelsize=16)
#     ax[i].yaxis.set_ticks_position('both')
#     ax[i].set_xlabel(r'$x/L$', fontsize=20)
#     ax[i].set_ylabel(r'$J / \langle[\tau]_{\Lambda}\rangle$', fontsize=20)

# ax[1].yaxis.set_label_position('right')

# ax[0].legend(fontsize=16)#, bbox_to_anchor=(1.35, 1.025))
# plt.show()
# # plt.savefig('../plots/paper_plots_final/stoch.pdf'.format(kind), bbox_inches='tight', dpi=300)
# # plt.close()
