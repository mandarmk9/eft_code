#!/usr/bin/env python3

#import libraries
import os
import pickle
import pandas
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from functions import read_sim_data, AIC, sub_find, binning, spectral_calc, param_calc_ens, smoothing
from tqdm import tqdm
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"



A = [-0.05, 1, -0.5, 11]
Nfiles = 51
Lambda_int = 3
Lambda = Lambda_int * (2*np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'

# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

path, n_runs = 'cosmo_sim_1d/sim_k_1_11/run1/', 8
# path, n_runs = 'cosmo_sim_1d/multi_k_sim/run1/', 8
# path, n_runs = 'cosmo_sim_1d/sim_3_15/run1/', 8
# path, n_runs = 'cosmo_sim_1d/multi_sim_3_15_33/run1/', 8

# path, n_runs = 'cosmo_sim_1d/final_sim_k_1_11/run1/', 16
# path, n_runs = 'cosmo_sim_1d/new_sim_k_1_11/run1/', 24
n_use = 8
mode = 1

flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

f1 = 'curve_fit'
# f1 = ''
nbins_x = 10
nbins_y = 10
npars = 3

per = 43
# fde_method = 'algorithm'
fde_method = 'percentile'

N = 51
# folder_name = '/test_hier/'
folder_name = '/new_hier/data_{}/L{}/'.format(kind, Lambda_int)
# plots_folder = '/test/multi_sim_3_15_33/'
plots_folder = '/paper_plots_final/'

# folder_name = '/new_hier/' #N_bins = 10000
# folder_name = '/data_even_coarser/' #N_bins = 1000
# folder_name = '/data_coarse/' #N_bins = 100
zero = 0
a_list, ctot2_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, ctot2_5_list, ctot2_6_list, err1_list, err2_list, chi_list, t_err, a_list4, err4_list = [], [], [], [], [], [], [], [], [], [], [], [], []
ens = True
c1_list, c2_list = [], []
for j in tqdm(range(zero, N)):
    # if j == 24:
    #     pass
    # else:
    sol = param_calc_ens(j, Lambda, path, A, mode, kind, n_runs, n_use, fitting_method=f1, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars, fde_method=fde_method, per=per, folder_name=folder_name, ens=ens)
    a_list.append(sol[0])
    ctot2_list.append(sol[2])
    ctot2_2_list.append(sol[3])
    ctot2_3_list.append(sol[4])
    ctot2_4_list.append(sol[21])
    ctot2_5_list.append(sol[24])
    ctot2_6_list.append(sol[25])
    err4_list.append(sol[-1])
    # print('err', sol[-1])
    err1_list.append(sol[6])
    err2_list.append(sol[7])
    chi_list.append(sol[10])
    t_err.append(sol[-9])
    # # print('a = ', sol[0], 'ctot2: ', sol[2], ',', sol[3], ',', sol[-2])
    # tau_l = sol[12]
    # dv_l = sol[14]
    # dc_l = sol[-1]
    # rho_b = 27.755 / sol[0]**3
    # tD = np.mean(tau_l*dc_l) / rho_b
    # tT = np.mean(tau_l*dv_l) / rho_b
    # DT = np.mean(dc_l*dv_l)
    # TT = np.mean(dv_l*dv_l)
    # DD = np.mean(dc_l*dc_l)
    # rhs = (tD / DT) - (tT / TT)
    # lhs = (DD / DT) - (DT / TT)
    # cs2 = rhs / lhs
    # cv2 = (DD*cs2 - tD) / DT
    # c1_list.append(cs2+cv2)
    # c2_list.append(tD/ DD)


ctot2_4_list = np.array(ctot2_4_list)
err4_list = np.array(err4_list)
a_list = np.array(a_list)

# df = pandas.DataFrame(data=[a_list, ctot2_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, ctot2_5_list, ctot2_6_list])
# file = open("./{}/ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'wb')
# pickle.dump(df, file)
# file.close()


# file = open("./{}/ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'rb')
# read_file = pickle.load(file)
# a_list, ctot2_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, ctot2_5_list, ctot2_6_list = np.array(read_file)
# file.close()



# ctot2_4, a_list4, err4 = [], [], []
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=18, y=1.01)
# ax.set_title(r'$\Lambda = {} \,k_{{\mathrm{{f}}}}$ ({}), $N_{{\mathrm{{bins}}}} = {}$'.format(int(Lambda/(2*np.pi)), kind_txt, 10000), fontsize=18, y=1.01)
# ax.set_title(r'$\Lambda = {} \,k_{{\mathrm{{f}}}}$ ({})'.format(Lambda_int, kind_txt), fontsize=18, y=1.01)
ax.set_title(rf'$\Lambda = {Lambda_int} \,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=18, y=1.01)


ax.set_xlabel(r'$a$', fontsize=20)
# ax.set_ylabel('$c_{\mathrm{tot}}^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=20)
ax.set_ylabel('$c_{\mathrm{tot}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=20)


# print(ctot2_4_list)
for j in range(N):
    # # if j == 0:
    # #     cond = False
    # #     distance = 0
    # # else:
    # #     distance = np.abs(ctot2_4_list[j] - ctot2_4_list[j-1])
    # #     cond = distance < 0.7
    # cond = True
    # if cond:
    #     print(j, a_list[j], ctot2_4_list[j], ctot2_3_list[j], ctot2_list[j])#, distance)
    #     ctot2_4.append(ctot2_4_list[j])
    #     a_list4.append(a_list[j])
    #     err4.append(err4_list[j])
    if flags[j] == 1:
        sc_line = ax.axvline(a_list[j], c='teal', lw=0.5, ls='dashed', zorder=1)

    else:
        # sc_line = ax.axvline(0.5, c='teal', lw=1, zorder=1)
        # print('boo')
        pass

ctot2_list = np.array(ctot2_list)

# k = np.fft.ifftshift(2.0 * np.pi * np.arange(-a_list4.size/2, a_list4.size/2))
# L = 100
# ctot2_list = smoothing(ctot2_list, k, L, kind='gaussian')
# ctot2_2_list = smoothing(ctot2_2_list, k, L, kind='gaussian')
# ctot2_3_list = smoothing(ctot2_3_list, k, L, kind='gaussian')
# ctot2_4_list = smoothing(ctot2_4_list, k, L, kind='gaussian')
# err4_list = smoothing(err4_list, k, L, kind='gaussian')


# print(np.array(ctot2_2_list[:N]).max())

# ax.fill_between(a_list, ctot2_list-t_err, ctot2_list+t_err, color='darkslategray', alpha=0.55, zorder=2)
ctot2_line, = ax.plot(a_list[:N], ctot2_list[:N], c='k', lw=1.5, zorder=4, marker='o') #from tau fit
ctot2_4_line, = ax.plot(a_list[:N], ctot2_4_list[:N], c='seagreen', lw=1.5, marker='x') #F6P
# ctot2_line, = ax.plot(a_3list[:N], ctot2_list[:N], c='k', lw=1.5, zorder=4, marker='o') #from tau fit

# ctot2_2_line, = ax.plot(a_list[:N], ctot2_2_list[:N], c='cyan', lw=1.5, marker='*') #M&W
# ctot2_5_line, = ax.plot(a_list[:N], ctot2_5_list[:N], c='magenta', lw=1.5, marker='v') #SC
# # ctot2_3_line, = ax.plot(a_list[:N], ctot2_3_list[:N], c='orange', lw=1.5, marker='v', zorder=3) #B+12

# ctot2_6_line, = ax.plot(a_list[:N], ctot2_6_list[:N], c='orange', lw=1.5, marker='+', zorder=1) #SC\delta
# # ctot2_4_line, = ax.plot(a_list[:N], ctot2_4_list[:N], c='lightseagreen', lw=1.5, marker='+', zorder=1) #DDE

# # ctot2_4_line, = ax.plot(a_list[:N], ctot2_4_list[:N], c='xkcd:dried blood', lw=1.5, marker='+', zorder=1) #DDE
# # ctot2_4_line, = ax.plot(a_list[:N], ctot2_6par[:N], c='k', ls='dashed', lw=1.5, marker='o', zorder=4) #B+12

# # ctot2_4_err = ax.fill_between(a_list[:N], ctot2_4_list[:N]-err4_list[:N], ctot2_4_list[:N]+err4_list[:N], color='darkslategray', alpha=0.55, zorder=2)
# # ctot2_4_err = ax.fill_between(a_3list[:N], ctot2_list[:N]-err_list[:N], ctot2_list[:N]+err_list[:N], color='darkslategray', alpha=0.55, zorder=2)

# slope = (ctot2_2_list[5]-ctot2_2_list[4]) / (a_list[5]-a_list[4])
# pred_line, = ax.plot(a_list[:N], slope*a_list[:N], ls='dashdot', c='green', lw=1, zorder=0) #DDE

# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line, ctot2_4_line, pred_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'Spatial Corr', r'Spatial Corr from $\delta_{\ell}$', r'$c^{2}_{\mathrm{tot}} \propto a$'], fontsize=14, framealpha=0.95)

# plt.legend(handles=[sc_line, ctot2_line, ctot2_4_line, ctot2_2_line, ctot2_5_line, ctot2_6_line, pred_line],\
#      labels=[r'$a_\mathrm{sc}$', r'F3P', r'F6P', r'M\&W', r'SC', r'SC$\delta$', r'$c^{2}_{\mathrm{tot}} \propto a$'], fontsize=14, framealpha=0.95)


# print(slope)
# plt.legend(handles=[ctot2_line, ctot2_2_line, ctot2_3_line, pred_line], labels=[r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'$c^{2}_{\mathrm{tot}} \propto a$'], fontsize=14, framealpha=1)

# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line, (ctot2_4_line, ctot2_4_err)], labels=[r'$a_\mathrm{sc}$', r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'DDE'.format(per)], fontsize=14, framealpha=1, loc=2)
# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line, pred_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'$c^{2}_{\mathrm{tot}} \propto a$'], fontsize=14, loc=2, framealpha=0.75)
# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, pred_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$c^{2}_{\mathrm{tot}} \propto a$'], fontsize=14, loc=2, framealpha=0.75)

# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line, pred_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'$c^{2}_{\mathrm{tot}} \propto a$'], fontsize=14, loc=2, framealpha=0.75)
# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line, ctot2_4_line, pred_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$', r'DDE', r'$c^{2}_{\mathrm{tot}} \propto a$'], fontsize=14, loc=2, framealpha=0.75)


# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, loc=1, framealpha=1)
# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, loc=1, framealpha=1)

# plt.legend(handles=[sc_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, loc=1, framealpha=1)
# plt.legend(handles=[sc_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'M\&W', r'$\mathrm{C^{+12}}$'], fontsize=14, loc=1, framealpha=1)

# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W'], fontsize=14, loc=1, framealpha=1)
# plt.legend(handles=[sc_line], labels=[r'$a_\mathrm{sc}$'], fontsize=14)
# ax.set_ylim(-0.2, 2.05)

ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.yaxis.set_ticks_position('both')
# if kind == 'sharp':
#     ax.set_ylim(0.1, 3)
# else:
#     ax.set_ylim(0.5, 3.25)

# plt.savefig('../plots/test/new_paper_plots/ctot2_ev_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# plt.savefig('../plots/test/new_paper_plots/multi_k_ctot2_{}.png'.format(kind), bbox_inches='tight', dpi=150)
# plt.close()
# # # # plt.savefig('../plots/test/new_paper_plots/ctot2_ev_{}.png'.format(kind), bbox_inches='tight', dpi=150)
# # # # # plt.savefig('../plots/test/new_paper_plots/fde_per_test_median/ctot2_ev_{}_{}.png'.format(kind, int(per)), bbox_inches='tight', dpi=150)
# # # #
# # # # # plt.savefig('../plots/test/new_paper_plots/fit_comps/curve_fit/{}_{}_ctot2_ev_{}.png'.format(n_runs, data_cov, kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# # # # # plt.savefig('../plots/test/new_paper_plots/fit_comps/{}_ctot2_ev_{}.png'.format(nbins_x, kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# # # # # plt.savefig('../plots/test/new_paper_plots/fit_comps/{}_{}_ctot2_ev_{}_median_w_binned_fit.png'.format(n_runs, data_cov, kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# plt.show()
# # print(ctot2_list[N-1], ctot2_2_list[N-1])
plt.savefig(f'../plots/{plots_folder}/ctot2_ev_{kind}.pdf', bbox_inches='tight', dpi=300)
plt.close()
