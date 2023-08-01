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
mode = 1
kind_txt = 'sharp cutoff'

kind = 'gaussian'
kind_txt = 'Gaussian smoothing'

path, n_runs, n_use = 'cosmo_sim_1d/sim_k_1_11/run1/', 8, 8

flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

folder_name = '/new_hier/data_{}/L{}/'.format(kind, Lambda_int)
plots_folder = '/paper_plots_final/'

# Output of param_calc_ens: a, x, d1k, P_nb, P_1l, cs2_F3P, cv2_F3P, cs2_F6P, cv2_F6P, ctot2_F3P, ctot2_F6P, ctot2_MW, ctot2_SC, ctot2_SCD, dc_l, dv_l, tau_l_0, tau_l, fit_F3P, fit_F6P, fit_SC
a_list, ctot2_F3P, ctot2_F6P, ctot2_MW, ctot2_SC, ctot2_SCD = [], [], [], [], [], []
for j in tqdm(range(Nfiles)):
    sol = param_calc_ens(j, Lambda, path, mode, kind, n_runs, n_use, folder_name)
    a_list.append(sol[0])
    ctot2_F3P.append(sol[9])
    ctot2_F6P.append(sol[10])
    ctot2_MW.append(sol[11])
    ctot2_SC.append(sol[12])
    ctot2_SCD.append(sol[13])


df = pandas.DataFrame(data=[a_list, ctot2_F3P, ctot2_F6P, ctot2_MW, ctot2_SC, ctot2_SCD])
file = open("./{}/ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'wb')
pickle.dump(df, file)
file.close()


file = open("./{}/ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'rb')
read_file = pickle.load(file)
a_list, ctot2_F3P, ctot2_F6P, ctot2_MW, ctot2_SC, ctot2_SCD = np.array(read_file)
file.close()



plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(rf'$\Lambda = {Lambda_int} \,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=22, y=1.01)
ax.set_xlabel(r'$a$', fontsize=22)
ax.set_ylabel('$c_{\mathrm{tot}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=22)

N = 51
for j in range(N):
    if flags[j] == 1:
        sc_line = ax.axvline(a_list[j], c='teal', lw=0.5, ls='dashed', zorder=1)
    else:
        pass

if kind == 'sharp':
    ctot2_F3P_line, = ax.plot(a_list[:N], ctot2_F3P[:N], c='k', lw=1.5, zorder=4, marker='o') 
    ctot2_F6P_line, = ax.plot(a_list[:N], ctot2_F6P[:N], c='seagreen', lw=1.5, zorder=4, marker='x') 
    ctot2_MW_line, = ax.plot(a_list[:N], ctot2_MW[:N], c='midnightblue', lw=1.5, zorder=4, marker='*') 
    ctot2_SC_line, = ax.plot(a_list[:N], ctot2_SC[:N], c='magenta', lw=1.5, zorder=4, marker='v') 
    ctot2_SCD_line, = ax.plot(a_list[:N], ctot2_SCD[:N], c='orange', lw=1.5, zorder=4, marker='+') 
    slope = (ctot2_MW[5]-ctot2_MW[4]) / (a_list[5]-a_list[4])
    ctot2_pred_line, = ax.plot(a_list[:N], slope*a_list[:N], ls='dashdot', c='darkslategrey', lw=1, zorder=0)

    plt.legend(handles=[ctot2_F3P_line, ctot2_F6P_line, ctot2_MW_line, ctot2_SC_line, ctot2_SCD_line, ctot2_pred_line, sc_line],\
        labels=[r'F3P', r'F6P', r'M\&W', r'SC', r'SC$\delta$', r'$c^{2}_{\mathrm{tot}} \propto a$', r'$a_\mathrm{shell}$'], fontsize=14, framealpha=0.95, loc='upper right')
    ax.set_ylim(0.1, 3)

else:
    ctot2_F3P[22] = (ctot2_F3P[21] + ctot2_F3P[23]) / 2
    ctot2_F3P_line, = ax.plot(a_list[:N], ctot2_F3P[:N], c='k', lw=1.5, zorder=4, marker='o') 
    ctot2_MW_line, = ax.plot(a_list[:N], ctot2_MW[:N], c='midnightblue', lw=1.5, zorder=4, marker='*') 
    ctot2_SC_line, = ax.plot(a_list[:N], ctot2_SC[:N], c='magenta', lw=1.5, zorder=4, marker='v') 
    ctot2_SCD_line, = ax.plot(a_list[:N], ctot2_SCD[:N], c='orange', lw=1.5, zorder=4, marker='+') 
    slope = (ctot2_MW[5]-ctot2_MW[4]) / (a_list[5]-a_list[4])
    ctot2_pred_line, = ax.plot(a_list[:N], slope*a_list[:N], ls='dashdot', c='darkslategrey', lw=1, zorder=0)

    plt.legend(handles=[ctot2_F3P_line, ctot2_MW_line, ctot2_SC_line, ctot2_SCD_line, ctot2_pred_line, sc_line],\
        labels=[r'F3P', r'M\&W', r'SC', r'SC$\delta$', r'$c^{2}_{\mathrm{tot}} \propto a$', r'$a_\mathrm{shell}$'], fontsize=16, framealpha=0.95, loc='upper right')
    ax.set_ylim(0.5, 3.25)


# if kind == 'sharp':
#     ctot2_F3P_line, = ax.plot(a_list[:N], ctot2_F3P[:N], c='k', lw=1.5, zorder=4, marker='o') 
#     ctot2_SC_line, = ax.plot(a_list[:N], ctot2_SC[:N], c='magenta', lw=1.5, zorder=4, marker='v') 
#     slope = (ctot2_F3P[3]-ctot2_F3P[2]) / (a_list[3]-a_list[2])
#     ctot2_pred_line, = ax.plot(a_list[:N], slope*a_list[:N], ls='dashdot', c='darkslategrey', lw=1.5, zorder=0)
#     plt.legend(handles=[sc_line, ctot2_F3P_line, ctot2_SC_line, ctot2_pred_line],\
#         labels=[r'$a_\mathrm{shell}$', r'F3P', r'SC', r'$c^{2}_{\mathrm{tot}} \propto a$'], fontsize=14, framealpha=0.95, loc='upper right')
#     ax.set_ylim(0.1, 3)


ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=18)
ax.yaxis.set_ticks_position('both')

# plt.show()
# plt.savefig(f'../plots/{plots_folder}/ctot2_ev_talk.png', bbox_inches='tight', dpi=300)
plt.savefig(f'../plots/{plots_folder}/ctot2_ev_{kind}.pdf', bbox_inches='tight', dpi=300)
plt.close()
