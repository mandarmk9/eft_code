#!/usr/bin/env python3
import h5py
import pickle
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from functions import dc_in_finder, smoothing, dn, param_calc_ens, read_sim_data
from tqdm import tqdm

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
A = []
mode = 1
kind = 'sharp'
kind_txt = 'sharp cutoff'

# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

n_runs = 8
n_use = n_runs-1
j = 22
fm = '' #fitting method
nbins_x, nbins_y, npars = 10, 10, 3

def P13_finder(path, j, Lambdas, kind, mode):
    Nx = 2048
    L = 1.0
    dx = L/Nx
    x = np.arange(0, L, dx)
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    a_list, P11_list, P13_list = [], [], []
    a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
    dc_in = dc_in_finder(path, x, interp=True)[0]
    Nx = dc_in.size
    for Lambda in Lambdas:
        # print('Lambda = ', Lambda)
        Lambda *= (2 * np.pi)
        dc_in = smoothing(dc_in, k, Lambda, kind)
        F = dn(3, L, dc_in)
        d1k = (np.fft.fft(F[0]) / Nx)
        d2k = (np.fft.fft(F[1]) / Nx)
        d3k = (np.fft.fft(F[2]) / Nx)
        P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
        # P13 += ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k))) * (a**3)
        P11 = (d1k * np.conj(d1k)) * (a**2)
        P13_list.append(np.real(P13)[mode])
        P11_list.append(np.real(P11)[mode])
    return np.array(P13_list), np.array(P11_list)


def ctot_finder(Lambdas, path, j, A, mode, kind, n_runs, n_use):
    ctot2_list, ctot2_2_list, ctot2_3_list = [], [], []
    for Lambda in Lambdas:
        print('Lambda = ', Lambda)
        folder_name = '/new_hier/data_{}/L{}'.format(kind, Lambda)
        Lambda *= (2 * np.pi)
        sol = param_calc_ens(j, Lambda, path, A, mode, kind, fitting_method=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars, folder_name=folder_name)
        ctot2_list.append(sol[2])
        ctot2_2_list.append(sol[3])
        ctot2_3_list.append(sol[4])
    a = sol[0]
    return a, np.array(ctot2_list), np.array(ctot2_2_list), np.array(ctot2_3_list)

Lambdas = np.arange(3, 10)

# P13, P11 = P13_finder(path, j, Lambdas, kind, mode)
# df = pandas.DataFrame(data=[P13, P11])
# file = open("./{}/P13_lambda_{}_{}.p".format(path, kind, j), "wb")
# pickle.dump(df, file)
# file.close()

# # # a, ctot2, ctot2_2, ctot2_3 = ctot_finder(Lambdas, path, j, A, mode, kind, n_runs, n_use)
# a, ac_true, ac, ac2, ac3, ac4, err = alpha_c_lambda(Lambdas, path, j, A, mode, kind, n_runs, n_use, fm=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)
# df = pandas.DataFrame(data=[a, ac_true, ac, ac2, ac3, ac4, err])
# file = open("./{}/loops_with_lambda_{}_N{}.p".format(path, kind, j), "wb")
# pickle.dump(df, file)
# file.close()



def ratio_j(j, path, Lambdas, kind):
    P13, P11 = P13_finder(path, j, Lambdas, kind, mode)
    ac_true_list, ac1_list, ac2_list, ac3_list, ac4_list, ac5_list, ac_pred_list =  [], [], [], [], [], [], []
    err0_list, errp_list, errm_list = [], [], []
    for Lambda_int in Lambdas:
        file = open(f"./{path}/alpha_c_{kind}_{Lambda_int}.p", "rb")

        read_file = pickle.load(file)
        a, ac_true, ac1, ac2, ac3, ac4, ac5, _, _, _, ac_pred = np.array(read_file)
        ac_true_list.append(ac_true[j])
        ac1_list.append(ac1[j]) #F3P
        ac2_list.append(ac2[j]) #F6P   
        ac3_list.append(ac3[j]) #M&W   
        ac4_list.append(ac4[j]) #SC   
        ac5_list.append(ac5[j]) #SC\delta   
        ac_pred_list.append(ac_pred[j])    

    M = P11 / P13
    ratio_true = np.array(ac_true_list) * M
    ratio1 = np.array(ac1_list) * M
    ratio2 = np.array(ac2_list)* M
    ratio3 = np.array(ac3_list) * M
    ratio4 = np.array(ac4_list) * M
    ratio5 = np.array(ac5_list) * M
    ratio_pred = np.array(ac_pred_list) * M

    e = 0.1
    for m in range(Lambdas.size):
        Lambda = Lambdas[m]
        a0, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, f'new_hier/data_{kind}/L{Lambda_int}/')
        P_lin = np.abs(np.conj(d1k) * d1k) * a0**2
        acp = ac_true_list[m] + (P_nb[mode] * e / (2 * 100 * P_lin[mode]))
        acm = ac_true_list[m] - (P_nb[mode] * e / (2 * 100 * P_lin[mode]))
        errp_list.append(acp)
        errm_list.append(acm)

    return a[j], ratio_true, ratio1, ratio2, ratio3, ratio4, ratio5, ratio_pred, np.array(errp_list)*M, np.array(errm_list)*M

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1]})
ax[0].set_ylabel('$\eta(\Lambda, k=k_{\mathrm{f}})$', fontsize=24)
ax[1].set_xlabel(r'$\Lambda/k_{{\mathrm{{f}}}}$ ({})'.format(kind_txt), fontsize=24)
# ax[2].set_ylabel('$\eta(\Lambda)$', fontsize=24)
# titles = [r'from fit to $\langle\tau\rangle$', r'M\&W', r'Spatial Corr']
# ax[0].set_ylim(0, 1)
titles = [r'F3P', r'M\&W', r'SC']

a_list, ratios_true_list, ratios1_list, ratios2_list, ratios3_list, ratios4_list, ratios5_list, ratios_pred_list, errp_list, errm_list = [], [], [], [], [], [], [], [], [], []
for j in [11, 23, 50]:
    a, ratio_true, ratio1, ratio2, ratio3, ratio4, ratio5, ratio_pred, errp, errm = ratio_j(j, path, Lambdas, kind)
    a_list.append(a)
    ratios_true_list.append(ratio_true)
    ratios1_list.append(ratio1)
    ratios2_list.append(ratio2)
    ratios3_list.append(ratio3)
    ratios4_list.append(ratio4)
    ratios5_list.append(ratio5)
    errp_list.append(errp)
    errm_list.append(errm)


linestyles = ['solid', 'dashdot', 'dashed']
# ax[2].yaxis.set_label_position('right')
print(len(ratios_true_list))
for i in range(3):
    ax[i].set_title(titles[i], fontsize=24)

    ax[0].plot(Lambdas, ratios1_list[i], c='k', lw=1.5, ls=linestyles[i], marker='o')
    ax[1].plot(Lambdas, ratios3_list[i], c='midnightblue', lw=1.5, ls=linestyles[i],  marker='*')
    ax[2].plot(Lambdas, ratios4_list[i], c='magenta', lw=1.5, ls=linestyles[i],  marker='v')
    # ax[0].fill_between(Lambdas, errm_list[i], errp_list[i], color='darkslategray', alpha=0.5)

    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=22)
    ax[i].yaxis.set_ticks_position('both')
    #
    # if kind == 'sharp':
    #     anchor_x = 1.1

    # elif kind == 'gaussian':
    #     anchor_x = 1.52


ax[1].tick_params(labelleft=False)
ax[2].tick_params(labelleft=False, labelright=False)

fig.align_labels()
plt.subplots_adjust(wspace=0)

line1 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='solid')
line2 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashdot')
line3 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashed')

labels = ['a = {}'.format(a_list[0]), 'a = {}'.format(a_list[1]), 'a = {}'.format(a_list[2])]
handles = [line1, line2, line3]
# plt.legend(handles=handles, labels=labels, fontsize=22, bbox_to_anchor=(1.67,1.05))
plt.legend(handles=handles, labels=labels, fontsize=20, ncol=3, bbox_to_anchor=(1,1.23))

# labels = 
plt.savefig('../plots/paper_plots_final/loops_with_Lambda_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
plt.close()
# plt.show()

# for j in tqdm(range(22, 51)):
#     a, ratio_true, ratio1, ratio2, ratio3, ratio4, ratio_pred = ratio_j(j, path, Lambdas, kind)
#     ratios = [ratio_true, ratio1, ratio2, ratio3, ratio4, ratio_pred]
#     plt.rcParams.update({"text.usetex": True})
#     plt.rcParams.update({"font.family": "serif"})
#     fig, ax = plt.subplots(figsize=(10, 6))
#     labels=[r'from fit to $P_{N\mathrm{-body}}$', r'from fit to $\langle\tau\rangle$', r'M\&W', r'Spatial Corr', r'Spatial Corr from $\delta_{\ell}$', r'$\alpha_{c} \propto a^{2}$']
#     ax.set_ylim(0.2, 1)
#     ax.set_title(rf'$a = {a}$', fontsize=20)
#     ax.set_xlabel(rf'$\Lambda \;[k_{{\mathrm{{f}}}}]$ ({kind_txt})', fontsize=22)
#     ax.set_ylabel(r'$\eta(\Lambda)$', fontsize=22)

#     colours = ['g', 'k', 'cyan', 'magenta', 'orange', 'seagreen']
#     markers = ['x', 'o', '*', 'v', '+', '1']

#     for l in range(5):
#         ax.plot(Lambdas[:-2], ratios[l][:-2], c=colours[l], lw=1.5, marker=markers[l], markersize=8, label=labels[l])

#     ax.legend(fontsize=14, bbox_to_anchor=(0.995, 1.025), bbox_transform=ax.transAxes)
#     ax.minorticks_on()
#     ax.tick_params(axis='both', which='both', direction='in', labelsize=16)
#     ax.yaxis.set_ticks_position('both')
#     # plt.show()
#     plt.tight_layout()
#     plt.savefig(rf'../plots/paper_plots_final/loops_with_Lambda/eta_{j}.png', bbox_inches='tight', dpi=300)
#     plt.close()
