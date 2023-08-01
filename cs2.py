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

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

A = [-0.05, 1, -0.5, 11]
Lambda_int = 3
Lambda = Lambda_int * (2*np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'

# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

path = 'cosmo_sim_1d/sim_k_1_11/run1/'


flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

N = 51
folder_name = '/new_hier/data_{}/L{}/'.format(kind, Lambda_int)
plots_folder = '/paper_plots_final/'

file = open("./{}/fit_cs_L{}_{}.p".format(path, int(Lambda/(2*np.pi)), kind), 'rb')
read_file = pickle.load(file)
a_list, cs2_0, cv2_0, cs2_0_6par, cv2_0_6par = np.array(read_file)
file.close()

file = open("./{}/cross_corr_cs_L{}_{}.p".format(path, int(Lambda/(2*np.pi)), kind), 'rb')
read_file = pickle.load(file)
a_list, cs2_1, cv2_1, cD = np.array(read_file)
file.close()

# file = open("./{}/ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'rb')
# read_file = pickle.load(file)
# a_list, ctot2_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, ctot2_5_list, ctot2_6_list = np.array(read_file)
# print(a_list.size)
# file.close()



file = open(f"./{path}/ellipse_23.p", 'rb')
read_file = pickle.load(file)
a, cov, corr, mean_x, mean_y = np.array(read_file)
file.close()

c2_0 = cs2_0 + cv2_0
c2_1 = cs2_1 + cv2_1

flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})

fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle(rf'$\Lambda = {Lambda_int} \,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=20, y=0.975)

cs2_0_6par[22] = (cs2_0_6par[21] + cs2_0_6par[23]) / 2
cv2_0_6par[22] = (cv2_0_6par[21] + cv2_0_6par[23]) / 2

# cs2_0_6par[45] = (cs2_0_6par[44] + cs2_0_6par[46]) / 2
# cv2_0_6par[45] = (cv2_0_6par[44] + cv2_0_6par[46]) / 2


ax[0].plot(a_list, cs2_0, c='k', lw=2)
ax[0].plot(a_list, cs2_0_6par, c='seagreen', lw=2, ls='dashdot')
ax[0].plot(a_list, cs2_1, c='magenta', lw=2, ls='dashed')
ax[1].plot(a_list, -cv2_0, c='k', lw=2, label=r'F3P')
ax[1].plot(a_list, -cv2_0_6par, c='seagreen', lw=2, ls='dashdot', label='F6P')
ax[1].plot(a_list, -cv2_1, c='magenta', lw=2, ls='dashed', label=r'SC')

# # ax[0].plot(a_list, cs2_0, c='b', lw=1.5)
# # ax[0].plot(a_list, cs2_0_6par, c='r', lw=1.5, ls='dashdot')
# # ax[0].plot(a_list, cs2_1, c='k', lw=1.5, ls='dashed')
# ax[1].plot(a_list, c2_0, c='b', lw=1.5, label=r'F3P')
# ax[1].plot(a_list, cs2_0_6par + cv2_0_6par, c='r', lw=1.5, ls='dashdot', label='F6P')
# ax[1].plot(a_list, c2_1, c='k', lw=1.5, ls='dashed', label=r'SC')

ax[1].axvline(a_list[12], c='teal', lw=0.5, ls='dashed', zorder=1, label=r'$a_{\mathrm{shell}}$')

for j in range(N):
    if flags[j] == 1:
        ax[0].axvline(a_list[j], c='teal', lw=0.5, ls='dashed', zorder=1)
        ax[1].axvline(a_list[j], c='teal', lw=0.5, ls='dashed', zorder=1)
    else:
        pass

ax[0].set_ylabel('$c_{\mathrm{s}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=18)
ax[1].set_ylabel('$c_{\mathrm{v}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=18)
ax[1].set_xlabel(r'$a$', fontsize=18)
# handles, labels = ax[1].get_legend_handles_labels()
# handles.append(sc_line)
# labels.append(r'$a_{\mathrm_{shell}}$')

for j in range(2):
    ax[j].minorticks_on()
    ax[j].tick_params(axis='both', which='both', direction='in', labelsize=14)
    ax[j].yaxis.set_ticks_position('both')

fig.align_labels()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
def plot_confidence_ellipse(cov, corr, x, y, n_std, **kwargs):
    cov = cov[1:, 1:]
    corr = corr[1:, 1:]
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                    **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
    .rotate_deg(135) \
    .scale(scale_x, scale_y) \
    .translate(x, y)

    ellipse.set_transform(transf + inset.transData)
    return inset.add_patch(ellipse)

plt.legend(fontsize=12)
inset = inset_axes(ax[0],
                    width=0.95, # width = 30% of parent_bbox
                    height=0.95, # height : 1 inch
                    bbox_to_anchor=(-0.325, 0, 1, 1),
                    bbox_transform=ax[0].transAxes)

inset.scatter(mean_x[0], -mean_y[0], s=0)
inset.minorticks_on()
inset.tick_params(axis='both', which='both', direction='in', labelsize=8)
inset.yaxis.set_ticks_position('both')
inset.set_title(rf'$a = {a[0]}$', fontsize=8, x=0.3, y=0.75)
inset.set_xlabel('$c_{\mathrm{s}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=10)
inset.set_ylabel('$c_{\mathrm{v}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=10)
# inset.yaxis.set_label_position('right')
# inset_xlim = inset.get_xlim()
# print(inset_xlim)
inset.set_xlim(37.53, 37.79)
inset.set_ylim(35.25, 35.52)


plot_confidence_ellipse(cov[0], corr[0], mean_x[0], -mean_y[0], n_std=1, edgecolor='blue', lw=0.5)#, label=r'$1\sigma$')

plt.subplots_adjust(hspace=0)
# plt.tight_layout()
# plt.show()
plt.savefig(f'../plots/{plots_folder}/cs2_{kind}.pdf', bbox_inches='tight', dpi=300)
# # plt.savefig(f'../plots/{plots_folder}/ctot2_{kind}.png', bbox_inches='tight', dpi=300)
# 
plt.close()



# file = open(f"./{path}/ellipse.p", 'rb')
# read_file = pickle.load(file)
# cov, corr, C_3par, C_3par2 = np.array(read_file)
# file.close()
# a = 3.03

# C_3par = [0, C_3par[0], C_3par2[0]]

# cov, corr = cov[0], corr[0]


# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# fig, ax = plt.subplots()
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=14)
# ax.yaxis.set_ticks_position('both')
# ax.set_title(rf'$a = {a}$', fontsize=22)
# ax.set_xlabel('$c_{\mathrm{s}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=18)
# ax.set_ylabel('$c_{\mathrm{v}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=18)

# def plot_confidence_ellipse(cov, corr, x, y, n_std, **kwargs):
#     cov = cov[1:, 1:]
#     corr = corr[1:, 1:]

#     pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
#     # Using a special case to obtain the eigenvalues of this
#     # two-dimensional dataset.
#     ell_radius_x = np.sqrt(1 + pearson)
#     ell_radius_y = np.sqrt(1 - pearson)
#     ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
#                     **kwargs)

#     scale_x = np.sqrt(cov[0, 0]) * n_std
#     scale_y = np.sqrt(cov[1, 1]) * n_std

#     transf = transforms.Affine2D() \
#     .rotate_deg(45) \
#     .scale(scale_x, scale_y) \
#     .translate(x, y)

#     ellipse.set_transform(transf + ax.transData)
#     return ax.add_patch(ellipse)


# ax.scatter(C_3par[1], C_3par[2], c='red', s=20)
# # plot_confidence_ellipse(cov, corr, C_3par[1], C_3par[2], 1, edgecolor='blue')#, label=r'$1\sigma$')
# # plot_confidence_ellipse(cov, corr, C_3par[1], C_3par[2], 100, edgecolor='red', linestyle='dashed', label=r'$2\sigma$')


# # plt.legend(fontsize=14)
# plt.tight_layout()
# plt.show()

# plt.savefig('../plots/paper_plots_final/error_ellipse.png', bbox_inches='tight', dpi=300)
# plt.close()


# fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_title(rf'$\Lambda = {Lambda_int} \,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=18, y=1.01)

# ax.plot(a_list, cs2_0, c='b', lw=1.5, label=r'$c_{\mathrm{s}}^{2}$; spatial corr')
# ax.plot(a_list, cs2_1, c='k', lw=1.5, ls='dashed', label=r'$c_{\mathrm{s}}^{2}$; fit')

# # ax.plot(a_list, cv2_0, c='b', lw=1.5, label=r'$c_{\mathrm{v}}^{2}$; spatial corr')
# # ax.plot(a_list, cv2_1, c='k', lw=1.5, ls='dashed', label=r'$c_{\mathrm{v}}^{2}$; fit')

# # ax.plot(a_list, cD, c='b', lw=1.5, label=r'$c_{\mathrm{tot}}^{2}$; spatial corr from $\delta_{\ell}$')
# # ax.plot(a_list, cs2_0+cv2_0, c='k', lw=1.5, ls='dashed', label=r'$c_{\mathrm{tot}}^{2}$; fit')
# # ax.plot(a_list, cs2_1+cv2_1, c='r', lw=1.5, ls='dashdot', label=r'$c_{\mathrm{tot}}^{2}$; spatial corr')

# for j in range(N):
#     if flags[j] == 1:
#         ax.axvline(a_list[j], c='teal', lw=1, ls='dashed', zorder=1)

#     else:
#         pass


# ax.set_xlabel(r'$a$', fontsize=20)
# ax.set_ylabel('$c^{2}\;[H_{0}^{2}L^{2}]$', fontsize=20)
# plt.legend(fontsize=14)
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
# ax.yaxis.set_ticks_position('both')
# # plt.show()
# plt.savefig(f'../plots/{plots_folder}/cs2_{kind}.png', bbox_inches='tight', dpi=300)
# plt.close()
