#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas
import pickle

from functions import read_sim_data, AIC, BIC, spectral_calc
from scipy.optimize import curve_fit
from tau_fits import tau_calc
from matplotlib.patches import Ellipse

def calc(j, Lambda, path, mode, kind, n_runs, n_use, folder_name):
    a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)

    H = a**(-1/2)*100
    dv_l = -dv_l / (H)
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

    def fitting_function(X, a0, a1, a2):
        x1, x2 = X
        return a0 + a1*x1 + a2*x2

    guesses = 1, 1, 1
    C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    C_3par = C / (27.755 / a**3)

    corr = np.zeros(cov.shape)
    for i in range(3):
        corr[i,:] = [cov[i,j] / np.sqrt(cov[i,i]*cov[j,j]) for j in range(3)]

    def fitting_function(X, a0, a1, a2, a3, a4, a5):
        x1, x2 = X
        return a0 + a1*x1 + a2*x2 + a3*x1**2 + a4*x2**2 + a5*x1*x2

    guesses = 1, 1, 1, 1, 1, 1
    C, cov_6 = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    C_6par = C / (27.755 / a**3)

    return a, x, C_3par, C_6par, cov, corr

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

file_num = 23
folder_name = '/new_hier/data_{}/L{}/'.format(kind, int(Lambda/(2*np.pi)))

from tqdm import tqdm
c1_list, c2_list, c1_6par_list, c2_6par_list, a_list = [], [], [], [], []
# for file_num in tqdm(range(51)):
a, x, C_3par, C_6par, cov, corr = calc(file_num, Lambda, path, mode, kind, n_runs, n_use, folder_name)
# c1_list.append(C_3par[1])
# c2_list.append(C_3par[2])
# print(C_3par[1], C_3par[2], C_3par[1]+C_3par[2])

# c1_6par_list.append(C_6par[1])
# c2_6par_list.append(C_6par[2])

# print(C_6par[1], C_6par[2], C_6par[1]+C_6par[2])

# a_list.append(a)

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

# print(cov, corr, C_3par[1], C_3par[2])
# ax.scatter(C_3par[1], C_3par[2], c='red', s=20)
# plot_confidence_ellipse(cov, corr, C_3par[1], C_3par[2], 1, edgecolor='blue')#, label=r'$1\sigma$')
# # plot_confidence_ellipse(cov, corr, C_3par[1], C_3par[2], 100, edgecolor='red', linestyle='dashed', label=r'$2\sigma$')


# # plt.legend(fontsize=14)
# plt.tight_layout()
# plt.show()
# # plt.savefig('../plots/paper_plots_final/error_ellipse.png', bbox_inches='tight', dpi=300)
# # plt.close()

d_array = np.array([a, cov, corr, C_3par[1], C_3par[2]], dtype=object)
df = pandas.DataFrame(data=d_array)
file = open(f"./{path}/ellipse_{file_num}.p", 'wb')
pickle.dump(df, file)
file.close()
