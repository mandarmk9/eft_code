#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pickle

from matplotlib import ticker
from matplotlib.tri import Triangulation, LinearTriInterpolator
from functions import read_sim_data, percentile_fde, deriv_param_calc
from scipy.interpolate import griddata
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


Lambda_int = 3
Lambda = Lambda_int * (2*np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'

file_num = 0
grid_size = 5
m = 1

dels, thes, taus, a_list, ctot2_list, err_list = [], [], [], [], [], []
for file_num in range(50):
    # file_num = 
    path = 'cosmo_sim_1d/sim_k_1_11/run1/'
    a, x, d1k, dels, thes, taus, P_nb, P_1l = read_sim_data(path, Lambda, kind, file_num, folder_name=f'/new_hier/data_{kind}/L{Lambda_int}/')
    thes = thes * (np.sqrt(a) / 100)
    taus -= np.mean(taus)

    for run in range(1, 9):

        # if False:
        #     pass
        # else:
        path = 'cosmo_sim_1d/sim_k_1_11/run{}/'.format(run)
        a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, file_num, folder_name=f'/new_hier/data_{kind}/L{Lambda_int}/')
        dv_l = dv_l * (np.sqrt(a) / 100)
        tau_l -= np.mean(tau_l)

        dels = np.concatenate((dels, dc_l))
        thes = np.concatenate((thes, dv_l))
        taus = np.concatenate((taus, tau_l))

    sort_inds = np.argsort(dels)
    dels = dels[sort_inds]
    thes = thes[sort_inds]
    taus = taus[sort_inds]

    # points = (dels, thes)
    # values = taus

    # # x_0, x_n = dels.min(), dels.max()
    # # y_0, y_n = thes.min(), thes.max()
    # #
    # # fac = 0.75
    # # x_0, x_n = np.median(dels)-np.median(dels)/fac, np.median(dels)+np.median(dels)/fac
    # # y_0, y_n = np.median(thes)-np.median(thes)/fac, np.median(thes)+np.median(thes)/fac

    # ind_0 = np.argmin(dels**2 + thes**2)
    # fac = 5
    # x_n, y_n = dels[ind_0]*fac, thes[ind_0]*fac
    # x_grid = np.linspace(-x_n, x_n, grid_size)
    # y_grid = np.linspace(-y_n, y_n, grid_size)
    # X, Y = np.meshgrid(x_grid, y_grid)

    # dist, points_ = [], []
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         point = (X[i, j], Y[i, j])
    #         dist.append(np.sqrt(point[1]**2 + point[0]**2))
    #         points_.append([i,j])

    # p0 = points_[np.argmin(dist)]
    # grid_tau = griddata(points, values, (X, Y), method='nearest')

    # # p0 = [grid_size//2, grid_size//2]
    # p1 = [p0[0]+m, p0[1]]
    # p2 = [p0[0], p0[1]+m]
    # p3 = [p0[0]-m, p0[1]]
    # p4 = [p0[0], p0[1]-m]

    # del_x = X[p2[0], p2[1]] - X[p4[0], p4[1]]
    # del_y = Y[p1[0], p1[1]] - Y[p3[0], p3[1]]


    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\delta_{l}$', fontsize=16)
    ax.set_ylabel(r'$\theta_{l}$', fontsize=16)
    ax.set_title(r'$a = {}$'.format(a), fontsize=16)
    # ax.axhline(thes.max(), ls='dashed', c='k')
    # ax.axhline(thes.min(), ls='dashed', c='k')
    # ax.axvline(dels.min(), ls='dashed', c='k')
    # ax.axvline(dels.max(), ls='dashed', c='k')
    # ax.scatter(X, Y, c=grid_tau, s=5)
    # ax.plot(X[p0[0], p0[1]], Y[p0[0], p0[1]], c='k', marker='o', markersize=5)
    # ax.plot(X[p1[0], p1[1]], Y[p1[0], p1[1]], c='k', marker='o', markersize=2.5)
    # ax.plot(X[p2[0], p2[1]], Y[p2[0], p2[1]], c='k', marker='o', markersize=2.5)
    # ax.plot(X[p3[0], p3[1]], Y[p3[0], p3[1]], c='k', marker='o', markersize=2.5)
    # ax.plot(X[p4[0], p4[1]], Y[p4[0], p4[1]], c='k', marker='o', markersize=2.5)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=13.5)
    obj = ax.scatter(dels, thes, s=2, c='b')
    # cbar = fig.colorbar(obj, ax=ax)
    # cbar.ax.set_ylabel(r'$[\tau]_{\Lambda}\; [\mathrm{M}_{\mathrm{p}}H_{0}^{2}L^{-1}]$', fontsize=18)
    # print(dels.max(), thes.max())
    # plt.show()
    
    plt.savefig('../plots/paper_plots_final/dt_grid/dt_{0:02d}.png'.format(file_num), bbox_inches='tight', dpi=150)
    plt.close()

#     C0 = grid_tau[p0[0], p0[1]]
#     C1_del = (grid_tau[p2[0], p2[1]] - grid_tau[p4[0], p4[1]]) / (del_x)
#     C1_the = (grid_tau[p1[0], p1[1]] - grid_tau[p3[0], p3[1]]) / (del_y)
#     C1 = C1_del + C1_the
#     C_ = [C0, C1]

#     rho_b = 27.755 / a**3
#     ctot2_list.append(C_[1] / rho_b)
#     # err_list.append(err_[1] / rho_b)
#     a_list.append(a)
#     print('a = ', a, 'ctot2 = ', C_[1]/rho_b)


# df = pandas.DataFrame(data=[a_list, ctot2_list, err_list])
# file = open("./{}/nsim_dde_ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'wb')
# pickle.dump(df, file)
# file.close()

# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# fig, ax = plt.subplots()
# path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')
# file = open("./{}/ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'rb')
# read_file = pickle.load(file)
# a_list, ctot21_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, err4_list = np.array(read_file)
# file.close()
# for j in range(24):
#     if flags[j] == 1:
#         sc_line = ax.axvline(a_list[j], c='teal', lw=1, zorder=1)
#     else:
#         pass

# path = 'cosmo_sim_1d/sim_k_1_11/run8/'
# file = open("./{}/nsim_dde_ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'rb')
# read_file = pickle.load(file)
# a_list, ctot2_list, err_list = np.array(read_file)
# file.close()





# ax.set_xlabel(r'$a$', fontsize=18)
# ax.set_ylabel(r'$c_{\mathrm{tot}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=18)
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=13.5)
# ax.set_title(r'$\Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=16)
# ax.plot(a_list[:23], ctot2_list[:23], c='k', marker='v', lw=1.5, label='DDE')
# ax.plot(a_list[:23], ctot2_2_list[:23], c='cyan', lw=1.5, marker='*', label='M\&W') #M&W

# plt.legend()
# # ax.fill_between(a_list, ctot2_list-err_list, ctot2_list+err_list, color='darkslategray', alpha=0.5, rasterized=True)
# # plt.savefig('../plots/test/new_paper_plots/ctot2_dde_{}.png'.format(file_num), bbox_inches='tight', dpi=150)
# # plt.close()
# plt.show()
