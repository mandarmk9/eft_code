#!/usr/bin/env python3
import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# paths = ['cosmo_sim_1d/sim_k_1_7/run1/', 'cosmo_sim_1d/sim_k_1_11/run1/', 'cosmo_sim_1d/sim_k_1_15/run1/']
# paths = ['cosmo_sim_1d/amps_sim_k_1_11/run1/', 'cosmo_sim_1d/amps2_sim_k_1_11/run1/']
# paths = ['cosmo_sim_1d/multi_k_sim/run1/']
# paths = ['cosmo_sim_1d/multi_sim_3_15_33/run1/']

# path = 'cosmo_sim_1d/amp_ratio_test/run1/'
paths = ['cosmo_sim_1d/sim_k_1_11/run1/']

Nfiles = 50
# plots_folder = 'test/multi_sim_3_15_33/'
# title = r'\texttt{sim\_3\_15\_33}'

plots_folder = 'paper_plots_final/' #'test/new_paper_plots/'
title = r'\texttt{sim\_1\_11}'
# k1, k2, k3 = 3, 9, 27
k1, k2, k3 = 1, 3, 9


def k_NL_ext(path, Nfiles, ind='mean'):
    print(path)
    a_list, k_NL = [], []
    for j in tqdm(range(Nfiles)):
        moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
        nbody_filename = 'output_{0:04d}.txt'.format(j)
        moments_file = np.genfromtxt(path + moments_filename)
        a = moments_file[:,-1][0]
        nbody_file = np.genfromtxt(path + nbody_filename)
        Psi = nbody_file[:,1]
        x_nbody = nbody_file[:,-1]
        v_nbody = nbody_file[:,2]

        Psi *= (2*np.pi)
        if ind == 'mean':
            k_NL.append(1/np.mean(np.abs(Psi)))
        elif ind == 'median':
            k_NL.append(1/np.median(np.abs(Psi)))

        a_list.append(a)
        # std = np.std([Psi], ddof=1)
        # print('sigma = ', std)
        # print('mean(|Psi|)', np.mean(np.abs(Psi)))
        # print('a = {}, k_NL = {}\n'.format(np.round(a,4), np.round(1/np.mean(np.abs(Psi)), 3)))
    return a_list, k_NL

# k_NL_lists = []
# # ind = 'mean'
# # ind = 'median'
# inds = ['mean', 'median']
# # for j in range(len(paths)):
# # # j = 0
# for ind in inds:
#     lists = k_NL_ext(paths[0], Nfiles, ind)
#     a_list = lists[0]
#     k_NL_lists.append(lists[1])
#     df = pandas.DataFrame(data=k_NL_lists)
#     pickle.dump(df, open('./{}/k_NL_lists_{}.p'.format(paths[0], ind), 'wb'))

# path = paths[0]
# a_list = np.zeros(Nfiles)
# for j in range(Nfiles):
#     a_list[j] = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
#     # print('a = ', a_list[j])
# df = pandas.DataFrame(data=a_list)
# pickle.dump(df, open('./{}/a_list.p'.format(paths[0]), 'wb'))

# df = pandas.DataFrame(data=k_NL_lists)
# pickle.dump(df, open('./data/k_NL_lists_{}.p'.format(ind), 'wb'))

data = pickle.load(open("./{}/k_NL_lists_{}.p".format(paths[0], 'mean'), "rb" ))
data_median = pickle.load(open("./{}/k_NL_lists_{}.p".format(paths[0], 'median'), "rb" ))
a_list = np.array(pickle.load(open('./{}/a_list.p'.format(paths[0]), "rb" ))[0])

# A = [-0.05, 1, -0.04, 2, -0.03, 3, -0.02, 4, -0.01, 5, -1, 11]
# A = [-0.1, 1, -0.5, 11]


# A = [-0.05, 3, -0.1, 15, -0.5, 33]
A = [-0.05, 1, -0.5, 7, 0, 33]

from zel import initial_density
x = np.arange(0, 1, 1/1000)
a_sc = 1 / np.max(initial_density(x, A, 1))
# # print(a_sc)
k_NL_mean = np.array(data)[0]
k_NL_median = np.array(data_median)[0]

# a_list, k_NL_mean = k_NL_ext(path, Nfiles, ind='mean')
# a_list, k_NL_median = k_NL_ext(path, Nfiles, ind='median')


k_NL_lists = [k_NL_mean, k_NL_median]
labels = ['mean', 'median'] #[r'\texttt{sim\_k\_1\_7}', r'\texttt{sim\_k\_1\_11}', r'\texttt{sim\_k\_1\_15}']
# labels = [r'\texttt{sim\_multi\_k}: mean', r'\texttt{sim\_multi\_k}: median'] #[r'\texttt{sim\_k\_1\_7}', r'\texttt{sim\_k\_1\_11}', r'\texttt{sim\_k\_1\_15}']
# labels = [r'\texttt{amp\_ratio\_test}: mean', r'\texttt{amp\_ratio\_test}: median'] #[r'\texttt{sim\_k\_1\_7}', r'\texttt{sim\_k\_1\_11}', r'\texttt{sim\_k\_1\_15}']

colours = ['b', 'r', 'k']
linestyles = ['solid', 'dashdot', 'dotted']
plt.rcParams.update({"text.usetex": True})

fig, ax = plt.subplots()

ax.set_xlabel(r'$a$', fontsize=18)
ax.set_ylabel(r'$k_{\mathrm{NL}}/k_{\mathrm{f}}$', fontsize=18)
# ax.set_ylabel(r'$k_{\mathrm{f}}/k_{\mathrm{NL}}$', fontsize=18)

ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.set_title(f'{title}', fontsize=18)
# ax.axhline(k1, c='brown', ls='dashed', label=rf'$k={k1}$', lw=1)
# ax.axhline(k2, c='k', ls='dashed', label=rf'$k={k2}$', lw=1)
# ax.axhline(k3, c='magenta', ls='dashed', label=rf'$k={k3}$', lw=1)

# ax.axhline(k1, c='brown', ls='dashed', label=r'$k=k_{\mathrm{f}}$', lw=1)
# ax.axhline(k2, c='k', ls='dashed', label=rf'$k={k2}\,k_{{\mathrm{{f}}}}$', lw=1)
# ax.axhline(k3, c='magenta', ls='dashed', label=rf'$k={k3}\,k_{{\mathrm{{f}}}}$', lw=1)

print(k_NL_mean[-2:])
# print(a_list[15:17])


ax.axhline(k1, c='midnightblue', ls='dashed', label=r'$k_{\mathrm{f}}$', lw=1)
ax.axhline(k2, c='k', ls='dashed', label=rf'${k2}\,k_{{\mathrm{{f}}}}$', lw=1)
ax.axhline(k3, c='magenta', ls='dashed', label=rf'${k3}\,k_{{\mathrm{{f}}}}$', lw=1)

ax.axvline(a_sc, c='teal', ls='dashed', label=r'$a_{\mathrm{shell}}$', lw=1)
# ax.axhline(5, c='brown', ls='dashed', label=r'$k=5$', lw=1)
ax.minorticks_on()
ax.yaxis.set_ticks_position('both')

for j in range(2):
    ax.plot(a_list, k_NL_lists[j], c=colours[j], lw=1.5, ls=linestyles[j], label=labels[j])


# labels = [r'1-loop', r'2-loop']
# labels = [r'1-loop / 2-loop', r'2-loop']
# j = 0
# ax.plot(a_list, (1/k_NL_lists[j])**2 / (1/k_NL_lists[j])**3, c=colours[j], lw=1.5, ls=linestyles[j], label=labels[j])
# ax.axhline(1, c='brown', ls='dashed', label='1')
# ax.plot(a_list, (1/k_NL_lists[j])**2 / a_list**2, c=colours[j], lw=1.5, ls=linestyles[j], label=labels[j])

# file = open(f"./{paths[0]}/new_trunc_alpha_c_sharp_3.p", "rb")
# read_file = pickle.load(file)
# a_list, alpha_c_true, alpha_c_naive, alpha_c_naive2, alpha_c_naive3, alpha_c_naive4 = np.array(read_file)
# file.close()

# j = 1
# ax.plot(a_list, alpha_c_true, c=colours[j], lw=1.5, ls=linestyles[j], label=labels[j])

# j = 1
# ax.plot(a_list, (1/k_NL_lists[j])**4, c=colours[j], lw=1.5, ls=linestyles[j], label=labels[j])


plt.legend(fontsize=13)
plt.savefig(f'../plots/{plots_folder}/k_NL_ev.pdf', dpi=300)
# plt.savefig('../plots/test/new_paper_plots/k_NL_ev.pdf', dpi=300, bbox_inches='tight')
# # plt.savefig('../plots/multi_k_sim/k_NL_multi.png', dpi=150, bbox_inches='tight')
# # plt.savefig('../plots/ratio_test/k_NL_multi.png', dpi=150, bbox_inches='tight')
plt.close()

# plt.show()

# plt.savefig('../plots/loops_ratio.png', dpi=150, bbox_inches='tight')
# plt.close()

# data = pickle.load(open("./data/k_NL_lists.p", "rb" ))
# data_median = pickle.load(open("./data/k_NL_lists_median.p", "rb" ))
# a_list = np.array(pickle.load(open("./data/sim_k_1_11_a_list.p", "rb" ))[0])
# # k_NL_7 = [data[j][0] for j in range(data.shape[1])]
# k_NL_11 = [data[j][1] for j in range(data.shape[1])]
# k_NL_11_median = [data_median[j][1] for j in range(data_median.shape[1])]
# k_NL_15 = [data[j][2] for j in range(data.shape[1])]
#
# # k_NL_11 = [data[j][1] for j in range(data.shape[1])]
# # k_NL_11_median = [data_median[j][1] for j in range(
# # k_NL_lists = [k_NL_11, k_NL_11_median] #[k_NL_7, k_NL_11, k_NL_15]



# ax.set_ylabel(r'$N$')
# ax.set_xlabel(r'$\Psi\;[(2\pi h)^{-1}\;\mathrm{Mpc}]$')
#
# ax.hist(Psi, bins=50)
# # ax.hist(x_nbody, bins=50)
#
# ax.set_title('a = {}'.format(a))
# ax.set_title(title_txt, fontsize=18)
