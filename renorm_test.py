#!/usr/bin/env python3

#import libraries
# import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle
from functions import read_density, dc_in_finder, dn, smoothing, spec_from_ens
from tqdm import tqdm

def SPT(dc_in, L, a):
   """Returns the SPT PS upto 2-loop order"""
   Nx = dc_in.size
   F = dn(5, L, dc_in)
   d1k = (np.fft.fft(F[0]) / Nx)
   d2k = (np.fft.fft(F[1]) / Nx)
   d3k = (np.fft.fft(F[2]) / Nx)
   d4k = (np.fft.fft(F[3]) / Nx)
   d5k = (np.fft.fft(F[4]) / Nx)

   P11 = np.real((d1k * np.conj(d1k)) * (a**2))
   P12 = np.real(((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3))
   P22 = np.real((d2k * np.conj(d2k)) * (a**4))
   P13 = np.real(((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4))
   P14 = np.real(((d1k * np.conj(d4k)) + (d4k * np.conj(d1k))) * (a**5))
   P23 = np.real(((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a**5))
   P33 = np.real((d3k * np.conj(d3k)) * (a**6))
   P15 = np.real(((d1k * np.conj(d5k)) + (d5k * np.conj(d1k))) * (a**6))
   P24 = np.real(((d2k * np.conj(d4k)) + (d4k * np.conj(d2k))) * (a**6))
   return P11, P12, P22, P13, P14, P23, P33, P15, P24

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

Lambda_int = 3
Lambda = Lambda_int*(2*np.pi)
folder_name = '/new_hier/data_{}/L{}/'.format(kind, Lambda_int)

file = open(f"./{path}/alpha_c_{kind}_{Lambda_int}.p", "rb")
read_file = pickle.load(file)
a_list, alpha_c_true, alpha_c_F3P, alpha_c_F6P, alpha_c_MW, alpha_c_SC, alpha_c_SCD, _, _, _, alpha_c_pred = np.array(read_file)
file.close()

kf = (2*np.pi)
alpha_c_F3P /= kf**2
alpha_c_SC /= kf**2
x = np.arange(0, 1.0, 0.001)
dc_in, k = dc_in_finder(path, x, interp=True)
mode = 1
Nfiles = 51

P_lin = []
for j in range(51):
    a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
    sol = SPT(dc_in, 1.0, a)
    P11_un = sol[0][mode]
    P13_un = sol[1][mode]
    P22_un = sol[2][mode]
    P_lin.append(P11_un + 2*P13_un + P22_un)

dc_in = smoothing(dc_in, k, Lambda, kind)
P11, P12, P13, P22, a_list, P_nb, P_1l, P_eft = [], [], [], [], [], [], [], []
for j in range(Nfiles):
    a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
    P11_a, P12_a, P22_a, P13_a, _, _, _, _, _ = SPT(dc_in, 1.0, a)
    a_list.append(a)
    P11.append(P11_a[mode])
    P13.append(P13_a[mode]/2)
    P22.append(P22_a[mode])
    P12.append(P12_a[mode])


P11 = np.array(P11)
P13 = np.array(P13)
P22 = np.array(P22)
a_list = np.array(a_list)
P_lin = np.array(P_lin)


sol = spec_from_ens(Nfiles, Lambda, path, 1, kind, folder_name=folder_name)
P_nb, P_1l, P_eft = sol[2:5]

# alpha_ctr = P13 / 2*P11*(kf**2)
# alpha_ren = alpha_c_F3P - alpha_ctr
# P_eft_ren = P11 + (2*alpha_ren*(kf**2)*P11)

alpha_ren = (P_nb - 2*P13 - P11) / (2 * kf**2 * P11)
savename = 'renorm_time_cond1'
title = rf'$k = k_{{\mathrm{{f}}}},\,\Lambda_{{\mathrm{{ren}}}}=3\,k_{{\mathrm{{f}}}}$'

# alpha_ren = (P_nb - P11) / (2 * kf**2 * P11)
# savename = 'renorm_time_cond2'
# title = rf'$k = k_{{\mathrm{{f}}}},\,\Lambda_{{\mathrm{{ren}}}}=0$'


alpha_ctr = alpha_c_F3P - alpha_ren
# c = ((((0.05)**2 / (4 * kf**2))) + ((0.5**2 / (4 * (11*kf)**2)))) * 2 #* a**2 
f = ((0.05)**2 / (4 * kf**2)) / 2
# P13_ctr = -f*a_list**2 * kf**2 * P11

# P13_ctr = -2*alpha_ctr*(kf**2)*P11
# alpha_c = (P_nb - P_1l) / (2 * kf**2 * P11)

# import pandas
# df = pandas.DataFrame(data=[alpha_ctr, alpha_c_F3P, alpha_ren, P11])
# file = open(f"./data/renorms_{kind}.p", "wb")
# pickle.dump(df, file)
# file.close()

# ylabel = r'$a^{-2}L^{-1}P(k, a) \times 10^{4}$'
# ylabel = r'$\mathrm{log}10\left|a^{-4}L^{-1}P(k, a)\times 10^{4}\right|$'
# ylabel = r'$\left|a^{-4}L^{-1}P(k, a)\times 10^{6}\right|$'
# ylabel = r'$a^{-4}L^{-1}P(k, a)\times 10^{4}$'
ylabel = r'$k^{2}\alpha \times 10^{3}$'

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
# title = rf'$k = k_{{\mathrm{{ren}}}} = k_{{\mathrm{{f}}}},\,\Lambda = {Lambda_int}\,k_{{\mathrm{{f}}}}\,$({kind_txt})'
# title = rf'$k = k_{{\mathrm{{f}}}},\,\Lambda = {Lambda_int}\,k_{{\mathrm{{f}}}}\,$({kind_txt})'

ax.set_xlabel(r'$a$', fontsize=20)
ax.set_ylabel(ylabel, fontsize=20)

# ax.plot(a_list, P_nb*1e4/(a_list**2), c='b', lw=2.5, label=r'$P_{N\mathrm{-body}}$')
# ax.plot(a_list, P_1l*1e4/(a_list**2), c='brown', ls='dashdot', lw=2.5, label=r'$P_{\mathrm{tSPT-4}}$')
# ax.plot(a_list, P_eft*1e4/(a_list**2), c='k', ls='dashed', lw=2.5, label=r'$P_{\mathrm{EFT}}: \mathrm{F3P}$')
# ax.plot(a_list, P_eft_ren*1e4/(a_list**2), c='magenta', ls='dashed', lw=2.5, label=r'$P_{\mathrm{ren}}: \mathrm{F3P}$')

# ax.plot(a_list, np.abs(2*np.pi*1e4*P13/(a_list**(4))), c='b', lw=2.5, label=r'$P_{13}$')
# ax.plot(a_list, np.abs(2*np.pi*1e4*P13_ren/(a_list**(4))), c='k', ls='dashed', lw=2.5, label=r'$P_{\mathrm{ctr}}$')

# # ax.plot(a_list, np.abs(2*np.pi*1e6*P13/a_list**4), c='b', lw=2.5, label=r'$P_{13}$')
# # ax.plot(a_list, np.abs(2*np.pi*1e6*P13_ren/a_list**4), c='k', ls='dashed', lw=2.5, label=r'$P_{\mathrm{ctr}}$')

# ax.plot(a_list, P13, c='b', lw=2.5, label=r'$P_{13}$')
# ax.plot(a_list, P13_ctr, c='k', ls='dashed', lw=2.5, label=r'$P_{\mathrm{ctr}}$')


# c0 = -P13 / (P11 * (2*np.pi)**2)
# # alpha_ctr_model = alpha_ctr[0]*a_list**2 / a_list[0]**2 #c0 #* a_list**(1.001)
# # print(alpha_ctr[0] / a_list[0]**2)

# # c = -(((0.05)**2 / (2 * kf**2)))
# c = 2*(-(((0.05)**2 / (4 * kf**2))) - ((0.5**2 / (4 * (11*kf)**2))))

alpha_ctr_model = f*a_list**2  #c0 #* a_list**(1.001)

alpha_ctr_model2 = alpha_ctr[0]*a_list**2 / a_list[0]**2 #c0 #* a_list**(1.001)


ax.plot(a_list, alpha_c_F3P*(2*np.pi)*1e3, c='b', lw=2.5, label=r'$\alpha_{c}$')
ax.plot(a_list, alpha_ctr*(2*np.pi)*1e3, c='k', ls='dashed', lw=2.5, label=r'$\alpha_{\mathrm{ctr}}$')
ax.plot(a_list, alpha_ren*(2*np.pi)*1e3, c='r', ls='dashdot', lw=2.5, label=r'$\alpha_{\mathrm{ren}}$')
# ax.plot(a_list, alpha_c*(2*np.pi)*1e3, c='magenta', ls='dotted', lw=2.5, label=r'$\alpha_{c}$; derived')

# ax.plot(a_list, alpha_ctr_model*(2*np.pi)*1e3, c='magenta', ls='dotted', lw=2.5, label=r'$\alpha \propto a^{2}$')
# ax.plot(a_list, alpha_ctr_model*(2*np.pi)*1e3, c='magenta', ls='dotted', lw=2.5, label=r'$-a^{2}f(\Lambda)$')
# ax.plot(a_list, alpha_ctr_model2*(2*np.pi)*1e3, c='seagreen', ls='dotted', lw=2.5, label=r'$\alpha_{\mathrm{ctr}}(a=0.5)(a/0.5)^{2}$')

ax.set_ylim(-3.2, 2.1)
# ax.set_yscale('log')
# ax.set_xscale('log')

ax.tick_params(axis='both', which='both', direction='in', labelsize=14)
ax.set_title(f'{title}', fontsize=20)
ax.minorticks_on()
ax.yaxis.set_ticks_position('both')
# ax.set_xticklabels(ax.get_xticks(), minor=True)

plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig(f'../plots/paper_plots_final/{savename}.pdf', dpi=300)
# plt.savefig(f'../plots/paper_plots_final/alpha_ctr.pdf', dpi=300)
# plt.savefig(f'../plots/paper_plots_final/eft_ren.pdf', dpi=300)
plt.close()
# plt.show()



