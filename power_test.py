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

Lambda_int = 10
Lambda = Lambda_int*(2*np.pi)
folder_name = '/new_hier/data_{}/L{}/'.format(kind, Lambda_int)

# file = open(f"./{path}/alpha_c_{kind}_{Lambda_int}.p", "rb")
# read_file = pickle.load(file)
# a_list, alpha_c_true, alpha_c_F3P, alpha_c_F6P, alpha_c_MW, alpha_c_SC, alpha_c_SCD, _, _, _, alpha_c_pred = np.array(read_file)
# file.close()

kf = (2*np.pi)
# alpha_c_F3P /= kf**2
# alpha_c_SC /= kf**2
x = np.arange(0, 1.0, 0.001)
dc_in, k = dc_in_finder(path, x, interp=True)
mode = 1
Nfiles = 51
j = 35
dc_in = smoothing(dc_in, k, Lambda, kind)
a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
P11, P12, P22, P13, _, _, _, _, _ = SPT(dc_in, 1.0, a)

# f_Lambda = (P13 / (k**2 * P11 * a**2))
# c = -1/(4*np.pi**2)
# # c = -((2/(4*np.pi**2)))# + (2/(22*4*np.pi**2)))
# P13_model = c * k**2 * (P11**2)

# # f = -(((0.05)**2 / (2 * kf**2))) # + ((0.5)**2 / (2 * 121 * kf**2)))
# eta = 2 * a**2 * ((0.05**2 / (4 * kf**2)))# + (0.5**2 / (4 * (11*kf)**2)))
# f = -eta / a**2
# print('f = ', f)
# # f = -P11[1] / (a**2 * (2*np.pi)**2)
f = -((0.05**2 / (4 * kf**2)))

print('f = ', f)

P13_model = a**2 * k**2 * P11 * f

print(P13[:5])
print(P13_model[:5])

k /= (2*np.pi)

# # # ylabel = r'$a^{-2}L^{-1}P(k, a) \times 10^{4}$'
# # # ylabel = r'$\mathrm{log}10\left|a^{-4}L^{-1}P(k, a)\times 10^{4}\right|$'
# # # ylabel = r'$\left|a^{-4}L^{-1}P(k, a)\times 10^{6}\right|$'
# # # ylabel = r'$a^{-4}L^{-1}P(k, a)\times 10^{4}$'
# # ylabel = r'$k^{2}\alpha \times 10^{3}$'

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
# # title = rf'$k = k_{{\mathrm{{ren}}}} = k_{{\mathrm{{f}}}},\,\Lambda = {Lambda_int}\,k_{{\mathrm{{f}}}}\,$({kind_txt})'
# title = rf'$k = k_{{\mathrm{{f}}}},\,\Lambda = {Lambda_int}\,k_{{\mathrm{{f}}}}\,$({kind_txt})'
ax.scatter(k, P13, c='b', s=35)
ax.scatter(k, P13_model, c='k', s=25)
ax.set_xlim(0.5, 14.5)
plt.show()

# ax.set_xlabel(r'$a$', fontsize=20)
# ax.set_ylabel(ylabel, fontsize=20)

# # ax.plot(a_list, P_nb*1e4/(a_list**2), c='b', lw=2.5, label=r'$P_{N\mathrm{-body}}$')
# # ax.plot(a_list, P_1l*1e4/(a_list**2), c='brown', ls='dashdot', lw=2.5, label=r'$P_{\mathrm{tSPT-4}}$')
# # ax.plot(a_list, P_eft*1e4/(a_list**2), c='k', ls='dashed', lw=2.5, label=r'$P_{\mathrm{EFT}}: \mathrm{F3P}$')
# # ax.plot(a_list, P_eft_ren*1e4/(a_list**2), c='magenta', ls='dashed', lw=2.5, label=r'$P_{\mathrm{ren}}: \mathrm{F3P}$')

# # ax.plot(a_list, np.abs(2*np.pi*1e4*P13/(a_list**(4))), c='b', lw=2.5, label=r'$P_{13}$')
# # ax.plot(a_list, np.abs(2*np.pi*1e4*P13_ren/(a_list**(4))), c='k', ls='dashed', lw=2.5, label=r'$P_{\mathrm{ctr}}$')

# # # ax.plot(a_list, np.abs(2*np.pi*1e6*P13/a_list**4), c='b', lw=2.5, label=r'$P_{13}$')
# # # ax.plot(a_list, np.abs(2*np.pi*1e6*P13_ren/a_list**4), c='k', ls='dashed', lw=2.5, label=r'$P_{\mathrm{ctr}}$')

# # ax.plot(a_list, P13, c='b', lw=2.5, label=r'$P_{13}$')
# # ax.plot(a_list, P13_ren, c='k', ls='dashed', lw=2.5, label=r'$P_{\mathrm{ctr}}$')

# # ax.plot(a_list, P13, c='b', lw=2.5, label=r'$P_{13}$')
# # ax.plot(a_list, P13_ren, c='k', ls='dashed', lw=2.5, label=r'$P_{\mathrm{ctr}}$')


# c0 = -P13 / (P11 * (2*np.pi)**2)
# alpha_ctr_model = alpha_ctr[0]*a_list**2 / a_list[0]**2 #c0 #* a_list**(1.001)


# ax.plot(a_list, alpha_c_F3P*(2*np.pi)*1e3, c='b', lw=2.5, label=r'$\alpha_{c}$')
# ax.plot(a_list, alpha_ctr*(2*np.pi)*1e3, c='k', ls='dashdot', lw=2.5, label=r'$\alpha_{\mathrm{ctr}}$')
# ax.plot(a_list, alpha_ren*(2*np.pi)*1e3, c='r', ls='dashed', lw=2.5, label=r'$\alpha_{\mathrm{ren}}$')
# ax.plot(a_list, alpha_ctr_model*(2*np.pi)*1e3, c='magenta', ls='dotted', lw=2.5, label=r'$\alpha \propto a^{2}$')
# # ax.set_ylim(-3.2, 2.1)
# # ax.set_yscale('log')
# # ax.set_xscale('log')

# ax.tick_params(axis='both', which='both', direction='in', labelsize=14)
# ax.set_title(f'{title}', fontsize=20)
# ax.minorticks_on()
# ax.yaxis.set_ticks_position('both')
# # ax.set_xticklabels(ax.get_xticks(), minor=True)

# plt.legend(fontsize=13)
# plt.tight_layout()
# plt.savefig(f'../plots/paper_plots_final/renorm_time.pdf', dpi=300)
# # plt.savefig(f'../plots/paper_plots_final/alpha_ctr.pdf', dpi=300)
# # plt.savefig(f'../plots/paper_plots_final/eft_ren.pdf', dpi=300)
# plt.close()
# # plt.show()



