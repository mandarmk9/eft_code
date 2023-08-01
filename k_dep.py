#!/usr/bin/env python3

#import libraries
# import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.markers import MarkerStyle
from functions import read_density, dc_in_finder, dn
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
j = 50
a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
x = np.arange(0, 1.0, 0.001)
dc_in, k = dc_in_finder(path, x, interp=True)


_, P12, P22, P13, _, _, _, _, _ = SPT(dc_in, 1.0, a)

# del_J, del_J_corr = [], []
# for mode in np.arange(1, 13, 1):
#     mode = int(mode)
#     file = open(f"./{path}/stoch_del_{kind}_{Lambda_int}_{mode}.p", "rb")
#     read_file = pickle.load(file)
#     a_list, del_J_m, del_J_corr_m = np.array(read_file)
#     del_J.append(del_J_m[j])
#     del_J_corr.append(del_J_corr_m[j])
#     file.close()

# P_JJ = np.abs(k[1:13]**2 * del_J)**2
# P_JJ_corr = np.abs(k[1:13]**2 * del_J_corr)**2


k /= (2*np.pi)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
ax.set_title(r'$a = {}$'.format(a))
ax.set_ylabel(r'$\log_{10}P(k)$', fontsize=14)
ax.set_xlabel(r'$k\,[k_{\mathrm{f}}]$', fontsize=14)

m = 19
# print(P_JJ[1:8])
ax.tick_params(axis='both', which='both', direction='in')
# ax.scatter(k[:m], P_JJ[:m], s=50, c='b', label=r'from fit to $\langle\tau\rangle$')
# ax.scatter(k[:m], P_JJ_corr[:m], s=20, c='k', label='Spatial Corr')

ax.scatter(k[:m], P13[:m], s=20, c='k', label='Spatial Corr')


# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('both')
# ax.set_ylim(-15, 1)
ax.legend(fontsize=12)#, loc='lower left')#, loc=2, bbox_to_anchor=(1,1))
ax.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=False, top=False)
# ax.set_xlim(0.5, 16.5)

# plt.savefig(f'../plots/paper_plots_final/k_dep/.png', bbox_inches='tight', dpi=300)
# plt.close()
plt.show()
