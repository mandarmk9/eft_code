#!/usr/bin/env python3

#import libraries
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle
from functions import read_density, SPT, dc_in_finder, write_density

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
path_k1 = 'cosmo_sim_1d/sim_k_1/run1/'

def power_plot_calc(path, file_num):
   def P_nb_calc(path, j):
      dk_par, a, dx = read_density(path, j)
      x = np.arange(0, 1.0, dx)
      M0_par = np.real(np.fft.ifft(dk_par))
      M0_par = (M0_par / np.mean(M0_par)) - 1
      M0_k = np.fft.fft(M0_par) / M0_par.size
      P_nb_a = np.real(M0_k * np.conj(M0_k))
      k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2)) / (2*np.pi)
      return a, P_nb_a, k, np.real(M0_k[1])

   a, P_nb, k, _ = P_nb_calc(path, file_num)
   x = np.arange(0, 1, 1/k.size)
   dc_in, k_1l = dc_in_finder(path, x, interp=True)
   _, _, P_1l, P_2l = SPT(dc_in, 1.0, a)
   a, P_nb_k1, k, _ = P_nb_calc(path_k1, file_num)
   dc_in, k_1l = dc_in_finder(path_k1, x, interp=True)
   _, _, P_1l_k1, _ = SPT(dc_in, 1.0, a)

   diff_nb = np.abs(P_nb_k1 - P_nb) / P_nb
   P = np.log10(np.abs(P_nb))
   P_k1 = np.log10(np.abs(P_nb_k1))
   diff_spt = np.abs(np.abs(P_1l[:7]) - P_nb[:7]) / P_nb[:7]
   P_1l = np.log10(np.abs(P_1l))

   return a, k, P, P_k1, P_1l, diff_nb, diff_spt

a_0, k, P_0, P_k1_0, P_1l_0, diff_nb_0, diff_spt_0 = power_plot_calc(path, 11)
a_1, k, P_1, P_k1_1, P_1l_1, diff_nb_1, diff_spt_1 = power_plot_calc(path, 50)



plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
ax[0].set_title(rf'$a = {a_0}$', fontsize=24)
ax[1].set_title(rf'$a = {a_1}$', fontsize=24)
ax[0].scatter(k[:16], P_0[:16], s=35, c='b', marker=MarkerStyle('o', fillstyle='full'))
ax[0].scatter(k[:16], P_k1_0[:16], facecolors='none', edgecolor='r', marker=MarkerStyle('o', fillstyle='none'), s=35)
# ax[0].scatter(k[:16], P_1l_0[:16], s=35, c='seagreen', marker=MarkerStyle('d', fillstyle='full'))


ax[1].scatter(k[:16], P_1[:16], label=r'\texttt{sim\_1\_11}', s=35, c='b', marker=MarkerStyle('o', fillstyle='full'))
ax[1].scatter(k[:16], P_k1_1[:16], label=r'\texttt{sim\_1}', facecolors='none', edgecolor='r', marker=MarkerStyle('o', fillstyle='none'), s=35)
# ax[1].scatter(k[:16], P_1l_1[:16], label=r'\texttt{sim\_1\_11}; SPT-4', s=35, c='seagreen', marker=MarkerStyle('d', fillstyle='full'))
from matplotlib.ticker import MaxNLocator


# ax[0].axvline(1, c='teal', ls='dashed', lw=0.75, zorder=5)
# ax[1].axvline(1, c='teal', ls='dashed', lw=0.75, zorder=5)#label=r'$k_{\mathrm{f}}, 11\,k_{\mathrm{f}}$', zorder=1)
# ax[0].axvline(11, c='teal', ls='dashed', lw=0.75, zorder=5)
# ax[1].axvline(11, c='teal', ls='dashed', lw=0.75, zorder=5)#label=r'$11\,k_{\mathrm{f}}$', zorder=1)

ax[0].vlines(1, ymin=-30, ymax=np.log10(6.25e-4 * 1.71**2), colors=['teal'], ls='dashed', lw=0.75, zorder=5)
ax[0].vlines(11, ymin=-30, ymax=np.log10(6.25e-2 * 1.71**2), colors=['teal'], ls='dashed', lw=0.75, zorder=5)
ax[1].vlines(1, ymin=-30, ymax=np.log10(6.25e-4 * 6**2), colors=['teal'], ls='dashed', lw=0.75, zorder=5)
ax[1].vlines(11, ymin=-30, ymax=np.log10(6.25e-2 * 6**2), colors=['teal'], ls='dashed', lw=0.75, zorder=5)

# ax[1].axvline(1, c='teal', ls='dashed', lw=0.75, zorder=5)#label=r'$k_{\mathrm{f}}, 11\,k_{\mathrm{f}}$', zorder=1)
# ax[0].axvline(11, c='teal', ls='dashed', lw=0.75, zorder=5)
# ax[1].axvline(11, c='teal', ls='dashed', lw=0.75, zorder=5)#label=r'$11\,k_{\mathrm{f}}$', zorder=1)


for j in range(2):
   ax[j].set_xlabel(r'$k/k_{\mathrm{f}}$', fontsize=22)
   ax[j].tick_params(axis='both', which='both', direction='in')
   ax[j].minorticks_on()
   ax[j].yaxis.set_ticks_position('both')
   ax[j].tick_params(axis='both', which='major', labelsize=16)
   ax[j].set_ylim(-16, 0.5)
   ax[j].set_xlim(0.5, 15.5)
   ax[j].xaxis.set_major_locator(MaxNLocator(integer=True))

ax[0].set_ylabel(r'$\log_{10}[L^{-1}P(k)]$', fontsize=22)
ax[1].legend(fontsize=14, loc='lower right', framealpha=1, bbox_to_anchor=(0.15,1), ncol=1)
ax[1].tick_params(axis='both', which='both', labelleft=False)

plt.subplots_adjust(wspace=0)
ax02 = fig.add_axes([0.375, 0.25, 0.1, 0.2])

ax12 = fig.add_axes([0.6, 0.25, 0.1, 0.2])

for ax in [ax02, ax12]:
   ax.yaxis.set_ticks_position('both')
   ax.tick_params(axis='both', which='both', direction='in')
   ax.tick_params(axis='both', which='minor', bottom=False)
   # ax.tick_params(axis='x', which='minor', bottom=False, top=False)
   ax.tick_params(axis='both', which='major', labelsize=12)
   ax.set_xlim(0.5, 3.5)
   ax.set_xlabel(r'$k/k_{\mathrm{f}}$', fontsize=16)
   ax.set_ylabel(r'$\frac{\Delta P}{P}$', fontsize=16)
   ax.minorticks_on()

print(diff_nb_0[:5], diff_spt_0[:5], diff_nb_1[:5], diff_spt_1[:5])
ax02.scatter(k, diff_nb_0, facecolors='k', edgecolor='k', marker=MarkerStyle('^', fillstyle='full'), s=25)
# ax02.scatter(k[:7], diff_spt_0, c='seagreen', marker=MarkerStyle('d', fillstyle='full'), s=25)
ax12.scatter(k, diff_nb_1, facecolors='k', edgecolor='k', marker=MarkerStyle('^', fillstyle='full'), s=25)
# ax12.scatter(k[:7], diff_spt_1, c='seagreen', marker=MarkerStyle('d', fillstyle='full'), s=25)
ax02.set_ylim(0, 0.03)
ax12.set_ylim(0, 0.1)

# plt.show()
plt.savefig(f'../plots/paper_plots_final/PS.pdf', bbox_inches='tight', dpi=300)
plt.close()