#!/usr/bin/env python3

#import libraries
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle
from functions import read_density, SPT, dc_in_finder, write_density

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
path_k1 = 'cosmo_sim_1d/sim_k_1/run1/'
file_num = 0

# write_density(path_k1, 0, 51, 0.001)

def P_nb_calc(path, j):
   dk_par, a, dx = read_density(path, j)
   x = np.arange(0, 1.0, dx)
   M0_par = np.real(np.fft.ifft(dk_par))
   M0_par = (M0_par / np.mean(M0_par)) - 1
   M0_k = np.fft.fft(M0_par) / M0_par.size
   P_nb_a = np.real(M0_k * np.conj(M0_k))
   k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2)) / (2*np.pi)
   return a, P_nb_a, k, np.real(M0_k[1])

a, P_nb, k, dc_kf = P_nb_calc(path, file_num)
x = np.arange(0, 1, 1/k.size)
dc_in, k_1l = dc_in_finder(path, x, interp=True)
_, _, P_1l, P_2l = SPT(dc_in, 1.0, a)
a, P_nb_k1, k, dc_kf_k1 = P_nb_calc(path_k1, file_num)
dc_in, k_1l = dc_in_finder(path_k1, x, interp=True)
_, _, P_1l_k1, P_2l_k1 = SPT(dc_in, 1.0, a)

# print(np.abs(dc_kf - dc_kf_k1) / dc_kf)

k_1l /= (2*np.pi)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
ax.set_title(r'$a = {}$'.format(a), fontsize=18)
ax.set_ylabel(r'$\log_{10}[L^{-1}P(k)]$', fontsize=16)
ax.set_xlabel(r'$k/k_{\mathrm{f}}$', fontsize=16)

diff_nb = np.abs(P_nb_k1 - P_nb) / P_nb
P = np.log10(np.abs(P_nb))
P_k1 = np.log10(np.abs(P_nb_k1))

# P = (P_nb)
# P_k1 = (P_nb_k1)

diff_spt = np.abs(np.abs(P_1l[:7]) - P_nb[:7]) / P_nb[:7]

print(diff_spt)
diff_spt_k1 = np.abs(np.abs(P_1l_k1[:7]) - P_nb[:7]) / P_nb[:7]
diff_spt_k1_all = np.abs(np.abs(P_1l_k1[:7]) - P_nb_k1[:7]) / P_nb_k1[:7]

# print((P_1l_k1[:14]))

ax.scatter(k[:16], P[:16], label=r'\texttt{sim\_1\_11}; $N$-body', s=25, c='b', marker=MarkerStyle('o', fillstyle='full'))
ax.scatter(k[:16], P_k1[:16], label=r'\texttt{sim\_1}; $N$-body', facecolors='none', edgecolor='r', marker=MarkerStyle('o', fillstyle='none'), s=25)

ax.scatter(k_1l[:16], np.log10(np.abs(P_1l[:16])), label=r'\texttt{sim\_1\_11}; SPT-4', s=25, c='seagreen', marker=MarkerStyle('d', fillstyle='full'))

# ax.scatter(k_1l[:16], np.log10(np.abs(P_1l_k1[:16])), label=r'\texttt{sim\_1}; SPT-4', s=25, c='seagreen', marker=MarkerStyle('d', fillstyle='none'))

# ax.scatter(k_1l, P_1l, label=r'\texttt{sim\_1\_11}; SPT-4', s=25, c='seagreen', marker='v')
# ax.scatter(k_1l, P_1l_k1, label=r'\texttt{sim\_1}; SPT-4', s=25, c='r', marker='+')

ax.tick_params(axis='both', which='both', direction='in')

# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('both')
ax.set_ylim(-20.675, 2.5)
ax.legend(fontsize=12, loc='lower right', framealpha=1)#, loc=2, bbox_to_anchor=(1,1))
ax.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=False, top=False)
ax.set_xlim(0.5, 16.5)
ax.tick_params(axis='both', which='major', labelsize=13)

# plt.show()

# text_str = r'$\Delta P(k) = \frac{\left|P_{\texttt{\footnotesize sim\_1}} - P_{\texttt{\footnotesize sim\_1\_11}}\right|}{P_{\texttt{\footnotesize sim\_1\_11}}}$'

# # ax.text(1.01, 0.75, text_str, transform=ax.transAxes, fontsize=10)
# diff[0] = 0

left, bottom, width, height = [0.24, 0.25, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.set_xlim(0.5, 3.5)
ax2.yaxis.set_ticks_position('both')
ax2.tick_params(axis='both', which='both', direction='in')
ax2.tick_params(axis='y', which='minor', bottom=False)
ax2.set_ylim(0, 0.11)
ax2.set_xlabel(r'$k/k_{\mathrm{f}}$', fontsize=15)
ax2.set_ylabel(r'$\frac{\Delta P}{P}$', fontsize=15)
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False, top=False)
# ax2.yaxis.set_label_position('right')

ax2.scatter(k, diff_nb, facecolors='none', edgecolor='r', marker=MarkerStyle('o', fillstyle='none'), s=25)
ax2.scatter(k[:7], diff_spt, c='seagreen', marker=MarkerStyle('d', fillstyle='full'), s=25)
# ax2.scatter(k[:7], diff_spt_k1, c='seagreen', marker=MarkerStyle('d', fillstyle='none'), s=25)

ax2.tick_params(axis='both', which='major', labelsize=11)
# print(diff_spt)
# print(diff_spt_k1_all)

# diff_spt_k1 = np.abs(np.abs(P_1l_k1[:7]) - P_nb_k1[:7]) / P_nb_k1[:7]

# print(diff_spt[1:5]*100, diff_spt_k1[1:5]*100)

# print('N-body relative difference: ', diff_nb[1])
# plt.show()
plt.savefig(f'../plots/paper_plots_final/PS_{file_num}.pdf', bbox_inches='tight', dpi=300)
plt.close()

# print('c = ', c)
# print('n = ', n)
