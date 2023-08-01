#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from EFT_hier_solver import EFT_solve

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
j = 14
kind = 'sharp'
Lambda = 15 * (2*np.pi)
folder_name = '/hierarchy_even_coarser/'

a, x, d1k, dc_l, dv_l, tau_l, P_nb_a, P_1l_a_tr = EFT_solve(j, Lambda, path, kind, folder_name)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
ax.set_title(r'$a = {}$'.format(a), fontsize=16)
ax.set_xlabel(r'$x/L$', fontsize=14)
ylabel = r"$[\tau]_{\Lambda}$"

ax.set_ylabel(ylabel, fontsize=14)
ax.plot(x, -dv_l, c='k', lw=2)
ax.plot(x, dc_l*(100/np.sqrt(a)), c='b', lw=2)


# ax.legend(fontsize=12)
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.ticklabel_format(scilimits=(-2, 3))
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('both')
ax.minorticks_on()
plt.show()
# plt.savefig('../plots/test/new_paper_plots/new_hier/{}{}_{}.png'.format(MorC, nM, j), bbox_inches='tight', dpi=150)
# plt.close()
