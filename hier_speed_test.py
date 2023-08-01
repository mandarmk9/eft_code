#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from functions import is_between, hier_calc

j = 20
dx_grid = 0.01
path = 'cosmo_sim_1d/sim_k_1_11/run1/'
a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))

moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
moments_file = np.genfromtxt(path + moments_filename)
a = moments_file[:,-1][0]
x_old = moments_file[:,0]
M0_old = moments_file[:,2]
M1_old = moments_file[:,4]
C1_old = moments_file[:,5]
M2_old = moments_file[:,6]
C2_old = moments_file[:,7]

x_grid, M0, M1, M2, C0, C1, C2 = hier_calc(j, path, dx_grid)
# nbody_filename = 'output_{0:04d}.txt'.format(j)
# nbody_file = np.genfromtxt(path + nbody_filename)
# x_nbody = nbody_file[:,-1]
# v_nbody = nbody_file[:,2]
# off = 0#-dx_grid/2
# x_grid = np.arange(0-off, 1-off, dx_grid)
#
# M0 = np.zeros(x_grid.size)
# C1_nbody = np.zeros(x_grid.size)
# C2_nbody = np.zeros(x_grid.size)
#
# for j in range(x_grid.size):
#     if j == x_grid.size-1:
#         s = is_between(x_nbody, x_grid[0]+off, x_grid[1]+off)
#     else:
#         s = is_between(x_nbody, x_grid[j]+off, x_grid[j+1]+off)
#     vels = v_nbody[s[0]]
#     M0[j] = s[0].size
#     C1_nbody[j] = sum(vels) / len(vels)
#     C2_nbody[j] = sum(vels**2) / len(vels)
#
#
# M0_nbody = M0 / np.mean(M0)
#
# M0 = [[] for j in range(x_grid.size)]
# C1 = [[] for j in range(x_grid.size)]
#
# for m in range(x_nbody.size):
#     p = round(x_nbody[m]/dx_grid)
#     if p == x_grid.size:
#         p = 0
#     else:
#         pass
#     M0[p].append(m)
#     C1[p].append(v_nbody[m])
#
#
# M0 = [len(M0[j]) for j in range(len(M0))]
# C2 = [sum(np.array(C1[j])**2)/len(C1[j]) for j in range(len(C1))]
# C1 = [sum(C1[j])/len(C1[j]) for j in range(len(C1))]
# C1[-1] = C1[0]
# M0 /= np.mean(M0)


plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
ax.set_title(r'$a = {}$'.format(a), fontsize=16)
ax.set_xlabel(r'$x/L$', fontsize=14)
ax.set_ylabel(r'$1+\delta_{l}$', fontsize=14)

# ax.plot(x_grid, C1_nbody, c='k', lw=2, label='slow')
ax.plot(x_grid, C1, c='r', ls='dashed', lw=2, label='fast')
ax.plot(x_old, C1_old, c='b', ls='dotted', lw=2, label='old')

# ax.plot(x_grid, M0_nbody, c='k', lw=2, label='slow')
# ax.plot(x_grid, M0, c='r', ls='dashed', lw=2, label='fast')
# ax.plot(x_old, M0_old, c='b', ls='dotted', lw=2, label='old')

# ax.plot(x_grid, C2_nbody, c='k', lw=2, label='slow')
# ax.plot(x_grid, C2, c='r', ls='dashed', lw=2, label='fast')
# ax.plot(x_old, C2_old, c='b', ls='dotted', lw=2, label='old')


ax.legend(fontsize=12)
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.ticklabel_format(scilimits=(-2, 3))
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('both')
ax.minorticks_on()

plt.show()
