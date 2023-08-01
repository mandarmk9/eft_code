#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import h5py
from functions import is_between
from scipy.interpolate import interp1d

def hier_calc(j, path, dx_grid):
    print('\nCalculating moment hierarchy...')
    nbody_filename = 'output_{0:04d}.txt'.format(j)
    nbody_file = np.genfromtxt(path + nbody_filename)
    x_nbody = nbody_file[:,-1]
    v_nbody = nbody_file[:,2]

    par_num = x_nbody.size
    L = 1.0
    x_grid = np.arange(0, L, dx_grid)
    N = x_grid.size
    k_grid = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-N/2, N/2))
    x_grid = np.arange(0+dx/2, 1+dx/2, dx_grid)

    M0 = np.zeros(x_grid.size)
    M1 = np.zeros(x_grid.size)
    M2 = np.zeros(x_grid.size)
    C1 = np.zeros(x_grid.size)
    C2 = np.zeros(x_grid.size)
    for j in range(x_grid.size):
        if j == x_grid.size-1:
            s = is_between(x_nbody, x_grid[0]-dx/2, x_grid[1]-dx/2)
        else:
            s = is_between(x_nbody, x_grid[j]-dx/2, x_grid[j+1]-dx/2)
        vels = v_nbody[s[0]]
        M0[j] = s[0].size
        C1[j] = sum(vels) / len(vels)
        C2[j] = sum(vels**2) / len(vels)

    M0 /= np.mean(M0)
    M1 = M0 * C1

    M2 = C2 * M0
    C0 = M0
    return x_grid, M0, M1, M2, C0, C1, C2

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
j = 0
dx = 0.01
a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))


x, M0, M1, M2, C0, C1, C2 = hier_calc(j, path, dx)

filename = './hier_test/hier_{0:04d}.hdf5'.format(j)
file = h5py.File(filename, 'w')
file.create_group('Header')
header = file['Header']
a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
print('a = ', a)
header.attrs.create('a', a, dtype=float)
header.attrs.create('dx', dx, dtype=float)
moments = file.create_group('Moments')
moments.create_dataset('M0', data=M0)
moments.create_dataset('M1', data=M1)
moments.create_dataset('M2', data=M2)
moments.create_dataset('C0', data=C0)
moments.create_dataset('C1', data=C1)
moments.create_dataset('C2', data=C2)
file.close()


filename = './hier_test/hier_{0:04d}.hdf5'.format(j)
file = h5py.File(filename, mode='r')
header = file['/Header']
a = header.attrs.get('a')
dx = header.attrs.get('dx')

moments = file['/Moments']
mom_keys = list(moments.keys())
C0 = np.array(moments[mom_keys[0]])
C1 = np.array(moments[mom_keys[1]])
C2 = np.array(moments[mom_keys[2]])
M0 = np.array(moments[mom_keys[3]])
M1 = np.array(moments[mom_keys[4]])
M2 = np.array(moments[mom_keys[5]])
file.close()

# C2 -= C1**2

nbody_filename = 'output_{0:04d}.txt'.format(j)
nbody_file = np.genfromtxt(path + nbody_filename)
x_nbody = nbody_file[:,-1]
v_nbody = nbody_file[:,2]

moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
moments_file = np.genfromtxt(path + moments_filename)
a = moments_file[:,-1][0]
x_cell = moments_file[:,0]
M0_nbody = moments_file[:,2]
M1_nbody = moments_file[:,4]
M2_nbody = moments_file[:,6]

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
ax.set_title(r'$a = {}$'.format(a), fontsize=16)

# arr = M0

# ax.plot(x, arr, c='k', lw=2)
# ax.plot(x, np.flip(arr), c='r', lw=2, ls='dashed')
# ax.plot(x, arr-np.flip(arr), c='r', lw=2, ls='dashed')

# ax.plot(x, C2, c='k', lw=2)
ax.plot(x_cell, M0_nbody, c='r', lw=2)
ax.plot(x, M0, c='b', ls='dashed', lw=2)

print(x.size, M0.size)

# ax.set_xlim(0.995, 1.002)
ax.legend(fontsize=12)
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.ticklabel_format(scilimits=(-2, 3))
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('both')
ax.minorticks_on()

plt.show()
