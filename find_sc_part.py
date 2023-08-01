#!/usr/bin/env python3
"""A script for reading and plotting snapshots from cosmo_sim_1d"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from tqdm import tqdm
from functions import hier_calc
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

num = 11
path = f'cosmo_sim_1d/sim_k_1_{num}/run1/'


filename = path + '/particle_data.hdf5'
file = h5py.File(filename, mode='r')
a_list = np.array(file['/Scalefactors']['a'])
eul_coo = file['/Eulerian Coordinate']
lag_coo = file['/Lagrangian Coordinate']
vel = file['/Velocities']


def hier_sc_cont(j, path, dx_grid, mask):
    nbody_filename = 'output_{0:04d}.txt'.format(j)
    nbody_file = np.genfromtxt(path + nbody_filename)
    x_nbody = nbody_file[:,-1]
    v_nbody = nbody_file[:,2]

    x_grid = np.arange(0, 1, dx_grid)
    M0 = [[] for j in range(x_grid.size)]
    C1 = [[] for j in range(x_grid.size)]
    mask_field = [[] for j in range(x_grid.size)]
    count = 0
    for m in range(x_nbody.size):
        p = int(round(x_nbody[m]/dx_grid))
        if p == x_grid.size:
            p = 0
        else:
            pass
        if mask[m] == 1:
            M0[p].append(m)
            C1[p].append(v_nbody[m])
            mask_field[p].append(1)
            count += 1

    M0 = [len(M0[j]) for j in range(len(M0))]
    C2 = np.array([sum(np.array(C1[j])**2)/len(C1[j]) for j in range(len(C1))])
    C1 = [sum(C1[j])/len(C1[j]) for j in range(len(C1))]

    M0 /= np.mean(M0)
    M0 *= (count / x_nbody.size)
    M1 = M0 * C1
    M2 = C2 * M0
    C0 = M0
    # print(count, count/x_nbody.size)
    return x_grid, M0, M1, M2, C0, C1, C2, count/x_nbody.size
    
dx_grid = 0.001
folder_name = 'shell_crossed_hier/'

for j in tqdm(range(51)):
    # j = 12
    x = np.array(eul_coo[str(j)])
    q = np.array(lag_coo[str(j)])
    v = np.array(vel[str(j)])
    mask = np.array([int((x[i+1] > x[i])) for i in range(x.size-1)])
    mask = np.append(mask, 1)
    x_grid, M0, M1, M2, C0, C1, C2, frac = hier_sc_cont(j, path, dx_grid, mask)
    # x_grid_all, M0_all, M1_all, M2_all, C0_all, C1_all, C2_all = hier_calc(j, path, dx_grid)

    # plt.plot(x_grid, C2_all-C2, c='b')
    
    # plt.show()

    filepath = path + '/{}/'.format(folder_name)
    try:
        os.makedirs(filepath, 0o755)
    except:
        pass

    filename = 'hier_{0:04d}.hdf5'.format(j)
    file = h5py.File(filepath+filename, 'w')
    file.create_group('Header')
    header = file['Header']
    a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
    header.attrs.create('a', a, dtype=float)
    header.attrs.create('dx', dx_grid, dtype=float)
    header.attrs.create('frac', frac, dtype=float)

    moments = file.create_group('Moments')
    moments.create_dataset('M0', data=M0)
    moments.create_dataset('M1', data=M1)
    moments.create_dataset('M2', data=M2)
    moments.create_dataset('C0', data=C0)
    moments.create_dataset('C1', data=C1)
    moments.create_dataset('C2', data=C2)

    file.close()


