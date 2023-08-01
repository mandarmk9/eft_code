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
flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')


filename = path + '/particle_data.hdf5'
file = h5py.File(filename, mode='r')
a_list = np.array(file['/Scalefactors']['a'])
eul_coo = file['/Eulerian Coordinate']
lag_coo = file['/Lagrangian Coordinate']
vel = file['/Velocities']

dx_grid = 0.001
folder_name = 'shell_crossed_hier/'
Nfiles = 51

count = 0
count_list = [0]
for j in tqdm(range(1, Nfiles)):
    if j <= 35:
        x_old = np.array(eul_coo[str(j-1)])
        mask_old = np.array([int((x_old[i+1] > x_old[i])) for i in range(x_old.size-1)])
        mask_old = np.append(mask_old, 1)

        x_new = np.array(eul_coo[str(j)])
        mask_new = np.array([int((x_new[i+1] > x_new[i])) for i in range(x_new.size-1)])
        mask_new = np.append(mask_new, 1)

    else:
        x_old = np.array(eul_coo[str(j-1)])
        mask_old = np.array([int((x_old[i+1] < x_old[i])) for i in range(x_old.size-1)])
        mask_old = np.append(mask_old, 1)

        x_new = np.array(eul_coo[str(j)])
        mask_new = np.array([int((x_new[i+1] < x_new[i])) for i in range(x_new.size-1)])
        mask_new = np.append(mask_new, 1)


    count += sum(mask_old - mask_new)
    count_list.append(count)

count_list = np.array(count_list) / x_new.size
np.savetxt(fname=path+'/count_fraction.txt', X=count_list, newline='\n')

count = np.loadtxt(fname=path+'/count_fraction.txt', delimiter='\n')

fig, ax = plt.subplots()
ax.plot(a_list, count_list, c='b', lw=2)
for j in range(Nfiles):
    if flags[j] == 1:
        ax.axvline(a_list[j], lw=0.5, c='teal', ls='dashed')
plt.show()


# x_grid, M0, M1, M2, C0, C1, C2, frac = hier_sc_cont(j, path, dx_grid, mask)
# # x_grid_all, M0_all, M1_all, M2_all, C0_all, C1_all, C2_all = hier_calc(j, path, dx_grid)

#     # plt.plot(x_grid, C2_all-C2, c='b')
    
#     # plt.show()

#     filepath = path + '/{}/'.format(folder_name)
#     try:
#         os.makedirs(filepath, 0o755)
#     except:
#         pass

#     filename = 'hier_{0:04d}.hdf5'.format(j)
#     file = h5py.File(filepath+filename, 'w')
#     file.create_group('Header')
#     header = file['Header']
#     a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
#     header.attrs.create('a', a, dtype=float)
#     header.attrs.create('dx', dx_grid, dtype=float)
#     header.attrs.create('frac', frac, dtype=float)

#     moments = file.create_group('Moments')
#     moments.create_dataset('M0', data=M0)
#     moments.create_dataset('M1', data=M1)
#     moments.create_dataset('M2', data=M2)
#     moments.create_dataset('C0', data=C0)
#     moments.create_dataset('C1', data=C1)
#     moments.create_dataset('C2', data=C2)

#     file.close()


