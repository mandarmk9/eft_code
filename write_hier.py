#!/usr/bin/env python3
import os
import multiprocessing as mp
from functions import write_hier, write_density

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# dx = 0.01
dx = 0.0001

# #coarse: dx = 0.0005, hierarchy: dx = 0.0001, even coarser: dx = 0.001, old: 1/62500
# # folder_name = '/hierarchy_coarse/'
# # folder_name = '/hierarchy_even_coarser/'
# # folder_name = '/test_for_slurm/'
# # folder_name = '/hierarchy_old/'
folder_name = '/hierarchy/'

def write(path, dx, folder_name):
    # path = 'cosmo_sim_1d/multi_k_sim/run{}/'.format(j)
    print(path)
    write_hier(0, 51, path, dx, folder_name)


tasks = []
for j in range(0,1):
    # path = 'cosmo_sim_1d/gaussian_ICs/'
    path = f'cosmo_sim_1d/sim_k_1_11/run{j+1}/'
    # path = 'cosmo_sim_1d/multi_sim_3_15_33/run{}/'.format(j)
    p = mp.Process(target=write, args=(path, dx, folder_name,))
    tasks.append(p)
    p.start()

for task in tasks:
    p.join()


# path = 'cosmo_sim_1d/sim_k_1_11/run{}/'.format(1)
# write_density(path, 50, 51, 0.001)



# path = 'cosmo_sim_1d/sim_k_1/run1/'
# write_hier(0, 50, path, dx)

# for j in range(2, 9):
#     path = 'cosmo_sim_1d/sim_k_1_7/run{}/'.format(j)
#     print(path)
#     write_hier(0, 50, path, dx)


# for j in range(1, 2):
#     # path = 'cosmo_sim_1d/sim_k_1/run{}/'.format(j)
#     path = 'cosmo_sim_1d/multi_k_sim/run{}/'.format(j)
#     print(path)
#     write_hier(0, 51, path, dx)
