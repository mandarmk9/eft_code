#!/usr/bin/env python3
import time
import numpy as np
import multiprocessing as mp
from functions import write_sim_data

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
A = []

# path = 'cosmo_sim_1d/sim_k_1/run1/'
# A = [-0.05, 1, 0, 11]

# path = 'cosmo_sim_1d/multi_k_sim/run1/'
# A = []

# path = 'cosmo_sim_1d/multi_sim_3_15_33/run1/'
# A = []

n_runs = 8
mode = 1
kinds = ['sharp', 'gaussian']
kind_txts = ['sharp cutoff', 'Gaussian smoothing']
# which = 0

# read_folder_name = '/hierarchy_coarse/'
# write_folder_name = '/new_hier_coarse/'

# read_folder_name = '/hierarchy_even_coarser/'
# write_folder_name = '/data_even_coarser/'

read_folder_name =  '' #'/hierarchy/'
write_folder_name = '/new_hier/'

# read_folder_name = '/hierarchy_new/'
# write_folder_name = '/new_hier/'


# read_folder_name = '/hierarchy_old/'
# write_folder_name = '/new_hier_old/'


# read_folder_name = '/test/'
# write_folder_name = '/test_hier/'


tmp_st = time.time()
# n = 5
zero, Nfiles = 0, 51
def write(j, Lambda, path, A, kind, mode, run, read_folder_name='', write_folder_name=''):
    path = path[:-2] + '{}/'.format(run)
    write_sim_data(j, Lambda, path, A, kind, mode, read_folder_name, write_folder_name)

for which in range(1, 2):
    kind = kinds[which]
    kind_txt = kind_txts[which]
    for Lambda in range(10, 13):
        print('\nLambda = {} ({})'.format(Lambda, kind_txt))
        folder_name = write_folder_name + '/data_{}/L{}'.format(kind, Lambda)
        Lambda *= (2*np.pi)
        for j in range(zero, Nfiles):#, Nfiles):
            print('Writing {} of {}'.format(j+1, Nfiles))
            tasks = []
            for run in range(1,9):
                p = mp.Process(target=write, args=(j, Lambda, path, A, kind, mode, run, read_folder_name, folder_name,))
                tasks.append(p)
                p.start()
            for task in tasks:
                p.join()
tmp_end = time.time()
print('multiprocessing takes {}s'.format(np.round(tmp_end-tmp_st, 3)))
