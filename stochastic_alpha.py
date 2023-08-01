#!/usr/bin/env python3

#import libraries
import pandas
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from functions import read_sim_data, param_calc_ens
from alpha_c_function import alpha_c_function
from tqdm import tqdm
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda_int = 3

kind = 'sharp'
kind_txt = 'sharp cutoff'
kind = 'gaussian'
kind_txt = 'Gaussian smoothing'
Nfiles = 51

folder_name = '/new_hier/data_{}/L{}/'.format(kind, Lambda_int)

# for Lambda_int in [3, 4, 5, 6, 7, 8, 9]:
for Lambda_int in [10, 11, 12]:#, 11, 12]:
    folder_name = f'/new_hier/data_{kind}/L{Lambda_int}/'
    Lambda = Lambda_int * (2 * np.pi)
    for mode in tqdm(range(0, 20)):
        a_list, alpha_c_true, alpha_c_F3P, alpha_c_F6P, alpha_c_MW,\
            alpha_c_SC, alpha_c_SCD, del_J_F3P, del_J_F6P, del_J_SC, alpha_c_pred = alpha_c_function(Nfiles, Lambda, path, mode, kind, n_runs=8, n_use=10, H0=100, folder_name=folder_name)

        # df = pandas.DataFrame(data=[a_list, del_J_F3P, del_J_F6P, del_J_SC])
        # file = open(f"./{path}/stoch_del_{kind}_{Lambda_int}_{mode}.p", "wb")
        # pickle.dump(df, file)
        # file.close()

        df = pandas.DataFrame(data=[a_list, alpha_c_SC, alpha_c_F3P])
        file = open(f"./{path}/alphas_{kind}_{Lambda_int}_{mode}.p", "wb")
        pickle.dump(df, file)
        file.close()
