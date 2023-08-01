#!/usr/bin/env python3

#import libraries
import pandas
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from functions import read_sim_data, param_calc_ens, tau_ext
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from tqdm import tqdm

def alpha_c_function(Nfiles, Lambda, path, mode, kind, n_runs=8, n_use=8, H0=100, folder_name=''):
    print('\npath = {}'.format(path))
    #define lists to store the data
    a_list = np.zeros(Nfiles)

    #An and Bn for the integral over the Green's function
    An_F3P = np.zeros(Nfiles)
    Bn_F3P = np.zeros(Nfiles)
    Pn_F3P = np.zeros(Nfiles)
    Qn_F3P = np.zeros(Nfiles)

    An_F6P = np.zeros(Nfiles)
    Bn_F6P = np.zeros(Nfiles)
    Pn_F6P = np.zeros(Nfiles)
    Qn_F6P = np.zeros(Nfiles)

    An_MW = np.zeros(Nfiles)
    Bn_MW = np.zeros(Nfiles)
    Pn_MW = np.zeros(Nfiles)
    Qn_MW = np.zeros(Nfiles)

    An_SC = np.zeros(Nfiles)
    Bn_SC = np.zeros(Nfiles)
    Pn_SC = np.zeros(Nfiles)
    Qn_SC = np.zeros(Nfiles)

    An_SCD = np.zeros(Nfiles)
    Bn_SCD = np.zeros(Nfiles)
    Pn_SCD = np.zeros(Nfiles)
    Qn_SCD = np.zeros(Nfiles)

    AJ_F3P = np.zeros(Nfiles)
    BJ_F3P = np.zeros(Nfiles)
    PJ_F3P = np.zeros(Nfiles)
    QJ_F3P = np.zeros(Nfiles)

    AJ_F6P = np.zeros(Nfiles)
    BJ_F6P = np.zeros(Nfiles)
    PJ_F6P = np.zeros(Nfiles)
    QJ_F6P = np.zeros(Nfiles)

    AJ_SC = np.zeros(Nfiles)
    BJ_SC = np.zeros(Nfiles)
    PJ_SC = np.zeros(Nfiles)
    QJ_SC = np.zeros(Nfiles)
    
    ctot2_list = np.zeros(Nfiles)


    #the densitites
    P_nb = np.zeros(Nfiles)
    P_lin = np.zeros(Nfiles)
    P_1l_tr = np.zeros(Nfiles)
    dk_lin = np.zeros(Nfiles, dtype=complex)

    sol = tau_ext(4, Lambda, path, mode, kind, folder_name)
    a1, ctot2_1 = sol[0], sol[-1]

    sol = tau_ext(5, Lambda, path, mode, kind, folder_name)
    a2, ctot2_2 = sol[0], sol[-1]
    slope = (ctot2_2-ctot2_1) / (a2-a1)

    a_list = np.array([np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j)) for j in range(Nfiles)])
    C_P = 4*slope*(a_list[0]**(9/2)) / (45 * (100**2) * a_list**(5/2))
    C_Q = (slope*a_list[0]**2 / (5*100**2)) * np.ones(a_list.size)
    alpha_c_0 = C_P - C_Q #this is the part of alpha_c integrated from 0 to a0. Add this to the integral from a0 to a

    a_list = np.array([np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j)) for j in range(Nfiles)])


    #initial scalefactor
    a0 = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(0))
    q = np.genfromtxt(path + 'output_{0:04d}.txt'.format(0))[:,0]

    for file_num in tqdm(range(Nfiles)):
        # filename = '/output_hierarchy_{0:03d}.txt'.format(file_num)
        #the function 'EFT_solve' return solutions of all modes + the EFT parameters
        ##the following line is to keep track of 'a' for the numerical integration
        if file_num > 0:
            a0 = a

        a, x, d1k, P_nb_a, P_1l_a_tr, cs2_F3P, cv2_F3P, cs2_F6P, cv2_F6P, ctot2_F3P, ctot2_F6P, ctot2_MW, ctot2_SC, ctot2_SCD,\
             dc_l, dv_l, tau_l_0, tau_l, fit_F3P, fit_F6P, fit_SC = param_calc_ens(file_num, Lambda, path, mode, kind, n_runs, n_use, folder_name=folder_name)

        ctot2_list[file_num] = ctot2_MW

        Nx = x.size
        k = np.fft.ifftshift(2.0 * np.pi * np.arange(-Nx/2, Nx/2))
        P_nb[file_num] = P_nb_a[mode]
        P_1l_tr[file_num] = P_1l_a_tr[mode]
        P_lin[file_num] = (np.abs(d1k**2)*a**2)[mode]
        dk_lin[file_num] = d1k[mode] * a

        J_F3P = np.real(np.fft.fft(tau_l - fit_F3P)[mode]) / tau_l.size
        J_F6P = np.real(np.fft.fft(tau_l - fit_F6P)[mode]) / tau_l.size
        J_SC = np.real(np.fft.fft(tau_l_0 - fit_SC)[mode]) / tau_l.size


        ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
        if file_num > 0:
            da = a - a0

            #F3P
            Pn_F3P[file_num] = ctot2_F3P * (a**(5/2)) #for calculation of alpha_c
            Qn_F3P[file_num] = ctot2_F3P

            #F6P
            Pn_F6P[file_num] = ctot2_F6P * (a**(5/2)) #for calculation of alpha_c
            Qn_F6P[file_num] = ctot2_F6P

            #M&W
            Pn_MW[file_num] = ctot2_MW * (a**(5/2)) #for calculation of alpha_c
            Qn_MW[file_num] = ctot2_MW

            #SC
            Pn_SC[file_num] = ctot2_SC * (a**(5/2)) #for calculation of alpha_c
            Qn_SC[file_num] = ctot2_SC

            #SCD
            Pn_SCD[file_num] = ctot2_SCD * (a**(5/2)) #for calculation of alpha_c
            Qn_SCD[file_num] = ctot2_SCD

            #Stoch F3P
            PJ_F3P[file_num] = J_F3P * (a**(3/2)) #for calculation of alpha_c
            QJ_F3P[file_num] = J_F3P / a

            #Stoch F6P
            PJ_F6P[file_num] = J_F6P * (a**(3/2)) #for calculation of alpha_c
            QJ_F6P[file_num] = J_F6P / a

            #Stoch SC
            PJ_SC[file_num] = J_SC * (a**(3/2)) #for calculation of alpha_c
            QJ_SC[file_num] = J_SC / a

    #A second loop for the integration
    for j in range(1, Nfiles):
        An_F3P[j] = np.trapz(Pn_F3P[:j], a_list[:j])
        Bn_F3P[j] = np.trapz(Qn_F3P[:j], a_list[:j])

        An_F6P[j] = np.trapz(Pn_F6P[:j], a_list[:j])
        Bn_F6P[j] = np.trapz(Qn_F6P[:j], a_list[:j])

        An_MW[j] = np.trapz(Pn_MW[:j], a_list[:j])
        Bn_MW[j] = np.trapz(Qn_MW[:j], a_list[:j])

        An_SC[j] = np.trapz(Pn_SC[:j], a_list[:j])
        Bn_SC[j] = np.trapz(Qn_SC[:j], a_list[:j])

        An_SCD[j] = np.trapz(Pn_SCD[:j], a_list[:j])
        Bn_SCD[j] = np.trapz(Qn_SCD[:j], a_list[:j])

        AJ_F3P[j] = np.trapz(PJ_F3P[:j], a_list[:j])
        BJ_F3P[j] = np.trapz(QJ_F3P[:j], a_list[:j])

        AJ_F6P[j] = np.trapz(PJ_F6P[:j], a_list[:j])
        BJ_F6P[j] = np.trapz(QJ_F6P[:j], a_list[:j])

        AJ_SC[j] = np.trapz(PJ_SC[:j], a_list[:j])
        BJ_SC[j] = np.trapz(QJ_SC[:j], a_list[:j])

    #calculation of the Green's function integral
    C = 2 / (5 * H0**2)

    An_F3P /= (a_list**(5/2))
    An_F6P /= (a_list**(5/2))
    An_MW /= (a_list**(5/2))
    An_SC /= (a_list**(5/2))
    An_SCD /= (a_list**(5/2))

    AJ_F3P /= (a_list**(3/2))
    AJ_F6P /= (a_list**(3/2))
    AJ_SC /= (a_list**(3/2))

    BJ_F3P *= a_list
    BJ_F6P *= a_list
    BJ_SC *= a_list

    del_J_F3P = C * (AJ_F3P - BJ_F3P)
    del_J_F6P = C * (AJ_F6P - BJ_F6P)
    del_J_SC = C * (AJ_SC - BJ_SC)
    
    alpha_c_true = k[mode]**2 * ((P_nb - P_1l_tr) / (2 * P_lin * k[mode]**2))
    alpha_c_F3P = k[mode]**2 * ((C * (An_F3P - Bn_F3P)) + alpha_c_0)
    alpha_c_F6P = k[mode]**2 * ((C * (An_F6P - Bn_F6P)) + alpha_c_0)
    alpha_c_MW = k[mode]**2 * ((C * (An_MW - Bn_MW)) + alpha_c_0)
    alpha_c_SC = k[mode]**2 * ((C * (An_SC - Bn_SC)) + alpha_c_0)
    alpha_c_SCD = k[mode]**2 * ((C * (An_SCD - Bn_SCD)) + alpha_c_0)
    alpha_c_pred = -2*slope*a_list**2 / (9 * 100**2)

    return a_list, alpha_c_true, alpha_c_F3P, alpha_c_F6P, alpha_c_MW, alpha_c_SC, alpha_c_SCD, del_J_F3P, del_J_F6P, del_J_SC, alpha_c_pred


path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# path = 'cosmo_sim_1d/sim_k_1_7/run1/'

Lambda_int = 13
Lambda = Lambda_int * (2 * np.pi)
mode = 1
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

Nfiles = 51
folder_name = '/new_hier/data_{}/L{}/'.format(kind, Lambda_int)
plots_folder = '/paper_plots_final/'

folder_name = '/new_hier/data_{}/L{}'.format(kind, int(Lambda/(2*np.pi)))
a_list, alpha_c_true, alpha_c_F3P, alpha_c_F6P,\
     alpha_c_MW, alpha_c_SC, alpha_c_SCD, del_J_F3P, del_J_F6P, del_J_SC, alpha_c_pred = alpha_c_function(Nfiles, Lambda, path, mode, kind, folder_name=folder_name)


df = pandas.DataFrame(data=[a_list, alpha_c_true, alpha_c_F3P, alpha_c_F6P, alpha_c_MW, alpha_c_SC, alpha_c_SCD, del_J_F3P, del_J_F6P, del_J_SC, alpha_c_pred])
file = open(f"./{path}/alpha_c_{kind}_{Lambda_int}.p", "wb")
pickle.dump(df, file)
file.close()