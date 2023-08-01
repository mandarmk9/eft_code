#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt

from functions import *
from zel import eulerian_sampling

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
path_k1 = 'cosmo_sim_1d/sim_k_1/run1/'
A = [-0.05, 1, -0.0, 11]
Lambda_int = 3
Lambda = Lambda_int * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'

folder_name = '/hierarchy_new/'
def extract_sm_fields(path, file_num, Lambda, kind, sm=True):
    a, dx, M0_nbody, M1_nbody, M2_nbody, C0_nbody, C1_nbody, C2_nbody = read_hier(path, file_num, folder_name)
    x = np.arange(0, 1, dx)
    dc = M0_nbody #dc is 1+\delta
    v = C1_nbody
    k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))

    if sm == True:
        dc = smoothing(dc, k, Lambda, kind)
        v = smoothing(v, k, Lambda, kind)

    dc_k = np.fft.fft(dc)
    dc_k[2:-1] = 0
    dc = np.real(np.fft.ifft(dc_k))
    return a, x, dc, v

a_list, ratio_list = [], []
for j in range(22):
    # if j != 23:
    a_0, x, dc_0, v_0 = extract_sm_fields(path, j, Lambda, kind)
    a_0, x_k1, dc_k1_0, v_k1_0 = extract_sm_fields(path_k1, j, Lambda, kind)#, sm=False)
    q1 = np.real(np.fft.fft(dc_0)[1]) / dc_0.size
    q2 = np.real(np.fft.fft(dc_k1_0)[1]) / dc_0.size
    ratio = q1 / q2 #/ np.real(np.fft.fft(dc_0)[1])
    ratio_list.append(ratio)
    a_list.append(a_0)
    print(a_0, q1, q2)
fig, ax = plt.subplots()
ax.plot(a_list, ratio_list)
plt.savefig('../plots/test/new_paper_plots/test_0.png', dpi=300)
plt.close()