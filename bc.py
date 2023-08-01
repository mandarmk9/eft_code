#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from functions import read_sim_data




kind = 'sharp'
Lambda = 3 * (2 * np.pi )
file_num = 16
dcs = []
fig, ax = plt.subplots()
for m in range(1, 9):
    path = f'cosmo_sim_1d/sim_k_1_11/run{m}/'
    a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, file_num)

    dcs.append(dc_l)

    ax.plot(x, dc_l)

Nt = len(dcs)
tau_l = sum(np.array(dcs)) / Nt
ax.plot(x, tau_l, ls='dashed')
# ax.plot(x, dcs[0], ls='dashed')

plt.show()