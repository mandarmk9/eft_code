#!/usr/bin/env python3

import os
import numpy as np
# loc = 'cosmo_sim_1d/sim_k_1_11/run1/data_sharp/L6/'

Lambda = 1
for run in range(1,2):
    # for Lambda in range(2, 7):
    # loc = 'cosmo_sim_1d/sim_k_1_11/run{}/data_gaussian/L{}/'.format(run, Lambda)
    loc = 'cosmo_sim_1d/sim_k_1_11/run2/'.format(run, Lambda)
    # loc = 'cosmo_sim_1d/sim_k_1_11/'

    files = os.listdir(loc)
    change_index = 1

    for j in range(len(files)):
        # print(files[j])
        # if 'aout_' in files[j]:
        if 'output_hierarchy_' in files[j]:
            # print(files[j])

            try:
                old_filename = loc + files[j]
                old_index_string = files[j][-7:-4]
                if 21 <= int(old_index_string) <= 50:
                    new_index_string = str(int(old_index_string))
                # print(old_index_string, new_index_string)
                # if int(old_index_string) > 22:
                    new_index = int(old_index_string) + 1# + 1)
                    new_index_string = '{0:03d}'.format(new_index)
                    new_filename = old_filename.replace(old_index_string, new_index_string)
                    print(old_filename, new_filename)
                    os.rename(old_filename, new_filename)
                # # print(old_filename, old_index_string)
            except Exception as e: print(e)

    # old_index = int(old_index_string)
    # if old_index > 528:
    # new_index = old_index + change_index
    # print(old_filename)#, new_filename)
        # break
