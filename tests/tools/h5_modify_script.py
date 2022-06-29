
# %%
import os
import h5py
import argparse


path_s = r'20_DEP_KD_JL_AAexposure_control.h5' # sourse file
path_d = r'20_DEP_KD_JL_AAexposure.h5' # file to append to

group_s = r'/raw/samp'
group_d = r'/raw/samp'

idx_s = list(range(77, 186))
idx_d = [id - 22 for id in idx_s] # -22

# 55 to 78 change to 187 to 210 (+132)
# %% change names in destination file
with h5py.File(path_d, 'a') as fd:
    for id in range(55, 79):
        print(id)
        # copy
        fd[group_d + '/' + str(id + 132)] = fd[group_d + '/' + str(id)] 
        # delete
        del fd[group_d + '/' + str(id)]

   


# %% move idx_s to d
    with h5py.File(path_s, 'r') as fs:
        with h5py.File(path_d, 'a') as fd:
            parent_group_id = fd.require_group(group_d)
            
            for id_s, id_d in zip(idx_s, idx_d):
                print(id_d)
                # Copy fs:group_s to fd:group_d
                fs.copy(group_s + '/' + str(id_s), parent_group_id, name=str(id_d))
        
            # check the datasets if copied
            # print(fd[group_d].keys())


# %%
