
# %%
import os
import h5py
import argparse


path_s = r'20240613095729.h5' # sourse file
path_d = r'20240613095729.h5' # file to append to

group_s = r'/data/samp'
group_d = r'/data/ref'

idx_s = list(range(5))
idx_d = list(range(5)) 

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
# %% copy inside path_s,  idx_s to d
with h5py.File(path_s, 'a') as f:
        # parent_group_id = f.require_group(group_s)
        for id_s, id_d in zip(idx_s, idx_d):
            print(id_d)
            # Copy f:group_s to f:group_d
            # f.copy(group_s + '/' + str(id_s), parent_group_id, name=str(id_d))
    
        # check the datasets if copied
        print(f.keys())
        # print(f[group_d].keys())
        # f.copy('data/samp', 'data/ref')
        del f['data/ref']
        f.create_dataset('data/ref', data=f['data/samp'][()], dtype=h5py.special_dtype(vlen=str)) 

        print(f['data/ref'][()])
        print(f['data/samp'][()])
        print(f['data/exp_ref'][()])



# %%
