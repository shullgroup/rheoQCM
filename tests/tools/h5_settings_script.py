

# %%
import os
import h5py
import json

# %% read settings from h5 file

path = './tools/0.3 g HPC and ALA 2 AA (1).h5'
with h5py.File(path, 'r') as fh:
    settings = json.loads(fh['settings'][()])


# %%
settings['harmdata']['ref']=settings['harmdata']['samp']


# %% write settings to h5 file

with h5py.File(path, 'a') as fh:
    if 'settings' in fh:
        data_settings =  fh['settings']
        data_settings[()] = json.dumps(settings)
    else:
        fh.create_dataset('settings', data=json.dumps(settings))
# %%
