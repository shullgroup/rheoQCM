

# %%
import os
import h5py
import json

# %% read settings from h5 file

path = './tools/TFA_ED.h5'
with h5py.File(path, 'r') as fh:
    settings = json.loads(fh['settings'][()])

# %% write settings to h5 file

with h5py.File(path, 'a') as fh:
    if 'settings' in fh:
        data_settings =  fh['settings']
        data_settings[()] = json.dumps(settings)
    else:
        fh.create_dataset('settings', data=json.dumps(settings))