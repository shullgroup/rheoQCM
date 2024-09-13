

# %%
import os
import h5py
import json

# %% read settings from h5 file

path = './0.2g_AAP(0.6)_2TFA_ED(1).h5'
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
