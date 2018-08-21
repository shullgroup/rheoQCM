#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 01:23:49 2018

@author: ken

"""

# %%
# import os
# print(os.path.abspath('__file__'))
# exit(0)
import matplotlib.pyplot as plt
try:
    from Qifeng_sampledefs import sample_dict
except ImportError: # in case running Jupyter in the QCM_py folder
    from QCMFuncs.Qifeng_sampledefs import sample_dict
try:
    import QCM_functions as qcm
except ImportError: # in case running Jupyter in the QCM_py folder
    import QCMFuncs.QCM_functions as qcm

parms = {}  # parameters to pass to qcm.analyze
parms['dataroot'] = qcm.find_dataroot('qifeng')
parms['figlocation'] = 'datadir' # save data in 
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters
parms['imagetype'] = 'pdf'  # default is 'svg'

######### run samples below ##########
#%% 2:1 
qcm.analyze(sample['DGEBA-Jeffamine400_RT_3'], parms)

#%% 2:1 
qcm.analyze(sample['DGEBA-Jeffamine400_RT_2'], parms)

#%% 2:1 
qcm.analyze(sample['DGEBA-Jeffamine400_RT'], parms)

#%% 2:1 
qcm.analyze(sample['DGEBA-Jeffamine2000_RT_4_2'], parms)

#%% 2:1 
qcm.analyze(sample['DGEBA-Jeffamine2000_RT_4'], parms)

#%% 2:1 
qcm.analyze(sample['DGEBA-Jeffamine2000_RT_3'], parms)

#%% 2:1 
qcm.analyze(sample['DGEBA-Jeffamine2000_RT_2'], parms)

#%% 2:1 good
qcm.analyze(sample['DGEBA-Jeffamine230_RT_5'], parms)

#%% 2:1 thickj
qcm.analyze(sample['DGEBA-Jeffamine230_RT_4'], parms)

#%% 1:1 good
qcm.analyze(sample['DGEBA-Jeffamine230_RT_3'], parms)

#%%
qcm.analyze(sample['DGEBA-Jeffamine230_RT_2'], parms)

#%%
qcm.analyze(sample['DGEBA-Jeffamine230_RT'], parms)

#%%
qcm.analyze(sample['DGEBA-Jeffamine2000_RT'], parms)
#%%
qcm.analyze(sample['cryt_2_BCB_air_after_LN2'], parms)
#%%
qcm.analyze(sample['cryt_2_BCB_LN2'], parms)
#%%
qcm.analyze(sample['cryt_2_BCB_air'], parms)
