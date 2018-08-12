#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 01:23:49 2018

@author: ken

"""

# %%
import matplotlib.pyplot as plt
from Qifeng_sampledefs import sample_dict
from QCM_functions import QCManalyze
parms = {}  # parameters to pass to QCManalyze
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters
parms['imagetype'] = 'pdf'  # default is 'svg'
#%%
QCManalyze(sample['cryt_2_BCB_air'], parms)
#%%
QCManalyze(sample['cryt_2_BCB_LN2'], parms)
#%%
QCManalyze(sample['cryt_2_BCB_air_after_LN2'], parms)

#%%
QCManalyze(sample['DGEBA-Jeffamine2000_RT'], parms)

#%%
QCManalyze(sample['DGEBA-Jeffamine230_RT'], parms)

#%%
QCManalyze(sample['DGEBA-Jeffamine230_RT_2'], parms)
jj
#%% 1:1 good
QCManalyze(sample['DGEBA-Jeffamine230_RT_3'], parms)
j
#%% 2:1 thickj
QCManalyze(sample['DGEBA-Jeffamine230_RT_4'], parms)

#%% 2:1 good
QCManalyze(sample['DGEBA-Jeffamine230_RT_5'], parms)

#%% 2:1 
QCManalyze(sample['DGEBA-Jeffamine2000_RT_2'], parms)

#%% 2:1 
QCManalyze(sample['DGEBA-Jeffamine2000_RT_3'], parms)

#%% 2:1 
QCManalyze(sample['DGEBA-Jeffamine2000_RT_4'], parms)
