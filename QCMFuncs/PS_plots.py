#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 01:23:49 2018

@author: ken

"""

# %%
import sys
import os
print(sys.path)
print('current directory is', os.getcwd())

# %%
import matplotlib.pyplot as plt
import QCM_functions as qcm
from PS_sampledefs import sample_dict 

parms = {}  # parameters to pass to QCManalyze
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters
parms['imagetype'] = 'pdf'  # default is 'svg'
parms['dataroot'] = 'data_PS'

# %%  Temperature dependence for 3k PS sample 
qcm.analyze(sample['PS_3k_cool'], parms)

# %%  Temperature dependence for 3k PS sample
qcm.analyze(sample['PS_30k'], parms)

# %%  Temperature dependence for 3k PS sample
qcm.analyze(sample['PS_192k'], parms)
