#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 01:23:49 2018

@author: ken

"""

# %%
import matplotlib.pyplot as plt
from QCM_functions import QCManalyze, process_raw
# change PS_sampledefs to correct file as appropriate
from PS_sampledefs import sample_dict 

parms = {}  # parameters to pass to QCManalyze
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters
parms['imagetype'] = 'pdf'  # default is 'svg'

# %%  Temperature dependence for 3k PS sample 
QCManalyze(sample['PS_3k_cool'], parms)

# %%  Temperature dependence for 3k PS sample
QCManalyze(sample['PS_30k'], parms)

# %%  Temperature dependence for 3k PS sample
QCManalyze(sample['PS_192k'], parms)
