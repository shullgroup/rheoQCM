#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 01:23:49 2018

@author: ken

"""

# %%
import matplotlib.pyplot as plt
import QCM_functions as qcm
import numpy as np
# change PS_sampledefs to correct file as appropriate
from Depolo_sampledefs import sample_dict 

parms = {}  # parameters to pass to qcm.analyze
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters
parms['imagetype'] = 'pdf'  # default is 'svg'
parms['dataroot'] = qcm.find_dataroot('depolo')


# %%  Gwen's reasonably thick linseed oil sample
qcm.analyze(sample['linseed_bulk'], parms)

# %% sample with PS film
qcm.analyze(sample['linseed_PS'], parms)
