#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 01:23:49 2018

@author: ken

"""
# This is an example runfile that illustrates the use of QCManalyze
# it make use of the example_sampledefs.py, where which defines the
# files to be analyzed
# %%
import matplotlib.pyplot as plt
from example_sampledefs import sample_dict
from QCM_functions import QCManalyze
parms = {}  # parameters to pass to QCManalyze
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters
parms['imagetype'] = 'pdf'  # default is 'svg'

# %%  75k PMMA sample
QCManalyze(sample['PMMA_75k'], parms)

# %%  Temperature dependene for 3k PS sample
QCManalyze(sample['PS_3k_cool'], parms)

# add additional lines as needed