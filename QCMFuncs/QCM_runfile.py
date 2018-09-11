#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 01:23:49 2018

@author: ken

"""

# %%
import matplotlib.pyplot as plt
from QCM_functions import QCManalyze, process_raw
from PS_sampledefs import sample_dict

parms = {}  # parameters to pass to QCManalyze
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters
parms['imagetype'] = 'pdf'  # default is 'svg'


# %%  75k PMMA sample from Meredith
QCManalyze(sample['PMMA_75k_T01'], parms)
plt.show()

# %%plot the bare crystal data from one of David's samples
dict = process_raw(sample['Bare_xtal'], 'film')
dict['rawfig'].savefig('figures/refdata.pdf')

# %%Qifeng's epoxy sample for bulk analysis
QCManalyze(sample['DGEBA-Jeffamine2000_RT'], parms)

# %%  75k PMMA sample from Tom
QCManalyze(sample['PMMA_75k_S04'], parms)

# %%  75k PMMA sample from Tom
QCManalyze(sample['PMMA_75k_S05'], parms)

# %%  Temperature dependence for 3k PS sample
QCManalyze(sample['PS_3k_cool'], parms)
