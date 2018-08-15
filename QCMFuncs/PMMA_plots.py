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
from PMMA_sampledefs import sample_dict 

parms = {}  # parameters to pass to QCManalyze
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters
parms['imagetype'] = 'pdf'  # default is 'svg'

# %%  75k PMMA sample from Meredith
QCManalyze(sample['PMMA_75k_T01'], parms)

# %%  75k PMMA sample from Tom
QCManalyze(sample['PMMA_75k_S04_postanneal'], parms)
# %% 75 k PMMA sample Tom Post Annealing
QCManalyze(sample['PMMA_75k_S04_TiO2coating'], parms)
# %% 75 k PMMA sample Tom Post Exposure 2000J/cm^2
QCManalyze(sample['PMMA_75k_S04_TiO2_2000I'], parms)
# %% 75 k PMMA sample Tom Post Exposure 4000 J/cm^2
QCManalyze(sample['PMMA_75k_S04_TiO2_4000I'], parms)
# %% 75 k PMMA sample Tom Post Exposure 6000 J/cm^2
QCManalyze(sample['PMMA_75k_S04_TiO2_6000I'], parms)
# %% 75 k PMMA sample Tom Post Exposure 36 min
QCManalyze(sample['PMMA_75k_S04_TiO2_36min'], parms)
# %% 75 k PMMA sample Tom Post Exposure 46 min
QCManalyze(sample['PMMA_75k_S04_TiO2_46min'], parms)
# %% 75 k PMMA sample Tom Post Exposure 56 min
QCManalyze(sample['PMMA_75k_S04_TiO2_56min'], parms)
# %% 75 k PMMA sample Tom Post Exposure 66 min
QCManalyze(sample['PMMA_75k_S04_TiO2_66min'], parms)
# %%  75k PMMA sample from Tom
QCManalyze(sample['PMMA_75k_S05'], parms)
# %%  crystal test QCMS06
QCManalyze(sample['PMMA_75k_S06_test'], parms)
# %%  S06 PMMA film
QCManalyze(sample['PMMA_75k_S06'], parms)
# %%  S08 PMMA film
QCManalyze(sample['PMMA_75k_S08'], parms)
# %%  S08 TiO2 PMMA film
QCManalyze(sample['PMMA_75k_S08_TiO2'], parms)
# %%  S08 TiO2 PMMA film
QCManalyze(sample['PMMA_75k_S08_TiO2_1day'], parms)
# %%  S08 TiO2 PMMA film
QCManalyze(sample['PMMA_75k_S08_TiO2_2day'], parms)
# %%  S08 TiO2 PMMA film
QCManalyze(sample['PMMA_75k_S08_TiO2_3day'], parms)
# %%  S08 TiO2 PMMA film
QCManalyze(sample['PMMA_75k_S08_TiO2_5day'], parms)
# %%  S08 TiO2 PMMA film
QCManalyze(sample['PMMA_75k_S08_TiO2_6day'], parms)
# %%  S08 TiO2 PMMA film
QCManalyze(sample['PMMA_75k_S08_TiO2_6day_2'], parms)
# %%  S08 TiO2 PMMA film
QCManalyze(sample['PMMA_75k_S08_TiO2_7day'], parms)
# %%  S08 TiO2 PMMA film
QCManalyze(sample['PMMA_75k_S08_TiO2_8day'], parms)