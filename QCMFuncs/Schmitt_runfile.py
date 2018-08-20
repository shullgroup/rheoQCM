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
from Schmitt_sampledefs import sample_dict 

parms = {}  # parameters to pass to qcm.analyze
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters
parms['imagetype'] = 'pdf'  # default is 'svg'
parms['dataroot'] = qcm.find_dataroot('schmitt')

spectrafig = qcm.plot_spectra({}, sample['PMMA_75k_S08_TiO2_6day_2'], np.arange(20))
qcm.plot_spectra(spectrafig, sample['PMMA_75k_S08_TiO2_8day'], np.arange(20))
#qcm.plot_spectra(spectrafig, sample['PMMA_75k_S08_TiO2_7day'], [1])
# %%  75k PMMA sample from Meredith
qcm.analyze(sample['PMMA_75k_T01'], parms)

# %%  75k PMMA sample from Tom
qcm.analyze(sample['PMMA_75k_S04_postanneal'], parms)
# %% 75 k PMMA sample Tom Post Annealing
qcm.analyze(sample['PMMA_75k_S04_TiO2coating'], parms)
# %% 75 k PMMA sample Tom Post Exposure 2000J/cm^2
qcm.analyze(sample['PMMA_75k_S04_TiO2_2000I'], parms)
# %% 75 k PMMA sample Tom Post Exposure 4000 J/cm^2
qcm.analyze(sample['PMMA_75k_S04_TiO2_4000I'], parms)
# %% 75 k PMMA sample Tom Post Exposure 6000 J/cm^2
qcm.analyze(sample['PMMA_75k_S04_TiO2_6000I'], parms)
# %% 75 k PMMA sample Tom Post Exposure 36 min
qcm.analyze(sample['PMMA_75k_S04_TiO2_36min'], parms)
# %% 75 k PMMA sample Tom Post Exposure 46 min
qcm.analyze(sample['PMMA_75k_S04_TiO2_46min'], parms)
# %% 75 k PMMA sample Tom Post Exposure 56 min
qcm.analyze(sample['PMMA_75k_S04_TiO2_56min'], parms)
# %% 75 k PMMA sample Tom Post Exposure 66 min
qcm.analyze(sample['PMMA_75k_S04_TiO2_66min'], parms)
# %%  75k PMMA sample from Tom
qcm.analyze(sample['PMMA_75k_S05'], parms)
# %%  crystal test QCMS06
qcm.analyze(sample['PMMA_75k_S06_test'], parms)
# %%  S06 PMMA film
qcm.analyze(sample['PMMA_75k_S06'], parms)
# %%  S08 PMMA film
qcm.analyze(sample['PMMA_75k_S08'], parms)
# %%  S08 TiO2 PMMA film
qcm.analyze(sample['PMMA_75k_S08_TiO2'], parms)
# %%  S08 TiO2 PMMA film
qcm.analyze(sample['PMMA_75k_S08_TiO2_1day'], parms)
# %%  S08 TiO2 PMMA film
qcm.analyze(sample['PMMA_75k_S08_TiO2_2day'], parms)
# %%  S08 TiO2 PMMA film
qcm.analyze(sample['PMMA_75k_S08_TiO2_3day'], parms)
# %%  S08 TiO2 PMMA film
qcm.analyze(sample['PMMA_75k_S08_TiO2_5day'], parms)
# %%  S08 TiO2 PMMA film
qcm.analyze(sample['PMMA_75k_S08_TiO2_6day'], parms)
# %%  S08 TiO2 PMMA film
qcm.analyze(sample['PMMA_75k_S08_TiO2_6day_2'], parms)

# %%  S08 TiO2 PMMA film
qcm.analyze(sample['PMMA_75k_S08_TiO2_7day'], parms)
# %%  S08 TiO2 PMMA film
qcm.analyze(sample['PMMA_75k_S08_TiO2_8day'], parms)