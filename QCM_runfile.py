#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 01:23:49 2018

@author: ken

"""

# %%
import matplotlib.pyplot as plt
import QCM_sampledefs
from QCM_functions import QCManalyze

# define some globar variables that are passed to the solve function, and
# which don't need to be changed every time

# %%  toms PMMA 75k sample
plt.close('all')
QCManalyze(QCM_sampledefs.PMMA_75k_Tom())


# %%  Meredith's PMMA 75k sample
plt.close('all')
QCManalyze(QCM_sampledefs.PMMA_75k())

# %%  This is the spuncast sample
plt.close('all')
QCManalyze(QCM_sampledefs.PSF_spun())

# %%  This is the spuncast polysulfone sample at 130
plt.close('all')
QCManalyze(QCM_sampledefs.PSF_spun_130())

# %%  This is the spuncast polysulfone at RT after being heated to 130
plt.close('all')
QCManalyze(QCM_sampledefs.PSF_spun_heated_cooled())

# %%  This is the floated polysulfone sample
plt.close('all')
QCManalyze(QCM_sampledefs.PSF_float())

# %%  This is the floated polysulfone sample at 130C
plt.close('all')
QCManalyze(QCM_sampledefs.PSF_float_130())

# %%  This is the floated polysulfone sample at RT after being heated to 130
plt.close('all')
QCManalyze(QCM_sampledefs.PSF_float_heated_cooled())

# %%
plt.close('all')

sample = QCM_sampledefs.PS_3k_1()
QCManalyze(sample)

# %%
plt.close('all')

sample = QCM_sampledefs.PS_3k_cool()
QCManalyze(sample)

