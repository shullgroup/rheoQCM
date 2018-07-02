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
import example_sampledefs
from QCM_functions import QCManalyze

# %%  75k PMMA sample
QCManalyze(example_sampledefs.PMMA_75k())

# %%  Temperature dependene for 3k PS sample
QCManalyze(example_sampledefs.PS_3k_cool())
plt.show()
