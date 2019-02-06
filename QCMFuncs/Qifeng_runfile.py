#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 01:23:49 2018

@author: ken

"""

# %%
# import os
# print(os.path.abspath('__file__'))
# exit(0)
import matplotlib.pyplot as plt
try:
    from Qifeng_sampledefs import sample_dict
except ImportError: # in case running Jupyter in the QCM_py folder
    from QCMFuncs.Qifeng_sampledefs import sample_dict
try:
    import QCM_functions as qcm
except ImportError: # in case running Jupyter in the QCM_py folder
    import QCMFuncs.QCM_functions as qcm

parms = {}  # parameters to pass to qcm.analyze
parms['dataroot'] = qcm.find_dataroot('qifeng')
parms['figlocation'] = 'datadir' # save data in 
parms['close_on_click_switch'] = False
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters  
parms['imagetype'] = 'png'  # default is 'svg'

######### run samples below ##########
#%% 2:1 20190201 samp
qcm.analyze(sample['D400_5_N2_CO2__samp'], parms)
exit(0)
#%% 2:1 20190201 ref
# qcm.analyze(sample['D400_5_N2_CO2__ref'], parms)

#%% 2:1 20190129 samp
qcm.analyze(sample['D400_4_N2_RH__samp'], parms)
#%% 2:1 20190129 ref
qcm.analyze(sample['D400_4_N2_RH__ref'], parms)






#%% 2:1 20181029 
qcm.analyze(sample['DGEBA-PACM_RT_2_cured'], parms)

exit(0)
#%% 2:1 20181029 
# qcm.analyze(sample['DGEBA-PACM_RT_cured'], parms)

#%% 2:1 20181029 
# qcm.analyze(sample['DGEBA-Jeffamine400_RT_3_cured'], parms)

#%% 2:1 20181029 
# qcm.analyze(sample['DGEBA-Jeffamine400_RT_2_cured'], parms)

#%% 2:1 20181029 
# qcm.analyze(sample['DGEBA-Jeffamine230_RT_5_cured'], parms)

#%% 2:1 20181029 
# qcm.analyze(sample['DGEBA-Jeffamine230_RT_3_cured'], parms)

#%% 2:1 20180917 
qcm.analyze(sample['DGEBA-Jeffamine2000_RT_7'], parms)

#%% 2:1 20180914 (film dewetted)
# qcm.analyze(sample['DGEBA-Jeffamine2000_RT_6'], parms)

#%% 2:1 20180828 (film dewetted)
# qcm.analyze(sample['DGEBA-Jeffamine2000_RT_5'], parms) 

#%% 2:1 20180824 good but low G
# qcm.analyze(sample['DGEBA-PACM_RT_2'], parms) 

#%% 2:1 20180823 thick  10 um high phi
# qcm.analyze(sample['DGEBA-PACM_RT'], parms)

#%% 2:1 20180817 good
# qcm.analyze(sample['DGEBA-Jeffamine400_RT_3'], parms)

#%% 2:1 20180814 solving error 
# qcm.analyze(sample['DGEBA-Jeffamine400_RT_2'], parms)

#%% 2:1 20180813 dewetted
# qcm.analyze(sample['DGEBA-Jeffamine400_RT'], parms)

#%% 2:1 20180810 dewetted after 2 days
# qcm.analyze(sample['DGEBA-Jeffamine2000_RT_4_2'], parms)

#%% 2:1 20180808 dewetted after 2 days
# qcm.analyze(sample['DGEBA-Jeffamine2000_RT_4'], parms)

#%% 2:1 20180807 not uniform due to dewetting
# qcm.analyze(sample['DGEBA-Jeffamine2000_RT_3_2'], parms)

#%% 2:1 20180806 not uniform 
# qcm.analyze(sample['DGEBA-Jeffamine2000_RT_3'], parms)

#%% 2:1 20180803 not uniform
# qcm.analyze(sample['DGEBA-Jeffamine2000_RT_2'], parms)

#%% 2:1 20180727 good
# qcm.analyze(sample['DGEBA-Jeffamine230_RT_5'], parms)

#%% 2:1 thick
# qcm.analyze(sample['DGEBA-Jeffamine230_RT_4'], parms)

#%% 20180724 1:1 good
# qcm.analyze(sample['DGEBA-Jeffamine230_RT_3'], parms)

#%%
# qcm.analyze(sample['DGEBA-Jeffamine230_RT_2'], parms)

#%%
# qcm.analyze(sample['DGEBA-Jeffamine230_RT'], parms)

#%%
# qcm.analyze(sample['DGEBA-Jeffamine2000_RT'], parms)
#%%
# qcm.analyze(sample['cryt_2_BCB_air_after_LN2'], parms)
#%%
# qcm.analyze(sample['cryt_2_BCB_LN2'], parms)
#%%
# qcm.analyze(sample['cryt_2_BCB_air'], parms)
