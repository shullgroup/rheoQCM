'''
This script plot the same sample tested by Matlab program 
and Python program, respectively, to demonstrate how to use 
QCM_functions to do the data analysis.

This script is supposed to run with QCMFuns as root folder.
Otherwise, please change:
import package (example_sampledefs and QCM_functions) path and parms['dataroot']
'''


# %%
import os
import matplotlib.pyplot as plt
from example_sampledefs import sample_dict
import QCM_functions as qcm

parms = {}  # parameters to pass to qcm.analyze
parms['dataroot'] = os.path.join(os.getcwd(), 'example_data')
parms['figlocation'] = 'datadir' # save data in 
parms['close_on_click_switch'] = False
parms['nx'] = 20 # number of points to calculate if filmindex is not defined
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters  
parms['imagetype'] = 'png'  # default is 'svg'


################################

#%% mat
# qcm.analyze(sample['polymer_matlab'], parms)

#%% h5
qcm.analyze(sample['polymer_h5'], parms)