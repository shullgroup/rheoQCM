
# %%
# import os
# print(os.path.abspath('__file__'))
# exit(0)
import matplotlib.pyplot as plt
try:
    from BCB_sampledefs import sample_dict
except ImportError: # in case running Jupyter in the QCM_py folder
    from QCMFuncs.BCB_sampledefs import sample_dict
try:
    import QCM_functions as qcm
except ImportError: # in case running Jupyter in the QCM_py folder
    import QCMFuncs.QCM_functions as qcm

parms = {}  # parameters to pass to qcm.analyze
parms['dataroot'] = r'C:\Users\ShullGroup\Documents\User Data\WQF\GoogleDriveSync\Side_projects\Alvin_BCB'
parms['figlocation'] = 'datadir' # save data in 
parms['close_on_click_switch'] = False
parms['nx'] = 20 # number of points to calculate if filmindex is not defined
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters  
parms['imagetype'] = 'png'  # default is 'svg'

######### run samples below ##########
#%% 2:1 20181210
qcm.analyze(sample['BCB_4'], parms)

#%% 2:1 20181210
qcm.analyze(sample['BCB_3'], parms)

#%% 2:1 20180327
qcm.analyze(sample['BCB_2_20180406_refit'], parms)

#%% 2:1 20180327
qcm.analyze(sample['BCB_2_RIE_etched_20180413'], parms)

#%% 2:1 20180327
qcm.analyze(sample['BCB_2_recured_20180425'], parms)

#%% 2:1 20180319
qcm.analyze(sample['BCB_1'], parms)

#%% 2:1 20180319
qcm.analyze(sample['BCB_1_RIE_etched_20180413'], parms)

#%% 2:1 20180319
qcm.analyze(sample['BCB_1_recured_20180425'], parms)
