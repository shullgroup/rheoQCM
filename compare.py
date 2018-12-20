'''
compare matlab and python data
polymer_matlab.mat
polymer.h5
'''

import matplotlib.pyplot as plt
import QCMFuncs.QCM_functions as qcm
import modules.QCM as QCM
import numpy as np

qcmc = QCM.QCM()
qcmc.f1 = 5e6 # Hz
qcmc.rh = 3

nh = [3,5,3]

delfstar = {
    1: -11374.5837019132 + 1j*2.75600512576013,
    3: -34446.6126772240 + 1j*41.5415621054838,
    5: -58249.9656346552 + 1j*79.9003634583878,
}

overlayer = {'drho':0, 'grho_rh':0, 'phi':0}

n = 3
drho = 0.003
grho_rh = 1e9
phi = 20
dlam_rh = 0.05

print(qcm.rhcalc(nh, dlam_rh, phi))
print(qcmc.rhcalc(nh, dlam_rh, np.deg2rad(phi)))


dfst = qcm.delfstarcalc(n, drho, grho_rh, phi, overlayer)
dfstc = qcmc.delfstarcalc(n, drho, grho_rh, np.deg2rad(phi), overlayer)
print(dfst)
print(dfstc)
print(np.real(dfstc))
print(np.imag(dfstc))
exit(0)




def sample_dict():
    sample = {}  # individual sample dictionaries get added to this

    samplename = 'polymer_matlab'
    sample[samplename] = {
    'samplename': 'polymer_matlab',
    'datadir': 'data/polymer_matlab',
    'barefile': 'polymer_matlab_bare',
    'filmfile': 'polymer_matlab',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['353'],
    'nhplot': [1, 3, 5]
    }
    return sample

parms = {}  # parameters to pass to qcm.analyze
parms['close_on_click_switch'] = False
parms['dataroot'] = qcm.find_dataroot('')
parms['figlocation'] = 'datadir' # save data in 
sample = sample_dict()  # read sample dictionary

# specify any non-default parameters  
parms['imagetype'] = 'png'  # default is 'svg'



qcm.analyze(sample['polymer_matlab'], parms)

drho, grho_rh, phi, dlam_rh, err = qcmc.solve_general(nh, delfstar, overlayer)

print('drho', drho)
print('grho_rh', grho_rh)
print('phi', phi)
print('dlam_rh', phi)
print('err', err)
