# %%

from modules.QCM import *
import QCM_functions as qcmfun
import numpy as np
import copy
import pandas as pd

qcm = QCM()
qcm.f1 = 5e6 #5002468 # Hz
qcm.g1 = 16  # Hz
qcm.refh = 3
# qcm.refto = 1
# qcm.fit_method = 'lmfit'

# set experimental errors to 0
qcm.g_err_min = 0 # error floor for gamma
qcm.f_err_min = 0 # error floor for f
qcm.err_frac = 0 # error in f or gamma as a fraction of gamma


nh = [3,5,3]
harms = [1, 3, 5]
samp = 'PS'
# samp = 'PS_water'
samp = 'PS_water2'
samp = 'PET_PBS'

if samp == 'BCB':
    delfstar = {
        1: -28206.4782657343 + 1j*5.6326137881,
        3: -87768.0313369799 + 1j*155.716064797,
        5: -159742.686586637 + 1j*888.6642467156,
    }
    film = {0: {'calc': False, 'drho': 2.8e-06, 'grho': 3e+17, 'phi': 0, 'n': 3}, 1: {'calc': True}}
elif samp == 'water':
    delfstar = {
        1: -694.15609764494 + 1j*762.8726222543,
        3: -1248.7983004897833 + 1j*1215.1121711257,
        5: -1641.2310467399657 + 1j*1574.7706516819,
    }
    film = {0: {'calc': False, 'drho': 2.8e-06, 'grho': 3e+17, 'phi': 0, 'n': 3}, 1: {'calc': True}}
elif samp == 'PS':
    delfstar = {
        1: -17976.05155692622 + 1j*4.9702365159206146,
        3: -55096.2819308918 + 1j*5.28570043096034,
        5: -95888.85117323324 + 1j*26.76581997773431,
    }
    film = {0: {'calc': False, 'drho': 2.8e-06, 'grho': 3e+17, 'phi': 0, 'n': 3}, 1: {'calc': True}}
elif samp == 'PS_water': # ref to air
    delfstar = {
        1: -18740.0833361046 + 1j*709.1201809073,
        3: -56445.09063657 + 1j*1302.7285967785,
        5: -97860.1416540742 + 1j*1943.5972185125,
    }
    film = {
        0: {'calc': False, 'drho': 2.8e-06, 'grho': 3e+17, 'phi': 0, 'n': 3}, 
        1: {'calc': True},
        2: {'calc': False, 'drho': 0.5347e-3, 'grho': 86088e3, 'phi': np.pi/2, 'n': 3}}
elif samp == 'PS_water2': # ref to S 8 (hH2O)
    delfstar = {
        1: -18045.9272384596 + 1j*-53.7524413470001,
        3: -55196.2923360802 + 1j*87.6164256528002,
        5: -96218.9106073342 + 1j*368.8265668306,
    }
    film = {
        0: {'calc': False, 'drho': 2.8e-06, 'grho': 3e+17, 'phi': 0, 'n': 3}, 
        1: {'calc': True},
        2: {'calc': False, 'drho': 0.5347e-3, 'grho': 86088e3, 'phi': np.pi/2, 'n': 3}}
elif samp == 'PS_water3': # ref to R 19 (H2O)
    delfstar = {
        1: -17845.878287589177 + 1j*-6.106914860800089,
        3: -54958.28525063768 + 1j*62.071071251499916,
        5: -95887.36210607737 + 1j*283.76939751400005,
    }
    film = {
        0: {'calc': False, 'drho': 2.8e-06, 'grho': 3e+17, 'phi': 0, 'n': 3}, 
        1: {'calc': True},
        2: {'calc': False, 'drho': 0.5347e-3, 'grho': 86088e3, 'phi': np.pi/2, 'n': 3}}
elif samp == 'PET_PBS':
    # delfstar = {
    #     1: -18509.7782613868 + 1j*747.6284643851,
    #     3: -54797.9162914753 + 1j*1306.5316867216,
    #     5: -93851.0474332124 + 1j*2319.8000874687,
    # }

    delfstar_overlayer = {
        1: -753.160689003766 + 1j*734.3562839223,
        3: -1418.76131652482 + 1j*1126.617726934,
        5: -1948.36904231831 + 1j*1456.9099428171,
    }
    delfstar_0 = { #20210728 S#295 refto air
        1: -18377 + 1j*734.1,
        3: -54373 + 1j*1249.9,
        5: -92873 + 1j*2061.9,
    }
    delfstar_1 = { #20210728 S#295 ref to water
        1: -17623 - 1j*0.26011,
        3: -52954 + 1j*123.33,
        5: -90924 + 1j*604.99,
    }

    film = {
        0: {'calc': False, 'drho': 2.8e-06, 'grho': 3e+17, 'phi': 0, 'n': 3}, 
        1: {'calc': True},
        2: {'calc': False, 'drho': np.inf, 'grho': 86088e3, 'phi': np.pi/2, 'n': 3}}

# %% calc w/ only one layer prop is unknown
qcm.fit_method = 'lmfit'
print(samp, 'refto 0')
qcm.refto = 0
film[1]['grho'] = 1e12
new_film = copy.deepcopy(film)
brief_props, props = qcm.solve_general_delfstar_to_prop(nh, delfstar_0, new_film, calctype='LL', bulklimit=.5, nh_interests=harms, brief_report=False, 
# prop_guess={'phi': 0.09, 'drho':3e-3}
)

print(brief_props)

# calc w/ both layers' prop are unknown
print(samp, 'refto 1')


# print(delfstar_film)
new_film = copy.deepcopy(film)
qcm.refto = 1
brief_props, props = qcm.solve_general_delfstar_to_prop(nh, delfstar_1, new_film, calctype='LL', bulklimit=.5, nh_interests=harms, brief_report=False)

print(brief_props)

# %%
df0 = pd.DataFrame({})
df0[1] = [delfstar_0[1]]
df0[3] = [delfstar_0[3]]
df0[5] = [delfstar_0[5]]

print(qcmfun.solve_for_props(df0, '353', calctype='LL', reftype='bare', overlayer={'drho': np.inf, 'grho3': 1e8, 'phi': 90})
)

df1 = pd.DataFrame({})
df1[1] = [delfstar_1[1]]
df1[3] = [delfstar_1[3]]
df1[5] = [delfstar_1[5]]

print(qcmfun.solve_for_props(df1, '353', calctype='LL', reftype='overlayer', overlayer={'drho': np.inf, 'grho3': 1e8, 'phi': 90})
)


# %% water ref to crystal
film = qcm.build_single_layer_film(prop_default['water'])
qcm.calctype = 'LL'
qcm.refto = None
for n in [1, 3, 5]:
    print(qcm.calc_delfstar(n, film))

# %% single layer
# water
layers = {
    'film': {
      'phi': 90,
      'drho': np.inf,
      'grho3': 100000000.0},
    'electrode': 
      {'drho': 2.8e-6, 'grho3': 3e17, 'phi': 0
      }
}
print(qcmfun.calc_delfstar(1, layers, calctype='SLA', reftype='bare'))

# electrode
elect = {0: prop_default['electrode'].copy()}
elect[0]['calc'] = True
print(elect)
qcm.calctype = 'LL'
print(qcm.calc_delfstar(1, elect))


# %% make a film in water
polymerfilm = {
    # electrode
    0: {'calc': False, 'drho': 2.8e-06, 'grho': 3e+17, 'phi': 0, 'n': 3},
    # # polymer
    # 1: {'calc': True,
    # 'grho': 0.5e12,
    # 'phi': np.pi*10/180,
    # 'drho': 5e-3,
    # 'n': 3},
    
    1: {'calc': True, # 20210728 to S#295 calc from qcmfun
    'grho': 1.752417e+12,
    'phi': np.pi*3.569943/180,
    'drho': 3.068e-3,
    'n': 3},

    # 1: {'calc': True, 'drho': 5e-3, 'grho': 1e12, 'phi': 5.729577951308232*np.pi/180, 'n': 3}, # phi=0.1 rad
    # water
    2: {'calc': False, 
    'drho': np.inf,
    'grho': 1e8,
    'phi': np.pi/2,
    'n': 3}}


qcm.refto = 0
qcm.calctype = 'LL'
delfstar_tot = {
    n: qcm.calc_delfstar(n, polymerfilm) for n in [1, 3, 5]
}
delfstar_ref = { # water+electrode
    n: qcm.calc_delfstar(n, qcm.get_ref_layers(polymerfilm)) for n in [1, 3, 5]
}
delfstar_film_sub = { # polymer by subtraction
    n: delfstar_tot[n] - delfstar_ref[n] for n in [1, 3, 5]
}

film_only = qcm.remove_layer_n(polymerfilm, 2) 
delfstar_film = { # polymer
    n: qcm.calc_delfstar(n, film_only) for n in [1, 3, 5]
}

qcm.refto=1
delfstar_tot_refto1 = { # polymer refto 1
    n: qcm.calc_delfstar(n, polymerfilm) for n in [1, 3, 5]
}

print('tot refo 0')
print(delfstar_tot)
print('tot refo 1')
print(delfstar_tot_refto1)
print('ref')
print(delfstar_ref)
print('film w/o overlayer')
print(delfstar_film)
print('film = tot0 - ref')
print(delfstar_film_sub)


# %%
new_polymerfilm = copy.deepcopy(polymerfilm)
# refto == 0 and delfstar_tot 
# works the same as
# refto == 1 and delfstar_film

qcm.refto = 0
brief_props, props = qcm.solve_general_delfstar_to_prop(nh, delfstar_tot, new_polymerfilm, calctype='LL', bulklimit=.5)

print(brief_props)


qcm.refto = 1
brief_props, props = qcm.solve_general_delfstar_to_prop(nh, delfstar_film_sub, new_polymerfilm, calctype='LL', bulklimit=.5)

print(brief_props)


# %% qcmfun
layers = {
    'overlayer': {
      'phi': 90,
      'drho': np.inf,
      'grho3': 1e8}, 
    # 'film': {
    #   'phi': 10,
    #   'drho': 5e-3,
    #   'grho3': 0.5e12},
    # 'film': {
    #   'phi': 3.569943,
    #   'drho': 0.003068,
    #   'grho3': 1.752417e+12},
    'film': {'drho': 5e-3, 'grho3': 1e12, 'phi': 5.729577951308232}, # phi=0.1 rad
    'electrode': 
      {'drho': 2.8e-6, 'grho3': 3e17, 'phi': 0
      }
}
qcmfun_tot = {}
qcmfun_tot_to_overlayer = {}
qcmfun_film = {}
qcmfun_film_sub = {}
qcmfun_overlayer = {}
for n in [1, 3, 5]:
    qcmfun_tot[n] = qcmfun.calc_delfstar(n, layers, calctype='LL', reftype='bare')
    
    qcmfun_tot_to_overlayer[n] = qcmfun.calc_delfstar(n, layers, calctype='LL', reftype='overlayer')
    
    layers_f = {}
    layers_f['film'] = layers['film']
    layers_f['electrode'] = layers['electrode']
    qcmfun_film[n] = qcmfun.calc_delfstar(n, layers_f, calctype='LL', reftype='bare')
    
    layers_ol = {}
    layers_ol['film'] = layers['overlayer']
    layers_ol['electrode'] = layers['electrode']
    qcmfun_overlayer[n] = qcmfun.calc_delfstar(n, layers_ol, calctype='LL', reftype='bare')

    qcmfun_film_sub[n] = qcmfun_tot[n] - qcmfun_overlayer[n]

print('qcmfun tot')
print(qcmfun_tot)
print('qcmfun tot ref to overlayer')
print(qcmfun_tot_to_overlayer)
print('qcmfun overlayer')
print(qcmfun_overlayer)
print('qcmfun film only')
print(qcmfun_film)
print('qcmfun film = tot - overlayer')
print(qcmfun_film_sub)


# %%
# reconstruct delfstar dic
df1 = pd.DataFrame({})
df1[1] = [qcmfun_tot_to_overlayer[1]]
df1[3] = [qcmfun_tot_to_overlayer[3]]
df1[5] = [qcmfun_tot_to_overlayer[5]]

print(qcmfun.solve_for_props(df1, '353', calctype='LL', reftype='overlayer', overlayer={'drho': np.inf, 'grho3': 1e8, 'phi': 90}))


df0 = pd.DataFrame({})
df0[1] = [qcmfun_tot[1]]
df0[3] = [qcmfun_tot[3]]
df0[5] = [qcmfun_tot[5]]

print(qcmfun.solve_for_props(df0, '353', calctype='LL', reftype='bare', overlayer={'drho': np.inf, 'grho3': 1e8, 'phi': 90}))

#####################################################








#####################################################
# test functions

# %%
delfstar_0 = { #20210728 S#295 refto air
    1: -18377 + 1j*734.1,
    3: -54373 + 1j*1249.9,
    5: -92873 + 1j*2061.9,
}
drho_qcm = qcm.calc_drho(nh[0], delfstar_0, dlam_refh, phi)
print(drho_qcm)



drho_qcmfun = -(qcmfun.sauerbreym(nh[0], delfstar_0[nh[0]].real) / qcmfun.normdelfstar(nh[0], dlam_refh, phi).real)
print(drho_qcmfun)

# %%
print(qcmfun.sauerbreym(nh[0], delfstar_1[nh[0]]))
print(qcm.sauerbreym(nh[0], delfstar_1[nh[0]]))
print(qcmfun.normdelfstar(nh[0], dlam_refh, phi*180/np.pi))
print(qcm.normdelfstar(nh[0], dlam_refh, phi))

# %%
print(qcmfun.dlam(nh[0], dlam_refh, phi*180/np.pi))
print(qcm.dlam(nh[0], dlam_refh, phi))


# %%

print(qcm.grho_from_dlam(qcm.refh, drho, dlam_refh, phi))
print(qcmfun.grho_from_dlam(qcm.refh, drho, dlam_refh, phi*180/np.pi))


# %% check calc_ZL
layers = {
    # 1: 
    #   {'drho': 2.8e-6, 'grho3': 3e17, 'phi': 0
    #   },
    # 2: {
    #   'phi': 10,
    #   'drho': 5e-3,
    #   'grho3': 0.5e12},
    1: {

      'phi': 90,
      'drho': np.inf,
      'grho3': 1e8}, 
}
print(qcm.get_ref_layers(polymerfilm))
# print(qcm.calc_ZL(3, qcm.get_ref_layers(polymerfilm), 0))
# print(qcm.calc_ZL(3, polymerfilm, 0))
print(qcm.remove_layer_0(qcm.get_ref_layers(polymerfilm)))
print(qcm.calc_ZL(3, qcm.remove_layer_0(qcm.get_ref_layers(polymerfilm)), 0))
print(layers)
print(qcmfun.calc_ZL(3, layers, 0, calctype='SLA'))



# %%
