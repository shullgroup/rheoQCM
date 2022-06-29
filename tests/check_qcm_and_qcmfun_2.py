# %%

from modules.QCM import *
import QCM_functions as qcmfun
import numpy as np
import copy
import pandas as pd


# %% inintiate qcm & qcmfun modules and build films
qcm = QCM()
qcm.f1 = 5e6 #5002468 # Hz
qcm.g1 = 50  # Hz

# set experimental errors to 0
qcm.g_err_min = 0 # error floor for gamma
qcm.f_err_min = 0 # error floor for f
qcm.err_frac = 0 # error in f or gamma as a fraction of gamma

qcm.refh = 3
qcm.refto = 0
qcm.calctype = 'LL'

nh = [3,5,3]

harms = [1, 3, 5]


def delf_df(fdelfstar):
    """ reconstruct delfstar dic """
    df = pd.DataFrame({})
    df[1] = [fdelfstar[1]]
    df[3] = [fdelfstar[3]]
    df[5] = [fdelfstar[5]]
    return df



film_prop_set = {
    'grho_Pag_cm3': 1e9, # in Pa g/cm^3
    'phi_deg': 5.729577951308232, # in deg. = 0.1 rad
    'drho_um': 5, # in um 
    'n': 3, 
}

film_prop_inh2o_expt = {# real data in water 20210728 S#295
    'grho_Pag_cm3': 1.75241714e9, # in Pa g/cm^3
    'phi_deg': 3.56994297, 
    'drho_um': 3.06769369, # in um 
    'n': 3, 
}

film_prop_inair_expt = {# real data in air 20210728 S#292
    'grho_Pag_cm3': 1.713e9, # in Pa g/cm^3
    'phi_deg': 3.7667, 
    'drho_um': 3.0562, # in um 
    'n': 3, 
}



# film_prop = film_prop_set
film_prop = film_prop_inh2o_expt
# film_prop = film_prop_inair_expt

## film layers:
# 0: electrode
# 1: film
# 2: water

# Build the layers for qcm
layers_012 = {
    0: {'calc': False, 'drho': 2.8e-06, 'grho': 3e+17, 'phi': 0, 'n': 3}, 
    1: {'calc': True, 'drho': film_prop['drho_um']*1e-3, 'grho': film_prop['grho_Pag_cm3']*1e3, 'phi': film_prop['phi_deg']*np.pi/180, 'n': film_prop['n']}, # phi=0.1 rad
    2: {'calc': False, 'drho': np.inf, 'grho': 1e8, 'phi': np.pi/2, 'n': 3},
}
layers_0 = {
    0: {'calc': True, 'drho': 2.8e-06, 'grho': 3e+17, 'phi': 0, 'n': 3},
} 
layers_01 = {
    0: {'calc': False, 'drho': 2.8e-06, 'grho': 3e+17, 'phi': 0, 'n': 3}, 
    1: {'calc': True, 'drho': film_prop['drho_um']*1e-3, 'grho': film_prop['grho_Pag_cm3']*1e3, 'phi': film_prop['phi_deg']*np.pi/180, 'n': film_prop['n']}, # phi=0.1 rad
}
layers_02 = {
    0: {'calc': False, 'drho': 2.8e-06, 'grho': 3e+17, 'phi': 0, 'n': 3}, 
    2: {'calc': True, 'drho': np.inf, 'grho': 1e8, 'phi': np.pi/2, 'n': 3}
}


# Build the layers for qcmfun
flayers_012 = {
    'electrode': {'drho': 2.8e-06, 'grho3': 3e+17, 'phi': 0}, 
    'film': {'drho': film_prop['drho_um']*1e-3, 'grho3': film_prop['grho_Pag_cm3']*1e3, 'phi': film_prop['phi_deg']}, # phi=0.1 rad
    'overlayer': {'drho': np.inf, 'grho3': 1e8, 'phi': 90},
}
flayers_0 = {
    'film': {'drho': 2.8e-06, 'grho3': 3e+17, 'phi': 0},
} 
flayers_01 = {
    'electrode': {'drho': 2.8e-06, 'grho3': 3e+17, 'phi': 0}, 
    'film': {'drho': film_prop['drho_um']*1e-3, 'grho3': film_prop['grho_Pag_cm3']*1e3, 'phi': film_prop['phi_deg']}, # phi=0.1 rad
}
flayers_02 = {
    'electrode': {'drho': 2.8e-06, 'grho3': 3e+17, 'phi': 0}, 
    'film': {'drho': np.inf, 'grho3': 1e8, 'phi': 90, 'n': 3}
}


# %% calc delfstar of all layers
# all ref to 0
delfstar_012 = {
    n: qcm.calc_delfstar(n, layers_012) for n in harms
}
# we change to refto=None to get the electrode layer shift itself
qcm.refto = None
delfstar_0 = {
    n: qcm.calc_delfstar(n, layers_0) for n in harms
}
qcm.refto = 0 # change it back
delfstar_01 = {
    n: qcm.calc_delfstar(n, layers_01) for n in harms
}
delfstar_02 = {
    n: qcm.calc_delfstar(n, layers_02) for n in harms
}

delfstar_012_2 = {
    n: delfstar_012[n]-delfstar_02[n] for n in harms
}

print('qcm\nlayers_012\n{}\nlayers_0(electrode)\n{}\nlayers_01\n{}\nlayers_02\n{}\nlayers_012_2\n{}\n'.format(delfstar_012, delfstar_0, delfstar_01, delfstar_02, delfstar_012_2))


# qcmfun
fdelfstar_012 = {
    n: qcmfun.calc_delfstar(n, flayers_012, calctype='LL', reftype='bare') for n in harms
}
fdelfstar_0 = {
    n: qcmfun.calc_delfstar(n, flayers_0, calctype='LL', reftype='bare') for n in harms
}
fdelfstar_01 = {
    n: qcmfun.calc_delfstar(n, flayers_01, calctype='LL', reftype='bare') for n in harms
}
fdelfstar_02 = {
    n: qcmfun.calc_delfstar(n, flayers_02, calctype='LL', reftype='bare') for n in harms
}
fdelfstar_012_02 = {
    n: fdelfstar_012[n]-fdelfstar_02[n] for n in harms
}
print('qcmfcn\nflayers_012\n{}\nflayers_0(electrode)\n{}\nflayers_01\n{}\nflayers_02\n{}\nflayers_012_02\n{}\n'.format(fdelfstar_012, fdelfstar_0, fdelfstar_01, fdelfstar_02, fdelfstar_012_02))


# %% calculate layers_012 reference to 0
new_layers_012 = copy.deepcopy(layers_012)
# refto == 0 and delfstar_tot 
# works the same as
# refto == 1 and delfstar_film

qcm.refto = 0
grho_refh, phi, drho, dlam_refh, err = qcm.solve_general_delfstar_to_prop(nh, delfstar_012, new_layers_012, calctype='LL', bulklimit=.5)

print(new_layers_012)
print('grho_refh', grho_refh)
print('phi', phi)
print('drho', drho)
print('dlam_refh', dlam_refh)
print('err', err)

# claculate back with the layers
delfstar_012_new = {
    n: qcm.calc_delfstar(n, new_layers_012) for n in harms
}
print(delfstar_012_new)
print(delfstar_012)


print('\n\n')
print('\n\n')

# calculate layers_012 reference to 2
new_layers_012_2 = copy.deepcopy(layers_012)
# refto == 0 and delfstar_tot 
# works the same as
# refto == 1 and delfstar_film

qcm.refto = 1
grho_refh, phi, drho, dlam_refh, err = qcm.solve_general_delfstar_to_prop(nh, delfstar_012_2, new_layers_012, calctype='LL', bulklimit=.5)

print(new_layers_012_2)
print('grho_refh', grho_refh)
print('phi', phi)
print('drho', drho)
print('dlam_refh', dlam_refh)
print('err', err)

# claculate back with the layers
delfstar_012_2_new = {
    n: qcm.calc_delfstar(n, new_layers_012) for n in harms
}
print(delfstar_012_2)
print(delfstar_012_2_new)


# %%
# qcmfun
df = delf_df(fdelfstar_012)

fprop = qcmfun.solve_for_props(df, '353', calctype='LL', reftype='bare', overlayer={'drho': np.inf, 'grho3': 1e8, 'phi': 90})
print('grho3:', fprop.grho3.values[0])
print('phi:', fprop.phi.values[0])
print('drho:', fprop.drho.values[0])

print('\n\n')

# qcmfun
df_2 = delf_df(fdelfstar_012_02)

fprop_2 = qcmfun.solve_for_props(df_2, '353', calctype='LL', reftype='overlayer', overlayer={'drho': np.inf, 'grho3': 1e8, 'phi': 90})
print('grho3:', fprop_2.grho3.values[0])
print('phi:', fprop_2.phi.values[0])
print('drho:', fprop_2.drho.values[0])







# %%
################################





# expt delfstar






######################################

# %%

delfstar_01_expt = {# real data in air 20210728 S#292
  1: -17498 + 1j*1.3951,
  3: -52552 + 1j*56.795,
  5: -90065 + 1j*280.49,
}

delfstar_012_0_expt = { #20210728 S#295 refto air
    1: -18377 + 1j*734.1,
    3: -54373 + 1j*1249.9,
    5: -92873 + 1j*2061.9,
}

delfstar_012_02_expt = { #20210728 S#295 ref to water
    1: -17623 - 1j*0.26011,
    3: -52954 + 1j*123.33,
    5: -90924 + 1j*604.99,
}


# %% use the expt data
# qcmfun
df = delf_df(delfstar_012_0_expt)

fprop = qcmfun.solve_for_props(df, '353', calctype='LL', reftype='bare', overlayer={'drho': np.inf, 'grho3': 1e8, 'phi': 90})
print('grho3:', fprop.grho3.values[0])
print('phi:', fprop.phi.values[0])
print('drho:', fprop.drho.values[0])
print('df_calc1:', fprop.df_calc1.values[0])
print('df_calc3:', fprop.df_calc3.values[0])
print('df_calc5:', fprop.df_calc5.values[0])
print(delfstar_012_0_expt)

print('\n\n')

# qcmfun
df_2 = delf_df(delfstar_012_02_expt)

fprop_2 = qcmfun.solve_for_props(df_2, '353', calctype='LL', reftype='overlayer', overlayer={'drho': np.inf, 'grho3': 1e8, 'phi': 90})
print('grho3:', fprop_2.grho3.values[0])
print('phi:', fprop_2.phi.values[0])
print('drho:', fprop_2.drho.values[0])



# %%
new_layers_012 = copy.deepcopy(layers_012)
# refto == 0 and delfstar_tot 
# works the same as
# refto == 1 and delfstar_film

qcm.refto = 0
grho_refh, phi, drho, dlam_refh, err = qcm.solve_general_delfstar_to_prop(nh, delfstar_012_0_expt, new_layers_012, calctype='LL', bulklimit=.5)

print(new_layers_012)
print('grho_refh', grho_refh)
print('phi', phi)
print('drho', drho)
print('dlam_refh', dlam_refh)
print('err', err)

# claculate back with the layers
delfstar_012_0_exptcalc = {
    n: qcm.calc_delfstar(n, new_layers_012) for n in harms
}
print(delfstar_012_0_exptcalc)
print(delfstar_012_0_expt)


print('\n\n')
print('\n\n')

# calculate layers_012 reference to 2
new_layers_012_2 = copy.deepcopy(layers_012)
# refto == 0 and delfstar_tot 
# works the same as
# refto == 1 and delfstar_film

qcm.refto = 1
grho_refh, phi, drho, dlam_refh, err = qcm.solve_general_delfstar_to_prop(nh, delfstar_012_02_expt, new_layers_012_2, calctype='LL', bulklimit=.5)

print(new_layers_012_2)
print('grho_refh', grho_refh)
print('phi', phi)
print('drho', drho)
print('dlam_refh', dlam_refh)
print('err', err)

# claculate back with the layers
delfstar_012_02_exptcalc = {
    n: qcm.calc_delfstar(n, new_layers_012) for n in harms
}
print(delfstar_012_02_exptcalc)
print(delfstar_012_02_expt)

