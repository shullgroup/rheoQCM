#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:19:59 2018 and updated continuously
most recent version at the following link:
https://github.com/shullgroup/rheoQCM/tree/master/QCMFuncs/QCM_functions.py
@author: Ken Shull (k-shull@northwestern.edu)
"""

import numpy as np
import sys
import os
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
import time
import shutil
from mpmath import findroot
from scipy.io import loadmat
from pylab import meshgrid
import pandas as pd
from copy import deepcopy, copy
import matplotlib.gridspec as gridspec

try:
  from kww import kwwc, kwws
  # kwwc returns: integral from 0 to +infinity dt cos(omega*t) exp(-t^beta)
  # kwws returns: integral from 0 to +infinity dt sin(omega*t) exp(-t^beta)
except ImportError:
  pass

# setvvalues for standard constants
Zq = 8.84e6  # shear acoustic impedance of at cut quartz
f1 = 5e6  # fundamental resonant frequency
    
openplots = 4
drho_q = Zq/(2*f1)
e26 = 9.65e-2

# Half bandwidth of unloaed resonator (intrinsic dissipation on crystalline quartz)
g0_default = 50

T_coef_default = {'f': {1: [0.00054625, 0.04338, 0.08075, 0],
                       3: [0.0017, -0.135, 8.9375, 0],
                       5: [0.002825, -0.22125, 15.375, 0]},
                  'g': {1: [0, 0, 0, 0],
                       3: [0, 0, 0, 0],
                       5: [0, 0, 0, 0]}}

electrode_default = {'drho': 2.8e-3, 'grho3': 3.0e14, 'phi': 0}
water = {'drho':np.inf, 'grho3':1e8, 'phi':90}
air = {'drho':np.inf, 'grho3':0, 'phi':90}

# make an uncertainty dictionary we'll use for everything
# the values of 0.05 in err_frac may be a bit high
uncertainty_dict_default = {'err_frac':[0.05,0.05]}
for n in [1,3,5,7,9]:
    uncertainty_dict_default[n] = [n*15, 2]
    
# make dictionary of default titles
titles_default =  ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)',
                   '(g)', '(h)']
    
# make a dictionary of the potential axis labels
axlabels = {'grho3': r'$|G_3^*|\rho$ (Pa $\cdot$ g/cm$^3$)',
            'phi': r'$\phi$ (deg.)',
            'tanphi': r'$\tan \phi$',
            'grho3p': r'$G^\prime_3\rho$ (Pa $\cdot$ g/cm$^3$)',
            'grho3pp': r'$G^{\prime\prime}_3\rho$ (Pa $\cdot$ g/cm$^3$)',
            'drho': r'$d\rho$ ($\mu$m$\cdot$g/cm$^3$)',
            'drho_nm':r'$d\rho$ (nm$\cdot$g/cm$^3$)',
            'drho_change': r'change in $d\rho$ ($\mu$m$\cdot$g/cm$^3$)',
            'drho_change_nm': r'change in $d\rho$ (nm$\cdot$g/cm$^3$)',
            'drho_norm': r'$d\rho / (d\rho)_{dry}$',
            'jdp': r'$J^{\prime \prime}/\rho$ (Pa$^{-1}\cdot$cm$^3$/g)',
            'temp':r'$T$ ($^\circ$C)',
            'etarho3':r'$|\eta_3^*| \rho$ (mPa$\cdot$s$\cdot$g/cm$^3$)',
            'delf':r'$\Delta f$ (Hz)',
            'delg':r'$\Delta \Gamma$ (Hz)',
            'delf_n':r'$\Delta f/n$ (Hz)',
            'delg_n':r'$\Delta \Gamma /n$ (Hz)',
            's':'t (s)',
            'min':'t (min)',
            'hr':'t (hr)',
            'day':'t (day)',
            'index':'index'}

def sig_figs(x, n):
    # rounds x to n significant figures
    if x == np.nan or x == np.inf:
        return x
    else:
        round_num = -int(np.floor(np.log10(abs(x))))+n-1
    return (round(x, round_num))

def find_nearest_idx(values, array):
    """
    Find index of a point with value closest to the one specified.
    args:
        values (list or numpy array):
            Values that we want to be close to.
        array (numpy array):
            Input array to choose values from.

    returns:
        idx (numpy array):
            Array of indices in 'array' that give values closest
            to 'values'.
    """
    # make values a numpy array if it isn't already
    values = np.asarray(values).reshape(1, -1)[0, :]
    idx = np.zeros(values.size, dtype=int)
    for i in np.arange(values.size):
        idxval = np.searchsorted(array, values[i], side="left")
        if idxval > 0 and (idxval == len(array) or
                           np.abs(values[i] - array[idxval-1]) <
                        np.abs(values[i] - array[idxval])):
            idx[i] = idxval-1
        else:
            idx[i] = idxval
    return idx


def find_idx_in_range(x, range):
    """
    Find indices of array with values within a specified range.
    args:
        x (numpy array):
            Input array containing the values of interest.
        range (2 element list or numpy array):
            Minimum and maximum values of the specified range.

    returns:
        idx (numpy array):
            Numpy array of indices of x that are within range.
    """
    if range[0] == range[1]:
        idx = np.arange(x.shape[0]).astype(int)
    else:
        idx = np.where((x >= range[0]) &
                       (x <= range[1]))[0]
    return idx


def add_eta_axis(ax):
    """
    Add right-hand y-axis to ghro3 plot with value for complex viscosity

    Parameters
    ----------
    ax : axis
        The axis that we are adding to.

    Returns
    -------
    Returns the new axis, but we generally don't use that for anything.

    """
    ax2 = ax.twinx()  
    omega = 2*np.pi*3*5e6
    axlim = ax.get_ylim()
    ylim2 = tuple((lim/omega)*1000 for lim in axlim)
    ax2.set_ylim(ylim2)
    ax2.set_ylabel(axlabels['etarho3'])
    return ax2
    

def add_D_axis(ax):
    """
    Add right hand axis with dissipation.
    args:
        ax (axis handle):
            Axis that we are going to work with.

    returns:
        axD (axis handle):
            Aix with dissipation added.
    """
    axD = ax.twinx()
    axD.set_ylabel(r'$\Delta D_n$ (ppm)')
    axlim = ax.get_ylim()
    ylim = tuple(2*lim/5 for lim in axlim)
    axD.set_ylim(ylim)
    return axD


def sauerbreyf(n, drho, **kwargs):

    """
    Find indices of array with values within a specified range.
    args:
        x (numpy array):
            Input array containing the values of interest.
        range (2 element list or numpy array):
            Minimum and maximum values of the specified range.

    returns:
        idx (numpy array):
            Numpy array of indices of x that are within range.
    """
    return 2*n*f1 ** 2*drho/Zq


def sauerbreym(n, delf):
    """
    Calculate Sauerbrey thickness from frequency shift.
    args:
        n (int):
            Harmonic of interest.
        delf (real):
            Frequency shift in Hz.

    returns:
        Sauerbrey mass in kg/m^2.
    """
    return -delf*Zq/(2*n*f1 ** 2)


def etarho(n, props):
    """
    Use power law formulation to get |eta*|rho at specified harmonic,
    with properties at n=3 as an input.

    args:
        n (int):
            Harmonic of interest.
        props (dictionary):
            Dictionary of material properties, which must contain
            grho3 and phi. This can also be a dictionary of dictionaries,
            in which case a numpy array of viscosity values is returned.

    returns:
        |eta*|rho at harmonic of interest in SI units.
    """

    # first handle the case where we have a single dictionary of properties
    if 'grho3' in props.keys():
        grho_mag = grho(n, props)
        etarho_mag = grho_mag/(2*np.pi*f1*n)

    # otherwise we handle the case where we are calculating for a dictionary
    # of property values, generally named for the indices in a dataframe
    else:
        etarho_mag = np.array([])
        for key in props.keys():
            grho_mag = grho(n, props[key])
            etarho_val = grho_mag/(2*np.pi*f1*n)
            etarho_mag = np.append(etarho_mag, etarho_val)
    return etarho_mag


def grho(n, props):
    """
    Use power law formulation to get G*\rho at different harmonics,
    with properties at n=3 as an input.
    args:
        n (int):
            Harmonic of interest.
        props (dictionary:
            Dictionary of material properties, which must contain
            grho3 and phi.

    returns:
        G*\rho at harmonic of interest.
    """
    grho3 = props['grho3']
    phi = props['phi']
    return grho3*(n/3) ** (phi/90)


def calc_grho3(n, grhostar):
    """
    Use power law formulation to get |G*|\rho at n=3.
    args:
        n (int):
            Harmonic of interest.
        grhostar (complex):
            G*\rho at harmonic of interest.

    returns:
        grho3:
            |G*|\rho (SI units) at n=3.
        phi:
            phase angle (degrees) - assumed independent of n.
    """
    phi = np.angle(grhostar, deg=True)
    grhon = abs(grhostar)
    grho3 = grhon*(3/n)**(phi/90)
    return grho3, phi


def calc_jdp(grho):
    """
    Calculte the loss compliance, J", normalized by rho, with Gstar*rho
    as the input.
    args:
        grho:
            complex Gstar multiplied by density

    returns:
        jdp:
            imaginary part of complex Jstar (shear compliance), divided by
            density
    """
    return (1/abs(grho))*np.sin(np.angle(grho))


def grho_from_dlam(n, drho, dlam, phi):
    """
    Obtain |G*|\rho from d/lambda.
    args:
        n (int):
            Harmonic of interest.
        drho (real):
            Mass thickness in kg/m^2.
        dlam (real):
            d/lambda at harmonic of interest
        phi (real):
            phase angle

    returns:
        G*\rho at harmonic of interest.
    """
    return (drho*n*f1*np.cos(np.deg2rad(phi/2))/dlam) ** 2


def grho_bulk(n, delfstar):
    """
    Obtain |G*|\rho from for bulk material (infinite thicknes).
    args:
        n (int):
            Harmonic of interest.
        delfstar (complex number or numpy array of complex numbers):
            Complex frequency shift in Hz.

    returns:
        |G*|\rho at harmonic of interest.
    """

    return (np.pi*Zq*abs(delfstar[n])/f1) ** 2


def phi_bulk(n, delfstar):
    """
    Obtain phase angle for bulk material (infinite thicknes).
    args:
        n (int):
            Harmonic of interest.
        delfstar (complex number or numpy array of complex numbers):
            Complex frequency shift in Hz.
    
    returns:
        phase angle at harmonic of interest.
    """
    return -np.degrees(2*np.arctan(np.real(delfstar[n]) /
                       np.imag(delfstar[n])))


def deltarho_bulk(n, delfstar, **kwargs):
    """
    Calculate decay length multiplied by density for bulk material.
    args:
        n (int):
            Harmonic of interest.
        delfstar (complex number or numpy array of complex numbers):
            Complex frequency shift in Hz.
            
    returns:
        Decay length multiplied by density (SI units).
    """
    return -Zq*abs(delfstar[n])**2/(2*n*f1**2*delfstar[n].real)


def calc_D(n, props, delfstar, calctype):
    """
    Calculate D (dk*, thickness times complex wave number).
    args:
        n (int):
            Harmonic of interest.
        props (dictionary):
            Dictionary of material properties, which must contain
            grho3 and phi.
        delfstar (complex):
            Complex frequency shift at harmonic of interest (Hz).
        calctype (string):
            - 'SLA' (default): small load approximation with power law model
            - 'LL': Lu Lewis equation, using default or provided electrode props
            - 'Voigt': small load approximation

    returns:
        D (complex) at harmonic of interest.
    """
    drho = props['drho']
    # set switch to handle ase where drho = 0
    if drho == 0:
        return 0
    else:
        return 2*np.pi*(n*f1+delfstar)*drho/zstar_bulk(n,
                       props, calctype)


def zstar_bulk(n, props, calctype):
    """
    Calculate complex acoustic impedance for bulk material.
    args:
        n (int):
            Harmonic of interest.
        props (dictionary):
            Dictionary of material properties, which must contain
            grho3 and phi.
        calctype (string):
            - 'SLA' (default): small load approximation with power law model
            - 'LL': Lu Lewis equation, using default or provided electrode props
            - 'Voigt': small load approximation

    returns:
        square root of gstar*\rho.
    """
    grho3 = props['grho3']
    if calctype != 'Voigt':
        grho = grho3*(n/3)**(props['phi']/90)
        grhostar = grho*np.exp(1j*np.pi*props['phi']/180)
    else:
        # Qsense version: constant G', G" linear in omega
        greal = grho3*np.cos(np.radians(props['phi']))
        gimag = (n/3)*np.sin(np.radians(props['phi']))
        grhostar = (gimag**2+greal**2)**(0.5)*(np.exp(1j *
                 np.radians(props['phi'])))
    return grhostar ** 0.5


def calc_delfstar_sla(ZL):
    """
    Calculate complex frequency shift from load impedance using small
    load approximation.
    args:
        ZL (complex):
            complex load impedance (SI units).

    returns:
        Complex frequency shift, delfstar (Hz).
    """
    return f1*1j*ZL/(np.pi*Zq)

def calc_ZL_sla(delfstar):
    """
    Calculate complex load impedance from complex freq. shift using small
    load approximation (the inverse of calc_dlfstar_sla)
    args:
        delfstar (complex):
            complex frequency shift relative to bare crystal.

    returns:
        Complex load impedance in SI units .
    """
    return -1j*np.pi*Zq/f1


def calc_ZL(n, layers, delfstar, calctype):
    """
    Calculate complex load impendance for stack of layers of known props.
    args:
        n (int):
            Harmonic of interest.

        layers (dictionary):
            Dictionary of material dictionaries specifying the properites of
            each layer. These dictionaries are labeled by integers from 
            Layer_min to Layer_max, with Layer_min
            being the layer in contact with the QCM.  Each material dictionary must
            include values for 'grho3, 'phi' and 'drho'. The dictionary for
            Layer_max can include the film impedance, Zf.

        delfstar (complex):
            Complex frequency shift at harmonic of interest (needed for Lu-
            Lewis calculation).

        calctype (string):
            - 'SLA' (default): small load approximation with power law model
            - 'LL': Lu Lewis equation, using default or provided electrode props
            - 'Voigt': small load approximation

    returns:
        ZL (complex):
            Complex load impedance (stress over velocity) in SI units
    """

    N = len(layers)
    Z = {}; D = {}; L = {}; S = {}
    layer_nums = layers.keys()
    layer_min = min(layer_nums)
    layer_max = max(layer_nums)

    # we use the matrix formalism to avoid typos and simplify the extension
    # to large N.
    for i in np.arange(layer_min, layer_max):
        Z[i] = zstar_bulk(n, layers[i], calctype)
        D[i] = calc_D(n, layers[i], delfstar, calctype)
        L[i] = np.array([[np.cos(D[i])+1j*np.sin(D[i]), 0],
                 [0, np.cos(D[i])-1j*np.sin(D[i])]])

    # get the terminal matrix from the properties of the last layer
    if 'Zf' in layers[layer_max].keys():
        Zf_max = layers[layer_max]['Zf'][n]
    else:
        D[layer_max] = calc_D(n, layers[layer_max], delfstar, calctype)
        Zf_max = 1j*zstar_bulk(n, layers[layer_max], calctype)*np.tan(D[layer_max])

    # if there is only one layer, we're already done
    if N == 1:
        return Zf_max

    Tn = np.array([[1+Zf_max/Z[layer_max-1], 0],
          [0, 1-Zf_max/Z[layer_max-1]]])

    uvec = L[layer_max-1]@Tn@np.array([[1.], [1.]])

    for i in np.arange(layer_max-2, 0, -1):
        S[i] = np.array([[1+Z[i+1]/Z[i], 1-Z[i+1]/Z[i]],
          [1-Z[i+1]/Z[i], 1+Z[i+1]/Z[i]]])
        uvec = L[i]@S[i]@uvec

    rstar = uvec[1, 0]/uvec[0, 0]
    return Z[1]*(1-rstar)/(1+rstar)

def delete_layer(old_layers, num):
    # removes l layer, shifting all the higher levels down 1
    # also works for highest layer
    layer_nums = old_layers.keys()
    layer_max = max(layer_nums)
    
    new_layers = deepcopy(old_layers)
    # delete the relevant layer and all higher layers
    for i in np.arange(num, layer_max+1):
        del new_layers[i]
    
    # now copy the relevant higher levels back in 
    # skip this step for num<1
    if num>=1:
        for i in np.arange(num,layer_max):
            new_layers[i] = old_layers[i+1]

    return new_layers



def calc_delfstar(n, layers, **kwargs):
    """
    Calculate complex frequency shift for stack of layers.
    args:
        n (int):
            Harmonic of interest.

        layers (dictionary):
            Dictionary of material dictionaries specifying the properites of
            each layer. 
            Generally a dictionary with 'film' that may also contain
            'overlayer' and/or 'electrode'.  
            If it doesn't contain 'film' it contains dictionaries
            labeled from 1 to N, with 1
            being the layer in contact with the QCM.  Each dictionary must
            include values for 'grho3, 'phi' and 'drho'.

    kwargs:
        calctype (string):
            - 'SLA' (default): small load approximation with power law model
            - 'LL': Lu Lewis equation, using default or provided electrode props
            - 'Voigt': small load approximation

        reftype (string)
           - 'bare' (default): ref is determined by removing layer 1 and 2 
               from the stack
        - 'overlayer' if layer 2 exists then the reference
               is determined by removing layer 1 from the layers stack
 

    returns:
        delfstar (complex):
            Complex frequency shift (Hz).
    """

    calctype = kwargs.get('calctype', 'SLA')
    reftype = kwargs.get('reftype', 'bare')
    if not layers:  # if layers is empty {}
        return np.nan

    ZL = calc_ZL(n, layers, 0, calctype)
    if (reftype=='overlayer') and (2 in layers.keys()):
        layers_ref = delete_layer(layers, 1)
        ZL_ref = calc_ZL(n, layers_ref, 0, calctype)
    else:
        ZL_ref = 0
    
    del_ZL = ZL-ZL_ref
    
    if calctype != 'LL':
        # use the small load approximation in all cases where calctype
        # is not explicitly set to 'LL'
        return calc_delfstar_sla(del_ZL)

    else:
        # this is the most general calculation
        # use default electrode if it's not specified
        layers_all = deepcopy(layers)
        layers_ref = deepcopy(layers)
        layers_all = {1: layers['electrode'], 2: layers['film']}
        layers_ref = {1: layers['electrode']}
        if 0 not in layers:
            layers[0] = electrode_default
        
        # not totally sure reftype is accounted or in all of this.
        layers_all = deepcopy(layers)
        layers_ref = delete_layer(deepcopy(layers), 1)

        ZL_all = calc_ZL(n, layers_all, 0, calctype)
        delfstar_sla_all = calc_delfstar_sla(ZL_all)
        ZL_ref = calc_ZL(n, layers_ref, 0, calctype)
        delfstar_sla_ref = calc_delfstar_sla(ZL_ref)

        def solve_Zmot(x):
            delfstar = x[0] + 1j*x[1]
            Zmot = calc_Zmot(n,  layers_all, delfstar, calctype)
            return [Zmot.real, Zmot.imag]

        sol = optimize.root(solve_Zmot, [delfstar_sla_all.real,
                                         delfstar_sla_all.imag])
        dfc = sol.x[0] + 1j * sol.x[1]

        def solve_Zmot_ref(x):
            delfstar = x[0] + 1j*x[1]
            Zmot = calc_Zmot(n,  layers_ref, delfstar, calctype)
            return [Zmot.real, Zmot.imag]

        sol = optimize.root(solve_Zmot_ref, [delfstar_sla_ref.real,
                                             delfstar_sla_ref.imag])
        dfc_ref = sol.x[0] + 1j * sol.x[1]
        return dfc-dfc_ref


def calc_Zmot(n, layers, delfstar, calctype, **kwargs):
    """
    Calculate motional impedance (used for .
    args:
        n (int):
            Harmonic of interest.

        layers (dictionary):
            Dictionary of material dictionaries specifying the properites of
            each layer. These dictionaries are labeled from 1 to N, with 1
            being the layer in contact with the QCM.  Each dictionary must
            include values for 'grho3, 'phi' and 'drho'.

        calctype (string):
            Generally passed from calling function.  Should always be 'LL'.

    kwargs:
        g0 (real):
            Dissipation at n for dissipation.  Default value set at top
            of QCM_functions.py (typically 50).

    returns:
        delfstar (complex):
            Complex frequency shift (Hz).
    """
    g0 = kwargs.get('g0', g0_default)
    om = 2 * np.pi * (n*f1 + delfstar)
    Zqc = Zq * (1 + 1j*2*g0/(n*f1))

    Dq = om*drho_q/Zq
    secterm = -1j*Zqc/np.sin(Dq)
    ZL = calc_ZL(n, layers, delfstar, calctype)
    # eq. 4.5.9 in Diethelm book
    thirdterm = ((1j*Zqc*np.tan(Dq/2))**-1 +
                 (1j*Zqc*np.tan(Dq/2) + ZL)**-1)**-1
    Zmot = secterm + thirdterm
    # uncomment next 4 lines to account for piezoelectric stiffening
    # dq = 330e-6  # only needed for piezoelectric stiffening calc.
    # epsq = 4.54; eps0 = 8.8e-12; C0byA = epsq * eps0 / dq; ZC0byA = C0byA / (1j*om)
    # ZPE = -(e26/dq)**2*ZC0byA
    # Zmot=Zmot+ZPE
    return Zmot


def calc_dlam(n, film):
    """
    Calculate d/lambda at specified harmonic.
    args:
        n (int):
            Harmonic of interest.

        film (dictionary):
            Dictionary of film properties, including values for
            'grho3, 'phi' and 'drho'.

    returns:
        delfstar (real):
            d/lambda at specified harmonic.
    """
    return calc_D(n, film, 0, 'SLA').real/(2*np.pi)

def calc_dlam_from_dlam3(n, dlam3, phi):
    """
    Calculate d/lamda at specified harmonic from its value at n/3
    args:
        n (int):
            Harmonic of interest.
        
        dlam3 (real):
            Value of d/lambda at n=3
            
        phi (real):
            phase angle in degrees
    """
    return dlam3*(n/3)**(1-phi/180)
    

def calc_lamrho(n, grho3, phi):
    """
    Calculate lambda*\rho at specified harmonic.
    args:
        n (int):
            Harmonic of interest.

        grho3 (real):
            |G*|\rho  at n=3 (SI units).

        phi (real):
            Phase angle (degrees).

    returns:
        shear wavelength times density in SI units
    """
    # calculate lambda*rho
    grho = grho3*(n/3) ** (phi/90)
    return np.sqrt(grho)/(n*f1*np.cos(np.deg2rad(phi/2)))


def calc_deltarho(n, grho3, phi):
    """
    Calculate delta*\rho at specified harmonic.
    args:
        n (int):
            Harmonic of interest.

        grho3 (real):
            |G*|\rho  at n=3 (SI units).

        phi (real):
            Phase angle (degrees).

    returns:
        decay length times density in SI units
    """
    # calculate delta*rho (decay length times density)
    return calc_lamrho(n, grho3, phi)/(2*np.pi*np.tan(np.radians(phi/2)))


def phi_from_grho3_sadman(grho3):
    # linear relationship between phi and Grho3 suggested
    # by Kazi's hydrophobic polyelectrolyte complex paper
    logG = np.log10(grho3)
    if logG <=5:
        return 90
    elif logG >=9:
        return 0
    else:
        return 90-90*(logG-5)/4


def dlam(n, dlam3, phi):
    """
    Calculate d/lambda at specified harmonic.
    args:
        n (int):
            Harmonic of interest.

        dlam3 (real):
            d/lambda at n=3.

        phi (real):
            Phase angle (degrees).

    returns:
        d/lambda at specified harmonic
    """
    return dlam3*(int(n)/3) ** (1-phi/180)


def normdelfstar(n, dlam3, phi):
    """
    Calculate complex frequency shift normzlized by Sauerbrey shift.
    args:
        n (int):
            Harmonic of interest.

        dlam3 (real):
            d/lambda at n=3.

        phi (real):
            Phase angle (degrees).

    returns:
        delfstar normalized by Sauerbrey value
    
    """

    return -np.tan(2*np.pi*dlam(n, dlam3, phi)*(1-1j*np.tan(np.deg2rad(phi/2)))) / \
            (2*np.pi*dlam(n, dlam3, phi)*(1-1j*np.tan(np.deg2rad(phi/2))))
            
def normdelfstar_liq(n, dlamval, phi, drho, overlayer):
    """
    Calculate normalized complex frequency shift for material immersed in a liquid
    the plot is no longer universal and so depends on the harmonic.  This is for n=3
    Small load approximation assumed here, as we mostly use this for visualization

    Parameters
    ----------
    n : integer
        harmonic of interest
    dlam : TYPE
        d/lambda for the film. - generally assumed to be n=3
    phi : TYPE
        phi for the film.
    drho: 
        drho for the film
    overlayer : dictionary of overlayer properties
        Must include 'grho', and 'phi',  'drho' assumed to be infinite if not listed.

    Returns
    -------
    Complex frequency shift, normalized by Sauerbrey shift of film

    """
    R = calc_delfstar(n, {1:overlayer})/sauerbreyf(n, drho)
    D = 2*np.pi*dlamval*(1-1j*np.tan(phi*np.pi/360))
    return (D**(-2)+R**2)/(1/(D*np.tan(D)) + R)


def normdelf_bulk(n, dlam3, phi):
    """
    Frequency shift normalized by value for bulk material.
    args:
        n (int):
            Harmonic of interest.

        dlam3 (real):
            d/lambda at n=3.

        phi (real):
            Phase angle (degrees).

    returns:
        delf normalized bulk value
    """

    answer = np.real(2*np.tan(2*np.pi*dlam(n, dlam3, phi) *
                          (1-1j*np.tan(np.deg2rad(phi/2)))) /
            (np.sin(np.deg2rad(phi))*(1-1j*np.tan(np.deg2rad(phi/2)))))
    return answer


def normdelg_bulk(n, dlam3, phi):
    """
    Dissipation shift normalized by value for bulk material.
    args:
        n (int):
            Harmonic of interest.

        dlam3 (real):
            d/lambda at n=3.

        phi (real):
            Phase angle (degrees).

    returns:
        delg normalized bulk value
    """

    return -np.imag(np.tan(2*np.pi*dlam(n, dlam3, phi) *
                           (1-1j*np.tan(np.deg2rad(phi/2)))) /
            ((np.cos(np.deg2rad(phi/2)))**2*(1-1j*np.tan(np.deg2rad(phi/2)))))


def rhcalc(calc, dlam3, phi):
    """
    Calculate harmonic ratio from material properties.
    args:
        calc (3 character string):
            Calculation string in format a.b:c.

        dlam3 (real):
            d/lambda at n=3.

        phi (real):
            Phase angle (degrees).

    returns:
        Harmonic ratio.
    """
    nvals = calc.split('_')[0]
    if '.' not in nvals:
        return np.nan
    else:
        return normdelfstar(nvals.split('.')[0], dlam3, phi).real / \
            normdelfstar(nvals.split('.')[1], dlam3, phi).real


def rh_from_delfstar(calc, delfstar):
    """
    Determine harmonic ratio from experimental delfstar.
    args:
        calc (character string):
            Calculation string - requires n1.n2_n3 format

        delfstar (complex):
            ditionary of complex frequency shifts, included all harmonics
            within calc.

    returns:
        Harmonic ratio for n1, n2
    """
    # calc here is the calc string (i.e., '353')
    nf = calc.split('_')[0]
    n1 = int(nf.split('.')[0])
    n2 = int(nf.split('.')[1])
    return (n2/n1)*delfstar[n1].real/delfstar[n2].real


def rdcalc(calc, dlam3, phi):
    """
    Calculate dissipation ratio from material properties.
    args:
        calc (character string):
            Calculation string.  
            Ratio taken for first value of gamma

        dlam3 (real):
            d/lambda at n=3.

        phi (real):
            Phase angle (degrees).

    returns:
        Harmonic ratio.
    """
    n = int(calc.split('_')[-1][0])
    return -(normdelfstar(n, dlam3, phi).imag /
        normdelfstar(n, dlam3, phi).real)


def rd_from_delfstar(n, delfstar):
    """
    Determine dissipation ratio from experimental delfstar.
    args:
        n (integer):
            Harmonic of interest

        delfstar (complex):
            ditionary of complex frequency shifts, harmonic of interest

    returns:
        Dissipation ratio.
    """
    return -delfstar[n].imag/delfstar[n].real


def bulk_props(delfstar):
    """
    Determine properties of bulk material from complex frequency shift.
    
    args:
        delfstar (complex):
            Complex frequency shift (at any harmonic).

    returns:
        grho:
            harmonic where delfstar was measured.
        phi:
            Phase angle in degrees, at harmonic where delfstar was measured.
    """

    grho = (np.pi*Zq*abs(delfstar)/f1) ** 2
    phi = -np.degrees(2*np.arctan(delfstar.real /
                      delfstar.imag))
    return grho, min(phi, 90)


def nvals_from_calc(calc):
    '''
    Get nvalues nf and ng used by solution

    args:
        calc (String):
            Calculation string.

    Returns:
        nf: harmonic used to fit delf
        
        ng: harmonic used to fit delg
        
        n_all:  total number of harmonics that are fit
        
        n_unique:  unique harmonics used in calculation

    '''
    nf = calc.split('_')[0].split('.')
    nf = [int(x) for x in nf]
    
    if len(calc.split('_'))>1:
        ng = calc.split('_')[1].split('.')
        ng = [int(x) for x in ng]
    elif len(calc.split('_')) == 1:
        ng = []
    else:
        sys.exit(f'not an allowed value of calc: {calc}')
    n_all = len(nf) + len(ng)
    n_unique = list(set(nf+ng))
    return nf, ng, n_all, n_unique

def nvals_from_df_soln(df_soln):
    '''
    Get experimental harmonics appearing in solution dataframe

    args:
        df_soln (dataframe):
            solution dataframe.

    Returns:
        nvals: (list of integers)
            experimental harmonics in the dataframe
            
    '''

    nvals = []
    for n in [1,3,5,7,9,11,13,15,17,19]:
        if f'df_expt{n}' in df_soln.columns.values:
            nvals = nvals + [n]
            
    return nvals


def update_calc(calc):
    """
    update calc to current format with underscore separating \
    harmonics used for frequency and dissipation \
    first replace colon with underscore if needed
    """
    calc = calc.replace(':', '_')
    if not '.' in calc and not '_' in calc and len(calc)>1:
        calc = '.'.join(calc)

    # now we add the underscore if needed
    if not '_' in calc:
        if len(calc.split('.'))==3:
            calc = (calc.split('.')[0]+'.'+
                    calc.split('.')[1]+'_'+
                    calc.split('.')[2])
        elif len(calc.split('.')) == 2:
            calc = (calc.split('.')[0]+'_'+
                    calc.split('.')[1])
    return calc

def find_nplot(delfstar):
    # returns the list of harmonics in the dataframe
    nplot = []
    # consider possibility that our data has harmonics up to n=21
    for n in np.arange(1, 22, 2):
        if n in delfstar.keys():
            nplot = nplot + [n]
    return nplot

    
def make_soln_df(delfstar, calc, props_calc, layers_in,  **kwargs):
    """
    Create solution dataframe from input delfstar dataframe
    
    args:
        delfstar (dataframe):
            input dataframe, typically from read_xlsx
            
        calc (string):
            specifies harmonics to use
            
        props_calc (list of strings):
            properties to calculate
            
        layers_in (dictionary):
            input layers dictionary, just used to see how many layers we have
            
    kwargs:
        gmax (float):
            rows with dg>gmax within calc are discarded
            
        props (list of strings):
            properties that we will calculate
            default is ['grho3', 'phi', 'drho']
            
        calctype ('string'):
            Type of calculation to be performed
    
            - 'SLA' (default): small load approximation with power law model.
            
            - 'LL': Lu Lewis equation, using default or provided electrode \
                props.

            - 'Voigt': small load approximation with Voigt model
                     
    """
    gmax = kwargs.get('gmax', 20000)
    calctype = kwargs.get('calctype', 'SLA')
    
    if not 't_next' in delfstar.keys():
        delfstar = add_t_diff(delfstar)
        
    # check to see if there are any nan values in the harmonics we need  
    delfstar_mod = deepcopy(delfstar)
    n_unique = nvals_from_calc(calc)[3]
    delfstar_mod  = delfstar_mod.dropna(subset = n_unique)

    # also set delfstar to nan for gamma exceeding gmax
    for n in n_unique:
          delfstar_mod.drop(delfstar_mod[np.imag(delfstar_mod[n]) > gmax].index, 
                      inplace = True)
    
    df_soln = delfstar_mod[['t', 't_prev', 't_next', 'temp']].copy()

    # now add complex frequency and frequency shifts
    npts = len(df_soln.index)
    complex_series = np.empty(len(df_soln), dtype = np.complex128)
    for n in find_nplot(delfstar):
        df_soln.insert(df_soln.shape[1], f'f_expt{n}', delfstar[f'{n}_dat'])
        df_soln.insert(df_soln.shape[1], 'df_expt'+str(n), delfstar[n])
        df_soln.insert(df_soln.shape[1], 'df_calc'+str(n), complex_series)
    
    # add calc to each row
    calc_array = np.array(npts*[calc])
    df_soln.insert(df_soln.shape[1], 'calc', calc_array)
    
    # add properties used in calculation
    object_series = np.empty(npts, dtype = object)
    df_soln.insert(df_soln.shape[1], 'props_calc', object_series)
    
    # now add columns for properties in all layers
    real_series = np.zeros(npts, dtype=np.float64)
    
    for layernum in layers_in.keys():
        for prop in ['grho3', 'phi', 'drho']:
            df_soln.insert(df_soln.shape[1], f'{prop}_{layernum}', real_series)
        
    # add column for dlam3_1
    df_soln.insert(df_soln.shape[1], 'dlam3_1', real_series)
    
    # now add columns for Jacobian and layers
    df_soln.insert(df_soln.shape[1], 'jacobian', object_series)
    df_soln.insert(df_soln.shape[1], 'layers', object_series)
  
    # now add columns for 'calctype'
    calctype_array = np.array(npts*[calctype])
    df_soln.insert(df_soln.shape[1], 'calctype', calctype_array)
       
    return df_soln, delfstar_mod


def compare_calc_expt(layers, row, calc, **kwargs):
    """
    Compare experimental and calculated values of delfstar
    
    args:
        layers (datafame):
            input properties 
        row (series):
            single row from input delfstar dataframe, obtained from iterrows
            
    kwargs:
        calctype ('string'):
            Type of calculation to be performed
    
            - 'SLA' (default): small load approximation with power law model.
            
            - 'LL': Lu Lewis equation, using default or provided electrode \
                props.

            - 'Voigt': small load approximation with Voigt model
            
        reftype (string):
            Specification of the reference. 
        
            - 'overlayer' (default):  If overlayer exists, then the \
                reference is the overlayer on a bare crystal.
            
            - 'bare': Reference is bare crystal, even if overlayer exists.
            
    returns:
        Difference between experimental and calculated df, dg at harmonics \
            determined by row.calc
            
    """
    calctype = kwargs.get('calctype', 'SLA')
    reftype = kwargs.get('reftype', 'overlayer')
    vals = []
    # update calc to current format
    calc = update_calc(calc)
                       
    # now we figure out which harmonics we use to fit to delg (ng)
    # or delf (nf)
    nf, ng, n_all, n_unique = nvals_from_calc(calc)
    for n in nf: 
        val = (calc_delfstar(n, layers, calctype=calctype,
                               reftype=reftype).real -
                 row[n].real)
        vals.append(val)
    for n in ng: 
        val = (calc_delfstar(n, layers, calctype=calctype,
                               reftype=reftype).imag -
                 row[n].imag)
        vals.append(val)
    return vals


def update_layers(props_calc, values, layers):
    # updates layers dictionary, substituting the specified values
    # into the properties specified by props_calc
    # make sure order of values corresponds to order of props_calc
    for input_string, value in zip(props_calc, values): 
        [prop, layer] = input_string.split('_')
        layer = int(layer)
        if value != 'no_change':
            layers[layer][prop] = value
    return layers


def guess_from_layers(props, layers):
    guess = [None]*len(props)
    for i, input_string in enumerate(props): 
        [prop, layer] = input_string.split('_')
        layer = int(layer)
        guess[i] = layers[layer][prop]
    return guess


def update_df_soln(df_soln, soln, idx, layers, props_calc):
    for layer_num in layers.keys():
        for prop in ['grho3', 'phi', 'drho']:
            df_soln.loc[idx, f'{prop}_{layer_num}'] = layers[layer_num][prop] 
    
    nvals = nvals_from_df_soln(df_soln)
    for n in nvals:
        df_soln.at[idx, f'df_calc{n}']=calc_delfstar(n, layers)
        
    df_soln.at[idx, 'layers'] = deepcopy(layers)
    df_soln.at[idx, 'jacobian'] = (soln['jac']).astype(object)
    df_soln.at[idx, 'dlam3_1'] = calc_dlam(3, layers[1])
    df_soln.at[idx, 'props_calc'] = props_calc
        
    return df_soln


def solve_for_props(delfstar, calc, props_calc, layers_in, **kwargs):
    """
    Solve the QCM equations to determine the properties.

    args:
        delfstar (dataframe):
            Input dataframe containing complex frequency shifts,
            generally generated by read_xlsx.
            
        calc (string):
            Calculation type in form 'x_y.z' or
            x.y_z. Numbers before the _ are harmonics used to fit against
            frequency shift.  Numbers after the _ are the harmonics used to
            fit against dissipation.  A single number means we use the
            Sauerbrey shift.  Two numbers (x_y) means we fix drho at the                  
            specfied value.
            
        props_calc (list of strings):
            properties we want to solve for.  Generally in form
            'grho3_n', 'phi_n', 'drho_n', where n is the layer
            if no n is given ('grho3', for example) n is assumed to be 1
            
        layers (dictionary):
            Dictionary with layer properties
            format is {1:{'grho3':val, 'phi':val, 'drho':val}}
              (with other layers specified in the same way)
            

    kwargs:
        calctype (string):
            Type of calculation to be performed
            
            - 'SLA' (default): small load approximation with power law model.
            
            - 'LL': Lu Lewis equation, using default or provided electrode \
                props.

            - 'Voigt': small load approximation with Voigt model
            
            
        guess (dictionary):
            Dictionary with initial guesses for properties    

        lb (dictionary):  
            dictionary of lower bounds. keys must correspond to props.
            ex: {'ghro3_1:1e8', 'phi_1:0', 'drho_1:0'}
            
        ub (dictionary):  
            dictionary of upper bounds. keys must correspond to props.
            ex: {'ghro3_1:1e13', 'phi_1:90', 'drho_1:0'}

        reftype (string):
            Specification of the reference. 
            
            - 'bare' (default): Reference is bare crystal.
        
            - 'overlayer':  reference corresponds to drho_1=0.
            (only used if layer 2 exists)
            
        gmax (real):
            Maximum value of dissipation (in Hz) for calculation.
            - default is 20,000 Hz
        
        accuracy (real):
            Max difference between actual and back-calculated delf, delg
            - deault is 1 Hz 
            - this is not the uncertainty in delf, delg but is used to check
                that a solution exists
            
    returns:
        df_soln (dataframe):
            Dataframe with properties added, deleting rows with any NaN \
                values that didn't allow calculation to be performed.

    """
    # add layer number as 1 if not specified
    layers = deepcopy(layers_in)
    for i, prop in enumerate(props_calc):
        if len(prop.split('_'))==1:
            props_calc[i]=prop+'_1'
                       
    calctype = kwargs.get('calctype', 'SLA')
    reftype = kwargs.get('reftype', 'bare')
    gmax = kwargs.get('gmax', 20000)
    
    # set default upper and lower bounds and guessfor properties
    default_prop_min = {'grho3':1e4, 'phi':0, 'drho':0}
    default_prop_max = {'grho3':1e13, 'phi':90, 'drho':3e-2}

    lb_default = [None]*len(props_calc)
    ub_default = [None]*len(props_calc)
    
    for i, prop_string in enumerate(props_calc):
        prop = prop_string.split('_')[0]
        lb_default[i] = default_prop_min[prop]
        ub_default[i] = default_prop_max[prop]
    
    lb = np.array(kwargs.get('lb', lb_default))
    ub = np.array(kwargs.get('ub', ub_default))
   
    guess = guess_from_layers(props_calc, layers)
    # make sure guess is in the right bounds
    for k in np.arange(len(guess)):
        guess[k]=max(guess[k], lb[k])
        guess[k]=min(guess[k], ub[k])
      
    # set required accuracy for solution
    accuracy =kwargs.get('accuracy', 1)

    # create df_soln dataframe                   
    df_soln, delfstar_mod = make_soln_df(delfstar, calc, props_calc, layers,
                           gmax=gmax, calctype=calctype) 
  
    for idx, row in delfstar_mod.iterrows(): 
        def ftosolve(x):
            layers_solve = update_layers(props_calc, x, layers)
            return compare_calc_expt(layers_solve, row, calc,
                                     calctype=calctype, reftype=reftype)
        
        try:
            soln = optimize.least_squares(ftosolve, guess, bounds=(lb, ub))
        except:
            print(f'error at index {idx}')
            
        # make sure sufficienty accurate solutions was found
        if soln['fun'].max() > accuracy:
            df_soln.drop(idx, inplace=True)
            continue
        
        layers = update_layers(props_calc, soln['x'], layers)
        df_soln = update_df_soln(df_soln, soln, idx, layers, props_calc)
      
    return df_soln
                             

def restrict_dlam(soln, n, dlam):
    """
    Restrict soln dataframe for d/lanmbd(n)>dlam
    Parameters
    ----------
    soln : dataframe
        solution dataframe, generally generated by solve_for_props.
    n : integer
        harmonic to use for calculation of d/lambda.
    dlam : float
        minimum value of d/lambda to keep in dataframe.

    Returns
    -------
    soln : dataframe
        solution frame with new d/lambda column added (if n !=3), restricted
        to values where d/lambda at this harmonic is greater than critical value.

    """
    if n != 3:
        soln['dlam'+str(n)] = calc_dlam_from_dlam3(n, soln['dlam3'], 
                                                   soln['phi'])
        col = 'dlam'+str(n)
    else:
        col = 'dlam3'
    # the following format comes from
    # https://stackoverflow.com/questions/49781626/pandas-query-with-variable-as-column-name
    soln = soln.query('{0}>@dlam'.format(col))
    return soln


def solve_all(datadir, calc, **kwargs):
    """

    Parameters
    ----------
    datadir : string
        Directory for which solutions are obtained for all .xlsx files.
    **kwargs : 
        optional arguments passed along to read_xlsx, solve_for_props and
        make_prop_axes

    Returns
    -------
    df : dictionary
        dictionary of dataframes returned by read_xlsx
    soln : dictionary
        dictinoary of solutions returned by solve_for_props.
    figinfo : dictionary
        dictionary of figinfo returned by make_prop_axes

    """

    # function to solve for all .xlsx files in a directory

    df = {}
    soln = {}
    figinfo = {}
    
    # create a list of all the .xlsx files in the data directory
    files = glob(os.path.join(datadir, '*.xlsx'))
    
    # now do the analysis on each of these files
    for infile in files:
        plt.close('all')
        # get the filename
        filename = os.path.split(infile)[-1]
        
        # remove the .xlsx to get the prefix 
        prefix = filename.rsplit('.', 1)[0]
        df[prefix] = read_xlsx(infile, **kwargs)
        print('solving '+prefix + ' - ' +calc)
        time.sleep(3)
        soln[prefix] = solve_for_props(df[prefix], calc, **kwargs)
        
        # window title for property plots
        kwargs['num']=os.path.join(datadir, prefix+'_'+calc+'_props.pdf')
        figinfo[prefix] = make_prop_axes(**kwargs)
        plot_props(soln[prefix], figinfo[prefix])
        
        # now set the window title for the solution check
        kwargs['num']=os.path.join(datadir, prefix+'_'+calc+'_check.pdf')
        figinfo[prefix]['fig'].savefig(os.path.join(datadir, prefix+
                                                    '_'+calc+'_props.pdf'))
    return df, soln, figinfo


def make_err_axes(**kwargs):
    """
    kwargs:
        num (string):
            title for plot window
            

    returns:
        fig:
            Figure containing various error plots.
        ax:
            axes of the figure.
    """
    num = kwargs.get('num','error plot')
    fig, ax = plt.subplots(3,3, figsize=(9,9), constrained_layout=True,
                           num = num)
    return fig, ax


def make_err_plot(ax, soln, uncertainty_dict, **kwargs):
    
    # this function needs to be updated to account for the way we now
    # deal with uncertainty+_dict.
    """
    Determine errors in properties based on uncertainies in a delfstar.
    args:
        soln (dataframe):
            Input solution dataframe.
        uncertainty_dict (dictionary):
            Dictionary of uncertainties

    kwargs:
        idx (int or string):
            index of point in soln to use (default is 'min')
            numeric value of 'max' are also possible
        npts (int):
            number of points to include in error plots (default is 10)
        num (string):
            title of plot window
        label (string):
            label used for legend

    """

    # specify specific point in the solution dataframe to use
    idx = kwargs.get('idx', 0)  
    if idx == 'max':
        idx = soln['calc'].index.max()
    elif idx == 'min':
        idx = soln['calc'].index.min()
        
    npts = kwargs.get('npts', 10)
    label = kwargs.get('label', '')

    calctype = soln['calctype'][idx]
    calc = soln['calc'][idx]
        
    guess = {'grho3': soln['grho3'][idx],
             'phi': soln['phi'][idx],
             'drho': soln['drho'][idx]}

    delfstar_0 = {}

    # get list of harmonics we care about
    nvals = list(set(calc.split('.')))
    nvals = list(map(int, nvals))
    for n in nvals:
        delfstar_0[n] = soln['df_expt'+str(n)][idx]

    # now generate series of delfstar values based on the errors
    # set some parameters for the plots
    # frequency or dissipation shift
    mult = np.array([1, 1, 1j], dtype=complex)
    pos = {0:0, 1:0, 2:1}  # used to ef the relevant uncertainty is f or g
    forg = {0: 'f', 1: 'f', 2: '$\Gamma$'}
    prop_type = {0: 'grho3', 1: 'phi', 2: 'drho'}
    scale_factor = {0: 0.001, 1: 1, 2: 1000}

    # intialize values of delfstar
    delfstar_del = {}
    for k in [0, 1, 2]:
        delfstar_del[k] = {}
        for n in nvals:
            delfstar_del[k][n] = np.ones(npts)*delfstar_0[n]
            
    #adjust values of delfstar and calculate properties
    err = {}
    for k in [0, 1, 2]:
        n = int(calc.split('.')[k])
        var = forg[k]
        ax[k, 0].set_ylabel(r'$|G_3|\rho$ (Pa$\cdot$g/cm$^3$)')
        ax[k, 1].set_ylabel(r'$\phi$ (deg.)')
        ax[k, 2].set_ylabel(r'$d\rho$ ($\mu$m$\cdot$g/cm$^3$)')
        for col in [0, 1, 2]:
            ax[k, col].set_xlabel(r'$\Delta${}'.format(var) +
                                r'$_{}$'.format(n) +' (Hz)')

        n = int(calc.split('.')[k]) 
        err[k] = uncertainty_dict[n][pos[k]]
        delta = np.linspace(-err[k], err[k], npts)
        delfstar_del[k][n] = delfstar_del[k][n]+delta*mult[k]
        delfstar_df = pd.DataFrame.from_dict(delfstar_del[k])

        props = solve_for_props(delfstar_df, calc=calc,
                                          calctype=calctype, guess=guess)
        # make the property plots
        for p in [0, 1, 2]:
            ax[k, p].plot(delta, props[prop_type[p]]*scale_factor[p],
                       '-+', label = label)
            ax[k, p].legend()
            
            # now add point for actual solution
            ax[k, p].plot(0, soln[prop_type[p]][idx]*scale_factor[p],'or')

def calc_fstar_err (n, row, uncertainty_dict):
    """
    Calculate uncertainties in delf and delg (expressed as complex delfstar)
    from uncertainty_dict.
    
    Args
    ----------
    n : Integer
        Harmonic of Interest.

    row : Dataframe row
        Dictionary of input values, typically taken from a row of the      
        solution dataframe generated by solve_for_props.

    uncertainty_dict : Dictionary
        Dictionary with the following format 
            n1:[f_err, g_err], n2:[f_err, g_err]..., \\
            'err_frac':[f_frac, g_frac]
        'err_frac' is optional, and is taken as [0,0] if not specified
            
    Returns
    -------
    deflstar_err, where the real component is the uncertainty in delf
    and the imaginary component is the uncertainty in delf.

    """
    if uncertainty_dict == 'zeros':
        return 0
    
    if 'err_frac' not in uncertainty_dict.keys():
        uncertainty_dict['err_frac'] = [0.0, 0.0]
        
    # find the value of gamma for the harmonic of interest
    gamma = np.imag(row[f'f_expt{n}'])
    err_frac = uncertainty_dict['err_frac']
    error_vals = np.array(uncertainty_dict[n], dtype = 'float')
    
    # now we calculate f_err and g_err
    f_err = round((err_frac[0]*gamma + error_vals[0]), 1)
    g_err = round((err_frac[1]*gamma + error_vals[1]), 1)
    return f_err + 1j*g_err


def make_df_err(soln, uncertainty_dict):
    '''
    Calclate error in frequency shifts and dissipation

    args
    ----------
    soln : Dataframe 
        Data being considered (from solve_for_props).
        
    uncertainty_dict : dictionary
        Dictionary used to determine uncertainties in frequency and
        disipation values.  See definition of calc_fstar_err for the
        details.

    Returns
    -------
    dataframe with uncertainties in measured values of delf, delg


    '''
    nvals = nvals_from_df_soln(soln)
    # make the error dataframe that will be returned by the function
    # solution doesn't necessarily have to have the same values of 
    # props_calc or calc for every line
    df_err = pd.DataFrame()
    npts = len(soln)
    real_series = np.zeros(npts, dtype=np.float64)
    
    for n in nvals:
        df_err.insert(df_err.shape[1], f'f{n}_err', real_series)
        df_err.insert(df_err.shape[1], f'g{n}_err', real_series)

    if uncertainty_dict != 'zeros':        
        for idx, row in soln.iterrows():
            # extract uncertainty from dataframe if it is not [0, 0, 0]
            for n in nvals:
                df_err.loc[idx, f'f{n}_err'] = (
                    np.real(calc_fstar_err(n, row, uncertainty_dict)))

                df_err.loc[idx, f'g{n}_err'] = (
                           np.imag(calc_fstar_err(n, row, uncertainty_dict)))
    return df_err
            

def calc_prop_error(soln, uncertainty_dict):
    '''
    Calclate error in properties

    args
    ----------
    soln : Dataframe 
        Data being considered (from solve_for_props).
        
    uncertainty_dict : dictionary
        Dictionary used to determine uncertainties in frequency and
        disipation values.  See definition of calc_fstar_err for the
        details.

    Returns
    -------
    dataframe with errors in properties within props_calc

    '''
    # make the error dataframe that will be returned by the function
    # solution doesn't necessarily have to have the same values of 
    # props_calc or calc for every line
    prop_err=pd.DataFrame(index=soln.index)
    npts = len(soln)
    real_series = np.zeros(npts, dtype=np.float64)
    for idx, row in soln.iterrows():
        # this handles case where soln dataframe was not generated
        # by solve_for_props
        if 'props_calc' not in row:
            continue
        props_calc = row['props_calc']
        for prop in props_calc:
            if f'{prop}_err' not in prop_err.columns.values:
                prop_err.insert(prop_err.shape[1], f'{prop}_err', real_series)
    
        calc = row['calc']
        nf, ng, n_all, n_unique = nvals_from_calc(calc)

        # extract uncertainty from dataframe if it is not [0, 0, 0]
        if uncertainty_dict == 'zeros':
            # we don't calculate an uncertainty if n_all>3
            uncertainty = [0]*min(3, n_all)
        else:
            uncertainty = []
            for n in nf:
                delf_err = np.real(calc_fstar_err(n, row, uncertainty_dict))
                uncertainty.append(delf_err)
            for n in ng:
                delg_err = np.imag(calc_fstar_err(n, row, uncertainty_dict))
                uncertainty.append(delg_err)

        # extract the jacobian and turn it back into a numpy array of floats
        try:
            jacobian = np.array(row['jacobian'], dtype='float')
        except:
            jacobian = np.zeros([len(uncertainty), len(uncertainty)])
        try:
            deriv = np.linalg.inv(jacobian)
        except:
            deriv = np.zeros([len(uncertainty), len(uncertainty)])
    
        # determine error from Jacobian
        # this only works if the number of elements in calc is the 
        # same as the number of elements in props_calc (n equations and
        # n unknowns)

        n = len(props_calc)
        for p in np.arange(n):
            err_name = f'{props_calc[p]}_err'
            errval = 0
            for k in np.arange(n):
                errval = errval + (deriv[p, k]*uncertainty[k])**2
            prop_err.loc[idx, err_name] = np.sqrt(errval)
    return prop_err

        
def make_prop_axes(props, **kwargs):
    """
    Make a blank property figure.
    args:
                    
        props (list of strings):  
            props to include
            
            - 'drho_change' is change in drho, relative to drho_ref
            
            - 'drho_change_nm' is change in drho with d in nm
            
            - 'drho_norm' is drho divided by drho_ref
            - 'vgp' can be added as van Gurp-Palmen plot
            
            - 'vgp_lin'  and 'grho3_lin' put the grho3 on a linear scale
            
            - 'jdp' is loss compliance normalized by density
            
            - 'tanphi' is loss tangent
            
            - 'grho3p' storage modulus at n=3 times density
            
            - 'grho3pp'  loss modulus at n = 3 time density
            
            - 'etarho3'  complex viscosity (units of mPa-s-g/cm3)
            
            - 'cole-cole' plots grho3pp vs. grho3pp
            
            - 'temp' is temperature in degrees C
            
            - 's', 'hr', 'day' is time in appropriate unit
            
            - name of dataframe column can also be specified

    kwargs:.
            
        num (string):           
            window title (string):
            - (default is 'property fig')
            
        checks (boolean):
            True (default) if we plot the solution checks
            
        
        maps (booleaan):
            True (default is false) if we make the response maps
                    
        contour_range_units: 
            'Hz' (default) or Sauerbrey if we want to normalize
            zscale of response maps by Sauerbrey shift
            
        xunit (single string or list of strings):
            Units for x data.  Default is 'index', function currently handles
            - 's', 'min', 'hr', 'day', 'temp', or user specified value corresponding
                to a dataframe column
                
        xscale (string):
            'linear' (default) or 'log'
            
        xlabel (string):
            label for x axis.  Only used if user-specified for xunit is used
            currently must be same for all axes
            
        plotsize (tuple of 2 real numbers):
            size of individual plots.  
            - Default is (4, 3)
            
        sharex (Boolean):
            share x axis for zooming
            -(default=True)
                   
        no3 (Boolean):
            False (default) if we want to keep the '3' subscript in axis label for G


    returns:
        prop_axes dictionary with the following elements
        fig:
            Handle for the figure
        ax:
            Handle for the axes, returned as a 2d array of 
            dimensions (1, len(props))
        info:
            Dictionary with plot types for different axes
        xunit:
            Unit for the x axes
    """

    num = kwargs.get('num', 'property fig')
    maps = kwargs.get('maps', False)
    checks = kwargs.get('checks', True)
    xunit_input = kwargs.get('xunit', 'index')
    sharex = kwargs.get('sharex', True)
    xscale = kwargs.get('xscale', 'linear')
    no3 = kwargs.get('no3', False)
    contour_range = kwargs.get('contour_range')
    nprops = len(props)
    plotsize = kwargs.get('plotsize', (4,3))
    drho = kwargs.get('drho', 'Sauerbrey')
    dlim = kwargs.get('dlim', [0, 0.5])
    if nprops == 1 and not checks and not maps:
        titles = ['']
    else:
        titles = copy(titles_default)
     
    # dictionaries used for response map  and check solution data labels
    maplabels={}
    checklabels={}
    
    # specify the xunit dictionary and xlabel dictionary
    # all plots have same xunit if only one value is given
    xunit = {}
    xlabel = {}
    if type(xunit_input)==str:
        for p in np.arange(nprops):
            xunit[p] = xunit_input
    else:
        for p in np.arange(nprops):
            xunit[p]=xunit_input[p]
            
    # turn off sharex if not all axes have the same xunit
    for p in np.arange(nprops-1):
        if xunit[p]!=xunit[p+1]:
            sharex = False
            
    # set the x labels
    for p in np.arange(nprops):
        if xunit[p] in axlabels.keys():
            xlabel[p] = axlabels[xunit[p]]
        else:
            xlabel[p] = kwargs.get('xlabel', 'xlabel')
            
    # make the main figure
    # fig includes 'master' (the full fig), plus subfigures of 'props',
    # 'checks' and 'maps'
    # ax includes 'props', 'checks' and 'maps', in addition to sequential
    # listing of all axes within the master figure (ax[0], ax[1], etc.)
    fig = {}; ax = {}
    fig['master'] = plt.figure(constrained_layout = True, num = num)
    
    # build the GridSpec
    nrows = 1
    if checks:
        nrows = nrows+1
    if maps:
        nrows = nrows+1
    
    GridSpec = gridspec.GridSpec(ncols=12, nrows=nrows, figure= fig['master'])
    
    # add subfigures - details depend on nprops, which must be 1, 2 or 3
    # Subfigure - properties
    if nprops==1:
        fig['props'] = fig['master'].add_subfigure(GridSpec[0,3:9])
    else:
        fig['props'] = fig['master'].add_subfigure(GridSpec[0,:])
        
    ax['props'] = fig['props'].subplots(1,nprops, sharex=sharex, 
                                        squeeze = False).flatten()
    
    for k in np.arange(nprops):
        ax[k] = ax['props'][k]
        ax['props'][k].set_title(titles[k])
        
    iax = nprops-1  # running number of axes for axis labeling
    irow = 0  # running index of row

    if checks:
        # Subfigure - solution checks
        irow = irow+1
        if nprops == 3:
            fig['checks'] = fig['master'].add_subfigure(GridSpec[irow,1:11])
        else:
            fig['checks'] = fig['master'].add_subfigure(GridSpec[irow,:])
            
        ax['checks'] = fig['checks'].subplots(1,2, sharex=True)
        ax[iax+1]=ax['checks'][0]
        ax[iax+2]=ax['checks'][1]
        for k in [0, 1]:
            ax['checks'][k].set_title(titles[iax +1+k])
            # used xlabel for first prop plot as the xlabel for checks
            ax['checks'][k].set_xlabel(xlabel[0])
        iax = iax + 2
        ax['checks'][0].set_ylabel(axlabels['delf_n'])
        ax['checks'][1].set_ylabel(axlabels['delg_n'])
        
        gammascale = kwargs.get('gammascale', 'linear')
        ax['checks'][1].set_yscale(gammascale)
        
        # set linear or log scale
        for k in [0,1]:
            ax['checks'][k].set_xscale(xscale)
            
        # set variable we'll use to make sure we don't duplicate labels            
        for n in [1,3,5,7,9]:
            checklabels[n]=True
                
    if maps:    
        irow = irow+1
        # Subfigure - response maps
        if nprops == 3:
            fig['maps'] = fig['master'].add_subfigure(GridSpec[irow,1:11])
        else:
            fig['maps'] = fig['master'].add_subfigure(GridSpec[irow,:])

        ax['maps'] = fig['maps'].subplots(1,2, sharex=True)
        ax[iax+1]=ax['maps'][0]
        ax[iax+2]=ax['maps'][1]
        # make the response maps
        make_response_maps(fig['maps'],ax['maps'], drho, contour_range
                           =contour_range, dlim = dlim,
                           first_plot = iax+1)
        # set variable we'll use to make sure we don't duplicate labels
        for n in [1,3,5,7,9]:
            maplabels[n]=True
  
 
    # set the figure size
    # account for 2 extra rows for solution checks and response maps
    cols = max(2, nprops)
    figsize = (plotsize[0]*cols, plotsize[1]*nrows)
    fig['master'].set_size_inches(figsize)

        
    # change labels in case we don't want the 3 subscript for G
    if no3:
        axlabels['grho3'] = r'$|G^*|\rho$ (Pa $\cdot$ g/cm$^3$)'
        axlabels['grho3p'] = r'$G^\prime\rho$ (Pa $\cdot$ g/cm$^3$)'
        axlabels['grho3pp'] = r'$G^{\prime\prime}\rho$ (Pa $\cdot$ g/cm$^3$)'
    
    # extract 'linear' or 'log' scale factors from plots
    for p in np.arange(nprops):
        if 'grho3' in props[p]:
            ax['props'][p].set_yscale('log')
        else:
            ax['props'][p].set_yscale('linear') # linear plots by default
               
        props_parsed = props[p].split('.')
        if len(props_parsed)==2:
            yscale = props_parsed[1]
            ax['props'][p].set_yscale(yscale)
        
            if yscale == 'linear' and 'grho3' in props[p]:
                ax['props'][p].ticklabel_format(axis = 'y', style = 'sci',
                                                scilimits = (0, 0),
                                                useOffset = False)

    # now set the y labels 
    for p in np.arange(nprops):   
        # strip out layer number to make the connection to axlabels dictionary
        prop = props[p].split('.')[0].split('_')[0]
        if prop in axlabels.keys():
            ax['props'][p].set_ylabel(axlabels[prop])
            ax['props'][p].set_xlabel(xlabel[p])      
        elif prop == 'cole-cole':
            ax['props'][p].set_ylabel(axlabels['grho3pp'])
            ax['props'][p].set_xlabel(axlabels['grho3p'])
        elif 'vgp' in prop: 
            ax['props'][p].set_ylabel(axlabels['phi'])
            ax['props'][p].set_xlabel(axlabels['grho3'])
            
            # remove links to other axes if they exist
            if sharex:
                ax['props'][p].get_shared_x_axes().remove(ax['props'][p])
            xticker = matplotlib.axis.Ticker()
            ax['props'][p].xaxis.major = xticker
            
            # The new ticker needs new locator and formatters
            xloc = matplotlib.ticker.AutoLocator()
            xfmt = matplotlib.ticker.ScalarFormatter()
            
            ax['props'][p].xaxis.set_major_locator(xloc)
            ax['props'][p].xaxis.set_major_formatter(xfmt)
            
        elif 'df' in prop:
            n = prop[-1]
            # handle the possibility that we want to plot negative of df
            if '-df' in props[p]:
                ax['props'][p].set_ylabel(f'$-\Delta f_{{{n}}}$ (Hz)')
            else:
                ax['props'][p].set_ylabel(f'$\Delta f_{{{n}}}$ (Hz)')
            ax[p].set_xlabel(xlabel[p])
        elif 'dg' in prop:
            n = prop[-1]
            ax['props'][p].set_ylabel(f'$\Delta \Gamma_{{{n}}}$ (Hz)')
            ax['props'][p].set_xlabel(xlabel[p])
        else:
            ax['props'][p].set_xlabel('xlabel')
            ax['props'][p].set_ylabel('ylabel')
        ax['props'][p].set_title(titles[p])

    info = {'props':props, 'xunit':xunit, 'xscale':xscale,
            'maplabels':maplabels, 'checklabels':checklabels}
    return {'fig':fig, 'ax':ax, 'info':info}


def plot_props(soln, figinfo, **kwargs):
    """
    Add property data to an existing figure.

    args:
        soln (dataframe):
            Dataframe containing data to be plotted, typically output from
            solve_for_props.
        figinfo (dictionary)
            Dictionary containing 'fig', 'ax' and other info for plot.

    kwargs:

        xoffset (real or string, single value or list):
            Amount to subtract from x value for plotting (default is 0)
            'zero' means that the data are offset so that the minimum val 
            is at 0.
        xmult (real):
            Multiplicative factor for rescaling x data.
        fmt (string):
            Format sting for plotting.  Default is '+'   .
        label (string):
            label for plots.  Used to generate legend.  Default is 
            '', which will not generate a label.
        uncertainty_dict (dictionary or string):
            Information used to generate uncertainty in frequency and
            dissipation values, used to determine uncertainties in 
            calculated properties.  Default is 'zeros', in which case we 
            don't calculate property uncertainties.  See the definition of
            calc_fstar_err for details.
        nplot (list of integers):
            harmonics to plot, default is [3,5])
        drho_ref (real):
            reference drho for plots of drho_norm or drho_ref


    returns:
        Nothing is returned.  The function updates the existing figure.
    """
    
    if len(soln) ==0:
        print('solution data frame for plotting is empty')
        return
    fmt=kwargs.get('fmt', '+')
    label=kwargs.get('label', '')
    xoffset_input=kwargs.get('xoffset', 0)  
    uncertainty_dict = kwargs.get('uncertainty_dict', uncertainty_dict_default)
    drho_ref = kwargs.get('drho_ref', np.nan)
    nplot = kwargs.get('nplot', [3,5])
    col = {1:'CO', 3:'C1', 5:'C2', 7:'C3', 9:'C5'}
    
    # drop dataframe rows with all nan
    soln = soln.dropna(how='all') 
    
    # extract data from figinfo
    props = figinfo['info']['props']
    ax = figinfo['ax']
    fig = figinfo['fig']
    nprops = len(props)
    plots_to_make = kwargs.get('plots_to_make', np.arange(nprops))
    
    # create dataframe with calculated errors
    prop_error = calc_prop_error(soln, uncertainty_dict)
    xunit = figinfo['info']['xunit']
    xvals = {}
    
    # set the offset for the x values (apart from vgp plots)
    xoffset = {}
    
    # determine the plots we actually need to make. Sometimes we don't add
    # data to an existing axis
    pvals = []
    for p in np.arange(nprops):
        if drho_ref == np.nan and ((props[p] == 'drho_diff' ) or 
                                   (props[p] == 'drho_norm' )):
            sys.exit('need to specify a value of drho_norm')
        if p in plots_to_make:
            pvals.append(p)
    if type(xoffset_input) != list:
        for p in np.arange(nprops):
            xoffset[p] = xoffset_input
    else:
        for p in np.arange(nprops):
            xoffset[p]=xoffset_input[p]   
            
    for p in pvals:
        if xunit[p] == 's':
            xvals[p]=soln['t']
        elif xunit[p] == 'min':
            xvals[p]=soln['t']/60
        elif xunit[p] == 'hr':
            xvals[p]=soln['t']/3600
        elif xunit[p] == 'day':
            xvals[p]=soln['t']/(24*3600)
        elif xunit[p] == 'temp':
            xvals[p]=soln['temp']
        elif xunit[p] == 'index':
            xvals[p]=soln.index
        else:
            xvals[p]=soln[xunit[p]]
            
        if xoffset_input == 'zero':
            xoffset[p] = min(xvals[p])
                    
   
    # now make all of the plots
    for p in pvals:
        # yerr is zero unless we specify otherwise
        yerr = pd.Series(np.zeros(len(xvals[p])))
        prop = props[p].split('.')[0]
        if len(prop.split('_'))==1:
            prop = prop+'_1'
        layer = prop.split('_')[1]
        if 'grho3_' in prop: 
            xdata = xvals[p]
            ydata = soln[prop].astype(float)/1000
            if f'{prop}_err' in prop_error.columns.values:
                yerr = prop_error[f'{prop}_err']/1000
    
        elif 'etarho3_' in prop:
            xdata = xvals[p]
            # units are mPa-s for viscosity
            ydata = soln[f'grho3_{layer}'].astype(float)/(np.pi*1.5e7)
            if f'grho3_{layer}_err' in prop_error.columns.values:
                yerr = prop_error[f'grho3_{layer}_err']/(np.pi*1.5e7)

        elif 'phi_' in prop:
            xdata = xvals[p]
            ydata = soln[prop].astype(float)
            if f'{prop}_err' in prop_error.columns.values:
                yerr = prop_error[f'{prop}_err']
            
        elif 'tanphi' in prop:
            xdata = xvals[p]
            ydata = np.tan(np.pi*soln[f'phi_{layer}'].astype(float)/180)
            
        elif 'grho3p' in prop:
            xdata = xvals[p]
            ydata = (soln[f'grho3_{layer}'].astype(float)*
                     np.cos(np.pi*soln['phi'].astype(float)/180)/1000)

        elif 'grho3pp' in prop:
            xdata = xvals[p]
            ydata = (soln[f'grho3_{layer}'].astype(float)*
                     np.sin(np.pi*soln['phi'].astype(float)/180)/1000)
            
        elif 'drhonm' in prop:
            xdata = xvals[p]
            ydata = 1e6*soln[f'drho_{layer}'].astype(float)
            yerr = 1e6*soln['drho_err']
                          
        elif 'drho' in prop:
            xdata = xvals[p]
            ydata = 1000*soln[f'drho_{layer}'].astype(float)
            if f'{prop}_err' in prop_error.columns.values:
                yerr = 1000*prop_error[f'{prop}_err']

        elif 'vgp' in prop:
            xdata = soln['grho3_1'].astype(float)/1000
            ydata = soln['phi_1'].astype(float)

        elif 'cole-cole' in prop:
            xdata = (soln['grho3_1'].astype(float)*
                     np.cos(np.pi*soln['phi'].astype(float)/180)/1000)
            ydata = (soln['grho3_1'].astype(float)*
                     np.sin(np.pi*soln['phi'].astype(float)/180)/1000)

            
        elif 'jdp' in prop:
            xdata  = xvals[p]
            ydata = ((1000/soln['grho3_1'].astype(float))*
                     np.sin(soln['phi'].astype(float)*np.pi/180))

            
        elif prop == 'temp':
            xdata  = xvals[p]
            ydata = soln['temp'].astype(float)
            
        elif prop == 't':
            xdata  = xvals[p]
            ydata = soln['t'].astype(float)
            
        elif 'soln_expt' in prop:
            xdata  = xvals[p]    
            yvals = soln[props[p]].astype(complex)
            ydata = np.real(yvals)
            
        elif 'dg_expt' in prop:
            xdata  = xvals[p]
            key = props[p].replace('dg', 'soln')
            yvals = soln[key].astype(complex)
            ydata = np.imag(yvals)
            
        elif prop in soln.keys():
            xdata = xvals[p]
            ydata = soln[props[p]]
            if 'grho' in props[p]:
                ydata = ydata/1000
            elif 'drho' in props[p]:
                ydata = 1e3*ydata
        
        else:
            print('not a recognized prop type ('+props[p]+')')
            sys.exit()
        
        xdata = xdata.astype('float64')
                
        if (yerr == 0).all() or np.isnan(yerr).all():
            ax['props'][p].plot(xdata, ydata, fmt, label=label)
        else:
            ax['props'][p].errorbar(xdata, ydata, fmt=fmt, yerr=yerr, 
                                    label=label)
            
        if props[p] == 'vgp' or figinfo['info']['xscale'] == 'log':
            ax['props'][p].set_xscale('log')
        if props[p] == 'cole-cole':
            ax['props'][p].set_xscale('log')
            ax['props'][p].set_yscale('log')
            
            
    # now add the comparison plots of measured and calcuated values         
    # plot the experimental data first
    # keep track of max and min values for plotting purposes       
    
    if 'checks' in fig.keys():       
        # decide to use df1 or not
        plot_df1 = kwargs.get('plot_df1', False)
        
        # adjust nplot if any of the values don't exist in the dataframe
        for n in nplot:
            if not 'df_expt'+str(n) in soln.keys():
                nplot.remove(n)
        
        if len(xdata)==1:
            calcfmt = 'o'
        else:
            calcfmt = '-'
        df_min = []
        df_max = []
        dg_min = []
        dg_max = []
        
        # now plot the calculated values
        nfplot = nplot.copy()
        ngplot = nplot.copy()
        if not plot_df1:
            try:
                nfplot.remove(1)
            except ValueError:
                pass
        # set up colors we'll use for the calculated values
        col = {1:'C0', 3:'C1', 5:'C2', 7:'C3', 9:'C4'}
        # add uncertainties in delf, delg

        soln = add_fstar_err(soln, uncertainty_dict)
        soln['xvals'] = xvals[0]           
        for n in nfplot: 
            # drop nan values from dataframe to avoid problems with errorbar
            soln_tmp = soln.dropna(subset=[f'df_expt{n}'])
            dfval = np.real(soln_tmp[f'df_expt{n}'])/n
            ferr = np.real(soln_tmp[f'fstar_err{n}'])/n
            df_min.append(np.nanmin(dfval))
            df_max.append(np.nanmax(dfval))
            if figinfo['info']['checklabels'][n]:
                label_expt = f'n={n}: expt'
                label_calc = f'n={n}: calc'
            else:
                label_expt = ''
                label_calc = ''
            ax['checks'][0].errorbar(soln_tmp['xvals'], 
                       dfval, yerr = ferr, fmt='+', color = col[n],       
                       label=label_expt)
            calcvals = np.real(soln_tmp['df_calc'+str(n)])/n
            ax['checks'][0].plot(soln_tmp['xvals'], calcvals, calcfmt, 
                    color = col[n], markerfacecolor='none', 
                    label=label_calc)
            
            # don't include multiple harmonic labels
            figinfo['info']['checklabels'][n]=False
        df_min = min(df_min)
        df_max = max(df_max)
     
        for n in ngplot:
            # drop nan values from dataframe to avoid problems with errorbar
            soln_tmp = soln.dropna(subset=[f'df_expt{n}'])
            dgval = np.imag(soln_tmp['df_expt'+str(n)])/n
            gerr = np.imag(soln_tmp[f'fstar_err{n}'])/n
            dg_min.append(np.nanmin(dgval))
            dg_max.append(np.nanmax(dgval))
            ax['checks'][1].errorbar(soln_tmp['xvals'], dgval, yerr = gerr, 
                                     fmt = '+',  
                                     color = col[n])
            calcvals = np.imag(soln_tmp['df_calc'+str(n)])/n
            ax['checks'][1].plot(soln_tmp['xvals'], calcvals, calcfmt, 
                          color = col[n], markerfacecolor='none')
        dg_min = min(dg_min)
        dg_max = max(dg_max)    
        
        gammascale = kwargs.get('gammascale', 'linear')
        # change dissipation scale to log scale if needed
        if gammascale=='log':
            ax['checks'][1].set_yscale('log')
            
        # change x scale to log scale if needed
        xscale = kwargs.get('xscale', 'linear')
        if xscale == 'log':
            ax['checks'][0].set_xscale('log')
            ax['checks'][0].set_xscale('log')  
            
        # add lendend - single legend for both parts
        handles, labels = ax['checks'][0].get_legend_handles_labels()
        
        # sort legend entries
        order = np.argsort(labels)

        ax['checks'][0].legend([handles[idx] for idx in order],
                               [labels[idx] for idx in order], ncol=1, 
                               labelspacing=0.1, columnspacing=0, 
                               markerfirst=False, handletextpad=0.1,
                               bbox_to_anchor=(1.02, 1),
                               handlelength=1)
            
        # reset y axis limits for delf and delg if needed
        df_lim = kwargs.get('df_lim', 'expt')
        if df_lim == 'expt':
            delf_range = df_max - df_min
            delg_range = dg_max - dg_min
            ax['checks'][0].set_ylim([df_min - 0.05*delf_range, 
                                          df_max + 0.05*delf_range])
            if ax['checks'][1].get_yscale() == 'log' and dg_min>0:
                ax['checks'][1].set_ylim([0.9*dg_min, 1.1*dg_max])
            elif ax['checks'][1].get_yscale() == 'linear':
                ax['checks'][1].set_ylim([dg_min - 0.05*delg_range, 
                            dg_max +0.05*delg_range])
                
    # now add the response maps        
    if 'maps' in fig.keys():
        # add values to contour plots
        for n in nplot:
            dlam = calc_dlam_from_dlam3(n, soln['dlam3_1'], soln['phi_1'])
            if figinfo['info']['maplabels'][n]:
                label = 'n='+str(n)
            else:
                label = ''
            for k in [0,1]:
                ax['maps'][k].plot(dlam, soln['phi_1'], '-o',
                    label = label, mfc = col[n], mec = 'k', c=col[n])         
            # don't include multiple harmonic labels
            figinfo['info']['maplabels'][n]=False
        for k in [0, 1]:
            ax['maps'][k].legend(framealpha=1)
    

def read_xlsx(infile, **kwargs):
    """
    Create dataframe from input .xlsx file.

    args:
    ---------------------
    infile : String
        The full name of the input .xlsx file, generally exported from
        the RheoQCM program.

    kwargs:
    ---------------------
    restrict_to_marked : list of integers
        List of harmonics that must be marked 
        in order to be included. Default is [], so that we
        include everything.

    film_channel : string
        Sheet for data.  _channel' by default.

    ref_channel : string
        Source for reference frequency and dissipation. 

        - 'R_channel': 'R_channel' sheet from xlsx file) (default)  
    
        - 'S_channel': 'S_channel' sheet from xlsx file 
                
        - 'S_reference': 'S_reference' sheet from xlsx file  
                
        - 'R_reference': 'R_channel' sheet from xlsx file'  
                
        - 'self':  read delf and delg read directly from the data channel   


    ref_idx : numpy array of integers
        Index values to include in reference determination.
            
            - default is 'all', which takes everything
            - 'max' means we take the single value for which f3 is maximized

    film_idx: numpy array of integers or string
        Index values to include for film data.  Default is 'all' which 
        takes everthing.

    T_coef : dictionary or string
        Temperature coefficients for reference temp. shift  
            
        - calculated from ref. temp. data if not specified
        - set to the following dictionary if equal to 'default'
        
            {'f': {1: [0.00054625, 0.04338, 0.08075, 0],                      
            3: [0.0017, -0.135, 8.9375, 0],                
            5: [0.002825, -0.22125, 15.375, 0]}, 
            'g': {1: [0, 0, 0, 0], 
            3: [0, 0, 0, 0], 
            5: [0, 0, 0, 0]}}
            
        - other option is to specify the dictionary directly

    Tref : float
        Temperature at which reference frequency shift was determined.
        Default is 22C.
            
    Autodelete : Boolean
        True (default) if we want delete points at temperatures where
        we don't have a reference point'

    T_coef_plots : Boolean  
        True (default) to plot temp. dependent f and g for ref.  

    fref_shift : dictionary
        Shifts added to reference values.  
        Default is {1:0, 3:0, 5:0, 7:0, 9:0}.

    nvals : list of integers 
        Harmonics to include.  Default is [1, 3, 5, 7, 9].
        
    index_col : integer
        Column in Excel spreadsheet to use for the dataframe index.
        Default is 0.  Use None to make a new index.
    
    Guess_sheet : string
        Sheet name containing property guesses.  
        Used when calculating were eported into the Excel file

    returns:
        Input data converted to dataframe.
    """

    restrict_to_marked=kwargs.get('restrict_to_marked', [])
    film_channel=kwargs.get('film_channel', 'S_channel')
    film_idx=kwargs.get('film_idx', 'all')
    ref_channel=kwargs.get('ref_channel', 'R_channel')
    ref_idx=kwargs.get('ref_idx', 'all')
    T_coef_plots=kwargs.get('T_coef_plots', True)
    nvals_in=kwargs.get('nvals', [1, 3, 5, 7, 9])
    Tref=kwargs.get('Tref', 22)
    autodelete = kwargs.get('autodelete', True)
    index_col = kwargs.get('index_col', 0)
    
    # specify default bare crystal temperature coefficients
    T_coef=kwargs.get('T_coef', 'calculated')
    if T_coef == 'default':
        T_coef = T_coef_default 

    # read shifts that account for changes from stress levels applied
    # to different sample holders
    fref_shift=kwargs.get('fref_shift', {1: 0, 3: 0, 5: 0, 7:0, 9:0})
    

    if type(ref_idx) == str and ref_idx == 'max':
        df_ref=pd.read_excel(infile, sheet_name=ref_channel, header=0)
        ref_idx = np.array([df_ref['f3'].idxmax()])
        
    df=pd.read_excel(infile, sheet_name=film_channel, header=0,
                     index_col = index_col)
    if type(film_idx) != str:
        df=df[df.index.isin(film_idx)]
        
    # keep track of columns for output dataframe
    keep_column = []
        
    # read guess values if they exist
    guess_sheet = kwargs.get('guess_sheet', 'none')
    sheet_names = pd.ExcelFile(infile).sheet_names
    if guess_sheet in sheet_names:
        df_guess_sheet = pd.read_excel(infile, sheet_name = guess_sheet, 
                                       header=0, index_col = index_col)
        series = [{}for _ in range(len(df_guess_sheet))]
        for idx in df_guess_sheet.index:
            series[idx] = {'grho3':df_guess_sheet.loc[idx, 'grhos3'],
                       'phi':df_guess_sheet.loc[idx, 'phi'],
                       'drho':df_guess_sheet.loc[idx, 'drho']}
        df_guess = pd.DataFrame({'guess':series})
        
        # merge in the guesses, keeping the indices from df
        df = df.reset_index().merge(df_guess, left_index = True, 
                                    right_index = True,
                                    how="left").set_index('index')

        keep_column.append('guess')
        
    # include all values of n that we want and that exist in the input file
    nvals = []
    for n in nvals_in:
        if 'f'+str(n) in df.keys():
            nvals.append(n)
        
    # keep all rows unless we are told to check for specific marks
    df['keep_row']=1  
    for n in restrict_to_marked:
        df['keep_row']=df['keep_row']*df['mark'+str(n)]

    # Delete all rows that are not appropriately marked
    df=df[df.keep_row == 1]
    
    # now sort out which columns we want to keep in the dataframe
    keep_column.append('t')
    keep_column.append('temp')
    for n in nvals:
        keep_column.append(n)

    # add the temperature column to original dataframe if it does not exist
    # or contains all nan values, and set all Temperatures to Tref
    if ('temp' not in df.keys()) or (df.temp.isnull().values.all()):
        df['temp'] = Tref
        
    # add each of the values of delfstar
    if ref_channel == 'self':
        # this is the simplest read protocol, with delf and delg already in
        # the .xlsx file
        for n in nvals:
            df[n]=df['delf'+str(n)] + 1j*df['delg'+str(n)
                                ].round(1) - fref_shift[n]
            
    elif T_coef != 'calculated':
        # this is the case where the temperature coefficients are input
        # directly as a dictionary,  includes the case where we just use
        # the default values
        df_ref=pd.read_excel(infile, sheet_name=ref_channel, header=0)
        if type(ref_idx) != str:
            df_ref=df_ref[df_ref.index.isin(ref_idx)]
        T_coef_plots=False
        
        for n in nvals:
            # apply fref_shift if needed
            df_ref['f'+str(n)] = df_ref['f'+str(n)] + fref_shift[n]
            # adjust constant lffast elment in T_coef (the 
            # constant term) to give measured ref. values at Tref
            for val in ['f', 'g']:
                T_coef[val][n][3] = (T_coef[val][n][3] + 
                                     df_ref[val+str(n)].mean() -
                                     np.polyval(T_coef[val][n], Tref))
                
                # add absolute frequency and reference values to dataframe
                keep_column.append(val+str(n)+'_dat')
                keep_column.append(val+str(n)+'_ref')
                
                # set reference and film values
                df[val+str(n)+'_ref'] = np.polyval(T_coef[val][n],
                                                   df['temp'])
                df[val+str(n)+'_dat'] = df[val+str(n)]
            
            # keep track (of film and reference values in dataframe
            df[n]  = (df['f'+str(n)+'_dat'] - df['f'+str(n) + '_ref'] +
                  1j*(df['g'+str(n)+'_dat'] - df['g'+str(n) + '_ref']))
            

    else:
        # here we need to obtain T_coef from the info in the ref. channel
        df_ref=pd.read_excel(infile, sheet_name=ref_channel, header=0)
        if type(ref_idx) != str:
            df_ref=df_ref[df_ref.index.isin(ref_idx)]

        # if no temperature is listed or a specific reference temperature
        # is given we just average the values or take the max value
        if (('temp' not in df_ref.keys()) or 
            (df_ref.temp.isnull().values.all())):

            for k in np.arange(len(nvals)):
                # write the film and reference values to the data frame
                df[f'{nvals[k]}_ref'] = (df_ref[f'f{nvals[k]}'].mean()+
                                 1j*df_ref[f'g{nvals[k]}'].mean()).round(1)
                df[f'{nvals[k]}_dat'] = (df[f'f{nvals[k]}']+
                                 1j*df[f'g{nvals[k]}']).round(1)

            # set all the temperatures on df_ref to Tref
            df_ref['temp']=Tref

        else:
            # now we handle the case where we have a full range of
            # temperatures
            # reorder rerence data according to temperature
            df_ref=df_ref.sort_values('temp')

            # drop any duplicate temperature values
            df_ref=df_ref.drop_duplicates(subset='temp', keep='first')
            temp=df_ref['temp']
            for k in np.arange(len(nvals)):
                for var in ['f', 'g']:
                    # set T_coef to defaults to start
                    T_coef = T_coef_default
                    # get the reference values and plot them
                    ref_vals=df_ref[var+str(nvals[k])]
                    
                    # put temp and reference values into a temporary
                    # dataframe
                    data = [temp, ref_vals]
                    headers = ['temp', 'data']
                    df_tmp = pd.concat(data, axis=1, keys=headers).dropna()
            
                    # make the fitting function
                    T_coef[var][nvals[k]]=np.polyfit(df_tmp['temp'], 
                                                        df_tmp['data'], 3)

                    # plot the data if fit was not obtained
                    if np.isnan(T_coef[var][nvals[k]]).any():
                        fig, ax = plt.subplots(1,1, figsize=(4,3),
                                               constrained_layout=True)
                        ax.plot(df_tmp['temp'], df_tmp['data'])
                        ax.set_xlabel(r'$T$ $^\circ$C')
                        ax.set_ylabel(f'{var}{nvals[k]}')
                        print('Temp. coefs could not be obtained - see plot')
                        sys.exit()
                                                                        
                # write the film and reference values to the data frame
                df[f'{nvals[k]}_dat'] = (df[f'f{nvals[k]}']+
                                         1j*df[f'g{nvals[k]}']).round(1)
                fref = np.polyval(T_coef['f'][nvals[k]], df['temp'])
                gref = np.polyval(T_coef['g'][nvals[k]], df['temp'])
                df[f'{nvals[k]}_ref'] = (fref + 1j*gref).round(1)


        for k in np.arange(len(nvals)):
            # now write values of delfstar to the dataframe
            df[nvals[k]]=(df[f'{nvals[k]}_dat'] -
                          df[f'{nvals[k]}_ref'] -
                          fref_shift[nvals[k]]).round(1)

            # add absolute frequency and reference values to dataframe
            keep_column.append(f'{nvals[k]}_dat')
            keep_column.append(f'{nvals[k]}_ref')

    # add the constant applied shift to the reference values to the dataframe
    for n in nvals:
        if fref_shift[n]!= 0:
            df[str(n)+'_refshift']=fref_shift[n]
            keep_column.append(str(n)+'_refshift')

    if (T_coef_plots and ref_channel != 'self' and 
        len(df_ref.temp.unique()) > 1):
        T_range=[df['temp'].min(), df['temp'].max()]
        T_ref_range = [df_ref['temp'].min(), df_ref['temp'].max()]
        # create a filename for saving the reference temperature data
        filename = os.path.splitext(infile)[0]+'_Tref.pdf'
        plot_bare_tempshift(df_ref, T_coef, Tref, nvals, T_range, filename)
        if (autodelete and 
            (T_range[0] < T_ref_range[0] or T_range[1] > T_ref_range[1])):
            df_tmp = df.query('temp >= @T_ref_range[0] & temp <= @T_ref_range[1]')
            # determine number of deleted points
            n_del = len(df) - len(df_tmp)
            print (f'deleting {n_del} points that are outside the ref T range')
            df = df_tmp
            
    # eliminate rows with nan at n=3
    df = df.dropna(subset=[3]).copy()
    
    # add time increments
    df = add_t_diff(df)
    keep_column.insert(1, 't_prev')
    keep_column.insert(2, 't_next')

    return df[keep_column].copy()


def cull_df(df_in, **kwargs):
    """
    cull dense dataframes by eliminating based on time
    args:
        df_in (dataframe)
        input dataframe
        
    kwargs:  (must include either t_ratio or t_diff)
        t_ratio:  minimum ratio of sucessive time points
        t_dff:  minimu difference between successive time points (s)
        t_range (list of two real numbers):
            minimum and maximum time points to consider.  Time points outside
            this range are unaffected.
    """
    
    t_range = kwargs.get('t_range', [-np.inf, np.inf])
    # copy the input dataframe
    df = df_in.copy()
   
    # dataframe based on time restrictions
    df_t = df.query('t>=@t_range[0] and t<=@t_range[1]')
    idxvals = df_t.index.values
    i_ref = 0
    i_val = 1
    remove_idx = []  # list of indices to remove
    if 't_ratio' in kwargs.keys():
        t_ratio = kwargs.get('t_ratio')
        while i_val<len(idxvals):
            while (i_val<len(df_t) and 
                   (df_t.iloc[i_val]['t']/df_t.iloc[i_ref]['t'] < t_ratio)):
                remove_idx.append(idxvals[i_val])
                i_val = i_val+1
            i_ref = i_val
            i_val = i_val+1
        return df.drop(remove_idx)
    
    elif 't_diff' in kwargs.keys():
        t_diff = kwargs.get('t_diff')
        while i_val<len(idxvals):
            while (i_val<len(df_t) and 
                   (df_t.iloc[i_val]['t']-df_t.iloc[i_ref]['t'] < t_diff)):
                remove_idx.append(idxvals[i_val])
                i_val = i_val+1
            i_ref = i_val
            i_val = i_val+1
        return df.drop(remove_idx)
    
    else:
       print('need \'t_diff\' or \'t_ratio\' in kwargs')


def plot_bare_tempshift(df_ref, T_coef, Tref, nvals, T_range, filename):
    var=['f', 'g']
    n_num=len(nvals)
    # figure for comparision of experimental and fit delf/n, delg/n
    fig, ax=plt.subplots(2, n_num, figsize=(3*n_num, 6),
                                   constrained_layout=True)
    
    # figure for comparison of all delf/n
    fig2, ax2 = plt.subplots(1, 1, figsize = (4, 3), constrained_layout=True,
                             num=filename)
    ylabel={0: r'$\Delta f/n$ (Hz)', 1: r'$\Delta \Gamma/n$ (Hz)'}
    ax2.set_ylabel(r'$\Delta f/n$ (Hz)')
    ax2.set_xlabel(r'$T$ ($^\circ$C)')
    # for now I'll use a default temp. range to plot
    temp_fit=np.linspace(T_range[0], T_range[1], 100)
    for k in np.arange(len(nvals)):
        # plot fit values of delf/n for all harmonics
        vals = bare_tempshift(temp_fit, T_coef, Tref, nvals[k])['f']/nvals[k]
        ax2.plot(temp_fit, vals, '-', label=f'n={nvals[k]}')
        for p in [0, 1]:
            # plot themeasured values, relative to value at ref. temp.
            meas_vals=(df_ref[var[p]+str(nvals[k])] -
                         np.polyval(T_coef[var[p]][nvals[k]], Tref))
            meas_vals=meas_vals/nvals[k]
            ax[p, k].plot(df_ref['temp'], meas_vals, 'x', label = 'meas')

            # now plot the fit values
            ref_val=bare_tempshift(temp_fit, T_coef, Tref, nvals[k])[var[p]]
            ref_val=ref_val/nvals[k]
            ax[p, k].plot(temp_fit, ref_val, 'o', label='fit')

            # set axis labels and plot titles
            ax[p, k].set_xlabel(r'$T$ ($^\circ$C)')
            ax[p, k].set_ylabel(ylabel[p])
            ax[p, k].set_title('n='+str(nvals[k]))
            ax[p, k].legend()
            ymin=np.min([meas_vals.min(), ref_val.min()])
            ymax=np.max([meas_vals.max(), ref_val.max()])
            ax[p, k].set_ylim([ymin, ymax])
    fig.suptitle(filename)
    fig.savefig(filename)
    sumfile = os.path.basename(filename).split('.')[0]+'_sum.pdf'
    directory = os.path.dirname(filename)
    fig2.savefig(os.path.join(directory, sumfile))


def bare_tempshift(T, T_coef, Tref, n):
    f=np.polyval(T_coef['f'][n], T) - np.polyval(T_coef['f'][n], Tref)
    g=np.polyval(T_coef['g'][n], T) - np.polyval(T_coef['g'][n], Tref)
    return {'f': f, 'g': g}

def plot_delfstar(df, **kwargs):
    '''
    Simple way to plot delf and delg.
    Parameters
    ----------
    df : dataframe
        datframe of delfstar values in the format returned by qcm.read_excel.

    Optional arguments
    n : list of integers 
        harmonics to inlude (default is [3])
        
    xkey : string
        key within df that we'll use as our x variable.

    df_ref : dataframe
        dataframe for reference frequency shifts
    
    num : string
        window title
        
    Returns
    -------
    fig, ax for the plot

    '''
    num = kwargs.get('num', 'delfstar plot')
    nvals = kwargs.get('nvals', [3])
    xkey = kwargs.get('xkey', 'index')
    if 'df_ref' in kwargs.keys():
        fig, ax = plt.subplots(2, 2, figsize = (8, 6), 
                               constrained_layout = True, num = num)
    else:
        fig, ax = plt.subplots(1, 2, figsize = (8, 3), constrained_layout = True,
                           num = num)

    ax = ax.flatten()
    
    # set all the axis labels
    for k in np.arange(len(ax)): 
        ax[k].set_xlabel(xkey)

    ax[0].set_ylabel(r'$\Delta f_n/n$ (Hz)')
    ax[1].set_ylabel(r'$\Delta \Gamma _n/n$ (Hz)')
    
    if 'df_ref' in kwargs.keys():
        ax[2].set_ylabel(r'$f_n^{ref}$-mean($f_n^{ref}$) (Hz)')
        ax[3].set_ylabel(r'$\Gamma _n^{ref}$-mean($\Gamma_n^{ref}$) (Hz)')
    
    for n in nvals:
        label = 'n='+str(n)
        normdelfstar = df[n]/n
        x = df[xkey]
        ax[0].plot(x, np.real(normdelfstar), '.', label = label)
        ax[1].plot(x, np.imag(normdelfstar), '.', label = label)
        if 'df_ref' in kwargs.keys():
            df_ref = kwargs.get('df_ref')
            fref = df_ref['f'+str(n)+'_ref']
            gref = df_ref['f'+str(n)+'_ref']
            xref = df_ref[xkey]
            ax[2].plot(xref, fref - fref.mean(), '.', label = label)
            ax[3].plot(xref, gref - gref.mean(), '.', label = label)

    for k in np.arange(len(ax)): ax[k].legend()
    return fig, ax


def gstar_maxwell(wtau):
    """
    Normzlized g* for single Maxwell element.

    args:
        wtau (real):
            angular frequency times relaxation time


    returns:
        gstar:
            complex shear modulus normalized by unrelaxed value
    """
    return 1j*wtau/(1+1j*wtau)


def gstar_kww_single(wtau, beta):  # Transform of the KWW function
    """
    Normzlized g* for single kww element.

    args:
        wtau (real):
            angular frequency times relaxation time
        beta (real):
            kww exponent

    returns:
        gstar:
            complex shear modulus normalized by unrelaxed value
    """
    return wtau*(kwws(wtau, beta)+1j*kwwc(wtau, beta))

gstar_kww=np.vectorize(gstar_kww_single)


def gstar_rouse(wtau, n_rouse):
    """
    Normzlized g* for Rouse model.

    args:
        wtau (real):
            Angular frequency times relaxation time.
        n_rouse (integer):
            Number  of Rouse modes to consider.

    returns:
        gstar:
            complex shear modulus normalized by unrelaxed value
    """
    # make sure n_rouse is an integer if it isn't already
    n_rouse=int(n_rouse)

    rouse=np.zeros((len(wtau), n_rouse), dtype=complex)
    for p in 1+np.arange(n_rouse):
        rouse[:, p-1]=((wtau/p**2)**2/(1+wtau/p**2)**2 +
                                  1j*(wtau/p**2)/(1+wtau/p**2)**2)
    rouse=rouse.sum(axis=1)/n_rouse
    return rouse


def springpot(w, g0, tau, beta, sp_type, **kwargs):
    """
    Create a calculated curve of the complex shear moduus vs
    frequency for an aribtrary combination of Maxwell, Rouse, 
    kww (stretched exponential) and springpot (power law) elements.

    args:
        w (numpy array of real values):
            Angular frequencies.
        g0 (list of real values):
            Unrelaxed moduli.
        tau (list of real values):
            Relaxation times.
        beta (list of real values):
            Exponents
        sp_type (list of integers):
            Specifies the detailed combination of different springpot elments
            combined in series, and then in parallel.  For example, if type
            is [1,2,3],  there are three branches in parallel with one
            another:  the first one is element 1, the second one is a 
            series comination of elements 2 and 3, and the third one is a 
            series combination of 4, 5 and 6.

    kwargs:
        kww (list of integers):
            Elements that are kww elements. Default is [].
        maxwell (list of integers):
            Elements that are Maxwell elements. Default is [].
        rouse (list of integers):
            Elments that are Rouse elments

    returns:
        g_br (dictionary of numpy arrays)
            complex modulus for each parallel branch (summed to get gstar)
        gstar (numpy array):
            complex shear modulus normalized by unrelaxed value

    """

    # specify which elements are kww or Maxwell elements
    kww=kwargs.get('kww', [])
    maxwell=kwargs.get('maxwell', [])
    rouse=kwargs.get('rouse', [])

    # make values numpy arrays if they aren't already
    w = np.asarray(w).reshape(1, -1)[0, :]
    tau=np.asarray(tau).reshape(1, -1)[0, :]
    beta=np.asarray(beta).reshape(1, -1)[0, :]
    g0=np.asarray(g0).reshape(1, -1)[0, :]
    sp_type=np.asarray(sp_type).reshape(1, -1)[0, :]

    nw=len(w)  # number of frequencies
    n_br=len(sp_type)  # number of series branches
    n_sp=sp_type.sum()  # number of springpot elements
    sp_comp=np.empty((nw, n_sp), dtype=complex)  # element compliance
    br_g=np.empty((nw, n_br), dtype=complex)  # branch stiffness

    # calculate the compliance for each element
    for i in np.arange(n_sp):
        if i in maxwell:  # Maxwell element
            sp_comp[:, i]=1/(g0[i]*gstar_maxwell(w*tau[i]))
        elif i in kww:  # kww (stretched exponential) elment
            sp_comp[:, i]=1/(g0[i]*gstar_kww(w*tau[i], beta[i]))
        elif i in rouse:  # Rouse element, beta is number of rouse modes
            sp_comp[:, i]=1/(g0[i]*gstar_rouse(w*tau[i], beta[i]))
        else:  # power law springpot element
            sp_comp[:, i]=1/(g0[i]*(1j*w*tau[i]) ** beta[i])

    # sp_vec keeps track of the beginning and end of each branch
    sp_vec=np.append(0, sp_type.cumsum())

    #  g_br keeps track of the contribution from each branch
    g_br={}
    for i in np.arange(n_br):
        sp_i=np.arange(sp_vec[i], sp_vec[i+1])
        # branch compliance obtained by summing compliances within the branch
        br_g[:, i]=1/sp_comp[:, sp_i].sum(1)
        g_br[i]=br_g[:, i]

    # now we sum the stiffnesses of each branch and return the result
    g_tot=br_g.sum(1)
    return g_br, g_tot


def simon_data(w):
    #  glassy and rubbery modulus for simon data
    # J. Applied Polymer Science 76, 495–508 (2000).
    # glassy and rubbery moduli
    Gg = 8.58e8
    Gr = 4.63e6
    tau = np.logspace(-7, 4, 23)
    tau = np.append(tau, 1e5)
    g = [0.0215, 0.0215, 0.0215, 0.0215, 0.0267, 0.0267, 0.0375, 0.0405,
         0.0630,
         0.0630, 0.1054, 0.1160, 0.1160, 0.1653, 0.0561, 0.0561, 0.0199,
         0.0119,
         0.0055, 0.0028, 0.0008, 0.0002, 0.0003, 0.0003]
    g = np.array(g)
    g0 = (Gg-Gr)*g
    
    # everything added so far is a maxwell element
    maxwell = np.arange(len(tau), dtype=int)
    beta = np.ones(len(tau))
    
    # add spring (springpot with beta= 0) to add relaxed modulus
    tau = np.append(tau, 1)
    beta = np.append(beta, 0)
    g0 = np.append(g0, Gr)
    
    # all elements are in parallel with one another
    sp_type = np.ones(len(tau), dtype=int)
    
    return springpot(w, g0, tau, beta,sp_type, maxwell=maxwell)


def vft(T, Tref, B, Tinf):

    """
    Vogel Fulcher Tamman Equation.

    args:
        T (real):
            Temperature (deg. C).
        Tref:
            Reference Temp (deg. C) where shift factor = 1.
        B:
            B (units of Kelvin)
        Tinf
            Vogel temperature (deg. C)

    returns:
        lnaT:
            Natural log of the shift factor.
    """
    return -B/(Tref-Tinf) + B/(T-Tinf)


def add_fstar_err(df, uncertainty_dict):
    """
    Add fstar uncertainties based on values given in uncertainty dic

    Args
    ----------
    df : dataframe
        Input solution dataframe, typically generated by solve_for_props.
    uncertainty_dict : dictionary
        Dictionary used to determine uncertainty in frequency and 
        dissipation (see definition of calc_fstar_err.

    Returns
    -------
    Input dataframe with values of fstar_err added

    """

    # add uncertainty columns and set to zero for now
    n_list = []
    for n in [1, 3, 5, 7, 9]:
        if f'df_expt{n}' in df.keys():
            n_list.append(n)
            if f'fstar_err{n}' not in df.keys():
                df.insert(df.columns.get_loc(f'df_expt{n}'), 
                            f'fstar_err{n}', 0+1j*0)

    for idx, row in df.iterrows():
        for n in n_list:
            f_err = np.real(calc_fstar_err(n, row, uncertainty_dict))
            g_err = np.imag(calc_fstar_err(n, row, uncertainty_dict))
            fstar_err = f_err + 1j*g_err
            df.loc[idx,f'fstar_err{n}'] = fstar_err
            
    return df
            
def make_response_maps(fig, ax, drho, **kwargs):
    """
    Make response maps of the QCM response - delf and delg
    Args:
        fig (figure handle):
            figure to use for the plot
        ax (list of 2 axis handles):
            axes to use for plot
        drho (float or string):
            value of drho for the response map.  Set to 
            'Sauerbrey' (default) if we normalize by Sqauerbrey shift
        
    Kwargs:
        axtitles (list of 2 strings):
            axis titles (default is ['(a)', '(b)']
            
    """
    numxy=kwargs.get('numxy', 100)
    numz=kwargs.get('numz', 200)
    philim=kwargs.get('philim', [0.001, 90])
    dlim=kwargs.get('dlim', [0.001, 0.5])
    dlim[0] = max(dlim[0], 0.001)
    autoscale = kwargs.get('autoscale', False)
    contour_range = kwargs.get('contour_range', {0:[-3, 3], 1:[0,3]})
    first_plot = kwargs.get('first_plot', 0)

    def Zfunction(x, y):
        if drho == 'Sauerbrey':
            grho3 = np.nan
        else:
            grho3=grho_from_dlam(3,drho, x, y)
        fnorm=normdelf_bulk(3, x, y)
        gnorm=normdelg_bulk(3, x, y)

        Zstar = normdelfstar(3, x, y)
        # convert to Hz by multiplying by sauerbrey shift
        if drho !='Sauerbrey':
            Zstar = Zstar*sauerbreyf(1, drho)
            
        return np.real(Zstar), np.imag(Zstar), grho3, fnorm, gnorm
        
    # make meshgrid for contour
    phi=np.linspace(philim[0], philim[1], numxy)
    dlam=np.linspace(dlim[0], dlim[1], numxy)
    DLAM, PHI=meshgrid(dlam, phi)
    Z1, Z2, grho3, fnorm, gnorm  = Zfunction(DLAM, PHI)
       
    # specify the range of the Z values
    if autoscale:
        min0=Z1.min()
        max0=Z1.max()
        min1=Z2.min()
        max1=Z2.max()
    else:
        min0 = contour_range[0][0]
        max0 = contour_range[0][1]
        min1 = contour_range[1][0]
        max1 = contour_range[1][1]
 
    levels1=np.linspace(min0, max0, numz)
    levels2=np.linspace(min1, max1, numz)
    
    contour1=ax[0].contourf(DLAM, PHI, Z1, levels=levels1,
                              cmap='rainbow')
    contour2=ax[1].contourf(DLAM, PHI, Z2, levels=levels2,
                              cmap='rainbow')
    ax[0].sharex(ax[1])
    ax[0].sharey(ax[1])
    
    cbax1 = ax[0].inset_axes([1.05, 0, 0.1, 1])
    cbar1 = fig.colorbar(contour1, ax=ax[0], cax = cbax1)
    cbax2 = ax[1].inset_axes([1.05, 0, 0.1, 1])
    cbar2 = fig.colorbar(contour2, ax=ax[1], cax = cbax2)
    
    # set ticks for the colorbars
    cbar1.set_ticks(np.linspace(min0, max0, 11))
    cbar2.set_ticks(np.linspace(min1, max1, 11))
    
    # set ticks for the contour plots
    ax[0].set_yticks(np.linspace(philim[0], philim[1], 7))
 
    # set formatting for parameters that appear at the bottom of the plot
    # when mouse is moved
    def fmt(x, y):
        z1, z2, grho3, fnorm, gnorm = Zfunction(x, y)
        
        return 'd/lambda={x:.3f},  phi={y:.1f}, delfstar/n={z:.0f}, '\
                'grho3={grho3:.2e}, '\
                'fnorm={fnorm:.4f}, gnorm={gnorm:.4f}'.format(x=x, y=y,
                 z=z1+1j*z2,
                 grho3=grho3/1000,
                 fnorm=fnorm, gnorm=gnorm)

    for k in [0,1]:
        ax[k].set_xlabel(r'$d/\lambda_n$')
        ax[k].set_ylabel(r'$\Phi$ (deg.)')
        ax[k].format_coord=fmt
        
    if drho == 'Sauerbrey':
        title_units = [r'$\Delta f_n /\Delta f_{sn}$',
                       r'$\Delta \Gamma _n/ \Delta f_{sn}$']
    else:
        title_units = [r'$\Delta f_n$ (Hz): $d\rho$='+
                       f'{1000*drho}'+r' $\mu$m$\cdot$g/cm$^3$',
                       r'$\Delta \Gamma _n$ (Hz): $d\rho$='+
                       f'{1000*drho}'+r' $\mu$m$\cdot$g/cm$^3$']
        
    for k in [0, 1]:
        ax[k].set_title(f'{titles_default[k+first_plot]} {title_units[k]}')
            
        # set labels for contour plots
        ax[0].set_xlabel(r'$d/\lambda_n$')
        ax[0].set_ylabel(r'$\Phi$ ($\degree$)')
        ax[1].set_xlabel(r'$d/\lambda_n$')
        ax[1].set_ylabel(r'$\Phi$ ($\degree$)')
        
            
    return {'fig':fig, 'ax':ax, 'cbax1':cbax1, 'cbax2':cbax2,
            'cbar1':cbar1, 'cbar2':cbar2}



        


        
            

       


 

def check_n_dependence(soln, **kwargs):
    '''
    Compare experimental and back-calculated frequency and dissipation shifts
    for all harmonics for a given time point.

    args:
        df (dataframe):
            input solution to consider
        nvals (list of integers):
            harmonics to plot
            default is (1,3,5)

    kwargs:
        index (integer):
            index to consider (default is minimum value)
        suptitle (string):
            suptitle for figure (default is '')
            
    Returns:
        fig, ax for figure with n dependence of frequency and dissipation shifts

    '''
    idx = kwargs.get('index', soln.index.min())
    filename = kwargs.get('filename', 'n_dependence.pdf')
    suptitle = kwargs.get('suptitle', '')
    nvals = kwargs.get('nvals', [1,3,5])
    delf_expt=[]
    delf_calc=[]
    delg_expt=[]
    delg_calc=[]

    fig, ax = plt.subplots(1, 2, figsize=(7,3),constrained_layout=True, 
                           num=filename)
    for n in nvals:
        delf_expt.append(np.real(soln['df_expt'+str(n)][idx])/n)
        delf_calc.append(np.real(soln['df_calc'+str(n)][idx])/n)
        delg_expt.append(np.imag(soln['df_expt'+str(n)][idx])/n)
        delg_calc.append(np.imag(soln['df_calc'+str(n)][idx])/n)

    p = ax[0].plot(nvals, delf_expt, '+', label='expt')
    color = p[0].get_color()
    ax[0].plot(nvals, delf_calc, '-o', color=color, label='calc',
                      markerfacecolor='none')
    ax[1].plot(nvals, delg_expt, '+', label='expt')
    ax[1].plot(nvals, delg_calc, '-o', color=color, label='calc',
                      markerfacecolor='none')
    
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel(r'$\Delta f_n/n$ (Hz)')
    ax[1].set_ylabel(r'$\Delta \Gamma_n/n$ (Hz)')
    for p in [0,1]:
        ax[p].set_xlabel('$n$')
        ax[p].set_xticks(np.arange(min(nvals), max(nvals)+2,2))
    
    # mark the values that correspond to the solution
    calc = soln['calc'][idx].split('.')
    n1 = int(calc.split('.')[0])
    n2 = int(calc.split('.')[1])
    n3 = int(calc.split('.')[2])
    ax[0].plot(n1, np.real(soln['df_expt'+str(n1)][idx])/n1, 'ro')
    ax[0].plot(n2, np.real(soln['df_expt'+str(n2)][idx])/n2, 'ro')    
    ax[1].plot(n3, np.imag(soln['df_expt'+str(n3)][idx])/n2, 'ro') 
    
    # print the figure
    fig.suptitle(suptitle)
    fig.savefig(filename)

def add_QCM_functions():
    ken_path = '/home/ken/Mydocs/Github/rheoQCM/QCMFuncs'
    # copy QCM_functions to current working directory (ken only, so others have
    # updated version without fiddling with GitHub)
    if os.path.isdir(ken_path):
        shutil.copy(os.path.join(ken_path, 'QCM_functions.py'), os.getcwd())
        
def add_t_diff(df):
    # add time until previous and next point in the file.  Helpful if we want to
    # use data collected at beginning or end of a relaxation step
    df.insert(2, 't_prev', 'nan')
    df.insert(3, 't_next', 'nan')
    df['t_next'] = -df['t'].diff(periods=-1)    
    df.iloc[-1, df.columns.get_loc('t_next')] = np.inf
    df['t_prev'] = -df['t'].diff(periods=1)
    df.iloc[0, df.columns.get_loc('t_prev')] = np.inf
    return df

# extraction of data from MATLAB fig file
def getxy_from_MATLAB_fig(filename):
    d = loadmat(filename,squeeze_me=True, struct_as_record=False)
    matfig = d['hgS_070000']
    childs = matfig.children
    ax = [c for c in childs if c.type == 'axes']
    x={}
    y={}
    Color = {}
    Marker = {}
    nax = len(ax)
    for k in np.arange(len(ax)):
        x[k]={}
        y[k]={}
        Color[k]={}
        Marker[k]={}
        counter = 0    
        for line in ax[k].children:
            if line.type == 'graph2d.lineseries':            
                x[k][counter] = line.properties.XData
                y[k][counter] = line.properties.YData
                Color[k][counter] = line.properties.Color
                Marker[k][counter] = line.properties.Marker
            counter += 1
    return nax, x, y, Color, Marker

# now fit the Kotula model
def kotula_single(xi, Gmstar, Gfstar, xi_crit, s,t):
    gstar = np.array([], dtype = complex) 
    def ftosolve(gstar):  
        A = (1-xi_crit)/xi_crit
        gstar = ((1-xi)*(Gmstar**(1/s)-gstar**(1/s))/(Gmstar**(1/s)+A*gstar**(1/s)) +
                 xi*(Gfstar**(1/t)-gstar**(1/t))/(Gfstar**(1/t)+A*gstar**(1/t)))
        return gstar
    gstar =findroot(ftosolve, Gmstar)
    return complex(gstar)

kotula = np.vectorize(kotula_single)

def abs_kotula(xi, Gmstar, Gfstar, xi_crit, s, t):
    return abs(kotula(xi, Gfstar, Gmstar, xi_crit, s, t))





