#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Current version:  2025.06.18
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
from glob import glob
import time
from mpmath import findroot
from scipy.io import loadmat
from pylab import meshgrid
import pandas as pd
from copy import deepcopy, copy
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import re
import warnings

# Suppress the specific warning when working with All-nan arrays
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


try:
  import seaborn as sns
  # Set the Seaborn colorblind palette as the default color cycle for Matplotlib
  colorblind_palette = sns.color_palette("colorblind")
  plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colorblind_palette)
except ImportError:  
  pass


try:
  from kww import kwwc, kwws
  # kwwc returns: integral from 0 to +infinity dt cos(omega*t) exp(-t^beta)
  # kwws returns: integral from 0 to +infinity dt sin(omega*t) exp(-t^beta)
except ImportError:
  pass

            
# set up colors we'll use to plot different harmonics
col = {1:'C0', 3:'C1', 5:'C2', 7:'C3', 9:'C4'}

# setvvalues for standard constants
Zq = 8.84e6  # shear acoustic impedance of at cut quartz
f1_default = 5e6  # fundamental r_defaault_defaaultesonant frequency
e26 = 9.65e-2

# Half bandwidth of unloaed resonator (intrinsic dissipation on crystalline quartz)
g0_default = 50

# note that these values give constant delf/n for n=3, 5, 7
T_coef_default = {'f': {1: [0.00054625, 0.04338, 0.08075, 0],
                       3: [0.0017, -0.135, 8.938, 0],
                       5: [0.002833, -0.225, 14.890, 0],
                       7: [0.00397, -0.315, 20.855, 0],
                       9: [0.0051, -0.405,  26.8125, 0]},
                  'g': {1: [0, 0, 0, 0],
                        3: [0, 0, 0, 0],
                        5: [0, 0, 0, 0],
                        7: [0, 0, 0, 0],
                        9: [0, 0, 0, 0]}}

electrode_default = {'drho': 2.8e-3, 'grho3': 3.0e14, 'phi': 0}
water = {'drho':np.inf, 'grho3':9.4e7, 'phi':90}
air = {'drho':np.inf, 'grho3':0, 'phi':90}

   
# make dictionary of default titles
titles_default =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
# make a dictionary of the potential axis labels
axlabels = {'drho': r'$d\rho$ ($\mu$m$\cdot$g/cm$^3$)',
            'grho3': r'$|G_3^*|\rho$ (Pa $\cdot$ g/cm$^3$)',
            'phi': r'$\phi$ (deg.)',
            'phi.tan': r'tan$\phi$',
            'grho3p': r'$G^\prime_3\rho$ (Pa $\cdot$ g/cm$^3$)',
            'grho3pp': r'$G^{\prime\prime}_3\rho$ (Pa $\cdot$ g/cm$^3$)',
            'deltarho3': r'$\delta\rho$ ($\mu$m $\cdot$ g/cm$^3$)',
            'AF':r'AF',
            'jdp': r'$J^{\prime \prime}/\rho$ (Pa$^{-1}\cdot$cm$^3$/g)',
            'temp':r'$T$ ($^\circ$C)',
            'etarho3':r'$|\eta_3^*| \rho$ (mPa$\cdot$s$\cdot$g/cm$^3$)',
            'delf':r'$\Delta f$ (Hz)',
            'delg':r'$\Delta \Gamma$ (Hz)',
            'delf_n':r'$\Delta f_n/n$ (Hz)',
            'delg_n':r'$\Delta \Gamma _n /n$ (Hz)',
            's':'t (s)',
            'min':'t (min)',
            'hr':'t (hr)',
            'day':'t (day)',
            'index':'index'}

def drho_q(**kwargs):
    f1 = kwargs.get('f1', f1_default)
    return Zq/(2*f1)

def drho_label(ext):
    '''
    Create axis labels for drho.
    
    Parameters
    ----------
    ext : list of strings
        A list of potential switches, generlly obtained by splitting a 
        delimited input. Defaults to microns*density if empty, and is adjusted
        by the following:
            
        - nm:  units of nm instead of microns
        - diff:  change in mass (requires drho_ref for plotting)
        - norm:  normalized (requires drho_ref for plotting)

    Returns
    --------
    label : string
        String to use for axis label.
    '''
    str2 = '('
    # change to change in mass or normalzied mass, if desired
    if 'diff' in ext:
        str1 = r'change in $d\rho$ '
    elif 'norm' in ext:
        str1 = r'$d\rho$/$d\rho_{ref}'
        str3 = ''
    else:
        str1 = r'$d\rho$ '
        
    # now add g/m^2 or mg/m^2 unit if desired
    if 'add_m/a' in ext:
        if 'nm' in ext:
            str2 = str2 + r'mg/m$^2$ or '
        else:
            str2 = str2 + r'g/m$^2$ or '
        
    # use nm instead of microns if desired
    if 'nm' in ext:
        str3 = r'nm$\cdot$g/cm$^3$)'
    else:
        if 'norm' not in ext:
            str3 = r'$\mu$m$\cdot$g/cm$^3$)'
            
    return str1+str2+str3

def sig_figs(x, n):
    '''
    # rounds x to n significant figures
    
    Parameters
    ----------
    x : real
        number to be rounded
    n : integer
        significant figures to round to
        
    Returns
    ---------
    Round_nun : real
        input rounded to n significant figures
    '''
    if x == np.nan or x == np.inf:
        return x
    else:
        round_num = -int(np.floor(np.log10(abs(x))))+n-1
    return (round(x, round_num))


def find_nearest_idx(values, array):
    """
    Find index of a point with value closest to the one specified.
    
    Parameters
    ---------
    values : list or numpy array
        Values that we want to be close to.
    array : numpy array
        Input array to choose values from.

    Returns
    --------

    idx : numpy array
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
    
    Parameters
    ---------
        x : numpy array
            Input array containing the values of interest.
        range : 2 element list or numpy array
            Minimum and maximum values of the specified range.

    Returns
    ---------
        idx : numpy array
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
    ax2.set_yscale(ax.get_yscale())
    return ax2


def update_twin(ax):
    """
    Updates any axes that are twins of x (typically etarho3, logphi)

    Parameters
    ----------
    ax : axis
        Axis that has twins.

    Returns
    -------
    Twinned axis - generally not used

    """
    twinax = [a for a in ax.figure.axes if a is not ax and 
                   a.bbox.bounds == ax.bbox.bounds]
    twinax = twinax[0]
    
    if twinax.get_ylabel() == axlabels['phi']:
        yticks = ax.get_yticks()
        ylim = ax.get_ylim()
        twinax.set_yticks(yticks)
        twinax.set_ylim(ylim)
        twinax.set_yticklabels(np.tan(np.radians(ax.get_yticks())))
        twinax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    
    return twinax
    

def add_D_axis(ax):
    """
    Add right hand axis with dissipation.
    
    Parameters
    ---------
    ax : axis handle
        Axis that we are going to work with.

    Returns
    ---------
    axD : axis handle
        Axis with dissipation added.
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
    
    Parameters
    ----------
    n : interger
        Harmonic of interest.
    drho : float
        Mass per unit area.

    Returns
    --------
    deltaf : float
        Calculated Sauerbrey frequency shift.
    """
    f1 = kwargs.get('f1', f1_default)
    return 2*n*f1 ** 2*drho/Zq


def sauerbreym(n, delf, **kwargs):
    """
    Calculate Sauerbrey mass from frequency shift.
    
    Parameters
    ----------
    n : integer
        Harmonic of interest.
    delf : real float
        Frequency shift in Hz.

    Returns
    ---------
    drho : real float
        Sauerbrey mass in kg/m^2.
    """
    f1 = kwargs.get('f1', f1_default)
    return -delf*Zq/(2*n*f1 ** 2)


def etarho(n, props, **kwargs):
    """
    Use power law formulation to get |eta*|rho at specified harmonic,
    with properties at n=3 as an input.

    Parameters
    ----------
    n : integer
        Harmonic of interest.
    props : dictionary
        Dictionary of material properties, which must contain
        grho3 and phi. This can also be a dictionary of dictionaries,
        in which case a numpy array of viscosity values is returned.

    Returns
    --------
    eta_rho_mag : real
         |eta*|rho at harmonic of interest in SI units.
    """
    f1 = kwargs.get('f1', f1_default)
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
    Use power law formulation to get |G*|rho at different harmonics,
    with properties at n=3 as an input.
    
    Parameters
    -----------
    n : integer
        Harmonic of interest.
    props : dictionary:
        Dictionary of material properties, which must contain
        grho3 and phi.

    Returns
    --------
    grho_mag : real
        |G*rho| at harmonic of interest.
    """
    grho3 = props['grho3']
    phi = props['phi']
    return grho3*(n/3) ** (phi/90)


def calc_grho3(n, grhostar):
    """
    Use power law formulation to get |G*|rho at n=3.
    
    Parameters
    ----------
    
    n : integer
        Harmonic of interest.
    grhostar (complex):
        |G*|rho at harmonic of interest.

    Returns
    ---------
        grho3 : real
            |G*|rho (SI units) at n=3.
        
        phi : real
            Phase angle (degrees) - assumed independent of n.
    """
    phi = np.angle(grhostar, deg=True)
    grhon = abs(grhostar)
    grho3 = grhon*(3/n)**(phi/90)
    return grho3, phi


def calc_jdp(grho):
    """
    Calculte the loss compliance, J", normalized by rho, with Gstar*rho
    as the input.

    Parameters
    ---------
    grho : complex
        complex Gstar multiplied by density

    Returns
    -------
    jdp : real
        imaginary part of complex Jstar (shear compliance), divided by
        density
    """
    return (1/abs(grho))*np.sin(np.angle(grho))


def grho_from_dlam(n, drho, dlam, phi, **kwargs):
    """
    Obtain |G*|\rho from d/lambda.
    
    Parameters
    ----------
    n : integer
        Harmonic of interest.
    drho : real
        Mass thickness in kg/m^2.
    dlam : real
        d/lambda at harmonic of interest
    phi : real
        phase angle in degrees

    Returns
    -------
    grho : real
        |G*|*density at harmonic of interest.
    """
    f1 = kwargs.get('f1', f1_default)
    # min dlam value for calculation is 0.001
    dlam_new = copy(dlam)
    dlam_new[dlam_new==0]=0.001
    return (drho*n*f1*np.cos(np.deg2rad(phi/2))/dlam_new) ** 2


def grhostar_bulk(delfstar, **kwargs):
    """
    Obtain complex Gstar from for bulk material (infinite thicknes).
    
    Parameters
    ----------
    delfstar : complex or numpy array of complex numbers
        Complex frequency shift in Hz.

    Returns
    -------
    grhostar : complex
        |G*|*rho at harmonic corresonding to delfstar.
    """
    f1 = kwargs.get('f1', f1_default)
    return -(np.pi*Zq*delfstar/f1) ** 2


def deltarho_bulk(n, delfstar, **kwargs):
    """
    Calculate decay length multiplied by density for bulk material.
    
    Parameters
    ----------
    n : integer
        Harmonic of interest.
    delfstar : complex number or numpy array of complex numbers
        Complex frequency shift in Hz.
            
    Returns
    -------
    deltarho : real
        Decay length multiplied by density (SI units).
    """
    f1 = kwargs.get('f1', f1_default)
    return -Zq*abs(delfstar[n])**2/(2*n*f1**2*delfstar[n].real)


def calc_D(n, props, delfstar, **kwargs):
    """
    Calculate D (dk*, thickness times complex wave number).
    
    Parameters
    ----------
    n : integer
        Harmonic of interest.
    props : dictionary
        Dictionary of material properties, which must contain
        'grho3' and 'phi'.
    delfstar : complex
        Complex frequency shift at harmonic of interest (Hz).
        
    kwargs
    ----------
    calctype : string
        One of the following calculation types
        - 'SLA' (default): small load approximation with power law model
        - 'LL': Lu Lewis equation, using default or provided electrode props
        - 'Voigt': small load approximation

    Returns
    -------
        D : complex
            Thickness times complex wave number.
    """
    calctype = kwargs.get('calctype', 'SLA')
    f1 = kwargs.get('f1', f1_default)
    drho = props['drho']
    # ignore divide by zero errors that come up in some special cases
    with np.errstate(divide='ignore'):
        value = 2*np.pi*(n*f1+delfstar)*drho/zstar_bulk(n,
                       props, calctype)
    return value


def zstar_bulk(n, props, calctype):
    """
    Calculate complex acoustic impedance for bulk material.
    
    Parameters
    ----------
    n : integer
        Harmonic of interest.
    props : dictionary
        Dictionary of material properties, which must contain
        'grho3' and 'phi'.
    calctype : string
        One of the following calculation types:
        - 'SLA' (default): small load approximation with power law model
        - 'LL': Lu Lewis equation, using default or provided electrode props
        - 'Voigt': small load approximation

    Returns
    -------
        Acoustic impedance  (square root of gstar*density
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


def calc_delfstar_sla(ZL, **kwargs):
    """
    Calculate complex frequency shift from load impedance using small
    load approximation.
    args:
        ZL (complex):
            complex load impedance (SI units).

    returns:
        Complex frequency shift, delfstar (Hz).
    """
    f1 = kwargs.get('f1', f1_default)
    return f1*1j*ZL/(np.pi*Zq)


def calc_ZL(n, layers, delfstar, **kwargs):
    """
    Calculate complex load impendance for stack of layers of known props.
    Layers are assumed to be laterally homogeneous
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
    calctype = kwargs.get('calctype', 'SLA')

    # we use the matrix formalism to avoid typos and simplify the extension
    # to large N.
    for i in np.arange(layer_min, layer_max):
        Z[i] = zstar_bulk(n, layers[i], calctype)
        D[i] = calc_D(n, layers[i], delfstar, **kwargs)
        L[i] = np.array([[np.cos(D[i])+1j*np.sin(D[i]), 0],
                 [0, np.cos(D[i])-1j*np.sin(D[i])]])

    # get the terminal matrix from the properties of the last layer
    if 'Zf' in layers[layer_max].keys():
        Zf_max = layers[layer_max]['Zf'][n]
    else:
        D[layer_max] = calc_D(n, layers[layer_max], delfstar, 
                              **kwargs)
        Zf_max = 1j*zstar_bulk(n, layers[layer_max], 
                               calctype)*np.tan(D[layer_max])

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
    ZL = Z[layer_min]*(1-rstar)/(1+rstar)
    
    # account for the possibility of a fractional layer
    # only one of the layers can have fractional coverage
    for i in np.arange(layer_min, layer_max):
        if 'AF' in layers[i].keys():
            AF = layers[i]['AF']
            layers_ref = delete_layer(layers, 1)
            ZL_ref = calc_ZL(n, layers_ref, 0, **kwargs)
            ZL = AF*ZL+(1-AF)*ZL_ref
    return ZL

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

def calc_delfstar(n, layers_in, **kwargs):
    """
    Calculate complex frequency shift for stack of layers.
    args:
        n (int):
            Harmonic of interest.

        layers_in (dictionary):
            Dictionary of material dictionaries specifying the properites of
            each layer. 
            it contains dictionaries
            labeled from 1 to N, with 1
            being the layer in contact with the QCM.  Each dictionary must
            include values for 'grho3, 'phi' and 'drho'.
            layer 0 is the electrode itself.


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
    if not layers_in:  # if layers is empty {}
        return np.nan
    layers = deepcopy(layers_in)
    
    # run quick check to make sure we don't have an infinite layer 
    # not at the top
    for layernum in np.arange(min(layers.keys()), max(layers.keys())):
        if layers[layernum]['drho'] == np.inf:
            print('only outermost layer can have infinite thickness')

    ZL = calc_ZL(n, layers, 0, **kwargs)
    if (reftype=='overlayer') and (2 in layers.keys()):
        layers_ref = delete_layer(layers, 1)
        ZL_ref = calc_ZL(n, layers_ref, 0, **kwargs)
    else:
        ZL_ref = 0
    
    del_ZL = ZL-ZL_ref
    
    if calctype != 'LL':
        # use the small load approximation in all cases where calctype
        # is not explicitly set to 'LL'
        return calc_delfstar_sla(del_ZL, **kwargs)

    else:
        # this is the most general calculation
        # use default electrode if it's not specified

        if 0 not in layers:
            layers[0] = electrode_default
        
        layers_all = deepcopy(layers)
        if reftype == 'overlayer':
            layers_ref = delete_layer(deepcopy(layers), 1)
        else:
            layers_ref = {0:layers[0]}

        ZL_all = calc_ZL(n, layers_all, 0, **kwargs)
        delfstar_sla_all = calc_delfstar_sla(ZL_all, **kwargs)
        ZL_ref = calc_ZL(n, layers_ref, 0, **kwargs)
        delfstar_sla_ref = calc_delfstar_sla(ZL_ref, **kwargs)

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


def calc_Zmot(n, layers, delfstar, **kwargs):
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

    kwargs:
        g0 (real):
            Dissipation at n for dissipation.  Default value set at top
            of QCM_functions.py (typically 50).
            
        calctype (string):
            Generally passed from calling function.  Should always be 'LL'.

    returns:
        delfstar (complex):
            Complex frequency shift (Hz).
    """
    f1 = kwargs.get('f1', f1_default)
    g0 = kwargs.get('g0', g0_default)
    om = 2 * np.pi * (n*f1 + delfstar)
    Zqc = Zq * (1 + 1j*2*g0/(n*f1))

    Dq = om*drho_q/Zq
    secterm = -1j*Zqc/np.sin(Dq)
    ZL = calc_ZL(n, layers, delfstar, **kwargs)
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


def calc_dlam(n, film, **kwargs):
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
    return calc_D(n, film, 0, **kwargs).real/(2*np.pi)

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
    

def calc_lamrho(n, grho3, phi, **kwargs):
    """
    Calculate lambda*\rho at specified harmonic.
    args:
        n (int):
            Harmonic of interest.

        grho3 (real):
            |G*|$\rho$  at n=3 (SI units).

        phi (real):
            Phase angle (degrees).

    returns:
        shear wavelength times density in SI units
    """
    f1 = kwargs.get('f1', f1_default)
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
    '''
    linear relationship between phi and Grho3 suggested
    by Kazi's hydrophobic polyelectrolyte complex paper 
    SI units
    '''
    logG = np.log10(grho3)
    if logG <=8:
        return 90
    elif logG >=12:
        return 0
    else:
        return 90-90*(logG-8)/4


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
    D = 2*np.pi*dlam(n, dlam3, phi)*(1-1j*np.tan(np.deg2rad(phi/2)))
    return -np.sinc(D/np.pi)/np.cos(D)
    # note:  sinc(x)=sin(pi*x)/(pi*x)
            
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
    
    # to avoid divergence for phi = 0 we set 0.1 degree as floor for phi
    phi_new = copy(phi)
    if isinstance(phi_new, np.ndarray):
        phi_new[phi_new==0]=0.1
    answer = np.real(2*np.tan(2*np.pi*dlam(n, dlam3, phi_new) *
        (1-1j*np.tan(np.deg2rad(phi_new/2)))) /
        (np.sin(np.deg2rad(phi_new))*(1-1j*np.tan(np.deg2rad(phi_new/2)))))
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


def bulk_props(delfstar, **kwargs):
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
    f1 = kwargs.get('f1', f1_default)
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
    calc = update_calc(calc)
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
        if f'delfstar_expt_{n}' in df_soln.columns.values:
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
        if f'delfstar_expt_{n}' in delfstar.keys():
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
    gmax = kwargs.get('gmax', np.inf)
    calctype = kwargs.get('calctype', 'SLA')
    reftype = kwargs.get('reftype', 'bare')
    
    if 't' in delfstar.keys() and not 't_next' in delfstar.keys():
        delfstar = add_t_diff(delfstar)
        
    # check to see if there are any nan values in the harmonics we need  
    delfstar_mod = copy(delfstar)
    n_unique = nvals_from_calc(calc)[3]
    for n_to_drop in n_unique:
        delfstar_mod  = delfstar_mod.dropna(subset = 
                                            f'delfstar_expt_{n_to_drop}')

    # also set delfstar to nan for gamma exceeding gmax
    for n in n_unique:
        index_overdamped = delfstar_mod[(np.imag(delfstar_mod[f'delfstar_expt_{n}'])
                                         >gmax)].index
        delfstar_mod.drop(index_overdamped, inplace = True)
          
    # add time and temp infor if it exists
    var_list = []
    for var in ['t', 't_prev', 't_next', 'temp']:
        if var in delfstar_mod.keys():
            var_list.append(var)
          
    df_soln = copy(delfstar_mod[var_list])

    # now add complex frequency and frequency shifts
    npts = len(df_soln.index)
    complex_series = pd.Series([np.nan+1j*np.nan] * len(df_soln), dtype='complex128')
    for n in find_nplot(delfstar):
        # add experimental delf, delfstar and empty column for calc. delfstar
        if f'fstar_{n}_dat' in delfstar_mod.keys():
            # sometimes we only have delfstar and not the actual frequency
            df_soln.insert(df_soln.shape[1], f'fstar_expt_{n}', 
                           delfstar_mod[f'fstar_{n}_dat'])        
        df_soln.insert(df_soln.shape[1], f'delfstar_expt_{n}', 
                       delfstar_mod[f'delfstar_expt_{n}'])
        df_soln.insert(df_soln.shape[1], f'delfstar_calc_{n}', complex_series)
        
        
   
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
    
    # add column for AF_1
    df_soln['AF_1'] = 1.0
    
    # now add columns for Jacobian and layers
    df_soln.insert(df_soln.shape[1], 'jac', object_series)
    df_soln.insert(df_soln.shape[1], 'layers', object_series)
  
    # now add columns for 'calctype'
    calctype_array = np.array(npts*[calctype])
    df_soln.insert(df_soln.shape[1], 'calctype', calctype_array)
    
    # now add columns for 'reftype'
    reftype_array = np.array(npts*[reftype])
    df_soln.insert(df_soln.shape[1], 'reftype', reftype_array)
       
    return df_soln, delfstar_mod


def compare_calc_expt(layers, row, calc, **kwargs):
    """
    Compare experimental and calculated values of delfstar
    
    args:
        layers (datafame):
            input properties 
        row (series):
            single row from input delfstar dataframe, obtained from itertuples
            
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
                 np.real(getattr(row, f'delfstar_expt_{n}')))
        vals.append(val)
    for n in ng: 
        val = (calc_delfstar(n, layers, calctype=calctype,
                               reftype=reftype).imag -
                 np.imag(getattr(row, f'delfstar_expt_{n}')))
        vals.append(val)
    return vals


def extract_props(soln_df, props_calc):

    propvals=[]
    for prop in props_calc:
        if len(prop.split('_')==1):
            prop = prop+'_1'
        propvals.append(soln_df[props_calc])
    return propvals
        

def update_layers(props, values, layers, **kwargs):
    '''
    Updates layers dictionary, substituting the specified values
    into the properties specified by props_calc.  
    
    Parameters
    ----------
    props_calc : list
        List of values from the layers dictionary to be updated.
    soln_df : Either a list of values corresponding to the 
        value of props_calc (in the same order), or a dataframe row or pandas
        series from which they are taken.
    layers : Dictionary
        The updated layers dictionary.
    
    Returns
    -------
    Updated layers dictionary.
    '''
    # 
    # make sure order of values corresponds to order of props_calc
    if isinstance(values, pd.DataFrame):
        for input_string in props:
            [prop, layer] = input_string.split('_')
            layer = int(layer)
                    
    else:
        for input_string, value in zip(props, values): 
            [prop, layer] = input_string.split('_')
            layer = int(layer)
            if value != 'no_change':
                layers[layer][prop] = value
            if prop=='grho3' and 'phi_parms' in kwargs.keys():
                phi_parms=kwargs.get('phi_parms')
                layers[layer]['phi']=calc_phi(layers[layer]['grho3'], phi_parms)
    return layers

def add_layer_num(prop):
    # add _1 to property string if it is not already include
    if re.match(r'^(grho3|phi|drho)+(\..+)?$', prop):
        # Insert '_1' before the first dot
        prop = prop.replace('.', '_1.', 1) if '.' in prop else prop + '_1'
    return prop

def add_layer_nums(props):
    return [add_layer_num(prop) for prop in props]


def guess_from_layers(props, layers):
    guess = [None]*len(props)
    for i, input_string in enumerate(props): 
        [prop, layer] = input_string.split('_')
        layer = int(layer)
        guess[i] = layers[layer][prop]
    return guess


def update_df_soln(df_soln, soln, idx, layers, props_calc, reftype):
    for layer_num in layers.keys():
        for prop in ['grho3', 'phi', 'drho']:
            df_soln.loc[idx, f'{prop}_{layer_num}'] = layers[layer_num][prop] 
    
    if 'AF' in layers[1].keys():
        df_soln.loc[idx, 'AF_1'] = layers[1]['AF']
    
    nvals = nvals_from_df_soln(df_soln)
    for n in nvals:
        df_soln.at[idx, f'delfstar_calc_{n}']=calc_delfstar(n, layers,
                                                     reftype=reftype)
        
    df_soln.at[idx, 'layers'] = deepcopy(layers)
    df_soln.at[idx, 'jac'] = (soln['jac']).astype(object)
    df_soln.at[idx, 'dlam3_1'] = calc_dlam(3, layers[1])
    df_soln.at[idx, 'props_calc'] = props_calc
        
    return df_soln


def solve_for_props(delfstar, calc, props_calc, layers_in, **kwargs):
    """
    Solve the QCM equations to determine the properties.

    Parameters
    ----------
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
            

    kwargs
    --------
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
        ex: {'ghro3_1':1e8', 'phi_1':0, 'drho_1':0}

        
    ub (dictionary):  
        dictionary of upper bounds. keys must correspond to props.
        ex: {'ghro3_1':1e13, 'phi_1':90, 'drho_1':0}

    reftype (string):
        Specification of the reference. 
        
        - 'bare' (default): Reference is bare crystal.
    
        - 'overlayer':  reference corresponds to drho_1=0.
        (only used if layer 2 exists)
        
    gmax (real):
        Maximum value of dissipation (in Hz) for calculation.
        - default is 20,000 Hz
    
    accuracy (real):
        Max difference between acltual and back-calculated delf, delg
        - deault is 1 Hz 
        - this is not the uncertainty in delf, delg but is used to check
        that a solution exists
            
    showvals (Boolean):
        True if we want to displacy solutions as they are generated
        - default is False
        
    phi_parms (list)
        Values used to calculate phi from grho3, assuming linear relationship
        between phi and log(grho3) - used for polymer solutions
        - first value is liquid viscosity (times density)
        - second value is solid modulus (times density)
            
    Returns
    -------
    df_soln (dataframe):
        Dataframe with properties added, deleting rows with any NaN \
            values that didn't allow calculation to be performed.

    """
    # add layer number as 1 if not specified
    calc = update_calc(calc)
    layers = deepcopy(layers_in)
    for i, prop in enumerate(props_calc):
        # remove any plotting designations ('log', etc.)
        prop = prop.split('.')[0]
        if len(prop.split('_'))==1:
            props_calc[i]=prop+'_1'
        else:
            props_calc[i] = prop
                       
    calctype = kwargs.get('calctype', 'SLA')
    reftype = kwargs.get('reftype', 'bare')
    gmax = kwargs.get('gmax', 20000)
    
    # create df_soln dataframe                   
    df_soln, delfstar_mod = make_soln_df(delfstar, calc, props_calc, layers,
                           gmax=gmax, calctype=calctype, reftype=reftype) 
    
    if len(props_calc)==0:
        return df_soln
    
    # obtain starting guess from layers dictionary
    guess = guess_from_layers(props_calc, layers)
    
    # set default upper and lower bounds and guessfor properties
    default_prop_min = {'AF':0,'grho3':1e4, 'phi':0, 'drho':0}
    default_prop_max = {'AF':1,'grho3':1e13, 'phi':90, 'drho':3e-2}
    
    prop_min = kwargs.get('lb' ,default_prop_min)
    prop_max = kwargs.get('ub', default_prop_max)
    showvals = kwargs.get('showvals', False)

    # note that if we specify lb, make sure to specify ub as well
    lb = [None]*len(props_calc)
    for i, prop_string in enumerate(props_calc):
        prop = prop_string.split('_')[0]
        if prop_string in prop_min.keys():
            lb[i] = prop_min[prop_string]
        else:
            lb[i] = prop_min[prop]
        guess[i]=max(guess[i], lb[i])    
        

    ub = [None]*len(props_calc)
    for i, prop_string in enumerate(props_calc):
        prop = prop_string.split('_')[0]
        if prop_string in prop_max.keys():
            ub[i] = prop_max[prop_string]
        else:
            ub[i] = prop_max[prop]
        guess[i] = min(guess[i], ub[i])        
   
  
    # set required accuracy for solution
    accuracy =kwargs.get('accuracy', 1)


  
    for row in delfstar_mod.itertuples(): 
        def ftosolve(x):
            layers_solve = update_layers(props_calc, x, layers, **kwargs)
            return compare_calc_expt(layers_solve, row, calc,
                                     calctype=calctype, reftype=reftype)
        
        try:
            soln = optimize.least_squares(ftosolve, guess, bounds=(lb, ub))
        except:
            print(f'error at index {row.Index}')
            continue
            
        # make sure sufficienty accurate solutions was found
        if soln['fun'].max() > accuracy:
            df_soln.drop(row.Index, inplace=True)
            continue
        
        layers = update_layers(props_calc, soln['x'], layers)
        guess = guess_from_layers(props_calc, layers)
        df_soln = update_df_soln(df_soln, soln, row.Index, layers, props_calc,
                                 reftype)
        
        #display the calculated values as the program is running
        if showvals:  
            valstring = ""
            for i, prop in enumerate(props_calc):
                valstring.append(f"{prop}: {soln['x']:.2g}")
            print(valstring)
      
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
        column = 'dlam'+str(n)
    else:
        column = 'dlam3'
    # the following format comes from
    # https://stackoverflow.com/questions/49781626/pandas-query-with-variable-as-column-name
    soln = soln.query('{0}>@dlam'.format(column))
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
    figdic : dictionary
        dictionary returned by make_prop_axes

    """

    # function to solve for all .xlsx files in a directory

    df = {}
    soln = {}
    figdic = {}
    
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
        figdic[prefix] = make_prop_axes(**kwargs)
        plot_props(soln[prefix], figdic[prefix])
        
        # now set the window title for the solution check
        kwargs['num']=os.path.join(datadir, prefix+'_'+calc+'_check.pdf')
        figdic[prefix]['fig'].savefig(os.path.join(datadir, prefix+
                                                    '_'+calc+'_props.pdf'))
    return df, soln, figdic


def calc_phi(grho3, phi_parms):
    '''
    Esitmates phase angle from grho3, assuming linear relationshp between phi
    and log(grho3).  Based on Sadman, et al. Macromol. 50, 9417–9426 (2017).
    This assumes f1 = the default value of 5 MHz for now

    Parameters
    ----------
    grho3 : float
        Value of grho3 for calculation.
    phi_parms : list containing etarho, grho
        etarho:  liquid limit - viscosity times density for pure solvent
        grho: solid limit - modulus times density for pure polymer
        

    Returns
    -------
    phase angle (in degrees).

    '''
    f1 = 5e6
    eta = phi_parms[0]
    grho3_lo = 2*np.pi*3*f1*eta
    grho3_hi = phi_parms[1]
    
    # grho3 = grho3_in[(grho3_in>=grho3_lo) & (grho3_in<=grho3_hi)]

    phi =90-90*((np.log(grho3)-np.log(grho3_lo))/
                (np.log(grho3_hi)-np.log(grho3_lo)))
    return phi
        


def make_err_axes(**kwargs):
    """
    Parameters
    ----------
    kwargs: (optional arguments)
        num (string):
            title for plot window
            
    Returns
    ----------
        fig:
            Figure containing various error plots.
        ax:
            axes of the figure.
    """
    num = kwargs.get('num','error plot')
    fig, ax = plt.subplots(3,3, figsize=(9,9), constrained_layout=True,
                           num = num)
    return fig, ax

def err_fn_correlated_df(df_soln_in, fn_err):
    """
    Function to calculate property error if all values of df/n change by fn_err
    Parameters
    ----------
    df_soln_in : Dataframe
        Solution Dataframe
    fn_err : float
        error in df/n (same for each harmonic).

    Returns
    -------
    df_soln_out : Dataframe
        soln dataframe with property errors added

    """
    df_soln_out = deepcopy(df_soln_in)
    npts = len(df_soln_out.index)
 
    float_series = np.zeros(npts, dtype = np.float64)
    for prop in df_soln_out.iloc[0]['props_calc']:
        if f'{prop}_err_fn' not in df_soln_out.keys():
            df_soln_out.insert(df_soln_out.shape[1], 
                           f'{prop}_err_fn', float_series)
        else:
            df_soln_out.loc[:,f'{prop}_err_fn'] = float_series
    
    # now add columns for properties in all layers
    for idx, row in df_soln_in.iterrows(): 
        guess = guess_from_layers(row.props_calc, row.layers)
        nf, ng, n_all, n_unique = nvals_from_calc(row.calc)
        for n in nf:
            row[f'delfstar_expt_{n}'] = row[f'delfstar_expt_{n}'] +n*fn_err
        def ftosolve(x):
            layers_solve = update_layers(row.props_calc, x, row.layers)
            return compare_calc_expt(layers_solve, row, row.calc,
                                     calctype=row.calctype, 
                                     reftype=row.reftype)
        try:
            soln = optimize.least_squares(ftosolve, guess)
        except:
            print(f'error at index {row.Index}')
            continue
        
        for i, prop in enumerate(row.props_calc):
            df_soln_out.loc[idx, f'{prop}_err_fn'] = abs(soln['x'][i] - 
                        df_soln_out.loc[idx, f'{prop}'])
    
    return df_soln_out


def err_fn_correlated_row(row_in, fn_err):
    """
    Function to calculate property error if all values of df/n change by 
    fn_err, operating only on a single row of a dataframe generated by iterrows
    Parameters
    ----------
    row : Dataframe row
        Solution Dataframe
    fn_err : float
        error in df/n (same for each harmonic).

    Returns
    -------
    row_out : Dictionary
        input row with property errors added

    """
    row = deepcopy(row_in)
    # Return the original row if 'props_calc' does not exist
    if ('props_calc' not in row or row['props_calc'] is None or (not 
        isinstance(row['props_calc'], list)) or len(row['props_calc']) == 0):
        return row
    
    # handle trivial case where fn_err = 0
    if fn_err == 0:
        for i, prop in enumerate(row.props_calc):
            row[f'{prop}_err_fn'] = 0
        return row
        
 
    guess = guess_from_layers(row.props_calc, row.layers)
    nf, ng, n_all, n_unique = nvals_from_calc(row.calc)
    for n in nf:
        row_in[f'delfstar_expt_{n}'] = row_in[f'delfstar_expt_{n}'] +n*fn_err
    
    def ftosolve(x):
        layers_solve = update_layers(row_in.props_calc, x, row.layers)
        return compare_calc_expt(layers_solve, row_in, row.calc,
                                 calctype=row.calctype, 
                                 reftype=row.reftype)
    try:
        soln = optimize.least_squares(ftosolve, guess)
    except:
        print(f'error at index {row.Index} during fn_err calc)')
        return
    
    for i, prop in enumerate(row.props_calc):
        row[f'{prop}_err_fn'] = abs(soln['x'][i] - 
                    row[f'{prop}'])
    
    return row



def calc_fstar_err (n, row, f_error):
    """
    Calculate uncertainties in delf and delg (expressed as complex delfstar)
    from f_error.
    
    Parameters
    ----------
    n : Integer
        Harmonic of Interest.

    row : Dataframe row
        Dictionary of input values, typically taken from a row of the      
        solution dataframe generated by solve_for_props.

    f_error (list of 3 numbers):
        -uncertainty in f,g as a fraction of g
        
        -uncertainty in f/n (applied individually to harmonics)
        
        -uncertaintiy in correlated f/n (not used here)
             
        default is [0.05, 15, 0], set to [0,0,0] to elminate error bars
            
    Returns
    -------
    Dictionary with two calculated versions of error in measurd delfstar:
        'p':  uncorrelated error, determined from f_error[0,1]
        't':  total error, determined from f_error[0,1,2]

    """
        
    # find the value of gamma for the harmonic of interest
    fstar = getattr(row, f'fstar_expt_{n}')
    gamma = np.imag(fstar)
    
    # now we calculate f_err and g_err
    f_err_p = round((f_error[0]*gamma + n*f_error[1]), 1)
    f_err_t = f_err_p + n*f_error[2]

    g_err = round(f_error[0]*gamma, 1)
    
    fstar_err = {'p': f_err_p + 1j*g_err,
                 't': f_err_t + 1j*g_err}
    return fstar_err
           

def calc_prop_error(soln, f_error):
    '''
    Calclate error in properties

    Parameters
    ----------
    soln : Dataframe 
        Data being considered (from solve_for_props).
        
    f_error (list of 3 numbers):
        -uncertainty in f,g as a fraction of g
        
        -uncertainty in f/n (applied individually to harmonics)
        
        -uncertainty in correlated f/n (same error applied to all delf/n)
             
        default is [0.05, 15, 0]

    Returns
    -------
    dataframe with errors in properties within props_calc

    '''
    # make the error dataframe
    # Input solution doesn't necessarily have to have the same values of 
    # props_calc or calc for every line
    prop_err=pd.DataFrame(index=soln.index)
    npts = len(soln)
    real_series = np.zeros(npts, dtype=np.float64)
    
    # handle the case where we're using prop_plots to plot other data
    if ('props_calc' not in soln.keys() or soln['props_calc'].isna().all()):
        return pd.DataFrame(None)
    
    for prop in soln[soln['props_calc'].notna()].iloc[0]['props_calc']:
        err_namep = f'{prop}_err_p' # partial error from f_error[0,1]
        err_namet = f'{prop}_err_t' # total error form f_eroor[0,1,2]
        for err_name in [err_namep, err_namet]:
            if err_name not in prop_err.columns.values:
                prop_err.insert(prop_err.shape[1], err_name, real_series)
           
    for idx, row in soln.iterrows():
        # this handles case where soln dataframe was not generated
        # by solve_for_props
        if ('props_calc' not in row or row['props_calc'] is None or (not 
              isinstance(row['props_calc'], list)) or 
              len(row['props_calc']) == 0):
            continue
        # make dictionary where we'll keep the different contributions to error

        props_calc = row.props_calc
        calc = row.calc
        nf, ng, n_all, n_unique = nvals_from_calc(calc)

        # extract uncertainty from dataframe
        uncertainty_p = []
        for n in nf:
            fstar_err = calc_fstar_err(n, row, f_error)
            uncertainty_p.append(np.real(fstar_err['p']))
        for n in ng:
            fstar_err = calc_fstar_err(n, row, f_error)
            uncertainty_p.append(np.imag(fstar_err['p']))

        # extract the jacobian and turn it back into a numpy array of floats
        try:
            jacobian = np.array(row.jac, dtype='float')
        except:
            jacobian = np.zeros([len(uncertainty_p), len(uncertainty_p)])
        try:
            deriv = np.linalg.inv(jacobian)
        except:
            deriv = np.zeros([len(uncertainty_p), len(uncertainty_p)])
    
        # determine error from Jacobian
        # this only works if the number of elements in calc is the 
        # same as the number of elements in props_calc (n equations and
        # n unknowns)

        # include errors from correlated changes in df/n
        row_new = deepcopy(err_fn_correlated_row(row, f_error[2]))
        n = len(props_calc)
        if n != n_all:
            print(f'{n_all} elements in calc ({calc}) but {n} props' +
                  ' {props_calc} Cannot calculate error')
            return prop_err
        for p, prop in enumerate(props_calc):
            err2 = (row_new[f'{prop}_err_fn'])**2
            errp = 0
            for k in np.arange(n):
                try:
                    errp = errp + (deriv[p, k]*uncertainty_p[k])**2
                except:
                    print(f'error with {prop} in calc_prop_error')
            # errors can't exceed actual values of the properties
            prop_err.loc[idx, f'{prop}_err_p'] = (min(np.sqrt(errp), 
                                                soln.loc[idx, f'{prop}']))
            prop_err.loc[idx, f'{prop}_err_t'] = (min(np.sqrt(errp+err2),
                                                 soln.loc[idx, f'{prop}']))

    return prop_err

        
def make_prop_axes(propnames, **kwargs):
    '''
    Make a blank property figure.

    Parameters
    ----------
                    
    propnames : list of strings
        each in the format property_layer.ext. Here 
        layer is the layer number of the property, and ext is an optional
        argument that can include the following:
        
        
        'log': 
            Force plot on log scale.
            
        'add_m/a' (drho only):
            Add g/m^2 or mg/m^2 to y axis label.
            
        'nm' (drho only)
            Switch thickness unit to nm instead of default micron.
           
    Here is a list of all possible vlues for props:
    
        'drho': 
            Mass per area, generally microns*g/cm^3.
        
        'grho3':
            Density x magnitude of complex modulus at n=3.
        
        'phi':
            Phase angle (degrees), assumed constant at all harmonics used.
            
        'phi.tan':
            Loss tangent
            
        'vgp':
            Van Gurp-Palmen plot (phi vs. grho3).
        
        'jdp':
            Loss compliance normalized by density.
        
       
        'grho3p':
            Storage modulus at n=3 times density.
        
        'grho3pp':
            Loss modulus at n = 3 time density.
        
        'etarho3':
            Complex viscosity (units of mPa-s-g/cm3).
        
        'temp':
            Temperature in degrees C.
        
        's', 'hr', 'day':
            Time in correspoinding unit.
            
        dfn or dgn:
            frequency or dissipation shift for a given harmonic
        
        Name of any dataframe column:
            In this case you'll need to label the y axis manually.

    **kwargs:
        
        num (string):           
            window title (string): (default is 'property fig')
            
        checks (boolean):
            True (default) if we plot the solution checks
             
        maps (booleaan):
            True (default is false) if we make the response maps
            
        orientation ('string')
            'horizontal' is default, swith to 'vertical' for vertical
            plot graph (only valid if checks and maps are both False)
        
        contour_range (dictionary):
            range of contours in form {0:[min,max], 1:[min, max])} \
            default is {0:[-3, 3], 1:[0,3]})
                    
        contour_range_units: 
            'Hz' (default) or Sauerbrey if we want to normalize
            
        xunit (single string or list of strings):
            Units for x data.  Default is 'index', function currently handles
            - 's', 'min', 'hr', 'day', 'temp', or user specified value corresponding
                to any property or dataframe column
            - add .log to end of string to plot on log scale
                
        xlabel (string):
            label for x axis.  Only used if user-specified xunit is used
            String if same for all axes, otherwise list.
                         
        plotsize (tuple of 2 real numbers):
            size of individual plots.  
            - Default is (4, 3)
            
        sharex (list of integers):
            axes for potential sharing of x axis
            -(default is np.arange(nplots), where nplots is number of prop plots)
                   
        no3 (Boolean):
            False (default) if we want to keep the '3' 
            subscript in axis label for G
        
        gammascale (string):
            'log to plot dissipation on log scale'
            
        title_strings (list of 2 strings):
            characters to go before and after letters for parts of figures - default
            is ['(',')']
        
        title_fontweight (string):
            weight of axes titles - default is 'normal'
            
        title_loc (string):
            location of title - default is 'center' can also be 'left', 'right'
            


    Returns:
    ----------
    figdic: dictionary with the following elements:
        fig:
            Dictionary of main figure 'master' along with included
            subfigures.  It always includes the 'props' subfigure,
            and will also include the 'checks' and 'maps' subfigures
            if these were generated.

        ax:
            Dictionary of axes in the figure. Always has 'props',
            may have 'checks'  and 'maps'.  Numbers correspond to 
            flattened indices of all axes in the overall figure.
        info:
            Dictionary with info used for plotting.
    '''

    num = kwargs.get('num', 'property fig')
    maps = kwargs.get('maps', False)
    checks = kwargs.get('checks', True)
    orientation = kwargs.get('orientation', 'horizontal')
    xunit_input = kwargs.get('xunit', 'index')
    no3 = kwargs.get('no3', False)
    # add '_1' to property if it wasn't included explicitly
    propnames = add_layer_nums(propnames)  
    nprops = len(propnames)
    
    # make props dictionary that connects property axis number to the property
    props = {}
    for k in np.arange(nprops):
        props[k] = propnames[k]
    
    # vertical orientation gets rid of unwanted whitespace if we only have
    # one suplot
    if nprops==1 and not maps and not checks:
        orientation = 'vertical'
    sharex = kwargs.get('sharex', np.arange(nprops))
    plotsize = kwargs.get('plotsize', (4,3))
    
    title_strings = kwargs.get('title_strings', ['(', ')'])
    title_fontweight = kwargs.get('title_fontweight', 'normal')
    title_loc = kwargs.get('title_loc', 'center')
    
    # change labels in case we don't want the 3 subscript for G
    if no3:
        axlabels['grho3'] = r'$|G^*|\rho$ (Pa $\cdot$ g/cm$^3$)'
        axlabels['grho3p'] = r'$G^\prime\rho$ (Pa $\cdot$ g/cm$^3$)'
        axlabels['grho3pp'] = r'$G^{\prime\prime}\rho$ (Pa $\cdot$ g/cm$^3$)'
        
    if nprops == 1 and not checks and not maps:
        titles = ['']
        title_strings=['','']
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
        if nprops==0:
            xunit[0]=xunit_input
        for p in np.arange(0,nprops):
            xunit[p] = xunit_input
    else:
        for p in np.arange(nprops):
            xunit[p]=xunit_input[p]
            
            
    # set the x labels
    for p in np.arange(max(1, nprops)):  # account for fact that nprops might be 0
        xunit_base = xunit[p].split('.')[0].split('_')[0]
        if xunit_base in axlabels.keys():
            xlabel[p] = axlabels[xunit_base]
        else:
            # read the xlabel
            xlabel_input = kwargs.get('xlabel', 'xlabel')
            if type(xlabel_input)==str:
                xlabel[p] = xlabel_input
            else :
                xlabel[p] = xlabel_input[p]
        
            
    # make the main figure
    # fig includes 'master' (the full fig), plus subfigures of 'props',
    # 'checks' and 'maps'
    # listing of all axes within the master figure (ax[0], ax[1], etc.)
    fig = {}; ax = {}
    fig['master'] = plt.figure(constrained_layout = True, num = num)
    
    # build the GridSpec
    nrows = 0
    irow=-1
    if len(propnames)!=0:
        nrows = 1
    if checks:
        nrows = nrows+1
    if maps:
        nrows = nrows+1
    
    
    # create the gridspec and add property subfigure
    # start with vertial version
    if (not checks and not maps and orientation=='vertical'):
        GridSpec = gridspec.GridSpec(ncols=1, nrows=1, 
                                     figure= fig['master'])
        fig['props'] = fig['master'].add_subfigure(GridSpec[0,0])
    # now handle the standard horizontal verions
    else:
        GridSpec = gridspec.GridSpec(ncols=12, nrows=nrows, 
                                     figure= fig['master'])
        if nprops==1:
            fig['props'] = fig['master'].add_subfigure(GridSpec[0,3:9])
        else:
            fig['props'] = fig['master'].add_subfigure(GridSpec[0,:])


    # add the different axes to plots figure
    if nprops!=0:
        irow=irow+1
        ax['props']={}
        
        if orientation == 'vertical':
            ax['props'][0] = fig['props'].add_subplot(nprops, 1, 1)
        
            for p in np.arange(1, nprops):
                if xunit[p] == xunit[0] and p in sharex:
                    ax['props'][p] = fig['props'].add_subplot(nprops, 1, p+1, 
                                                        sharex = ax['props'][0])
                else:
                    ax['props'][p] = fig['props'].add_subplot(nprops, 1, p+1)
        else:
            ax['props'][0] = fig['props'].add_subplot(1, nprops, 1)
        
            for p in np.arange(1, nprops):
                if xunit[p] == xunit[0] and p in sharex:
                    ax['props'][p] = fig['props'].add_subplot(1, nprops, p+1, 
                                                        sharex = ax['props'][0])
                else:
                    ax['props'][p] = fig['props'].add_subplot(1, nprops, p+1)
                
        for p in np.arange(nprops):
            ax[p] = ax['props'][p]
            title = f'{title_strings[0]}{titles[p]}{title_strings[1]}'
            ax[p].set_title(title, fontweight=title_fontweight,
                            loc=title_loc)
            ax[p].set_xlabel(xlabel[p])

    iax = nprops-1  # running number of axes for axis labeling

    if checks:
        # Subfigure - solution checks
        irow = irow+1
        if nprops == 3:
            fig['checks'] = fig['master'].add_subfigure(GridSpec[irow,1:11])
        else:
            fig['checks'] = fig['master'].add_subfigure(GridSpec[irow,:])
        if nprops!=0:    
            ax['checks'] = {0:fig['checks'].add_subplot(1,2,1, 
                                                        sharex=ax['props'][0]),
                            1:fig['checks'].add_subplot(1,2,2, 
                                                        sharex=ax['props'][0])}
        else:
            ax['checks'] = {0:fig['checks'].add_subplot(1,2,1),
                            1:fig['checks'].add_subplot(1,2,2)}
        ax[iax+1]=ax['checks'][0]
        ax[iax+2]=ax['checks'][1]
        for k in [0, 1]:
            title = f'{title_strings[0]}{titles[iax +1+k]}{title_strings[1]}'
            ax['checks'][k].set_title(title, fontweight=title_fontweight,
                            loc=title_loc)
            # used xlabel for first prop plot as the xlabel for checks
            ax['checks'][k].set_xlabel(xlabel[0])
        iax = iax + 2
        ax['checks'][0].set_ylabel(axlabels['delf_n'])
        ax['checks'][1].set_ylabel(axlabels['delg_n'])
        
        gammascale = kwargs.get('gammascale', 'linear')
        ax['checks'][1].set_yscale(gammascale)
        
            
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
        kwargs['first_plot'] = iax+1
        make_response_maps(fig['maps'],ax['maps'], **kwargs)
        # set variable we'll use to make sure we don't duplicate labels
        for n in [1,3,5,7,9]:
            maplabels[n]=True
  
 
    # set the figure size
    # account for 2 extra rows for solution checks and response maps
    if orientation == 'vertical':
        figsize = (plotsize[0], plotsize[1]*nprops)
        
    else: 
        cols = max(2, nprops)
        figsize = (plotsize[0]*cols, plotsize[1]*nrows)
    
    fig['master'].set_size_inches(figsize)
    
  
    
    # now set the y labels 
    for p in np.arange(nprops):   
        # strip out layer number to make the connection to axlabels dictionary
        prop = props[p].split('.')[0].split('_')[0]
        # get other options that we might need
        ext = props[p].split('.')
        
        # start with the special case of tanphi
        if props[p] == 'phi.tan':
            ax['props'][p].set_ylabel(r'tan$\phi$')
            ax['props'][p].set_xlabel(xlabel[p])       
        
        # we also handle 'drho' separately to allow for case where thickness 
        # is in nm
        elif 'drho' in prop:
            ax['props'][p].set_ylabel(drho_label(ext))
            ax['props'][p].set_xlabel(xlabel[p])  
        
        # we get most of the axes labels from axlabels dictionary
        elif prop in axlabels.keys():
            ax['props'][p].set_ylabel(axlabels[prop])
            ax['props'][p].set_xlabel(xlabel[p])      
            
        elif 'df' in prop:
            n = prop[-1]
            # handle the possibility that we want to plot negative of df
            if '-df' in props[p]:
                ax['props'][p].set_ylabel(f'-\u0394 f_{{{n}}} (Hz)')
            else:
                ax['props'][p].set_ylabel(f'\u0394 f_{{{n}}}$ (Hz)')
            ax[p].set_xlabel(xlabel[p])
            
        elif 'dg' in prop:
            n = prop[-1]
            ax['props'][p].set_ylabel(f'\u0394 \u0393_{{{n}}} (Hz)')
            ax['props'][p].set_xlabel(xlabel[p])
            
           
        else:
            ax['props'][p].set_ylabel('ylabel')


    info = {'props':props, 'xunit':xunit,
            'maplabels':maplabels, 'checklabels':checklabels}
    

    return {'fig':fig, 'ax':ax, 'info':info}


def make_data_array(var, soln, prop_error, **kwargs):
    """
    Extract appropriate data vector from solution dataframe

    Parameters
    ----------
    var : string
        The variable to be extracted.
    soln : dataframe
        The dataframe the data is extracted from.
    prop_error : dataframe
        Property errors calcualted from f_error

    Returns
    -------
    data_array : numpy array of data
    err_array_p : numpy array of errors based on f_error[1,2]
    err_array_t : numpy array of errors based on f_error[1,2,3]

    """
    f1 = kwargs.get('f1', f1_default)
    ext = var.split('.')
    
    # set errors to zero by default
    err_array_p = np.zeros_like(soln.index)
    err_array_t = np.zeros_like(soln.index)
    
    # get layer number
    if len(ext[0].split('_'))==2:
        layer = ext[0].split('_')[1]
    else:
        layer = 1

    if ext[0] == 's':
        data_array = soln['t']
        
    elif ext[0] == 'min':
        data_array =soln['t']/60
        
    elif ext[0] == 'hr':
        data_array = soln['t']/3600
        
    elif ext[0] == 'day':
        data_array = soln['t']/(24*3600)
        
    elif ext[0] == 'temp':
        data_array = soln['temp']
        
    elif ext[0] == 'index':
        data_array = soln.index
        
    elif 'grho3' in ext[0]:
        # units are g/m^2 for grho3
        prop_name = f'grho3_{layer}'
        prop_err_name = prop_name+'_err'
        data_array = soln[prop_name].astype(float)/1000
        if f'{prop_err_name}_p' in prop_error.keys():
            err_array_p = prop_error[f'{prop_err_name}_p']/1000
            err_array_t = prop_error[f'{prop_err_name}_t']/1000

    elif 'etarho3' in ext[0]:
        # units are mPa-s for viscosity
        prop_name = f'grho3_{layer}'
        prop_err_name = prop_name+'_err'
        data_array = soln[prop_name].astype(float)/(2*np.pi*3*f1)
        if f'{prop_err_name}_p' in prop_error.keys():
            err_array_p = prop_error[f'grho3_{layer}_err_p']/(2*np.pi*3*f1)
            err_array_t = prop_error[f'grho3_{layer}_err_t']/(2*np.pi*3*f1)

    elif 'phi' in ext[0]:
        prop_name = f'phi_{layer}'
        prop_err_name = prop_name+'_err'
        phi_d = soln[prop_name].astype(float)
        phi_r = phi_d*np.pi/180
        if 'tan' in ext:
            data_array = np.tan(phi_r)
        else:
            data_array = phi_d
        if f'{prop_err_name}_p' in prop_error.keys():
            err_d_p = prop_error[f'{prop_err_name}_p']
            err_d_t = prop_error[f'{prop_err_name}_t']
            if 'tan' in ext:
                err_r_p = err_d_p*np.pi/180
                err_r_t = err_d_t*np.pi/180
                err_array_p = np.tan(phi_r+err_r_p)-np.tan(phi_r)
                err_array_t = np.tan(phi_r+err_r_t)-np.tan(phi_r)
            else:
                err_array_p = err_d_p
                err_array_t = err_d_t
       
    elif 'grho3p' in ext[0]:
       data_array = (soln[f'grho3_{layer}'].astype(float)*
                np.cos(np.pi*soln['phi'].astype(float)/180)/1000)

    elif 'grho3pp' in ext[0]:
       data_array = (soln[f'grho3_{layer}'].astype(float)*
                np.sin(np.pi*soln['phi'].astype(float)/180)/1000)
       
    elif 'deltarho3' in ext[0]:
       grho3 = soln[f'grho3_{layer}'].astype(float)
       phi = soln[f'phi_{layer}'].astype(float)
       data_array = 1000*calc_deltarho(3, grho3, phi)
                               
    elif 'drho' in ext[0]:
        prop_name = f'drho_{layer}'
        prop_err_name = prop_name+'_err'
        data_array = 1000*soln[f'drho_{layer}'].astype(float)
        
        # multiply by 1000 if units are nm instead of microns
        if 'nm' in ext:
            data_array = 1000*data_array
        if f'{prop_err_name}_p' in prop_error.keys():
            err_array_p = 1000*prop_error[f'{prop_err_name}_p']
            err_array_t = 1000*prop_error[f'{prop_err_name}_t']

    elif 'jdp' in ext[0]:
        data_array = ((1000/soln['grho3_1'].astype(float))*
                 np.sin(soln['phi'].astype(float)*np.pi/180))
      
    elif 'soln_expt' in ext[0]:
        data_array = soln[ext[0]].astype(complex)
        data_array = np.real(data_array)
       
    elif 'g_expt' in ext[0]:
        # this handles g_exptn and dg_expt_n for different n
        key = ext[0]
        data_array = np.real(soln[key].astype(complex))
       
    elif 'f_expt' in ext[0]:
        # this handles f_exptn and df_expn for different n
        key = ext[0]
        data_array = np.real(soln[key].astype(complex))
          
    elif ext[0] in soln.keys():
        data_array = soln[ext[0]]
   
    else:
        print(f'no data - not a recognized prop type ({ext[0]})')
        data_array=np.array([])
        data_array=np.array([])
        
    return data_array.astype('float64'), err_array_p,  err_array_t
   

def plot_props(soln, figdic, **kwargs):
    """
    Add property data to an existing figure.

    Parameters
    ----------
    soln " dataframe):
        Dataframe containing data to be plotted, typically output from
        solve_for_props.
    figdic : dictionary
        Dictionary containing 'fig', 'ax' and other info for plot.

    kwargs
    ------

    props (dictionary)
        Default taken as figdic['info']['plots'].  Keys correspond to the axes, 
        used for plotting, e.g., {1:'grho3_1.log', 2:'phi_1'}.
              
        
    xoffset (real or string, single value or list):
        Amount to subtract from x value for plotting (default is 0)
        'zero' means that the data are offset so that the minimum val 
        is at 0.
    xmult (real):
        Multiplicative factor for rescaling x data.
    fmt (string):
        Format sting for plotting.  Default is '+'.  Can include color designation
    prop_color (string)
        Valid color designation for property plots, not used if color included
        in fmt
    n_color (dictionary)
        Color designation for different harmonics in solution checks,  keys
        are harmonics and values are color designations
    
    label (string):
        label for plots.  Used to generate legend.  Default is 
        '', which will not generate a label.
    f_error (list of 3 numbers):
        -uncertainty in f,g as a fraction of g
        
        -uncertainty in f/n (applied individually to harmonics)
        
        -uncertaintiy in correlated f/n (applied to all harmonics)]
             
        default is [0.05, 15, 0], set to [0,0,0] to elminate error bars
    nplot (list of integers):
        harmonics to plot, default is [3,5])
    plot_df1 
        True if we want to plot df1
    drho_ref (real):
        reference drho for plots of drho_norm or drho_ref       
    linewidth (float):
            linewidth for plots


    Returns
    -------
    fig : figure handle-
        master figure
    ax : list of axes handles
        all axes listed numerically)
    """


    def extract_color_from_format(fmt):
        color_chars = {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}
        for char in fmt:
            if char in color_chars:
                return char
        return None

    if len(soln) ==0:
        print('solution data frame for plotting is empty')
        return
    fmt=kwargs.get('fmt', 'x')
           
    label_input=kwargs.get('label', '')
    xoffset_input=kwargs.get('xoffset', 0)  
    f_error = kwargs.get('f_error', [0.05, 15, 0])
    linewidth = kwargs.get('linewidth', 1)
    
    nplot = kwargs.get('nplot', [3,5])
    n_color = kwargs.get('n_color', {1:'C0', 3:'C1', 5:'C2', 7:'C3', 9:'C5'})
    
    # drop dataframe rows with all nan
    soln = soln.dropna(how='all') 
    
    # sort out which axes to use and whcih props to plot on these axes
    props = kwargs.get('props', figdic['info']['props'])
    

    ax = figdic['ax']
    fig = figdic['fig']
        
    # create dataframe with calculated errors
    prop_error = calc_prop_error(soln, f_error)
    xunit = figdic['info']['xunit']
    xoffset={}
    # set the offset for the x values 
    if type(xoffset_input) != list:
        for p in props.keys():
            xoffset[p] = xoffset_input
    else:
        for p in props.keys():
            xoffset[p]=xoffset_input[p]  
            
    # determine the plots we actually need to make. Sometimes we don't add
    # data to an existing axis
    # we need to keep xdata for each plot, because we'll use just
    # one of these arrays (usually the first one) for the solution checks
    xdata = {}
    
    for p in props.keys():
        # establish property color
        if 'C' in fmt:
            idx=fmt.find('C')
            prop_color = f'C{fmt[idx+1]}'
            fmt = fmt.replace(f'C{fmt[idx+1]}', '')
            
        elif extract_color_from_format(fmt)!=None:
            # handle case where we already have the olor in the fmt string
            prop_color = extract_color_from_format(fmt)
            fmt = fmt.replace(prop_color, '')
            
        elif 'prop_color' not in kwargs.keys():
            # use the next color in the normal color sequence
            prop_color = None
            
        else:
            prop_color = kwargs['prop_color']
            
        xdata[p] = make_data_array(xunit[p], soln, prop_error)[0]
        ydata, yerr_p, yerr_t  = make_data_array(props[p], soln, prop_error)
        
        # offset xdata if desired
        if xoffset[p] == 'zero':
            xoffset[p] = min(xdata[p])
        xdata[p] = xdata[p] - xoffset[p]
        
        # make sure y limits are okay (needed for log axes)
        y_min = []
        y_max = []
        if ax['props'][p].lines:
            y_min = [ax['props'][p].get_ylim()[0]]
            y_max = [ax['props'][p].get_ylim()[1]]
            
        y_min.append(min(ydata-yerr_t))
        y_max.append(max(ydata+yerr_t))
        y_min = min(y_min)
        y_max = max(y_max)
        label = label_input
        
        # create layer labels if more than 1 laer are used
        if len(soln.iloc[0]['layers'].keys())>1:
            layer_num = props[p].split('_')[1].split('.')[0]
            label = label_input + f' layer {layer_num}'
        if not np.any(yerr_t): 
            ax['props'][p].plot(xdata[p], ydata, fmt, label=label,
                                color = prop_color,
                                linewidth=linewidth)
            
        else:
            # plot with error bars
            ax['props'][p].errorbar(xdata[p], ydata, fmt=fmt, 
                            yerr=yerr_p, label=label, capsize = 3,
                            linewidth=linewidth,color = prop_color)
            
            # now plot extended error bars, including correlated error in f/n
            ax['props'][p].errorbar(xdata[p], ydata, yerr=yerr_t, 
                                   markersize = 0, color = prop_color,
                                   linewidth = linewidth, fmt = fmt)     
            
        y_range = y_max - y_min
        
        if 'log' in props[p].split('.'):
            ax['props'][p].set_yscale('log')
            
        if 'log' in xunit[p].split('.'):
            ax['props'][p].set_xscale('log')
            
        if ax['props'][p].get_yscale()=='log' and y_min>0:
            # set axis limits so we have 5% white space at top and bottom
            logrange = np.log(y_max/y_min)
            padding = np.exp(0.05*logrange)
            ax['props'][p].set_ylim([y_min/padding, y_max*padding])
        else:
            if y_range>0:
                ax['props'][p].set_ylim([y_min - 0.05*y_range, 
                            y_max +0.05*y_range])     
        
        # rest max phi to cut off meaningless phase angles 
        # this probably is okay for phi.tan as well
        if 'phi' in props[p] and ax['props'][p].get_ylim()[1]>92:
            ax['props'][p].set_ylim(top = 92)
        if 'phi' in props[p] and ax['props'][p].get_ylim()[0]<0:
            ax['props'][p].set_ylim(bottom = 0)
            
        ax['props'][p].legend()
            
    # now add the comparison plots of measured and calcuated values         
    # plot the experimental data first
    # keep track of max and min values for plotting purposes       
    if 'checks' in fig.keys(): 
        # decide to use df1 or not
        plot_df1 = kwargs.get('plot_df1', True)
        
        # adjust nplot if any of the values don't exist in the dataframe
        for n in nplot:
            if not f'delfstar_expt_{n}' in soln.keys():
                nplot.remove(n)
                   
        df_min = []
        df_max = []
        dg_min = []
        dg_max = []
        x_min = []
        x_max = []
        
        # start with previous limits if data were already plotted
        if ax['checks'][0].lines:
            df_min = [ax['checks'][0].get_ylim()[0]]
            df_max = [ax['checks'][0].get_ylim()[1]]
            dg_min = [ax['checks'][1].get_ylim()[0]]
            dg_max = [ax['checks'][1].get_ylim()[1]]
            x_min = [ax['checks'][1].get_xlim()[0]]
            x_max = [ax['checks'][1].get_xlim()[1]]   
            
        # now plot the calculated values 
        nfplot = []
        ngplot = []
        for n in nplot:
            if f'delfstar_expt_{n}' in soln.keys():
                nfplot.append(n)
                ngplot.append(n)

        if not plot_df1:
            try:
                nfplot.remove(1)
            except ValueError:
                pass

        # add uncertainties in delf, delg
        soln = add_fstar_err(soln, f_error)
        
        # use the first plotted property plot for the x values
        if len(props.keys())==0:
            # make sure we have xdata for nprops=0 case 
            soln['xdata'] = make_data_array(xunit[0], soln, prop_error)[0]
        else:
            soln['xdata'] = xdata[min(list(props.keys()))]    
        for n in nfplot: 
            # drop nan values from dataframe to avoid problems with errorbar
            soln_tmp = soln.dropna(subset=[f'delfstar_expt_{n}'])
            dfval = np.real(soln_tmp[f'delfstar_expt_{n}'])/n
            dfval2 = np.real(soln_tmp[f'delfstar_calc_{n}'])/n
            ferr_p = np.real(soln_tmp[f'fstar_err_p_{n}'])/n
            ferr_t = np.real(soln_tmp[f'fstar_err_t_{n}'])/n
            df_min.append(np.nanmin(dfval-ferr_t))
            df_min.append(np.nanmin(dfval2))
            df_max.append(np.nanmax(dfval+ferr_t))
            df_max.append(np.nanmax(dfval2))
            if figdic['info']['checklabels'][n]:
                label_expt = f'n={n}: expt'
                label_calc = f'n={n}: calc'
            else:
                label_expt = ''
                label_calc = ''
            # we have no calculated curves if there are no properties
            if len(props.keys())==0:
                label_calc = ''
                
            # mow make the plot with both 'partial' and 'total' error bars
            ax['checks'][0].errorbar(soln_tmp['xdata'], 
                       dfval, yerr = ferr_p, fmt='x', color = n_color[n],       
                       label=label_expt, capsize=3)
            ax['checks'][0].errorbar(soln_tmp['xdata'], 
                       dfval, yerr = ferr_t, fmt='x', color = n_color[n],       
                       linewidth = 0.5, markersize = 0)
            calcvals = np.real(soln_tmp[f'delfstar_calc_{n}'])/n
            ax['checks'][0].plot(soln_tmp['xdata'], calcvals, '-', 
                    color = n_color[n], markerfacecolor='none', 
                    label=label_calc)
            
            # don't include multiple harmonic labels
            figdic['info']['checklabels'][n]=False
        df_min = min(df_min)
        df_max = max(df_max)
     
        for n in ngplot:
            # drop nan values from dataframe to avoid problems with errorbar
            soln_tmp = soln.dropna(subset=[f'delfstar_expt_{n}'])
            dgval = np.imag(soln_tmp[f'delfstar_expt_{n}'])/n
            dgval2 = np.imag(soln_tmp[f'delfstar_calc_{n}'])/n
            gerr = np.imag(soln_tmp[f'fstar_err_t_{n}'])/n
            dg_min.append(np.nanmin(dgval-gerr))
            dg_min.append(np.nanmin(dgval2))
            dg_max.append(np.nanmax(dgval+gerr))
            dg_max.append(np.nanmax(dgval2))
            ax['checks'][1].errorbar(soln_tmp['xdata'], dgval, yerr = gerr, 
                                     fmt = 'x',  
                                     color = n_color[n],
                                     capsize = 3)
            calcvals = np.imag(soln_tmp[f'delfstar_calc_{n}'])/n
            ax['checks'][1].plot(soln_tmp['xdata'], calcvals, '-', 
                          color = n_color[n], markerfacecolor='none')
        dg_min = min(dg_min)
        dg_max = max(dg_max)   
        
        # now get range for x data
        x_min.append(min(soln_tmp['xdata']))
        x_max.append(max(soln_tmp['xdata']))        
        x_min = min(x_min)
        x_max = max(x_max)
                              
        # add legend - single legend for both parts
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
        
        # now set x limits
        x_range = x_max - x_min
        if ax['checks'][1].get_xscale() == 'log' and x_min>0:
            ax['checks'][1].set_xlim([0.9*x_min, 1.1*x_max])
        elif ax['checks'][1].get_xscale() == 'linear':
            ax['checks'][1].set_xlim([x_min - 0.05*x_range, 
                        x_max +0.05*x_range])
                
    # now add the response maps        
    if 'maps' in fig.keys():
        # add values to contour plots
        # get xlim so we can reset it if we plot outside the range
        xlim = ax['maps'][0].get_xlim()
        for n in nplot:
            dlam = calc_dlam_from_dlam3(n, soln['dlam3_1'], soln['phi_1'])
            if figdic['info']['maplabels'][n]:
                label = 'n='+str(n)
            else:
                label = ''
            for k in [0,1]:
                ax['maps'][k].plot(dlam, soln['phi_1'], '-o',
                    label = label, mfc = col[n], mec = 'k', c=col[n])         
            # don't include multiple harmonic labels
            figdic['info']['maplabels'][n]=False
        for k in [0, 1]:
            ax['maps'][k].legend(framealpha=1)
            ax['maps'][k].set_xlim(xlim)
    return figdic['fig']['master'], figdic['ax']
    

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
        Sheet for data.  'S_channel' by default.

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
            - 'last' means the last index (highest value)
            - 'first' means we take only the first index (highest value)

    film_idx: numpy array of integers or string
        Index values to include for film data.  Default is 'all' which 
        takes everthing.

    T_coef : dictionary or string
        Temperature coefficients for reference temp. shift  
            
        - calculated from ref. temp. data if not specified
        - set to the following dictionary if unspecified or set to 'default'     
            |  {'f': {1: [0.00054625, 0.04338, 0.08075, 0],           
            |  3: [0.0017, -0.135, 8.9375, 0],     
            |  5: [0.002833, -0.225, 14.89, 0]}, 
            |  7: [0.00397, -0.315, 20.855, 0]},
            |  9: [0.0051, -0.405,  26.8125, 0],
            |  'g': {1: [0, 0, 0, 0], 
            |  3: [0, 0, 0, 0], 
            |  5: [0, 0, 0, 0],
            |  7: [0, 0, 0, 0],
            |  9: [0, 0, 0, 0]}}
            
        - other option is to specify the dictionary directly

    Tref : float
        Temperature at which reference frequency shift was determined.
        Default is 22C.
            
    autodelete : Boolean
        True (default) if we want delete points at temperatures where
        we don't have a reference point'

    T_coef_plots : Boolean  
        True (default) to plot temp. dependent f and g for ref.  

    f_ref_shift : dictionary
        Shifts added to reference values.  
        Default is {1:0, 3:0, 5:0, 7:0, 9:0}.

    nvals : list of integers 
        Harmonics to include.  Default is [1, 3, 5, 7, 9].
        
    index_col : integer
        Column in Excel spreadsheet to use for the dataframe index.
        Default is 0.  Use None to make a new index.
    
       
    overlayer : dictionary
        dictionary containing values or 'grho', 'phi', and 'drho'
        corresponding to properties of top layer - delfstar for this
        layer is substracted from experimental values to give
        delfstar used in calculations

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
    T_coef = kwargs.get('T_coef', 'calculated')
    if T_coef == 'default':
        T_coef = deepcopy(T_coef_default)

    # read shifts that account for changes from stress levels applied
    # to different sample holders
    f_ref_shift=kwargs.get('fref_shift', {1: 0, 3: 0, 5: 0, 7:0, 9:0})
    

    if type(ref_idx)==str and ref_idx == 'max':
        df_ref=pd.read_excel(infile, sheet_name=ref_channel, header=0)
        ref_idx = [df_ref['f3'].idxmax()]
    elif type(ref_idx)== str and ref_idx == 'first':
        df_ref=pd.read_excel(infile, sheet_name=ref_channel, header=0)
        ref_idx = [df_ref.index[0]]
    elif type(ref_idx)==str and ref_idx == 'last':
        df_ref=pd.read_excel(infile, sheet_name=ref_channel, header=0)
        ref_idx = [df_ref.index[-1]]
        
    df=pd.read_excel(infile, sheet_name=film_channel, header=0,
                     index_col = index_col)
    if type(film_idx) != str:
        df=df[df.index.isin(film_idx)]
        
    # keep track of columns for output dataframe
    keep_column = []
                 
    # include all values of n that we want and that exist in the input file
    nvals = []
    for n in nvals_in:
        if f'f{n}' in df.keys():
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
        keep_column.append(f'delfstar_expt_{n}')

    # add the temperature column to original dataframe if it does not exist
    # or contains all nan values, and set all Temperatures to Tref
    if ('temp' not in df.keys()) or (df.temp.isnull().values.all()):
        df['temp'] = Tref
        
    # account for possibility of overlayer
    if 'overlayer' in kwargs.keys():
        overlayer = kwargs.get('overlayer')
        keep_column.append('overlayer')
        df['overlayer'] = [overlayer for _ in range(df.shape[0])]

        
    # add each of the values of delfstar
    if ref_channel == 'self':
        # this is the simplest read protocol, with delf and delg already in
        # the .xlsx file.  All we need to do is read the values and return them
        for n in nvals:
            df[f'delfstar_expt_{n}']=(df[f'delf{n}'].round(1) 
                                      + 1j*df[f'delg{n}'].round(1) - 
                                      f_ref_shift[n].round(1))
            df[f'delfstar_ref_{n}']='self'
            df[f'delfstar_dat_{n}']='self'
        return df [keep_column].copy()
    
    else:
        # read the reference data
        df_ref=pd.read_excel(infile, sheet_name=ref_channel, header=0)
        if type(ref_idx) != str:
            df_ref=df_ref[df_ref.index.isin(ref_idx)]  
            
        if ('temp' not in df_ref.keys()) or (df_ref.temp.isnull().values.all()):
            df_ref['temp'] = Tref
            # this is the common case where we don'to have a full set of 
            # referene temp data, but only values at Tref
            T_coef = deepcopy(T_coef_default)
            T_coef_plots = False
            autodelete = False
            
        # determine temperature coefficients if needed
        if T_coef == 'calculated':
            T_coef = fit_T_coef(df_ref, nvals, ['f', 'g'])


        for n in nvals:
            # check to see if the specified harmonic exists
            if 'f'+str(n) not in df_ref.keys():
                continue
            
            # apply fref_shift if needed
            df_ref['f'+str(n)] = df_ref['f'+str(n)] + f_ref_shift[n]
 
            for val in ['f', 'g']:  
                # determine the constant in the fit function by fitting
                # average values if need
                if T_coef[val][n][3] == 0:
                    if len(df_ref['temp'].unique())!=1:
                        # check to make sure its okay to average
                        print('averaging ref. vals over non-constant temp.')
                    avg = df_ref[val+str(n)].mean()
                    T_coef[val][n][3] = avg - np.polyval(T_coef[val][n],Tref)
                # add absolute frequency and reference values to dataframe
                keep_column.append(val+str(n)+'_dat')
                keep_column.append(val+str(n)+'_ref')
                
                # set reference and film values
                df[val+str(n)+'_ref'] = np.polyval(T_coef[val][n],
                                                   df['temp'])
                df[val+str(n)+'_dat'] = df[val+str(n)]
            
            # keep track of film and reference values in dataframe    
            df[f'fstar_{n}_dat'] = (df[f'f{n}'].round(1)+
                                    1j*df[f'g{n}'].round(1))
            fref = np.polyval(T_coef['f'][n], df['temp']).round(1)
            gref = np.polyval(T_coef['g'][n], df['temp']).round(1)
            df[f'fstar_{n}_ref'] = (fref + 1j*gref)
            df[f'delfstar_expt_{n}']=(df[f'fstar_{n}_dat'].round(1) -
                          df[f'fstar_{n}_ref'].round(1) - f_ref_shift[n])
            keep_column.append(f'fstar_{n}_dat')
            keep_column.append(f'fstar_{n}_ref')
         

    # add the constant applied shift to the reference values to the dataframe
    # also account for overlayer if it exists
    for n in nvals:
        if f_ref_shift[n]!= 0:
            df[f'{n}_refshift']=f_ref_shift[n]
            keep_column.append(f'{n}_refshift')
        if 'overlayer' in df.keys():
            df[f'delfstar_expt_{n}']=(df[f'delfstar_expt_{n}'] - 
                                 calc_delfstar(n, {1:overlayer}))


    T_range=[df['temp'].min(), df['temp'].max()]
    T_ref_range = [df_ref['temp'].min(), df_ref['temp'].max()]
    
    if T_coef_plots:
        plot_bare_tempshift(df_ref, T_coef, Tref, nvals, T_range)
        
    if (autodelete and 
        (T_range[0] < T_ref_range[0] or T_range[1] > T_ref_range[1])):
        df_tmp = df.query('temp >= @T_ref_range[0] & temp <= @T_ref_range[1]')
        # determine number of deleted points
        n_del = len(df) - len(df_tmp)
        print (f'deleting {n_del} points that are outside the ref T range')
        df = df_tmp
            
    # eliminate rows with nan at n=3
    df = df.dropna(subset=['delfstar_expt_3']).copy()
    
    # add time increments
    # df = add_t_diff(df)
    # keep_column.insert(1, 't_prev')
    # keep_column.insert(2, 't_next')

    return df[keep_column].copy()


def fit_T_coef(df, nvals, varvals):
    T_coef = {'f':{}, 'g':{}}
    # reorder reference data according to temperature
    df=df.sort_values('temp')
   
    # drop any duplicate temperature values
    df=df.drop_duplicates(subset='temp', keep='first')
    temp=df['temp']
    if len(temp.unique())<5:
        print('using default T_coef values')
        return T_coef_default
    
    for n in nvals:
        for var in varvals:
            # set T_coef to defaults to start
            # get the reference values and plot them
            ydata=df[var+str(n)]
            
            # make the fitting function
            idx = np.isfinite(temp) & np.isfinite(ydata)
            T_coef[var][n]=np.polyfit(temp[idx], ydata[idx], 3)

            # plot the data if fit was not obtained
            if np.isnan(T_coef[var][n]).any():
                fig, ax = plt.subplots(1,1, figsize=(4,3),
                                       constrained_layout=True)
                ax.plot(temp, ydata)
                ax.set_xlabel(r'$T$ $^\circ$C')
                ax.set_ylabel(f'{var}{n}')
                fig.show()
                print('Temp. coefs could not be obtained - see plot')
                sys.exit()
            
    return T_coef


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


def plot_bare_tempshift(df_ref, T_coef, Tref, nvals, T_range):
    var=['f', 'g']
    n_num=len(nvals)
    # figure for comparision of experimental and fit delf/n, delg/n
    fig, ax=plt.subplots(2, n_num, figsize=(3*n_num, 6),
                                   constrained_layout=True,
                                   num = 'bare crystal data: individual n')
    
    # figure for comparison of all delf/n
    fig2, ax2 = plt.subplots(1, 1, figsize = (4, 3), constrained_layout=True,
                             num='bare crystal data: summary')
    ylabel={0: r'$\Delta f/n$ (Hz)', 1: r'$\Delta \Gamma/n$ (Hz)'}
    ax2.set_ylabel(r'$\Delta f/n$ (Hz)')
    ax2.set_xlabel(r'$T$ ($^\circ$C)')
    
    # for now I'll use a default temp. range to plot
    temp_fit=np.linspace(T_range[0], T_range[1], 100)
    if 1 in nvals: nvals.remove(1) # con't plot values or n=1
    for k , n in enumerate(nvals):
        # plot fit values of delf/n for all harmonics
        vals = bare_tempshift(temp_fit, T_coef, Tref, n)['f']/n
        vals_default = bare_tempshift(temp_fit, T_coef_default, Tref, n)['f']/n
        ax2.plot(temp_fit, vals, f'{col[n]}-', label=f'n={n} fit')
        ax2.plot(temp_fit, vals_default, f'{col[n]}--', label=f'n={n} default fit')
        
        for p in [0, 1]:
            # plot themeasured values, relative to value at ref. temp.
            meas_vals=(df_ref[var[p]+str(n)] -np.polyval(T_coef[var[p]][n], 
                                                         Tref))
            meas_vals=meas_vals/n
            ax[p, k].plot(df_ref['temp'], meas_vals, '-', label = 'meas')

            # now plot the fit values
            ref_val=bare_tempshift(temp_fit, T_coef, Tref, n)[var[p]]
            ref_val=ref_val/n
            ax[p, k].plot(temp_fit, ref_val, '-', label='fit')
            
            # plot default T_coef values for comparison
            ref_val_default=bare_tempshift(temp_fit, T_coef_default, Tref, 
                                           n)[var[p]]
            ref_val_default=ref_val_default/n
            ax[p, k].plot(temp_fit, ref_val_default, '-', label='default')

            # set axis labels and plot titles
            ax[p, k].set_xlabel(r'$T$ ($^\circ$C)')
            ax[p, k].set_ylabel(ylabel[p])
            ax[p, k].set_title(f'n={n}')
            ax[p, k].legend()
            ymin=np.min([meas_vals.min(), ref_val.min()])
            ymax=np.max([meas_vals.max(), ref_val.max()])
            ax[p, k].set_ylim([ymin, ymax])
            
    ax2.legend()
    fig.suptitle('bare crystal data: indivdual n')
    fig2.suptitle('bare crystal data: summary')
    fig.show()
    fig2.show()


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

def gstar(gmag, phi):
    """
    Complex gstar from magnitude and phase (in degrees).

    args:
        gmag (real):
            magnitude of complex modulus
        phi (real)
            phas angle in degrees

    returns:
        gstar:
            complex shear modulus
    """
    return gmag*np.exp(1j*phi*np.pi/180)

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
            Frequencies.
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
        gstar (numpy array):
            complex moduli
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
    return g_tot


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


def add_fstar_err(df, f_error):
    """
    Add fstar uncertainties based on values given in uncertainty dic

    Args
    ----------
    df : dataframe
        Input solution dataframe, typically generated by solve_for_props.
    f_error (list of 3 numbers):
        -uncertainty in f,g as a fraction of g
        
        -uncertainty in f/n (applied individually to harmonics)
        
        -uncertaintiy in correlated f/n (not used here at this point]
             
        default is [0.05, 15, 0]

    Returns
    -------
    Input dataframe with values of fstar_err added

    """

    # add uncertainty columns and set to zero for now
    n_list = []
    for n in [1, 3, 5, 7, 9]:
        if f'delfstar_expt_{n}' in df.keys():
            n_list.append(n)
            for k in ['p', 't']:
                if f'fstar_err_{k}_{n}' not in df.keys():
                    df.insert(df.columns.get_loc(f'delfstar_expt_{n}'), 
                                f'fstar_err_{k}_{n}', 0+1j*0)

    for row in df.itertuples():
        for n in n_list:
            fstar_err = calc_fstar_err(n, row, f_error)
            for k in ['p', 't']:
                df.loc[row.Index,f'fstar_err_{k}_{n}'] = fstar_err[k]
            
    return df
            
def make_response_maps(fig, ax, **kwargs):
    """
    Make response maps of the QCM response - delf and delg
    Args:
        fig (figure handle):
            figure to use for the plot
        ax (list of 2 axis handles):
            axes to use for plot
        
    Kwargs:
        drho (float or string):
            value of drho for the response map.  Set to 
            'Sauerbrey' (default) if we normalize by Sqauerbrey shift
        numxy (integer):
            number of points in x and y directions (default 100)
        numz (integer):
            number of z contours (default 200)
        philim (list of 2 floats):
            range for phi axis (default [0.001, 90])
        dlim (list of 2 floats):
            range for d/lambda axis (default [0.001, 0.5])
        autoscale (Boolean):
            True if we authoscale the contours
        contour_range (dictionary):
            range of contours in form {0:[min,max], 1:[min, max])} \
            default is {0:[-3, 3], 1:[0,3]})
        first_plot (integer):
            number of first plot, if so sublabelfigure is correct (default 0)
        title_strings (list of 2 strings):
            characters to go before and after letters for parts of figures - 
            default is ['(',')']
        title_fontweight (string):
            weight of axes titles - default is 'normal'
            
        title_loc (string):
            location of title - default is 'center' can also be 'left', 'right'
        """
            
    numxy=kwargs.get('numxy', 100)
    numz=kwargs.get('numz', 200)
    philim=kwargs.get('philim', [0, 90])
    dlim=kwargs.get('dlim', [0, 0.5])
    autoscale = kwargs.get('autoscale', False)
    contour_range = kwargs.get('contour_range', {0:[-3, 3], 1:[0,3]})
    first_plot = kwargs.get('first_plot', 0)
    drho = kwargs.get('drho', 'Sauerbrey')
    title_strings = kwargs.get('title_strings', ['(', ')'])
    title_fontweight = kwargs.get('title_fontweight', 'normal')


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
        title_units = [r'$\Delta f_n/n$ (Hz): $d\rho$='+
                       f'{1000*drho:.3g}'+r' $\mu$m$\cdot$g/cm$^3$',
                       r'$\Delta \Gamma _n/n$ (Hz): $d\rho$='+
                       f'{1000*drho:.3g}'+r' $\mu$m$\cdot$g/cm$^3$']
        
    for k in [0, 1]:
        title = (f'{title_strings[0]}{titles_default[k+first_plot]}'+
                 f'{title_strings[1]} {title_units[k]}')
        ax[k].set_title(title, fontweight=title_fontweight)
        
            
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

       
def add_t_diff(df):
    # add time until previous and next point in the file.  Helpful if we want to
    # use data collected at beginning or end of a relaxation step
    # we skip this if 't' is not in the dataframe
    if 't' not in df.columns:
        return
    df.insert(2, 't_prev', 'nan')
    df.insert(3, 't_next', 'nan')
    df.loc[:,'t_next'] = -df.loc[:,'t'].diff(periods=-1)    
    df.at[df.index[-1], 't_next'] = np.inf
    df.loc[:,'t_prev'] = -df.loc[:,'t'].diff(periods=1)
    df.loc[0,'t_prev'] = np.inf
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


def kotula_gstar(xi, Gmstar, Gfstar, xi_crit, s,t):
    ''' 
    Kotua moldel of complex modulus as a function of filler fraction.
    
    Parameters
    ----------
    xi : real 
        Filler volume fraction.
    Gmstar : complex
        Matrix complex modulus.
    Gfstar : complex
        Filler complex modulus.
    xi_crit : real
        Critical filler volume fraction.
    s, t : exponents
    
    '''
    gstar = np.full_like(xi, 1, dtype = complex)
    for i, xival in np.ndenumerate(xi):
        def ftosolve(gstar):
            A = (1-xi_crit)/xi_crit
            func = ((1-xival)*(Gmstar**(1/s)-gstar**(1/s))/(Gmstar**(1/s)+
                     A*gstar**(1/s)) + xival*(Gfstar**(1/t)-gstar**(1/t))/
                     (Gfstar**(1/t)+A*gstar**(1/t)))
            return func
        gstar[i] =findroot(ftosolve, Gmstar)
    return gstar


def kotula_xi(gstar, Gmstar, Gfstar, xi_crit, s,t):
    ''' 
    Kotua moldel filler raction as a function of compex modulus.
    Basically the inverse of kotula-gstar, but we can solve
    this one anaytically
    
    Parameters
    ----------
    gstar : compplex 
        Filler volume fraction.
    Gmstar : complex
        Matrix complex modulus.
    Gfstar : complex
        Filler complex modulus.
    xi_crit : real
        Critical filler volume fraction.
    s, t : exponents
    
    '''
    A = (1-xi_crit)/xi_crit
    xi = (-A * Gmstar**(1/s) * gstar**(1/t) + A * gstar**((s + t)/(s * t)) -       
            Gfstar**(1/t) * Gmstar**(1/s) + Gfstar**(1/t) * gstar**(1/s)) / (A *  
            Gfstar**(1/t) * gstar**(1/s) - A * Gmstar**(1/s) * gstar**(1/t) +
            Gfstar**(1/t) * gstar**(1/s) - Gmstar**(1/s) * gstar**(1/t))
    return xi


def abs_kotula(xi, Gmstar, Gfstar, xi_crit, s, t):
    return abs(kotula_gstar(xi, Gfstar, Gmstar, xi_crit, s, t))

def vline(x, ax, **kwargs):
    '''
    Draws vertical line at x using existing limits

    Parameters
    ----------
    x : float
        location of the vertical line
    ax : axis
        axis on which to plot the line.

    Returns
    -------
    None.

    '''
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    linestyle = kwargs.get('linestyle', 'solid')
    color = kwargs.get('color', 'k')
    
    ax.vlines(x, ymin, ymax, color = color, linestyle = linestyle)
    
def hline(x, ax, **kwargs):
    '''
    Draws horizontal line at y using existing limits

    Parameters
    ----------
    y : float
        location of the horizontal line
    ax : axis
        axis on which to plot the line.

    Returns
    -------
    None.

    '''
    
    xmin = ax.get_xlim()[0]
    xmax = ax.get_xlim()[1]
    linestyle = kwargs.get('linestyle', 'solid')
    color = kwargs.get('color', 'k')
    plt.vlines(x, xmin, xmax, color = color, linestyle = linestyle)


