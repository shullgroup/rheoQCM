#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:19:59 2018

@author: ken
"""

import numpy as np
import sys
import os
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from glob import glob
import time
import shutil

import pandas as pd

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
err_frac_default = 3e-2  # error in f or gamma as a fraction of gamma
T_coef_default = {'f': {1: [0.00054625, 0.04338, 0.08075, 0],
                       3: [0.0017, -0.135, 8.9375, 0],
                       5: [0.002825, -0.22125, 15.375, 0]},
                  'g': {1: [0, 0, 0, 0],
                       3: [0, 0, 0, 0],
                       5: [0, 0, 0, 0]}}

electrode_default = {'drho': 2.8e-3, 'grho3': 3.0e14, 'phi': 0}
water = {'drho':np.inf, 'grho3':1e8, 'phi':90}

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


def grho_bulk(n, delfstar, **kwargs):
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
            each layer. These dictionaries are labeled from 1 to N, with 1
            being the layer in contact with the QCM.  Each dictionary must
            include values for 'grho3, 'phi' and 'drho'. The dictionary for
            layer N can include the film impedance, Zf

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

    # we use the matrix formalism to avoid typos and simplify the extension
    # to large N.
    for i in np.arange(1, N):
        Z[i] = zstar_bulk(n, layers[i], calctype)
        D[i] = calc_D(n, layers[i], delfstar, calctype)
        L[i] = np.array([[np.cos(D[i])+1j*np.sin(D[i]), 0],
                 [0, np.cos(D[i])-1j*np.sin(D[i])]])

    # get the terminal matrix from the properties of the last layer
    if 'Zf' in layers[N].keys():
        Zf_N = layers[N]['Zf'][n]
    else:
        D[N] = calc_D(n, layers[N], delfstar, calctype)
        Zf_N = 1j*zstar_bulk(n, layers[N], calctype)*np.tan(D[N])

    # if there is only one layer, we're already done
    if N == 1:
        return Zf_N

    Tn = np.array([[1+Zf_N/Z[N-1], 0],
          [0, 1-Zf_N/Z[N-1]]])

    uvec = L[N-1]@Tn@np.array([[1.], [1.]])

    for i in np.arange(N-2, 0, -1):
        S[i] = np.array([[1+Z[i+1]/Z[i], 1-Z[i+1]/Z[i]],
          [1-Z[i+1]/Z[i], 1+Z[i+1]/Z[i]]])
        uvec = L[i]@S[i]@uvec

    rstar = uvec[1, 0]/uvec[0, 0]
    return Z[1]*(1-rstar)/(1+rstar)


def calc_delfstar(n, layers, **kwargs):
    """
    Calculate complex frequency shift for stack of layers.
    args:
        n (int):
            Harmonic of interest.

        layers (dictionary):
            Dictionary of material dictionaries specifying the properites of
            each layer. These dictionaries are labeled from 1 to N, with 1
            being the layer in contact with the QCM.  Each dictionary must
            include values for 'grho3, 'phi' and 'drho'.

    kwargs:
        calctype (string):
            - 'SLA' (default): small load approximation with power law model
            - 'LL': Lu Lewis equation, using default or provided electrode props
            - 'Voigt': small load approximation

        reftype (string)
            - 'overlayer' (default):  if overlayer exists, then the reference
               is the overlayer on a bare crystal
            - 'bare': ref is bare crystal, even if overlayer exists

    returns:
        delfstar (complex):
            Complex frequency shift (Hz).
    """

    calctype = kwargs.get('calctype', 'SLA')
    reftype = kwargs.get('reftype', 'overlayer')
    if not layers:  # if layers is empty {}
        return np.nan

    # if layers is not empty:
    if 'overlayer' in layers:
        ZL = calc_ZL(n, {1: layers['film'], 2: layers['overlayer']},
                         0, calctype)
        if reftype == 'overlayer':
            ZL_ref = calc_ZL(n, {1: layers['overlayer']}, 0, calctype)
        else:
            ZL_ref = 0
        del_ZL = ZL-ZL_ref
    else:
        del_ZL = calc_ZL(n, {1: layers['film']}, 0, calctype)

    if calctype != 'LL':
        # use the small load approximation in all cases where calctype
        # is not explicitly set to 'LL'
        return calc_delfstar_sla(del_ZL)

    else:
        # this is the most general calculation
        # use defaut electrode if it's not specified
        if 'electrode' not in layers:
            layers['electrode'] = electrode_default

        layers_all = {1: layers['electrode'], 2: layers['film']}
        layers_ref = {1: layers['electrode']}
        if 'overlayer' in layers:
            layers_all[3] = layers['overlayer']
            if reftype == 'overlayer':
                layers_ref[2] = layers['overlayer']

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
            Calculation string ('353' for example).

        dlam3 (real):
            d/lambda at n=3.

        phi (real):
            Phase angle (degrees).

    returns:
        Harmonic ratio.
    """
    return normdelfstar(calc.split('.')[0], dlam3, phi).real / \
        normdelfstar(calc.split('.')[1], dlam3, phi).real


def rh_from_delfstar(calc, delfstar):
    """
    Determine harmonic ratio from experimental delfstar.
    args:
        calc (3 character string):
            Calculation string ('353' for example).

        delfstar (complex):
            ditionary of complex frequency shifts, included all harmonics
            within calc.

    returns:
        Harmonic ratio.
    """
    # calc here is the calc string (i.e., '353')
    n1 = int(calc.split('.')[0])
    n2 = int(calc.split('.')[1])
    return (n2/n1)*delfstar[n1].real/delfstar[n2].real


def rdcalc(calc, dlam3, phi):
    """
    Calculate dissipation ratio from material properties.
    args:
        calc (3 character string):
            Calculation string ('353' for example).

        dlam3 (real):
            d/lambda at n=3.

        phi (real):
            Phase angle (degrees).

    returns:
        Harmonic ratio.
    """
    return -(normdelfstar(calc.split('.')[2], dlam3, phi).imag /
        normdelfstar(calc.split('.')[2], dlam3, phi).real)


def rd_from_delfstar(n, delfstar):
    """
    Determine dissipation ratio from experimental delfstar.
    args:
        calc (3 character string):
            Calculation string ('353' for example).

        delfstar (complex):
            ditionary of complex frequency shifts, included all harmonics
            within calc.

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
            Magnitude of complex shear modulus, multipled by density, at
            harmonic where delfstar was measured.
        phi:
            Phase angle in degrees, at harmonic where delfstar was measured.
    """

    grho = (np.pi*Zq*abs(delfstar)/f1) ** 2
    phi = -np.degrees(2*np.arctan(delfstar.real /
                      delfstar.imag))
    return grho, min(phi, 90)


def thinfilm_guess(delfstar, calc):
    """
    Determine properties of bulk material from complex frequency shift.
    args:

        delfstar (complex):
            Complex frequency shift (at any harmonic).

        calc (3 character string):
            Calculation string ('353' for example).

    returns:
        drho:
            Mass per unit area in kg/m^2.
        grho3:
            Magnitude of complex shear modulus at n=3, multipled by density
            (SI units).
        phi:
            Phase angle in degrees.
    """
    # really a placeholder function until we develop a more creative strategy
    # for estimating the starting point
    n1 = int(calc.split('.')[0])
    n2 = int(calc.split('.')[1])
    n3 = int(calc.split('.')[2])

    rd_exp = -delfstar[n3].imag/delfstar[n3].real
    rh_exp = (n2/n1)*delfstar[n1].real/delfstar[n2].real
    lb = np.array([0, 0])  # lower bounds on dlam3 and phi
    ub = np.array([5, 90])  # upper bonds on dlam3 and phi

    # we solve the problem initially using the harmonic and dissipation
    # ratios, using the small load approximation
    # we also neglect the overlayer in this first calculation
    def ftosolve(x):
        return [rhcalc(calc, x[0], x[1])-rh_exp, rdcalc(calc, x[0], x[1])-rd_exp]
    guess = [0.05, 5]
    soln = optimize.least_squares(ftosolve, guess, bounds=(lb, ub))

    dlam3 = soln['x'][0]
    phi = soln['x'][1]
    drho = -(sauerbreym(n1, delfstar[n1].real) /
            normdelfstar(n1, dlam3, phi).real)
    grho3 = grho_from_dlam(3, drho, dlam3, phi)
    return drho, grho3, phi


def solve_for_props(delfstar, calc, **kwargs):
    """
    Solve the QCM equations to determine the properties.

    args:
        delfstar (dataframe):
            input dataframe containing complex frequency shifts,
            generally generated by delfstar_from_xlsx

        calc (string):
            string specifying calculation type in form 'x:y.z' or
            x.y:z. Numbers before the : are harmonics used to fit against
            frequency shift.  Numbers after the : are the harmonics used to
            fit against dissipation.  If no : exists ('x.y.z') we assume
            'x.y:z'.  If no . or : exists ('xyz') we assume 'x.y:z'.

    kwargs:
        calctype (string):
            - 'SLA' (default): small load approximation with power law model
            - 'LL': Lu Lewis equation, using default or provided electrode props
            - 'Voigt': small load approximation with Voigt model
            
        guess (dictionary):
            Dictionary with initial guesses for properties 
            ('grho3', 'phi', 'drho').   
        
        overlayer (dictionary):
            Dictionary with Properties of overlayer ('grho3', 'phi', 'drho').
            By default, there is no overlayer.

        drho (real):
            - 0 (default): standard calculation returning drho, grho3, phi
            - anything else:  mass per unit area (SI units) - returning grho3, phi


        lb (list of 3 numbers):  
            Lower bound for Grho3, phi, drho.
            
            - default [1e4, 0, 0]
            
        ub (list of 3 numbers):  
            Upper bound for Grho3, phi, drho.
            
            - default [1e13, 90, 3e-2]

        reftype (string):
            type of calculation 
        
            - 'overlayer' (default):  if overlayer exists, then the reference \
            is the overlayer on a bare crystal
            
            - 'bare': ref is bare crystal, even if overlayer exists
            
        gmax (real):
            maximum value of dissipation for calculation
            - default is 20,000 Hz
            
    returns:
        df_out (dataframe):
            dataframe with properties added, deleting rows with any NaN values \
            that didn't allow calculation to be performed

    """

    df_in = delfstar.T  # transpose the input dataframe
    overlayer = kwargs.get('overlayer', 'air')
    if overlayer == 'air':
        layers = {}
    else:        
        layers = {'overlayer': overlayer}


    nplot = []
    # consider possibility that our data has harmonics up to n=21
    for n in np.arange(1, 22, 2):
        if n in df_in.index:
            nplot = nplot + [n]
    calctype = kwargs.get('calctype', 'SLA')
    reftype = kwargs.get('reftype', 'overlayer')
    drho = kwargs.get('drho', 0)
    gmax = kwargs.get('gmax', 20000)
    
    # set upper and lower bounds
    lb = kwargs.get('lb', [1e4, 0, 0])
    ub = kwargs.get('ub', [1e13, 90, 3e-2])
    lb = np.array(lb)  # lower bounds on grho3, phi, drho
    ub = np.array(ub)  # upper bounds on grho3, phi, drho

    # add dots between numbers if needed
    if not '.' in calc and len(calc)<4:
        calc = '.'.join(calc)
    elif not '.' in calc and len(calc)>=4:
        print('calc has '+str(len(calc))+' digits with no periods')
        sys.exit()

    # now we add the colon if needed
    if not ':' in calc:
        if len(calc.split('.'))==3:
            calc = (calc.split('.')[0]+'.'+
                    calc.split('.')[1]+':'+
                    calc.split('.')[2])
        elif len(calc.split('.')) == 2:
            calc = (calc.split('.')[0]+':'+
                    calc.split('.')[1])
        elif len(calc.split('.')) == 1:
            calc = (calc.split('.')[0]+':'+
                    calc.split('.')[0])
                
                       
    # now we figure out which harmonics we use to fit to delg (ng)
    # or delf (nf)
    nf = calc.split(':')[0].split('.')
    nf = [int(x) for x in nf]
    
    ng = calc.split(':')[1].split('.')
    ng = [int(x) for x in ng]
    
    # nall is a list of all the harmonics we care about for this calculation
    nall = nf + ng
        
    # set up initial guess
    if drho != 0:
        fixed_drho = True
        if 'guess' in kwargs.keys():
            guess = kwargs['guess']
            grho3, phi = guess['grho3'], guess['phi']
        else:
            grho3 = 1e11
            phi = 45
    else:
        fixed_drho = False
        if 'guess' in kwargs.keys():
            guess = kwargs['guess']
            drho, grho3, phi = guess['drho'], guess['grho3'], guess['phi']
        else:
            start_key = np.min(df_in.keys())
            try:
                drho, grho3, phi = thinfilm_guess(df_in[start_key], calc)
            except:
                grho3 = 1e11
                phi = 45
                drho = 2e-3
                

    # set up the initial guess
    if fixed_drho:
        x0 = np.array([grho3, phi])
        lb = lb[0:2]
        ub = ub[0:2]
    else:
        x0 = np.array([grho3, phi, drho])

    # create output dataframe, starting with time and temperature
    df_time_temp = pd.DataFrame(columns = ['t', 'temp'], dtype = 'Float64',
                                index = df_in.keys())
    
    # now create dataframe with the complex frequency shifts
    delfstar_columns = []
    for n in nplot:
        delfstar_columns = delfstar_columns + ['df_expt'+str(n), 'df_calc'+str(n)]
    df_delfstar = pd.DataFrame(columns=delfstar_columns, dtype = 'complex128',
                                index = df_in.keys())

    # now we make the dataframe with calculated properties
    df_real = pd.DataFrame(columns = ['grho3', 'phi', 'drho', 'dlam3'],
                            dtype = 'Float64', index = df_in.keys())
    
    # now we make the dataframe with the objects
    df_object = pd.DataFrame(columns = ['jacobian'],
                            dtype = 'object', index = df_in.keys())
  
    # now we make the dataframe with the strings
    df_string = pd.DataFrame(columns = ['calc', 'calctype'],
                            dtype = 'string', index = df_in.keys())
                             
    # concatenate individual datafames to get df_out
    df_out = pd.concat([df_time_temp, df_delfstar, df_real, df_object,
                        df_string], axis = 1)
    
    # obtain the solution, using either the SLA or LL methods
    for i in df_in.columns:
        # if an initial guess exists in the input file, we use that
        if 'guess' in df_in[i].index:
            x0 = [df_in[i].guess['grho3'], df_in[i].guess['phi'],
                  df_in[i].guess['drho']]
        # check to see if there are any nan values in the harmonics we need
        # also set delfstar to nan for gamma exceeding gmax
        continue_flag = False
        for n in nall:
            if np.isnan(df_in[i][n]) or np.imag(df_in[i][n]) > gmax:
                continue_flag = True
                df_in[i][n]=np.nan  # may not need this anymore
                
        if continue_flag:
            continue
        
        # set up the function to solve
        def ftosolve(x):
            layers['film'] = {'grho3': x[0], 'phi': x[1]}
            if fixed_drho:
                layers['film']['drho'] = drho
            else:
                layers['film']['drho'] = x[2]
            vals = []
            for n in nf: 
                val = (calc_delfstar(n, layers, calctype=calctype,
                                       reftype=reftype).real -
                         df_in[i][n].real)
                vals.append(val)
            for n in ng: 
                val = (calc_delfstar(n, layers, calctype=calctype,
                                       reftype=reftype).imag -
                         df_in[i][n].imag)
                vals.append(val)
            return vals
               
        # make sure x0 is in the right bounds
        for k in np.arange(len(x0)):
            x0[k]=max(x0[k], lb[k])
            x0[k]=min(x0[k], ub[k])

        soln = optimize.least_squares(ftosolve, x0, bounds=(lb, ub))
        grho3 = soln['x'][0]
        phi = soln['x'][1]
        print(soln['x'])

        if not fixed_drho:
            drho = soln['x'][2]

        dlam3 = calc_dlam(3, layers['film'])
        jacobian = soln['jac']
        
        # now back calculate delfstar from the solution
        # add experimental and calculated values to the dataframe
        for n in nplot:
            delfstar_calc = calc_delfstar(n, layers, calctype=calctype,
                                             reftype = reftype)
            df_out.loc[i, 'df_calc'+str(n)] = round(delfstar_calc, 1)
            try:
                df_out.loc[i, 'df_expt'+str(n)] =  delfstar[n][i]
            except:
                df_out.loc[i, 'df_expt'+str(n)] =  'nan'

        # get t, temp, set to 'nan' if they don't exist
        if 't' in df_in[i].keys():
            t = np.real(df_in[i]['t'])
        else:
            t = np.nan
        if 'temp' in df_in[i].keys():
            temp = np.real(df_in[i]['temp'])
        else:
            temp = np.nan

        var = [grho3, phi, drho, dlam3, t, temp, jacobian, calc, calctype]
        var_name = ['grho3', 'phi', 'drho', 'dlam3', 't', 'temp',
                    'jacobian', 'calc', 'calctype']

        for k in np.arange(len(var)):
            df_out.at[i, var_name[k]] = var[k]
        
        # sometimes addition of nan values mess up the dtype for the outputs
        # here we make sure we recast everything to 'float64'
        for varname in ['grho3', 'phi', 'drho', 'dlam3', 't', 'temp']:
            df_out[varname] = df_out[varname].astype('float64')

        # set up the initial guess
        if fixed_drho:
            x0 = np.array([grho3, phi])
        else:
            x0 = np.array([grho3, phi, drho])

    return df_out


def solve_all(datadir, calc, **kwargs):
    """

    Parameters
    ----------
    datadir : string
        Directory for which solutions are obtained for all .xlsx files.
    **kwargs : 
        optional arguments passed along to read_xlsx, solve_for_props, 
        make_prop_axes and check_solution

    Returns
    -------
    df : dictionary
        dictionary of dataframes returned by read_xlsx
    soln : dictionary
        dictinoary of solutions returned by solve_for_props.
    figinfo : dictionary
        diectionary of figinfo returned by make_prop_axes

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
        prop_plots(soln[prefix], figinfo[prefix])
        
        # now set the window title for the solution check
        kwargs['num']=os.path.join(datadir, prefix+'_'+calc+'_check.pdf')
        check_solution(soln[prefix], **kwargs)
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


    idx = kwargs.get('idx', 0)  # specify specific point to use
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
        

def calc_error(soln, uncertainty):
    '''
    Calclate error in properties

    args
    ----------
    soln : Dataframe 
        Data being considered (from solve_for_props)
        
    uncertainty : list of 3 numbers
        Uncertainty in delta_f(n1), delta_f(n2), delta_g(n3)

    Returns
    -------
    input dictionary with errors for grho3, phi, drho added to it

    '''
    propname = {0:'grho3_err', 1:'phi_err', 2:'drho_err'}
    soln = soln.reindex(columns = soln.columns.tolist() +
                            ['grho3_err', 'phi_err', 'drho_err'])
    
    for index in soln.index:
        jacobian = soln.jacobian[index]
        
        try:
            deriv = np.linalg.inv(jacobian)
        except:
            deriv = np.zeros([len(uncertainty), len(uncertainty)])
    
        # determine error from Jacobian
        # p = property
        for p in np.arange(len(uncertainty)):  
            errval = 0
            for k in np.arange(len(uncertainty)):
                errval = errval + (deriv[p, k]*uncertainty[k])**2
            soln.loc[index, propname[p]] = np.sqrt(errval)
            
    return soln
        
        
def make_prop_axes(**kwargs):
    """
    Make a blank property figure.

    kwargs:.
        titles (list):  
            titles for axes
            - (defaults are (a), (b), (c), etc., unlabeled for 
                                 single plot)
            
        num (string):           
            window title (string):
            - (default is 'property fig')
            
        plots (list of strings):  
            plots to include (default is ['grho3', 'phi', 'drho'])
            
            - 'vgp' can be added as van Gurp-Palmen plot
            
            - 'vgp_lin'  and 'grho3_lin' put the grho3 on a linear scale
            
            - 'jdp' is loss compliance normalized by density
            
            -'tanphi' is loss tangent
            
            - 'Gprho3' storage modulus at n=3 times density
            
            - 'Gdprho3'  loss modulus at n = 3 time density
            
            - 'cole-cole' plots Gdprho3 vs. Gdprho3
            
            - 'temp' is temperature in degrees C
            
            - 's', 'hr', 'day' is time in appropriate unit
            
            - name of dataframe column can also be specified
            
        xunit (single string or list of strings):
            Units for x data.  Default is 'index', function currently handles
            - 's', 'min', 'hr', 'day', 'temp', or user specified value corresponding
                to a dataframe column
                
        xscale (string):
            'lin' (default) or 'log'
            
        xlabel (string):
            label for x axis.  Only used if user-specified for xunit is used
            currently must be same for all axes
            
        plotsize (tuple of 2 real numbers):
            size of individual plots.  
            - Defualt is (4, 3)
            
        sharex (Boolean):
            share x axis for zooming
            -(default=True)
            
        orientation (string)
            'horizontal' (default) for horizontal arrangement of figures
        
        no3 (Boolean):
            False (default) if we want to keep the '3' subscript in axis label for G


    returns:
        prop_axes dictionary with the following elements
        fig:
            Handle for the figure
        ax:
            Handle for the axes, returned as a 2d array of 
            dimensions (1, len(plots))
        info:
            Dictionary with plot types for different axes
        xunit:
            Unit for the x axes
    """

    num = kwargs.get('num', 'property fig')
    plots = kwargs.get('plots', ['grho3', 'phi', 'drho'])
    xunit_input = kwargs.get('xunit', 'index')
    sharex = kwargs.get('sharex', True)
    xscale = kwargs.get('xscale', 'lin')
    no3 = kwargs.get('no3', False)
    num_plots = len(plots)
    plotsize = kwargs.get('plotsize', (4,3))
    if num_plots == 1:
        default_titles = ['']
    else:
        default_titles =  ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    titles = kwargs.get('titles', default_titles)
    orientation = kwargs.get('orientation', 'horizontal')
    if orientation == 'horizontal':
        figsize = (plotsize[0]*num_plots, plotsize[1])
    else:
        figsize = (plotsize[0], num_plots*plotsize[1])
    
    # specify the xunit dictionary and xlabel dictionary
    # all plots have same xunit if only one value is given
    xunit = {}
    xlabel = {}
    ylabel = {}
    if type(xunit_input)==str:
        for p in np.arange(num_plots):
            xunit[p] = xunit_input
    else:
        for p in np.arange(num_plots):
            xunit[p]=xunit_input[p]
            
    # turn of sharex if not all axes have the same xunit
    for p in np.arange(num_plots-1):
        if xunit[p]!=xunit[p+1]:
            sharex = False

    if orientation == 'horizontal':
        fig, ax = plt.subplots(1, num_plots, figsize=figsize, num=num,
                               constrained_layout=True, squeeze=False,
                               sharex=sharex)
    else:
        fig, ax = plt.subplots(num_plots, 1, figsize=figsize, num=num,
                               constrained_layout=True, squeeze=False,
                               sharex=sharex)        
        
    ax = ax.flatten()

    for p in np.arange(num_plots):
        # set the x label
        if xunit[p] == 's':
            xlabel[p] = '$t$ (s)'
        elif xunit[p] == 'min':
            xlabel[p] = '$t$ (min.)'
        elif xunit[p] == 'hr':
            xlabel[p] = '$t$ (hr)'
        elif xunit[p] == 'day':
            xlabel[p] = '$t$ (days)'
        elif xunit[p] == 'temp':
            xlabel[p] = r'$T$ ($^\circ$C)'
        elif xunit[p] == 'index':
            xlabel[p] = 'index'
        else:
            xlabel[p] = kwargs.get('xlabel', 'xlabel')
            ylabel[p] = kwargs.get('ylabel', 'ylabel')
    
    # make a dictionary of the potential axis labels
    axlabels = {'grho3': r'$|G_3^*|\rho$ (Pa $\cdot$ g/cm$^3$)',
               'phi': r'$\phi$ (deg.)',
               'tanphi': r'$\tan \phi$',
               'gprho3': r'$G^\prime_3\rho$ (Pa $\cdot$ g/cm$^3$)',
               'gdprho3': r'$G^{\prime\prime}_3\rho$ (Pa $\cdot$ g/cm$^3$)',
               'drho': r'$d\rho$ ($\mu$m$\cdot$g/cm$^3$)',
               'jdp': r'$J^{\prime \prime}/\rho$ (Pa$^{-1}\cdot$cm$^3$/g)',
               'temp':r'$T$ ($^\circ$C)'
               }
    
    # change labels in case where we don't want the 3 subscript for G
    if no3:
        axlabels['grho3'] = r'$|G^*|\rho$ (Pa $\cdot$ g/cm$^3$)'
        axlabels['gprho3'] = r'$G^\prime\rho$ (Pa $\cdot$ g/cm$^3$)'
        axlabels['gdprho3'] = r'$G^{\prime\prime}\rho$ (Pa $\cdot$ g/cm$^3$)'
    
    for p in np.arange(num_plots):
        if plots[p] == 'grho3' or plots[p] == 'grho3_lin':
            ax[p].set_ylabel(axlabels['grho3'])
            ax[p].set_xlabel(xlabel[p])
        elif plots[p] == 'phi':
            ax[p].set_ylabel(axlabels['phi'])
            ax[p].set_xlabel(xlabel[p])
        elif plots[p] == 'tanphi':
            ax[p].set_ylabel(axlabels['tanphi'])
            ax[p].set_xlabel(xlabel[p])
        elif plots[p] == 'gprho3':
            ax[p].set_ylabel(axlabels['gprho3'])
            ax[p].set_xlabel(xlabel[p])
        elif plots[p] == 'gdprho3':
            ax[p].set_ylabel(axlabels['gdprho3'])
            ax[p].set_xlabel(xlabel[p])
        elif plots[p] == 'drho':
            ax[p].set_ylabel(axlabels['drho'])
            ax[p].set_xlabel(xlabel[p])
        elif plots[p] == 'cole-cole':
            ax[p].set_ylabel(axlabels['gdprho3'])
            ax[p].set_xlabel(axlabels['gprho3'])
        elif plots[p] == 'vgp' or plots[p] == 'vgp_lin':
            ax[p].set_ylabel(axlabels['phi'])
            ax[p].set_xlabel(axlabels['grho3'])
            
            # remove links to other axes if they exist
            if sharex:
                ax[p].get_shared_x_axes().remove(ax[p])
            xticker = matplotlib.axis.Ticker()
            ax[p].xaxis.major = xticker
            
            # The new ticker needs new locator and formatters
            xloc = matplotlib.ticker.AutoLocator()
            xfmt = matplotlib.ticker.ScalarFormatter()
            
            ax[p].xaxis.set_major_locator(xloc)
            ax[p].xaxis.set_major_formatter(xfmt)
            
        elif plots[p] == 'jdp':
            ax[p].set_ylabel(axlabels['jdp'])
            ax[p].set_xlabel(xlabel[p])
        elif plots[p] == 'temp':
            ax[p].set_ylabel(axlabels['temp'])
            ax[p].set_xlabel(xlabel[p])
        elif plots[p] == 's':
            ax[p].set_ylabel('t (s)')
            ax[p].set_xlabel(xlabel[p])
        elif plots[p] == 'min':
            ax[p].set_ylabel('t (min)')
            ax[p].set_xlabel(xlabel[p])
        elif plots[p] == 'hrs':
            ax[p].set_ylabel('t (hr)')
            ax[p].set_xlabel(xlabel[p])
        elif plots[p] == 'day':
            ax[p].set_ylabel('t (day)')
            ax[p].set_xlabel(xlabel[p])
        elif 'df' in plots[p]:
            n = plots[p][-1]
            ax[p].set_ylabel(f'$\Delta f_{{{n}}}$ (Hz)')
            ax[p].set_xlabel(xlabel[p])
        elif 'dg' in plots[p]:
            n = plots[p][-1]
            ax[p].set_ylabel(f'$\Delta \Gamma_{{{n}}}$ (Hz)')
            ax[p].set_xlabel(xlabel[p])
        else:
            ax[p].set_xlabel(xlabel[p])
            ax[p].set_ylabel(ylabel[p])
        ax[p].set_title(titles[p])

    info = {'plots':plots, 'xunit':xunit, 'xscale':xscale}
    ax = ax.flatten()
    return {'fig':fig, 'ax':ax, 'info':info}


def prop_plots(df, figinfo, **kwargs):
    """
    Add property data to an existing figure.

    args:
        df (dataframe):
            dataframe containing data to be plotted, typically output from
            solve_for_props
        figinfo (dictionary cotnaining fig, ax and other info for plot):

    kwargs:

        xoffset (real or string, (single value or list)):
            amount to subtract from x value for plotting (default is 0)
            'zero' means that the data are offset so that the minimum val is 0
        xumult (real):
            multiplicative factor we use to multiply the xdata by.
        fmt (string):
            Format sting: Default is '+'   .
        label (string):
            label for plots.  Used to generate legend.  Default is ''
        plotdrho (Boolean):
            Switch to plot mass data or not. Default is True
        uncertainty_dict (dictionary of 2-elment lists of real numbers):
            Uncertainty in f, gamma for each harmonic
        plots_to_make (list of strings):
            plots to make.  Equal to figinfo['info']['plots'] if not specified


    returns:
        Nothing is returned.  The function just updates an existing axis.
    """

    fmt=kwargs.get('fmt', '+')
    label=kwargs.get('label', '')
    xmult = kwargs.get('xmult', 1)
    xoffset_input=kwargs.get('xoffset', 0)
    uncertainty_dict = kwargs.get('uncertainty_dict', 'default')
    
    # extract data from figinfo
    plots = figinfo['info']['plots']
    ax = figinfo['ax']
    num_plots = len(plots)
    plots_to_make = kwargs.get('plots_to_make', figinfo['info']['plots'])
    calc = df['calc'][df.index.min()]
    calc_list = calc.split('.')
    
    # extract uncertainty from datafrae if it is not [0, 0, 0]
    if uncertainty_dict == 'default':
        uncertainty = [0]*len(calc_list)
    else:

        # handle case where calc is a single number, corresponding to the 
        # harmonic where we take frequency and dissipation (for fixed freq.)

        if len(calc_list) == 1:
            n1 = int(calc_list[0])
            uncertainty = [uncertainty_dict[n1][0],
                           uncertainty_dict[n1][1]]
        else: #  now handle the more normal case
            n1 = int(calc.split('.')[0])
            n2 = int(calc.split('.')[1])
            n3 = int(calc.split('.')[2])
            uncertainty = [uncertainty_dict[n1][0],
                           uncertainty_dict[n2][0],
                           uncertainty_dict[n3][1]]
    
    # add calculated errors to dataframe
    df = calc_error(df, uncertainty)
    xunit = figinfo['info']['xunit']
    
    xvals = {}
    
    # set the offset for the x values (apart from vgp plots)
    xoffset = {}
    
    # determine the plots we actually need to make. Sometimes we don't add
    # data to an existing axis
    pvals = []
    for p in np.arange(num_plots):
        if plots[p] in plots_to_make:
            pvals.append(p)
    if type(xoffset_input) != list:
        for p in np.arange(num_plots):
            xoffset[p] = xoffset_input
    else:
        for p in np.arange(num_plots):
            xoffset[p]=xoffset_input[p]   
            
    for p in pvals:
        if xunit[p] == 's':
            xvals[p]=df['t']
        elif xunit[p] == 'min':
            xvals[p]=df['t']/60
        elif xunit[p] == 'hr':
            xvals[p]=df['t']/3600
        elif xunit[p] == 'day':
            xvals[p]=df['t']/(24*3600)
        elif xunit[p] == 'temp':
            xvals[p]=df['temp']
        elif xunit[p] == 'index':
            xvals[p]=df.index
        else:
            xvals[p]=df[xunit[p]]
            
        if xoffset[p] == 'zero':
            xoffset[p] = min(xvals[p])
        
        xvals[p] = xmult*(xvals[p] - xoffset[p])
                
   
    # now make all of the plots
    for p in pvals:
        if plots[p] == 'grho3' or plots[p] == 'grho3_lin': 
            xdata = xvals[p]
            ydata = df['grho3'].astype(float)/1000
            yerr = df['grho3_err']/1000

        elif plots[p] == 'phi':
            xdata = xvals[p]
            ydata = df['phi'].astype(float)
            yerr = df['phi_err']
            
        elif plots[p] == 'tanphi':
            xdata = xvals[p]
            ydata = np.tan(np.pi*df['phi'].astype(float)/180)
            yerr = df['phi_err']  # approximate for now
            
        elif plots[p] == 'gprho3':
            xdata = xvals[p]
            ydata = (df['grho3'].astype(float)*
                     np.cos(np.pi*df['phi'].astype(float)/180)/1000)
            yerr = df['grho3_err'] # appoximate for now

        elif plots[p] == 'gdprho3':
            xdata = xvals[p]
            ydata = (df['grho3'].astype(float)*
                     np.sin(np.pi*df['phi'].astype(float)/180)/1000)
            yerr = df['grho3_err'] # approximate 
                          
        elif plots[p] == 'drho':
            xdata = xvals[p]
            ydata = 1000*df['drho'].astype(float)
            yerr = 1000*df['drho_err']
      
        elif plots[p] == 'vgp' or plots[p] == 'vgp_lin':
            xdata = df['grho3'].astype(float)/1000
            ydata = df['phi'].astype(float)
            yerr = pd.Series(np.zeros(len(xdata)))  # not really included yet
            
        elif plots[p] == 'cole-cole' or plots[p] == 'cole-cole_lin':
            xdata = (df['grho3'].astype(float)*
                     np.cos(np.pi*df['phi'].astype(float)/180)/1000)
            ydata = (df['grho3'].astype(float)*
                     np.sin(np.pi*df['phi'].astype(float)/180)/1000)
            yerr = pd.Series(np.zeros(len(xdata)))  # not really included yet
            
        elif plots[p] == 'jdp':
            xdata  = xvals[p]
            ydata = ((1000/df['grho3'].astype(float))*
                     np.sin(df['phi'].astype(float)*np.pi/180))
            yerr = pd.Series(np.zeros(len(xdata))) # may eventually add error for this one
            
        elif plots[p] == 'temp':
            xdata  = xvals[p]
            ydata = df['temp'].astype(float)
            yerr = pd.Series(np.zeros(len(xdata)))
            
        elif plots[p] == 't':
            xdata  = xvals[p]
            ydata = df['t'].astype(float)
            yerr = pd.Series(np.zeros(len(xdata)))
            
        elif 'df_expt' in plots[p]:
            xdata  = xvals[p]    
            yvals = df[plots[p]].astype(complex)
            ydata = np.real(yvals)
            yerr = pd.Series(np.zeros(len(xdata)))
            
        elif 'dg_expt' in plots[p]:
            xdata  = xvals[p]
            key = plots[p].replace('dg', 'df')
            yvals = df[key].astype(complex)
            ydata = np.imag(yvals)
            yerr = pd.Series(np.zeros(len(xdata)))
            
        elif (plots[p] in df.keys()):
            xdata = xvals[p]
            ydata = df[plots[p]]
            yerr = pd.Series(np.zeros(len(xdata)))
        
        else:
            print('not a recognized plot type ('+plots[p]+')')
            sys.exit()
        
        xdata = xdata.astype('float64')
                
        if (yerr == 0).all() or np.isnan(yerr).all():
            ax[p].plot(xdata, ydata, fmt, label=label)
        else:
            ax[p].errorbar(xdata, ydata, fmt=fmt, yerr=yerr, label=label)
            
        if plots[p] == 'vgp':
            ax[p].set_xscale('log')
        if plots[p] == 'grho3' or plots[p] == 'gprho3' or plots[p] == 'gdprho3':
            ax[p].set_yscale('log')
        if plots[p] == 'cole-cole':
            ax[p].set_xscale('log')
            ax[p].set_yscale('log')           
                

def read_xlsx(infile, **kwargs):
    """
    Create data frame from .xlsx file output by RheoQCM.

    args:
        :infile: (string) the full name of the input  .xlsx file

    kwargs:
        
        :restrict_to_marked: (list) List of frequencies that must be marked in order to be included.
        Default is [], so that we include everything.

        :film_channel: (string) sheet for data
            - 'S_channel' by default

        :ref_channel: (string) Source for reference frequency and dissipation:  

            - 'R_channel': 'R_channel' sheet from xlsx file) (default)  
        
            - 'S_channel': 'S_channel' sheet from xlsx file 
                    
            - 'S_reference': 'S_reference' sheet from xlsx file  
                    
            - 'R_reference': 'R_channel' sheet from xlsx file'  
                    
            - 'self':  read delf and delg read directly from the data channel   
                    

        :ref_idx: (numpy array) index values to include in reference determination  
            
            - default is 'all', which takes 
            - 'max' means we take the value for which f3 is maximized

        :film_idx: (numpy array) index values to include for film data  
            
            - default is 'all' which takes everthing

        :T_coef: (dictionary or string) Temperature coefficients for reference temp. shift  
            
            - calculated from ref. temp. data if not specified
            - set to the following dictionary if equal to 'default'
            
                {'f': {1: [0.00054625, 0.04338, 0.08075, 0],                      
                3: [0.0017, -0.135, 8.9375, 0],                
                5: [0.002825, -0.22125, 15.375, 0]}, 
                'g': {1: [0, 0, 0, 0], 
                3: [0, 0, 0, 0], 
                5: [0, 0, 0, 0]}}
                
            - other option is to specify the dictionary directly

        :Tref: (numeric)
            Temperature at which reference frequency shift was determined  
            - default is 22C

        :T_coef_plots: (Boolean)  
            set to True to plot temp. dependent f and g for ref.  
        
            - default is True

        :fref_shift: (dictionary)
            shifts added to reference values  
        
            - default is {1:0, 3:0, 5:0}

        :nvals: (list) harmonics to include:  
        
            - default is [1, 3, 5]
            
        :Ggess_sheet: (string) sheet name containing property guesses
    
            - used when calculating were eported into the Excel file

    returns:
        :df: (dataframe) Input data converted to dataframe
    """

    restrict_to_marked=kwargs.get('restrict_to_marked', [])
    film_channel=kwargs.get('film_channel', 'S_channel')
    film_idx=kwargs.get('film_idx', 'all')
    ref_channel=kwargs.get('ref_channel', 'R_channel')
    ref_idx=kwargs.get('ref_idx', 'all')
    T_coef_plots=kwargs.get('T_coef_plots', True)
    nvals_in=kwargs.get('nvals', [1, 3, 5, 7, 9])

    Tref=kwargs.get('Tref', 22)
    
    # specify default bare crystal temperature coefficients
    T_coef=kwargs.get('T_coef', 'calculated')
    if T_coef == 'default':
        T_coef = T_coef_default 

    # read shifts that account for changes from stress levels applied
    # to different sample holders
    fref_shift=kwargs.get('fref_shift', {1: 0, 3: 0, 5: 0, 7:0, 9:0})

    if ref_idx == 'max':
        df_ref=pd.read_excel(infile, sheet_name=ref_channel, header=0)
        ref_idx = np.array([df_ref['f3'].idxmax()])
        
    df=pd.read_excel(infile, sheet_name=film_channel, header=0,
                     index_col = 0)
    if type(film_idx) != str:
        df=df[df.index.isin(film_idx)]
        
    # keep track of columns for output dataframe
    keep_column = []
        
    # read guess values if they exist
    guess_sheet = kwargs.get('guess_sheet', 'none')
    sheet_names = pd.ExcelFile(infile).sheet_names
    if guess_sheet in sheet_names:
        df_guess_sheet = pd.read_excel(infile, sheet_name = guess_sheet, header=0,
                                 index_col = 0)
        series = [{}for _ in range(len(df_guess_sheet))]
        for idx in df_guess_sheet.index:
            series[idx] = {'grho3':df_guess_sheet.loc[idx, 'grhos3'],
                       'phi':df_guess_sheet.loc[idx, 'phi'],
                       'drho':df_guess_sheet.loc[idx, 'drho']}
        df_guess = pd.DataFrame({'guess':series})
        # merge in the guesses, keeping the indices from df
        df = df.reset_index().merge(df_guess, left_index = True, right_index = True,
                                    how="left").set_index('index')

        keep_column.append('guess')
        
    # include all values of n that we want and that exist in the input file
    nvals = []
    for n in nvals_in:
        if 'f'+str(n) in df.keys():
            nvals.append(n)
        

    df['keep_row']=1  # keep all rows unless we are told to check for specific marks
    for n in restrict_to_marked:
        df['keep_row']=df['keep_row']*df['mark'+str(n)]

    # delete rows we don't want to keep
    df=df[df.keep_row == 1]  # Delete all rows that are not appropriately marked

    # now sort out which columns we want to keep in the dataframe
    keep_column.append('t')
    keep_column.append('temp')
    for n in nvals:
        keep_column.append(n)

    # add the temperature column to original dataframe if it does not exist or
    # contains all nan values, and set all Temperatures to Tref
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
                T_coef[val][n][3] = (T_coef[val][n][3] + df_ref[val+str(n)].mean() -
                    np.polyval(T_coef[val][n], Tref))
                
                # add absolute frequency and reference values to dataframe
                keep_column.append(val+str(n)+'_dat')
                keep_column.append(val+str(n)+'_ref')
                
                # set reference and film values
                df[val+str(n)+'_ref'] = np.polyval(T_coef[val][n], df['temp'])
                df[val+str(n)+'_dat'] = df[val+str(n)]
            
            # keep track (of film and reference values in dataframe
            df[n]  = (df['f'+str(n)+'_dat'] - df['f'+str(n) + '_ref'] +
                  1j*(df['g'+str(n)+'_dat'] - df['g'+str(n) + '_ref']))
            

    else:
        # here we need to obtain T_coef from the info in the ref. channel
        df_ref=pd.read_excel(infile, sheet_name=ref_channel, header=0)
        if type(ref_idx) != str:
            df_ref=df_ref[df_ref.index.isin(ref_idx)]
        var=['f', 'g']

        # if no temperature is listed or a specific reference temperature
        # is given we just average the values or take the max value
        if ('temp' not in df_ref.keys()) or (df_ref.temp.isnull().values.all()):

            for k in np.arange(len(nvals)):
                for p in [0, 1]:
                    # get the reference values
                    ref_val=df_ref[var[p]+str(nvals[k])].mean()

                    # write the film and reference values to the data frame
                    df[var[p]+str(nvals[k])+'_dat']=df[var[p]+str(nvals[k])]
                    df[var[p]+str(nvals[k])+'_ref']=ref_val


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
                for p in [0, 1]:
                    # set T_coef to defaults to start
                    T_coef = T_coef_default
                    # get the reference values and plot them
                    ref_vals=df_ref[var[p]+str(nvals[k])]
                    
                    # put temp and reference values into a temporary dataframe
                    data = [temp, ref_vals]
                    headers = ['temp', 'data']
                    df_tmp = pd.concat(data, axis=1, keys=headers).dropna()
            
                    # make the fitting function
                    T_coef[var[p]][nvals[k]]=np.polyfit(df_tmp['temp'], 
                                                               df_tmp['data'], 3)
                    
                    
                    # adjust temperature coefficient to get correct value
                    # at ref temp
                    # T_coef[var[p]][nvals[k]][3]=(T_coef[var[p]][nvals[k]][3] +
                    #      refval - np.polyval(T_coef[var[p]][nvals[k]], Tref))
                    
                    # plot the data if fit was not obtained
                    if np.isnan(T_coef[var[p]][nvals[k]]).any():
                        fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
                        ax.plot(df_tmp['temp'], df_tmp['data'])
                        ax.set_xlabel(r'$T$ $^\circ$C')
                        ax.set_ylabel(var[p]+str(nvals[k]))
                        print('Temp. coefficients could not be obtained - see plot')
                        sys.exit()
                                                                        
                    # write the film and reference values to the data frame
                    df[var[p]+str(nvals[k])+'_dat']=df[var[p]+str(nvals[k])]
                    df[var[p]+str(nvals[k])+'_ref']=(
                         np.polyval(T_coef[var[p]][nvals[k]], df['temp']))

        for k in np.arange(len(nvals)):
            # now write values of delfstar to the dataframe
            df[nvals[k]]=(df['f'+str(nvals[k])+'_dat'] -
                          df['f'+str(nvals[k])+'_ref'] +
                      1j*(df['g'+str(nvals[k])+'_dat'] -
                          df['g'+str(nvals[k])+'_ref'])-fref_shift[nvals[k]]).round(1)

            # add absolute frequency and reference values to dataframe
            keep_column.append('f'+str(nvals[k])+'_dat')
            keep_column.append('f'+str(nvals[k])+'_ref')
            keep_column.append('g'+str(nvals[k])+'_dat')
            keep_column.append('g'+str(nvals[k])+'_ref')

    # add the constant applied shift to the reference values to the dataframe
    for n in nvals:
        if fref_shift[n]!= 0:
            df[str(n)+'_refshift']=fref_shift[n]
            keep_column.append(str(n)+'_refshift')

    if T_coef_plots and ref_channel != 'self' and len(df_ref.temp.unique()) > 1:
        T_range=[df['temp'].min(), df['temp'].max()]
        T_ref_range = [df_ref['temp'].min(), df_ref['temp'].max()]
        # create a filename for saving the reference temperature data
        filename = os.path.splitext(infile)[0]+'_Tref.pdf'
        plot_bare_tempshift(df_ref, T_coef, Tref, nvals, T_range, filename)
        if T_range[0] < T_ref_range[0] or T_range[1] > T_ref_range[1]:
            print ('deleting some points that are outside the reference temperature range')
            df = df.query('temp >= @T_ref_range[0] & temp <= @T_ref_range[1]')
            
    # eliminate rows with nan at n=3
    df = df.dropna(subset=[3]).copy()
    
    # add time increments
    df = add_t_diff(df)
    keep_column.insert(1, 't_prev')
    keep_column.insert(2, 't_next')

    return df[keep_column].copy()


def plot_bare_tempshift(df_ref, T_coef, Tref, nvals, T_range, filename):
    var=['f', 'g']
    n_num=len(nvals)
    fig, ax=plt.subplots(2, n_num, figsize=(3*n_num, 6),
                                   constrained_layout=True)
    ylabel={0: r'$\Delta f$ (Hz)', 1: r'$\Delta \Gamma$ (Hz)'}
    # for now I'll use a default temp. range to plot
    temp_fit=np.linspace(T_range[0], T_range[1], 100)
    for p in [0, 1]:
        for k in np.arange(len(nvals)):
            # plot themeasured values, relative to value at ref. temp.
            meas_vals=(df_ref[var[p]+str(nvals[k])] -
                         np.polyval(T_coef[var[p]][nvals[k]], Tref))
            ax[p, k].plot(df_ref['temp'], meas_vals, 'x', label = 'meas')

            # now plot the fit values
            ref_val=bare_tempshift(temp_fit, T_coef, Tref, nvals[k])[var[p]]
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



def bare_tempshift(T, T_coef, Tref, n):
    f=np.polyval(T_coef['f'][n], T) - np.polyval(T_coef['f'][n], Tref)
    g=np.polyval(T_coef['g'][n], T) - np.polyval(T_coef['g'][n], Tref)
    return {'f': f, 'g': g}


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
    args:
        w (numpy array of real values):
            Angular frequencies.
        g0 (list of real values):
            unrelaxed moduli
        tau (list of real values):
            relaxation times
        beta (list of real values):
            exponents
        sp_type (list of integers):
            Specifies the detailed combination of different springpot elments
            combined in series, and then in parallel.  For example, if type is
            [1,2,3],  there are three branches in parallel with one another:
            the first one is element 1, the second one is a series comination of
            elements 2 and 3, and the third one is a series combination of 4, 5 and 6.

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
    tau=np.asarray(tau).reshape(1, -1)[0, :]
    beta=np.asarray(beta).reshape(1, -1)[0, :]
    g0=np.asarray(g0).reshape(1, -1)[0, :]
    sp_type=np.asarray(sp_type).reshape(1, -1)[0, :]

    nw=len(w)  # number of frequencies
    n_br=len(sp_type)  # number of series branches
    n_sp=sp_type.sum()  # number of springpot elements
    sp_comp=np.empty((nw, n_sp), dtype=np.complex)  # element compliance
    br_g=np.empty((nw, n_br), dtype=np.complex)  # branch stiffness

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

def check_solution(df, **kwargs):
    '''
    Create contour plot of normf_g or rh_rd and verify that solution is correct.

    args:
        df (dataframe):
            input solution to consider

    kwargs:
        num (string):
            Also used for window title. Also used for filename
        numxy (int):
            number of grid points in x and y (default is 100)
        numz (int):
            number of z levels (default is 100)
        philim (list of two real numbers):
            min and max of phase angle 
            
            - default is [0, 90]
            
            - 'auto' scales to expt data)
            
        dlim (list of two real numbers):
            min and max of d/lambda 
            
            - (default is [0, 0.5]
               
            - 'auto' scales to expt data)
            
        nplot (list of integers):
            list of harmonics to plot 
            
            - default is [1,3,5]
            
        ratios (Boolean):
            plot rh and rd if True, delf, delg otherwise 
            
            - default is False
            
        autoscale (Boolean):
            auto scale z values to min and max of calculated values if True 
            
            - default is False
            
        label ('string'): 
            dataframe key to use to label individual points in 
            solution check  
            
            - default is 'temp'
            
        plot_solutions (Boolean): 
            True if we want to plot the solution checks for each point  
            
            - default is  False
            
        plot_interval (integer or string): 
            interval between successive solution plots
            
            - default is 1
            
            - 'firstlast': first and last points (by index)
            
            - 'first': first point (by index)
            
            - 'last':  last point by index
    
        xunit (string):
            Units for x data
            
            - Default is 'dlam'
            
            - function currently also handles 's', 'min', 'hr', 'day', 'temp', 'index',
              or the name of column in the input dataframe
              
        xlabel (string):
            label for string (typicall used when datframe column name is used for xunit)
            
        xoffset (real):
            Value subtracted from x data 
            - default is 0
            
        xmult (real):
            Multiplicative factor for x data
            - default is 1
            
        gammascale (string):
            'linear'  or 'log' for scale of dissipation axis
            default is linear
        
        write_pdf (Boolean):
            True (default)  if we want to write the pdf file
            
        contour_plots (Boolean):
            True (default) if we want to write the contour plots
            
        orientation ('string')
            'horizontal' (default) if delf and delg plots are in a row
            'vertical' if we want a column plot
            
        df_lim ('string')
            controls max and min of delf and delg for plotting
            - 'expt' (default) - based on max and min of experimental values
            - 'auto' accounts for all values, including calculated values
            
        plotsize (2-tubple)
            size of individual plots.  Default is (4,3)
        
        plot_df1 (Boolean):
            False (default) if we don't want to plot delf for n=1 in soln check'

            
    Returns:
        {'fig', 'ax'} - dictionary with fig and ax
        
    '''

    from pylab import meshgrid
    numxy=kwargs.get('numxy', 100)
    numz=kwargs.get('numz', 200)
    philim=kwargs.get('philim', [0.001, 90])
    dlim=kwargs.get('dlim', [0.001, 0.5])
    xscale = kwargs.get('xscale', 'linear')
    xmult = kwargs.get('xmult', 1)
    # having d of 0 causes some problems.  Change lower limit to be at least 0.001
    dlim[0] = max(dlim[0], 0.001)
    nplot=kwargs.get('nplot', [1, 3, 5])
    ratios=kwargs.get('ratios', False)
    autoscale=kwargs.get('autoscale', False)
    plot_solutions=kwargs.get('plot_solutions', False)
    plot_interval = kwargs.get('plot_interval', 1)
    idxmin=df.index[0]
    calc=df['calc'][idxmin]
    num=kwargs.get('num', 'solution_check.pdf')
    xunit=kwargs.get('xunit', 'dlam')
    xoffset = kwargs.get('xoffset', 0)
    gammascale = kwargs.get('gammascale', 'linear')
    write_pdf = kwargs.get('write_pdf', True)
    contour_plots = kwargs.get('contour_plots', True)
    # we don't make the contour plots if we have any np.inf values
    if (df['drho']==np.inf).any():
        contour_plots = False
    orientation = kwargs.get('orientation', 'horizontal')
    df_lim = kwargs.get('df_lim', 'expt')
    plotsize = kwargs.get('plotsize', (4,3))
    plot_df1 = kwargs.get('plot_df1', False)
    
    # adjust nplot if any of the values don't' exist in the dataframe
    for n in nplot:
        if not 'df_expt'+str(n) in df.keys():
            nplot.remove(n)
        
    # make sure values of experimental and calculated frequency shifts are cast
    # as complex numbers
    for n in nplot:
        df['df_expt'+str(n)] = df['df_expt'+str(n)].astype(complex)
        df['df_calc'+str(n)] = df['df_calc'+str(n)].astype(complex)
        
    # set up x labels for plots of actual and back-calculated shifts
    if xunit == 's':
        xlabel='$t$ (s)'
        df.loc[:,'xvals']=xmult*(df.loc[:,'t']-xoffset)
    elif xunit == 'min':
        xlabel='$t$ (min.)'
        df.loc[:,'xvals']=xmult*(df.loc[:,'t']/60-xoffset)
    elif xunit == 'hr':
        xlabel='$t$ (hr)'
        df.loc[:,'xvals']=xmult*(df.loc[:,'t']/3600-xoffset)
    elif xunit == 'day':
        xlabel='$t$ (days)'
        df.loc[:,'xvals']=xmult*(df.loc[:, 't']/(24*3600)-xoffset)
    elif xunit == 'temp':
        xlabel=r'$T$ ($^\circ$C)'
        df.loc[:, 'xvals']=xmult*(df.loc[:,'temp']-xoffset)
    elif xunit == 'index':
        xlabel = 'index'
        df.loc[:, 'xvals']=df.index
    elif xunit in df.keys():
        xlabel = kwargs.get('xlabel', 'xlabel')
        df.loc[:, 'xvals'] = xmult*(df.loc[:, xunit]-xoffset)
    else:
        xlabel=r'$d/\lambda_3$'
        df.loc[:,'xvals'] = df.loc[:,'dlam3']
        
    def Zfunction(x, y):
        # function used for calculating the z values
        # this Z is the value plotted in the contour plot and NOT the impedance
        drho=df['drho'][idxmin]
        grho3=grho_from_dlam(3, drho, x, y)
        fnorm=normdelf_bulk(3, x, y)
        gnorm=normdelg_bulk(3, x, y)
        if ratios:
            n1=int(calc.split('.')[0])
            n2=int(calc.split('.')[1])
            n3=int(calc.split('.')[2])
            Z1=np.real(normdelfstar(n2, x, y))/np.real(normdelfstar(n1, x, y))
            Z2=-np.imag(normdelfstar(n3, x, y))/np.real(normdelfstar(n3, x, y))
        else:
            delfstar=sauerbreyf(1, drho)*normdelfstar(3, x, y)
            Z1=np.real(delfstar)
            Z2=np.imag(delfstar)
        return Z1, Z2, drho, grho3, fnorm, gnorm
        
    if contour_plots:
        # make meshgrid for contour
        phi=np.linspace(philim[0], philim[1], numxy)
        dlam=np.linspace(dlim[0], dlim[1], numxy)
        DLAM, PHI=meshgrid(dlam, phi)
        Z1, Z2, drho, grho3, fnorm, gnorm=Zfunction(DLAM, PHI)

        # specify the range of the Z values
        if autoscale:
            min1=Z1.min()
            max1=Z1.max()
            min2=Z2.min()
            max2=Z2.max()
        else:
            if ratios:
                min1=-1
                max1=1.2
                min2=0
                max2=2
            else:
                min1=-3*sauerbreyf(1, df['drho'][idxmin])
                max1=3*sauerbreyf(1, df['drho'][idxmin])
                min2=0
                max2=10000
    
        levels1=np.linspace(min1, max1, numz)
        levels2=np.linspace(min2, max2, numz)
        
    # make the axes and link them for auto-zooming
    if contour_plots:
        ntypes = 2
    else:
        ntypes = 1
    
    if orientation == 'horizontal': 
        fig, ax=plt.subplots(ntypes, 2, figsize=(2*plotsize[0], ntypes*plotsize[1]), 
                             sharex=False, sharey=False,
                             num=num, constrained_layout=True)
        ax = ax.flatten(order = 'C')
    else:
        fig, ax=plt.subplots(2, ntypes, figsize=(ntypes*plotsize[0], 2*plotsize[1]), 
                                sharex=False, sharey=False,
                                num=num, constrained_layout=True)
        ax = ax.flatten(order = 'F') 
        
    ax[0].sharex(ax[1])
    ax[0].set_ylabel(r'$\Delta f/n$ (Hz)')
    ax[1].set_ylabel(r'$\Delta\Gamma/n$ (Hz)')
    ax[0].set_title('(a)')
    ax[1].set_title('(b)')
    
    if contour_plots:
        contour1=ax[2].contourf(DLAM, PHI, Z1, levels=levels1,
                                  cmap='rainbow')
        contour2=ax[3].contourf(DLAM, PHI, Z2, levels=levels2,
                                  cmap='rainbow')
        ax[2].sharex(ax[3])
        ax[2].sharey(ax[3])
        cbax2 = ax[2].inset_axes([1.05, 0, 0.1, 1])
        cbar2 = fig.colorbar(contour1, ax=ax[2], cax = cbax2)
        cbax3 = ax[3].inset_axes([1.05, 0, 0.1, 1])
        cbar3 = fig.colorbar(contour2, ax=ax[3], cax = cbax3)

        # set labels for contour plots
        ax[2].set_xlabel(r'$d/\lambda_n$')
        ax[2].set_ylabel(r'$\Phi$ ($\degree$)')
        ax[3].set_xlabel(r'$d/\lambda_n$')
        ax[3].set_ylabel(r'$\Phi$ ($\degree$)')

        # add titles
        if ratios:
            ax[2].set_title('(c) '+ calc + r' $r_h$')
            ax[3].set_title('(d) '+ calc + r' $r_d$')
        else:
            ax[2].set_title('(c) ' + calc + r' $\Delta f /n$ (Hz)')
            ax[3].set_title('(d) ' + calc + r' $\Delta\Gamma /n$ (Hz)')
        
    # set formatting for parameters that appear at the bottom of the plot
    # when mouse is moved
    def fmt(x, y):
        if ratios:
            z1, z2, drho, grho3, fnorm, gnorm=Zfunction(x, y)
            return 'd/lambda={x:.3f},  phi={y:.1f}, rh={z1:.5f}, rd={z2:.5f}, '\
                    'drho={drho:.2f}, grho3={grho3:.2e}, '\
                    'fnorm={fnorm:.4f}, gnorm={gnorm:.4f}'.format(x=x, y=y,
                     z1=z1, z2=z2,
                     drho=1000*drho, grho3=grho3/1000, fnorm=fnorm, gnorm=gnorm)

        else:
            z1, z2, drho, grho3, fnorm, gnorm=Zfunction(x, y)
            return 'd/lambda={x:.3f},  phi={y:.1f}, delfstar/n={z:.0f}, '\
                    'drho={drho:.2f}, grho3={grho3:.2e}, '\
                    'fnorm={fnorm:.4f}, gnorm={gnorm:.4f}'.format(x=x, y=y,
                     z=z1+1j*z2,
                     drho=1000*drho, grho3=grho3/1000, fnorm=fnorm, gnorm=gnorm)

    # set up standar color scheme for the different harmonics
    col = {}
    for n in np.arange(1,22,2):
        col[n]='C'+str(int((n-1)/2))
        
    # now add the comparison plots of measured and calcuated values         
    # plot the experimenta data first
    # keep track of max and min values for plotting purposes
    if len(df['xvals'])==1:
        calcfmt = 'o'
    else:
        calcfmt = '-'
    df_min = []
    df_max = []
    dg_min = []
    dg_max = []
    df['xvals'] = df['xvals'].astype('float64')
    
    # now plot the calculated values
    nfplot = nplot.copy()
    ngplot = nplot.copy()
    if not plot_df1:
        try:
            nfplot.remove(1)
        except ValueError:
            pass
        
    for n in nfplot: 
        dfval = np.real(df['df_expt'+str(n)])/n
        df_min.append(np.nanmin(dfval))
        df_max.append(np.nanmax(dfval))
        nstr=str(n)+': expt' 
        ax[0].plot(df['xvals'].astype('float64'), 
                   dfval, '+', label='n='+nstr, color = col[n])
        ax[0].plot(df['xvals'], np.real(df['df_calc'+str(n)])/n, calcfmt, 
                      color = col[n], markerfacecolor='none', label='calc')
    df_min = min(df_min)
    df_max = max(df_max)
 
    for n in ngplot:
        dgval = np.imag(df['df_expt'+str(n)])/n
        dg_min.append(np.nanmin(dgval))
        dg_max.append(np.nanmax(dgval))
        nstr=str(n)+': expt' 
        ax[1].plot(df['xvals'].astype('float64'),
                   dgval, '+', label='n='+nstr, color = col[n])
        ax[1].plot(df['xvals'], np.imag(df['df_calc'+str(n)])/n, calcfmt, 
                      color = col[n], markerfacecolor='none', label='calc')

    dg_min = min(dg_min)
    dg_max = max(dg_max)    
        
    # change dissipation scale to log scale if needed
    if gammascale=='log':
        ax[1].set_yscale('log')
        
    # change scale to log scale if needed
    if xscale == 'log':
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')    

    ax[0].legend(ncol=1, labelspacing=0.1, columnspacing=0, 
                    markerfirst=False, handletextpad=0.1,
                    bbox_to_anchor=(1.02, 1),
                    handlelength=1)
    ax[1].legend(ncol=1, labelspacing=0.1, columnspacing=0, 
                    markerfirst=False, handletextpad=0.1,
                    bbox_to_anchor=(1.02, 1),
                    handlelength=1)
    for k in [0, 1]: ax[k].set_xlabel(xlabel) 
        
    # reset y axis limits for delf and delg if needed
    if df_lim == 'expt':
        delf_range = df_max - df_min
        delg_range = dg_max - dg_min
        ax[0].set_ylim([df_min - 0.05*delf_range, df_max +0.05*delf_range])
        if gammascale == 'log' and dg_min>0:
            ax[1].set_ylim([0.9*dg_min, 1.1*dg_max])
        elif gammascale == 'linear':
            ax[1].set_ylim([dg_min - 0.05*delg_range, 
                        dg_max +0.05*delg_range])

    # add values to contour plots
    if contour_plots:
        for n in nplot:
            dlam = calc_dlam_from_dlam3(n, df['dlam3'], df['phi'])
            ax[2].plot(dlam, df['phi'], '-o', markerfacecolor='none',
                          label = 'n='+str(n), color = col[n])
            ax[3].plot(dlam, df['phi'], '-o', markerfacecolor='none',
                          label = 'n='+str(n), color = col[n])
            
        for k in [2, 3]:
            ax[k].legend()
            ax[k].format_coord=fmt
    
            # reset axis limits
            ax[k].set_xlim(dlim)
            ax[k].set_ylim(philim)


    # create a PdfPages object - one solution check per page
    if write_pdf:
        pdf=PdfPages(num)
    
    # we only take every nth row, where n = plot_interval
    if plot_interval == 'firstlast':
        df_plot = df.iloc[[0, -1], :]
    elif plot_interval =='first':
        df_plot = df.iloc[[0], :]
    elif plot_interval == 'last':
        df_plot = df.iloc[[-1], :]
    else:
        df_plot = df.iloc[::plot_interval, :]
    if plot_solutions and write_pdf:
        idxnum = 0 # keeps track of the fact that we don't always start from idx=0
        for idx, row in df_plot.iterrows():
            idxnum = idxnum + 1
            curves={}

            # indicate where the solution is being taken
            print('writing solution '+str(idxnum)+' of '+str(len(df_plot)))

            # label for data point in case we want use it
            curves[0]=ax[1, 0].plot(row['dlam3'], row['phi'], 'kx', 
                            markersize=14)
            
            curves[1]=ax[1, 1].plot(row['dlam3'], row['phi'], 'wx', 
                            markersize=14)
            
            # now plot the lines for the solution
            rh=rhcalc(calc, row['dlam3'], row['phi'])
            rd=rdcalc(calc, row['dlam3'], row['phi'])

            def solutions(phi, guess):
                def ftosolve_rh(d):
                    return rhcalc(calc, d, phi)-rh

                def ftosolve_rd(d):
                    return rdcalc(calc, d, phi)-rd

                soln_rh=optimize.least_squares(ftosolve_rh, guess[0])
                soln_rd=optimize.least_squares(ftosolve_rd, guess[1])
                return {'phi': phi, 'd_rh': soln_rh['x'][0], 'd_rd': soln_rd['x'][0],
                        'resid_rh':soln_rh['fun'][0], 'resid_rd':soln_rd['fun'][0]}

            npts=25
            dcalc=pd.DataFrame(columns=['phi', 'd_rh', 'd_rd'])

            # starting guess is the actual solution, and then we work outward
            # from there
            for phiend in philim:
                guess=[row['dlam3'], row['dlam3']]
                phivals=np.linspace(row['phi'], phiend, npts)
                for phival in phivals:
                    soln=solutions(phival, guess)
                    # break if there is not solution for this value of phi
                    # for either rd or rh
                    if soln['resid_rh'] > 1e-4 or soln['resid_rd']>1e-4:
                        break
                    dcalc=dcalc.append(soln, ignore_index=True)
                    guess=[soln['d_rh'], soln['d_rd']]

            dcalc=dcalc.sort_values(by=['phi'])
            
            # now plot the curves of constant rh and rd
            curves[2]=ax[2].plot(dcalc['d_rh'], dcalc['phi'], 'k-',
                                    label =r'$r_h$')
            curves[3]=ax[2].plot(dcalc['d_rd'], dcalc['phi'], 'k--',
                                    label = r'$r_d$')
            curves[4]=ax[3].plot(dcalc['d_rh'], dcalc['phi'], 'w-',
                                    label = r'$r_h$')
            curves[5]=ax[3].plot(dcalc['d_rd'], dcalc['phi'], 'w--',
                                    label = r'$r_d$')
            
            for k in [2,3]:
                ax[k].legend(ncol=2)
                ax[k].set_xlim(left=0)
                      
            pdf.savefig()
            if idxnum>1:
                for k in np.arange(6):
                    curves[k][0].remove()
    else:
        if write_pdf:
            pdf.savefig()
                
    if write_pdf: pdf.close()
    figinfo = {'fig':fig, 'ax':ax} 
    if contour_plots:
        figinfo['colorbars'] =[cbar2, cbar3]
    return figinfo


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
    df.loc[:,'t_next'] = -df.loc[:,'t'].diff(periods=-1)
    df.iloc[-1, df.columns.get_loc('t_next')] = np.inf
    df['t_prev'] = -df['t'].diff(periods=1)
    df.iloc[0, df.columns.get_loc('t_prev')] = np.inf
    return df
