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
from matplotlib.backends.backend_pdf import PdfPages

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
                       5: [0, 0, 0, 9]}}

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


def fstar_err_calc(n, delfstar, layers, **kwargs):
    """
    Calculate the error in delfstar.
    args:
        n (int):
            Harmonic to consider.
        delfstar (numpy array):
            Array of complex frequency shifts at n.
        layers (dictionary):
            Property stack for different layers considered in the calculation.

    kwargs:
        g0 (real):
            Dissipation at n for dissipation.  Default value set at top
            of QCM_functions.py (typically 50).
        err_frac:
            Uncertainty in delf, delg as a fraction of gamma.  Default set
            at top of QCM_functions.py (typically 0.03).

    returns:
        fstar_err (numpy array of complex values):
            Array errors in delf (real part) and delg (imag part).
    """

    g0 = kwargs.get('g0', g0_default)
    err_frac = kwargs.get('err_frac', err_frac_default)

    g = g0 + np.imag(delfstar)

    if 'overlayer' in layers:
        delg = calc_delfstar(n, {'film': layers['overlayer']},
                                             calctype='SLA')
        g = g+delg

    # start by specifying the error input parameters
    fstar_err = err_frac*abs(g)*(1+1j)
    return fstar_err


def sauerbreyf(n, drho):
    return 2*n*f1 ** 2*drho/Zq
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


def deltarho_bulk(n, delfstar):
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
    # decay length multiplied by density
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
    return normdelfstar(calc[0], dlam3, phi).real / \
        normdelfstar(calc[1], dlam3, phi).real


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
    n1 = int(calc[0])
    n2 = int(calc[1])
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
    return -(normdelfstar(calc[2], dlam3, phi).imag /
        normdelfstar(calc[2], dlam3, phi).real)


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


def bulk_props(delfstar):
    """
    Determine properties of bulk material from complex frequency shift.
    args:
        calc (3 character string):
            Calculation string ('353' for example).

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
    n1 = int(calc[0])
    n2 = int(calc[1])
    n3 = int(calc[2])

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
            3 character string specifying calculation type (e.g., '353');
            only third character is used for filmtype='bulk'

    kwargs:
        calctype (string):
            - 'SLA' (default): small load approximation with power law model
            - 'LL': Lu Lewis equation, using default or provided electrode props
            - 'Voigt': small load approximation

        drho (real):
            - 0 (default): standard calculation returning drho, grho3, phi
            - anything else:  mass per unit area (SI units) - returning grho3, phi

        newtonian (Boolean): (only used for 'bulk' filmtype)
            - False (default): standard bulk calculation
            - True: phase angle assumed to be 90 degrees and viscosity obtained
              from dissipation

        err_frac (real):  Error in delf and delg as a fraction of delg
            - Default is zero

        lb (list of 3 numbers):  Lower bound for Grho3, phi, drho - default [1e4, 0, 0]
        ub (list of 3 numbers):  Upper bound for Grho3, phi, drho - default [1e13, 90, 3e-2]

        reftype (string)
            - 'overlayer' (default):  if overlayer exists, then the reference
               is the overlayer on a bare crystal
            - 'bare': ref is bare crystal, even if overlayer exists

    returns:
        df_out (dataframe):
            dataframe with properties added, deleting rows with any NaN values \
            that didn't allow calculation to be performed

    """

    err_frac = kwargs.get('err_frac', 0)
    df_in = delfstar.T  # transpose the input dataframe
    if 'overlayer' in kwargs.keys():
        layers = {'overlayer': kwargs['overlayer']}
    else:
        layers = {}

    nplot = []
    for n in [1, 3, 5, 7, 9]:
        if n in df_in.index:
            nplot = nplot + [n]
    calctype = kwargs.get('calctype', 'SLA')
    reftype = kwargs.get('reftype', 'overlayer')
    drho = kwargs.get('drho', 0)
    newtonian = kwargs.get('newtonian', False)

    # set upper and lower bounds
    lb = kwargs.get('lb', [1e4, 0, 0])
    ub = kwargs.get('ub', [1e13, 90, 3e-2])
    lb = np.array(lb)  # lower bounds drho, grho3, phi
    ub = np.array(ub)  # upper bounds drho, grho3, phi

    if drho != 0:
        fixed_drho = True
        n1 = int(calc[0]); n2 = int(calc[0]); n3 = int(calc[0])

        if 'guess' in kwargs.keys():
            guess = kwargs['guess']
            grho3, phi = guess['grho3'], guess['phi']
        else:
            grho3 = 1e11
            phi = 45
            n1 = n3  # fixed thickness uses delta f and delta gamma from same harmonic
    else:
        fixed_drho = False
        n1 = int(calc[0]); n2 = int(calc[1]); n3 = int(calc[2])

        if 'guess' in kwargs.keys():
            guess = kwargs['guess']
            drho, grho3, phi = guess['drho'], guess['grho3'], guess['phi']
        else:
            start_key = np.min(df_in.keys())
            drho, grho3, phi = thinfilm_guess(df_in[start_key], calc)

    # set up the initial guess
    if fixed_drho:
        x0 = np.array([grho3, phi])
        lb = lb[0:2]
        ub = ub[0:2]
    else:
        x0 = np.array([grho3, phi, drho])

    # create keys for output dictionary and dataframe
    out_rows = []
    for n in nplot:
        out_rows = out_rows + ['df_expt'+str(n), 'df_calc'+str(n)]

    out_rows = out_rows+['drho', 'drho_err', 'grho3', 'grho3_err', 'phi', 'phi_err',
                       'dlam3', 't', 'temp']

    data = {}
    props = {}  # properties for each layer
    for element in out_rows:
        data[element] = np.array([])

    for n in nplot:
        data['df_expt'+str(n)] = np.array([])
        data['df_calc'+str(n)] = np.array([])

    # obtain the solution, using either the SLA or LL methods
    for i in df_in.columns:
        if np.isnan([df_in[i][n1], df_in[i][n2], df_in[i][n3]]).any():
            continue
        if not fixed_drho:
            def ftosolve(x):
                layers['film'] = {'grho3': x[0], 'phi': x[1], 'drho': x[2]}
                return ([calc_delfstar(n1, layers, calctype=calctype,
                                       reftype=reftype).real -
                         df_in[i][n1].real,
                         calc_delfstar(n2, layers, calctype=calctype,
                                       reftype = reftype).real -
                         df_in[i][n2].real,
                         calc_delfstar(n3, layers, calctype=calctype,
                                       reftype = reftype).imag -
                         df_in[i][n3].imag])
        else:
            def ftosolve(x):
                layers['film'] = {'drho': drho, 'grho3': x[0], 'phi': x[1]}
                if not newtonian:
                    return ([calc_delfstar(n1, layers, calctype=calctype,
                                           reftype = reftype).real -
                             df_in[i][n1].real,
                             calc_delfstar(n1, layers, calctype=calctype,
                                           reftype = reftype).imag -
                             df_in[i][n1].imag])
                else:  # set frequency shift equal to dissipation shift in this case
                    return ([calc_delfstar(n1, layers, calctype=calctype,
                                           reftype = reftype).real +
                             df_in[i][n1].imag,
                             calc_delfstar(n1, layers, calctype=calctype,
                                           reftype = reftype).imag -
                             df_in[i][n1].imag])

        # initialize the output uncertainties
        err = {}
        soln = optimize.least_squares(ftosolve, x0, bounds=(lb, ub))
        grho3 = soln['x'][0]
        phi = soln['x'][1]
        print(soln['x'])

        if not fixed_drho:
            drho = soln['x'][2]

        layers['film'] = {'drho': drho, 'grho3': grho3, 'phi': phi}
        dlam3 = calc_dlam(3, layers['film'])
        jac = soln['jac']

        try:
            deriv = np.linalg.inv(jac)
        except:
            deriv = np.zeros([len(x0), len(x0)])

        # put the input uncertainties into a 3 element vector
        delfstar_err = np.zeros(3)
        delfstar_err[0] = fstar_err_calc(n1, df_in[i][n1], layers,
                                         err_frac=err_frac).real
        delfstar_err[1] = fstar_err_calc(n2, df_in[i][n2], layers,
                                         err_frac=err_frac).real
        delfstar_err[2] = fstar_err_calc(n3, df_in[i][n3], layers,
                                         err_frac=err_frac).imag

        # determine error from Jacobian
        err = {}
        for p in np.arange(len(x0)):  # p = property
            err[p] = 0
            for k in np.arange(len(x0)):
                err[p] = err[p]+(deriv[p, k]*delfstar_err[k])**2
            err[p] = np.sqrt(err[p])

        if fixed_drho:
            err[2] = np.nan

        # now back calculate delfstar from the solution
        delfstar_calc = {}

        # add experimental and calculated values to the dictionary
        for n in nplot:
            delfstar_calc[n] = calc_delfstar(n, layers, calctype=calctype,
                                             reftype = reftype)
            data['df_calc'+str(n)] = (np.append(data['df_calc'+str(n)],
                                             round(delfstar_calc[n], 1)))
            try:
                data['df_expt'+str(n)] = (np.append(data['df_expt'+str(n)],
                                             delfstar[n][i]))
            except:
                data['df_expt' +
                    str(n)] = np.append(data['df_expt'+str(n)], 'nan')

        # get t, temp, set to 'nan' if they doesn't exist
        if 't' in df_in[i].keys():
            t = np.real(df_in[i]['t'])
        else:
            t = np.nan
        if 'temp' in df_in[i].keys():
            temp = np.real(df_in[i]['temp'])
        else:
            temp = np.nan

        var_name = ['grho3', 'phi', 'drho', 'grho3_err', 'phi_err', 'drho_err',
                    'dlam3', 't', 'temp']
        var = [grho3, phi, drho, err[0], err[1], err[2],
               dlam3, t, temp]

        for k in np.arange(len(var_name)):
            data[var_name[k]] = np.append(data[var_name[k]], var[k])

        props[i] = layers['film']
        # set up the initial guess
        if fixed_drho:
            x0 = np.array([grho3, phi])
        else:
            x0 = np.array([grho3, phi, drho])

    # add these calculated values to existing dataframe
    df_out = pd.DataFrame(data)
    df_out['props'] = props
    df_out['calc'] = calc
    return df_out


def make_err_plot(df_in, **kwargs):
    """
    Determine errors in properties based on uncertainies in a delfstar.
    args:
        df_in (dataframe):
            Input data.

    kwargs:
        idx (int):
            index of point in df_in to use (default is 0)
        npts (int):
            number of points to include in error plots (default is 10)
        err_frac (real):
            error in delfstar as a fraction of gamma.
        err_range (real):
            multiplicative factor that expands err range beyond err_frac.


    returns:
        fig:
            Figure containing various error plots.
        ax:
            axes of the figure.
    """
    fig = make_prop_axes(figsize=(12, 3))
    ax = fig['ax']
    idx = kwargs.get('idx', 0)  # specify specific point to use
    npts = kwargs.get('npts', 10)
    err_frac = kwargs.get('err_frac', err_frac_default)
    # >1 to extend beyond calculated err
    err_range = kwargs.get('err_range', 1)
    err_range = max(1, err_range)
    calctype = df_in['calctype'][idx]
    calc = df_in['calc'][idx]
    deriv = df_in['deriv'][idx]
    delfstar_err = df_in['delfstar_err'][idx]
    guess = {'grho3': df_in['grho3'][idx],
             'phi': df_in['phi'][idx],
             'drho': df_in['drho'][idx]}

    delfstar_0 = {}

    for nstr in list(set(calc)):
        n = int(nstr)
        delfstar_0[n] = df_in[n][idx]

    # now generate series of delfstar values based on the errors
    delfstar_del = {}

    # set some parameters for the plots
    # frequency or dissipation shift
    mult = np.array([1, 1, 1j], dtype=complex)
    forg = {0: 'f', 1: 'f', 2: '$\Gamma$'}
    prop_type = {0: 'grho3', 1: 'phi', 2: 'drho'}
    scale_factor = {0: 0.001, 1: 1, 2: 1000}
    marker = {0: '+', 1: 's', 2: 'o'}

    # intialize values of delfstar
    for k in [0, 1, 2]:
        delfstar_del[k] = {}
        for n in [3, 5]:
            delfstar_del[k][n] = np.ones(npts)*delfstar_0[n]

    # adjust values of delfstar and calculate properties
    for k in [0, 1, 2]:
        ax[0,k].set_xlabel(r'$(X-X_0)/X^{err}$')
        n = int(calc[k])
        err = err_range*delfstar_err[k]
        delta = np.linspace(-err, err, npts)
        delfstar_del[k][n] = delfstar_del[k][n]+delta*mult[k]
        delfstar_df = pd.DataFrame.from_dict(delfstar_del[k])
        props = solve_for_props(delfstar_df, calc=calc,
                                          calctype=calctype, guess=guess)
        # make the property plots
        for p in [0, 1, 2]:
            ax[0,p].plot(delta/delfstar_err[k], props[prop_type[p]]*scale_factor[p],
                       marker=marker[k], linestyle='none',
                       label=r'X='+forg[k]+'$_'+str(n)+'$')

    # reset color cycles so dervitave plots match the color scheme
    for p in [0, 1, 2]:
        ax[0,p].set_prop_cycle(None)
        for k in [0, 1, 2]:
            err = delfstar_err[k]
            xdata = np.array([-err_range, 0, err_range])
            ydata = (np.ones(3)*df_in[prop_type[p]][idx]*scale_factor[p] +
                     err*xdata*scale_factor[p]*deriv[p][k])
            ax[0,p].plot(xdata, ydata, '-')

        # now add the originally calculated error
            err0 = df_in[prop_type[p]+'_err']
            ax[0,p].errorbar(0, df_in[prop_type[p]][idx]*scale_factor[p],
                           yerr=err0*scale_factor[p], color='k')

    ax[0,0].legend(loc='center', bbox_to_anchor=(-0.5, 0, 0, 1))
    sub_string = {}
    for k in [0, 1, 2]:
        n = calc[k]
        sub_string[k] = (forg[k]+'$_'+n+'^{err}=$' +
                         f'{delfstar_err[k]:.0f}'+' Hz')
    title_string = (sub_string[0]+'; '+sub_string[1]+'; '+sub_string[2] +
                  '; '+'err_frac='+str(err_frac))
    fig.suptitle(r''+title_string)
    fig.tight_layout()
    return fig, ax


def make_prop_axes(**kwargs):
    """
    Make a blank property figure.

    kwargs:.
        titles (dictionary):
            titles for axes (defaults are (a), (b), (c), etc.)
        num (string):
            window title (default is 'property fig')
        plots (list of strings):
            plots to include (default is ['grho3', 'phi', 'drho'])
            'vgp' can be added as van Gurp-Palmen plot
            'vgp_lin'  and 'grho3_lin' put the grho3 on a linear scale
            'jdp' is loss compliance normalized by density
            'temp' is temperature in degrees C
            't' is time in seconds
            
        xunit (string):
            Units for x data.  Default is 'index', function currently handles
            's', 'min', 'hr', 'day', 'temp', or user specified value corresponding
            to a dataframe column
            
        xlabel (string):
            label for x axis.  Only used if user-specified for xunit is used
            
        figsize (tuple of 2 real numbers):
            size of figure.  Defualt is (3*num of plots, 3)


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
    titles = kwargs.get('titles', {0: '(a)', 1: '(b)', 2: '(c)', 3: '(d)'})
    xunit = kwargs.get('xunit', 'index')
    num_plots = len(plots)
    figsize = kwargs.get('figsize', (3*num_plots, 3))

    fig, ax = plt.subplots(1, num_plots, figsize=figsize, num=num,
                           constrained_layout=True, squeeze=False)

    # set the x label
    if xunit == 's':
        xlabel = '$t$ (s)'
    elif xunit == 'min':
        xlabel = '$t$ (min.)'
    elif xunit == 'hr':
        xlabel = '$t$ (hr)'
    elif xunit == 'day':
        xlabel = '$t$ (days)'
    elif xunit == 'temp':
        xlabel = r'$T$ ($^\circ$C)'
    elif xunit == 'index':
        xlabel = 'index'
    else:
        xlabel = kwargs.get('xlabel', 'xlabel')

    # make a dictionary of the potential axis labels
    axlabels = {'grho3': r'$|G_3^*|\rho$ (Pa $\cdot$ g/cm$^3$)',
               'phi': r'$\phi$ (deg.)',
               'drho': r'$d\rho$ ($\mu$m$\cdot$g/cm$^3$)',
               'jdp': r'$J^{\prime \prime}/\rho$ (Pa$^{-1}\cdot$cm$^3$/g)',
               'temp':r'$T$ ($^\circ$C)'
               }
    
    for p in np.arange(num_plots):
        if plots[p] == 'grho3' or plots[p] == 'grho3_lin':
            ax[0,p].set_ylabel(axlabels['grho3'])
            ax[0,p].set_xlabel(xlabel)
        elif plots[p] == 'phi':
            ax[0,p].set_ylabel(axlabels['phi'])
            ax[0,p].set_xlabel(xlabel)
        elif plots[p] == 'drho':
            ax[0,p].set_ylabel(axlabels['drho'])
            ax[0,p].set_xlabel(xlabel)
        elif plots[p] == 'vgp' or plots[p] == 'vgp_lin':
            ax[0,p].set_ylabel(axlabels['phi'])
            ax[0,p].set_xlabel(axlabels['grho3'])
        elif plots[p] == 'jdp':
            ax[0,p].set_ylabel(axlabels['jdp'])
            ax[0,p].set_xlabel(xlabel)
        elif plots[p] == 'temp':
            ax[0,p].set_ylabel(axlabels['temp'])
            ax[0,p].set_xlabel(xlabel)
        elif plots[p] == 't':
            ax[0,p].set_ylabel('t (s)')
            ax[0,p].set_xlabel(xlabel)
        if num_plots > 1:
            ax[0,p].set_title(titles[p])

    info = {'plots':plots, 'xunit':xunit}
    return {'fig':fig, 'ax':ax, 'info':info}


def prop_plots(df, figinfo, **kwargs):
    """
    Add property data to an existing figure.

    args:
        df (dataframe):
            dataframe containing data to be plotted
        figinfo (dictionary cotnaining fig, ax and other info for plot):

    kwargs:

        xoffset (real or string):
            amount to subtract from x value for plotting (default is 0)
            'zero' means that the data are offset so that the minimum val is 0
        fmt (string):
            Format sting: Default is '+'   .
        label (string):
            label for plots.  Used to generate legend.  Default is ''
        plotdrho (Boolean):
            Switch to plot mass data or not. Default is True
        err_plot
            True if you want to include errorbars (default is False)

    returns:
        Nothing is returned.  The function just updates an existing axis.
    """

    fmt=kwargs.get('fmt', '+')
    label=kwargs.get('label', '')
    xoffset=kwargs.get('xoffset', 0)
    err_plot = kwargs.get('err_plot', False)
    xunit = figinfo['info']['xunit']

    if xunit == 's':
        xvals=df['t']
    elif xunit == 'min':
        xvals=df['t']/60
    elif xunit == 'hr':
        xvals=df['t']/3600
    elif xunit == 'day':
        xvals=df['t']/(24*3600)
    elif xunit == 'temp':
        xvals=df['temp']
    elif xunit == 'index':
        xvals=df.index
    else:
        xvals=df[xunit]
        
    if xoffset == 'zero':
        xoffset = min(xvals)
    
    xvals = xvals - xoffset
                
    plots = figinfo['info']['plots']
    ax = figinfo['ax']
    num_plots = len(plots)
    
    # now make all of the plots
    for p in np.arange(num_plots):
        if plots[p] == 'grho3' or plots[p] == 'grho3_lin': 
            xdata = xvals
            ydata = df['grho3']/1000
            yerr = df['grho3_err']/1000

        elif plots[p] == 'phi':
            xdata = xvals
            ydata = df['phi']
            yerr = df['phi_err']
                
        elif plots[p] == 'drho':
            xdata = xvals
            ydata = 1000*df['drho']
            yerr = 1000*df['drho_err']
      
        elif plots[p] == 'vgp' or plots[p] == 'vgp_lin':
            xdata = df['grho3']/1000
            ydata = df['phi']
            
        elif plots[p] == 'jdp':
            xdata  = xvals
            ydata = (1000/df['grho3'])*np.sin(df['phi']*np.pi/180) 
            
        elif plots[p] == 'temp':
            xdata  = xvals
            ydata = df['temp']    
            
        elif plots[p] == 't':
            xdata  = xvals
            ydata = df['t']  
        
        else:
            print('not a recognized plot type ('+plots[p]+')')
            sys.exit()
                
        if err_plot:
            ax[0, p].errorbar(xdata, ydata, fmt=fmt, yerr=yerr, label=label)
        else:
            ax[0, p].plot(xdata, ydata, fmt, label=label)
            
        if plots[p] == 'vgp':
                ax[0, p].set_xscale('log')
        if plots[p] == 'grho3':
                ax[0, p].set_yscale('log')
                

def read_xlsx(infile, **kwargs):
    """
    Create data frame from.xlsx file output by RheoQCM.

    args:
        infile (string):
            full name of input  .xlsx file

    kwargs:
        restrict_to_marked (list):
            List of frequencies that must be marked in order to be included.
            Default is [], so that we include everything.

        film_channel (string):
            sheet for data:  'S_channel' by default

        ref_channel (string)
            Source for reference (bare crystal) frequency and dissipation,
            - 'R_channel'  ('R_channel' sheet from xlsx file) (default)
            - 'S_channel'  ('S_channel' sheet from xlsx file)
            - 'S_reference'  ('S_reference' sheet from xlsx file)
            - 'R_reference'  ('R_channel' sheet from xlsx file)
            - 'self'  (read delf and delg read directly from the data channel.)
            - 'T_coef' - Taken directly from T_coef dictionary

        ref_idx (numpy array):
            index values to include in reference determination
            - default is 'all', which takes everything

        film_idx (numpy array)
            index values to include for film data
            default is 'all' which takes everthing

        T_coef (dictionary):
            Temperature coefficients for reference temp. shift
            - default values used if not specified

        Tref: (numeric)
            Temperature at which reference frequency shift was determined
            - default is 22C

        T_coef_plots (Boolean):  set to True to plot temp. dependent f and g for ref.
            - default is True

        T_shift (dictionary): shifts added to reference values
            - default is {1:0, 3:0, 5:0}

        nvals (list): harmonics to include
            - default is [1, 3, 5]


    returns:
        df:
            Input data converted to dataframe
    """

    restrict_to_marked=kwargs.get('restrict_to_marked', [])
    film_channel=kwargs.get('film_channel', 'S_channel')
    film_idx=kwargs.get('film_idx', 'all')
    ref_channel=kwargs.get('ref_channel', 'R_channel')
    ref_idx=kwargs.get('ref_idx', 'all')
    T_coef_plots=kwargs.get('T_coef_plots', True)
    nvals=kwargs.get('nvals', [1, 3, 5])

    Tref=kwargs.get('Tref', 22)
    # specify default bare crystal temperature coefficients
    T_coef=kwargs.get('T_coef', T_coef_default)

    # read shifts that account for changes from stress levels applied
    # to different sample holders
    T_shift=kwargs.get('T_shift', {1: 0, 3: 0, 5: 0})


    df=pd.read_excel(infile, sheet_name=film_channel, header=0)
    if type(film_idx) != str:
        df=df[df.index.isin(film_idx)]

    df['keep_row']=1  # keep all rows unless we are told to check for specific marks
    for n in restrict_to_marked:
        df['keep_row']=df['keep_row']*df['mark'+str(n)]

    # delete rows we don't want to keep
    df=df[df.keep_row == 1]  # Delete all rows that are not appropriately marked

    # now sort out which columns we want to keep in the dataframe
    keep_column=['t']
    for n in nvals:
        keep_column.append(n)

    # keep the temperature column if it exists
    if 'temp' in df.keys():
        keep_column.append('temp')

    # add each of the values of delfstar
    if ref_channel == 'T_coef':
        T_coef_plots=False
        for n in nvals:
            ref_f=np.polyval(T_coef['f'][n], df['temp'])
            ref_g=np.polyval(T_coef['g'][n], df['temp'])
            fstar_ref=ref_f+1j*ref_g
            fstar=df['f'+str(n)] + 1j*df['g'+str(n)]
            df[n]=fstar - fstar_ref - T_shift[n]  # -AS

    elif ref_channel == 'self':
        # this is the standard read protocol, with delf and delg already in
        # the .xlsx file
        for n in nvals:
            df[n]=df['delf'+str(n)] + 1j*df['delg'+str(n)
                                ].round(1) - T_shift[n]  # -AS

    else:
        # here we need to obtain T_coef from the info in the ref. channel
        df_ref=pd.read_excel(infile, sheet_name=ref_channel, header=0)
        if type(ref_idx) != str:
            df_ref=df_ref[df_ref.index.isin(ref_idx)]
        var=['f', 'g']

        # if no temperature is listed or a specific reference temperature
        # is given we just average the values
        if ('temp' not in df_ref.keys()) or (df_ref.temp.isnull().values.all()):
            for k in np.arange(len(nvals)):
                for p in [0, 1]:
                    # get the reference values and plot them
                    ref_val=df_ref[var[p]+str(nvals[k])].mean()

                    # write the film and reference values to the data frame
                    df[var[p]+str(nvals[k])+'_dat']=df[var[p]+str(nvals[k])]
                    df[var[p]+str(nvals[k])+'_ref']=ref_val

                    # adjust temperature coefficient to get correct value
                    # at ref temp
                    T_coef[var[p]][nvals[k]][3]=(T_coef[var[p]][nvals[k]][3] +
                            ref_val - np.polyval(T_coef[var[p]][nvals[k]], Tref))

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

                    # get the reference values and plot them
                    ref_vals=df_ref[var[p]+str(nvals[k])]

                    # make the fitting function
                    T_coef[var[p]][nvals[k]]=np.polyfit(temp, ref_vals, 3)
                    
                    # plot the data if fit was not obtained
                    if np.isnan(T_coef[var[p]][nvals[k]]).any():
                        fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
                        ax.plot(temp, ref_vals)
                        ax.set_xlabel(r'$T$ $^\circ$C')
                        ax.set_ylabel(var[p]+str(nvals[k]))
                        print('Temp. coefficients could not be obtained - see plot')
                        sys.exit()
                                                                        
                    # write the film and reference values to the data frame
                    df[var[p]+str(nvals[k])+'_dat']=df[var[p]+str(nvals[k])]
                    df[var[p]+str(nvals[k])+'_ref']=(
                         np.polyval(T_coef[var[p]][nvals[k]], df['temp']))

        for k in np.arange(len(nvals)):
            # now write values delfstar to the dataframe
            df[nvals[k]]=(df['f'+str(nvals[k])+'_dat'] -
                          df['f'+str(nvals[k])+'_ref'] +
                      1j*(df['g'+str(nvals[k])+'_dat'] -
                          df['g'+str(nvals[k])+'_ref'])-T_shift[nvals[k]]).round(1)  # -AS

            # add absolute frequency and reference values to dataframe
            keep_column.append('f'+str(nvals[k])+'_dat')
            keep_column.append('f'+str(nvals[k])+'_ref')
            keep_column.append('g'+str(nvals[k])+'_dat')
            keep_column.append('g'+str(nvals[k])+'_ref')

    # add the constant applied shift to the reference values to the dataframe -AS
    for n in nvals:
        df[str(n)+'_refshift']=T_shift[n]
        keep_column.append(str(n)+'_refshift')

    if T_coef_plots and ref_channel != 'self' and len(df_ref.temp.unique()) > 1:
        T_range=[df['temp'].min(), df['temp'].max()]
        T_ref_range = [df_ref['temp'].min(), df_ref['temp'].max()]
        # create a filename for saving the reference temperature data
        filename = os.path.splitext(infile)[0]+'_Tref.pdf'
        plot_bare_tempshift(df_ref, T_coef, Tref, nvals, T_range, filename)
        if T_range[0] < T_ref_range[0] or T_range[1] > T_ref_range[1]:
            print ('experimental data outside of reference range')
            sys.exit()

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
            ref_vals=bare_tempshift(temp_fit, T_coef, Tref, nvals[k])[var[p]]
            ax[p, k].plot(temp_fit, ref_vals, 'o', label='fit')

            # set axis labels and plot titles
            ax[p, k].set_xlabel(r'$T$ ($^\circ$C)')
            ax[p, k].set_ylabel(ylabel[p])
            ax[p, k].set_title('n='+str(nvals[k]))
            ax[p, k].legend()
            ymin=np.min([meas_vals.min(), ref_vals.min()])
            ymax=np.max([meas_vals.max(), ref_vals.max()])
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
            input dataframe to consider

    kwargs:
        filename (string):
            Filename for pdf.  Also used for window title
        numxy (int):
            number of grid points in x and y (default is 100)
        numz (int):
            number of z levels (default is 100)
        philim (list of two real numbers):
            min and max of phase angle (default is [0, 90], 'auto' scales to expt data)
        dlim (list of two real numbers):
            min and max of d/lambda (default is [0, 0.5], 'auto' scales to expt data)
        nplot (list of integers):
            list of harmonics to plot (default is [1,3,5])
        ratios (Boolean):
            plot rh and rd if True, delf, delg otherwise (default is False)
        autoscale (Boolean):
            auto scale z values to min and max of calculated values if True
            default is False
        label ('string'): dataframe key to use to label individual points in
            solution check (default is 'temp')
        plot_solutions (Boolean): True if we want to plot the solution checks
            for each point (default = False)
        plot_interval (integer): interval between successive solution plots
            (default is 1)
        xunit (string):
            Units for x data.  Default is 'dlam', function currently also handles
            's', 'min', 'hr', 'day', 'temp'
        xoffset (real):
            Value subtracted from x data (default is zero)
            
    Returns:
        fig, ax for solutioncheck figure

    '''

    from pylab import meshgrid
    numxy=kwargs.get('numxy', 100)
    numz=kwargs.get('numz', 200)
    philim=kwargs.get('philim', [0.001, 90])
    dlim=kwargs.get('dlim', [0.001, 0.5])
    # having d of 0 causes some problems.  Change lower limit to be at least 0.001
    dlim[0] = max(dlim[0], 0.001)
    nplot=kwargs.get('nplot', [1, 3, 5])
    ratios=kwargs.get('ratios', False)
    autoscale=kwargs.get('autoscale', False)
    plot_solutions=kwargs.get('plot_solutions', False)
    plot_interval = kwargs.get('plot_interval', 1)
    idxmin=df.index[0]
    calc=df['calc'][idxmin]
    filename=kwargs.get('filename', 'solution_check.pdf')
    xunit=kwargs.get('xunit', 'dlam')
    xoffset = kwargs.get('xoffset', 0)

    # set up x labels for plots of actual and back-calculated shifts
    if xunit == 's':
        xlabel='$t$ (s)'
        df.loc[:,'xvals']=df.loc[:,'t']-xoffset
    elif xunit == 'min':
        xlabel='$t$ (min.)'
        df.loc[:,'xvals']=df.loc[:,'t']/60-xoffset
    elif xunit == 'hr':
        xlabel='$t$ (hr)'
        df.loc[:,'xvals']=df.loc[:,'t']/3600-xoffset
    elif xunit == 'day':
        xlabel='$t$ (days)'
        df.loc[:,'xvals']=df.loc[:, 't']/(24*3600)-xoffset
    elif xunit == 'temp':
        xlabel=r'$T$ ($^\circ$C)'
        df.loc[:, 'xvals']=df.loc[:,'temp']-xoffset
    else:
        xlabel=r'$d/\lambda_3$'
        df.loc[:,'xvals'] = df.loc[:,'dlam3']
        
    # make the axes
    fig, ax=plt.subplots(2, 2, figsize=(10, 6), sharex=False, sharey=False,
                           num=filename, constrained_layout=True)

    # make meshgrid for contour
    phi=np.linspace(philim[0], philim[1], numxy)
    dlam=np.linspace(dlim[0], dlim[1], numxy)
    DLAM, PHI=meshgrid(dlam, phi)

    # need to use n=3 in this calculation, since
    # normdelfstar assumes third harmonic in its definition

    def Zfunction(x, y):
        # function used for calculating the z values
        # this Z is the value plotted in the contour plot and NOT the impedance
        drho=df['drho'][idxmin]
        grho3=grho_from_dlam(3, drho, x, y)
        fnorm=normdelf_bulk(3, x, y)
        gnorm=normdelg_bulk(3, x, y)
        if ratios:
            n1=int(calc[0])
            n2=int(calc[1])
            n3=int(calc[2])
            Z1=np.real(normdelfstar(n2, x, y))/np.real(normdelfstar(n1, x, y))
            Z2=-np.imag(normdelfstar(n3, x, y))/np.real(normdelfstar(n3, x, y))
        else:
            delfstar=sauerbreyf(1, drho)*normdelfstar(3, x, y)
            Z1=np.real(delfstar)
            Z2=np.imag(delfstar)
        return Z1, Z2, drho, grho3, fnorm, gnorm

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
            max2=5000

    levels1=np.linspace(min1, max1, numz)
    levels2=np.linspace(min2, max2, numz)

    contour1=ax[0, 0].contourf(DLAM, PHI, Z1, levels=levels1,
                              cmap='rainbow')
    contour2=ax[0, 1].contourf(DLAM, PHI, Z2, levels=levels2,
                              cmap='rainbow')

    fig.colorbar(contour1, ax=ax[0, 0])
    fig.colorbar(contour2, ax=ax[0, 1])

    # set label of ax[1]
    ax[0, 0].set_xlabel(r'$d/\lambda_3$')
    ax[0, 0].set_ylabel(r'$\Phi$ ($\degree$)')
    ax[1, 0].set_ylabel(r'$\Delta f/n$ (Hz)')

    ax[0, 1].set_xlabel(r'$d/\lambda_3$')
    ax[0, 1].set_ylabel(r'$\Phi$ ($\degree$)')
    ax[1, 1].set_ylabel(r'$\Delta\Gamma/n$ (Hz)')

    # add titles
    if ratios:
        ax[0, 0].set_title(calc + r': $r_h$')
        ax[0, 1].set_title(calc + r': $r_d$')
    else:
        ax[0, 0].set_title(calc + r': $\Delta f /n$ (Hz)')
        ax[0, 1].set_title(calc + r': $\Delta\Gamma /n$ (Hz)')

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

    # now add the experimental data
    # variable to keep track of differentplots

    col={1: 'C0', 3: 'C1', 5: 'C2'}

    for n in nplot:
        nstr=str(n)+' (expt)'
        # compare experimental and calculated frequency fits
        ax[1, 0].plot(df['xvals'], np.real(df['df_expt'+str(n)])/n, '+', color=col[n],
                     label='n='+nstr)
        ax[1, 0].plot(df['xvals'], np.real(df['df_calc'+str(n)])/n, '-', color=col[n])

        # now compare experimental and calculated dissipation
        ax[1, 1].plot(df['xvals'], np.imag(df['df_expt'+str(n)])/n, '+', color=col[n],
                     label='n='+nstr)
        ax[1, 1].plot(df['xvals'], np.imag(df['df_calc'+str(n)])/n, '-', color=col[n])

    # add values to contour plots for n=3
    ax[0, 0].plot(df['dlam3'], df['phi'], 'k-')
    ax[0, 1].plot(df['dlam3'], df['phi'], 'k-')

    for k in [0, 1]:
        ax[1, k].legend()
        ax[0, k].format_coord=fmt
        ax[1, k].set_xlabel(xlabel)

    # reset axis limits
    ax[0, 0].set_xlim(dlim)
    ax[0, 1].set_xlim(dlim)
    ax[0, 0].set_ylim(philim)
    ax[0, 1].set_ylim(philim)

    # create a PdfPages object - one solution check per page
    pdf=PdfPages(filename)
    
    # we only take every nth row, where n = plot_interval
    df_plot = df.iloc[::plot_interval, :]
    if plot_solutions:
        idxnum = 0 # keeps track of the fact that we don't always start from idx=0
        for idx, row in df_plot.iterrows():
            idxnum = idxnum + 1
            curves={}

            # indicate where the solution is being taken
            print('writing solution '+str(idxnum)+' of '+str(len(df_plot)))
            for k in [0, 1]:
                curves[0+k]=ax[0, k].plot(row['dlam3'], row['phi'], 'kx', 
                                markersize=14, label = 'x='+str(row['xvals']))

            for p in np.arange(len(nplot)):
                n = nplot[p]
                curves[2+2*p]=ax[1, 0].plot(row['xvals'],
                            np.real(row['df_expt'+str(n)])/n, 'x',
                             markersize=14, color=col[n])
                curves[3+2*p]=ax[1, 1].plot(row['xvals'], 
                            np.imag(row['df_expt'+str(n)])/n, 'x',
                             markersize=14, color=col[n])

            # now plot the lines for the solution
            rh=rhcalc(calc, row['dlam3'], row['phi'])
            rd=rdcalc(calc, row['dlam3'], row['phi'])

            def solutions(phi, guess):
                def ftosolve_rh(d):
                    return rhcalc(calc, d, phi)-rh

                def ftosolve_rd(d):
                    return rdcalc(calc, d, phi)-rd

                soln_rh=optimize.least_squares(ftosolve_rh, guess[0], bounds=dlim)
                soln_rd=optimize.least_squares(ftosolve_rd, guess[1], bounds=dlim)
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
            for k in [0, 1]:
                curves[2+2*len(nplot)+k]=ax[0, k].plot(dcalc['d_rh'], dcalc['phi'], 'w-')
                curves[4+2*len(nplot)+k]=ax[0, k].plot(dcalc['d_rd'], dcalc['phi'], 'w--')
                
            # add titles
            if ratios:
                ax[0, 0].set_title(calc + r': $r_h$='+f'{rh:.4f}')
                ax[0, 1].set_title(calc + r': $r_d$='+f'{rd:.4f}')
            else:
                ax[0, 0].set_title(calc + r': $\Delta f /n$ (Hz)')
                ax[0, 1].set_title(calc + r': $\Delta\Gamma /n$ (Hz)')
        
            pdf.savefig()
            
            for k in np.arange(6+2*len(nplot)):
                curves[k][0].remove()
    else:
        pdf.savefig()
                
    pdf.close()
    return fig, ax
