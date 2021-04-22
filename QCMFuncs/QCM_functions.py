#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:19:59 2018

@author: ken
"""

import numpy as np
import scipy.optimize as optimize
from scipy.interpolate import InterpolatedUnivariateSpline
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
g0_default = 50 # Half bandwidth of unloaed resonator (intrinsic dissipation on crystalline quartz)
err_frac_default = 3e-2 # error in f or gamma as a fraction of gamma

electrode_default = {'drho':2.8e-3, 'grho3':3.0e14, 'phi':0}


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
    values = np.asarray(values).reshape(1, -1)[0,:]
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

    g=g0 + np.imag(delfstar)
    
    if 'overlayer' in layers:
        delg = calc_delfstar(n, {'film':layers['overlayer']}, 
                                             calctype='SLA')
        g=g+delg
            
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
    phi=np.angle(grhostar, deg=True)
    grhon=abs(grhostar)
    grho3=grhon*(3/n)**(phi/90)
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
    #decay length multiplied by density
    return -Zq*abs(delfstar[n])**2/(2*n*f1**2*delfstar[n].real)


def calc_D(n, props, delfstar,calctype):
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
                       props,calctype)


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
        greal=grho3*np.cos(np.radians(props['phi']))
        gimag=(n/3)*np.sin(np.radians(props['phi']))
        grhostar=(gimag**2+greal**2)**(0.5)*(np.exp(1j*
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
    return f1*1j/(np.pi*Zq)*ZL


def calc_ZL(n, layers, delfstar, calctype):
    """
    Calculate complex load impendance for stack of layers.
    args:
        n (int):
            Harmonic of interest.
            
        layers (dictionary):
            Dictionary of material dictionaries specifying the properites of
            each layer. These dictionaries are labeled from 1 to N, with 1
            being the layer in contact with the QCM.  Each dictionary must 
            include values for 'grho3, 'phi' and 'drho'.
            
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
        Z[i] = zstar_bulk(n, layers[i],calctype)
        D[i] = calc_D(n, layers[i], delfstar, calctype)
        L[i] = np.array([[np.cos(D[i])+1j*np.sin(D[i]), 0],
                 [0, np.cos(D[i])-1j*np.sin(D[i])]])

    # get the terminal matrix from the properties of the last layer
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

    rstar = uvec[1,0]/uvec[0,0]
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
       
    returns:
        delfstar (complex):
            Complex frequency shift (Hz).
    """
    
    calctype = kwargs.get('calctype', 'SLA')
    if not layers: # layers is empty {}
        return np.nan

    # there is data
    if 'overlayer' in layers:
        ZL = calc_ZL(n, {1:layers['film'], 2:layers['overlayer']},
                         0,calctype)
        ZL_ref = calc_ZL(n, {1:layers['overlayer']}, 0, calctype)
        del_ZL = ZL-ZL_ref
    else:
        del_ZL = calc_ZL(n, {1:layers['film']}, 0, calctype)

    if calctype != 'LL':
        # use the small load approximation in all cases where calctype
        # is not explicitly set to 'LL'
        return calc_delfstar_sla(del_ZL)

    else:
        # this is the most general calculation
        # use defaut electrode if it's not specified
        if 'electrode' not in layers:
            layers['electrode'] = electrode_default

        layers_all = {1:layers['electrode'], 2:layers['film']}
        layers_ref = {1:layers['electrode']}
        if 'overlayer' in layers:
            layers_all[3]=layers['overlayer']
            layers_ref[2] = layers['overlayer']

        ZL_all = calc_ZL(n, layers_all, 0, calctype)
        delfstar_sla_all = calc_delfstar_sla(ZL_all)
        ZL_ref = calc_ZL(n, layers_ref, 0,calctype)
        delfstar_sla_ref = calc_delfstar_sla(ZL_ref)


        def solve_Zmot(x):
            delfstar = x[0] + 1j*x[1]
            Zmot = calc_Zmot(n,  layers_all, delfstar, calctype)
            return [Zmot.real, Zmot.imag]

        sol = optimize.root(solve_Zmot, [delfstar_sla_all.real,
                                         delfstar_sla_all.imag])
        dfc = sol.x[0] + 1j* sol.x[1]

        def solve_Zmot_ref(x):
            delfstar = x[0] + 1j*x[1]
            Zmot = calc_Zmot(n,  layers_ref, delfstar, calctype)
            return [Zmot.real, Zmot.imag]

        sol = optimize.root(solve_Zmot_ref, [delfstar_sla_ref.real,
                                             delfstar_sla_ref.imag])
        dfc_ref = sol.x[0] + 1j* sol.x[1]
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
    om = 2 * np.pi *(n*f1 + delfstar)
    Zqc = Zq * (1 + 1j*2*g0/(n*f1))

    Dq = om*drho_q/Zq
    secterm = -1j*Zqc/np.sin(Dq)
    ZL = calc_ZL(n, layers, delfstar,calctype)
    # eq. 4.5.9 in Diethelm book
    thirdterm = ((1j*Zqc*np.tan(Dq/2))**-1 + (1j*Zqc*np.tan(Dq/2) + ZL)**-1)**-1
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
    grho=grho3*(n/3) ** (phi/90)
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

    return np.real(2*np.tan(2*np.pi*dlam(n, dlam3, phi)*
                          (1-1j*np.tan(np.deg2rad(phi/2)))) / \
            (np.sin(np.deg2rad(phi))*(1-1j*np.tan(np.deg2rad(phi/2)))))
            

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

    return -np.imag(np.tan(2*np.pi*dlam(n, dlam3, phi)*
                           (1-1j*np.tan(np.deg2rad(phi/2)))) / \
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
    return -(normdelfstar(calc[2], dlam3, phi).imag / \
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
                
    returns:
        df_out (dataframe):  
            dataframe with properties added, deleting rows with any NaN values \
            that didn't allow calculation to be performed
            
    """
    
    err_frac=kwargs.get('err_frac', 0)
    df_in = delfstar.T  # transpose the input dataframe
    if 'overlayer' in kwargs.keys():
        layers={'overlayer':kwargs['overlayer']}
    else:
        layers={}

    nplot = []
    for n in [1,3,5,7,9]:
        if n in df_in.index:
            nplot = nplot + [n]
    calctype = kwargs.get('calctype', 'SLA')
    drho = kwargs.get('drho', 0)
    newtonian = kwargs.get('newtonian', False)

    
    # set upper and lower bounds
    lb = kwargs.get('lb', [1e5, 0, 0])
    ub = kwargs.get('ub', [1e13, 90, 3e-2])
    lb = np.array(lb)  # lower bounds drho, grho3, phi
    ub = np.array(ub)  # upper bounds drho, grho3, phi
    
    if drho!=0:
        fixed_drho=True
        n1 = int(calc[0]); n2 = int(calc[0]); n3 = int(calc[0])

        if 'guess' in kwargs.keys():
            guess=kwargs['guess']
            grho3, phi = guess['grho3'], guess['phi']
        else:
            grho3=1e11
            phi=45
            n1=n3 # fixed thickness uses delta f and delta gamma from same harmonic
    else:
        fixed_drho=False
        n1 = int(calc[0]); n2 = int(calc[1]); n3 = int(calc[2])

        if 'guess' in kwargs.keys():
            guess=kwargs['guess']
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
    props = {} # properties for each layer
    for element in out_rows:
        data[element] = np.array([])
    
    for n in nplot:
        data['df_expt'+str(n)]=np.array([])
        data['df_calc'+str(n)]=np.array([])
    
    # obtain the solution, using either the SLA or LL methods
    for i in df_in.columns:
        if np.isnan([df_in[i][n1], df_in[i][n2], df_in[i][n3]]).any():
            continue
        if not fixed_drho:
            def ftosolve(x):
                layers['film'] = {'grho3':x[0], 'phi':x[1], 'drho':x[2]}
                return ([calc_delfstar(n1, layers, calctype=calctype).real-
                         df_in[i][n1].real,
                         calc_delfstar(n2, layers, calctype=calctype).real-
                         df_in[i][n2].real,
                         calc_delfstar(n3, layers, calctype=calctype).imag-
                         df_in[i][n3].imag])
        else:
            def ftosolve(x):
                layers['film'] = {'drho':drho, 'grho3':x[0], 'phi':x[1]}
                if not newtonian:
                    return ([calc_delfstar(n1, layers, calctype=calctype).real-
                             df_in[i][n1].real,
                             calc_delfstar(n1, layers, calctype=calctype).imag-
                             df_in[i][n1].imag])
                else:  # set frequency shift equal to dissipation shift in this case
                    return ([calc_delfstar(n1, layers, calctype=calctype).real+
                             df_in[i][n1].imag,
                             calc_delfstar(n1, layers, calctype=calctype).imag-
                             df_in[i][n1].imag])

        # initialize the output uncertainties
        err = {}
        soln = optimize.least_squares(ftosolve, x0, bounds=(lb, ub))
        grho3 = soln['x'][0]
        phi = soln['x'][1]
        print(soln['x'])

        if not fixed_drho:
            drho = soln['x'][2]

        layers['film'] = {'drho':drho, 'grho3':grho3, 'phi':phi}
        dlam3 = calc_dlam(3, layers['film'])
        jac = soln['jac']

        try:
            deriv = np.linalg.inv(jac)
        except:
            deriv = np.zeros([len(x0),len(x0)])

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
            err[p]=0
            for k in np.arange(len(x0)):
                err[p] = err[p]+(deriv[p, k]*delfstar_err[k])**2
            err[p] = np.sqrt(err[p])

        if fixed_drho:
            err[2]=np.nan

        # now back calculate delfstar from the solution
        delfstar_calc = {}

        # add experimental and calculated values to the dictionary
        for n in nplot:
            delfstar_calc[n] = calc_delfstar(n, layers, calctype=calctype)
            data['df_calc'+str(n)]=(np.append(data['df_calc'+str(n)], 
                                             round(delfstar_calc[n],1)))
            try:
                data['df_expt'+str(n)]=(np.append(data['df_expt'+str(n)], 
                                             delfstar[n][i]))
            except:
                data['df_expt'+str(n)]=np.append(data['df_expt'+str(n)], 'nan')
                
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
    fig, ax = make_prop_axes(figsize=(12,3))
    idx = kwargs.get('idx', 0) # specify specific point to use
    npts= kwargs.get('npts', 10)
    err_frac = kwargs.get('err_frac', err_frac_default)
    err_range=kwargs.get('err_range', 1) # >1 to extend beyond calculated err
    err_range=max(1, err_range) 
    calctype=df_in['calctype'][idx]
    calc=df_in['calc'][idx]
    deriv=df_in['deriv'][idx]    
    delfstar_err = df_in['delfstar_err'][idx]
    guess = {'grho3':df_in['grho3'][idx],
             'phi':df_in['phi'][idx],
             'drho':df_in['drho'][idx]}

    delfstar_0={}
    
    for nstr in list(set(calc)):
        n=int(nstr)
        delfstar_0[n]=df_in[n][idx]
    
    # now generate series of delfstar values based on the errors
    delfstar_del={}
    
    # set some parameters for the plots
    mult=np.array([1,1,1j], dtype=complex) # frequency or dissipation shift
    forg={0:'f', 1:'f', 2:'$\Gamma$'}
    prop_type = {0:'grho3', 1:'phi', 2:'drho'}
    scale_factor = {0:0.001, 1:1, 2:1000}
    marker = {0:'+', 1:'s', 2:'o'}
    
    # intialize values of delfstar
    for k in [0,1,2]:
        delfstar_del[k]={}
        for n in [3,5]:
            delfstar_del[k][n]=np.ones(npts)*delfstar_0[n]
            
    # adjust values of delfstar and calculate properties
    for k in [0,1,2]:
        ax[k].set_xlabel(r'$(X-X_0)/X^{err}$')
        n = int(calc[k])
        err = err_range*delfstar_err[k]
        delta=np.linspace(-err, err, npts)
        delfstar_del[k][n]=delfstar_del[k][n]+delta*mult[k]
        delfstar_df=pd.DataFrame.from_dict(delfstar_del[k])
        props = solve_for_props(delfstar_df, calc=calc,
                                          calctype=calctype, guess=guess)
        # make the property plots
        for p in [0,1,2]:
            ax[p].plot(delta/delfstar_err[k], props[prop_type[p]]*scale_factor[p], 
                       marker=marker[k], linestyle='none',
                       label=r'X='+forg[k]+'$_'+str(n)+'$')
            
     
    # reset color cycles so dervitave plots match the color scheme
    for p in [0, 1, 2]:
        ax[p].set_prop_cycle(None)
        for k in [0, 1, 2]:
            err = delfstar_err[k]
            xdata = np.array([-err_range, 0, err_range])
            ydata = (np.ones(3)*df_in[prop_type[p]][idx]*scale_factor[p]+
                     err*xdata*scale_factor[p]*deriv[p][k])
            ax[p].plot(xdata, ydata, '-')
        
        # now add the originally calculated error
            err0=df_in[prop_type[p]+'_err']
            ax[p].errorbar(0, df_in[prop_type[p]][idx]*scale_factor[p],
                           yerr=err0*scale_factor[p], color='k')
            
    ax[0].legend(loc='center', bbox_to_anchor=(-0.5, 0, 0, 1))
    sub_string={}
    for k in [0, 1, 2]:
        n=calc[k]
        sub_string[k]=(forg[k]+'$_'+n+'^{err}=$'+
                         f'{delfstar_err[k]:.0f}'+' Hz')
    title_string=(sub_string[0]+'; '+sub_string[1]+'; '+sub_string[2]+
                  '; '+'err_frac='+str(err_frac))
    fig.suptitle(r''+title_string)
    fig.tight_layout()
    return fig, ax


def make_prop_axes(**kwargs):
    """
    Make a blank property figure.

    kwargs:
        filmtype (string):
            - 'thin': standard thin film solution, with three axes
            - 'bulk': bulk solution with two axes
        xlabel (string):
            label for x axes (default is 'index')     .
        titles (dictionary):
            titles for axes (keys are 0, 1, 2)
        num (string):
            window title (default is 'property fig')
            
    returns:
        fig: 
            Figure containing property plots.
        ax:
            axes of the figure.
    """

    filmtype = kwargs.get('filmtype','thin')
    num = kwargs.get('num','property fig')
    xlabel = kwargs.get('xlabel', 'index')
    titles = kwargs.get('titles', {0:'(a)', 1:'(b)', 2:'(c)'})

    if filmtype != 'bulk':
        figsize=kwargs.get('figsize',(9,3))
        fig, ax = plt.subplots(1,3, figsize=figsize, num=num)
        ax[2].set_ylabel(r'$d\rho$ ($\mu$m$\cdot$g/cm$^3$)')
    else:
        figsize=kwargs.get('figsize',(6,3))
        fig, ax = plt.subplots(1,2, figsize=figsize, num=num)

    ax[0].set_ylabel(r'$|G_3^*|\rho$ (Pa $\cdot$ g/cm$^3$)')
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    ax[1].set_ylabel(r'$\phi$ (deg.)')

    # set xlabel 
    for i in np.arange(len(ax)):
        ax[i].set_xlabel(xlabel)
        ax[i].set_title(titles[i])
    fig.tight_layout()
    return fig, ax


def prop_plots(df, ax, **kwargs):
    """
    Add property data to an existing figure.
    
    args:
        df (dataframe):
            dataframe containing data to be plotted
        ax (axes handle):
            axes to use for plotting

    kwargs:
        xunit (string):
            Units for x data.  Default is 'index', function currently handles
            's', 'min', 'hr', 'day', 'temp'
        fmt (string):
            Format sting: Default is '+'   .
        legend (string):
            Legend for plots.  Default is 'none'
        plotdrho (Boolean):
            Switch to plot mass data or not. Default is True
        err_plot
            True if you want to include errorbars (default is false)
            
    returns:
        Nothing is returned.  The function just updates an existing axis.
    """
    xunit=kwargs.get('xunit', 'index')
    fmt=kwargs.get('fmt','+')
    legend=kwargs.get('legend','none')
    plotdrho=kwargs.get('plotdrho', True)
     
    if xunit =='s':
        xdata = df['t']
        xlabel ='$t$ (s)'
    elif xunit == 'min':
        xdata = df['t']/60
        xlabel ='$t$ (min.)'
    elif xunit == 'hr':
        xdata = df['t']/3600
        xlabel ='$t$ (hr)'
    elif xunit == 'day':
        xdata = df['t']/(24*3600)
        xlabel ='$t$ (days)'
    elif xunit == 'temp':
        xdata = df['temp']
        xlabel = r'$T$ ($^\circ$C)'
    
    else:
        xdata = df.index
        xlabel ='index'

    ax[0].errorbar(xdata, df['grho3']/1000, fmt=fmt, yerr=df['grho3_err']/1000,
                                                    label=legend)
    ax[1].errorbar(xdata, df['phi'], fmt=fmt, yerr=df['phi_err'], label=legend)
    
    if len(ax)==3 and plotdrho:
        ax[2].errorbar(xdata, df['drho']*1000, fmt=fmt, yerr=df['drho_err']*1000,
                      label=legend)
    
    for k in np.arange(len(ax)):
        ax[k].set_xlabel(xlabel)
        ax[k].set_yscale('linear')
        if legend!='none':
            ax[k].legend()
 
            
def vgp_plot(df, ax, **kwargs):
    """
    Add property data to existing van Gurp-Palmen axes.
    
    args:
        df (dataframe):
            dataframe containing data to be plotted
        ax (axes handle):
            axes to use for plotting

    kwargs:
        fmt (string):
            Format sting: Default is '+'   .
        legend (string):
            Legend for plots.  Default is 'none'
            
    returns:
        Nothing is returned.  The function just updates an existing axis.
    """
    fmt=kwargs.get('fmt','+')
    legend=kwargs.get('legend','none')
     
    ax.semilogx(df['grho3']/1000, df['phi'], fmt, label=legend)
  
    if legend!='none':
        ax.legend()
            

def check_plots(df, ax, nplot, **kwargs):
    """
    Add measured and recalculated complex frequency shifts to existing axes.
    
    args:
        df (dataframe):
            dataframe containing data to be plotted
        ax (axes handle):
            axes to use for plotting
        nplot (list of integers):
            harmonics to include on the plot

    kwargs:
        xunit (string):
            Units for x data.  Default is 'index', function currently handles
            's', 'min', 'hrs', 'day'
            
    returns:
        Nothing returned. The function just updates an existing axis set.
    """
    xunit=kwargs.get('xunit', 'index')
     
    if xunit =='s':
        xdata = df['t']
        xlabel ='$t$ (s)'
    elif xunit == 'min':
        xdata = df['t']/60
        xlabel ='$t$ (min.)'
    elif xunit == 'hr':
        xdata = df['t']/3600
        xlabel ='$t$ (hrs)'
    elif xunit == 'day':
        xdata = df['t']/(24*3600)
        xlabel ='$t$ (days)'
    else:
        xdata =xdata = df['index']
        xlabel ='index'
    
    col = {1:'C0', 3:'C1', 5:'C2', 7:'C3', 9:'C4'}
    # compare measured and calculated delfstar
    for n in nplot:
        ax[0].plot(xdata, -np.real(df['df_expt'+str(n)]),'+', color=col[n])
        ax[1].plot(xdata, np.imag(df['df_expt'+str(n)]),'+', color=col[n])
        ax[0].plot(xdata, -np.real(df['df_calc'+str(n)]),'-', color=col[n])
        ax[1].plot(xdata, np.imag(df['df_calc'+str(n)]),'-', color=col[n])
    
    for k in [0,1]:
        ax[k].set_xlabel(xlabel)
        ax[k].set_yscale('log')


def make_check_axes(**kwargs):
    """
    Make blank figure for plotting comparison of actual and recalculated 
    version os delfstar.

    kwargs:
        num (string):
            Window title default is 'delfstar check'.
        figsize (2 elment tuple of numbers):
            Figure size (default is (6,3))
            
    returns:
        fig: 
            Figure containing delfstar plots.
        ax:
            Axes of the figure.
    """
    # set up axes to compare measured and calculated delfstar values
    num = kwargs.get('num','delfstar check')
    figsize=kwargs.get('figsize',(6,3))
    fig, ax = plt.subplots(1,2, figsize=figsize, num=num)

    ax[0].set_ylabel(r'-$\Delta f$ (Hz))')
    ax[1].set_ylabel(r'$\Delta \Gamma$ (Hz)')

    # set xlabel to 'index' by default
    for i in [0, 1]:
        ax[i].set_xlabel('index')
    fig.tight_layout()
    return fig, ax


def make_delf_axes(**kwargs):
    """
    Make blank figure for plotting delfstar values.

    kwargs:
        num (string):
            Window title.
        figsize (2 elment tuple of numbers):
            Figure size (default is (6,3))
            
    returns:
        fig: 
            Figure containing delfstar plots.
        ax:
            Axes of the figure.
    """
    num = kwargs.get('num','delf fig')
    fig, ax = plt.subplots(1,2, figsize=(6,3), num=num)
    for i in [0, 1]:
        ax[i].set_xlabel(r'index')
    ax[0].set_ylabel(r'$\Delta f/n$ (kHz)')
    ax[1].set_ylabel(r'$\Delta \Gamma /n$ (kHz)')
    fig.tight_layout()
    return fig, ax


def make_vgp_axes(**kwargs):
    """
    Make blank figure for van Gurp-Palmen plot.

    kwargs:
        num (string):
            Window title.
        figsize (2 elment tuple of numbers):
            Figure size (default is (3,3))
            
    returns:
        fig: 
            Figure containing delfstar plots.
        ax:
            Axes of the figure.
    """
    figsize=kwargs.get('figsize', (4,3))
    num = kwargs.get('num','VGP plot')
    fig, ax = plt.subplots(1,1, figsize=figsize, num=num)
    ax.set_xlabel((r'$|G_3^*|\rho$ (Pa $\cdot$ g/cm$^3$)'))
    ax.set_ylabel(r'$\phi$ (deg.)')
    fig.tight_layout()
    return fig, ax

def get_bare_fstar(infile, **kwargs):
    """
    get the bare crystal ref. data at one specific temperature
    
    args:
        infile (string):
            full name of input .xlsx file
            
    kwargs:
        restrict_to_marked (list):
            List of frequencies that must be marked in order to be included.
            Default is [], so that we include everything.  We rerturn just
            the average value of all the rows read in
        
            
        ref_source (string):
            Channel for reference (bare crystal) frequency and dissipation,
            typically one of the following:
                - 'S_channel'
                - 'S_reference'
                - 'R-channel'
                - 'R-reference'                '
                           
    returns:
        df:  
            Input data converted to dataframe   
        
    """
    restrict_to_marked = kwargs.get('restrict_to_marked',[])
    ref_source = kwargs.get('ref_source', 'S_reference')

    df = pd.read_excel(infile, sheet_name = ref_source, header=0)
    
    # sort out which harmonics are included
    nvals = []
    for n in [1,3,5,7,9]:
        if 'f'+str(n) in df.columns:
            nvals=nvals+[n]

    df['keep_row']=1  # keep all rows unless we are told to check for specific marks
    for n in restrict_to_marked:
        df['keep_row'] = df['keep_row']*df['mark'+str(n)]

    # delete rows we don't want to keep
    df = df[df.keep_row==1] # Delete all rows that are not appropriately marked
    
    # now sort out which columns we want to keep in the dataframe
    keep_column=[]
    for n in nvals:
        keep_column.append('f'+str(n))
        keep_column.append('g'+str(n))

    df = df[keep_column] 
            
    return df.mean()



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
        
        data_channel (string):
            Channel for data:  'S' (default) or 'R'
            
        ref_source (string):
            Source for reference (bare crystal) frequency and dissipation,
              'self' (default) means we ignore it and read delf and delg
              directly from the data channel. Otherwise we specify one of 
              the following:
                - 'S_channel'  ('S-channel' sheet from xlsx file)
                - 'S_reference'  ('S-reference' sheet from xlsx file)
                - 'R-channel'  ('R-channel' sheet from xlsx file)
                - 'R-reference'  ('R-channel' sheet from xlsx file)  
                -  df_ref (series with df1, dg1, df3, dg3, ...;  in this
                           case Tref and Tf_coeff must also be specified,
                           if not equal to the defaults)
        
        ref_index (numpy array):
            index values to include in reference determination 
            default is 'all', which takes everything
        
        Tref:
            Temperature that df_ref corresponds to
            
        Tf_coeff: polynomial coefficients for bare crystal frequency shifts
                           
    returns:
        df:  
            Input data converted to dataframe           
    """
    
    restrict_to_marked = kwargs.get('restrict_to_marked',[])
    data_channel = kwargs.get('data_channel', 'S')
    data_name = data_channel+'_channel'
    ref_source = kwargs.get('ref_source', 'self')
    Tref = kwargs.get('Tref', 22)
    Tf_coeff = kwargs.get('Tf_coeff', 
                          {1:[0.00054625, 0.04338, 0.08075, 0],
                           3:[0.0017, -0.135, 8.9375, 0],
                           5:[0.002825, -0.22125, 15.375, 0]})

    df = pd.read_excel(infile, sheet_name=None, header=0)[data_name]
    
    # sort out which harmonics are included
    nvals = []
    for n in [1,3,5,7,9]:
        if 'f'+str(n) in df.columns:
            nvals=nvals+[n]

    df['keep_row']=1  # keep all rows unless we are told to check for specific marks
    for n in restrict_to_marked:
        df['keep_row'] = df['keep_row']*df['mark'+str(n)]

    # delete rows we don't want to keep
    df = df[df.keep_row==1] # Delete all rows that are not appropriately marked
    
    # now sort out which columns we want to keep in the dataframe
    keep_column=['t']
    for n in nvals:
        keep_column.append(n)
    
    # add each of the values of delfstar
    if isinstance(ref_source, pd.Series):
        for n in nvals:
            fref = (ref_source['f'+str(n)]+
                    bare_tempshift(Tf_coeff, df['temp'], Tref, n))
            gref = ref_source['g'+str(n)]
            fstar_ref = fref + 1j*gref
            fstar = df['f'+str(n)] + 1j*df['g'+str(n)]
            df[n] = (fstar - fstar_ref)

    elif ref_source == 'self':
        # this is the standard read protocol, with delf and delg already in 
        # the .xlsx file
        for n in nvals:
            df[n] = df['delf'+str(n)] + 1j*df['delg'+str(n)].round(1)            
            
    else:
        # here we need to get the reference frequencies from the 
        # data in the reference channel

        n_num = len(nvals)
        df_ref = pd.read_excel(infile, sheet_name=ref_source, header=0)
        vars=['f', 'g']
        # if no temperature is listed we just average the values
        if ('temp' not in df_ref.keys()) or (df.temp.isnull().values.all()):
            for k in np.arange(len(nvals)):
                for p in [0,1]:
                    # get the reference values and plot them
                    ref_val=df_ref[vars[p]+str(nvals[k])].mean()
                    
                    #write the film and reference values to the data frame
                    df[vars[p]+str(nvals[k])+'_dat']=df[vars[p]+str(nvals[k])]
                    df[vars[p]+str(nvals[k])+'_ref']=ref_val
                    
        # now we handle the case where we have a full range of 
        # temperatures
        else:
            keep_column.append('temp')
            fig, ax = plt.subplots(2, n_num, figsize=(3*n_num,6),
                               constrained_layout=True)        
            # reorder rerence data according to temperature
            df_ref=df_ref.sort_values('temp')
            
            # drop any duplicate temperature values
            df_ref = df_ref.drop_duplicates(subset='temp', keep='first')
            temp = df_ref['temp']
            for k in np.arange(len(nvals)):
                for p in [0,1]:
                    ax[p, k].set_title(vars[p]+str(nvals[k]))
                    # get the reference values and plot them
                    ref_vals=df_ref[vars[p]+str(nvals[k])]
                    ax[p, k].plot(temp, ref_vals, '.')
                    ax[p, k].set_xlabel(r'$T$ ($^\circ$C)')
                    
                    # make the fitting function and plot it, along with the
                    # values corresponding to temperatures from the data
                    # set with the film
                    #fit=InterpolatedUnivariateSpline(temp, ref_vals, k=1, ext=3)
                    fit = np.polyfit(temp, ref_vals, 3)
                    ax[p, k].plot(temp, np.polyval(fit, temp), '-')
                    #ax[p, k].plot(df['temp'], fit(df['temp']), '+')
                    
                    #write the film and reference values to the data frame
                    df[vars[p]+str(nvals[k])+'_dat']=df[vars[p]+str(nvals[k])]
                    df[vars[p]+str(nvals[k])+'_ref']=np.polyval(fit, df['temp'])
 
            # label the plot
            fig.suptitle('ref temp coeffs: '+str(fit))
        
        for k in np.arange(len(nvals)):
            # now write values delfstar to the dataframe
            df[nvals[k]]=(df['f'+str(nvals[k])+'_dat']-
                          df['f'+str(nvals[k])+'_ref']+
                      1j*(df['g'+str(nvals[k])+'_dat']-
                          df['g'+str(nvals[k])+'_ref'])).round(1)

        # add absolute frequency and reference values to dataframe
        for n in nvals:
            keep_column.append('f'+str(n)+'_dat')
            keep_column.append('f'+str(n)+'_ref')
            keep_column.append('g'+str(n)+'_dat')
            keep_column.append('g'+str(n)+'_ref')

    return df[keep_column].copy()


def plot_bare_tempshift(Tf_coeff, T, Tref):
    fig, ax = plt.subplots(1, 1, figsize=(4,4), constrained_layout=True)
    for n in [1, 3, 5]:
        fitvals = bare_tempshift(Tf_coeff, T, Tref, n)
        ax.plot(T, fitvals, label = 'n='+str(n))
    ax.legend(loc='best')
    ax.set_title('$T_{ref}=$'+str(Tref)+r'$^\circ$C')
    ax.set_xlabel(r'$T\:^\circ$C')
    ax.set_ylabel(r'$\Delta f_n$ (Hz)')


def bare_tempshift(Tf_coeff, T, Tref, n):
    return np.polyval(Tf_coeff[n],T) - np.polyval(Tf_coeff[n],Tref)


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

gstar_kww = np.vectorize(gstar_kww_single)


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
    n_rouse = int(n_rouse)

    rouse=np.zeros((len(wtau), n_rouse), dtype=complex)
    for p in 1+np.arange(n_rouse):
        rouse[:, p-1] = ((wtau/p**2)**2/(1+wtau/p**2)**2 +
                                  1j*(wtau/p**2)/(1+wtau/p**2)**2)
    rouse = rouse.sum(axis=1)/n_rouse
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
        gstar (numpy array):  
            complex shear modulus normalized by unrelaxed value         
    """

    # specify which elements are kww or Maxwell elements
    kww = kwargs.get('kww',[])
    maxwell = kwargs.get('maxwell',[])
    rouse = kwargs.get('rouse', [])

    # make values numpy arrays if they aren't already
    tau = np.asarray(tau).reshape(1, -1)[0,:]
    beta = np.asarray(beta).reshape(1, -1)[0,:]
    g0 = np.asarray(g0).reshape(1, -1)[0,:]
    sp_type = np.asarray(sp_type).reshape(1, -1)[0,:]
    
    nw = len(w)  # number of frequencies
    n_br = len(sp_type)  # number of series branches
    n_sp = sp_type.sum()  # number of springpot elements
    sp_comp = np.empty((nw, n_sp), dtype=np.complex)  # element compliance
    br_g = np.empty((nw, n_br), dtype=np.complex)  # branch stiffness

    # calculate the compliance for each element
    for i in np.arange(n_sp):
        if i in maxwell:  # Maxwell element
            sp_comp[:, i] = 1/(g0[i]*gstar_maxwell(w*tau[i]))            
        elif i in kww:  #  kww (stretched exponential) elment
            sp_comp[:, i] = 1/(g0[i]*gstar_kww(w*tau[i], beta[i]))
        elif i in rouse:  # Rouse element, beta is number of rouse modes
            sp_comp[:, i] = 1/(g0[i]*gstar_rouse(w*tau[i], beta[i]))
        else:  # power law springpot element
            sp_comp[:, i] = 1/(g0[i]*(1j*w*tau[i]) ** beta[i])

    # sp_vec keeps track of the beginning and end of each branch
    sp_vec = np.append(0, sp_type.cumsum())
    
    #  g_br keeps track of the contribution from each branch
    g_br = {}
    for i in np.arange(n_br):
        sp_i = np.arange(sp_vec[i], sp_vec[i+1])
        # branch compliance obtained by summing compliances within the branch
        br_g[:, i] = 1/sp_comp[:, sp_i].sum(1)
        g_br[i]=br_g[:,i]

    # now we sum the stiffnesses of each branch and return the result
    g_tot = br_g.sum(1)
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
    from pylab import meshgrid
    '''
    Create contour plot of normf_g or rh_rd and verify that solution is correct.
    
    args:
        df (dataframe):
            input dataframe to consider
                
    kwargs:
        wintitle (string):
            Window title.
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
          
    '''

    numxy = kwargs.get('numxy', 100)
    numz = kwargs.get('numz', 200)
    philim = kwargs.get('philim', [0, 90])
    dlim = kwargs.get('dlim', [0.001, 0.5])
    nplot = kwargs.get('nplot', [1,3,5])
    ratios = kwargs.get('ratios', False)
    autoscale = kwargs.get('autoscale', False)
    label = kwargs.get('label', 'temp')
    plot_solutions = kwargs.get('plot_solutions', False)
    idxmin=df.index[0]
    calc = df['calc'][idxmin]
    
    # make the axes
    fig, ax = plt.subplots(2,2, figsize=(10,8), sharex=False, sharey=False,
                           num=calc, constrained_layout=True)
                       
    # make meshgrid for contour
    phi = np.linspace(philim[0], philim[1], numxy) 
    dlam = np.linspace(dlim[0], dlim[1], numxy)
    DLAM, PHI = meshgrid(dlam, phi)

    # need to use n=3 in this calculation, since
    # normdelfstar assumes third harmonic in its definition
    
    def Zfunction(x,y):
        #function used for calculating the z values
        # this Z is the value plotted in the contour plot and NOT the impedance
        drho = df['drho'][idxmin]
        grho3 = grho_from_dlam(3, drho, x, y)
        fnorm = normdelf_bulk(3, x, y)
        gnorm = normdelg_bulk(3, x, y)
        if ratios:
            n1=int(calc[0])
            n2=int(calc[1])
            n3=int(calc[2])
            Z1 = np.real(normdelfstar(n2, x, y))/np.real(normdelfstar(n1, x, y))
            Z2 = -np.imag(normdelfstar(n3, x, y))/np.real(normdelfstar(n3, x, y))
        else:
            delfstar = sauerbreyf(1, drho)*normdelfstar(3, x, y)
            Z1 = np.real(delfstar)
            Z2 = np.imag(delfstar)
        return Z1, Z2, drho, grho3, fnorm, gnorm

    Z1, Z2, drho, grho3, fnorm, gnorm = Zfunction(DLAM, PHI)
    
    # specify the range of the Z values
    if autoscale:
        min1 = Z1.min()
        max1 = Z1.max()
        min2 = Z2.min()
        max2 = Z2.max()
    else:
        if ratios:
            min1 = -1
            max1 = 1.2
            min2 = 0
            max2 = 2
        else:
            min1 = -3*sauerbreyf(1, df['drho'][df.index[idxmin]])
            max1 = 3*sauerbreyf(1, df['drho'][df.index[idxmin]])
            min2 = 0
            max2 = 5000
        
    levels1 = np.linspace(min1, max1, numz)
    levels2 = np.linspace(min2, max2, numz)
    
    contour1 = ax[0,0].contourf(DLAM, PHI, Z1, levels=levels1, 
                              cmap='rainbow')
    contour2 =  ax[0,1].contourf(DLAM, PHI, Z2, levels=levels2,
                              cmap='rainbow')
    
    fig.colorbar(contour1, ax=ax[0,0])
    fig.colorbar(contour2, ax=ax[0,1])

    # set label of ax[1]
    ax[0,0].set_xlabel(r'$d/\lambda_3$')
    ax[0,0].set_ylabel(r'$\Phi$ ($\degree$)')
    ax[1,0].set_ylabel(r'$\Delta f/n$ (Hz)')

    ax[0,1].set_xlabel(r'$d/\lambda_3$')
    ax[0,1].set_ylabel(r'$\Phi$ ($\degree$)')
    ax[1,1].set_ylabel(r'$\Delta\Gamma/n$ (Hz)')
    
    # add titles
    if ratios:
        ax[0,0].set_title(r'$r_h$ - '+calc)
        ax[0,1].set_title(r'$r_d$ - '+calc)
    else:         
        ax[0,0].set_title(r'$\Delta f /n$ (Hz) - '+calc)
        ax[0,1].set_title(r'$\Delta\Gamma /n$ (Hz) - '+calc)
    
    # set formatting for parameters that appear at the bottom of the plot
    # when mouse is moved
    def fmt(x, y):
        if ratios:
            z1, z2, drho, grho3, fnorm, gnorm=Zfunction(x,y)
            return  'd/lambda={x:.3f},  phi={y:.1f}, rh={z1:.5f}, rd={z2:.5f}, '\
                    'drho={drho:.2f}, grho3={grho3:.2e}, '\
                    'fnorm={fnorm:.4f}, gnorm={gnorm:.4f}'.format(x=x, y=y, 
                     z1=z1, z2=z2,
                     drho=1000*drho, grho3=grho3/1000, fnorm=fnorm, gnorm=gnorm)    
                      
        else:
            z1, z2, drho, grho3, fnorm, gnorm=Zfunction(x,y)
            return  'd/lambda={x:.3f},  phi={y:.1f}, delfstar/n={z:.0f}, '\
                    'drho={drho:.2f}, grho3={grho3:.2e}, '\
                    'fnorm={fnorm:.4f}, gnorm={gnorm:.4f}'.format(x=x, y=y, 
                     z=z1+1j*z2, 
                     drho=1000*drho, grho3=grho3/1000, fnorm=fnorm, gnorm=gnorm)    
                         
    # now add the experimental data
    # variable to keep track of differentplots
    dvals= {}
    phisol=np.array([])
    df_expt={}
    df_calc={}
    legend_label=np.array([])
    col={1:'C0', 3:'C1',5:'C2'}
    for idx, row in df.iterrows():
        phisol =  np.append(phisol, row['phi'])

    for n in nplot:
        nstr=str(n)
        dvals[n] = np.array([])
        df_expt[n] = np.array([])
        df_calc[n] = np.array([])
        
        # extract the calculated propoerties from the dataframe
        for idx, row in df.iterrows():
            film = {'drho':row['drho'], 'grho3':row['grho3'],
                    'phi':row['phi']}
            dvals[n] = np.append(dvals[n], calc_dlam(n, film))
            df_expt[n] = np.append(df_expt[n], row['df_expt'+nstr])
            df_calc[n] = np.append(df_calc[n], row['df_calc'+nstr])        
            if label in df.keys():
                lab = label+'='+str(row[label])
            else:
                lab = 'idx='+str(idx)
            legend_label = np.append(legend_label, lab)
            
            
    for n in nplot: 
        nstr=str(n)+' (expt)'
        # compare experimental and calculated frequency fits     
        ax[1,0].plot(dvals[3], np.real(df_expt[n])/n, '+', color = col[n],
                     label = 'n='+nstr)
        ax[1,0].plot(dvals[3], np.real(df_calc[n])/n, '-', color = col[n])
                
        # now compare experimental and calculated dissipation
        ax[1,1].plot(dvals[3], np.imag(df_expt[n])/n, '+', color = col[n],
                     label = 'n='+nstr)
        ax[1,1].plot(dvals[3], np.imag(df_calc[n])/n, '-',
                     color = col[n])
        
    # add values to contour plots for n=3
    ax[0,0].plot(dvals[3], phisol, 'k-')
    ax[0,1].plot(dvals[3], phisol, 'k-')  
    
    for k in [0,1]:  
        ax[1,k].legend()
        ax[0,k].format_coord = fmt
        ax[1,k].set_xlabel(r'$d/\lambda _3$')
        
    # reset axis limits
    ax[0,0].set_xlim(dlim)
    ax[0,1].set_xlim(dlim)
    ax[0,0].set_ylim(philim)
    ax[0,1].set_ylim(philim)
    
    # create a PdfPages object - one solution check per page
    pdf = PdfPages('solution_'+calc+'.pdf')
    pdf.savefig(fig)

    if plot_solutions:
        for idx in np.arange(len(phisol)):
            curves = {}
    
            # indicate where the solution is being taken
            print('writing solution '+str(idx)+' of '+str(len(phisol)-1))
            for k in [0,1]:
                curves[0+k]=ax[0,k].plot(dvals[3][idx], phisol[idx], 'kx', markersize=14,
                                         label = legend_label[idx])
                
            for n in nplot:         
                curves[2+(n-1)/2]=ax[1,0].plot(dvals[3][idx], 
                            np.real(df_expt[n][idx])/n, 'kx',
                             markersize=14, color=col[n])
                curves[5+(n-1)/2]=ax[1,1].plot(dvals[3][idx], np.imag(df_expt[n][idx])/n, 'kx',
                             markersize=14, color=col[n])
        
        # now plot the lines for the solution
            dlam3 = dvals[3][idx]
            rh = rhcalc(calc, dlam3, phisol[idx])
            rd = rdcalc(calc, dlam3, phisol[idx])
               
            def solutions(phi, guess):
                def ftosolve0(d):
                    return rhcalc(calc, d, phi)-rh
            
                def ftosolve1(d):
                    return rdcalc(calc, d, phi)-rd
                
                soln0 = optimize.least_squares(ftosolve0, guess[0], bounds=dlim)
                soln1 = optimize.least_squares(ftosolve1, guess[1], bounds=dlim)
                return {'phi':phi, 'd_rh':soln0['x'][0], 'd_rd':soln1['x'][0]}
        
            npts = 25
            dcalc=pd.DataFrame(columns=['phi','d_rh', 'd_rd'])
      
            # starting guess is the actual solution, and then we work outward
            # from there
            for phiend in philim:
                guess = [dlam3, dlam3]
                phivals = np.linspace(phisol[idx], phiend, npts)
                for phival in phivals:
                    soln = solutions(phival, guess)
                    dcalc=dcalc.append(soln,ignore_index=True)
                    guess = [soln['d_rh'], soln['d_rd']]
    
            dcalc = dcalc.sort_values(by=['phi'])
            for k in [0,1]:
                curves[8+k]=ax[0,k].plot(dcalc['d_rh'], dcalc['phi'], 'k-')
                curves[10+k]=ax[0,k].plot(dcalc['d_rd'], dcalc['phi'], 'k--')
                
            curves[12]=ax[0,0].legend()
                
            pdf.savefig(fig)
            
            for k in np.arange(12):
                curves[k][0].remove()
                
    pdf.close()
    return fig, ax




