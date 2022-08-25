'''
This module is a modified version of QCM_functions.
It is used for UI, but you can also use it for data analysis.
The input and return data are all in form of DataFrame.
This module doesn't have plotting functions.

NOTE: Differnt to other modules, the harmonics used in this module are all INT.
'''


import importlib
from typing import Dict
import numpy as np
import pandas as pd
from scipy import optimize
from lmfit import Minimizer, minimize, Parameters, fit_report, printfuncs

import logging
logger = logging.getLogger(__name__)

kww_spec = importlib.util.find_spec('kww')
found_kww = kww_spec is not None
if found_kww:
    from kww import kwwc, kwws
    # kwwc returns: integral from 0 to +infinity dt cos(omega*t) exp(-t^beta)
    # kwws returns: integral from 0 to +infinity dt sin(omega*t) exp(-t^beta)
else:
    logger.warning('kww module is not found!')


# variable limitions
dlam_refh_range = (0, 10) # (0, 5)
drho_range = (0, 3e-2) # m kg/m^3 = 1000 um g/cm^3 = 1000 g/m^2
# drho_range = (0, np.inf) # m kg/m^3 = 1000 um g/cm^3 = 1000 g/m^2
grho_refh_range = (1e4, 1e14) # Pa kg/m^3 = 1/1000 Pa g/cm^3
phi_range = (0, np.pi/2) # rad 

bulk_drho = np.inf # default bulk thickness

#  Zq (shear acoustic impedance) of quartz = rho_q * v_q
Zq = {
    'AT': 8.84e6,  # kg m−2 s−1
    'BT': 0e6,     # kg m−2 s−1
}

e26 = 9.65e-2 # piezoelectric stress coefficient (e = 9.65·10−2 C/m2 for AT-cut quartz)
d26 = 3.1e-9 # piezoelectric strain coefficient (d = 3.1·10−9 m/V for AT-cut quartz)
g0 = 10 # 10 Hz, Half bandwidth (HWHM) of unloaed resonator (intrinsic dissipation on crystalline quartz)
dq = 330e-6  # only needed for piezoelectric stiffening calc.
epsq = 4.54
eps0 = 8.8e-12
C0byA = epsq * eps0 / dq 

prop_default = {
    'electrode': {'calc': False, 'grho': 3.0e14, 'phi': 0, 'drho': 2.8e-3, 'n': 3},  # n here is harmonic.
    'air':       {'calc': False, 'grho': 0, 'phi': 0, 'drho': 0, 'n': 1}, #???
    'water':     {'calc': False, 'grho': 1e8, 'phi': np.pi / 2, 'drho': bulk_drho, 'n': 3}, #?? water ar R.T.
}



def nh2i(nh):
    '''
    convert harmonic (str) to index (int) 
    since only odd harmonics are stored in list
    '''
    if isinstance(nh, str):
        nh = int(nh)
    return int((nh - 1) / 2)
    

def nhcalc2nh(nhcalc):
    '''
    convert nhcalc (str) to list of harmonics (int) in nhcalc
    '''
    return [int(s) for s in nhcalc] 




class QCM:
    def __init__(self, cut='AT'):
        '''
        phi in rad for calculation and exported as degree
        '''
        self.Zq = Zq[cut]  # kg m−2 s−1. shear acoustic impedance of AT cut quartz
        #TODO add zq by cuts
        self.f1 = None # 5e6 Hz fundamental resonant frequency
        self.g1 = None # dissipation of 1st harmonic of bare
        self.f0s = None # f0 of all harmonics in DICT with int(harm) as key
        self.g0s = None # g0 of all harmonics in DICT with int(harm) as key
        self.g_err_min = 1 # error floor for gamma
        self.f_err_min = 1 # error floor for f
        self.err_frac = 3e-2 # error in f or gamma as a fraction of gamma

        self.refh = None # reference harmonic for calculation

        self.refto = 0 # ref to electrode (0) or knownlayers (1), or None to crystal

        self.calctype = 'SLA' # 'SLA', 'LL', 'Voigt' NOTE: not case sensitive
        # 'Voigt': Qsense version: constant G', G" linear in omega
        # default values

        self.piezoelectric_stiffening = False # set True if including piezoelectric stiffening

        # self.fit_method = 'lmfit'
        self.fit_method = 'scipy'
        

    def get_prop_by_name(self, name):
        return prop_default.get(name, prop_default['air']) # if name does not exist, use air ?


    def fstar_err_calc(self, delfstar):
        ''' 
        calculate the error in delfstar
        delfstar: complex number 
        film: dict with all properties calculated
        '''

        # NOTE: since all delfstar is supposed to reference to bare crystal. We do not correct it for now !!
        # if len(film) > 1: # with multiple layers
        #     fstar = self.calc_delfstar(n, film, self.calctype)
        # else: # single layer
        #     fstar = delfstar

        # start by specifying the error input parameters
        fstar_err = np.zeros(1, dtype=np.complex128)
        fstar_err = (self.f_err_min + self.err_frac * np.imag(delfstar)) + 1j*(self.g_err_min + self.err_frac * np.imag(delfstar))
        return fstar_err
        # ?? show above both imag?


    def sauerbreyf(self, n, drho):
        ''' delf_sn from Sauerbrey eq'''
        # n = int(n) if isinstance(n, str) else n
        return 2 * n * self.f1**2 * drho / self.Zq


    def sauerbreym(self, n, delf):
        ''' mass from Sauerbrey eq'''
        return -delf * self.Zq / (2 * n * self.f1**2)


    def grho(self, n, grho_refh, phi): # old func
        ''' grho of n_th harmonic'''
        return grho_refh * (n/self.refh)**(phi / (np.pi/2))
        

    def grhostar_from_refh(self, n, grho_refh, phi):
        return self.grho(n, grho_refh, phi) * np.exp(1j*phi)
        # return self.grhostar(self.grho(n, grho_refh, phi), phi) 
        # two returns are the same


    def grhostar(self, grho, phi):
        '''return complex value of grhostar from grho (modulus) and phi '''
        return grho * np.exp(1j * phi)


    def grho_from_dlam(self, n, drho, dlam_refh, phi):
        return (drho * n * self.f1 * np.cos(phi / 2) / dlam_refh)**2


    def etarho(self, n, grho_n):
        ''' 
        viscosity at nth harmonic 
        grho_n: grho at nth haromonic
        eta = G / (2*pi*f)
        '''
        return grho_n / (2 * np.pi * n * self.f1)


    def calc_lamrho(self, n, grho_n, phi_n):
        '''
        calculate rho*lambda
        grho_n & phi_n of harmonic n
        '''
        return np.sqrt(grho_n) / (n * self.f1 * np.cos(phi_n / 2))


    def calc_delrho(self, n, grho, phi):
        '''
        decay length 
        '''
        return self.calc_lamrho(n, grho, phi) / (2 * np.pi * np.tan(phi / 2))


    def D(self, n, grho_refh, phi, drho):
        return 2*np.pi*drho*n*self.f1*(np.cos(phi/2) - 1j * np.sin(phi/2)) / (self.grho(n, grho_refh, phi)) ** 0.5


    def DfromZ(self, n, drho, Zstar):
        return 2 * np.pi * n * self.f1 * drho / Zstar


    def zstarfilm(self, n, drho, grhostar):
        if grhostar == 0:
            answer = 0
        else:
            answer = self.zstarbulk(grhostar) * np.tan(2 * np.pi * n * self.f1 * drho / self.zstarbulk(grhostar)) 
        return answer


    def rstar(self, n, grho_refh, phi, drho, overlayer={'drho': 0, 'gho_refh': 0, 'phi': 0}):
        # overlayer is dictionary with drho, grho_refh and phi
        grhostar_1 = self.grhostar_from_refh(n, grho_refh, phi)
        grhostar_2 = self.grhostar_from_refh(n, overlayer.get('grho_refh', 0), overlayer.get('phi', 0))
        zstar_1 = self.zstarbulk(grhostar_1)
        zstar_2 = self.zstarfilm(n, overlayer.get('drho', 0), grhostar_2)   
        return zstar_2 / zstar_1
    
    
    # calcuated complex frequency shift for single layer
    def delfstarcalc(self, n, grho_refh, phi, drho, overlayer):
        rstar = self.rstar(n, grho_refh, phi, drho, overlayer)
        # overlayer is dictionary with drho, grho_refh and phi
        calc = -(self.sauerbreyf(n, drho)*np.tan(self.D(n, grho_refh, phi, drho)) / self.D(n, grho_refh, phi, drho))*(1-rstar**2) / (1+1j*rstar*np.tan(self.D(n, grho_refh, phi, drho)))
        
        # handle case where drho = 0, if it exists
        calc[np.where(drho==0)]=0
        return calc


    def d_lamcalc(self, n, grho_refh, phi, drho):
        return drho*n*self.f1*np.cos(phi/2) / np.sqrt(self.grho(n, grho_refh, phi))


    def thin_film_gamma(self, n, drho, jdprime_rho):
        return 8 * np.pi**2 * n**3 * self.f1**4 * drho**3 * jdprime_rho / (3 * self.Zq)
        # same master expression, replacing grho3 with jdprime_rho3


    def grho_refh(self, jdprime_rho_refh, phi):
        return np.sin(phi)/jdprime_rho_refh


    ###### new funcs #########
    def grho_from_material(self, n, material): # func new
        '''
        n: the harmonic we want to calculate
        '''
        grho_refh = material['grho']
        phi = material['phi']
        refh = material['n']
        return grho_refh * (n / refh)**(phi / (np.pi / 2))


    def calc_grho_ncalc(self, grhostar, n, ncalc):
        '''
        from fun calc_grho3 in QCM_functions

        calculate value of grho3 from given value of Grho at another harmonic,
        assuming power law behavior
        gstar: the complex modulus at harmonic n
        ncalc: the harm of grho to calc
        RETURN:  grho_calc, phi
        '''
        phi = np.angle(grhostar) # in radian
        grho_n = abs(grhostar)
        grho_calc = grho_n * (ncalc / n)**(phi / (np.pi/2))
        # we return values of grho_calc and phi which return the correct value of grhostar at the nth harmonic
        return grho_calc, phi


    def calc_grho_refh(self, grhostar, n):
        '''calculate grho_refh from grhostar of nth harmonic'''
        return self.calc_grho_ncalc(grhostar, n, self.refh)


    def calc_D(self, n, material, delfstar):
        '''
        Calculate D (dk*, thickness times complex wave number).
        
        take the material from a single layer {'grho': 0, 'phi': 0, 'drho': 0,  'n': 1}
        '''
        # logger.info('material %s', material) 
        # logger.info('delfstar %s', delfstar) 
        drho = material['drho']
        # set switch to handle as where drho = 0
        if drho == 0:
            return 0
        else:
            return 2 * np.pi * (n * self.f1 + delfstar) * drho / self.zstar_bulk(n, material)
            #?? should delfstar be delfstar.real?
            #?? can we replace (n * self.f1 + delfstar) with fstar.real


    def zstar_bulk(self, n, material):
        # logger.info('material %s', material) 
        if self.calctype.upper() != 'VOIGT': 
            grho = self.grho_from_material(n, material)  #check for error here
            phi = material['phi']
            grhostar = self.grhostar(grho, phi) 
        else: # Qsense version: constant G', G" linear in omega
            r = material['n']
            grho_r = material['grho']
            greal = grho_r * np.cos(material['phi'])
            gimag= grho_r * np.sin(material['phi']) * (n/r) # NOTE: qcmfucn might missed grho_r
            grhostar = (gimag**2 + greal**2)**0.5 * (np.exp(1j*material['phi']))

        return grhostar**0.5

    def calc_delfstar_sla(self, ZL):
        return self.f1 * 1j / (np.pi * self.Zq) * ZL


    def calc_ZL(self, n, layers, delfstar):
        # layers is a dictionary of dictionaries
        # each dictionary is named according to the layer number
        # layer 0 is closest to the quartz

        if not layers: # no layers are defined
            return 0

        N = len(layers)
        # logger.info('N', N) 
        Z = {}; D = {}; L = {}; S = {}

        # we use the matrix formalism to avoid typos.
        for i, layer_n in enumerate(sorted(layers.keys()), start=1): # iterate except the last layer
            D[i] = self.calc_D(n, layers[layer_n], delfstar)
            if i == N:
                # we skip asign values for Z[N] and L[N]
                break  
            Z[i] = self.zstar_bulk(n, layers[layer_n])
            L[i] = np.array([
                [np.cos(D[i]) + 1j * np.sin(D[i]), 0], 
                [0, np.cos(D[i]) - 1j * np.sin(D[i])]
            ], dtype=complex)

            # logger.info('i', i) 
            # logger.info('Zi', Z[i]) 
            # logger.info('Di', D[i]) 
            # logger.info('Li', L[i]) 

        # get the terminal matrix from the properties of the last layer
        top_n = sorted(layers.keys())[-1]
        Zf_N = 1j * self.zstar_bulk(n, layers[top_n]) * np.tan(D[N])

        # logger.info('top_n', top_n) 
        # logger.info('DN', D[N]) 
        # logger.info('Zf_n', Zf_N) 

        # if there is only one layer, we're already done
        if N == 1:
            return Zf_N

        Tn = np.array([
            [1 + Zf_N / Z[N-1], 0],
            [0, 1 - Zf_N / Z[N-1]]
            ], dtype=complex)

        # logger.info('L[N-1]', L[N-1]) 
        # logger.info('Tn', Tn) 

        uvec = L[N-1] @ Tn @ np.array([[1.], [1.]])
        # logger.info('uvec', uvec) 

        for i in np.arange(N-2, 0, -1):
            S[i] = np.array([
                [1 + Z[i+1] / Z[i], 1 - Z[i+1] / Z[i]],
                [1 - Z[i+1] / Z[i], 1 + Z[i+1] / Z[i]]
            ])
            # logger.info('S[i]', S[i]) 
            uvec = L[i] @ S[i] @ uvec
            # logger.info('uvec', uvec) 

        rstar = uvec[1,0] / uvec[0,0]
        # logger.info('rstar', rstar) 
        # logger.info('ZL', Z[1] * (1 - rstar) / (1 + rstar)) 

        return Z[1] * (1 - rstar) / (1 + rstar)


    def calc_delfstar(self, n, layers):
        '''
        ref to electrode (0) or knownlayers (1) or None to crystal
        '''
        if not layers: # layers is empty {}
            return np.nan

        if self.refto == 1:
            layers_ref = self.get_ref_layers(layers)
        elif self.refto == 0:
            # layers_ref = self.get_ref_layers(layers)
            layers_ref = {0: layers[0]}
        elif self.refto is None:
            layers_ref = {}

        logger.info('layers: %s', layers)
        logger.info('layers_ref: %s', layers_ref)
        
        # there is data
        if self.calctype.upper() == 'SLA':
            # use the small load approximation in all cases where calctype
            # is not explicitly set to 'LL'

            layers = self.remove_layer_0(layers)
            layers_ref = self.remove_layer_0(layers_ref)
            ZL = self.calc_ZL(n, layers, 0)
            if self.refto == 1:
                ZL_ref = self.calc_ZL(n, layers_ref, 0)
                del_ZL = ZL - ZL_ref
            else: # refto == 0 or None
                del_ZL = ZL
            return self.calc_delfstar_sla(del_ZL)

        elif self.calctype.upper() == 'LL':
            # this is the most general calculation
            # use default electrode if its not specified
            if 0 not in layers: 
                layers[0] = prop_default['electrode']

            def solve_Zmot(x, layers):
                delfstar = x[0] + 1j * x[1]
                Zmot = self.calc_Zmot(n, layers, delfstar)
                return [np.real(Zmot), np.imag(Zmot)]

            ZL_all = self.calc_ZL(n, layers, 0)
            delfstar_sla_all = self.calc_delfstar_sla(ZL_all)
            sol = optimize.root(solve_Zmot, [np.real(delfstar_sla_all), np.imag(delfstar_sla_all)], args=(layers,))
            dfc = sol.x[0] + 1j * sol.x[1]
            # logger.info('dfc', dfc) 

            if self.refto is None:
                return dfc

            ZL_ref = self.calc_ZL(n, layers_ref, 0)
            delfstar_sla_ref = self.calc_delfstar_sla(ZL_ref)
            sol = optimize.root(solve_Zmot, [np.real(delfstar_sla_ref), np.imag(delfstar_sla_ref)], args=(layers_ref,))
            dfc_ref = sol.x[0] + 1j * sol.x[1]
            # logger.info('dfc_ref', dfc_ref) 
                # logger.info('dfc_ref', dfc_ref) 
            # logger.info('dfc_ref', dfc_ref) 
                # logger.info('dfc_ref', dfc_ref) 
            # logger.info('dfc_ref', dfc_ref) 

            return dfc - dfc_ref

        else:
            return np.nan


    def calc_delfstar_from_single_material(self, n, material):
        '''
        convert material to a single layer and return delfstar
        '''
        layers = self.build_single_layer_film(material)
        return self.calc_delfstar( n, layers)


    def calc_Zmot(self, n, layers, delfstar):
        '''
        Calculate motional impedance 
        returns: delfstar, Complex frequency shift in Hz.
        '''
        om = 2 * np.pi * (n * self.f1 + delfstar)
        Zqc = self.Zq * (1 + 1j * 2 * self.g1 / (n * self.f1)) # NOTE: changed g0 to self.g1


        self.drho_q = self.Zq / (2 * self.f1)
        Dq = om * self.drho_q / self.Zq
        secterm = -1j * Zqc / np.sin(Dq)
        ZL = self.calc_ZL(n, layers, delfstar)
        # eq. 4.5.9 in book
        thirdterm = ((1j * Zqc * np.tan(Dq/2))**-1 + (1j * Zqc * np.tan(Dq / 2) + ZL)**-1)**-1
        Zmot = secterm + thirdterm

        if self.piezoelectric_stiffening:
            ZC0byA = C0byA / (1j*om)
            # can always be neglected as far as we can tell
            ZPE = -(e26 / dq)**2 * ZC0byA  # ZPE accounts for piezoelectric stiffening anc
            Zmot += ZPE

        # logger.info('Zmot shape %s', Zmot.shape) 
        # logger.info('Zmot %s', Zmot) 
        return Zmot


    def calc_dlam(self, n, material):
        '''
        Calculate D (dk*, thickness times complex wave number).
        material: {grho, phi, drho, n}
        '''
        return np.real(self.calc_D(n, material, 0)) / (2 * np.pi)


    ##### end new funcs ######


    def dlam(self, n, dlam_refh, phi):
        return dlam_refh * (n/self.refh)**(1-phi/np.pi)


    def normdelfstar(self, n, dlam_refh, phi):
        '''
        Calculate complex frequency shift normzlized by Sauerbrey shift.
        '''
        dlam_n = self.dlam(n, dlam_refh, phi)
        return -np.tan(2*np.pi*dlam_n*(1-1j*np.tan(phi/2))) / (2*np.pi*dlam_n*(1-1j*np.tan(phi/2)))


    def calc_drho(self, n1, delfstar, dlam_refh, phi):
        return -self.sauerbreym(n1, np.real(delfstar[n1])) / np.real(self.normdelfstar(n1, dlam_refh, phi))


    def rhcalc(self, nh, dlam_refh, phi):
        ''' nh: list '''
        return np.real(self.normdelfstar(nh[0], dlam_refh, phi)) /  np.real(self.normdelfstar(nh[1], dlam_refh, phi))


    def rh_from_delfstar(self, nh, delfstar):
        ''' this func is the same as rhexp!!! '''
        n1 = int(nh[0])
        n2 = int(nh[1])
        if np.real(delfstar[n2]) == 0:
            return np.nan
        else:
            return (n2/n1)*np.real(delfstar[n1])/np.real(delfstar[n2])


    def rdcalc(self, nh, dlam_refh, phi):
        return -np.imag(self.normdelfstar(nh[2], dlam_refh, phi)) / np.real(self.normdelfstar(nh[2], dlam_refh, phi))


    def rdexp(self, nh, delfstar):
        ''' not using '''
        return -np.imag(delfstar[nh[2]]) / np.real(delfstar[nh[2]])


    def rd_from_delfstar(self, n, delfstar):
        ''' dissipation ratio calculated for the relevant harmonic '''
        if np.real(delfstar[n]) == 0:
            return np.nan
        else:
            return -np.imag(delfstar[n]) / np.real(delfstar[n])


    ######## functions for bulk ########
    
    # zstar_bulk() is used for thin layer too. So, it is not listed here


    def zstarbulk(self, grhostar):
        '''
        the difference between zstarbulk(grhostar) and zstar_bulk(n, material) is they take different variables. 
        '''
        return grhostar ** 0.5

    
    def delrho_bulk(self, n, delfstar):
        '''decay length multiplied by density'''
        return -self.Zq * abs(delfstar[n])**2 / (2 * n * self.f1**2 * np.real(delfstar[n]))


    # calculated complex frequency shift for bulk layer
    def delfstarcalc_bulk(self, n, grho_refh, phi):
        return ((self.f1*np.sqrt(self.grho(n, grho_refh, phi)) / (np.pi*self.Zq)) * (-np.sin(phi/2)+ 1j * np.cos(phi/2)))


    def delfstarcalc_bulk_from_film(self, n, film):
        # TODO add fun to find the bulk layer
        material = self.get_calc_material(film) 
        grho_refh = self.grho_from_material(self.refh, material)
        phi = material['phi']
        return self.delfstarcalc_bulk(n, grho_refh, phi)


    def bulk_guess(self, delfstar):
        ''' get the bulk solution for grho and phi '''
        grho_refh = self.grho_bulk(delfstar)
        phi = self.phi_bulk(delfstar)

        dlam_refh = self.bulk_dlam_refh(grho_refh, phi)

        return [dlam_refh, phi]


    def bulk_dlam_refh(self, grho_refh, phi):
        ''' get bulk dlam by setting it a quarter wavelength for now'''
        # calculate rho*lambda
        lamrho_refh = self.calc_lamrho(self.refh, grho_refh, phi)
        # logger.info('grho_refh %s', grho_refh) 
        # logger.info('lamrho_refh %s', lamrho_refh) 
        # we need an estimate for drho. We only use this approach if it is
        # reasonably large. We'll put it at the quarter wavelength condition for now
        drho = lamrho_refh / 4

        film = {'drho':drho, 'grho':grho_refh, 'phi':phi, 'n': self.refh}
        
        dlam_refh = self.calc_dlam(self.refh, film)

        return dlam_refh


    def grho_bulk(self, delfstar, n=None):
        '''
        bulk model
        calculate grho reference to refh
        '''
        if n is None:
            n = self.refh
        return (np.pi * self.Zq * abs(delfstar[n]) / self.f1) ** 2


    def phi_bulk(self, delfstar, n=None):
        '''
        bulk model
        calculate phi
        '''
        if n is None:
            n = self.refh
        return min(np.pi / 2, -2 * np.arctan(np.real(delfstar[n]) / np.imag(delfstar[n]))) # limit phi <= pi/2
        # return -2 * np.arctan(np.real(delfstar[n]) / np.imag(delfstar[n])) # phi will be limited to < pi/2 when exported in solve_single_queue


    def bulk_props(self, delfstar, n=None):
        # get the bulk solution for grho and phi
        #??
        return [
            self.grho_bulk(delfstar, n), # grho
            self.phi_bulk(delfstar, n), # phi
            bulk_drho, # drho OR make it inf?
        ]


    ######## end of bulk functions ########


    def guess_from_props(self, material):
        '''
        # TODO: check if prop_guress from calling function is material or material
        '''
        # dlam_refh = self.calc_dlam(self.refh, material)
        try:
            grho_refh = self.grho_from_material(self.refh, material)
        except:
            grho_refh = material.get('grho', np.nan)
        phi = material.get('phi', np.nan)
        drho = material.get('drho', np.nan)
        return [grho_refh, phi, drho]


    def thinfilm_guess(self, delfstar, nh):
        ''' 
        guess the thin file properties by delfstar
        return grho, phi, drho, dlam_refh
        '''
        # first pass at solution comes from rh and rd
        rd_exp = self.rd_from_delfstar(nh[2], delfstar) # nh[2]
        rh_exp = self.rh_from_delfstar(nh, delfstar) # nh[0], nh[1]
        # logger.info('rd_exp %s', rd_exp) 
        # logger.info('rh_exp %s', rh_exp) 
        
        n1 = nh[0]
        n2 = nh[1]
        n3 = nh[2]
        # solve the problem
        if ~np.isnan(rd_exp) and ~np.isnan(rh_exp):
            # logger.info('rd_exp, rh_exp is not nan') 

            logger.info('use thin film guess') 
            dlam_refh, phi = 0.05, np.pi/180*5

            if self.fit_method == 'lmfit': # easier to read
                # we solve the problem initially using the harmonic and dissipation
                # ratios, using the small load approximation
                def residual(pars, rh_exp, rd_exp, nh): # function for residual to solve dlam & phi
                    vals = pars.valuesdict()
                    dlam = vals['dlam']
                    phi = vals['phi']
                    return [self.rhcalc(nh, dlam, phi)-rh_exp, self.rdcalc(nh, dlam, phi)-rd_exp]
                params = Parameters()
                params.add('dlam', value=dlam_refh, min=dlam_refh_range[0], max=dlam_refh_range[1])
                params.add('phi', value=phi, min=phi_range[0], max=phi_range[1])

                # logger.info(x0) 
                soln = minimize(
                    residual,
                    params,
                    method='least_squares',
                    args=(rh_exp, rd_exp, nh),
                    nan_policy='omit', # ('raise' default, 'propagate', 'omit')
                )
                # logger.info(soln['x']) 
                dlam_refh = soln.params.get('dlam').value
                phi =soln.params.get('phi').value
                drho = self.calc_drho(n1, delfstar, dlam_refh, phi)
                grho_refh = self.grho_from_dlam(self.refh, drho, dlam_refh, phi)
            else: # scipy
                lb = np.array([dlam_refh_range[0], phi_range[0]])  # lower bounds on dlam_refh and phi
                ub = np.array([dlam_refh_range[1], phi_range[1]])  # upper bonds on dlam_refh and phi

                # we solve the problem initially using the harmonic and dissipation
                # ratios, using the small load approximation
                def ftosolve(x): # solve dlam & phi
                    return [self.rhcalc(nh, x[0], x[1])-rh_exp, self.rdcalc(nh, x[0], x[1])-rd_exp]

                x0 = np.array([dlam_refh, phi])
                # logger.info(x0) 
                soln = optimize.least_squares(ftosolve, x0, bounds=(lb, ub))
                # logger.info(soln['x']) 
                dlam_refh = soln['x'][0]
                phi =soln['x'][1]
                drho = self.calc_drho(n1, delfstar, dlam_refh, phi)
                grho_refh = self.grho_from_dlam(self.refh, drho, dlam_refh, phi)
        else:
            grho_refh, phi, drho, dlam_refh = np.nan, np.nan, np.nan, np.nan
        
        return grho_refh, phi, drho, dlam_refh
 

    def convert_D_to_gamma(self, D_dsiptn, n):
        '''
        this function convert given D_dsiptn (QCM-D) to gamma used in this program
        D_dsiptn: dissipation of QCM-D
        n: int 
        '''
        return 0.5 * n * self.f1 * D_dsiptn


    def convert_gamma_to_D(self, gamma, n):
        '''
        this function convert given gamma to D (QCM-D)
        D_dsiptn: dissipation of QCM-D
        harm: str. 
        '''
        return 2 * gamma / (n * self.f1)


    def isbulk(self, rd_exp, bulklimt):
        return rd_exp >= bulklimt


    ########################################################



    ########################################################


    def solve_single_queue_to_prop(self, nh, qcm_queue, calctype=None, film={}, bulklimit=0.5, nh_interests=None, brief_report=False):
        '''
        This function is for the cases you want props (dict) from the qcm_queue directly
        
        solve the property of a single test.
        nh: list of int
        qcm_queue:  QCM data. df (shape[0]=1) 
        calctype: 'SLA' / 'LL'
        film: dict of the film layers information
        return grho_refh, phi, drho, dlam_ref, err
        '''
        if calctype is not None:
            self.calctype = calctype

        # logger.info('calctype %s', calctype) 
        #TODO this may be replaced
        film = self.replace_layer_0_prop_with_known(film)

        # get fstar
        fstars = qcm_queue.fstars.iloc[0] # list
        # get delfstar
        delfstars = qcm_queue.delfstars.iloc[0] # list
        # logger.info('fstars %s', fstars) 
        # logger.info(delfstars) 
        # convert list to dict to make it easier to do the calculation
        # fstar = {int(i*2+1): fstar[i] for i, fstar in enumerate(fstars)}
        delfstar = {int(i*2+1): dfstar for i, dfstar in enumerate(delfstars)}
        # logger.info(delfstar) 

        # set f1
        f0s = qcm_queue.f0s.iloc[0]
        f0s = {int(i*2+1): f0 for i, f0 in enumerate(f0s)}
        g0s = qcm_queue.g0s.iloc[0]
        g0s = {int(i*2+1): g0 for i, g0 in enumerate(g0s)}

        self.f0s = f0s
        self.g0s = g0s

        if np.isnan(list(f0s.values())).all():
            self.f1 = np.nan
        else:
            for k in sorted(f0s.keys()):
                if ~np.isnan(f0s[k]): # the first non na harmonic
                    self.f1 = f0s[k] / k
                    self.g1 = g0s[k] / k
                    logger.info('self.f1 %s', self.f1)
                    break

            # use np find in dict values. may have issue with lower ver Python since dict is not ordered        
            # first_notnan = np.argwhere(~np.isnan(list(f0s.values())))[0][0] # find out index of the first freq is not nan
            # use this value calculate f1 = fn/n (in case f1 is not recorded)
            # self.f1 = f0s[first_notnan] / (first_notnan * 2 + 1)

        # logger.info('f1 %s, self.f1) 

        brief_props, props = self.solve_general_delfstar_to_prop(nh, delfstar, film, prop_guess={}, bulklimit=bulklimit, nh_interests=nh_interests, brief_report=brief_report)

        return brief_props, props


    def solve_single_queue(self, nh, qcm_queue, mech_queue, calctype=None, film={}, bulklimit=0.5):
        '''
        solve the property of a single test.
        nh: list of int
        qcm_queue:  QCM data. df (shape[0]=1) 
        mech_queue: initialized property data. df (shape[0]=1)
        calctype: 'SLA' / 'LL'
        film: dict of the film layers information
        return mech_queue

        NOTE: n used in this function is int
        '''
        # get the marks [1st, 3rd, 5th, ...]
        marks = qcm_queue.marks.iloc[0]

        # find available harmonics by the mark is not nan or None (0 or 1)
        nh_interests = [i*2+1 for i, mark in enumerate(marks) if (not np.isnan(mark)) and (mark is not None)]

        # logger.info('film before calc %s', film) 
        # we force the function to do all calculations here
        brief_props, props = self.solve_single_queue_to_prop(nh, qcm_queue, film=film, bulklimit=bulklimit, nh_interests=nh_interests)

        def replace_sublist_in_series_by_dict(s, dic):
            '''
            s: with cell value is a list
            dic: with key values are [1, 3, 5]
            we replace the values in list by matching the (index*2+1) with keys in dict
            '''
            return s.apply(lambda x:[dic[i*2+1] if i*2+1 in dic else v for i, v in enumerate(x)])
        
        # props has all the results we want to update to mech_queue
        for col in mech_queue.columns:
            if col in props.keys(): # check if the desired value is returned in the dict
                mech_queue[col] = replace_sublist_in_series_by_dict(mech_queue[col], props[col])

        return mech_queue
        ########## TODO 


    def solve_general_delfstar_to_prop(self, nh, delfstar, film, calctype=None, prop_guess={}, bulklimit=0.5, nh_interests=None, brief_report=False):
        '''
        solve the property of a single test.
        nh: list of int
        delfstar: dict {harm(int): complex, ...}
        film: dict e.g.: {0: 'calc': False, 'drho': 0, 'grho_refh': 0, 'phi': 0}
        bulklimt: 0.5 by default. rd > bulklimt use bulk calculation
        nh_interests: a list of harmonics (int) to return calculated values. if None, [1, ... max(nh)] 
        brief_report: True, calculate less factors. Both True and False will return props as 2nd dict (When it is True, only experiment data related factors are returned in props)
        return brief_props, props
        '''
        if calctype is not None:
            self.calctype = calctype

        if nh_interests is None: # no list is given
            nh_interests = [i+1 for i in np.arange(max(nh)) if i%2 == 0]

        # first pass at solution comes from rh and rd
        rd_exp = self.rd_from_delfstar(nh[2], delfstar) # nh[2]
        rh_exp = self.rh_from_delfstar(nh, delfstar) # nh[0], nh[1]
        # logger.info('rd_exp %s', rd_exp) 
        # logger.info('rh_exp %s', rh_exp) 
   

        # input variables - this is helpfulf for the error analysis
        # define sensibly names partial derivatives for further use
        err = {}
        err_names=['grho_refh', 'phi', 'drho']
        # initiate  err
        for key in err_names:
            err[key] = np.nan

        # initiate empty values
        grho_refh, phi, drho, dlam_refh, rd_calc, rh_calc = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        delfstar_calc, normdelfstar_calcs = {}, {}

        props = {} # dict used for returen detailed calculation

        # we put those props not related to mechanical properties ahead of calculating to make sure to have them enven the mechnical calculation is failed
        props['delf_exps'] = {n: np.real(delfstar[n]) for n in nh_interests}
        props['delg_exps'] = {n: np.imag(delfstar[n]) for n in nh_interests}
        props['delfn_exps'] = {n: props['delf_exps'][n] / n for n in nh_interests}
        props['delgn_exps'] = {n: props['delg_exps'][n] / n for n in nh_interests}

        props['sauerbreyms'] = {n: self.sauerbreym(n, props['delf_exps'][n]) for n in nh_interests}

        props['delD_exps'] = {n: self.convert_gamma_to_D(props['delg_exps'][n], n) for n in nh_interests}

        props['rd_exps'] = {n: self.rd_from_delfstar(n, delfstar) for n in nh_interests}
        props['rh_exp'] = {n: rh_exp for n in nh_interests}


        # if there is na values, stop solving and return nans
        if np.isnan(rd_exp) or np.isnan(rh_exp):
            logger.warning('rd_exp and/or rh_exp have nan') 
            return [
                dict(
                    grho_refh=grho_refh,
                    phi=phi, 
                    drho=drho, 
                    dlam_refh=dlam_refh, 
                    rd_exp=rd_exp,
                    rh_exp=rh_exp,
                    rd_calc=rd_calc,
                    rh_calc=rh_calc,
                    err=err, 
                    nh=nh,
                    delfstar=delfstar,
                    delfstar_calc=delfstar_calc,
                    normdelfstar_calcs=normdelfstar_calcs
                ), 
                props
            ] 

        # solve the problem
        if not film: # film is not built
            film = self.build_single_layer_film()

        n1 = nh[0]
        n2 = nh[1]
        n3 = nh[2]

        # set the bounds for solutions
        lb = np.array([grho_refh_range[0], phi_range[0], drho_range[0]])  # lower bounds ongrho and phi, drho
        ub = np.array([grho_refh_range[1], phi_range[1], drho_range[1]])  # upper bounds on grho and phi, drho
        
        ## set functions and initial values
        # logger.info('rd_exp, rh_exp is not nan') 
            # logger.info('rd_exp, rh_exp is not nan') 
        # logger.info('rd_exp, rh_exp is not nan') 
        isbulk = self.isbulk(rd_exp, bulklimit)


        # now let's get the guess (initial) values for calculating first
        '''
        The guess will overwrite the given guess
        if guess failed, we go to bulk or thinfilm guess
        guess doesn't have to be assigned with all values
        '''
        grho_refh_guess, phi_guess, drho_guess = np.nan, np.nan, np.nan
        if prop_guess: # prop_guess is a film dict {'drho', 'grho_refh', 'phi'}
            logger.info('use guess') 
            # logger.info('use prop guess') 
                # logger.info('use prop guess') 
            # logger.info('use prop guess') 
            grho_refh_guess, phi_guess, drho_guess = self.guess_from_props(prop_guess)
        if np.isnan([grho_refh_guess, phi_guess, drho_guess]).any():
            if isbulk: # bulk
                logger.info('use bulk guess') 
                grho_refh, phi, drho = self.bulk_props(delfstar)
                # dlam_refh = self.bulk_dlam_refh(grho_refh, phi)
            else: # thin layer
                logger.info('use thin film guess') 
                # logger.info('use thin film guess') 
                    # logger.info('use thin film guess') 
                # logger.info('use thin film guess') 
                grho_refh, phi, drho, dlam_refh = self.thinfilm_guess(delfstar, nh)
            # replace nan guess values
        # use given value over write guess value
        if ~np.isnan(grho_refh_guess):
            grho_refh = grho_refh_guess
        if ~np.isnan(phi_guess):
            phi = phi_guess
        if ~np.isnan(drho_guess):
            drho = drho_guess

        '''
        use the value given in film calc layer as constrain (fixed prop)
        value of n is not considered here!
        '''
        vary_vars = {'grho': True, 'phi': True, 'drho': True}
        calc_material = self.get_calc_material(film)
        if any([True for key in calc_material if key in ('grho', 'phi', 'drho')]): 
            grho_refh_constrain, phi_constrain, drho_constrain = self.guess_from_props(calc_material)
            # conditions to set if the value to be vary
            if ~np.isnan(grho_refh_constrain):
                vary_vars['grho'] = False  
                grho_refh = grho_refh_constrain
                err_names.remove('grho_refh') # fixed value has no covar

            if ~np.isnan(phi_constrain):
                vary_vars['phi'] = False  
                phi = phi_constrain
                err_names.remove('phi')

            if ~np.isnan(drho_constrain):
                vary_vars['drho'] = False  
                drho = drho_constrain
                if 'drho' in err_names:
                    err_names.remove('drho')

        '''
        Build the function for solving 
        '''
        if self.fit_method == 'lmfit':
            logger.info('use lmfit')

            def residual(pars,):
                vals = pars.valuesdict()
                grho = vals['grho']
                phi = vals['phi']
                drho = vals['drho']
                layers = self.set_calc_layer_val(film, grho, phi, drho)
            
                if isbulk:
                    # n3 == self.refh !!
                    return [
                        np.real(self.calc_delfstar(self.refh, layers)) - np.real(delfstar[self.refh]),
                        np.real(self.calc_delfstar(self.refh, layers)) - np.real(delfstar[self.refh])
                    ] 
                else: # thin film
                    return [
                        np.real(self.calc_delfstar(n1, layers)) - np.real(delfstar[n1]),
                        np.real(self.calc_delfstar(n2, layers)) - np.real(delfstar[n2]),
                        np.imag(self.calc_delfstar(n3, layers)) - np.imag(delfstar[n3])
                    ]

            # set parameters for minimize
            params = Parameters()
            params.add('grho', value=grho_refh, vary=vary_vars['grho'], min=grho_refh_range[0], max=grho_refh_range[1])
            params.add('phi', value=phi, vary=vary_vars['phi'], min=phi_range[0], max=phi_range[1])
            if isbulk:
                params.add('drho', value=drho, vary=False)
                err_names = err_names[0:1]
            else:
                params.add('drho', value=drho, vary=vary_vars['drho'], min=phi_range[0], max=phi_range[1])

        else: # use scipy
            if isbulk: # bulk              
                # use bounds for grho and phi only
                lb = lb[0:-1]
                ub = ub[0:-1]

                # initial value
                x0 = np.array([grho_refh, phi])

                # remove drho from error names for error calculation
                err_names = err_names[0:1]

                # define the solution function
                def ftosolve(x):
                    layers = self.set_calc_layer_val(film, x[0], x[1], bulk_drho) # set inf to grho, phi, drho to x[0], x[1], respectively

                    # n3 == self.refh !!
                    calc_delfstar = self.calc_delfstar(self.refh, layers)
                    return ([
                        np.real(calc_delfstar) - np.real(delfstar[self.refh]),
                        np.imag(calc_delfstar) - np.imag(delfstar[self.refh])
                    ])
            else: # thin layer
                logger.info('use thin film guess') 
                # initial value
                x0 = np.array([grho_refh, phi, drho])

                # define the solution function
                def ftosolve(x):
                    layers = self.set_calc_layer_val(film, x[0], x[1], x[2]) # set grho, phi, drho to x[0], x[1], x[2], respectively
                    return ([
                        np.real(self.calc_delfstar(n1, layers)) - np.real(delfstar[n1]),
                        np.real(self.calc_delfstar(n2, layers)) - np.real(delfstar[n2]),
                        np.imag(self.calc_delfstar(n3, layers)) - np.imag(delfstar[n3])
                    ])
            # if ~vary_vars['grho']:
            #     lb[0] = grho_refh
            #     ub[0] = grho_refh
            # if ~vary_vars['phi']:
            #     lb[1] = phi
            #     ub[1] = phi
            # if isbulk and ~np.isnan(drho_constrain):
            #     lb[2] = drho
            #     ub[2] = drho
            
        if ~np.isnan([grho_refh, phi, drho]).all() and grho_refh_range[0]<=grho_refh<=grho_refh_range[1] and phi_range[0]<=phi<=phi_range[1] and (isbulk or drho_range[0]<=drho<=drho_range[1]):
            logger.warning('film guess in range') 
            
            logger.info('delfstars: %s, %s, %s', delfstar[n1], delfstar[n2], delfstar[n3])
            
            # recalculate solution to give the uncertainty, if solution is viable
            if self.fit_method == 'lmfit':
                try:
                    soln = minimize(
                        residual,
                        params,
                        method='least_squares',
                        nan_policy='omit',
                    )
                    
                    grho_refh = soln.params.get('grho').value
                    phi = soln.params.get('phi').value
                    drho = soln.params.get('drho').value

                    # put the input uncertainties into a n element vector
                    delfstar_err = np.zeros(3)
                    delfstar_err[0] = np.real(self.fstar_err_calc(delfstar[n1]))
                    delfstar_err[1] = np.real(self.fstar_err_calc(delfstar[n2]))
                    delfstar_err[2] = np.imag(self.fstar_err_calc(delfstar[n3]))

                    covar = soln.covar
                    logger.info('covar %s', covar) 
                    
                    for i, nm in enumerate(err_names):
                        err[nm] = covar[i, :].sum()
                        err[nm] = np.sqrt(err[nm]) if err[nm] >= 0 else 0 # sometimes covar<0
                except:
                    logger.exception('error occurred while solving the thin film.')
            else: # scipy
                logger.info('x0: %s, lb: %s, ub: %s', x0, lb, ub)
                try:
                    soln = optimize.least_squares(ftosolve, x0, bounds=(lb, ub))

                    grho_refh = soln['x'][0]
                    phi = soln['x'][1]
                    if isbulk: # bulk
                        drho = bulk_drho
                    else:
                        logger.info('drho guess: %s', drho)
                        drho = soln['x'][2]
                        logger.info('drho recalc: %s', drho)

                    # put the input uncertainties into a n element vector
                    if isbulk: # bulk
                        delfstar_err = np.zeros(2)
                        delfstar_err[0] = np.real(self.fstar_err_calc(delfstar[n3]))
                        delfstar_err[1] = np.imag(self.fstar_err_calc(delfstar[n3]))
                    else: # thin film
                        delfstar_err = np.zeros(3)
                        delfstar_err[0] = np.real(self.fstar_err_calc(delfstar[n1]))
                        delfstar_err[1] = np.real(self.fstar_err_calc(delfstar[n2]))
                        delfstar_err[2] = np.imag(self.fstar_err_calc(delfstar[n3]))

                    jac = soln['jac']
                    # logger.info('jac %s', jac) 
                        # logger.info('jac %s', jac) 
                    # logger.info('jac %s', jac) 
                    try:
                        deriv = np.linalg.inv(jac)
                        # logger.info('jac_inv %s', jac_inv) 
                            # logger.info('jac_inv %s', jac_inv) 
                        # logger.info('jac_inv %s', jac_inv) 
                    except:
                        logger.warning('set deriv to 0') 
                        deriv = np.zeros(jac.shape)
                    
                    for i, nm in enumerate(err_names):
                        err[nm] = 0 # initialize error
                        for ii in np.arange(jac.shape[0]):
                            err[nm] += (deriv[i, ii] * delfstar_err[ii])**2
                        err[nm] = np.sqrt(err[nm]) 
                except:
                    logger.exception('error occurred while solving the thin film.')

            # update calc layer prop
            film = self.set_calc_layer_val(film, grho_refh, phi, drho)
            # logger.info('film after 2nd sol %s', film) 
                # logger.info('film after 2nd sol %s', film) 
            # logger.info('film after 2nd sol %s', film) 
            
            dlam_refh = self.calc_dlam(self.refh, self.get_calc_material(film))
            # comment above line to use the dlam_refh from soln

        else:
            logger.warning('film guess out of range') 
            grho_refh, phi, drho, dlam_refh = np.nan, np.nan, np.nan, np.nan # set the used values back to nan
            return [
                dict(
                    grho_refh=grho_refh,
                    phi=phi, 
                    drho=drho, 
                    dlam_refh=dlam_refh, 
                    rd_exp=rd_exp,
                    rh_exp=rh_exp,
                    rd_calc=rd_calc,
                    rh_calc=rh_calc,
                    err=err, 
                    nh=nh,
                    delfstar=delfstar,
                    delfstar_calc=delfstar_calc,
                    normdelfstar_calcs=normdelfstar_calcs
                ), 
                props
            ] 

        ###############################
        # for easier use out of GUI, here we calculate all values into dict
        ###############################

        # add nan to fill delfstar with nhs no values 
        for n in np.arange(1, max(nh_interests)+1, 2):
            if n not in delfstar:
                delfstar[n] = np.nan

        # these values are used for calculating others
        delfstar_calc = {n: self.calc_delfstar(n, film) for n in list(set(nh_interests) | set(nh))} # we need the harmonics in nh for other calculation
        normdelfstar_calcs = {n: self.normdelfstar(n, dlam_refh, phi) for n in nh_interests} # calculated normalized delfstar
        # normdelfstar_calcs = {n: np.real(delfstar_calc[n]) / delfsn[n] for n in nh_interests} # this is a test. it should be the same as normdelf_calcs[nh2i(n)] NOTE: they are not the same as tested
        rd_calc = self.rd_from_delfstar(nh[2], delfstar_calc) # single value to nh[2].
        rh_calc = self.rh_from_delfstar(nh, delfstar_calc)

        if brief_report:
            return [
                dict(
                    grho_refh=grho_refh,
                    phi=phi, 
                    drho=drho, 
                    dlam_refh=dlam_refh, 
                    rd_exp=rd_exp,
                    rh_exp=rh_exp,
                    rd_calc=rd_calc,
                    rh_calc=rh_calc,
                    err=err, 
                    nh=nh,
                    delfstar=delfstar,
                    delfstar_calc=delfstar_calc,
                    normdelfstar_calcs=normdelfstar_calcs
                ), 
                props
            ] 


        # harmonic depended variables
        props['delfsn'] = {n: self.sauerbreyf(n, drho) for n in nh_interests} # fsn from sauerbrey eq

        props['delf_calcs'] = {n: np.real(delfstar_calc[n]) for n in nh_interests}
        props['delg_calcs'] = {n: np.imag(delfstar_calc[n]) for n in nh_interests}
        props['delfn_calcs'] = {n: props['delf_calcs'][n] / n for n in nh_interests}
        props['delgn_calcs'] = {n: props['delg_calcs'][n] / n for n in nh_interests}

        props['delD_calcs'] = {n: self.convert_gamma_to_D(props['delg_calcs'][n], n) for n in nh_interests}


        props['rd_calcs'] = {n: self.rd_from_delfstar(n, delfstar_calc) for n in nh_interests}
        
        props['dlams'] = {n: self.dlam(n, dlam_refh, phi) for n in nh_interests}
        # props['dlams'] = {n: self.calc_dlam(n, film) for n in nh_interests} # more calculation

        props['grhos'] = {n: self.grho(n, grho_refh, phi) for n in nh_interests}
        props['grhos_err'] = {n: self.grho(n, err['grho_refh'], phi) for n in nh_interests} # assume errors follow power law, too

        props['etarhos'] = {n: self.etarho(n, props['grhos'][n]) for n in nh_interests}
        props['etarhos_err'] = {n: self.etarho(n, props['grhos_err'][n]) for n in nh_interests}

        props['lamrhos'] = {n: self.calc_lamrho(n, props['grhos'][n], phi) for n in nh_interests}

        if self.isbulk(rd_exp, bulklimit):
            props['delrhos'] = {n: self.delrho_bulk(n, delfstar) for n in nh_interests}
        else:
            props['delrhos'] = {n: self.calc_delrho(n,  props['grhos'][n], phi) for n in nh_interests}

        props['normdelf_calcs'] = {n: np.real(normdelfstar_calcs[n]) for n in nh_interests} 
        props['normdelg_calcs'] = {n: np.imag(normdelfstar_calcs[n]) for n in nh_interests} 
        props['normdelf_exps'] = {n: np.real(delfstar[n]) / props['delfsn'][n] for n in nh_interests}
        props['normdelg_exps'] = {n: np.imag(delfstar[n]) / props['delfsn'][n] for n in nh_interests}

        # variables w/ single values
        # we use repeated value for each harmonic to keep the same structure as other variables.
        props['drho'] = {n: drho for n in nh_interests}
        props['drho_err'] = {n: err['drho'] for n in nh_interests}
        props['phi'] = {n: min(np.pi/2, phi) for n in nh_interests}
        props['phi_err'] = {n: err['phi'] for n in nh_interests}

        props['rd_calc'] = {n: rh_calc for n in nh_interests}
        props['rh_calc'] = {n: rh_calc for n in nh_interests}
        
        # {n:  for n in nh_interests}
        return [
            dict(
                grho_refh=grho_refh,
                phi=phi, 
                drho=drho, 
                dlam_refh=dlam_refh, 
                rd_exp=rd_exp,
                rh_exp=rh_exp,
                rd_calc=rd_calc,
                rh_calc=rh_calc,
                err=err, 
                nh=nh,
                delfstar=delfstar,
                delfstar_calc=delfstar_calc,
                normdelfstar_calcs=normdelfstar_calcs
            ), 
            props
        ] 


    def all_nhcaclc_harm_not_na(self, nh, qcm_queue):
        '''
        check if all harmonics in nhcalc are not na
        nh: list of strings
        qcm_queue: qcm data (df) of a single queue
        return: True/False
        '''
        # logger.info(nh) 
        # logger.info(nh2i(nh[0])) 
        # logger.info(qcm_queue.delfstars.values) 
        # logger.info(qcm_queue.delfstars.iloc[0]) 
        if np.isnan(qcm_queue.delfstars.iloc[0][nh2i(nh[0])].real) or np.isnan(qcm_queue.delfstars.iloc[0][nh2i(nh[1])].real) or np.isnan(qcm_queue.delfstars.iloc[0][nh2i(nh[2])].imag):
            return False
        else:
            return True
        
        # for h in set(nh):
        #     if np.isnan(qcm_queue.delfstar[nh2i(h)]): # both real and imag are not nan
        #         return False
        # return True


    def analyze(self, nhcalc, queue_ids, qcm_df, mech_df):
        # sample, parms
        '''
        calculate with qcm_df and save to mech_df
        '''
        nh = nhcalc2nh(nhcalc) # list of harmonics (int) in nhcalc
        for queue_id in queue_ids: # iterate all ids
            # logger.info('queue_id %s', queue_id) 
            # logger.info('qcm_df %s', qcm_df) 
            # logger.info(type(qcm_df)) 
            # queue index
            idx = qcm_df[qcm_df.queue_id == queue_id].index.astype(int)[0]
            # qcm data of queue_id
            qcm_queue = qcm_df.loc[[idx], :].copy() # as a dataframe
            # mechanic data of queue_id
            mech_queue = mech_df.loc[[idx], :].copy()  # as a dataframe

            # obtain the solution for the properties
            if self.all_nhcaclc_harm_not_na(nh, qcm_queue):
                # solve a single queue
                mech_queue = self.solve_single_queue(nh, qcm_queue, mech_queue)
                # save back to mech_df
                # logger.info(mech_df.loc[[idx], :].to_dict()) 
                # logger.info(mech_queue.to_dict()) 
                # set mech_queue index the same as where it is from for update
                # logger.info(mech_df.delg_calcs) 
                mech_queue.index = [idx]
                mech_df.update(mech_queue)
                # logger.info(mech_df) 
                # logger.info(mech_df.delg_calcs) 
            else:
                # since the df already initialized with nan values, nothing todo
                pass
        return mech_df


    def convert_mech_unit(self, mech_df):
        '''
        convert unit of grho, phi, drho from IS to those convient to use
        input: df or series 
        '''
        # logger.info(type(mech_df)) 
        # logger.info(mech_df) 
        df = mech_df.copy()
        cols = mech_df.columns
        for col in cols:
            # logger.info(col) 
            if any([st in col for st in ['drho', 'lamrho', 'delrho', 'sauerbreyms']]): # delta m
                # logger.info('x1000') 
                df[col] = df[col].apply(lambda x: list(np.array(x) * 1000) if isinstance(x, list) else x * 1000) # from m kg/m3 to um g/cm3
            elif any([st in col for st in ['grho', 'etarho']]):
                # logger.info('x1/1000') 
                df[col] = df[col].apply(lambda x: list(np.array(x) / 1000) if isinstance(x, list) else x / 1000) # from Pa kg/m3 to Pa g/cm3 (~ Pa)
            elif 'phi' in col:
                # logger.info('rad2deg') 
                df[col] = df[col].apply(lambda x: list(np.rad2deg(x)) if isinstance(x, list) else np.rad2deg(x)) # from rad to deg
            elif 'D_' in col: # dissipation
                df[col] = df[col].apply(lambda x: list(np.array(x) * 1e6) if isinstance(x, list) else x * 1e6) # D to ppm
            else:
                # logger.info('NA') 
                pass
        return df


    def convert_mech_unit_data(self, data, varname):
        '''
        convert unit of grho, phi, drho from IS to those convient to use
        data: int, array, list
        varname: str

        '''
        if isinstance(data, list):
            data = np.array(data)

        # logger.info(col) 
        if varname in ['drho', 'lamrho', 'delrho', 'sauerbreyms']: # delta m
            # logger.info('x1000') 
            data = data * 1000 # from m kg/m3 to um g/cm3
        elif varname in ['grho', 'etarho']:
            # logger.info('x1/1000') 
            data = data / 1000 # from Pa kg/m3 to Pa g/cm3 (~ Pa)
        elif 'phi' in varname:
            # logger.info('rad2deg') 
            data = np.rad2deg(data) # from rad to deg
        elif 'D' in varname: # dissipation
            data = data * 1e6 # D to ppm
        else:
            # logger.info('NA') 
            pass
        return data


    def single_harm_data(self, var, qcm_df):
        '''
        get variables calculate from single harmonic
        variables listed in DataSaver.mech_keys_multiple
        to keep QCM and DataSaver independent from each other, we don't use import for each other
        ['delf_calcs', 'delg_calcs', 'normdelfs', 'rds']
        '''
        # logger.info(var) 
        if var == 'delf_calcs':
            return qcm_df.delfs
        if var == 'delg_calcs':
            return qcm_df.delgs
        if var == 'normdelfs':
            return None # TODO
        if var == 'rds':
            # rd_from_delfstar(self, n, delfstar)
            s = qcm_df.delfstars.copy()
            s.apply(lambda delfstars: [self.rd_from_delfstar(n, delfstars) for n in range(len(delfstars))])
            return s


    ####### film structure functions ######

    def get_calc_layer_num(self, film):
        ''' 
        film ={
            0:{'calc': True/False,
                'grho': float, # in Pa kg/m^3 = 1/1000 Pa g/cm^3
                'phi': float, # in rad
                'drho': float, # in m kg/m^3 = 1000 um g/cm^3
               },
            1:
            2:
            ...
        }
        This function return n (int) of the layer with 'calc' is True
        '''
        calc_n = None
        for n, layer in film.items():
            if 'calc' in layer and layer['calc']:
                calc_n = n
                break
        
        return calc_n


    def get_calc_material(self, film):
        ''' 
        film ={
            0:{'calc': True/False,
                'grho': float, # in Pa g/cm^3
                'phi': float, # in rad
                'drho': float, # in um g/cm^3
               },
            1:
            2:
            ...
        }
        This function return the material of the layer 'calc' is True
            {'calc': True/False,
                'grho': float, # in Pa g/cm^3
                'phi': float, # in rad
                'drho': float, # in um g/cm^3
               }
        '''
        n = self.get_calc_layer_num(film)

        if n is None:
            return None
        else:
            return film[n]


    def get_calc_layer(self, film):
        ''' 
        film ={
            0:{'calc': True/False,
                'grho': float, # in Pa g/cm^3
                'phi': float, # in rad
                'drho': float, # in um g/cm^3
               },
            1:
            2:
            ...
        }
        This function return the film contain only the layer with 'calc' is True
        '''
        new_film = {}
        n = self.get_calc_layer_num(film)
        if n is not None:
            new_film[n] = film[n]

        return new_film


    def set_calc_layer_val(self, film, grho, phi, drho):
        ''' 
        film ={
            0:{ 'calc': True/False,
                'grho': float, # in Pa g/cm^3
                'phi':  float, # in rad
                'drho': float, # in um g/cm^3
               },
            1:
            2:
            ...
        }
        This function set the values of the layer with 'calc' is True
        '''
        if not film: # film has no layers
            # logger.info('build a single layer film') 
            film = self.build_single_layer_film() # make a single layer film

        calcnum = self.get_calc_layer_num(film)
        film[calcnum].update(grho=grho, phi=phi, drho=drho, n=self.refh)
        return film


    def set_layer_n_val(self, film, layer_n, **kwargs):
        ''' 
        film ={
            0:{ 'calc': True/False,
                'grho': float, # in Pa g/cm^3
                'phi':  float, # in rad
                'drho': float, # in um g/cm^3
               },
            1:
            2:
            ...
        }
        This function set the values of the layer n
        but will not check if the layer number is continuous, it is only as simple update of the dictionary of the layer n.
        layer_n: int
        '''
        film[layer_n].update(**kwargs)
        return film


    def build_single_layer_film(self, material=None):
        '''
        build a single layer film for calc
        material: dict of properties {'grho': ***, 'phi': ***, 'drho': ***, 'n': #}
        '''
        if material is None:
            return self.replace_layer_0_prop_with_known({0:{'calc': False}, 1:{'calc': True}})
        else:
            # set material as calc layer
            # material['calc'] = True # this change the input material value!! 
            new_film = self.replace_layer_0_prop_with_known({0:{'calc': False}, 1:{**material}})
            new_film[1]['calc'] = True
            return new_film


    def get_ref_layers(self, film):
        ''' 
        This function return the film contain all the layers with 'calc' is False
        '''
        new_film = {}
        for n, layer in film.items():
            if 'calc' not in layer:
                layer['calc'] = False
            if not layer['calc']:
                new_film[n] = layer.copy()
        for n in new_film.keys():
            new_film[n]['calc'] = True # we change 'calc' all True for calculating delfstar (self.calc_delfstar())
        return new_film


    def separate_calc_ref_layers(self, film):
        return self.get_calc_layer(film), self.get_ref_layers(film)


    def remove_layer_n(self, film, n):
        new_film = film.copy()
        new_film.pop(n, None)
        return new_film


    def remove_layer_0(self, film):
        new_film = film.copy()
        new_film.pop(0, None)
        return new_film


    def replace_layer_0_prop_with_known(self, film):
        '''
        this function replace layer 0 prop with prop_default['electrode']
        It will be replaced with better way
        '''
        new_film = film.copy()
        if 0 in new_film.keys():
            new_film[0] = prop_default['electrode']
            # new_film[0]['calc'] = False # this layer should not be the unkown layer generally

        for n, layer in new_film.items(): # add 'calc': False to dict if does not exist in layer
            if 'calc' not in layer:
                new_film[n]['calc'] = False

        return new_film

    
    ##################### springport functions ########################
    def gstar_maxwell(self, wtau):  
        ''' Maxwell element '''
        return 1j * wtau / (1 + 1j * wtau)


    # def gstar_kww_single(self, wtau, beta):  
    #     ''' Transform of the KWW function '''
    #     return wtau * (kwws(wtau, beta) + 1j * kwwc(wtau, beta))
    # gstar_kww = np.vectorize(gstar_kww_single)
    
    
    # use decortor, should be the same as above commented
    @np.vectorize
    def gstar_kww(self, wtau, beta):  
        ''' Transform of the KWW function '''
        return wtau * (kwws(wtau, beta) + 1j * kwwc(wtau, beta))


    def gstar_rouse(self, wtau, n_rouse):
        # make sure n_rouse is an integer if it isn't already
        n_rouse = int(n_rouse)

        rouse=np.zeros((len(wtau), n_rouse), dtype=complex)
        for p in 1+np.arange(n_rouse):
            rouse[:, p-1] = (wtau / p**2)**2/(1 + wtau / p**2)**2 + 1j*(wtau / p**2) / (1 + wtau / p**2)**2
        rouse = rouse.sum(axis=1) / n_rouse
        return rouse


    def springpot(self, w, g0, tau, beta, sp_type, **kwargs):
        # this function supports a combination of different springpot elments
        # combined in series, and then in parallel.  For example, if type is
        # [1,2,3],  there are three branches
        # in parallel with one another:  the first one is element 1, the
        # second one is a series comination of elements 2 and 3, and the third
        # one is a series combination of 4, 5 and 6.
        
        # specify which elements are kww or Maxwell elements
        kww = kwargs.get('kww',[])
        maxwell = kwargs.get('maxwell',[])

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
                sp_comp[:, i] = 1 / (g0[i] * self.gstar_maxwell(w * tau[i]))            
            elif i in kww:  #  kww (stretched exponential) elment
                sp_comp[:, i] = 1 / (g0[i] * self.gstar_kww(w * tau[i], beta[i]))
            else:  # power law springpot element
                sp_comp[:, i] = 1 / (g0[i] * (1j * w * tau[i]) **beta[i])

        # sp_vec keeps track of the beginning and end of each branch
        sp_vec = np.append(0, sp_type.cumsum())
        for i in np.arange(n_br):
            sp_i = np.arange(sp_vec[i], sp_vec[i+1])
            # branch compliance obtained by summing compliances within the branch
            br_g[:, i] = 1 / sp_comp[:, sp_i].sum(1)

        # now we sum the stiffnesses of each branch and return the result
        return br_g.sum(1)


    def vogel(self, T, Tref, B, Tinf):
        logaT = -B / (Tref - Tinf) + B / (T - Tinf)
        return logaT


if __name__ == '__main__':
    qcm = QCM()
    qcm.f1 = 5e6 # Hz
    qcm.refh = 3

    # delrho = qcm.calc_delrho(qcm.refh, 96841876.71961978, 1.5707963266378169)
    # print(delrho)
    # exit(0)

    nh = [3,5,3]

    samp = 'PS'

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


    grho_refh, phi, drho, dlam_refh, err = qcm.solve_general_delfstar_to_prop(nh, delfstar, film, calctype='LL', bulklimit=.5)

    print('drho', drho)
    print('grho_refh', grho_refh)
    print('phi', phi)
    print('dlam_refh', phi)
    print('err', err)


