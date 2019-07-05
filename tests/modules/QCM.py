'''
This module is a modified version of QCM_functions.
It is used for UI, but you can also use it for data analysis.
The input and return data are all in form of DataFrame.
This module doesn't have plotting functions.

NOTE: Differnt to other modules, the harmonics used in this module are all INT.
'''


import numpy as np
from scipy import optimize
from lmfit import Minimizer, minimize, Parameters, fit_report, printfuncs

import pandas as pd


# variable limitions
dlam_refh_range = (0, 10) # (0, 5)
drho_range = (0, 2e-2) # m kg/m^3 = 1000 um g/cm^3 = 1000 g/m^2
# drho_range = (0, np.inf) # m kg/m^3 = 1000 um g/cm^3 = 1000 g/m^2
grho_refh_range = (1e4, 1e14) # Pa kg/m^3 = 1/1000 Pa g/cm^3
phi_range = (0, np.pi/2) # rad 

#  Zq (shear acoustic impedance) of quartz = rho_q * v_q
Zq = {
    'AT': 8.84e6,  # kg m−2 s−1
    'BT': 0e6,  # kg m−2 s−1
}

e26 = 9.65e-2 # piezoelectric stress coefficient (e = 9.65·10−2 C/m2 for AT-cut quartz)
d26 = 3.1e-12 # piezoelectric strain coefficient (d = 3.1·10−9 m/V for AT-cut quartz)
g0 = 10 # 10 Hz, Half bandwidth (HWHM) of unloaed resonator (intrinsic dissipation on crystalline quartz)
dq = 330e-6  # only needed for piezoelectric stiffening calc.
epsq = 4.54
eps0 = 8.8e-12
C0byA = epsq * eps0 / dq 

prop_default = {
    'electrode': {'drho': 2.8e-6, 'grho': 3.0e17, 'phi': 0, 'n': 3},  # n here is relative harmonic.
    'air':       {'drho': 0, 'grho': 0, 'phi': 0, 'n': 1}, #???
    'water':     {'drho': np.inf, 'grho': 1e8, 'phi': np.pi / 2, 'n': 1}, #?? water ar R.T.
}

# fit_method = 'lmfit'
fit_method = 'scipy'

def nh2i(nh):
    '''
    convert harmonic (str) to index (int) 
    since only odd harmonics are stored in list
    '''
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
        self.g_err_min = 10 # error floor for gamma
        self.f_err_min = 50 # error floor for f
        self.err_frac = 3e-2 # error in f or gamma as a fraction of gamma

        self.refh = None # reference harmonic for calculation
        # default values
        # self.nhcalc = '355' # harmonics used for calculating
        # self.nhplot = [1, 3, 5] # harmonics used for plotting (show calculated data)
        
        # self.air_default = air_default
        # self.electrode_default = electrode_default


    def get_prop_by_name(self, name):
        return prop_default.get(name, prop_default['air']) # if name does not exist, use air ?


    def fstar_err_calc(self, fstar):
        ''' 
        calculate the error in delfstar
        fstar: complex number 
        '''
        # start by specifying the error input parameters
        fstar_err = np.zeros(1, dtype=np.complex128)
        fstar_err = (self.f_err_min + self.err_frac * np.imag(fstar)) + 1j*(self.g_err_min + self.err_frac * np.imag(fstar))
        return fstar_err
        # ?? show above both imag?


    def sauerbreyf(self, n, drho):
        ''' delf_sn from Sauerbrey eq'''
        # n = int(n) if isinstance(n, str) else n
        return 2 * n * self.f1**2 * drho / self.Zq


    def sauerbreym(self, n, delf):
        ''' mass from Sauerbrey eq'''
        return delf * self.Zq / (2 * n * self.f1**2)


    def grho(self, n, grho_refh, phi): # old func
        ''' grho of n_th harmonic'''
        return grho_refh * (n/self.refh) ** (phi)


    def grhostar(self, n, grho_refh, phi):
        return self.grho(n, grho_refh, phi) * np.exp(1j*phi)


    def grho_from_dlam(self, n, drho, dlam, phi):
        return (drho * n * self.f1 * np.cos(phi / 2) / dlam)**2


    def grho_bulk(self, delfstar, n=None):
        '''
        bulk model
        calculate grho reference to refh
        '''
        if n is None:
            n = self.refh
        return (np.pi * self.Zq * abs(delfstar[self.refh]) / self.f1) ** 2


    def phi_bulk(self, delfstar, n=None):
        '''
        bulk model
        calculate phi
        '''
        if n is None:
            n = self.refh
        return min(np.pi / 2, -2 * np.arctan(np.real(delfstar[n]) / np.imag(delfstar[n]))) # limit phi <= pi/2


    def etarho(self, n, grho_n):
        ''' 
        viscosity at nth harmonic 
        grho_n: grho at nth haromonic
        eta = G / (2*pi*f)
        '''
        return grho_n / (2 * np.pi * n * self.f1)


    def calc_lamrho(self, n, grho, phi):
        '''
        calculate rho*lambda
        grho & phi of harmonic n
        '''
        return np.sqrt(grho) / (n * self.f1 * np.cos(phi / 2))


    def calc_delrho(self, n, grho, phi):
        '''
        decaa quarter of wavelength
        '''
        return self.calc_lamrho(n, grho, phi) / 4


    def D(self, n, drho, grho_refh, phi):
        return 2*np.pi*drho*n*self.f1*(np.cos(phi/2) - 1j * np.sin(phi/2)) / (self.grho(n, grho_refh, phi)) ** 0.5


    def DfromZ(self, n, drho, Zstar):
        return 2 * np.pi * n * self.f1 * drho / Zstar


    def zstarbulk(self, grhostar):
        return grhostar ** 0.5


    def zstarfilm(self, n, drho, grhostar):
        if grhostar == 0:
            answer = 0
        else:
            answer = self.zstarbulk(grhostar) * np.tan(2 * np.pi * n * self.f1 * drho / self.zstarbulk(grhostar)) 
        return answer


    def rstar(self, n, drho, grho_refh, phi, overlayer={'drho': 0, 'gho_refh': 0, 'phi': 0}):
        # overlayer is dictionary with drho, grho_refh and phi
        grhostar_1 = self.grhostar(n, grho_refh, phi)
        grhostar_2 = self.grhostar(n, overlayer.get('grho_refh', 0), overlayer.get('phi', 0))
        zstar_1 = self.zstarbulk(grhostar_1)
        zstar_2 = self.zstarfilm(n, overlayer.get('drho', 0), grhostar_2)   
        return zstar_2 / zstar_1
    
    
    # calcuated complex frequency shift for single layer
    def delfstarcalc(self, n, drho, grho_refh, phi, overlayer):
        rstar = self.rstar(n, drho, grho_refh, phi, overlayer)
        # overlayer is dictionary with drho, grho_refh and phi
        calc = -(self.sauerbreyf(n, drho)*np.tan(self.D(n, drho, grho_refh, phi)) / self.D(n, drho, grho_refh, phi))*(1-rstar**2) / (1+1j*rstar*np.tan(self.D(n, drho, grho_refh, phi)))
        
        # handle case where drho = 0, if it exists
        calc[np.where(drho==0)]=0
        return calc


    # calculated complex frequency shift for bulk layer
    def delfstarcalc_bulk(self, n, grho_refh, phi):
        return ((self.f1*np.sqrt(self.grho(n, grho_refh, phi)) / (np.pi*self.Zq)) * (-np.sin(phi/2)+ 1j * np.cos(phi/2)))


    def d_lamcalc(self, n, drho, grho_refh, phi):
        return drho*n*self.f1*np.cos(phi/2) / np.sqrt(self.grho(n, grho_refh, phi))


    def thin_film_gamma(self, n, drho, jdprime_rho):
        return 8*np.pi ** 2*n ** 3*self.f1 ** 4*drho ** 3*jdprime_rho / (3*self.Zq)
        # same master expression, replacing grho3 with jdprime_rho3


    def grho_refh(self, jdprime_rho_refh, phi):
        return np.sin(phi)/jdprime_rho_refh


    ###### new funcs #########
    def grho_from_material(self, n, material): # func new
        grho_refh = material['grho']
        phi = material['phi']
        return grho_refh * (n/self.refh) ** (phi)


    def calc_D(self, n, material, delfstar):
        '''
        take the material from a single layer {'drho': 0, 'grho': 0, 'phi': 0, 'n': 1}
        '''
        # print('material', material) #testprint
        # print('delfstar', delfstar) #testprint
        drho = material['drho']
        # set switch to handle as where drho = 0
        if drho == 0:
            return 0
        else:
            return 2 * np.pi * (n * self.f1 + delfstar) * drho / self.zstar_bulk(n, material)
            #?? should delfstar be delfstar.real?
            #?? can we replace (n * self.f1 + delfstar) with fstar.real


    def zstar_bulk(self, n, material):
        # print('material', material) #testprint
        grho_refh = material['grho']
        refh = material['n']
        phi = material['phi']
        grho = grho_refh * (n / refh) ** (phi / (np.pi / 2))  #check for error here
        grhostar = grho * np.exp(1j * phi)
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
        # print('N', N) #testprint
        Z = {}; D = {}; L = {}; S = {}

        # we use the matrix formalism to avoid typos.
        for i, layer_n in enumerate(sorted(layers.keys())[:-1]): # iterate except the last layer
            i += 1 # start from 1, so add 1 to each i
            Z[i] = self.zstar_bulk(n, layers[layer_n])
            D[i] = self.calc_D(n, layers[layer_n], delfstar)
            L[i] = np.array([
                [np.cos(D[i]) + 1j * np.sin(D[i]), 0], 
                [0, np.cos(D[i]) - 1j * np.sin(D[i])]
            ], dtype=complex)

            # print('i', i) #testprint
            # print('Zi', Z[i]) #testprint
            # print('Di', D[i]) #testprint
            # print('Li', L[i]) #testprint

        # get the terminal matrix from the properties of the last layer
        top_n = sorted(layers.keys())[-1]
        D[N] = self.calc_D(n, layers[top_n], delfstar)
        Zf_N = 1j * self.zstar_bulk(n, layers[top_n]) * np.tan(D[N])

        # print('top_n', top_n) #testprint
        # print('DN', D[N]) #testprint
        # print('Zf_n', Zf_N) #testprint

        # if there is only one layer, we're already done
        if N == 1:
            return Zf_N

        Tn = np.array([
            [1 + Zf_N / Z[N-1], 0],
            [0, 1 - Zf_N / Z[N-1]]
            ], dtype=complex)

        # print('L[N-1]', L[N-1]) #testprint
        # print('Tn', Tn) #testprint

        uvec = L[N-1] @ Tn @ np.array([[1.], [1.]])
        # print('uvec', uvec) #testprint

        for i in np.arange(N-2, 0, -1):
            S[i] = np.array([
                [1 + Z[i+1] / Z[i], 1 - Z[i+1] / Z[i]],
                [1 - Z[i+1] / Z[i], 1 + Z[i+1] / Z[i]]
            ])
            # print('S[i]', S[i]) #testprint
            uvec = L[i] @ S[i] @ uvec
            # print('uvec', uvec) #testprint

        rstar = uvec[1,0] / uvec[0,0]
        # print('rstar', rstar) #testprint
        # print('ZL', Z[1] * (1 - rstar) / (1 + rstar)) #testprint

        return Z[1] * (1 - rstar) / (1 + rstar)


    def calc_delfstar(self, n, layers, calctype):
        '''
        refto air (0) or knowlayers (1)
        '''
        refto = 0
        if not layers: # layers is empty {}
            return np.nan

        # there is data
        if calctype.upper() == 'SLA':
            # use the small load approximation in all cases where calctype
            # is not explicitly set to 'LL'

            ZL = self.calc_ZL(n, self.remove_layer_0(layers), 0)
            if refto == 1:
                ZL_ref = self.calc_ZL(n, self.remove_layer_0(self.get_ref_layers(layers)), 0)
                del_ZL = ZL - ZL_ref
    
                return self.calc_delfstar_sla(del_ZL)
            else:
                return self.calc_delfstar_sla(ZL)

        elif calctype.upper() == 'LL':
            # this is the most general calculation
            # use defaut electrode if it's not specified
            if 0 not in layers: 
                layers[0] = prop_default['electrode']

            ZL_all = self.calc_ZL(n, layers, 0)
            delfstar_sla_all = self.calc_delfstar_sla(ZL_all)
            
            def solve_Zmot(x):
                delfstar = x[0] + 1j * x[1]
                Zmot = self.calc_Zmot(n, layers, delfstar)
                return [np.real(Zmot), np.imag(Zmot)]

            sol = optimize.root(solve_Zmot, [np.real(delfstar_sla_all), np.imag(delfstar_sla_all)])
            dfc = sol.x[0] + 1j * sol.x[1]
            # print('dfc', dfc) #testprint

            if refto == 1:
                layers_ref = self.get_ref_layers(layers)
                ZL_ref = self.calc_ZL(n, layers_ref, 0)
                delfstar_sla_ref = self.calc_delfstar_sla(ZL_ref)

                def solve_Zmot_ref(x):
                    delfstar = x[0] + 1j * x[1]
                    Zmot = self.calc_Zmot(n, layers_ref, delfstar)
                    return [np.real(Zmot), np.imag(Zmot)]

                sol = optimize.root(solve_Zmot_ref, [np.real(delfstar_sla_ref), np.imag(delfstar_sla_ref)])
                dfc_ref = sol.x[0] + 1j * sol.x[1]
                # print('dfc_ref', dfc_ref) #testprint

                return dfc - dfc_ref
            else:
                return dfc
        else:
            return np.nan


    def calc_Zmot(self, n, layers, delfstar):
        om = 2 * np.pi * (n * self.f1 + delfstar)
        Zqc = self.Zq * (1 + 1j * 2 * g0 / (n * self.f1))
        ZC0byA = C0byA / (1j*om)
        # can always be neglected as far as I can tell
        ZPE = -(e26 / dq)**2 * ZC0byA  # ZPE accounts for piezoelectric stiffening anc

        self.drho_q = self.Zq / (2 * self.f1)
        Dq = om * self.drho_q / self.Zq
        secterm = -1j * Zqc / np.sin(Dq)
        ZL = self.calc_ZL(n, layers, delfstar)
        # eq. 4.5.9 in book
        thirdterm = ((1j * Zqc * np.tan(Dq/2))**-1 + (1j * Zqc * np.tan(Dq / 2) + ZL)**-1)**-1
        Zmot = secterm + thirdterm  + ZPE

        # print('Zmot shape', Zmot.shape) #testprint
        # print('Zmot', Zmot) #testprint
        return Zmot


    def calc_dlam(self, n, film):
        return np.real(self.calc_D(n, film, 0)) / (2 * np.pi)


    ##### end new funcs ######


    def dlam(self, n, dlam_refh, phi):
        return dlam_refh*(n/self.refh) ** (1-phi/np.pi)


    def normdelfstar(self, n, dlam_refh, phi):
        dlam_n = self.dlam(n, dlam_refh, phi)
        return -np.tan(2*np.pi*dlam_n*(1-1j*np.tan(phi/2))) / (2*np.pi*dlam_n*(1-1j*np.tan(phi/2)))


    def calc_drho(self, n1, delfstar, dlam_refh, phi):
        return self.sauerbreym(n1, np.real(delfstar[n1])) / np.real(self.normdelfstar(n1, dlam_refh, phi))


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
        return -np.imag(delfstar[nh[2]]) / np.real(delfstar[nh[2]])


    def rd_from_delfstar(self, n, delfstar):
        ''' dissipation ratio calculated for the relevant harmonic '''
        if np.real(delfstar[n]) == 0:
            return np.nan
        else:
            return -np.imag(delfstar[n])/np.real(delfstar[n])


    def bulk_guess(self, delfstar):
        ''' get the bulk solution for grho and phi '''
        grho_refh = self.grho_bulk(delfstar)
        phi = self.phi_bulk(delfstar)

        # calculate rho*lambda
        lamrho_refh = self.calc_lamrho(self.refh, grho_refh, phi)
        # print('grho_refh', grho_refh) #testprint
        # print('lamrho_refh', lamrho_refh) #testprint
        # we need an estimate for drho. We only use this approach if it is
        # reasonably large. We'll put it at the quarter wavelength condition for now

        drho = lamrho_refh / 4

        film = {'drho':drho, 'grho':grho_refh, 'phi':phi, 'n': self.refh}
        
        dlam_refh = self.calc_dlam(self.refh, film)

        return [dlam_refh, min(phi, np.pi/2)]

    
    def bulk_props(self, delfstar, n=None):
        # get the bulk solution for grho and phi
        #??
        return [
            np.inf, # drho
            self.grho_bulk(delfstar, n), # grho
            self.phi_bulk(delfstar, n), # phi
        ]


    def guess_from_props(self, film):
        dlam_refh = self.calc_dlam(self.refh, film)
        return [dlam_refh, phi]


    def thinfilm_guess(self, delfstar):
        ''' 
        really a placeholder function until we develop a more creative strategy
        for estimating the starting point 
        '''
        return [0.05, np.pi/180*5] # in rad


    ########################################################



    ########################################################


    def solve_single_queue_to_prop(self, nh, qcm_queue, calctype='SLA', film={}):
        '''
        solve the property of a single test.
        nh: list of int
        qcm_queue:  QCM data. df (shape[0]=1) 
        calctype: 'SLA' / 'LL'
        film: dict of the film layers information
        return drho, grho_refh, phi, dlam_ref, err
        '''
        # get fstar
        fstars = qcm_queue.fstars.iloc[0] # list
        # get delfstar
        delfstars = qcm_queue.delfstars.iloc[0] # list
        # print('fstars', fstars) #testprint
        # print(delfstars) #testprint
        # convert list to dict to make it easier to do the calculation
        # fstar = {int(i*2+1): fstar[i] for i, fstar in enumerate(fstars)}
        delfstar = {int(i*2+1): dfstar for i, dfstar in enumerate(delfstars)}
        # print(delfstar) #testprint

        # set f1
        f0s = qcm_queue.f0s.iloc[0]
        if np.isnan(f0s).all():
            self.f1 = np.nan
        else:
            first_notnan = np.argwhere(~np.isnan(f0s))[0][0] # find out index of the first freq is not nan
            # use this value calculate f1 = fn/n (in case f1 is not recorded)
            self.f1 = f0s[first_notnan] / (first_notnan * 2 + 1)
        # print('f1', self.f1) #testprint

        # fstar_err ={}
        # for n in nhplot: 
        #     fstar_err[n] = self.fstar_err_calc(fstar[n])


        # set up to handle one or two layer cases
        # overlayer set to air if it doesn't exist in soln_input
        #! check the last layer. it should be air or water or so with inf thickness
        #! if rd > 0.5 and nl < layers, set out layers to {0, 0, 0}

        drho, grho_refh, phi, dlam_refh, err = self.solve_general(nh, delfstar, calctype, film, prop_guess={})

        return drho, grho_refh, phi, dlam_refh, err


    def solve_single_queue(self, nh, qcm_queue, mech_queue, calctype='SLA', film={}):
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
        # print('calctype', calctype) #testprint
        #TODO this may be replaced
        film = self.replace_layer_0_prop_with_known(film)

        # print('film before calc', film) #testprint
        drho, grho_refh, phi, dlam_refh, err = self.solve_single_queue_to_prop(nh, qcm_queue, calctype=calctype, film=film)

        # update calc layer prop
        film = self.set_calc_layer_val(film, drho, grho_refh, phi)
        # print('film after calc', film) #testprint

        # now back calculate delfstar, rh and rd from the solution
        # get the marks [1st, 3rd, 5th, ...]
        marks = qcm_queue.marks.iloc[0]

        delfstars = qcm_queue.delfstars.iloc[0] # list
        delfstar = {int(i*2+1): dfstar for i, dfstar in enumerate(delfstars)}

        # find where the mark is not nan or None
        nhplot = [i*2+1 for i, mark in enumerate(marks) if (not np.isnan(mark)) and (mark is not None)]
        
        delfstar_calc = {}
        delfsn = {i*2+1: self.sauerbreyf(i*2+1, drho) for i, mark in enumerate(marks)} # fsn from sauerbrey eq
        normdelfstar_calcs = {}

        delf_calcs = mech_queue.delf_calcs.iloc[0].copy()
        delg_calcs = mech_queue.delg_calcs.iloc[0].copy()
        rd_exps = mech_queue.rd_exps.iloc[0].copy()
        rd_calcs = mech_queue.rd_calcs.iloc[0].copy()
        dlams = mech_queue.dlams.iloc[0].copy()
        lamrhos = mech_queue.lamrhos.iloc[0].copy()
        delrhos = mech_queue.delrhos.iloc[0].copy()
        grhos = mech_queue.dlams.iloc[0].copy()
        grhos_err = mech_queue.dlams.iloc[0].copy()
        etarhos = mech_queue.etarhos.iloc[0].copy()
        etarhos_err = mech_queue.etarhos_err.iloc[0].copy()
        normdelf_exps = mech_queue.normdelf_exps.iloc[0].copy()
        normdelf_calcs = mech_queue.normdelf_calcs.iloc[0].copy()
        normdelg_exps = mech_queue.normdelg_exps.iloc[0].copy()
        normdelg_calcs = mech_queue.normdelg_calcs.iloc[0].copy()
        # print('delf_calcs', delf_calcs) #testprint
        # print(type(delf_calcs)) #testprint
        for n in nhplot:
            delfstar_calc[n] = self.calc_delfstar(n, film, calctype)
            delf_calcs[nh2i(n)] = np.real(delfstar_calc[n])
            delg_calcs[nh2i(n)] = np.imag(delfstar_calc[n])
            
            rd_calcs[nh2i(n)] = self.rd_from_delfstar(n, delfstar_calc)

            rd_exps[nh2i(n)] = self.rd_from_delfstar(n, delfstar)

            dlams[nh2i(n)] = self.dlam(n, dlam_refh, phi)
            # dlams[nh2i(n)] = self.calc_dlam(n, film) # more calculation
            grhos[nh2i(n)] = self.grho(n, grho_refh, phi)
            grhos_err[nh2i(n)] = self.grho(n, err['grho_refh'], phi) # supose errors follow power law, too
            etarhos[nh2i(n)] = self.etarho(n, grhos[nh2i(n)])
            etarhos_err[nh2i(n)] = self.etarho(n, grhos_err[nh2i(n)]) # supose errors follow power law, too
            lamrhos[nh2i(n)] = self.calc_lamrho(n, grhos[nh2i(n)], phi) 
            delrhos[nh2i(n)] = self.calc_delrho(n, grhos[nh2i(n)], phi) 

            normdelfstar_calcs[n] = self.normdelfstar(n, dlam_refh, phi) # calculated normalized delfstar
            # normdelf_exps[nh2i(n)] = np.real(delfstar_calc[n]) / delfsn[n] # this is a test. it should be the same as normdelf_calcs[nh2i(n)] NOTE: they are not the same as tested
            normdelf_exps[nh2i(n)] = np.real(delfstar[n]) / delfsn[n] 
            normdelf_calcs[nh2i(n)] = np.real(normdelfstar_calcs[n])
            # normdelg_exps[nh2i(n)] = np.imag(delfstar_calc[n]) / delfsn[n] # this is a test. it should be the same as normdelg_calcs[nh2i(n)] NOTE: they are not the same as tested
            normdelg_exps[nh2i(n)] = np.imag(delfstar[n]) / delfsn[n] 
            normdelg_calcs[nh2i(n)] = np.imag(normdelfstar_calcs[n])
            # normdelg_calcs[nh2i(n)] = np.imag(delfstar_calc[n]) / delfsn[n] # test

        rh_exp = self.rh_from_delfstar(nh, delfstar)
        rh_calc = self.rh_from_delfstar(nh, delfstar_calc)
        # rh_calc = self.rhcalc(nh, dlam_refh, phi)
        # print('delf_calcs', delf_calcs) #testprint
        # print('delg_calcs', delg_calcs) #testprint

        # repeat values for single value
        tot_harms = len(delf_calcs)
        mech_queue['drho'] = [[drho] * tot_harms] # in kg/m2
        mech_queue['drho_err'] = [[err['drho']] * tot_harms] # in kg/m2
        mech_queue['phi'] = [[phi] * tot_harms] # in rad
        mech_queue['phi_err'] = [[err['phi']] * tot_harms] # in rad
        mech_queue['rh_exp'] = [[rh_exp] * tot_harms]
        mech_queue['rh_calc'] = [[rh_calc] * tot_harms]


        # multiple values in list
        mech_queue['grhos'] = [grhos] # in Pa kg/m3
        mech_queue['grhos_err'] = [grhos_err] # in Pa kg/m3 
        mech_queue['etarhos'] = [etarhos] # in Pa s kg/m3
        mech_queue['etarhos_err'] = [etarhos_err] # in Pa s kg/m3 
        mech_queue['dlams'] = [dlams] # in na
        mech_queue['lamrhos'] = [lamrhos] # in kg/m2
        mech_queue['delrhos'] = [delrhos] # in kg/m2
        
        mech_queue['delf_exps'] = qcm_queue['delfs']
        mech_queue['delf_calcs'] = [delf_calcs]
        mech_queue['delg_exps'] = qcm_queue['delgs']
        mech_queue['delg_calcs'] = [delg_calcs]
        mech_queue['normdelf_exps'] =[normdelf_exps]
        mech_queue['normdelf_calcs'] =[normdelf_calcs]
        mech_queue['normdelg_exps'] =[normdelg_exps]
        mech_queue['normdelg_calcs'] =[normdelg_calcs]
        mech_queue['rd_exps'] = [rd_exps]
        mech_queue['rd_calcs'] = [rd_calcs]

        # print(mech_queue['delf_calcs']) #testprint
        # print(mech_queue['delg_calcs']) #testprint
        # TODO save delfstar, deriv {n1:, n2:, n3:}
        # print(mech_queue)    #testprint

        return mech_queue
        ########## TODO 


    def solve_general(self, nh, delfstar, calctype, film, prop_guess={}):
        '''
        solve the property of a single test.
        nh: list of int
        delfstar: dict {harm(int): complex, ...}
        overlayer: dict e.g.: {'drho': 0, 'grho_refh': 0, 'phi': 0}
        return drho, grho_refh, phi, dlam_refh, err
        '''
        # input variables - this is helpfulf for the error analysis
        # define sensibly names partial derivatives for further use
        deriv = {}
        err = {}
        err_names=['drho', 'grho_refh', 'phi']

        # initiate deriv and err
        for key in err_names:
            deriv[key] = np.nan
            err[key] = np.nan

        if not film: # film is not built
            film = self.build_single_layer_film()

        # first pass at solution comes from rh and rd
        rd_exp = self.rd_from_delfstar(nh[2], delfstar) # nh[2]
        rh_exp = self.rh_from_delfstar(nh, delfstar) # nh[0], nh[1]
        # print('rd_exp', rd_exp) #testprint
        # print('rh_exp', rh_exp) #testprint
        
        n1 = nh[0]
        n2 = nh[1]
        n3 = nh[2]
        # solve the problem
        if ~np.isnan(rd_exp) and ~np.isnan(rh_exp):
            # print('rd_exp, rh_exp is not nan') #testprint
            
            if rd_exp > 0.5: # bulk
                # print('use bulk guess') #testprint
                dlam_refh, phi = self.bulk_guess(delfstar)
            elif prop_guess: # value{'drho', 'grho_refh', 'phi'}
                # print('use prop guess') #testprint
                dlam_refh, phi = self.guess_from_props(**prop_guess)
            else:
                # print('use thin film guess') #testprint
                dlam_refh, phi = self.thinfilm_guess(delfstar)

            # print('dlam_refh', dlam_refh) #testprint
            # print('phi', phi) #testprint
            
            if fit_method == 'lmfit': # this part is the old protocal w/o jacobian
                pass
            else: # scipy
                lb = np.array([dlam_refh_range[0], phi_range[0]])  # lower bounds on dlam_refh and phi
                ub = np.array([dlam_refh_range[1], phi_range[1]])  # upper bonds on dlam_refh and phi

                # we solve the problem initially using the harmonic and dissipation
                # ratios, using the small load approximation
                def ftosolve(x):
                    return [self.rhcalc(nh, x[0], x[1])-rh_exp, self.rdcalc(nh, x[0], x[1])-rd_exp]

                x0 = np.array([dlam_refh, phi])
                # print(x0) #testprint
                soln1 = optimize.least_squares(ftosolve, x0, bounds=(lb, ub))
                # print(soln1['x']) #testprint
                dlam_refh = soln1['x'][0]
                phi =soln1['x'][1]
                drho = self.calc_drho(n1, delfstar, dlam_refh, phi)
                grho_refh = self.grho_from_dlam(self.refh, drho, dlam_refh, phi)

            # print('solution of 1st solving:') #testprint
            # print('dlam_refh', dlam_refh) #testprint
            # print('phi', phi) #testprint
            # print('drho', drho) #testprint
            # print('grho_refh', grho_refh) #testprint
            
            # we solve it again to get the Jacobian with respect to our actual
            # input variables - this is helpfulf for the error analysis
            if drho_range[0]<=drho<=drho_range[1] and grho_refh_range[0]<=grho_refh<=grho_refh_range[1] and phi_range[0]<=phi<=phi_range[1]:
                # print('1st solution in range') #testprint
                
                if fit_method == 'lmfit': # this part is the old protocal w/o jacobian 
                    pass
                else: # scipy
                    x0 = np.array([drho, grho_refh, phi])
                    
                    lb = np.array([drho_range[0], grho_refh_range[0], phi_range[0]])  # lower bounds drho, grho3, phi
                    ub = np.array([drho_range[1], grho_refh_range[1], phi_range[1]])  # upper bounds drho, grho3, phi

                    def ftosolve2(x):
                        layers = self.set_calc_layer_val(film, x[0], x[1], x[2]) # set drho, grho, phi to x[0], x[1], x[2], respectively
                        return ([
                            np.real(delfstar[n1]) - np.real(self.calc_delfstar(n1, layers, calctype)),
                            np.real(delfstar[n2]) - np.real(self.calc_delfstar(n2, layers, calctype)),
                            np.imag(delfstar[n3]) - np.imag(self.calc_delfstar(n3, layers, calctype))
                        ])
                    
                    # recalculate solution to give the uncertainty, if solution is viable
                    soln2 = optimize.least_squares(ftosolve2, x0, bounds=(lb, ub))
                    drho = soln2['x'][0]
                    grho_refh = soln2['x'][1]
                    phi = soln2['x'][2]
                    
                    # update calc layer prop
                    film = self.set_calc_layer_val(film, drho, grho_refh, phi)
                    # print('film after 2nd sol', film) #testprint
                    
                    # dlam_refh = self.calc_dlam(self.refh, film)
                    # comment above line to use the dlam_refh from soln1

                    # put the input uncertainties into a 3 element vector
                    delfstar_err = np.zeros(3)
                    delfstar_err[0] = np.real(self.fstar_err_calc(delfstar[n1]))
                    delfstar_err[1] = np.real(self.fstar_err_calc(delfstar[n2]))
                    delfstar_err[2] = np.imag(self.fstar_err_calc(delfstar[n3]))

                    jac = soln2['jac']
                    # print('jac', jac) #testprint
                    try:
                        jac_inv = np.linalg.inv(jac)
                        # print('jac_inv', jac_inv) #testprint

                        for i, k in enumerate(err_names):
                            deriv[k]={0:jac_inv[i, 0], 1:jac_inv[i, 1], 2:jac_inv[i, 2]}
                            err[k] = ((jac_inv[i, 0]*delfstar_err[0])**2 + 
                                    (jac_inv[i, 1]*delfstar_err[1])**2 +
                                    (jac_inv[i, 2]*delfstar_err[2])**2)**0.5
                    except:
                        print('error calculation failed!') 
                        pass
            else:
                print('1st solution out of range') 


        if np.isnan(rd_exp) or np.isnan(rh_exp) or not deriv or not err: # failed to solve the problem
            print('2nd solving failed') 
            # assign the default value first
            drho = np.nan
            grho_refh = np.nan
            phi = np.nan
            dlam_refh = np.nan
            for k in err_names:
                err[k] = np.nan

        # print('drho', drho) #testprint
        # print('grho_refh', grho_refh) #testprint
        # print('phi', phi) #testprint
        # print('dlam_refh', phi) #testprint
        # print('err', err) #testprint
        delrho = self.calc_delrho(self.refh, grho_refh, phi)
        # print('delrho', delrho) #testprint

        return drho, grho_refh, phi, dlam_refh, err


    def all_nhcaclc_harm_not_na(self, nh, qcm_queue):
        '''
        check if all harmonics in nhcalc are not na
        nh: list of strings
        qcm_queue: qcm data (df) of a single queue
        return: True/False
        '''
        # print(nh) #testprint
        # print(nh2i(nh[0])) #testprint
        # print(qcm_queue.delfstars.values) #testprint
        # print(qcm_queue.delfstars.iloc[0]) #testprint
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
            # print('queue_id', queue_id) #testprint
            # print('qcm_df', qcm_df) #testprint
            # print(type(qcm_df)) #testprint
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
                # print(mech_df.loc[[idx], :].to_dict()) #testprint
                # print(mech_queue.to_dict()) #testprint
                # set mech_queue index the same as where it is from for update
                # print(mech_df.delg_calcs) #testprint
                mech_queue.index = [idx]
                mech_df.update(mech_queue)
                # print(mech_df) #testprint
                # print(mech_df.delg_calcs) #testprint
            else:
                # since the df already initialized with nan values, nothing todo
                pass
        return mech_df


    def convert_mech_unit(self, mech_df):
        '''
        convert unit of drho, grho, phi from IS to those convient to use
        input: df or series 
        '''
        # print(type(mech_df)) #testprint
        # print(mech_df) #testprint
        df = mech_df.copy()
        cols = mech_df.columns
        for col in cols:
            # print(col) #testprint
            if any([st in col for st in ['drho', 'lamrho', 'delrho']]):
                # print('x1000') #testprint
                df[col] = df[col].apply(lambda x: list(np.array(x) * 1000) if isinstance(x, list) else x * 1000) # from m kg/m3 to um g/cm3
            elif any([st in col for st in ['grho', 'etarho']]):
                # print('x1/1000') #testprint
                df[col] = df[col].apply(lambda x: list(np.array(x) / 1000) if isinstance(x, list) else x / 1000) # from Pa kg/m3 to Pa g/cm3 (~ Pa)
            elif 'phi' in col:
                # print('rad2deg') #testprint
                df[col] = df[col].apply(lambda x: list(np.rad2deg(x)) if isinstance(x, list) else np.rad2deg(x)) # from rad to deg
            else:
                # print('NA') #testprint
                pass
        return df


    def single_harm_data(self, var, qcm_df):
        '''
        get variables calculate from single harmonic
        variables listed in DataSaver.mech_keys_multiple
        to keep QCM and DataSaver independently, we don't use import for each other
        ['delf_calcs', 'delg_calcs', 'normdelfs', 'rds']
        '''
        # print(var) #testprint
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
                'drho': float, # in um g/cm^3
                'grho': float, # in Pa g/cm^3
                'drho': float, # in deg
               },
            1:
            2:
            ...
        }
        This function return n (int) of the layer with 'calc' is True
        '''
        for n, layer in film.items():
            if layer['calc']:
                return n


    def get_calc_layer(self, film):
        ''' 
        film ={
            0:{'calc': True/False,
                'drho': float, # in um g/cm^3
                'grho': float, # in Pa g/cm^3
                'drho': float, # in deg
               },
            1:
            2:
            ...
        }
        This function return the film contain only the layer with 'calc' is True
        '''
        new_film = {}
        for n, layer in film.items():
            if layer['calc']:
                new_film[n] = layer
                break # only one layer with 'calc' True is allowed
        return new_film


    def set_calc_layer_val(self, film, drho, grho, phi):
        ''' 
        film ={
            0:{'calc': True/False,
                'drho': float, # in um g/cm^3
                'grho': float, # in Pa g/cm^3
                'drho': float, # in deg
               },
            1:
            2:
            ...
        }
        This function return n (int) of the layer with 'calc' is True
        '''
        if not film: # film has no layers
            # print('build a single layer film') #testprint
            film = self.build_single_layer_film() # make a single layer film

        calcnum = self.get_calc_layer_num(film)
        film[calcnum].update(drho=drho, grho=grho, phi=phi, n=self.refh)
        return film


    def build_single_layer_film(self):
        '''
        build a single layer film for calc
        '''
        return self.replace_layer_0_prop_with_known({0:{'calc': False}, 1:{'calc': True}})


    def get_ref_layers(self, film):
        ''' 
        This function return the film contain all the layers with 'calc' is False
        '''
        new_film = {}
        for n, layer in film.items():
            if not layer['calc']:
                new_film[n] = layer
        return new_film


    def separate_calc_ref_layers(self, film):
        return self.get_calc_layer(film), self.get_ref_layers(film)


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
            new_film[0]['calc'] = False # this layer will not be the unkown layer
        return new_film


if __name__ == '__main__':
    qcm = QCM()
    qcm.f1 = 5e6 # Hz
    qcm.refh = 3

    # delrho = qcm.calc_delrho(qcm.refh, 96841876.71961978, 1.5707963266378169)
    # print(delrho)
    # exit(0)

    nh = [3,5,3]

    samp = 'PS_water'

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


    drho, grho_refh, phi, dlam_refh, err = qcm.solve_general(nh, delfstar, 'LL', film)

    print('drho', drho)
    print('grho_refh', grho_refh)
    print('phi', phi)
    print('dlam_refh', phi)
    print('err', err)


