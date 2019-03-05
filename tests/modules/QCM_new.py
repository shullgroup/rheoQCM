'''
This module is a modified version of QCM_functions.
It is used for UI, but you can also use it for data analysis.
The input and return data are all in form of DataFrame.
This module doesn't have plotting functions.

NOTE: Differnt to other modules, the harmonics used in this module are all INT.
'''


import numpy as np
from scipy.optimize import least_squares
from lmfit import Minimizer, minimize, Parameters, fit_report, printfuncs

import pandas as pd


# variable limitions
dlam_rh_range = (0, 5)
drho_range = (0, 1e-2) # m kg/m^3 = 1000 um g/cm^3 = 1000 g/m^2
grho_rh_range = (1e4, 1e10) # Pa kg/m^3 = 1/1000 Pa g/cm^3
phi_range = (0, np.pi/2) # rad 

e26 = 9.65e-2 # piezoelectric stress coefficient (e = 9.65·10−2 C/m2 for AT-cut quartz)
electrode_default = {'drho':2.8e-3, 'grho3':3.0e14, 'phi':0}
air_default = {'drho':np.inf, 'grho3':0, 'phi':0} #???

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
    def __init__(self):
        '''
        phi in rad for calculation and exported as degree
        '''
        self.Zq = 8.84e6  # kg m−2 s−1. shear acoustic impedance of AT cut quartz
        #TODO add zq by cuts
        self.f1 = None # 5e6 Hz fundamental resonant frequency
        self.g_err_min = 10 # error floor for gamma
        self.f_err_min = 50 # error floor for f
        self.err_frac = 3e-2 # error in f or gamma as a fraction of gamma
        self.drho_q = self.Zq / (2 * self.f1)

        self.rh = None # reference harmonic for calculation
        # default values
        # self.nhcalc = '355' # harmonics used for calculating
        # self.nhplot = [1, 3, 5] # harmonics used for plotting (show calculated data)



    def fstar_err_calc(self, fstar):
        ''' 
        calculate the error in delfstar
        fstar: complex number 
        '''
        # start by specifying the error input parameters
        fstar_err = np. zeros(1, dtype=np.complex128)
        fstar_err = (self.f_err_min + self.err_frac * np.imag(fstar)) + 1j*(self.g_err_min + self.err_frac*np.imag(fstar))
        return fstar_err
        # ?? show above both imag?


    def sauerbreyf(self, n, drho):
        ''' delf_sn from Sauerbrey eq'''
        return 2 * n * self.f1**2 * drho / self.Zq


    def sauerbreym(self, n, delf):
        ''' mass from Sauerbrey eq'''
        return delf * self.Zq / (2 * n * self.f1**2)


    # def grho(self, n, grho_rh, phi): # old func
    #     ''' grho of n_th harmonic'''
    #     return grho_rh * (n/self.rh) ** (phi)


    def grho(self, n, material): # func new
        grho_rh = material['grho_rh']
        phi = material['phi']
        return grho_rh * (n/self.rh) ** (phi)


    def grho_from_dlam(self, n, drho, dlam, phi):
        return (drho * n * self.f1 * np.cos(phi / 2) / dlam)**2


    def grho_rh_bulk(self, delfstar):
        '''
        bulk model
        calculate grho reference to rh
        '''
        return (np.pi * self.Zq * abs(delfstar[self.rh]) / self.f1) ** 2


    def phi_bulk(self, n, delfstar):
        '''
        bulk model
        calculate phi
        '''
        return -2 * np.arctan(np.real(delfstar[n]) / np.imag(delfstar[n]))


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

    ###### new funcs #########
    def calc_D(n, material, delfstar):
        drho = material['drho']
        # set switch to handle ase where drho = 0
        if drho == 0:
            return 0
        else:
            return 2*np.pi*(n*f1+delfstar)*drho/zstar_bulk(n, material)

    def zstar_bulk(n, material):
        grho3 = material['grho3']
        grho = grho3*(n/3)**(material['phi']/90)  #check for error here
        grhostar = grho*np.exp(1j*np.pi*material['phi']/180)
        return grhostar ** 0.5

    def calc_delfstar_sla(ZL):
        return f1*1j/(np.pi*Zq)*ZL

    def calc_ZL(n, layers, delfstar):
        # layers is a dictionary of dictionaries
        # each dictionary is named according to the layer number
        # layer 1 is closest to the quartz

        N = len(layers)
        Z = {}; D = {}; L = {}; S = {}

        # we use the matrix formalism to avoid typos.
        for i in np.arange(1, N):
            Z[i] = zstar_bulk(n, layers[i])
            D[i] = calc_D(n, layers[i], delfstar)
            L[i] = np.array([[np.cos(D[i])+1j*np.sin(D[i]), 0],
                    [0, np.cos(D[i])-1j*np.sin(D[i])]])

        # get the terminal matrix from the properties of the last layer
        D[N] = calc_D(n, layers[N], delfstar)
        Zf_N = 1j*zstar_bulk(n, layers[N])*np.tan(D[N])

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


    def calc_delfstar(n, layers, calctype):
        if not layers: # layers is empty {}
            return np.nan

        # there is data
        if 'overlayer' in layers:
            ZL = calc_ZL(n, {1:layers['film'], 2:layers['overlayer']}, 0)
            ZL_ref = calc_ZL(n, {1:layers['overlayer']}, 0)
            del_ZL = ZL-ZL_ref
        else:
            del_ZL = calc_ZL(n, {1:layers['film']}, 0)

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

            ZL_all = calc_ZL(n, layers_all, 0)
            delfstar_sla_all = calc_delfstar_sla(ZL_all)
            ZL_ref = calc_ZL(n, layers_ref, 0)
            delfstar_sla_ref = calc_delfstar_sla(ZL_ref)


            def solve_Zmot(x):
                delfstar = x[0] + 1j*x[1]
                Zmot = calc_Zmot(n,  layers_all, delfstar)
                return [Zmot.real, Zmot.imag]

            sol = optimize.root(solve_Zmot, [delfstar_sla_all.real,
                                            delfstar_sla_all.imag])
            dfc = sol.x[0] + 1j* sol.x[1]

            def solve_Zmot_ref(x):
                delfstar = x[0] + 1j*x[1]
                Zmot = calc_Zmot(n,  layers_ref, delfstar)
                return [Zmot.real, Zmot.imag]

            sol = optimize.root(solve_Zmot_ref, [delfstar_sla_ref.real,
                                                delfstar_sla_ref.imag])
            dfc_ref = sol.x[0] + 1j* sol.x[1]

            return dfc-dfc_ref


    def calc_Zmot(n, layers, delfstar):
        om = 2 * np.pi *(n*f1 + delfstar)
        g0 = 10 # Half bandwidth of unloaed resonator (intrinsic dissipation on crystalline quartz)
        Zqc = Zq * (1 + 1j*2*g0/(n*f1))
        dq = 330e-6  # only needed for piezoelectric stiffening calc.
        epsq = 4.54; eps0 = 8.8e-12; C0byA = epsq * eps0 / dq; ZC0byA = C0byA / (1j*om)
        ZPE = -(e26/dq)**2*ZC0byA  # ZPE accounts for oiezoelectric stiffening anc
        # can always be neglected as far as I can tell

        Dq = om*drho_q/Zq
        secterm = -1j*Zqc/np.sin(Dq)
        ZL = calc_ZL(n, layers, delfstar)
        # eq. 4.5.9 in book
        thirdterm = ((1j*Zqc*np.tan(Dq/2))**-1 + (1j*Zqc*np.tan(Dq/2) + ZL)**-1)**-1
        Zmot = secterm + thirdterm  +ZPE

        return Zmot



    def calc_dlam(n, film):
        return calc_D(n, film, 0).real/(2*np.pi)

    ##### end new funcs ######

    def dlam(self, n, dlam_rh, phi):
        return dlam_rh*(n/self.rh) ** (1-phi/np.pi)


    def normdelfstar(self, n, dlam_rh, phi):
        return -np.tan(2*np.pi*self.dlam(n, dlam_rh, phi)*(1-1j*np.tan(phi/2))) / (2*np.pi*self.dlam(n, dlam_rh, phi)*(1-1j*np.tan(phi/2)))


    def drho(self, n1, delfstar, dlam_rh, phi):
        return self.sauerbreym(n1, np.real(delfstar[n1])) / np.real(self.normdelfstar(n1, dlam_rh, phi))


    def rhcalc(self, nh, dlam_rh, phi):
        ''' nh list '''
        return np.real(self.normdelfstar(nh[0], dlam_rh, phi)) /  np.real(self.normdelfstar(nh[1], dlam_rh, phi))


    def rhexp(self, nh, delfstar):
        return (nh[1]/nh[0])*np.real(delfstar[nh[0]]) / np.real(delfstar[nh[1]])

    def rh_from_delfstar(self, nh, delfstar):
        ''' this func is the same as rhexp!!! '''
        n1 = int(nh[0])
        n2 = int(nh[1])
        return (n2/n1)*np.real(delfstar[n1])/np.real(delfstar[n2])


    def rdcalc(self, nh, dlam_rh, phi):
        return -np.imag(self.normdelfstar(nh[2], dlam_rh, phi)) / np.real(self.normdelfstar(nh[2], dlam_rh, phi))


    def rdexp(self, nh, delfstar):
        return -np.imag(delfstar[nh[2]]) / np.real(delfstar[nh[2]])


    def rd_from_delfstar(self, n, delfstar):
        ''' dissipation ratio calculated for the relevant harmonic '''
        return -np.imag(delfstar[n])/np.real(delfstar[n])


    def bulk_guess(self, delfstar):
        ''' get the bulk solution for grho and phi '''
        grho_rh = (np.pi*self.Zq*abs(delfstar[self.rh])/self.f1) ** 2
        phi = -2*np.arctan(np.real(delfstar[self.rh]) / np.imag(delfstar[self.rh]))

        # calculate rho*lambda
        lamrho_rh = np.sqrt(grho_rh)/(self.rh*self.f1*np.cos(phi/2))

        # we need an estimate for drho.  We only use this approach if it is
        # reasonably large.  We'll put it at the quarter wavelength condition
        # for now

        drho = lamrho_rh / 4
        dlam_rh = self.d_lamcalc(self.rh, drho, grho_rh, phi)

        return [dlam_rh, min(phi, np.pi/2)]


    def guess_from_props(self, drho, grho_rh, phi):
        dlam_rh = self.d_lamcalc(self.rh, drho, grho_rh, phi)
        return [dlam_rh, phi]


    def thinfilm_guess(self, delfstar):
        ''' 
        really a placeholder function until we develop a more creative strategy
        for estimating the starting point 
        '''
        return [0.05, np.pi/180*5]


    ########################################################



    ########################################################


    def solve_single_queue(self, nh, qcm_queue, mech_queue):
        '''
        solve the property of a single test.
        nh: list of int
        qcm_queue:  QCM data. df (shape[0]=1) 
        mech_queue: initialized property data. df (shape[0]=1)
        return mech_queue
        '''
        # get fstar
        fstars = qcm_queue.fstars.iloc[0] # list
        # get delfstar
        delfstars = qcm_queue.delfstars.iloc[0] # list
        print('fstars', fstars) #testprint
        print(delfstars) #testprint
        # convert list to dict to make it easier to do the calculation
        # fstar = {int(i*2+1): fstar[i] for i, fstar in enumerate(fstars)}
        delfstar = {int(i*2+1): dfstar for i, dfstar in enumerate(delfstars)}
        print(delfstar) #testprint
        # get the marks [1st, 3rd, 5th, ...]
        marks = qcm_queue.marks.iloc[0]
        # find where the mark is not nan or None
        nhplot = [i*2+1 for i, mark in enumerate(marks) if mark != np.nan and mark is not None ]

        # fstar_err ={}
        # for n in nhplot: 
        #     fstar_err[n] = self.fstar_err_calc(fstar[n])


        # set up to handle one or two layer cases
        # overlayer set to air if it doesn't exist in soln_input
        if 'overlayer' in qcm_queue.keys():
            overlayer = qcm_queue.overlayer.iat[0, 0]
        else:
            overlayer = {'drho':0, 'grho_rh':0, 'phi':0}


        drho, grho_rh, phi, dlam_rh, err = self.solve_general(nh, delfstar, overlayer)

        # now back calculate delfstar, rh and rd from the solution
        delfstar_calc = {}
        rh = {}
        rd_exp = {}
        rd_calc = {}
        delf_calcs = mech_queue.delf_calcs.iloc[0].copy()
        delg_calcs = mech_queue.delg_calcs.iloc[0].copy()
        rd_exps = mech_queue.rd_exps.iloc[0]
        rd_calcs = mech_queue.rd_calcs.iloc[0]
        print('delf_calcs', delf_calcs) #testprint
        print(type(delf_calcs)) #testprint
        for n in nhplot:
            delfstar_calc[n] = self.delfstarcalc(n, drho, grho_rh, phi, overlayer)
            delf_calcs[nh2i(n)] = np.real(delfstar_calc[n])
            delg_calcs[nh2i(n)] = np.imag(delfstar_calc[n])
            
            rd_calc[n] = self.rd_from_delfstar(n, delfstar_calc)
            rd_calcs[nh2i(n)] = rd_calc[n]
            rd_exp[n] = self.rd_from_delfstar(n, delfstar)
            rd_exps[nh2i(n)] = rd_exp[n]
        rh = self.rh_from_delfstar(nh, delfstar_calc)
        print('delf_calcs', delf_calcs) #testprint
        print('delg_calcs', delg_calcs) #testprint

        # drho = 1000*results[nh]['drho']
        # grho3 = results[nh]['grho3']/1000
        # phi = results[nh]['phi']
    
        # save back to mech_queue
        # single value
        # mech_queue['drho'] = [drho] # in kg/m2
        # mech_queue['drho_err'] = [err['drho']] # in kg/m2
        # mech_queue['grho_rh'] = [grho_rh] # in Pa kg/m3
        # mech_queue['grho_rh_err'] = [err['grho_rh']] # in Pa kg/m3
        # mech_queue['phi'] = [phi] # in rad
        # mech_queue['phi_err'] = [err['phi']] # in rad
        # mech_queue['dlam_rh'] = [dlam_rh] # in na
        # mech_queue['lamrho'] = [np.nan] # in kg/m2
        # mech_queue['delrho'] = [np.nan] # in kg/m2
        # mech_queue['delf_delfsn'] = [np.nan]
        # mech_queue['rh'] = [rh]

        # repeat values for single value
        tot_harms = len(delf_calcs)
        mech_queue['drho'] = [[drho] * tot_harms] # in kg/m2
        mech_queue['drho_err'] = [[err['drho']] * tot_harms] # in kg/m2
        mech_queue['grho_rh'] = [[grho_rh] * tot_harms] # in Pa kg/m3
        mech_queue['grho_rh_err'] = [[err['grho_rh']] * tot_harms] # in Pa kg/m3
        mech_queue['phi'] = [[phi] * tot_harms] # in rad
        mech_queue['phi_err'] = [[err['phi']] * tot_harms] # in rad
        mech_queue['dlam_rh'] = [[dlam_rh] * tot_harms] # in na
        mech_queue['lamrho'] = [[np.nan] * tot_harms] # in kg/m2
        mech_queue['delrho'] = [[np.nan] * tot_harms] # in kg/m2
        mech_queue['delf_delfsn'] = [[np.nan] * tot_harms]
        mech_queue['rh'] = [[rh] * tot_harms]

        # multiple values in list
        mech_queue['delf_exps'] = qcm_queue['delfs']
        mech_queue['delf_calcs'] = [delf_calcs]
        mech_queue['delg_exps'] = qcm_queue['delgs']
        mech_queue['delg_calcs'] = [delg_calcs]
        mech_queue['delg_delfsn_exps'] =[[np.nan] * tot_harms]
        mech_queue['delg_delfsn_clacs'] =[[np.nan] * tot_harms]
        mech_queue['rd_exps'] = [rd_exps]
        mech_queue['rd_calcs'] = [rd_calcs]

        print(mech_queue['delf_calcs']) #testprint
        print(mech_queue['delg_calcs']) #testprint
        # TODO save delfstar, deriv {n1:, n2:, n3:}
        print(mech_queue)    #testprint
        return mech_queue
        ########## TODO 


    def solve_general(self, nh, delfstar, overlayer, prop_guess={}):
        '''
        solve the property of a single test.
        nh: list of int
        delfstar: dict {harm(int): complex, ...}
        overlayer: dicr e.g.: {'drho': 0, 'grho_rh': 0, 'phi': 0}
        return drho, grho_rh, phi, dlam_rh, err
        '''
        # input variables - this is helpfulf for the error analysis
        # define sensibly names partial derivatives for further use
        deriv = {}
        err = {}
        err_names=['drho', 'grho_rh', 'phi']

        # first pass at solution comes from rh and rd
        rd_exp = self.rdexp(nh, delfstar) # nh[2]
        rh_exp = self.rhexp(nh, delfstar) # nh[0], nh[1]
        print('rd_exp', rd_exp) #testprint
        print('rh_exp', rh_exp) #testprint

        n1 = nh[0]
        n2 = nh[1]
        n3 = nh[2]

        # solve the problem
        if ~np.isnan(rd_exp) or ~np.isnan(rh_exp):
            print('rd_exp, rh_exp is not nan') #testprint
            # TODO change here for the model selection
            if prop_guess: # value{'drho', 'grho_rh', 'phi'}
                dlam_rh, phi = self.guess_from_props(**prop_guess)
            elif rd_exp > 0.5:
                dlam_rh, phi = self.bulk_guess(delfstar)
            else:
                dlam_rh, phi = self.thinfilm_guess(delfstar)

            print('dlam_rh', dlam_rh) #testprint
            print('phi', phi) #testprint
            
            if fit_method == 'lmfit':
                params1 = Parameters()
                params1.add('dlam_rh', value=dlam_rh, min=dlam_rh_range[0], max=dlam_rh_range[1])
                params1.add('phi', value=phi, min=phi_range[0], max=phi_range[1])

                def residual1(params, rh_exp, rd_exp):
                    # dlam_rh = params['dlam_rh'].value
                    # phi = params['phi'].value
                    return [self.rhcalc(nh, dlam_rh, phi)-rh_exp, self.rdcalc(nh, dlam_rh, phi)-rd_exp]

                mini = Minimizer(
                    residual1,
                    params1,
                    fcn_args=(rh_exp, rd_exp),
                    # nan_policy='omit',
                )
                soln1 = mini.leastsq(
                    # xtol=1e-7,
                    # ftol=1e-7,
                )

                print(fit_report(soln1))  #testprint
                print('success', soln1.success) #testprint
                print('message', soln1.message) #testprint
                print('lmdif_message', soln1.lmdif_message) #testprint

                dlam_rh = soln1.params.get('dlam_rh').value
                phi =soln1.params.get('phi').value
                drho = self.drho(n1, delfstar, dlam_rh, phi)
                grho_rh = self.grho_from_dlam(self.rh, drho, dlam_rh, phi)
            else: # scipy
                lb = np.array([dlam_rh_range[0], phi_range[0]])  # lower bounds on dlam3 and phi
                ub = np.array([dlam_rh_range[1], phi_range[1]])  # upper bonds on dlam3 and phi

                def ftosolve(x):
                    return [self.rhcalc(nh, x[0], x[1])-rh_exp, self.rdcalc(nh, x[0], x[1])-rd_exp]

                x0 = np.array([dlam_rh, phi])
                print(x0) #testprint
                soln1 = least_squares(ftosolve, x0, bounds=(lb, ub))
                print(soln1['x']) #testprint
                dlam_rh = soln1['x'][0]
                phi =soln1['x'][1]
                drho = self.drho(n1, delfstar, dlam_rh, phi)
                grho_rh = self.grho_from_dlam(self.rh, drho, dlam_rh, phi)

            print('solution of 1st solving:') #testprint
            print('dlam_rh', dlam_rh) #testprint
            print('phi', phi) #testprint
            print('drho', phi) #testprint
            print('grho_rh', grho_rh) #testprint

            # we solve it again to get the Jacobian with respect to our actual
            # input variables - this is helpfulf for the error analysis
            if drho_range[0]<=drho<=drho_range[1] and grho_rh_range[0]<=grho_rh<=grho_rh_range[1] and phi_range[0]<=phi<=phi_range[1]:
                
                print('1st solution in range') #testprint
                if fit_method == 'lmfit':
                    params2 = Parameters()

                    params2.add('drho', value=dlam_rh, min=drho_range[0], max=drho_range[1])
                    params2.add('grho_rh', value=grho_rh, min=grho_rh_range[0], max=grho_rh_range[1])
                    params2.add('phi', value=phi, min=phi_range[0], max=phi_range[1])

                    def residual2(params, delfstar, overlayer, n1, n2, n3):
                        drho = params['drho'].value
                        grho_rh = params['grho_rh'].value
                        phi = params['phi'].value
                        return ([np.real(delfstar[n1]) -
                                np.real(self.delfstarcalc(n1, drho, grho_rh, phi, overlayer)),
                                np.real(delfstar[n2]) -
                                np.real(self.delfstarcalc(n2, drho, grho_rh, phi, overlayer)),
                                np.imag(delfstar[n3]) -
                                np.imag(self.delfstarcalc(n3, drho, grho_rh, phi, overlayer))])
                    
                    mini = Minimizer(
                        residual2,
                        params2,
                        fcn_args=(delfstar, overlayer, n1, n2, n3),
                        # nan_policy='omit',
                    )
                    soln2 = mini.least_squares(
                        # xtol=1e-7,
                        # ftol=1e-7,
                    )
                    print(soln2.params.keys()) #testprint
                    print(soln2.params['drho']) #testprint
                    print(fit_report(soln2)) 
                    print('success', soln2.success)
                    print('message', soln2.message)
                    print('lmdif_message', soln1.lmdif_message)

                    # put the input uncertainties into a 3 element vector
                    delfstar_err = np.zeros(3)
                    delfstar_err[0] = np.real(self.fstar_err_calc(delfstar[n1]))
                    delfstar_err[1] = np.real(self.fstar_err_calc(delfstar[n2]))
                    delfstar_err[2] = np.imag(self.fstar_err_calc(delfstar[n3]))
                    
                    # initialize the uncertainties

                    # recalculate solution to give the uncertainty, if solution is viable
                    drho = soln2.params.get('drho').value
                    grho_rh = soln2.params.get('grho_rh').value
                    phi = soln2.params.get('phi').value
                    dlam_rh = self.d_lamcalc(self.rh, drho, grho_rh, phi)
                    jac = soln2.params.get('jac') #TODO ???
                    print('jac', jac) #testprint
                    jac_inv = np.linalg.inv(jac)

                    for i, k in enumerate(err_names):
                        deriv[k]={0:jac_inv[i, 0], 1:jac_inv[i, 1], 2:jac_inv[i, 2]}
                        err[k] = ((jac_inv[i, 0]*delfstar_err[0])**2 + 
                                (jac_inv[i, 1]*delfstar_err[1])**2 +
                                (jac_inv[i, 2]*delfstar_err[2])**2)**0.5
                else: # scipy
                    x0 = np.array([drho, grho_rh, phi])
                    
                    lb = np.array([drho_range[0], grho_rh_range[0], phi_range[0]])  # lower bounds drho, grho3, phi
                    ub = np.array([drho_range[1], grho_rh_range[1], phi_range[1]])  # upper bounds drho, grho3, phi

                    def ftosolve2(x):
                        return ([
                            np.real(delfstar[n1]) - np.real(self.delfstarcalc(n1, x[0], x[1], x[2], overlayer)),
                            np.real(delfstar[n2]) - np.real(self.delfstarcalc(n2, x[0], x[1], x[2], overlayer)),
                            np.imag(delfstar[n3]) - np.imag(self.delfstarcalc(n3, x[0], x[1], x[2], overlayer))
                            ]
                        )
                    
                    # put the input uncertainties into a 3 element vector
                    delfstar_err = np.zeros(3)
                    delfstar_err[0] = np.real(self.fstar_err_calc(delfstar[n1]))
                    delfstar_err[1] = np.real(self.fstar_err_calc(delfstar[n2]))
                    delfstar_err[2] = np.imag(self.fstar_err_calc(delfstar[n3]))

                    # recalculate solution to give the uncertainty, if solution is viable
                    soln2 = least_squares(ftosolve2, x0, bounds=(lb, ub))
                    drho = soln2['x'][0]
                    grho_rh = soln2['x'][1]
                    phi = soln2['x'][2]
                    dlam_rh = self.d_lamcalc(self.rh, drho, grho_rh, phi)
                    jac = soln2['jac']
                    print('jac', jac) #testprint
                    jac_inv = np.linalg.inv(jac)
                    print('jac_inv', jac_inv) #testprint

                    for i, k in enumerate(err_names):
                        deriv[k]={0:jac_inv[i, 0], 1:jac_inv[i, 1], 2:jac_inv[i, 2]}
                        err[k] = ((jac_inv[i, 0]*delfstar_err[0])**2 + 
                                (jac_inv[i, 1]*delfstar_err[1])**2 +
                                (jac_inv[i, 2]*delfstar_err[2])**2)**0.5

        if np.isnan(rd_exp) or np.isnan(rh_exp) or not deriv or not err: # failed to solve the problem
            print('2nd solving failed') 
            # assign the default value first
            drho = np.nan
            grho_rh = np.nan
            phi = np.nan
            dlam_rh = np.nan
            for k in err_names:
                err[k] = np.nan

        print('drho', drho) #testprint
        print('grho_rh', grho_rh) #testprint
        print('phi', phi) #testprint
        print('dlam_rh', phi) #testprint
        print('err', err) #testprint

        return drho, grho_rh, phi, dlam_rh, err


    def all_nhcaclc_harm_not_na(self, nh, qcm_queue):
        '''
        check if all harmonics in nhcalc are not na
        nh: list of strings
        qcm_queue: qcm data (df) of a single queue
        return: True/False
        '''
        print(nh) #testprint
        print(nh2i(nh[0])) #testprint
        print(qcm_queue.delfstars.values) #testprint
        print(qcm_queue.delfstars.iloc[0]) #testprint
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
            print('queue_id', queue_id) #testprint
            # print('qcm_df', qcm_df) #testprint
            print(type(qcm_df)) #testprint
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
                print(mech_df.loc[[idx], :].to_dict()) #testprint
                print(mech_queue.to_dict()) #testprint
                # set mech_queue index the same as where it is from for update
                print(mech_df.delg_calcs) #testprint
                mech_queue.index = [idx]
                mech_df.update(mech_queue)
                # print(mech_df) #testprint
                print(mech_df.delg_calcs) #testprint
            else:
                # since the df already initialized with nan values, nothing todo
                pass
        return mech_df


    def convert_mech_unit(self, mech_df):
        '''
        convert unit of drho, grho, phi from IS to those convient to use
        input: df or series 
        '''
        print(type(mech_df)) #testprint
        print(mech_df) #testprint
        df = mech_df.copy()
        cols = mech_df.columns
        for col in cols:
            if 'drho' in col:
                df[col] = df[col].apply(lambda x: list(np.array(x) * 1000) if isinstance(x, list) else x * 1000) # from m kg/m3 to um g/cm2
            elif 'grho' in col:
                df[col] = df[col].apply(lambda x: list(np.array(x) / 1000) if isinstance(x, list) else x / 1000) # from Pa kg/m3 to Pa g/cm3 (~ Pa)
            elif 'phi' in col:
                df[col] = df[col].apply(lambda x: list(np.rad2deg(x)) if isinstance(x, list) else np.rad2deg(x)) # from rad to deg
        return df


    def single_harm_data(self, var, qcm_df):
        '''
        get variables calculate from single harmonic
        variables listed in DataSaver.mech_keys_multiple
        to keep QCM and DataSaver independently, we don't use import for each other
        ['delf_calcs', 'delg_calcs', 'delg_delfsns', 'rds']
        '''
        print(var) #testprint
        if var == 'delf_calcs':
            return qcm_df.delfs
        if var == 'delg_calcs':
            return qcm_df.delgs
        if var == 'delg_delfsns':
            return None # TODO
        if var == 'rds':
            # rd_from_delfstar(self, n, delfstar)
            s = qcm_df.delfstars.copy()
            s.apply(lambda delfstars: [self.rd_from_delfstar(n, delfstars) for n in range(len(delfstars))])
            return s


if __name__ == '__main__':
    qcm = QCM()
    qcm.f1 = 5e6 # Hz
    qcm.rh = 3

    nh = [3,5,3]

    delfstar = {
        1: -11374.5837019132 + 1j*2.75600512576013,
        3: -34446.6126772240 + 1j*41.5415621054838,
        5: -58249.9656346552 + 1j*79.9003634583878,
    }

    overlayer = {'drho':0, 'grho_rh':0, 'phi':0}

    drho, grho_rh, phi, dlam_rh, err = qcm.solve_general(nh, delfstar, overlayer)

    print('drho', drho)
    print('grho_rh', grho_rh)
    print('phi', phi)
    print('dlam_rh', phi)
    print('err', err)


