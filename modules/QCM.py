'''
This module is a modified version of QCM_functions.
It is used for UI, but you can also use it for data analysis.
The input and return data are all in form of DataFrame.
This module doesn't have plotting functions.
'''


import numpy as np
from scipy.optimize import least_squares
from lmfit import Model, minimize, Parameters, fit_report, printfuncs

import pandas as pd


class QCM:
    def __init__(self):
        '''
        phi in rad for calculation and exported as degree
        '''
        self.zq = 8.84e6  # shear acoustic impedance of quartz
        self.f1 = None # 5e6 Hz fundamental resonant frequency
        self.g_err_min = 10 # error floor for gamma
        self.f_err_min = 50 # error floor for f
        self.err_frac = 3e-2 # error in f or gamma as a fraction of gamma

        self.rh = None # reference harmonic for calculation
        # default values
        self.nhcalc = [3, 5, 5] # harmonics used for calculating
        self.nhplot = [1, 3, 5] # harmonics used for plotting (show calculated data)



    def fstar_err_calc(self, fstar):
        ''' calculate the error in delfstar '''
        # start by specifying the error input parameters
        fstar_err = np. zeros(1, dtype=np.complex128)
        fstar_err = (max(self.f_err_min, self.err_frac* np.real(fstar)) + 1j*max(self.g_err_min, self.err_frac*np.imag(fstar)))
        return fstar_err


    def sauerbreyf(self, n, drho):
        ''' delf_sn from Sauerbrey eq'''
        return 2 * n * self.f1**2 * drho / self.zq


    def sauerbreym(self, n, delf):
        ''' mass from Sauerbrey eq'''
        return delf * self.zq / (2 * n * self.f1**2)


    def grho(self, n, grho_rh, phi):
        ''' grho of n_th harmonic'''
        return grho_rh * (n/self.rh) ** (phi)


    def grhostar(self, n, grho_rh, phi):
        return self.grho(n, grho_rh, phi) * np.exp(1j*phi)


    def grho_from_dlam(self, n, drho, dlam, phi):
        return (drho * n * self.f1 * np.cos(phi / 2) / dlam)**2


    def grho_rh_bulk(self, delfstar):
        return (np.pi * self.zq * abs(delfstar[self.rh]) / self.f1) ** 2


    def phi_bulk(self, n, delfstar):
        return -2 * np.arctan(np.real(delfstar[n]) / np.imag(delfstar[n]))


    def lamrho_rh_calc(self, grho_rh, phi):
        return np.sqrt(grho_rh) / (self.f1 * self.f1 * np.cos(phi / 2))


    def D(self, n, drho, grho_rh, phi):
        return 2*np.pi*drho*n*self.f1*(np.cos(phi/2) - 1j * np.sin(phi/2)) / (self.grho(n, grho_rh, phi)) ** 0.5


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


    def rstar(self, n, drho, grho_rh, phi, overlayer):
        # overlayer is dictionary with drho, grho_rh and phi
        grhostar_1 = self.grhostar(n, grho_rh, phi)
        grhostar_2 = self.grhostar(n, overlayer['grho_rh'], overlayer['phi'])
        zstar_1 = self.zstarbulk(grhostar_1)
        zstar_2 = self.zstarfilm(n, overlayer['drho'], grhostar_2)   
        return zstar_2 / zstar_1
    
    
    # calcuated complex frequency shift for single layer
    def delfstarcalc(self, n, drho, grho_rh, phi, overlayer):
        rstar = self.rstar(n, drho, grho_rh, phi, overlayer)
        # overlayer is dictionary with drho, grho_rh and phi
        calc = -(self.sauerbreyf(n, drho)*np.tan(self.D(n, drho, grho_rh, phi)) / self.D(n, drho, grho_rh, phi))*(1-rstar**2) / (1+1j*rstar*np.tan(self.D(n, drho, grho_rh, phi)))
        
        # handle case where drho = 0, if it exists
        # calc[np.where(drho==0)]=0
        return calc


    # calculated complex frequency shift for bulk layer
    def delfstarcalc_bulk(self, n, grho_rh, phi):
        return ((self.f1*np.sqrt(self.grho(n, grho_rh, phi)) / (np.pi*self.zq)) * (-np.sin(phi/2)+ 1j * np.cos(phi/2)))


    def d_lamcalc(self, n, drho, grho_rh, phi):
        return drho*n*self.f1*np.cos(phi/2) / np.sqrt(self.grho(n, grho_rh, phi))


    def thin_film_gamma(self, n, drho, jdprime_rho):
        return 8*np.pi ** 2*n ** 3*self.f1 ** 4*drho ** 3*jdprime_rho / (3*self.zq)
        # same master expression, replacing grho3 with jdprime_rho3


    def grho_rh(self, jdprime_rho_rh, phi):
        return np.sin(phi)/jdprime_rho_rh


    def dlam(self, n, dlam_rh, phi):
        return dlam_rh*(int(n)/self.rh) ** (1-phi/2)


    def normdelfstar(self, n, dlam_rh, phi):
        return -np.tan(2*np.pi*self.dlam(n, dlam_rh, phi)*(1-1j*np.tan(phi/2))) / (2*np.pi*self.dlam(n, dlam_rh, phi)*(1-1j*np.tan(phi/2)))


    def drho(self, n1, delfstar, dlam_rh, phi):
        return self.sauerbreym(n1, np.real(delfstar[n1])) / np.real(self.normdelfstar(n1, dlam_rh, phi))


    def rhcalc(self, nh, dlam_rh, phi):
        ''' nh string e.g. '353' ??? '''
        return self.normdelfstar(nh[0], dlam_rh, phi).np.real /  self.normdelfstar(nh[1], dlam_rh, phi).real


    def rhexp(self, nh, delfstar):
        return (nh[2]/nh[1])*np.real(delfstar[nh[1]]) / np.real(delfstar[nh[2]])

    def rh_from_delfstar(self, nh, delfstar):
        ''' nh here is the calc string (i.e., '353') '''
        n1 = int(nh[0])
        n2 = int(nh[1])
        return (n2/n1)*np.real(delfstar[n1])/np.real(delfstar[n2])


    def rdcalc(self, nh, dlam_rh, phi):
        return -np.imag(self.normdelfstar(nh[2], dlam_rh, phi)) / np.real(self.normdelfstar(nh[2], dlam_rh, phi))


    def rdexp(self, nh, delfstar):
        return -np.imag(delfstar[nh[3]]) / np.real(delfstar[nh[3]])


    def rd_from_delfstar(self, n, delfstar):
        ''' dissipation ratio calculated for the relevant harmonic '''
        return -np.imag(delfstar[n])/np.real(delfstar[n])


    def bulk_guess(self, delfstar):
        ''' get the bulk solution for grho and phi '''
        grho_rh = (np.pi*self.zq*abs(delfstar[self.rh])/self.f1) ** 2
        phi = -2*np.arctan(np.real(delfstar[self.rh]) / np.imag(delfstar[self.rh]))

        # calculate rho*lambda
        lamrho_rh = np.sqrt(grho_rh)/(self.rh*self.f1*np.cos(phi/2))

        # we need an estimate for drho.  We only use this approach if it is
        # reasonably large.  We'll put it at the quarter wavelength condition
        # for now

        drho = lamrho_rh / 4
        dlam_rh = self.d_lamcalc(self.rh, drho, grho_rh, phi)

        return [dlam_rh, min(phi, 90)]


    def guess_from_props(self, drho, grho_rh, phi):
        dlam_rh = self.d_lamcalc(self.rh, drho, grho_rh, phi)
        return [dlam_rh, phi]


    def thinfilm_guess(self, delfstar):
        ''' 
        really a placeholder function until we develop a more creative strategy
        for estimating the starting point 
        '''
        return [0.05, 5]


########################################################



########################################################


    def solve_general(self, soln_input):
        # set up to handle one or two layer cases
        # overlayer set to air if it doesn't exist in soln_input
        overlayer = soln_input.get('overlayer', {'drho':0, 'grho_rh':0, 'phi':0})
        nhplot = soln_input['nhplot']
        nh = soln_input['nh']
        n1 = int(nh[0])
        n2 = int(nh[1])
        n3 = int(nh[2])
        delfstar = soln_input['delfstar']

        # first pass at solution comes from rh and rd
        rd_exp = self.rdexp(nh, delfstar) # nh[3]
        rh_exp = self.rhexp(nh, delfstar) # nh[1], nh[2]

        if 'prop_guess' in soln_input:
            drho = soln_input['propguess']['drho']
            grho_rh = soln_input['propguess']['grho_rh']
            phi = soln_input['propguess']['phi']
            soln1_guess = self.guess_from_props(drho, grho_rh, phi)
        elif rd_exp > 0.5:
            soln1_guess = self.bulk_guess(delfstar)
        else:
            soln1_guess = self.thinfilm_guess(delfstar)

        lb = np.array([0, 0      ])  # lower bounds on dlam_rh and phi
        ub = np.array([5, np.pi/2])  # upper bonds on dlam_rh and phi
        ## WHY ub is

        def ftosolve(x):
            return [self.rhcalc(nh, x[0], x[1])-rh_exp, self.rdcalc(nh, x[0], x[1])-rd_exp]

        soln1 = least_squares(ftosolve, soln1_guess, bounds=(lb, ub))

        dlam_rh = soln1['x'][0]
        phi = soln1['x'][1]
        drho = self.drho(n1, delfstar, dlam_rh, phi)
        grho_rh = self.grho_from_dlam(self.rh, drho, dlam_rh, phi)

        # we solve it again to get the Jacobian with respect to our actual
        # input variables - this is helpfulf for the error analysis
        x0 = np.array([drho, grho_rh, phi])
        
        lb = np.array([0,    1e7,  0        ])  # lower bounds drho, grho_rh, phi
        ub = np.array([1e-2, 1e13, np.pi / 2])  # upper bounds drho, grho_rh, phi

        def ftosolve2(x):
            return ([np.real(delfstar[n1]) -
                    np.real(self.delfstarcalc(n1, x[0], x[1], x[2], overlayer)),
                    np.real(delfstar[n2]) -
                    np.real(self.delfstarcalc(n2, x[0], x[1], x[2], overlayer)),
                    np.imag(delfstar[n3]) -
                    np.imag(self.delfstarcalc(n3, x[0], x[1], x[2], overlayer))])
        
        # put the input uncertainties into a 3 element vector
        delfstar_err = np.zeros(3)
        delfstar_err[0] = np.real(soln_input['delfstar_err'][n1])
        delfstar_err[1] = np.real(soln_input['delfstar_err'][n2])
        delfstar_err[2] = np.imag(soln_input['delfstar_err'][n3])
        
        # initialize the uncertainties
        err = {}
        err_names=['drho', 'grho_rh', 'phi']

        # recalculate solution to give the uncertainty, if solution is viable
        if np.all(lb<x0) and np.all(x0<ub):
            soln2 = least_squares(ftosolve2, x0, bounds=(lb, ub))
            drho = soln2['x'][0]
            grho_rh = soln2['x'][1]
            phi = soln2['x'][2]
            dlam_rh = self.d_lamcalc(self.rh, drho, grho_rh, phi)
            jac = soln2['jac']
            jac_inv = np.linalg.inv(jac)
            for k in range(3):
                err[err_names[k]] = ((jac_inv[k, 0]*delfstar_err[0])**2 + 
                                    (jac_inv[k, 1]*delfstar_err[1])**2 +
                                    (jac_inv[k, 2]*delfstar_err[2])**2)**0.5
        else:
            drho = np.nan
            grho_rh = np.nan
            phi = np.nan
            dlam_rh = np.nan
            for k in range(3):
                err[err_names[k]] = np.nan
            
        # now back calculate delfstar, rh and rd from the solution
        delfstar_calc = {}
        rh = {}
        rd = {}
        for n in nhplot:
            delfstar_calc[n] = self.delfstarcalc(n, drho, grho_rh, phi, overlayer)
            rd[n] = self.rd_from_delfstar(n, delfstar_calc)
        rh = self.rh_from_delfstar(nh, delfstar_calc)

        soln_output = {
            'drho': drho, 
            'grho_rh': grho_rh, 
            'phi': phi, 
            'dlam_rh': dlam_rh, 
            'delfstar_calc': delfstar_calc, 
            'rh': rh, 
            'rd': rd,
        }
        
        soln_output['err'] = err
        return soln_output


    def null_solution(self, nhplot):
        soln_output = {'drho':np.nan, 'grho_rh':np.nan, 'phi':np.nan, 'dlam_rh':np.nan,
                'err':{'drho':np.nan, 'grho_rh':np.nan, 'phi': np.nan}}
        delfstar_calc = {}
        rh = {}
        rd = {}
        for n in nhplot:
            delfstar_calc[n] = np.nan
            rd[n] = np.nan
        rh = np.nan
        soln_output['rd'] = rd
        soln_output['rh'] = rh
        soln_output['delfstar_calc'] = delfstar_calc
        
        return soln_output


    def add_delfstar(self, df):
        '''
        convert delfs and delgs in df to delfstar (pd.series) for calculation
        '''
        # get delf and delg in form of [n1, n3, n5, ...]
        delfs = df.loc['delfs']
        delgs = df.loc['delgs']

        # convert to array
        delf_arr = np.array(delfs.values.tolist())
        delg_arr = np.array(delgs.values.tolist())

        # get delfstar as array
        delfstar_arr = delf_arr + 1j * delg_arr
        df['delfstars'] = list(delfstar_arr)

    def analyze_single(self):
        pass
    
    def analyze_series(self, sample, parms):
        # read in the optional inputs, assigning default values if not assigned
        nhplot = sample.get('nhplot', [1, 3, 5])

        sample['nhcalc'] = sample.get('nhcalc', '355')
        imagetype = parms.get('imagetype', 'svg')

        # initialize the dictionary we'll use to keep track of the points to plot
        idx = {}

        # plot and process the film data
        film, bare = self.process_raw(sample, 'film')

        # move getting index out of for loop for getting index from dict

        # pick the points that we want to analyze and add them to the plots
        for data_dict in [bare, film]:
            data_dict['fstar_err'] = {}
            idx = data_dict['idx']
            print(idx)
            for n in nhplot:
                data_dict['fstar_err'][n] = np.zeros(data_dict['n_all'], dtype=np.complex128)
                for i in range(len(queue_list)):
                    data_dict['fstar_err'][n][i] = self.fstar_err_calc(data_dict['fstar'][n][i])
                    f = np.real(data_dict['fstar'][n][i])/n
                    g = np.imag(data_dict['fstar'][n][i])
                    f_err = np.real(data_dict['fstar_err'][n][i])/n
                    g_err = np.imag(data_dict['fstar_err'][n][i])

        # adjust nhcalc to account to only include calculations for for which
        # the data exist
        # !!! this will be done in UI
        sample['nhcalc'] = self.nhcalc_in_nhplot(sample['nhcalc'], nhplot) 

        # now calculate the frequency and dissipation shifts
        delfstar = {}
        delfstar_err = {}
        film['fstar_ref']={}
        
        # if the number of temperatures is 1, we use the average of the 
        # bare temperature readings
        for n in nhplot:
            film['fstar_ref'][n] = np.zeros(film['n_all'], dtype=np.complex128)
            film['fstar_ref'][n][film['idx']] = bare['fstar'][n][bare['idx']]
        
        for i in range(len(queue_list)):
            idxf = film['idx'][i]
            delfstar[i] = {}
            delfstar_err[i] ={}
            for n in nhplot: 
                delfstar[i][n] = (film['fstar'][n][idxf] - film['fstar_ref'][n][idxf])
                delfstar_err[i][n] = self.fstar_err_calc(film['fstar'][n][idxf])

        # now do all of the calculations and plot the data
        soln_input = {'nhplot': nhplot}
        results = {}

        # now we set calculation of the desired solutions
        for nh in sample['nhcalc']:
            results[nh] = {'drho': np.zeros(nx), 'grho_rh': np.zeros(nx),
                        'phi': np.zeros(nx), 'dlam_rh': np.zeros(nx),
                        'lamrho_rh': np.zeros(nx), 'rd': {}, 'rh': {},
                        'delfstar_calc': {}, 'drho_err': np.zeros(nx),
                        'grho_rh_err': np.zeros(nx), 'phi_err': np.zeros(nx)}
            for n in nhplot:
                results[nh]['delfstar_calc'][n] = (np.zeros(nx,
                                                dtype=np.complex128))
                results[nh]['rd'][n] = np.zeros(nx)

            results[nh]['rh'] = np.zeros(nx)

            for i in np.arange(nx):
                # obtain the solution for the properties
                soln_input['nh'] = nh
                soln_input['delfstar'] = delfstar[i]
                soln_input['delfstar_err'] = delfstar_err[i]
                if (np.isnan(delfstar[i][int(nh[0])].real) or 
                    np.isnan(delfstar[i][int(nh[1])].real) or
                    np.isnan(delfstar[i][int(nh[2])].imag)):                
                    soln = null_solution(nhplot)
                else:
                    soln = solve_general(soln_input)

                results[nh]['drho'][i] = soln['drho']
                results[nh]['grho_rh'][i] = soln['grho_rh']
                results[nh]['phi'][i] = soln['phi']
                results[nh]['dlam_rh'][i] = soln['dlam_rh']
                results[nh]['drho_err'][i] = soln['err']['drho']
                results[nh]['grho_rh_err'][i] = soln['err']['grho_rh']
                results[nh]['phi_err'][i] = soln['err']['phi']

                for n in nhplot:
                    results[nh]['delfstar_calc'][n][i] = (
                    soln['delfstar_calc'][n])
                    results[nh]['rd'][n][i] = soln['rd'][n]
                results[nh]['rh'][i] = soln['rh']

            # add property data to the property figure
            drho = 1000*results[nh]['drho']
            grho3 = results[nh]['grho3']/1000
            phi = results[nh]['phi']
            drho_err = 1000*results[nh]['drho_err']
            grho3_err = results[nh]['grho3_err']/1000
            phi_err = results[nh]['phi_err']


def nhcalc_in_nhplot(nhcalc_in, nhplot):
    # there is probably a more elegant way to do this
    # only consider harmonics in nhcalc that exist in nhplot
    nhcalc_out = []
    nhplot = list(set(nhplot))
    for nh in nhcalc_in:
        nhlist = list(set(nh))
        nhlist = [int(i) for i in nhlist]
        if all(elem in nhplot for elem in nhlist):
            nhcalc_out.append(nh)
    return nhcalc_out

