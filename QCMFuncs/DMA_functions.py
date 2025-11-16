#Import any individual functions from outside packages 
#that are used in your functions.
#These are called dependencies.
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.special import gamma as gammaf
from scipy.special import digamma
from pymittagleffler import mittag_leffler

palette = ['#0093F5', '#F08E2C', '#000000', '#424EBD', '#B04D25', '#75CA85', '#C892D6']


# function used to figure out number of lines to ignore in DMA input file
def first_numbered_line(file_path):
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.lstrip()
            if stripped_line and stripped_line[0].isdigit():
                return line_number
    return None


#Function definitions with docstrings
def readDMA(path, **kwargs):
    '''
    Returns a DataFrame from DMA temp sweep experiment.

    Parameters
    ----------
    path : Path
        Path object to the temperature sweep experiment txt file.
    instrument : str, default 'g2'
        Instrument flag for different output formats.  Two options here are
        'g2' for the TA Instruments ARES G2 or 'rsa3' for the old TA RSAIII.
    skiprows : int, default 2
        Number of rows to skip at beginning of file to remove header.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing relevant experimental data.

    '''
    
    instrument = kwargs.get('instrument', 'g2');
    skiprows = kwargs.get('skiprows', first_numbered_line(path))


    if instrument=='rsa3':
        #if old DMA, read data with these labels
        # may need to update this to define other variables 
        #vif it is still being used
        with open(path, 'r') as f:
            df = pd.read_csv(f, delimiter="\t", skiprows=skiprows,
                             usecols=[0,1,2,3],
                             names=['temp','storage','loss','tand'])

    
    # if newer G2 DMA, use these labels
    else:
        with open(path, 'r') as f:
            df = pd.read_csv(f, sep='\t', skiprows=skiprows,
                             usecols=[0,1,2,3,4,5,6,7],
                             names=['w','t','temp','strain','stress',
                                    'tand','storage','loss'])
            df['freq'] = df['w']/(2*np.pi)
        
    
    f['phi'] = np.degrees(np.arctan(df['tand']))
    return df
	
    
def readStressRelax(path, **kwargs):
    '''Returns the time and modulus data for a stress relaxation test'''
    with open(path, 'r') as f:
        df = pd.read_csv(f, sep='\t', skiprows=[1])
        t = df['Step time']
        mod = df['Modulus']
    
    return t, mod

def plotStressRelax(*arg, **kwargs):
    '''Returns a general plot of relaxation modulus with time'''
    
    norm = kwargs.get('norm', True);
    yaxis = kwargs.get('yaxis','log');        
    
    palette = ['#0093F5', '#F08E2C', '#000000', '#424EBD', '#B04D25', '#75CA85', '#C892D6']
    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
    
    a = 0;
    for A in arg:
        time = A[0][1:]; mod = A[1][1:];
        norm_mod = [m/mod[1] for m in mod]
        #set modulus normalization
        if norm:
            modulus = norm_mod;
            ylabel = 'Normalized Relaxation Modulus';
        else:
            modulus = mod;
            ylabel = 'Relaxation Modulus, E(t) (Pa)'
        #set log or linear yaxis
        if yaxis == 'log':
            ax.loglog(time,modulus, '-', color=palette[a])    
        elif yaxis == 'linear':
            ax.semilogx(time,modulus, '-', color=palette[a])
        a = a + 1;
        
    ax.set_xlabel('Time (s)'); ax.set_ylabel(ylabel);
    return plt.show()
    
def plotDMA(df, **kwargs):
    '''Returns a general plot of storage & loss moduli and tand vs T'''
    
    tand_min = kwargs.get('tand_min', 0.003); tand_max = kwargs.get('tand_max', 2);
    mod_min = kwargs.get('mod_min', 1e5); mod_max = kwargs.get('mod_max', 1e10);
    temp_min = kwargs.get('temp_min', -120)
    temp_max = kwargs.get('temp_max', 225)
    
    legendloc = kwargs.get('legendloc','best')
    legend_adjust = kwargs.get('legend_adjust', None)
    legendsize = kwargs.get('legendsize',10)
    exclude_storage = kwargs.get('exclude_storage', False)
    exclude_loss = kwargs.get('exclude_loss', False)
    exclude_tand = kwargs.get('exclude_tand', False)
    
    title = kwargs.get('title', None)
    save = kwargs.get('save', False); savepath = kwargs.get('savepath', None)
    
    
    palette = ['#0093F5', '#F08E2C', '#000000']
    

    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
    ax.set_ylim(mod_min, mod_max)
    ax.set_xlim(temp_min, temp_max)
    ax.set_xlabel('Temperature ($^oC$)')
    ax.set_ylabel('Storage and Loss Moduli (Pa)')
    legend_elements = []
    
    if exclude_storage:
        ax.set_ylabel('Loss Modulus (Pa)')
    else:
        p1, = ax.semilogy(df['temp'], df['storage'], '.-', ms=8, lw=2, 
                          color=palette[0], label="E'")
        legend_elements.append(p1)
    
    if exclude_loss:
        ax.set_ylabel('Storage Modulus (Pa)')
    else:
        p2, = ax.semilogy(df['temp'], df['loss'], '.-', ms=8, lw=2, 
                    color=palette[1], label='E"')
        legend_elements.append(p2)
    
    if exclude_tand:
        pass
    else:
        twin1 = ax.twinx()
        p3, = twin1.semilogy(df['temp'], df['tand'], '.-', ms=8, lw=2, 
                             color=palette[2], label='tan$\\delta$')
        legend_elements.append(p3)
        twin1.set_ylim(tand_min, tand_max)
        twin1.set_ylabel('tan$\\delta$')

    
    
    ax.set_title(title)
    ax.legend(handles=legend_elements, loc=legendloc, bbox_to_anchor=legend_adjust,
              prop={'size': legendsize})
    
    if save == True:
        plt.savefig(savepath)
        plt.close()
        
    else:
        return plt.show()
    
def plottTS(df, ax, prop, **kwargs):
    '''Returns plots of storage modulus, loss modulus, and 
    tandelta vs. frequency for each temperature'''
    title = kwargs.get('title', '')
    
    tempstep = kwargs.get('tempstep', 2.5)
    
    aT = kwargs.get('aT', None)
    bT = kwargs.get('bT', None)

       
    Tmin = round(min(df['temp']),0)
    Tmax = round(max(df['temp']),0)
    num_temps = int(round((Tmax-Tmin)/tempstep,0) + 1)
    
    #set color scale depending on number of testing temperatures
    cmap = plt.cm.magma;
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                        cmaplist, cmap.N)
    temps = np.linspace(Tmin, Tmax, num_temps)
    mult = round(256/num_temps)
    norm = mpl.colors.BoundaryNorm(temps, cmap.N)
    
    # set all horizontal shift factors to 1 if not provided
    if aT == None:
        aT = {}
        for t in temps:
            aT[t] = 1
    
    # set all vertical shift factors to 1 if not provided
    if bT == None:
        bT = {}
        for t in temps:
            bT[t] = 1
    
        # now plot the desired property
        for t in temps:
            i = int(round(t - Tmin)/tempstep)
            ax.loglog(df.query('temp > @t-0.5 & temp < @t+0.5')['freq']*aT[t], 
                         df.query('temp > @t-0.5 & temp < @t+0.5')[prop]*bT[t], 
                         '.-', ms=10, lw=3, color=cmaplist[i*mult])

    
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = [];
        plt.colorbar(sm, ax=ax, cmap=cmap, norm=norm, 
                          label='Temperature ($^{\\circ}$C)')
        ax.set_title(title)


def VFTtau(T, tauref, Tref, B, Tinf):
    lnaT=-B/(Tref-Tinf)+B/(T-Tinf)
    return tauref*np.exp(lnaT)

def fitVFT(ax, aT, **kwargs):
    '''Fits tTS shift factors to Vogel-Fulcher-Tammann form and plots aT vs. T'''
    
    title = kwargs.get('title', None);
    Bguess = kwargs.get('Bguess', 3000);
    
    Tref = float(next(k for k, v in aT.items() if v == 1))
    Tguess = Tref-50
    
    palette = ['#0093F5', '#F08E2C', '#000000']
    
    #plot aT vs. T
    ax.set_xlabel('T ($^{\\circ}$C)')
    ax.set_ylabel(r'$a_T$')
    Tvals = [float(x) for x in aT.keys()]
    aTvals = [aT[i] for i in list(aT.keys())]
    ax.semilogy(Tvals, aTvals, 'o', color=palette[0], label = 'Expt.')
    
    # define the fitting function (Vogel-Fulcher-Tamman Eq.)
    def lnaT_VFT(T, B, Tinf):
        return -B/(Tref-Tinf)+B/(T-Tinf)

    # do the curve fit
    popt, pcov = curve_fit(lnaT_VFT, Tvals, np.log(aTvals), 
                           p0=[Bguess, Tguess], 
                           maxfev=5000,
                           bounds=([500, Tref-100],[5000, Tref-20]))

    # extract the fit values for B and Tinf
    B, Tinf = popt
    
    # Assign errors from fitting to physical variables
    B_err, Tinf_err = np.sqrt(np.diag(pcov))

    # now plot the fit values
    Tfit = np.linspace(min(Tvals), max(Tvals), 100)
    aTfit = np.exp(lnaT_VFT(Tfit, B, Tinf))
    ax.semilogy(Tfit, aTfit, '--', linewidth=2, color=palette[1], label = f'B={B:.0f}K; $T_\\infty$={Tinf:.0f}$^\\circ$C')

    # add the legend
    ax.legend(); ax.set_title(title);
        
    return B, B_err, Tinf, Tinf_err

def fitArrhenius(aT, **kwargs):
    '''Fits tTS shift factors to Arrhenius form and plots aT vs. T'''
    
    title = kwargs.get('title', ' ');
    savepath = kwargs.get('savepath', None)
    
    palette = ['#0093F5', '#F08E2C', '#000000']
    
    for k in aT.keys():
        if aT[k] == 1:
            Tref = float(k)
    
    #plot aT vs. T
    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
    ax.set_xlabel('Temperature ($^{\\circ}$C)')
    ax.set_ylabel(r'$a_T$')
    Tvals = [float(x) for x in aT.keys()]
    aTvals = [aT[i] for i in list(aT.keys())]
    ax.semilogy(Tvals, aTvals, 'o', color=palette[0], label = 'Expt.')
    
    # define the fitting function (Arrhenius)
    def lnaT_Arrhenius(T, Ea):
        return (Ea/8.314)*(1/(T+273) - 1/(Tref+273))

    # do the curve fit
    popt, pcov = curve_fit(lnaT_Arrhenius, Tvals, np.log(aTvals), maxfev=5000,
                           absolute_sigma=False)

    # extract the fit values for Ea
    Ea = popt[0]
    
    # Assign errors from fitting to physical variables
    Ea_err = np.sqrt(np.diag(pcov)[0])
    #print(Ea_err/1000)

    # now plot the fit values
    Tfit = np.linspace(min(Tvals), max(Tvals), 100)
    aTfit = np.exp(lnaT_Arrhenius(Tfit, Ea))
    ax.semilogy(Tfit, aTfit, '--', linewidth=2, color=palette[1], 
                label = f'$E_a$={Ea/1000:.0f} $\pm$ {Ea_err/1000:.0f} kJ/mol')

    # add the legend
    ax.legend(); ax.set_title(title);
    
    if savepath:
        plt.savefig(savepath)
        #plt.close()
    
    return Ea, Ea_err

def Eg_VFT(B, Tg, Tinf, **kwargs):
    '''Calculates effective activation energy of glass transition from VFT parameters'''
    
    B_err = kwargs.get('B_err', 0)
    Tg_err = kwargs.get('Tg_err', 0)
    Tinf_err = kwargs.get('Tinf_err', 0)
    
    R = 8.3145 # universal gas constant
    
    Eg = R*B*(Tg + 273)**2/(Tg - Tinf)**2
    Bvar = B_err**2*(Eg/B)**2
    Tgvar = Tg_err**2*(2*R*B*(Tg+273)/(Tg - Tinf)**3)**2
    Tinfvar = Tinf_err**2*(2*R*B*(Tg+273**2)/(Tg - Tinf)**3)**2
    Eg_err = np.sqrt(Bvar + Tgvar + Tinfvar)
    return Eg, Eg_err

def fitPowerLaw(df, **kwargs):
    '''Fits modulus (or other data) and time values to power law model'''
    
    df['lnmod'] = np.log(df['modulus'])
    df['lntime'] = np.log(df['time'])
    df = df.dropna()
    
    #plot modulus vs. t
    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Relaxation Modulus (Pa)')
    
    ax.loglog(df['time'], df['modulus'], 'o', color=palette[0], label = 'Expt.')
    
    def lnPowerLaw(logt, lnG0, m):
        return lnG0 - m*logt
    
    # do the curve fit
    popt, pcov = curve_fit(lnPowerLaw, df['lntime'], df['lnmod'], 
                           bounds=((np.log(1e4),0),(np.log(1e12),1)),
                            p0=[np.log(1e7), 0.5], maxfev=5000)#,
                            #sigma=[x for x in df['lnmod']])

    # extract the fit values for parameters
    lnG0, m = popt;
    lnG0_err, m_err = [float(np.sqrt(p)) for p in np.diag(pcov)]
    G0_err = np.exp(lnG0 + lnG0_err) - np.exp(lnG0)

    # now plot the fit values
    tfit = np.linspace(min(df['lntime']), max(df['lntime']), 1000)
    modfit = lnPowerLaw(tfit, lnG0, m)
    ax.loglog([np.exp(t) for t in tfit], [np.exp(m) for m in modfit], '--', linewidth=2, color=palette[1],
                label = f'$G_0$={np.exp(lnG0):.1e} $\pm$ {G0_err:.1e} Pa;\n m={m:.2f} $\pm$ {m_err:.2f}')

    # add the legend
    ax.legend()
    return plt.show()

def fitHybrid(df, B, Tinf, **kwargs):
    '''Fits data to product VFT/Arrhenius form and plots data vs. T'''
    
    Eaguess = kwargs.get('Eaguess', 1.9e5)
    tau0_arrguess = kwargs.get('tau0_arrguess', 1e-12)
    tau0_vftguess = kwargs.get('tau0_vftguess', 1e-12)
    title = kwargs.get('title', None)
    
    palette = ['#0093F5', '#F08E2C', '#000000']
    
    #plot aT vs. T
    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
    ax.set_xlabel('T ($^{\\circ}$C)')
    ax.set_ylabel('Relaxation Time (s)')
    ax.semilogy(df['temp'], df['tau'], 'o', color=palette[0], label = 'Expt.')
    
    # define the fitting function (Hybrid)
    def addVFTArrhenius(T, tau0_vft, tau0_arr, Ea):
        R = 8.3145 # universal gas constant
        return tau0_vft*np.exp(B/(T - Tinf)) + tau0_arr*np.exp(Ea/(R*(T + 273)))

    # do the curve fit
    popt, pcov = curve_fit(addVFTArrhenius, df['temp'], df['tau'], 
                           bounds=((1e-16, 1e-16, 1.0e5),(1e-9, 1e-9, 2.2e5)),
                           p0=[tau0_vftguess, tau0_arrguess, Eaguess], maxfev=5000,
                           sigma=[t for t in df['tau']], absolute_sigma=True)

    # extract the fit values for tau00 and Ea
    tau0_vft, tau0_arr, Ea = popt
    tau0_vft_err, tau0_arr_err, Ea_err = [float(np.sqrt(p)) for p in np.diag(pcov)]

    # now plot the fit values
    Tfit = np.linspace(min(df['temp']), max(df['temp']), 100)
    taufit = addVFTArrhenius(Tfit, tau0_vft, tau0_arr, Ea)
    ax.semilogy(Tfit, taufit, '--', linewidth=2, color=palette[1],
                label = f'$E_a$={Ea/1000:.0f} kJ/mol;\n $\u03c4_0,Arr$={tau0_arr:0.2e} s; \n $\u03c4_0,VFT$={tau0_vft:0.2e} s')
    ax.set_title(title)
    # add the legend
    ax.legend()
    return print(tau0_vft_err/tau0_vft, tau0_arr_err/tau0_arr, Ea_err/Ea)

def fitMaxwell(df, **kwargs):
    '''Fits time and modulus data from stress relaxation experiment
    to exponential decay of single Maxwell element'''
    
    residual = kwargs.get('residual', None)
    ylims = kwargs.get('ylims', [1e4, 1e7])
    title = kwargs.get('title', None)
    
    #plot E vs. t
    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Relaxation Modulus $G(t)$ (Pa)')
    
    df.dropna()
    
    ax.loglog(df['time'], df['modulus'], '.', color=palette[0], label = 'Expt.')
    
    if residual:
        #define maxwell response with residual modulus
        def maxwell(t, G0, tau):
            return residual + G0*np.exp(-t/tau)
    
    else:
        #define maxwell element response
        def maxwell(t, G0, tau):
            return G0*np.exp(-t/tau)
    
    # do the curve fit
    popt, pcov = curve_fit(maxwell, df['time'], df['modulus'],
                           maxfev=5000,
                           p0=[1e7, 10],
                           bounds=([1e5,0.1],[1e8,1e5]))

    # extract the fit value for tau
    G0, tau = popt
    G0_err, tau_err = np.sqrt(np.diag(pcov))

    # now plot the fit values
    tfit = np.logspace(np.log10(min(df['time'])), np.log10(max(df['time'])), 1000)
    Gfit = maxwell(tfit, G0, tau)
    ax.loglog(tfit, Gfit, '--', linewidth=2, color=palette[1],
                label = f'$\u03c4={tau:.1f} \pm {tau_err:.1f}$ s')

    # add the legend
    ax.legend()
    ax.set_title(title)
    ax.set_ylim(ylims)
    return tau, tau_err

def fitKWW(df, **kwargs):
    '''Fits time and modulus data from stress relaxation experiment
    to exponential decay of stretched exponential form'''
    
    Emin = kwargs.get('Emin', 0.01)
    norm = kwargs.get('norm', False)
    residual = kwargs.get('residual', None)
    
    ylims = kwargs.get('ylims', [1e4, 2e7])
    title = kwargs.get('title', '')
    tts = kwargs.get('tts', False)
    markevery = kwargs.get('markevery', 1)
    savepath = kwargs.get('savepath', None)
    
    #plot E vs. t
    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
    
    if tts:
        ax.set_xlabel('Time / $a_T$ (s)')
        ax.set_ylabel('G(t) / $b_T$ (Pa)')
    else:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Relaxation Modulus G(t)')

    df['lnmod'] = np.log(df['modulus'])
    df = df.dropna()
    
    if norm:
        ax.semilogx(df['time'], df['mod_norm'], '.', color=palette[0], label = 'Expt.')
        
    else:
        ax.loglog(df['time'], df['modulus'], '.', markevery=markevery, 
                  color=palette[0], label = 'Expt.')
    
    if residual:
        def KWW(t, G0, tau, beta):
            return residual + G0*np.exp(-(t/tau)**beta)
    else:
        #define KWW
        def KWW(t, G0, tau, beta):
            return G0*np.exp(-(t/tau)**beta)
    
    # do the curve fit
    popt, pcov = curve_fit(KWW, df['time'], df['modulus'],
                           p0=[1e7, 100, 0.95],
                           bounds=((1e5, 1e-1, 0.5),(1e8, 1e5, 1)),
                           maxfev=1000)

    # extract the fit value for tau and beta
    G0, tau, beta = popt
    G0_err, tau_err, beta_err = np.sqrt(np.diag(pcov))
    
    #calculate ensemble average tau and uncertainty
    avgtau = (tau/beta)*gammaf(1/beta)
    tauvar = tau_err**2 * ((1/beta)*gammaf(1/beta))**2
    betavar = beta_err**2 * (tau*gammaf(1/beta)*(beta + digamma(1/beta))/beta**3)**2
    avgtau_err = np.sqrt(tauvar + betavar)

    # now plot the fit values
    tfit = np.logspace(np.log10(min(df['time'])), np.log10(max(df['time'])), 100)
    modfit = KWW(tfit, G0, tau, beta)
    
    # residuals = df['modulus'] - modfit
    # chi_sq = np.sum(residuals**2/modfit)
    # dof = len(modfit) - len(popt)
    # chi_sq_dof = chi_sq/dof
    
    ax.loglog(tfit, modfit, '--', linewidth=2, color=palette[1],
                label = f'$\u03c4*={tau:.1f} \pm {tau_err:0.1f}$ s \n $\u03b2={beta:0.2f} \pm {beta_err:0.2f}$ \n $\u27e8\u03c4\u27e9={avgtau:.1f} \pm {avgtau_err:0.1f}$ s')

    # add the legend
    ax.legend()
    ax.set_title(title)
    ax.set_ylim(ylims)
    plt.show()
    
    if savepath:
        plt.savefig(savepath)
    
    return avgtau, avgtau_err

def MLf(z, a):
    '''
    Returns an array of values corresponding to the evaluation of the
    one parameter Mittag-Leffler function for an array of values z.

    Parameters
    ----------
    z : array
        Array of numbers to evaluate.
    a : float
        Mittag-Leffler parameter a or alpha.

    Returns
    -------
    array
        Array of output values.
    
    Notes
    -----
    - This function is designed primarily for real-valued z and especially
        z = -t^a where t > 0.
    - This was built with help from Stack Overflow and the paper by Gorenflo,
        Loutchko, and Luchko 2002.

    '''
    z = np.atleast_1d(z)
    if a == 0:
        return 1/(1 - z)
    elif a == 1:
        return np.exp(z)
    elif a > 1 or all(z > 0):
        k = np.arange(100)
        return np.polynomial.polynomial.polyval(z, 1/gammaf(a*k + 1))

    # a helper for tricky case, from Gorenflo, Loutchko & Luchko
    def _MLf(z, a):
        if z < 0:
            f = lambda x: (np.exp(-x*(-z)**(1/a)) * x**(a-1)*np.sin(np.pi*a)
                          / (x**(2*a) + 2*x**a*np.cos(np.pi*a) + 1))
            return 1/np.pi * quad(f, 0, np.inf)[0]
        elif z == 0:
            return 1
        else:
            return MLf(z, a)
    return np.vectorize(_MLf)(z, a)

def ML2f(z, a, b, **kwargs):
    '''
    Returns an array of values corresponding to the evaluation of the
    two parameter Mittag-Leffler function for an array of values z.

    Parameters
    ----------
    z : array
        Array of numbers to evaluate.
    a : float
        Mittag-Leffler parameter a or alpha.
    b : float
        Mittag-Leffler parameter b or beta
    N : int, default 14
        Number of summation terms for numerical approximation

    Returns
    -------
    array
        Array of output values.
    
    Notes
    -----
    - This function is designed primarily for real-valued z and especially
        z = -t^a where t > 0.
    - The default number of summation terms (14) should provide minimum
        error achievable for default precision in python. Adjust terms
        for speed or precision as needed.
    - This was built based on the hyperbolic Hankel contour approximation
        in the paper by W. McLean 2021.

    '''
    
    N = kwargs.get('N', 14) #number of summation terms for 2 parameter ML
    
    z = np.atleast_1d(z)
    if b == 1:
        return MLf(z, a)
    else:
        def _QNfunc(z, a, b, N):
            
            #define constants
            
            phi = 1.17210
            h = 1.08180/N
            mu = 4.49198*N
            A = 2*phi - np.pi/2

            # intermediate functions for creating lookup table to improve speed
            def wfunc(u):
                return mu*(1 + np.sin(1j*u - phi))

            def Cnfunc(w, nh):
                return np.exp(w)*np.cos(1j*nh - phi)

            table = pd.DataFrame(data=np.arange(0,N+1), columns=['n'])
            table['nh'] = h*table['n']
            table['wnh'] = [wfunc(u) for u in table['nh']]
            table['Cn'] = [Cnfunc(w, nh) for w,nh in zip(table['wnh'], table['nh'])]
            
            def ffunc(w, z, a, b):
                return (w**(a-b))/(w**a - z)
            
            # summation
            total = 0
            for n in table['n']:
                if n == 0:
                    total += table['Cn'][n]*ffunc(table['wnh'][n], z, a, b).real
                else:
                    total += 2*(table['Cn'][n]*ffunc(table['wnh'][n], z, a, b)).real
            
            return A*total
        
        return np.vectorize(_QNfunc)(z, a, b, N)

def fitFracMaxwell(df, **kwargs):
    '''
    Reads a dataframe of stress relaxation data and fits 
    to a fractional Maxwell model according to the formulation laid out
    in Jaishankar & McKinley 2013

    Parameters
    ----------
    df : pd.DataFrame
        This should be a dataframe with a column of time values labelled 'time'
        and a column of modulus values labelled 'modulus'
    model : str, default None
        Flag used to fix exponents if expecting spring or dashpot for one of 
        the springpots.  Use 'gel' to fit to fractional Maxwell gel, which 
        uses a perfect spring; or use 'liquid' to fit to fractional Maxwell 
        liquid, which uses a perfect dashpot.
    aguess : float, default 0.9
        Initial guess for the higher power law exponent.
    bguess : float, default 0.05
        Initial guess for the lower power law exponent.
    Gguess : float, default 1e7
        Initial guess for the first quasi-property.
    Vguess : float, default 1e8
        Initial guess for the second quasi-property.
    residual : float, default None
        Residual modulus to fit with model if expected or known.
    params_out : boolean, default False
        Option to output all the fitting parameters
    ylims : list, default [1e4, 1e7]
        y-axis limits
    title : str, default None
        Optional title for the output plot.
    tts : boolean, default False
        When False, x-axis is Time, but when True, x-axis is aT*Time
    savepath : str, default None
        Path to location to save plot.  If None, does not save.
    

    Returns
    -------
    A plot showing the experimental data and the fit.
    tau : float
        The fractional generalization of relaxation time in seconds.

    '''
    
    model = kwargs.get('model', None)
    
    aguess = kwargs.get('aguess', 0.9)
    bguess = kwargs.get('bguess', 0.05)
    Gguess = kwargs.get('Gguess', 1e7)
    Vguess = kwargs.get('Vguess', 1e8)
    
    residual = kwargs.get('residual', None)
    
    params_out = kwargs.get('params_out', False)
    
    ylims = kwargs.get('ylims', [1e4, 1e7])
    title = kwargs.get('title', None) # change title of plot
    tts = kwargs.get('tts', False)
    markevery = kwargs.get('markevery', 1)
    label = kwargs.get('label', 'Expt.')
    
    savepath = kwargs.get('savepath', None)
    
    # Need that 0 <= b < a <= 1
    if bguess > aguess:
        bguess, aguess = aguess, bguess
        
    if aguess == bguess:
        print('We need a > b.')
    
    if aguess < 0 or bguess < 0 or aguess == 0:
        print('The exponents must be between 0 and 1.')
    
    if residual:
        def fracMaxwell(time, G, a, V, b):
            prefactor = [G*t**(-b) for t in time]
            MLtime = [(-G/V)*t**(a-b) for t in time] # transformation of time for
                                                        # Mittag-Leffler input
            MLa = a - b # first parameter for the Mittag-Leffler function
            MLb = 1 - b # second parameter for the Mittag-Leffler function
            
            mleval = [mittag_leffler(t, MLa, MLb) for t in MLtime]
            return [residual + np.real(p*m) for p,m in zip(prefactor,mleval)]
    else:
        def fracMaxwell(time, G, a, V, b):
            prefactor = [G*t**(-b) for t in time]
            MLtime = [(-G/V)*t**(a-b) for t in time] # transformation of time for
                                                        # Mittag-Leffler input
            MLa = a - b # first parameter for the Mittag-Leffler function
            MLb = 1 - b # second parameter for the Mittag-Leffler function
            
            mleval = [mittag_leffler(t, MLa, MLb) for t in MLtime]
            return [np.real(p*m) for p,m in zip(prefactor,mleval)]
    
    # clean up dataframe
    df.reset_index()
    df = df.dropna()
    
    #plot G vs. t
    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
    
    if tts:
        ax.set_xlabel('Time / $a_T$ (s)')
        ax.set_ylabel('G(t) / $b_T$ (Pa)')
    else:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Relaxation Modulus G(t)')
        
    ax.loglog(df['time'], df['modulus'], '.', color=palette[0], 
              markevery=markevery, label = label)
    
    if model == 'gel':
        # use an elastic spring
        b = 0 
        berr = 0
        
        # do the fit
        popt, pcov = curve_fit(lambda time, G, a, V : fracMaxwell(time, G, a, V, b),
                               df['time'], df['modulus'], p0=[Gguess, aguess,
                                                              Vguess],
                               bounds=([1e4, 0.3, 1e4],[1e9, 1, 1e11]))
        
        # extract parameters and uncertainties
        G, a, V = popt
        Gerr, aerr, Verr = np.sqrt(np.diag(pcov))
    
    elif model == 'liquid':
        # use a viscous dashpot
        a = 1
        aerr = 0
        
        # do the fit
        popt, pcov = curve_fit(lambda time, G, V, b : fracMaxwell(time, G, a, V, b),
                               df['time'], df['modulus'], p0=[Gguess, Vguess,
                                                              bguess],
                               bounds=([1e4, 1e4, 0],[1e9, 1e11, 0.5]))
        
        # extract parameters and uncertainties
        G, a, V = popt
        Gerr, aerr, Verr = np.sqrt(np.diag(pcov))
    
    else:
    
        # do the fit
        popt, pcov = curve_fit(fracMaxwell, df['time'], df['modulus'],
                               p0=[Gguess, aguess, Vguess, bguess],
                               bounds=([1e4, 0.3, 1e4, 0],[1e9, 1, 1e11, 0.2]))
        
        # extract parameters and uncertainties
        G, a, V, b = popt
        Gerr, aerr, Verr, berr = np.sqrt(np.diag(pcov))
    
    tau = (V/G)**(1/(a-b)) # fractional equivalent of characteristic relaxation time
    
    #calculate error propagation for tau
    dtaudV = tau/(V*(a-b))
    dtaudG = tau/(G*(a-b))
    dtauda = tau*np.log(V/G)/(a-b)**2
    dtaudb = tau*np.log(V/G)/(a-b)**2
    
    tauerr = np.sqrt((Gerr*dtaudG)**2 + (Verr*dtaudV)**2 + (aerr*dtauda)**2 + (berr*dtaudb)**2)
    
    if model == 'gel':
        fitlabel = f'$\u03c4={tau:.1f} \pm {tauerr:.1f}$ s \n $\u03b1={a:0.2f} \pm {aerr:0.2f}$ \n $\u03b2=0$'
    elif model == 'liquid':
        fitlabel = f'$\u03c4={tau:.1f} \pm {tauerr:.1f}$ s \n $\u03b1=0$ \n $\u03b2={b:0.2f} \pm {berr:0.2f}$'
    else:
        fitlabel = f'$\u03c4={tau:.1f} \pm {tauerr:.1f}$ s \n $\u03b1={a:0.2f} \pm {aerr:0.2f}$ \n $\u03b2={b:0.2f} \pm {berr:0.2f}$'
    
    tfit = np.logspace(np.log10(min(df['time'])), np.log10(max(df['time'])), 100)
    modfit = fracMaxwell(tfit, G, a, V, b)
    ax.loglog(tfit, modfit, '--', linewidth=2, color=palette[1],
              markevery=10,
              label = fitlabel)

    # add the legend and such
    ax.legend()
    ax.set_title(title)
    ax.set_ylim(ylims)
    #ax.set_xlim([min(df['time']), max(df['time'])])
    plt.show()
    
    if savepath:
        plt.savefig(savepath)
    
    if params_out:
        return tau, tauerr, G, Gerr, a, aerr, V, Verr, b, berr
    else:
        return tau, tauerr

def findTg(temp, data):
    '''Returns the Tg for a dataset using the max of tand or loss modulus'''
    
    #find highest value of data
    maxi = np.argmax(data)
    
    #set fit range around data max
    fitrange_temp = temp[maxi-5:maxi+5]
    fitrange_data = data[maxi-5:maxi+5]
    
    #harmonic fit to range
    pfit = np.polyfit(fitrange_temp,fitrange_data,2)
    
    #find max in fitted data and associated temperature
    tempfit = np.linspace(temp[maxi-5],temp[maxi+5],100)
    datafit = [pfit[0]*x**2 + pfit[1]*x + pfit[2] for x in tempfit]
    fitmaxi = np.argmax(datafit)
    
    Tg = round(tempfit[fitmaxi],1)
    return print(Tg)

def findTandMax(df):
    '''Returns the temperature of maximum in tand for frequency sweep data'''
    
    invT = [];
    for i in np.arange(0, len(df.keys()), 1):
        try:
            #find max in tand
            maxi = np.argmax(df[list(df.keys())[i]]['Tan(delta)'])
            #set fit range
            fitrange_temp = df[list(df.keys())[i]]['Temperature'][maxi-2:maxi+2]
            fitrange_tand = df[list(df.keys())[i]]['Tan(delta)'][maxi-2:maxi+2]
            #harmonic fit to range
            pfit = np.polyfit(fitrange_temp,fitrange_tand,2)
            
            #find max in fitted data and associated temperature
            tempfit = np.linspace(df[list(df.keys())[i]]['Temperature'][maxi-2],
                                  df[list(df.keys())[i]]['Temperature'][maxi+2],50)
            tandfit = [pfit[0]*x**2 + pfit[1]*x + pfit[2] for x in tempfit]
            fitmaxi = np.argmax(tandfit)
            t = tempfit[fitmaxi] +273
            
            invT.append(1/t)
        except KeyError:
            invT.append(np.nan)
        except TypeError:
            invT.append(np.nan)
        
    return invT

def freqEa(path, **kwargs):
    '''Fits temperature and frequency data to measure the Ea of tand peak'''
    
    freq_num = kwargs.get('freq_num',31);
    title = kwargs.get('title',' ')
    
    #read in tTS by frequency
    A = readtTS(path, keys='freq', freq_num=freq_num);
    #find inverse Temperatures of each peak in tand by frequency
    B = findTandMax(A);
    #fitting log10(frequencies)
    freqs = np.linspace(-2,1,freq_num).tolist()
    
    for j in reversed(np.arange(0,len(B))):
        if float('-inf') < B[j] < float('inf'):
            continue
        else:
            B.pop(j);
            freqs.pop(j);
    
    
    #linear fit between log10(frequency) and inverse temperature
    linfit = np.polyfit(B,freqs,1)
    fitx = [(y - linfit[1])/linfit[0] for y in freqs]
    slope = linfit[0];
    Ea = round(-1*slope*8.314e-3*np.log(10),0)
    
    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)

    p1, = ax.plot([1000*b for b in B], freqs, 'o', color=palette[0], label='Expt. Data')
    p2, = ax.plot([1000*x for x in fitx], freqs, '--', color=palette[1], label=f'$E_a$={Ea:.0f}kJ/mol')
    ax.set_ylabel('$log_{10}$ f'); ax.set_xlabel('Inverse Temperature (1000$K^{-1}$)')
    ax.legend(); ax.set_title(title)


def findEpr(temp, stor):
    '''Returns the rubbery storage modulus minimum and temperature'''
    mini = np.argmin(stor)
    T = temp[mini]
    Epr = np.min(stor)
    return Epr, T

def compStor(path,f1,f2,f3,f4):
    viridis = ['#440154', '#39568C', '#1F968B', '#73D055', '#FDE725', '#000000']
    palette = ['#0093F5', '#F08E2C', '#000000', '#424EBD', '#B04D25']
    A = readDMA(path, f1); B = readDMA(path, f2); C = readDMA(path, f3);
    D = readDMA(path, f4); #E = readDMA(path, f5)
    temp1 = A[0]; temp2 = B[0]; temp3 = C[0]; temp4 = D[0]; #temp5 = E[0];
    stor1 = A[1]; stor2 = B[1]; stor3 = C[1]; stor4 = D[1]; #stor5 = E[1];
    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
    p1, = ax.semilogy(temp1, stor1, '.-', markersize=10, linewidth=3, color=palette[0], label='DGEBA/MDA')
    p2, = ax.semilogy(temp2, stor2, '.-', markersize=10, linewidth=3, color=palette[1], label='DGEBA/DTDA')
    p3, = ax.semilogy(temp3, stor3, '.-', markersize=10, linewidth=3, color=palette[3], label='BGPDS/MDA')
    p4, = ax.semilogy(temp4, stor4, '.-', markersize=10, linewidth=3, color=palette[4], label='BGPDS/DTDA')
    #p5, = ax.semilogy(temp5, stor5, '.-', markersize=10, linewidth=3, color=viridis[4], label='100%')
    
    ax.set_ylim(1e6, 1e10)
    ax.set_xlim(-125, 225)
    ax.set_xlabel('Temperature ($^oC$)')
    ax.set_ylabel('Storage Modulus, E\' (Pa)')
    ax.set_title('FDS Space Comparison')
    ax.legend(handles=[p1,p2,p3,p4])#,p5])
    return plt.show()

def compLoss(path_base, *f, **kwargs):
    
    temp_min = kwargs.get('temp_min', -125); temp_max = kwargs.get('temp_max', 150);
    mod_min = kwargs.get('mod_min', 1e7); mod_max = kwargs.get('mod_max', 5e8);
    legendsize = kwargs.get('legendsize', 10);
    
    viridis = ['#440154', '#39568C', '#1F968B', '#73D055', '#FDE725', '#000000']
    palette = ['#0093F5', '#F08E2C', '#000000', '#424EBD', '#B04D25', '#75CA85', '#C892D6']
    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)

    for i in np.arange(0,len(f)):
        A = readDMA(path_base.joinpath(f[i]));
        temp = A[0]; loss = A[2];
        ax.semilogy(temp, loss, '-', ms=10, lw=3, color=palette[i], label=str(f[i]).rstrip('.txt'))

    
    ax.set_ylim(mod_min, mod_max)
    ax.set_xlim(temp_min, temp_max)
    ax.set_xlabel('Temperature ($^oC$)')
    ax.set_ylabel('Loss Modulus, E" (Pa)')
    ax.set_title('E" Comparison')
    ax.legend(prop={'size': legendsize})
    return plt.show()

def compTand(path,*f):
    viridis = ['#440154', '#39568C', '#1F968B', '#73D055', '#FDE725', '#000000']
    palette = ['#0093F5', '#F08E2C', '#000000', '#424EBD', '#B04D25', '#75CA85', '#C892D6']
    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
    
    for i in np.arange(0,len(f)):
        A = readDMA(path,f[i]);
        temp = A[0]; tand = A[3];
        ax.semilogy(temp, tand, '.-', ms=10, lw=3, color=palette[i], label=str(f[i]).rstrip('-1.txt'))
        
    # A = readDMA(path, f1); B = readDMA(path, f2); C = readDMA(path, f3);
    # D = readDMA(path, f4); #E = readDMA(path, f5)
    # temp1 = A[0]; temp2 = B[0]; temp3 = C[0]; temp4 = D[0];
    # tand1 = A[3]; tand2 = B[3]; tand3 = C[3]; tand4 = D[3]; #tand5 = E[3];
    
    # p1, = ax.semilogy(temp1, tand1, '.-', markersize=10, linewidth=3, color=palette[0], label='DGEBA/MDA')
    # p2, = ax.semilogy(temp2, tand2, '.-', markersize=10, linewidth=3, color=palette[1], label='DGEBA/DTDA')
    # p3, = ax.semilogy(temp3, tand3, '.-', markersize=10, linewidth=3, color=palette[3], label='BGPDS/MDA')
    # p4, = ax.semilogy(temp4, tand4, '.-', markersize=10, linewidth=3, color=palette[4], label='BGPDS/DTDA')
    # #p5, = ax.semilogy(temp, tand5, '.-', markersize=10, linewidth=3, color=viridis[4], label='100%')
    
    ax.set_ylim(0.01, 2)
    ax.set_xlim(-125, 225)
    ax.set_xlabel('Temperature ($^oC$)')
    ax.set_ylabel('tan$\\delta$')
    ax.set_title('FDS tan$\\delta$ Comparison')
    ax.legend()
    return plt.show()

def fitTwoGaussian(path, **kwargs):
    '''Fits sub-Tg tan delta or loss modulus data to two Gaussian peaks'''
    
    betaprime_ctr = kwargs.get('betaprime_ctr', 25);
    var = kwargs.get('var', 'tand');
    
    if var == 'tand':
        #read in file temperature and tand of interest
        df = readDMA(path)
        i_max = np.argmin(df['tand'][75:108]) + 75
        
        df = df.iloc[0:i_max]
        df['phi'] = np.rad2deg(np.arctan(df['tand'])) #convert tand to phase angle in degrees

        #linear baseline between first and last points
        baseline = [(df['tand'].iloc[0] + ((df['tand'].iloc[-1] - df['tand'].iloc[0])/(df['temp'].iloc[-1] - df['temp'].iloc[0]))*(i - df['temp'].iloc[0])) for i in df['temp']]
        tand_base = np.subtract(np.array(df['tand']),np.array(baseline))

        def twoGaussian(x, *params):
            y = np.zeros_like(x);
            ctr1 = params[0]; ctr2 = params[3];
            amp1 = params[1]; amp2 = params[4];
            wid1 = params[2]; wid2 = params[5];
            y = y + amp1*np.exp(-((x - ctr1)/wid1)**2) + amp2*np.exp(-((x - ctr2)/wid2)**2)
            return y

        #guesses for Temp, amplitude, and width for beta and SS peaks
        guess = [-60, 5e-2, 10, betaprime_ctr, 5e-2, 10]; 

        #fit function to data and calculate each individual peak
        popt, pcov = curve_fit(twoGaussian, df['temp'], tand_base, p0=guess)
        fit = twoGaussian(df['temp'], *popt)
        peak1 = [popt[1]*np.exp(-((x - popt[0])/popt[2])**2) for x in df['temp']]
        peak2 = [popt[4]*np.exp(-((x - popt[3])/popt[5])**2) for x in df['temp']]

        # palette = ['#0093F5', '#F08E2C', '#000000', '#424EBD', '#B04D25']
        # fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)

        # p1, = ax.plot(temp, tand_base, '.-', color=palette[2], label='Expt. Data')
        # p2, = ax.plot(temp, fit, '--', color=palette[0], label='Dual Gaussian Fit')
        # p3, = ax.plot(temp, peak1, '--', color=palette[1], label='Beta Peak Fit')
        # p4, = ax.plot(temp, peak2, '--', color=palette[4], label='Disulfide Peak Fit')
        # ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Baselined tan$\delta$');
        # ax.legend(prop={'size': 9})
        
        return popt#,plt.show()
    
    elif var == 'loss':
        #read in file temperature and loss modulus of interest
        A = readDMA(path)
        i_max = np.argmin(A[3][75:108]) + 75
        temp = [T for T in A[0]][0:i_max]; #convert temperature to K
        tand = np.array(A[3][0:i_max]);
        loss = np.array(A[2][0:i_max]);
        phi = [math.degrees(math.atan(d)) for d in tand] #convert tand to phase angle in degrees

        #linear baseline between first and last points
        baseline = [(loss[0] + ((loss[-1] - loss[0])/(temp[-1] - temp[0]))*(i - temp[0])) for i in temp]
        loss_base = np.subtract(np.array(loss),np.array(baseline))

        def twoGaussian(x, *params):
            y = np.zeros_like(x);
            ctr1 = params[0]; ctr2 = params[3];
            amp1 = params[1]; amp2 = params[4];
            wid1 = params[2]; wid2 = params[5];
            y = y + amp1*np.exp(-((x - ctr1)/wid1)**2) + amp2*np.exp(-((x - ctr2)/wid2)**2)
            return y

        #guesses for Temp, amplitude, and width for beta and SS peaks
        guess = [-60, 1e8, 10, betaprime_ctr, 1e8, 10]; 

        #fit function to data and calculate each individual peak
        popt, pcov = curve_fit(twoGaussian, temp, loss_base, p0=guess, bounds=((-90, 1e6, 5, -10, 1e6, 5),(-30, 1e9, 50, 80, 1e9, 50)))
        fit = twoGaussian(temp, *popt)
        peak1 = [popt[1]*np.exp(-((x - popt[0])/popt[2])**2) for x in temp]
        peak2 = [popt[4]*np.exp(-((x - popt[3])/popt[5])**2) for x in temp]

        # palette = ['#0093F5', '#F08E2C', '#000000', '#424EBD', '#B04D25']
        # fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)

        # p1, = ax.plot(temp, loss_base, '.-', color=palette[2], label='Expt. Data')
        # p2, = ax.plot(temp, fit, '--', color=palette[0], label='Dual Gaussian Fit')
        # p3, = ax.plot(temp, peak1, '--', color=palette[1], label='Beta Peak Fit')
        # p4, = ax.plot(temp, peak2, '--', color=palette[4], label='Disulfide Peak Fit')
        # ax.set_xlabel('Temperature ($^{\circ}$C)'); ax.set_ylabel('Baselined E"');
        # ax.legend(prop={'size': 9})
        
        return popt#,plt.show()
    

def fitGaussian(path, **kwargs):
    '''Fits sub-Tg tan delta data to single Gaussian peak'''
    
    i_max = kwargs.get('i_max', 80);
    
    #read in file temperature and tand of interest
    A = readDMA(path)
    temp = [T + 273 for T in A[0]][0:i_max]; #convert temperature to K
    tand = np.array(A[3][0:i_max]);
    loss = np.array(A[2][0:i_max]);
    phi = [math.degrees(math.atan(d)) for d in tand] #convert tand to phase angle in degrees

    #linear baseline between first and last points
    baseline = [(tand[0] + ((tand[-1] - tand[0])/(temp[-1] - temp[0]))*(i - temp[0])) for i in temp]
    tand_base = np.subtract(np.array(tand),np.array(baseline))

    def Gaussian(x, *params):
        y = np.zeros_like(x);
        ctr = params[0];
        amp = params[1];
        wid = params[2];
        y = y + amp*np.exp(-((x - ctr)/wid)**2)
        return y

    #guesses for Temp, amplitude, and width for peak
    guess = [210, 5e-2, 10]; 

    #fit function to data and calculate each individual peak
    popt, pcov = curve_fit(Gaussian, temp, tand_base, p0=guess)
    fit = Gaussian(temp, *popt)

    # palette = ['#0093F5', '#F08E2C', '#000000', '#424EBD', '#B04D25']
    # fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)

    # p1, = ax.plot(temp, tand_base, '.-', color=palette[2], label='Expt. Data')
    # p2, = ax.plot(temp, fit, '--', color=palette[0], label='Gaussian Fit')
    # ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Baselined tan$\delta$');
    # ax.legend(prop={'size': 9})
    
    return popt#,plt.show()

#This if statement should be included at the end of any module
#If you don't include this, when you import the module, it will run
#the module script from top to bottom instead of only importing the
#functions	
if __name__ == "__main__":
	pass