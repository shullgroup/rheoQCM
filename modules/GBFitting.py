'''
class for fitting of G and B
'''
import numpy as np
from lmfit import Model, Minimizer, minimize, Parameters, fit_report, printfuncs
from lmfit.models import ConstantModel
from scipy.signal import find_peaks 

# from UISettings import settings_init

# initiate the parameters (for test)
distance = 1e3  # in Hz 
width = 10 # in Hz


class GBFitting:
    def __init__(self):
        # self.gmodel = Model(self.fun_G)
        # self.bmodel = Model(self.fun_B)
        self.gmodel, self.bmodel, self.gpars, self.bpars = make_models_pars(n=1)

########### peak finding functions ###########
def findpeaks(array, output, sortstr=None, npeaks=np.inf, minpeakheight=-np.inf, 
            threshold=0, minpeakdistance=0, widthreference=None, minpeakwidth=0, maxpeakwidth=np.inf):
    '''
    output: 'indices' or 'values'
    sortstr: 'ascend' or 'descend'
    '''
    indices = np.array([]).astype('int64')
    values = np.array([]).astype('float64')
    data = np.atleast_1d(array).astype('float64')
    if data.size < 3:
        return np.array([])

    hnpeaks = 0
    diffs = data[1:]-data[:-1]
    for i in range(diffs.size-1):
        if hnpeaks >= npeaks:
            break
        if diffs[i] > 0 and diffs[i+1] < 0:
            lthreshold = np.absolute(diffs[i])
            rthreshold = np.absolute(diffs[i+1])
            if data[i+1] >= minpeakheight and lthreshold >= threshold and rthreshold >= threshold:
                indices = np.append(indices, i+1)
                values = np.append(values, data[i+1])
                hnpeaks = hnpeaks + 1

    indices_copy = np.copy(indices)
    if sortstr:
        if sortstr.lower() == 'ascend':
            order = np.argsort(values)
            values = np.sort(values)
            for i in range(order.size):
                indices[i] = indices_copy[order[i]]
        elif sortstr.lower() == 'descend':
            order = np.argsort(-values)
            values = -np.sort(-values)
            for i in range(order.size):
                indices[i] = indices_copy[order[i]]

    if output.lower() == 'indices':
        return indices
    elif output.lower() == 'values':
        return values

def findpeaks_py(array, output=None, sortstr=None, threshold=None, prominence=None):
    '''

    '''
    indeces, prop = find_peaks(
        array, 
        threshold=threshold, 
        distance=distance, 
        prominence=prominence,
        width=width,
    )

    indices_copy = np.copy(indices)
    if sortstr:
        if sortstr.lower() == 'ascend':
            order = np.argsort(values)
            values = np.sort(values)
            for i in range(order.size):
                indices[i] = indices_copy[order[i]]
        elif sortstr.lower() == 'descend':
            order = np.argsort(-values)
            values = -np.sort(-values)
            for i in range(order.size):
                indices[i] = indices_copy[order[i]]

    if output.lower() == 'indices':
        return indices
    elif output.lower() == 'values':
        return values

def guess_peak_factors(freq, cen_index, resonance):
    ''' 
    guess the factors of a peak.
    input:
        freq: frequency
        cen_index: index of center of the peak
        resonance: G or B or modulus
    output:
        amp: amplitude
        cen: peak center
        half_wid: half-maxium hal-width (HMHW)
    '''

    cen = freq[cen_index] # peak center
    # determine the estimated associated conductance (or susceptance) value at the resonance peak
    Gmax = resonance[cen_index] 
    # determine the estimated half-max conductance (or susceptance) of the resonance peak
    half_Gmax = (Gmax-np.amin(resonance))/2 + np.amin(resonance)
    amp = Gmax-np.amin(resonance)
    half_wid = np.absolute(freq[np.where(np.abs(half_Gmax-resonance)==np.min(np.abs(half_Gmax-resonance)))[0][0]] -  cen)
    return amp, cen, half_wid

########### initial values guess functions ###
def params_guess(f, G, B, n, n_policy='max', method='gmax'):
    '''
    guess initial values based on given method
    if method == 'bmax': use max susceptance
    'gmax': use max conductance
    'derivative': use modulus
    n_policy: 'max' or 'forced'
    '''
    # determine the structure field that should be used to extract out the initial-guessing method
    if method == 'bmax': # use max susceptance
        resonance = B
        x = f
    elif method == 'dev': # use derivative
        resonance = np.sqrt(np.diff(G)**2 + np.diff(B)**2) # use modulus
        x = f[:-1] + np.diff(f) # change f size and shift
    elif method == 'prev': # use previous value
        # this conditin should not go to this function
        return
    else:
        resonance = G
        x = f

    phi = 0
    peak_guess = {}

    # indeces = findpeaks(resonance, output='indices', sortstr='descend')
    
    if not indeces:
        return 
    
    amp, _, half_wid = guess_peak_factors(f, indeces[0], G) # factors of highest peak
    
    if method == 'dev':
        # guess phase angle if derivatave method used
        phi = np.arcsin(G[0] / np.sqrt(G[0]**2 + B[0]**2))

    #TODO check found peaks

    #TODO use other method to guess if failed


    # check 
    if n > len(indeces) and n.lower() != 'forced':
        n = len(indeces) # change n to detected number of peaks
    
    
    for i, idx in enumerate(indeces):
        peak_guess[i] = {
            'amp': amp, 
            'cen': x[idx], 
            'wid': half_wid, 
            'phi': phi
        }

    return n, peak_guess

########### fitting functions ################
def fun_G(x, amp, cen, wid, phi):
    ''' 
    function of relation between frequency (f) and conductance (G) 
    '''
    return amp * (4 * wid**2 * x**2 * np.cos(phi) - 2 * wid * x * np.sin(phi) * (cen**2 - x**2)) / (4 * wid**2 * x**2 + (cen**2 -x**2)**2)

def fun_B(x, amp, cen, wid, phi):
    ''' 
    function of relation between frequency (f) and susceptance (B) 
    '''
    return amp * (4 * wid**2 * x**2 * np.sin(phi) + 2 * wid * x * np.cos(phi) * (cen**2 - x**2)) / (4 * wid**2 * x**2 + (cen**2 -x**2)**2)

def make_gmod(n):
    '''
    make complex model of G (w/o params) for multiple (n) peaks
    input:
    n:    number of peaks
    '''
    gmod = ConstantModel(prefix='g_')
    for i in np.arange(n):
        gmod_i = Model(fun_G, prefix='p'+str(i)+'_', name='g'+str(i))
        gmod += gmod_i
    return gmod

def make_bmod(n):
    '''
    make complex model of B (w/o params) for multiple (n) peaks
    input:
    n:    number of peaks
    '''
    bmod = ConstantModel(prefix='g_')
    for i in np.arange(n):
        bmod_i = Model(fun_B, prefix='p'+str(i)+'_', name='b'+str(i))
        bmod += bmod_i
    return bmod

def make_gbmodel(n=1):
    '''
    make complex model for multiple peaks
    input:
    n:    number of peaks
    '''
    gmod = ConstantModel(prefix='g_')
    bmod = ConstantModel(prefix='b_')

    for i in np.arange(n):
        # gmod and bmod sharing the same varible so use the same prefix
        gmod_i = Model(fun_G, prefix='p'+str(i)+'_', name='g'+str(i))
        bmod_i = Model(fun_B, prefix='p'+str(i)+'_', name='b'+str(i))
        gmod += gmod_i
        bmod += bmod_i
    
    return gmod, bmod

def make_models(n=1):
    '''
    Since minimizeResult class doesn't have eval_components method, we will make complex models with single peak for evaluation
    input:
        n:    number of peaks
    output:
        gmods = list of n models of G
        bmods = list of n models of B
    '''
    gmods = []
    bmods = []
    gc = ConstantModel(prefix='g_')
    bc = ConstantModel(prefix='b_')

    for i in np.arange(n):
        # gmod and bmod sharing the same varible so use the same prefix
        gmod_i = Model(fun_G, prefix='p'+str(i)+'_', name='g'+str(i)) + gc
        bmod_i = Model(fun_B, prefix='p'+str(i)+'_', name='b'+str(i)) + bc
        gmods.append(gmod_i)
        bmods.append(bmod_i)
    
    return gmods, bmods

def make_models_pars(n=1):
    '''
    make complex model for multiple peaks
    input:
    n:    number of peaks
    '''
    gmod = ConstantModel(prefix='g_', name='cg')
    gpars = gmod.make_params(c=0)
    bmod = ConstantModel(prefix='b_', name='cb')
    bpars = bmod.make_params(c=0)

    for i in np.arange(1, n+1):
        # gmod and bmod sharing the same varible so use the same prefix
        gmod_i = Model(fun_G, prefix='p'+str(i)+'_', name='g'+str(i))
        gpars.update(gmod_i.make_params())
        gpars['p'+str(i)+'_amp'].set(0, min=0)
        gpars['p'+str(i)+'_cen'].set()
        gpars['p'+str(i)+'_wid'].set(1, min=1)
        gpars['p'+str(i)+'_phi'].set(0, min=-np.pi/2, max=np.pi/2)
        bmod_i = Model(fun_B, prefix='p'+str(i)+'_', name='b'+str(i))
        bpars.update(bmod_i.make_params())
        bpars['p'+str(i)+'_amp'].set(0, min=0)
        bpars['p'+str(i)+'_cen'].set()
        bpars['p'+str(i)+'_wid'].set(1, min=1)
        bpars['p'+str(i)+'_phi'].set(0, min=-np.pi/2, max=np.pi/2)
        gmod += gmod_i
        bmod += bmod_i
    
    return gmod, bmod, gpars, bpars

def res_GB(params, f, G, B, **kwargs):
    '''
    residual of both G and B
    '''
    # gmod and bmod have to be assigned to real models
    gmod = kwargs.get('gmod')
    bmod = kwargs.get('bmod')
    eps = kwargs.get('eps', None)
    # eps = 100
    # eps = (G - np.amin(G))
    # eps = pow((G - np.amin(G)*1.001), 1/2)
    
    residual_G = G - gmod.eval(params, x=f)
    residual_B = B - bmod.eval(params, x=f)

    if eps is None:
        return np.concatenate((residual_G, residual_B))
    else:
        return np.concatenate((residual_G * eps, residual_B * eps))

def set_params(f, G, B, n=1, peak_guess=None):
    ''' set the parameters for fitting '''

    params = Parameters()
    for i in np.arange(n):
        if peak_guess is None: 
            amp = np.amax(G) - np.amin(G)
            cen = np.mean(f)
            wid = (np.amax(f) - np.amin(f)) / 6
            phi = 0
        else:
            amp = peak_guess[i]['amp']
            cen = peak_guess[i]['cen']
            wid = peak_guess[i]['wid']
            phi = peak_guess[i]['phi']

        params.add(
            'p'+str(i)+'_amp',      # amplitude (G)
            value=amp,              # init: peak height
            min=0,                  # lb
            max=np.inf,             # ub
        )
        params.add(
            'p'+str(i)+'_cen',      # center 
            value=cen,              # init: average f
            min=np.amin(f),         # lb: assume peak is in the range of f
            max=np.amax(f),         # ub: assume peak is in the range of f
        )
        params.add(
            'p'+str(i)+'_wid',                 # width (fwhm)
            value=wid,                         # init: half range
            min=1,                             # lb
            max=(np.amax(f) - np.amin(f)) * 2, # ub: assume peak is in the range of f
        )
        params.add(
            'p'+str(i)+'_phi',       # phase shift
            value=phi,               # init value: peak height
            min=-np.pi / 2,          # lb
            max=np.pi / 2,           # ub
        )
      
    params.add(
        'g_c',              # initialize G_offset
        value=np.amin(G),    # init G_offset = mean(G)
        # min=-np.inf,        # lb
        # max=np.amax(G)/2,    # ub
    )        
    params.add(
        'b_c',              # initialize B_offset
        value=np.mean(B),   # init B_offset = mean(B)
        # min=np.amin(B)/2,    # lb
        # max=np.amin(B)/2,    # ub
    )
    return params

def minimize_GB(f, G, B, n=1, cen_guess=None, wid_guess=None, factor=None):
    '''
    use leasesq to fit
    '''
   
    # set data for fitting
    if not None in [cen_guess, wid_guess, factor]:
        condition = np.where((f >= cen_guess - wid_guess * factor) & (f <= cen_guess + wid_guess * factor))
        f, G, B = f[condition], G[condition], B[condition]

    # eps = None
    eps = pow((G - np.amin(G)*1.001), 1/2) # residual weight
     
    # set the models
    gmod, bmod = make_gbmodel(n)
    # set params with data
    params = set_params(f, G, B, n=n)
    
    # minimize with leastsq
    # mini = Minimizer(residual, params, fcn_args=(f, G, B))
    # result = mini.leastsq(xtol=1.e-10, ftol=1.e-10)
    result = minimize(res_GB, params, method='leastsq', args=(f, G, B), kws={'gmod': gmod, 'bmod': bmod, 'eps': eps}, xtol=1.e-18, ftol=1.e-18)

    print(fit_report(result)) 
    print('success', result.success)
    print('message', result.message)
    print('lmdif_message', result.lmdif_message)
    return result

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    try:
        from AccessMyVNA_dummy import AccessMyVNA
    except:
        from Modules.AccessMyVNA_dummy import AccessMyVNA 

    # gbfitting = GBFitting()
    # # print(gbfitting.gmodel.param_names)
    # # print(gbfitting.bmodel.param_names)
    # # print(gbfitting.gpars)
    # # print(gbfitting.bpars)

    # gmod = gbfitting.gmodel
    # bmod = gbfitting.bmodel
    accvna = AccessMyVNA()
    _, f, G = accvna.GetScanData(nWhata=-1, nWhatb=15)
    _, _, B = accvna.GetScanData(nWhata=-1, nWhatb=16)
    # G = G * 1e3
    # B = B * 1e3

    n = 2

    result = minimize_GB(f, G, B, n, )
    params = set_params(f, G, B, n)
    # result = minimize(res_GB, params, method='leastsq', args=(f, G, B), kws={'eps': pow((G - np.amin(G)*1.001), 1/2), 'n': n}, xtol=1.e-10, ftol=1.e-10)
    # eixt(0)
    # print(fit_report(result)) 
    # print('success', result.success)
    # print('message', result.message)
    # print('lmdif_message', result.lmdif_message)
    print('params', result.params.get('p1_cen').value)
    print('params', result.params.get('p1_cen').stderr)
    print('params', result.params.valuesdict())
    # print(result.params)
    # print(params['p1_amp'].vary)

    # exit(0)
    gmod, bmod = make_gbmodel(n)
    gmods, bmods = make_models(n)
    # gpars = gmod.guess(G, x=f) #guess() not implemented for CompositeModel
    plt.figure()
    plt.plot(f, G, 'bo')
    plt.plot(f, gmod.eval(result.params, x=f), 'k--')
    if n > 1:
        for i in range(n):
            plt.plot(f, gmods[i].eval(result.params, x=f))
    plt.twinx()
    plt.plot(f, B, 'go')
    plt.plot(f, bmod.eval(result.params, x=f), 'k--')
    if n > 1:
        for i in range(n):
            plt.plot(f, bmods[i].eval(result.params, x=f))

    plt.figure()
    plt.plot(G, B, 'bo')
    plt.plot(gmod.eval(result.params, x=f), bmod.eval(result.params, x=f), 'k--')
    if n > 1:
        for i in range(n):
            plt.plot(gmods[i].eval(result.params, x=f), bmods[i].eval(result.params, x=f))

    plt.figure()
    Y = np.sqrt(np.diff(G)**2 + np.diff(B)**2)
    Y_fit = np.sqrt(np.diff(gmod.eval(result.params, x=f))**2 + np.diff(bmod.eval(result.params, x=f))**2)
    print(len(f[0:-1]), len(np.diff(f)))
    df = f[0:-1] + np.diff(f)
    plt.plot(df, Y, 'bo')
    plt.plot(df, Y_fit, 'k--')
    if n > 1:
        for i in range(n):
            Y_fit = np.sqrt(np.diff(gmods[i].eval(result.params, x=f))**2 + np.diff(bmods[i].eval(params, x=f))**2)
            plt.plot(df, Y)

    plt.show()
    exit(0)

    plot_components = True
    # plot results
    fig = plt.figure()
    plt.plot(f, G, 'bo')
    if plot_components:
        # generate components
        comps = result.eval_components(x=f)
        plt.plot(f, 10*comps['cg'], 'k--')
        plt.plot(f, 10*comps['cb'], 'r-')
    else:
        plt.plot(f, result.init_fit, 'k--')
        plt.plot(f, result.best_fit, 'r-')

    result = bmod.fit(B, params=params, x=f)
    print(result.fit_report())
    plt.show()
    exit(1)
    plot_components = False
    # plot results
    fig = plt.figure()
    plt.plot(f, B, 'bo')
    if plot_components:
        # generate components
        comps = result.eval_components(x=f)
        plt.plot(f, 10*comps['jump'], 'k--')
        plt.plot(f, 10*comps['gaussian'], 'r-')
    else:
        plt.plot(f, result.init_fit, 'k--')
        plt.plot(f, result.best_fit, 'r-')
    plt.show()