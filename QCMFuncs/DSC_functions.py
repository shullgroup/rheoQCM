#Import any individual functions from outside packages 
#that are used in your functions.
#These are called dependencies.
# updated version of this file is maintained at
# https://github.com/shullgroup/rheoQCM/blob/master/QCMFuncs/DSC_functions.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.signal import savgol_filter

from scipy.optimize import curve_fit

def is_numeric(cell):
    '''
    Check if a cell has numeric data. Used to start reading data files.
    
    Parameters
    ----------
    cell : str
        String which may be numeric (cell in a given row)
    
    Returns
    -------
    bool
        If cell is numeric (True) or not (False)
    
    '''
    try:
        float(cell)
        return True
    except ValueError:
        return False
    
def readDSC(path, **kwargs):
    '''
    Read txt file from DSC experiment and convert to a DataFrame
    
    Parameters
    ----------
    path : Path
        Path object to the .txt file containing the DSC data

    mode : str, default 'conv'
        Flag for mode of DSC, i.e. conventional (conv) vs. moduluated (mdsc)

    savgol_window : int, default 151
        Window length for filtering derivative data.  Must be odd int

    savgol_polyoder : int, default 4
        Polynomial degree for filtering derivative data.  Must be 
        less than window length.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing relevant experimental data, i.e.
        temperature, norm. heat flow, deriv. heat flow
        
    '''
    mode = kwargs.get('mode', 'conv')
    savgol_window = kwargs.get('savgol_window', 151)
    savgol_polyorder = kwargs.get('savgol_polyorder', 4)

    if mode == 'mdsc':
        with open(path, 'r') as f:
            #check which row starts the data
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if any(is_numeric(cell) for cell in row if cell.strip()):
                    start_row = i
                    break
                    
            df = pd.read_csv(f, delimiter="\t", skiprows=start_row,
                            usecols=[0,1,2,3,7],
                            names=['time','temp','q_rev','q_non','dq_revdT'])

        # drop nans
        df = df.dropna()
        # smooth q_rev data
        # uses Savitsky-Golay filter with default window 151 and polyorder 4
        # because raw exported data extremely coarse
        df['q_rev'] = savgol_filter(df['q_rev'], 
                                window_length=savgol_window, 
                                polyorder=savgol_polyorder)


        return df
    
    elif mode == 'conv':
        with open(path, 'r') as f:
            #check which row starts the data
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if any(is_numeric(cell) for cell in row if cell.strip()):
                    start_row = i
                    break
                    
            df = pd.read_csv(f, delimiter="\t", skiprows=start_row,
                            usecols=[0,1,2],
                            names=['time','temp','q'])

        df = df.dropna()
       
        dq = np.gradient(df['q'])
        dT = np.gradient(df['temp'])
        
        df['dqdT'] = np.divide(
                            dq,
                            dT,
                            out=np.full_like(dq, np.nan),
                            where=dT != 0)


        return df

def plotDSC(df, **kwargs):
    '''
    Generate typical plots for DSC experiments. Emphasize primarily placed
    on finding Tg as opposed to other transitions for now.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experimental data read in from the readDSC function
    mode : str, default 'conv'
        DSC mode used for experiment. Options are 'conv' for conventional DSC
        (i.e. heating and cooling ramps) or 'mdsc' for temperature modulated DSC.
    title : str, default None
        Title for the plot.
    savepath : Path, default None
        Path for saving the plot if you want to.
    min_temp : float, default None
        Minimum temperature to use for the DSC experiment.
    max_temp : float, default None
        Maximum temperature to use for the DSC experiment.
    showTg : bool, default True
        Option to show or not show Tg chosen from maximum in derivative
        e.g. if using the fitGaussian function below to find Tg.
    no_legend : bool, default False
        Option to remove legend e.g. if using legend generated from fitGaussian
    legendloc : int or str, default 0
        Location of legend if need to manually change
    legendsize : float, default 9.
        Size of legend if need to manually change
    show_deriv_ticks : bool, default False
        Option to show the tick marks on the twin axis for the derivative heat
        flow. Default False because typically too crowded.

    Returns
    -------
    Tg : float
        Glass transition temperature (Tg) in deg. Celsius chosen as maximum in 
        derivative of heat flow.
    twin : mpl.axes.Axes
        Axes instance used for plotting the derivative heat flow.  Output used
        for carrying to fitGaussian function for better calculation of 

    '''

    mode = kwargs.get('mode', 'conv')
    title = kwargs.get('title', None)
    savepath = kwargs.get('savepath', None)
    min_temp = kwargs.get('min_temp', None)
    max_temp = kwargs.get('max_temp', None)
    no_legend = kwargs.get('no_legend', False)
    legendloc = kwargs.get('legendloc', 0)
    legendsize = kwargs.get('legendsize', 10.)
    show_deriv_ticks = kwargs.get('show_deriv_ticks', False)

    # replace infs with nan and drop nans
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    # filter temperatures
    if min_temp != None:
        df = df.query('temp > @min_temp')
    if max_temp != None:
        df = df.query('temp < @max_temp')

    # choose which mode: conventional or mdsc
    if mode == 'conv':
        q = 'q'
        peak = df['dqdT'].min()
        Tg = float(df.query('dqdT == @peak')['temp'].iloc[0])
    elif mode == 'mdsc':
        q = 'q_rev'
        peak = df['dq_revdT'].min()
        Tg = float(df.query('dq_revdT == @peak')['temp'].iloc[0]) 

    #typical DSC plot with derivative to show Tg
    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
    twin = ax.twinx()
    twin._get_lines = ax._get_lines

    # plot q and dqdT
    ax.plot(df['temp'], df[f'{q}'], '-', label='q')
    twin.plot(df['temp'], -df[f'd{q}dT'], '--', label='dq/dT')
    
    # fit data to Gaussian for better Tg selection
    Tg, dT = fitGaussian(df, ax=twin)
    
    ax.set_xlabel('Temperature ($^\\circ$C)')
    ax.set_xlim([min_temp-5, max_temp+15])
    ax.set_ylabel('Norm. Heat Flow (W/g)')
    twin.set_ylabel('Deriv. Heat Flow (W/g*$^{\\circ}$C)')
    ax.set_title(title)
    
    if no_legend:
        pass
    else:
        handles, labels = ax.get_legend_handles_labels()
        handles_t, labels_t = twin.get_legend_handles_labels()
        ax.legend(
            handles=[*handles,*handles_t],
            labels = [*labels,*labels_t],
            loc=legendloc,
            prop={'size':legendsize})
    ax.annotate('Exo Up',(5,5),xycoords='axes points')
    
    if not show_deriv_ticks:
        twin.set_yticks([])
    
    if savepath:
        plt.savefig(savepath)

    return Tg, twin

def fitGaussian(df, ax, **kwargs):
    '''
    Fits data to single Gaussian peak and adds to DSC plot

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing DSC data.
    ax : mpl.axes.Axes
        Axes object to plot Gaussian fit on. Typically twin from DSC plot.
    return_err : bool, default False
        Option to return uncertainties for Tg and dT

    Returns
    -------
    Tg : float
        Glass transition temperature from center of Gaussian fit in deg. C
    Tg_err : float
        Uncertainty in Tg fit in deg. C
    dT : float
        Breadth of glass transition from width of Gaussian fit in deg. C
    dT_err : float
        Uncertainty in dT fit in deg. C

    '''
    
    return_err = kwargs.get('return_err', False)

    def Gaussian(x, *params):
        y = np.zeros_like(x);
        ctr = params[0];
        amp = params[1];
        wid = params[2];
        y0 = params[3];
        y = y + y0 + amp*np.exp(-((x - ctr)/(2*wid))**2)
        return y

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    #guesses for Temp, amplitude, and width for peak
    maxdqdT = df['dqdT'].min()
    guess = [df.query('dqdT == @maxdqdT')['temp'].iloc[0], 3e-2, 10, 0]
    
    #fit function to data and plot peak
    popt, pcov = curve_fit(Gaussian, df['temp'], -df['dqdT'], p0=guess, 
                           bounds=([-100,0,0,-1],[200,1,30,1]))
    fit_temp = np.linspace(df['temp'].min(),df['temp'].max(),num=1000)
    fit = Gaussian(fit_temp, *popt)
    dT = popt[2]
    Tg = popt[0]
    # parameter uncertainties
    perr = np.sqrt(np.diag(pcov))
    Tg_err = perr[0]
    dT_err = perr[2]

    ax.plot(fit_temp, fit, ':', color='k',
            label=f'T$_g = {Tg:0.1f} ^\\circ$C \n $\u03b4T = {dT:0.1f} ^\\circ$C')
    
    
    if return_err:
        return Tg, Tg_err, dT, dT_err
    else:
        return Tg, dT