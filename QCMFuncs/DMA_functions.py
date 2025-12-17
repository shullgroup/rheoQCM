#Import any individual functions from outside packages 
#that are used in your functions.
#These are called dependencies.
# updated version of this file is maintained at
# https://github.com/shullgroup/rheoQCM/blob/master/QCMFuncs/DMA_functions.py
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import quad
from scipy.special import gamma as gammaf
from scipy.special import digamma
from pymittagleffler import mittag_leffler
from matplotlib.patches import FancyArrowPatch


def double_headed_arrow(ax, x1, y1, x2, y2, 
                        color='C0', linewidth=2, 
                        mutation_scale=15, 
                        zorder=3, 
                        **kwargs):
    """
    Draw a double-headed arrow from (x1, y1) to (x2, y2) on the given axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x1, y1, x2, y2 : float
        Coordinates in data space.
    color : str
        Arrow color.
    linewidth : float
        Line width.
    mutation_scale : float
        Controls the size of the arrowheads.
    zorder : int
        Drawing order.
    kwargs : dict
        Passed through to FancyArrowPatch (e.g., alpha, linestyle).
    """
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='<->',           # double-headed
        mutation_scale=mutation_scale,
        color=color,
        linewidth=linewidth,
        zorder=zorder,
        shrinkA=0.0, shrinkB=0.0,   # no shrinking at ends
        # By default, FancyArrowPatch uses data coordinates from the Axes
        **kwargs
    )
    ax.add_patch(arrow)
    return arrow

palette = ['#0093F5', '#F08E2C', '#000000', '#424EBD', '#B04D25', 
           '#75CA85', '#C892D6', '#007d00']

# axis labels
axlabels = {'storage':r'$E^\prime$ (Pa)',
           'loss': r'$E^{\prime\prime}$ (Pa)',
           'phi':r'$\phi$ (deg.)'}

# function used to figure out number of lines to ignore in DMA input file

def first_numbered_line(file_path):
    """
    Find the line number of the first line in a file that starts with a digit.

    This function reads a text file line by line and returns the line number
    of the first non-empty line whose first non-whitespace character is a digit.
    If no such line exists, it returns None.

    Parameters:
    ----------
    file_path : str
        The path to the text file to be read.

    Returns:
    -------
    int or None
        The line number (1-based index) of the first line starting with a digit,
        or None if no such line is found.
    """
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
    skiprows : int
        Number of rows to skip at beginning of file to remove header. Default skips 
        all lines until the first one beginning with
        a nun-numeric character.

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
        
    
    df['phi'] = np.degrees(np.arctan(df['tand']))
    return df
	
    

def readStressRelax(path, **kwargs):
    """
    Read stress relaxation test data from a tab-delimited file.

    This function reads a text file containing stress relaxation data and
    returns the time and modulus values as pandas Series objects.

    Parameters
    ----------
    path : str
        Path to the tab-delimited file containing the data.
    **kwargs : dict, optional
        Additional keyword arguments (currently unused).

    Returns
    -------
    tuple of pandas.Series
        A tuple containing:
        - t : pandas.Series
            Time values from the 'Step time' column.
        - mod : pandas.Series
            Modulus values from the 'Modulus' column.

    Notes
    -----
    - The file is expected to have columns named 'Step time' and 'Modulus'.
    - The second row of the file is skipped during reading (skiprows=[1]).

    Example
    -------
    >>> t, mod = readStressRelax('data/stress_relax.txt')
    >>> print(t.head(), mod.head())
    """
    with open(path, 'r') as f:
        df = pd.read_csv(f, sep='\t', skiprows=[1])
        t = df['Step time']
        mod = df['Modulus']

    return t, mod



def plotStressRelax(*arg, **kwargs):

    """
    Plot relaxation modulus versus time for one or more datasets.
    
    This function generates a plot of relaxation modulus as a function of time
    for viscoelastic materials, using data provided in the arguments. Each
    dataset should be a tuple or list containing two arrays: time values and
    corresponding modulus values. The function supports normalization and
    different y-axis scales.
    
    Parameters
    ----------
    *arg : tuple of arrays
        One or more datasets, where each dataset is structured as:
        (time_array, modulus_array). Both arrays should be of equal length.
        The first element of each array is ignored (index 0), and plotting
        starts from index 1 onward.
    
    **kwargs : dict, optional
        norm : bool, default=True
            If True, normalizes the modulus values by the second modulus
            value in the dataset (mod[1]).
         : str, default='log'
            Determines the y-axis scale:
            - 'log' for log-log plot
            - 'linear' for semi-log plot (logarithmic x-axis, linear y-axis)
    
    Returns
    -------
    None
        Displays the plot using matplotlib's `plt.show()`.
    
    Notes
    -----
    - The function uses a predefined color palette (`palette`) for multiple
      datasets.
    - The figure size is fixed at (4, 3) inches with constrained layout.
    - Normalization adjusts modulus values relative to the second data point.
    
    Example
    -------
    >>> plotStressRelax((time_array, modulus_array), norm=True, yaxis='log')
    """

    norm = kwargs.get('norm', True)
    yaxis = kwargs.get('yaxis', 'log')

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    a = 0
    for A in arg:
        time = A[0][1:]
        mod = A[1][1:]
        norm_mod = [m / mod[1] for m in mod]

        # Set modulus normalization
        if norm:
            modulus = norm_mod
            ylabel = 'Normalized Relaxation Modulus'
        else:
            modulus = mod
            ylabel = 'Relaxation Modulus, E(t) (Pa)'

        # Set log or linear y-axis
        if yaxis == 'log':
            ax.loglog(time, modulus, '-', color=palette[a])
        elif yaxis == 'linear':
            ax.semilogx(time, modulus, '-', color=palette[a])
        a += 1

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    return plt.show()


def plot_tTS(df, ax, prop, **kwargs):
    """
    Plot time–temperature superposition (TTS) data for a specified property.

    This function creates log-log plots of a given viscoelastic property
    (e.g., storage modulus, loss modulus, or tan delta) versus frequency
    for multiple temperatures. Optional horizontal (aT) and vertical (bT)
    shift factors can be applied for TTS analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing experimental data. Must include columns:
        - 'temp': Temperature values (°C)
        - 'freq': Frequency values (Hz)
        - `prop`: The property to plot (e.g., 'E_storage', 'E_loss').

    ax : matplotlib.axes.Axes
        Matplotlib Axes object on which the plot will be drawn.

    prop : str
        Name of the column in `df` representing the property to plot.

    **kwargs : dict, optional
        title : str, default=''
            Title for the plot.
        tempstep : float, default=2.5
            Temperature increment for color mapping and grouping.
        aT : dataframe, default=None
            Horizontal shift factors for each temperature
            columns are labeled 'temp', and 'aT'
        bT : dict, default=None
            Vertical shift factors for each temperature.
            colums are labeled 'temp' and 'bT'

    Returns
    -------
    None
        Displays the plot on the provided Axes object and adds a colorbar.

    Notes
    -----
    - Color scale uses the 'magma' colormap.
    - Shift factors allow construction of master curves for TTS analysis.

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> plottTS(df, ax, 'E_storage', title='Storage Modulus vs Frequency')
    """
    title = kwargs.get('title', '')
    tempstep = kwargs.get('tempstep', 2.5)
    aT = kwargs.get('aT', None)
    bT = kwargs.get('bT', None)

    Tmin = round(min(df['temp']), 0)
    Tmax = round(max(df['temp']), 0)
    num_temps = int(round((Tmax - Tmin) / tempstep, 0) + 1)

    # Set color scale based on number of temperatures
    cmap = plt.cm.magma
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N
    )
    temps = np.linspace(Tmin, Tmax, num_temps)
    mult = round(256 / num_temps)
    norm = mpl.colors.BoundaryNorm(temps, cmap.N)

    # Default shift factors if not provided
    if aT is None:
        aT = {t: 1 for t in temps}
    if bT is None:
        bT = {t: 1 for t in temps}

    # Plot property for each temperature
    for t in temps:
        i = int(round(t - Tmin) / tempstep)
        subset = df.query('temp > @t-0.5 & temp < @t+0.5')
        if t not in aT.keys():
            continue
        ax.loglog(
            subset['freq'] * aT[t],
            subset[prop] * bT[t],
            '.-',
            ms=10,
            lw=3,
            color=cmaplist[i * mult]
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm._A = []
    plt.colorbar(
        sm,
        ax=ax,
        cmap=cmap,
        norm=norm,
        label='Temperature ($^{\\circ}$C)'
    )
    ax.set_title(title)
    if 'aT' not in list(kwargs.keys()):
        ax.set_xlabel (r'$f$ (s$^{-1}$)')
    else:
        ax.set_xlabel (r'$fa_T$ (s$^{-1}$)')
    if prop in axlabels.keys():
       ax.set_ylabel(axlabels[prop])



def calc_tau_VFT(T, tauref, Tref, B, Tinf):
    """
    Compute relaxation time using the Vogel–Fulcher–Tammann (VFT) equation.

    The VFT equation describes the temperature dependence of relaxation time
    in glass-forming systems and viscoelastic materials. It accounts for the
    dramatic increase in relaxation time as temperature approaches a limiting
    value (Tinf).

    Parameters
    ----------
    T : float
        Temperature at which to calculate the relaxation time (in Kelvin or °C,
        consistent with other parameters).
    tauref : float
        Reference relaxation time at the reference temperature Tref.
    Tref : float
        Reference temperature corresponding to tauref.
    B : float
        VFT constant related to material fragility.
    Tinf : float
        Ideal glass transition temperature (temperature at which relaxation
        time diverges).

    Returns
    -------
    float
        Relaxation time at temperature T.

    Notes
    -----
    The equation used is:
        ln(aT) = -B / (Tref - Tinf) + B / (T - Tinf)
        tau(T) = tauref * exp(ln(aT))

    Example
    -------
    >>> VFTtau(T=300, tauref=1e-3, Tref=298, B=800, Tinf=150)
    0.001234  # Example output
    """
    lnaT = -B / (Tref - Tinf) + B / (T - Tinf)
    return tauref * np.exp(lnaT)



def fitVFT(aT_in, **kwargs):
    """
    Fit time–temperature superposition (TTS) shift factors to the
    Vogel–Fulcher–Tammann (VFT) equation and plot aT vs. temperature.

    This function performs a nonlinear curve fit of experimental shift
    factors (aT) to the VFT model, estimates the parameters B and Tinf,
    and plots both the experimental data and the fitted curve.

    Parameters
    ----------

    T : numpy array
        temperatures
    aT_in : dictionary with columns 'T' and 'aT'
    **kwargs : dict, optional
        title : str, default=None
            Title for the plot.
        Bguess : float, default=3000
            Initial guess for the VFT constant B.
        ax : matplotlib.axes.Axes
        Matplotlib Axes object on which the plot will be drawn.

    Returns
    -------
    tuple
        (B, B_err, Tinf, Tinf_err)
        - B : float
            Fitted VFT constant.
        - B_err : float
            Standard error of B.
        - Tinf : float
            Fitted ideal glass transition temperature.
        - Tinf_err : float
            Standard error of Tinf.

    Notes
    -----
    - The VFT equation used is:
        ln(aT) = -B / (Tref - Tinf) + B / (T - Tinf)
    - Tref is determined as the temperature where aT = 1.

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> B, B_err, Tinf, Tinf_err = fitVFT(ax, aT_dict, title="VFT Fit")
    
    """
        
    title = kwargs.get('title', None)
    Bguess = kwargs.get('Bguess', 3000)
    
    #find referene temperature
    T = aT_in['T']
    aT = aT_in['aT']
    logaT = np.log(aT)
    ref_idx = np.argmin(np.abs(logaT))
    Tref = T[ref_idx]
    Tguess = Tref - 50

    # Define the VFT fitting function
    def lnaT_VFT(T, B, Tinf):
        return -B / (Tref - Tinf) + B / (T - Tinf)

    # Perform curve fitting
    popt, pcov = curve_fit(
        lnaT_VFT,
        T,
        logaT,
        p0=[Bguess, Tguess],
        maxfev=5000,
        bounds=([500, -273], [500000, Tref - 20])
    )

    # Extract fit values and errors
    B, Tinf = popt
    B_err, Tinf_err = np.sqrt(np.diag(pcov))

    if 'ax' in kwargs.keys():
        ax = kwargs.get('ax')
        # Plot aT vs. T
        ax.set_xlabel('T ($^{\\circ}$C)')
        ax.set_ylabel(r'$a_T$')
        ax.semilogy(T, aT, 'o', color=palette[0], label='Expt.')
        # Plot fitted curve
        Tfit = np.linspace(min(T), max(T), 100)
        aTfit = np.exp(lnaT_VFT(Tfit, B, Tinf))
        ax.semilogy(
            Tfit,
            aTfit,
            '--',
            linewidth=2,
            color=palette[1],
            label=f'B={B:.0f}K; $T_\\infty$={Tinf:.0f}$^\\circ$C'
        )
    
        # Add legend and title
        ax.legend()
        ax.set_title(title)

    return B, B_err, Tinf, Tinf_err, Tref

def lnaT_VFT(T, B, Tinf, Tref):
    lnaT = -B / (Tref - Tinf) + B / (T - Tinf)
    return lnaT


def aT_VFT(T, B, Tinf, Tref):
    lnaT = -B / (Tref - Tinf) + B / (T - Tinf)
    return np.exp(lnaT)


def Tg_DMA(B, Tinf, Tref, tau_ref, tau):
    def ftosolve(T):
        return np.log(calc_tau_VFT(T, tau_ref, Tref, B, Tinf)) - np.log(tau)
    soln = least_squares(ftosolve, Tinf+50, 
                                  bounds=(Tinf+10, Tinf+400))
    return soln['x']
    


def fitArrhenius(aT, **kwargs):
    """
    Fit time–temperature superposition (TTS) shift factors to Arrhenius form
    and plot aT vs. temperature.

    Parameters
    ----------
    aT : dict
        Dictionary of shift factors with temperature as keys (°C) and
        corresponding aT values.
    **kwargs : dict, optional
        title : str, default ' '
            Title for the plot.
        savepath : str, default None
            Path to save the plot. If None, the plot is not saved.

    Returns
    -------
    tuple
        Ea : float
            Activation energy (J/mol).
        Ea_err : float
            Standard error of Ea.

    Notes
    -----
    - The Arrhenius equation used is:
        ln(aT) = (Ea / R) * (1 / (T + 273) - 1 / (Tref + 273))
    - Tref is determined as the temperature where aT = 1.

    Example
    -------
    >>> Ea, Ea_err = fitArrhenius(aT_dict, title="Arrhenius Fit")
    """
    title = kwargs.get('title', ' ')
    savepath = kwargs.get('savepath', None)

    # Find reference temperature
    Tref = float(next(k for k, v in aT.items() if v == 1))

    # Plot aT vs. T
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    ax.set_xlabel('Temperature ($^{\\circ}$C)')
    ax.set_ylabel(r'$a_T$')
    Tvals = [float(x) for x in aT.keys()]
    aTvals = [aT[i] for i in list(aT.keys())]
    ax.semilogy(Tvals, aTvals, 'o', color=palette[0], label='Expt.')

    # Define Arrhenius fitting function
    def lnaT_Arrhenius(T, Ea):
        return (Ea / 8.314) * (1 / (T + 273) - 1 / (Tref + 273))

    # Perform curve fit
    popt, pcov = curve_fit(
        lnaT_Arrhenius,
        Tvals,
        np.log(aTvals),
        maxfev=5000,
        absolute_sigma=False
    )

    # Extract Ea and its error
    Ea = popt[0]
    Ea_err = np.sqrt(np.diag(pcov)[0])

    # Plot fitted curve
    Tfit = np.linspace(min(Tvals), max(Tvals), 100)
    aTfit = np.exp(lnaT_Arrhenius(Tfit, Ea))
    ax.semilogy(
        Tfit,
        aTfit,
        '--',
        linewidth=2,
        color=palette[1],
        label=f'$E_a$={Ea / 1000:.0f} ± {Ea_err / 1000:.0f} kJ/mol'
    )

    # Add legend and title
    ax.legend()
    ax.set_title(title)

    if savepath:
        plt.savefig(savepath)

    return Ea, Ea_err



def E_A_VFT(T, B, Tinf, **kwargs):
    """
    Calculate effective activation energy from VFT parameters.

    Parameters
    ----------
    B : float
        VFT constant.
    T : float
        temperature of interest (°C).
    Tinf : float
        Vogel temperature (°C).
    **kwargs : dict, optional
        B_err : float, default 0
            Uncertainty in B.
        Tg_err : float, default 0
            Uncertainty in Tg.
        Tinf_err : float, default 0
            Uncertainty in Tinf.

    Returns
    -------
    tuple
        Eg : float
            Effective activation energy (J/mol).
        Eg_err : float
            Uncertainty in Eg.
    """
    B_err = kwargs.get('B_err', 0)
    Tg_err = kwargs.get('Tg_err', 0)
    Tinf_err = kwargs.get('Tinf_err', 0)

    R = 8.3145  # universal gas constant

    E_A = R * B * (T + 273) ** 2 / (T - Tinf) ** 2
    Bvar = B_err ** 2 * (E_A / B) ** 2
    Tgvar = Tg_err ** 2 * (2 * R * B * (T + 273) / (T - Tinf) ** 3) ** 2
    Tinfvar = Tinf_err ** 2 * (2 * R * B * (T + 273) ** 2 / (T - Tinf) ** 3) ** 2
    E_A_err = np.sqrt(Bvar + Tgvar + Tinfvar)

    return E_A, E_A_err



def fitPowerLaw(df, **kwargs):
    """
    Fit modulus and time values to a power-law model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'time' and 'modulus'.

    Returns
    -------
    None
        Displays a plot of experimental data and fitted curve.
    """
    df['lnmod'] = np.log(df['modulus'])
    df['lntime'] = np.log(df['time'])
    df = df.dropna()

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Relaxation Modulus (Pa)')
    ax.loglog(df['time'], df['modulus'], 'o', color=palette[0], label='Expt.')

    def lnPowerLaw(logt, lnG0, m):
        return lnG0 - m * logt

    popt, pcov = curve_fit(
        lnPowerLaw,
        df['lntime'],
        df['lnmod'],
        bounds=((np.log(1e4), 0), (np.log(1e12), 1)),
        p0=[np.log(1e7), 0.5],
        maxfev=5000
    )

    lnG0, m = popt
    lnG0_err, m_err = [float(np.sqrt(p)) for p in np.diag(pcov)]
    G0_err = np.exp(lnG0 + lnG0_err) - np.exp(lnG0)

    tfit = np.linspace(min(df['lntime']), max(df['lntime']), 1000)
    modfit = lnPowerLaw(tfit, lnG0, m)
    ax.loglog(
        [np.exp(t) for t in tfit],
        [np.exp(m) for m in modfit],
        '--',
        linewidth=2,
        color=palette[1],
        label=(f'$G_0$={np.exp(lnG0):.1e} ± {G0_err:.1e} Pa;\n'
               f'm={m:.2f} ± {m_err:.2f}')
    )

    ax.legend()
    return plt.show()



def fitHybrid(df, B, Tinf, **kwargs):
    """
    Fit data to a hybrid VFT/Arrhenius model and plot relaxation time vs. T.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'temp' and 'tau'.
    B : float
        VFT constant.
    Tinf : float
        Ideal glass transition temperature.
    **kwargs : dict, optional
        Eaguess : float, default 1.9e5
        tau0_arrguess : float, default 1e-12
        tau0_vftguess : float, default 1e-12
        title : str, default None

    Returns
    -------
    None
        Displays plot and prints relative errors.
    """
    Eaguess = kwargs.get('Eaguess', 1.9e5)
    tau0_arrguess = kwargs.get('tau0_arrguess', 1e-12)
    tau0_vftguess = kwargs.get('tau0_vftguess', 1e-12)
    title = kwargs.get('title', None)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    ax.set_xlabel('T ($^{\\circ}$C)')
    ax.set_ylabel('Relaxation Time (s)')
    ax.semilogy(df['temp'], df['tau'], 'o', color=palette[0], label='Expt.')

    def addVFTArrhenius(T, tau0_vft, tau0_arr, Ea):
        R = 8.3145
        return (tau0_vft * np.exp(B / (T - Tinf)) +
                tau0_arr * np.exp(Ea / (R * (T + 273))))

    popt, pcov = curve_fit(
        addVFTArrhenius,
        df['temp'],
        df['tau'],
        bounds=((1e-16, 1e-16, 1.0e5), (1e-9, 1e-9, 2.2e5)),
        p0=[tau0_vftguess, tau0_arrguess, Eaguess],
        maxfev=5000,
        sigma=[t for t in df['tau']],
        absolute_sigma=True
    )

    tau0_vft, tau0_arr, Ea = popt
    tau0_vft_err, tau0_arr_err, Ea_err = [float(np.sqrt(p)) 
                                          for p in np.diag(pcov)]

    Tfit = np.linspace(min(df['temp']), max(df['temp']), 100)
    taufit = addVFTArrhenius(Tfit, tau0_vft, tau0_arr, Ea)
    ax.semilogy(
        Tfit,
        taufit,
        '--',
        linewidth=2,
        color=palette[1],
        label=(f'$E_a$={Ea / 1000:.0f} kJ/mol;\n'
               f'$\\tau_0,Arr$={tau0_arr:0.2e} s;\n'
               f'$\\tau_0,VFT$={tau0_vft:0.2e} s')
    )
    ax.set_title(title)
    ax.legend()
    return print(tau0_vft_err / tau0_vft, tau0_arr_err / tau0_arr, Ea_err / Ea)



def fitKWW(df, **kwargs):
    """
    Fit stress relaxation data to a stretched exponential (KWW) model.

    This function fits time and modulus data from a stress relaxation
    experiment to the Kohlrausch-Williams-Watts (KWW) stretched exponential
    form and plots the experimental data with the fitted curve.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'time' and 'modulus'.
    **kwargs : dict, optional
        norm : bool, default False
            If True, use normalized modulus for plotting.
        residual : float, default None
            Residual modulus to include in the fit.
        ylims : list, default [1e4, 2e7]
            y-axis limits for the plot.
        title : str, default ''
            Plot title.
        tts : bool, default False
            If True, x-axis is aT*Time; otherwise, x-axis is Time.
        markevery : int, default 1
            Marker frequency for plotting.
        savepath : str, default None
            Path to save the plot. If None, the plot is not saved.
        ax : axes handle
            Axis to plot the data and the fit

    Returns
    -------
    tuple
        avgtau : float
            Ensemble average relaxation time (s).
        avgtau_err : float
            Uncertainty in avgtau.

    Notes
    -----
    - The KWW equation used is:
        G(t) = G0 * exp(-(t / tau)^beta)
    - Ensemble average tau is computed as:
        ⟨tau⟩ = (tau / beta) * Gamma(1 / beta)

    Example
    -------
    >>> avgtau, avgtau_err = fitKWW(df, title="KWW Fit")
    """
    norm = kwargs.get('norm', False)
    residual = kwargs.get('residual', None)
    markevery = kwargs.get('markevery', 1)
    savepath = kwargs.get('savepath', None)

    # Plot E vs. t


    df['lnmod'] = np.log(df['modulus'])
    df = df.dropna()

    # Define KWW model
    def KWW(t, G0, tau, beta):
        return residual + G0 * np.exp(-(t / tau) ** beta) if residual \
            else G0 * np.exp(-(t / tau) ** beta)

    # Curve fitting
    popt, pcov = curve_fit(
        KWW,
        df['time'],
        df['modulus'],
        p0=[1e7, 100, 0.95],
        bounds=((1e5, 1e-1, 0.5), (1e8, 1e5, 1)),
        maxfev=1000
    )

    G0, tau, beta = popt
    G0_err, tau_err, beta_err = np.sqrt(np.diag(pcov))

    # Ensemble average tau and uncertainty
    avgtau = (tau / beta) * gammaf(1 / beta)
    tauvar = tau_err ** 2 * ((1 / beta) * gammaf(1 / beta)) ** 2
    betavar = (beta_err ** 2 *
               (tau * gammaf(1 / beta) *
                (beta + digamma(1 / beta)) / beta ** 3) ** 2)
    avgtau_err = np.sqrt(tauvar + betavar)

    # Plot fit
    tfit = np.logspace(np.log10(min(df['time'])),
                       np.log10(max(df['time'])), 100)
    modfit = KWW(tfit, G0, tau, beta)
    
    if 'ax' in kwargs.keys():
        ax = kwargs.get('ax')
        if norm:
            ax.semilogx(df['time'], df['mod_norm'], '.', color=palette[0],
                        label='Expt.', rasterized=True)
        else:
            ax.loglog(df['time'], df['modulus'], '.', markevery=markevery,
                      color=palette[0], label='Expt.',rasterized=True)
        ax.loglog(
            tfit,
            modfit,
            '--',
            linewidth=2,
            color=palette[1],
            label=(f'$\\tau^*={tau:.1f} \\pm {tau_err:.1f}$ s\n'
                   f'$\\beta={beta:.2f} \\pm {beta_err:.2f}$\n'
                   f'$\\langle\\tau\\rangle={avgtau:.1f} \\pm {avgtau_err:.1f}$ s')
        )
    
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
    """
    Fit stress relaxation data to a fractional Maxwell model.

    This function reads a DataFrame of stress relaxation data and fits it to
    a fractional Maxwell model according to Jaishankar & McKinley (2013).
    It supports variants for gels and liquids by fixing one of the fractional
    exponents.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns:
        - 'time': Time values (s)
        - 'modulus': Relaxation modulus values (Pa)
    model : str, default None
        Use 'gel' for fractional Maxwell gel (perfect spring) or 'liquid'
        for fractional Maxwell liquid (perfect dashpot).
    aguess : float, default 0.9
        Initial guess for the higher power-law exponent.
    bguess : float, default 0.05
        Initial guess for the lower power-law exponent.
    Gguess : float, default 1e7
        Initial guess for the first quasi-property.
    Vguess : float, default 1e8
        Initial guess for the second quasi-property.
    residual : float, default None
        Residual modulus to include in the fit if expected or known.
    params_out : bool, default False
        If True, returns all fitted parameters and their uncertainties.
    ylims : list, default [1e4, 1e7]
        y-axis limits for the plot.
    title : str, default None
        Optional title for the plot.
    tts : bool, default False
        If True, x-axis is aT*Time; otherwise, x-axis is Time.
    savepath : str, default None
        Path to save the plot. If None, the plot is not saved.

    Returns
    -------
    tuple
        tau : float
            Fractional generalization of relaxation time (s).
        tauerr : float
            Uncertainty in tau.
        If params_out=True, also returns:
            G, Gerr, a, aerr, V, Verr, b, berr

    Notes
    -----
    - Uses Mittag-Leffler function for fractional relaxation.
    - Requires that 0 <= b < a <= 1.

    Example
    -------
    >>> tau, tauerr = fitFracMaxwell(df, model='gel', title='FM Fit')
    """
    model = kwargs.get('model', None)
    aguess = kwargs.get('aguess', 0.9)
    bguess = kwargs.get('bguess', 0.05)
    Gguess = kwargs.get('Gguess', 1e7)
    Vguess = kwargs.get('Vguess', 1e8)
    residual = kwargs.get('residual', None)
    params_out = kwargs.get('params_out', False)
    ylims = kwargs.get('ylims', [1e4, 1e7])
    title = kwargs.get('title', None)
    tts = kwargs.get('tts', False)
    markevery = kwargs.get('markevery', 1)
    label = kwargs.get('label', 'Expt.')
    savepath = kwargs.get('savepath', None)

    # Ensure 0 <= b < a <= 1
    if bguess > aguess:
        bguess, aguess = aguess, bguess
    if aguess == bguess:
        print('We need a > b.')
    if aguess < 0 or bguess < 0 or aguess == 0:
        print('The exponents must be between 0 and 1.')

    # Define fractional Maxwell model
    def fracMaxwell(time, G, a, V, b):
        prefactor = [G * t ** (-b) for t in time]
        MLtime = [(-G / V) * t ** (a - b) for t in time]
        MLa = a - b
        MLb = 1 - b
        mleval = [mittag_leffler(t, MLa, MLb) for t in MLtime]
        if residual:
            return [residual + np.real(p * m) for p, m in zip(prefactor, mleval)]
        return [np.real(p * m) for p, m in zip(prefactor, mleval)]

    # Clean up DataFrame
    df.reset_index()
    df = df.dropna()

    # Plot experimental data
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    ax.set_xlabel('Time / $a_T$ (s)' if tts else 'Time (s)')
    ax.set_ylabel('G(t) / $b_T$ (Pa)' if tts else 'Relaxation Modulus G(t)')
    ax.loglog(df['time'], df['modulus'], '.', color=palette[0],
              markevery=markevery, label=label)

    # Fit based on model type
    if model == 'gel':
        b = 0
        berr = 0
        popt, pcov = curve_fit(
            lambda time, G, a, V: fracMaxwell(time, G, a, V, b),
            df['time'], df['modulus'],
            p0=[Gguess, aguess, Vguess],
            bounds=([1e4, 0.3, 1e4], [1e9, 1, 1e11])
        )
        G, a, V = popt
        Gerr, aerr, Verr = np.sqrt(np.diag(pcov))

    elif model == 'liquid':
        a = 1
        aerr = 0
        popt, pcov = curve_fit(
            lambda time, G, V, b: fracMaxwell(time, G, a, V, b),
            df['time'], df['modulus'],
            p0=[Gguess, Vguess, bguess],
            bounds=([1e4, 1e4, 0], [1e9, 1e11, 0.5])
        )
        G, V, b = popt
        Gerr, Verr, berr = np.sqrt(np.diag(pcov))

    else:
        popt, pcov = curve_fit(
            fracMaxwell,
            df['time'], df['modulus'],
            p0=[Gguess, aguess, Vguess, bguess],
            bounds=([1e4, 0.3, 1e4, 0], [1e9, 1, 1e11, 0.2])
        )
        G, a, V, b = popt
        Gerr, aerr, Verr, berr = np.sqrt(np.diag(pcov))

    # Compute tau and its uncertainty
    tau = (V / G) ** (1 / (a - b))
    dtaudV = tau / (V * (a - b))
    dtaudG = tau / (G * (a - b))
    dtauda = tau * np.log(V / G) / (a - b) ** 2
    dtaudb = tau * np.log(V / G) / (a - b) ** 2
    tauerr = np.sqrt((Gerr * dtaudG) ** 2 + (Verr * dtaudV) ** 2 +
                     (aerr * dtauda) ** 2 + (berr * dtaudb) ** 2)

    # Prepare fit label
    if model == 'gel':
        fitlabel = (f'τ={tau:.1f} ± {tauerr:.1f} s\n'
                    f'α={a:.2f} ± {aerr:.2f}\nβ=0')
    elif model == 'liquid':
        fitlabel = (f'τ={tau:.1f} ± {tauerr:.1f} s\nα=0\n'
                    f'β={b:.2f} ± {berr:.2f}')
    else:
        fitlabel = (f'τ={tau:.1f} ± {tauerr:.1f} s\n'
                    f'α={a:.2f} ± {aerr:.2f}\n'
                    f'β={b:.2f} ± {berr:.2f}')

    # Plot fit
    tfit = np.logspace(np.log10(min(df['time'])),
                       np.log10(max(df['time'])), 100)
    modfit = fracMaxwell(tfit, G, a, V, b)
    ax.loglog(tfit, modfit, '--', linewidth=2, color=palette[1],
              markevery=10, label=fitlabel)

    # Finalize plot
    ax.legend()
    ax.set_title(title)
    ax.set_ylim(ylims)
    plt.show()

    if savepath:
        plt.savefig(savepath)

    if params_out:
        return tau, tauerr, G, Gerr, a, aerr, V, Verr, b, berr
    return tau, tauerr



def findTg(temp, data):
    """
    Return the glass transition temperature (Tg) using the maximum of
    tan(delta) or loss modulus.

    Parameters
    ----------
    temp : array-like
        Temperature values.
    data : array-like
        Data values (e.g., tan(delta) or loss modulus).

    Returns
    -------
    None
        Prints the estimated Tg.
    """
    maxi = np.argmax(data)

    fitrange_temp = temp[maxi - 5:maxi + 5]
    fitrange_data = data[maxi - 5:maxi + 5]

    pfit = np.polyfit(fitrange_temp, fitrange_data, 2)

    tempfit = np.linspace(temp[maxi - 5], temp[maxi + 5], 100)
    datafit = [pfit[0] * x ** 2 + pfit[1] * x + pfit[2] for x in tempfit]
    fitmaxi = np.argmax(datafit)

    Tg = round(tempfit[fitmaxi], 1)
    return print(Tg)



def findTandMax(df):
    """
    Return temperatures of maximum tan(delta) for frequency sweep data.

    Parameters
    ----------
    df : dict
        Dictionary of DataFrames for different frequencies.

    Returns
    -------
    list
        List of inverse temperatures (1/T) for each frequency.
    """
    invT = []
    for i in np.arange(0, len(df.keys()), 1):
        try:
            maxi = np.argmax(df[list(df.keys())[i]]['Tan(delta)'])

            fitrange_temp = df[list(df.keys())[i]]['Temperature'][maxi - 2:maxi + 2]
            fitrange_tand = df[list(df.keys())[i]]['Tan(delta)'][maxi - 2:maxi + 2]

            pfit = np.polyfit(fitrange_temp, fitrange_tand, 2)

            tempfit = np.linspace(
                df[list(df.keys())[i]]['Temperature'][maxi - 2],
                df[list(df.keys())[i]]['Temperature'][maxi + 2],
                50
            )
            tandfit = [pfit[0] * x ** 2 + pfit[1] * x + pfit[2] for x in tempfit]
            fitmaxi = np.argmax(tandfit)
            t = tempfit[fitmaxi] + 273

            invT.append(1 / t)
        except (KeyError, TypeError):
            invT.append(np.nan)

    return invT



def freqEa(path, **kwargs):
    """
    Fit temperature and frequency data to measure activation energy (Ea)
    of tan(delta) peak.

    Parameters
    ----------
    path : str
        Path to DMA data file.
    **kwargs : dict, optional
        freq_num : int, default 31
            Number of frequencies.
        title : str, default ' '
            Plot title.

    Returns
    -------
    None
        Displays plot and prints Ea.
    """
    freq_num = kwargs.get('freq_num', 31)
    title = kwargs.get('title', ' ')

    A = readDMA(path, keys='freq', freq_num=freq_num)
    B = findTandMax(A)

    freqs = np.linspace(-2, 1, freq_num).tolist()

    for j in reversed(np.arange(0, len(B))):
        if float('-inf') < B[j] < float('inf'):
            continue
        else:
            B.pop(j)
            freqs.pop(j)

    linfit = np.polyfit(B, freqs, 1)
    fitx = [(y - linfit[1]) / linfit[0] for y in freqs]
    slope = linfit[0]
    Ea = round(-1 * slope * 8.314e-3 * np.log(10), 0)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    ax.plot([1000 * b for b in B], freqs, 'o', color=palette[0], 
            label='Expt. Data')
    ax.plot([1000 * x for x in fitx], freqs, '--', color=palette[1],
            label=f'$E_a$={Ea:.0f} kJ/mol')
    ax.set_ylabel('$log_{10}$ f')
    ax.set_xlabel('Inverse Temperature (1000$K^{-1}$)')
    ax.legend()
    ax.set_title(title)


def findEpr(temp, stor):
    """
    Return the rubbery storage modulus minimum and corresponding temperature.

    Parameters
    ----------
    temp : array-like
        Temperature values.
    stor : array-like
        Storage modulus values.

    Returns
    -------
    tuple
        Epr : float
            Minimum storage modulus.
        T : float
            Temperature at minimum storage modulus.
    """
    mini = np.argmin(stor)
    T = temp[mini]
    Epr = np.min(stor)
    return Epr, T



def compStor(path, f1, f2, f3, f4):
    """
    Compare storage modulus for multiple datasets and plot results.

    Parameters
    ----------
    path : str
        Base path to data files.
    f1, f2, f3, f4 : str
        Filenames for datasets.

    Returns
    -------
    None
        Displays a comparison plot.
    """
    A = readDMA(path, f1)
    B = readDMA(path, f2)
    C = readDMA(path, f3)
    D = readDMA(path, f4)

    temp1, stor1 = A[0], A[1]
    temp2, stor2 = B[0], B[1]
    temp3, stor3 = C[0], C[1]
    temp4, stor4 = D[0], D[1]

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    p1, = ax.semilogy(temp1, stor1, '.-', ms=10, lw=3, color=palette[0],
                      label='DGEBA/MDA')
    p2, = ax.semilogy(temp2, stor2, '.-', ms=10, lw=3, color=palette[1],
                      label='DGEBA/DTDA')
    p3, = ax.semilogy(temp3, stor3, '.-', ms=10, lw=3, color=palette[3],
                      label='BGPDS/MDA')
    p4, = ax.semilogy(temp4, stor4, '.-', ms=10, lw=3, color=palette[4],
                      label='BGPDS/DTDA')

    ax.set_ylim(1e6, 1e10)
    ax.set_xlim(-125, 225)
    ax.set_xlabel('Temperature ($^oC$)')
    ax.set_ylabel("Storage Modulus, E' (Pa)")
    ax.set_title('FDS Space Comparison')
    ax.legend(handles=[p1, p2, p3, p4])
    return plt.show()


def compLoss(path_base, *f, **kwargs):
    """
    Compare loss modulus for multiple datasets and plot results.

    Parameters
    ----------
    path_base : Path
        Base path to data files.
    *f : str
        Filenames for datasets.
    **kwargs : dict, optional
        temp_min : float, default -125
        temp_max : float, default 150
        mod_min : float, default 1e7
        mod_max : float, default 5e8
        legendsize : int, default 10

    Returns
    -------
    None
        Displays a comparison plot.
    """
    temp_min = kwargs.get('temp_min', -125)
    temp_max = kwargs.get('temp_max', 150)
    mod_min = kwargs.get('mod_min', 1e7)
    mod_max = kwargs.get('mod_max', 5e8)
    legendsize = kwargs.get('legendsize', 10)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    for i in np.arange(0, len(f)):
        A = readDMA(path_base.joinpath(f[i]))
        temp, loss = A[0], A[2]
        ax.semilogy(temp, loss, '-', ms=10, lw=3, color=palette[i],
                    label=str(f[i]).rstrip('.txt'))

    ax.set_ylim(mod_min, mod_max)
    ax.set_xlim(temp_min, temp_max)
    ax.set_xlabel('Temperature ($^oC$)')
    ax.set_ylabel('Loss Modulus, E" (Pa)')
    ax.set_title('E" Comparison')
    ax.legend(prop={'size': legendsize})
    return plt.show()


def compTand(path, *f):
    """
    Compare tan(delta) for multiple datasets and plot results.

    Parameters
    ----------
    path : str
        Base path to data files.
    *f : str
        Filenames for datasets.

    Returns
    -------
    None
        Displays a comparison plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    for i in np.arange(0, len(f)):
        A = readDMA(path, f[i])
        temp, tand = A[0], A[3]
        ax.semilogy(temp, tand, '.-', ms=10, lw=3, color=palette[i],
                    label=str(f[i]).rstrip('-1.txt'))

    ax.set_ylim(0.01, 2)
    ax.set_xlim(-125, 225)
    ax.set_xlabel('Temperature ($^oC$)')
    ax.set_ylabel('tan$\\delta$')
    ax.set_title('FDS tan$\\delta$ Comparison')
    ax.legend()
    return plt.show()



def fitTwoGaussian(path, **kwargs):
    """
    Fit sub-Tg tan delta or loss modulus data to two Gaussian peaks.

    Parameters
    ----------
    path : str
        Path to DMA data file.
    **kwargs : dict, optional
        betaprime_ctr : float, default 25
            Center for beta-prime peak.
        var : str, default 'tand'
            Variable to fit ('tand' or 'loss').

    Returns
    -------
    popt : ndarray
        Optimal parameters for the two Gaussian peaks.
    """
    betaprime_ctr = kwargs.get('betaprime_ctr', 25)
    var = kwargs.get('var', 'tand')

    if var == 'tand':
        df = readDMA(path)
        i_max = np.argmin(df['tand'][75:108]) + 75
        df = df.iloc[0:i_max]
        df['phi'] = np.rad2deg(np.arctan(df['tand']))

        baseline = [
            (df['tand'].iloc[0] +
             ((df['tand'].iloc[-1] - df['tand'].iloc[0]) /
              (df['temp'].iloc[-1] - df['temp'].iloc[0])) *
             (i - df['temp'].iloc[0]))
            for i in df['temp']
        ]
        tand_base = np.subtract(np.array(df['tand']), np.array(baseline))

        def twoGaussian(x, *params):
            ctr1, amp1, wid1, ctr2, amp2, wid2 = params
            return (amp1 * np.exp(-((x - ctr1) / wid1) ** 2) +
                    amp2 * np.exp(-((x - ctr2) / wid2) ** 2))

        guess = [-60, 5e-2, 10, betaprime_ctr, 5e-2, 10]
        popt, _ = curve_fit(twoGaussian, df['temp'], tand_base, p0=guess)
        return popt

    elif var == 'loss':
        A = readDMA(path)
        i_max = np.argmin(A[3][75:108]) + 75
        temp = [T for T in A[0]][0:i_max]
        loss = np.array(A[2][0:i_max])

        baseline = [
            (loss[0] + ((loss[-1] - loss[0]) / (temp[-1] - temp[0])) *
             (i - temp[0]))
            for i in temp
        ]
        loss_base = np.subtract(np.array(loss), np.array(baseline))

        def twoGaussian(x, *params):
            ctr1, amp1, wid1, ctr2, amp2, wid2 = params
            return (amp1 * np.exp(-((x - ctr1) / wid1) ** 2) +
                    amp2 * np.exp(-((x - ctr2) / wid2) ** 2))

        guess = [-60, 1e8, 10, betaprime_ctr, 1e8, 10]
        popt, _ = curve_fit(
            twoGaussian,
            temp,
            loss_base,
            p0=guess,
            bounds=((-90, 1e6, 5, -10, 1e6, 5), (-30, 1e9, 50, 80, 1e9, 50))
        )
        return popt

    
def fitGaussian(path, **kwargs):
    """
    Fit sub-Tg tan delta data to a single Gaussian peak.

    Parameters
    ----------
    path : str
        Path to DMA data file.
    **kwargs : dict, optional
        i_max : int, default 80
            Maximum index for data slice.

    Returns
    -------
    popt : ndarray
        Optimal parameters for the Gaussian peak.
    """
    i_max = kwargs.get('i_max', 80)

    A = readDMA(path)
    temp = [T + 273 for T in A[0]][0:i_max]
    tand = np.array(A[3][0:i_max])

    baseline = [
        (tand[0] + ((tand[-1] - tand[0]) / (temp[-1] - temp[0])) *
         (i - temp[0]))
        for i in temp
    ]
    tand_base = np.subtract(np.array(tand), np.array(baseline))

    def Gaussian(x, *params):
        ctr, amp, wid = params
        return amp * np.exp(-((x - ctr) / wid) ** 2)

    guess = [210, 5e-2, 10]
    popt, _ = curve_fit(Gaussian, temp, tand_base, p0=guess)
    return popt


def frac_lin_solid(f, Er, Eg, lambda_t, lambda_g, tau_g):
    """
    Compute the complex modulus for a fractional linear solid model.

    The fractional linear solid model is a viscoelastic model that uses
    fractional calculus to describe material behavior over a wide range
    of frequencies. This function calculates the complex modulus as a
    function of frequency and model parameters.

    Parameters
    ----------
    f : float or array-like
        Frequency (Hz) at which to compute the complex modulus.
    Er : float
        Relaxed modulus (low-frequency limit).
    Eg : float
        Glassy modulus (high-frequency limit).
    lambda_t : float
        Fractional exponent for the terminal relaxation process.
    lambda_g : float
        Fractional exponent for the glassy relaxation process.
    tau_g : float
        Characteristic relaxation time for the glassy process.

    Returns
    -------
    complex or ndarray
        Complex modulus at the given frequency or frequencies.

    Notes
    -----
    The equation used is:
        E*(f) = Er + (Eg - Er) / [(i * f * tau_g)^(-lambda_t) +
                                  (i * f * tau_g)^(-lambda_g)]

    Example
    -------
    >>> frac_lin_solid(f=1.0, Er=1e6, Eg=1e9,
    ...                lambda_t=0.5, lambda_g=0.1, tau_g=1e-3)
    (complex modulus value)
    """
    return Er + (Eg - Er) / (
        (1j * f * tau_g) ** (-lambda_t) + (1j * f * tau_g) ** (-lambda_g)
    )



def round_n(numbers, n_figures):
    """
    Rounds a single number or a NumPy array of numbers to n
    significant figures.

    Args:
        numbers (float|int|np.ndarray): The input number(s) to round.
        n_figures (int): The number of significant figures required.

    Returns:
        float|int|np.ndarray: The rounded number(s) as the original type
                              or np.ndarray.
    """
    is_scalar_input = np.isscalar(numbers)
    # Use numpy.array to handle single inputs the same way as arrays
    numbers_arr = np.array(numbers, dtype=float)

    # Handle the edge case where the input number(s) might be 0
    if np.all(numbers_arr == 0):
        if is_scalar_input:
            return 0
        else:
            # Return as array of zeros
            return np.zeros_like(numbers_arr, dtype=float)

    # Calculate the magnitude of the number(s) using log10
    magnitude = np.floor(np.log10(np.abs(numbers_arr)))[0]

    # Calculate the number of decimal places needed for standard rounding
    # The negative sign adjusts the magnitude for the standard round()
    # function's 'decimals' parameter
    decimal_places = n_figures - 1 - magnitude

    # Apply standard rounding element-wise
    # np.around takes 'decimals' as the second argument
    # Ensure decimals array is treated correctly as integer types for precision input
    rounded_numbers = np.around(
        numbers_arr, decimals=decimal_places.astype(int)
    )

    # If the input was a single scalar, return a single scalar result
    if is_scalar_input:
        # Use .item() to extract the scalar value from the 0D numpy array
        return rounded_numbers.item()
    else:
        return rounded_numbers







#This if statement should be included at the end of any module
#If you don't include this, when you import the module, it will run
#the module script from top to bottom instead of only importing the
#functions	
if __name__ == "__main__":
 	pass