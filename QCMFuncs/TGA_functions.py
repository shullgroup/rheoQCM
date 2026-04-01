#Import any individual functions from outside packages 
#that are used in your functions.
#These are called dependencies.
# updated version of this file is maintained at
# https://github.com/shullgroup/rheoQCM/blob/master/QCMFuncs/DSC_functions.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def readTGA(path):
    '''
    Read txt file from TGA experiment and convert to a DataFrame
    
    Parameters
    ----------
    path : Path
        Path object to the .txt file containing the TGA data
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing relevant experimental data, i.e.
        temperature, weight, weight derivative, etc.
        
    '''

    
    data_rows = []
    start_row_index = None

    with open(path, 'r') as f:
        lines = f.readlines()

    # Find the first numeric row to determine where data starts
    for i, line in enumerate(lines):
        # We use split() just for the "check" to see if the line contains numbers
        test_cells = line.split()
        if test_cells and is_numeric(test_cells[0]):
            start_row_index = i
            break

    if start_row_index is None:
        raise ValueError("Could not find numeric data in the file.")

    # Process data from the start_row onwards
    for line in lines[start_row_index:]:
        if not line.strip():
            continue
            
        # Instead of splitting by spaces, we "slice" the string by character position.
        # each column is 14 wide
        row_cells = []
        for i in range(0, len(line), 14):
            chunk = line[i : i + 14].strip()
            if chunk:
                row_cells.append(chunk)
        
        # Convert strings to floats
        if row_cells and is_numeric(row_cells[0]):
            numeric_row = [float(x) for x in row_cells]
            data_rows.append(numeric_row)

    # 3. Create DataFrame and clean up
    df = pd.DataFrame(data_rows)
    
    df = df[[1, 2, 4]]
    df.columns = ['temp', 'time', 'wt']

    # convert to wt%
    df['wt_pct'] = df['wt']/df['wt'].iloc[0]*100


    return df

def plotTGA(df, **kwargs):
    '''
    Plot the data from a TGA experiment.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing TGA data processed from readTGA method

    Td1 : bool, default True
        Flag to plot and annotate the temperature at 1% mass degradation

    Td5 : bool, default False
        Flag to plot and annotate the temperature at 5% mass degradation

    deriv : bool, default False
        Flag to plot the derivative of wt% w.r.t. temperature

    title : str, default None
        Optional title for the plot

    legendloc : int or str, default 0
        Location for legend.  Default is 0/'best'

    savepath : Path, default None
        Path to save of the plot
    
    Returns
    -------
    variable : type
        description
    
    Notes
    -----
    - Will add return of degradation temperatures
    
    '''
    
    Td1 = kwargs.get('Td1', True)
    Td5 = kwargs.get('Td5', False)
    deriv = kwargs.get('deriv', False)
    ax = kwargs.get('ax', None)
    label = kwargs.get('label', None)
    title = kwargs.get('title', None)
    legendloc = kwargs.get('legendloc', 0)
    savepath = kwargs.get('savepath', None)

    #typical TGA plot, annotate Td5%, option for Td1%
    if ax:
        pass
    else:
        fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
        ax.set_xlabel('Temperature ($^\\circ$C)')
        ax.set_ylabel('Weight %')
        ax.set_ylim([0,105])
        ax.set_yticks([0,20,40,60,80,100])
        ax.set_title(title)

    p1 = ax.plot(df['temp'], df['wt_pct'], '-', label=f'{label} wt%')
    

    if Td5:
        df_Td5 = df.query('wt_pct > 94.95 and wt_pct < 95.05')
        Td5 = np.mean(df_Td5['temp'])
        Td5_coords = kwargs.get('Td5_coords', (Td5+20, 95))
        ax.plot(Td5, [95], 'o', color = '#000000')
        ax.annotate(f'T$_{{d5}}$ = {round(Td5)} $^{{\\circ}}$C', Td5_coords)
    
    if Td1:
        
        df_Td1 = df.query('wt_pct > 98.95 and wt_pct < 99.05')
        Td1 = np.mean(df_Td1['temp'])
        Td1_coords = kwargs.get('Td1_coords', (Td1+30, 98))
        ax.plot(Td1, [99], 'o', color = '#000000')
        ax.annotate(f'T$_{{d1}}$ = {round(Td1)} $^{{\\circ}}$C', Td1_coords)

    if deriv:
        twin = plt.twinx()
        p2 = twin.plot(df.query('temp > 35')['temp'], 
                       -df.query('temp > 35')['dwt_pct'],
                        '-', label=f'{label} dwt%/dT')
        twin.set_ylabel('Derivative Wt Loss %')

        ps = p1 + p2
        labels = [l.get_label() for l in ps]
        ax.legend(ps, labels, loc=legendloc)

    if savepath:
        plt.savefig(savepath)

    if Td5 and not Td1:
        return Td5
    
    if Td1 and not Td5:
        return Td1
    
    if Td1 and Td5:
        return Td1, Td5

