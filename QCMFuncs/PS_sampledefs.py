def sample_dict():
    sample = {}  # individual sample dictionaries get added to this

    samplename = 'PS_3k_cool'
    sample[samplename] = {
        'samplename': samplename,
        'filetype': 'mat',                            # file type of data file(s) 'mat' or 'h5'. default is 'mat'.
        # 'filmchn': 'samp',                            # For 'h5' only. define the channel where film data is stored. defaule is 'samp'
        'datadir': '',                                # the relative path of file in 'dataroot'. Note: don't include '.' at the beginning of the relative path
        'barefile': 'Urma_bare_cool',                 # For 'mat' only. bare file name. 'h5' file will use the bare defined in its file
        'filmfile': 'Wilfred_3k_cool',                # file name where the film data is stored
        'baretrange': [0, 900],                       # index range to be used for bare. default is [0, 0]
        'filmtrange': [0, 900],                       # index range to be used for film. default is [0, 0]
        # 'filmindex': range(0, 156, 10),               # array form (range, list, tuple) of index of points to be calculated and plotted. 
        # 'xscale': 'linear',                           # x-axis (time or temperature) scale. 'linear' or 'log'. default is 'linear'
        # 'freqref':                                  # leave this commented
        'Temp': [120, 110, 100, 90, 80, 70, 60, 40],  # For 'mat' only. list of temperature steps. default is [22]
        # 'xtype': 't',                                 # For 'h5' only. data used for x-axis. 't' or 'temp'. default is 't'
        'nhplot': [3, 5],                             # list of harmonics to be plotted. default is [1, 3, 5]
        'nhcalc': ['355'],                            # list of harmonic combinations to be calculated. default is ['353']
        }

    samplename = 'PS_30k'
    sample[samplename] = {
        'samplename': samplename,
        'barefile': 'Bare_30k_5',
        'filmfile': 'PS_30k_5',
        'baretrange': [0, 900],
        'filmtrange': [0, 1000],
        'Temp': [30, 60, 80, 90, 100, 110, 120, 130, 140],  
        'nhplot': [1, 3, 5],
        'nhcalc': ['355']
        }

    samplename = 'PS_192k'
    sample[samplename] = {
        'samplename': samplename,
        'barefile': 'Bare_192k_1',
        'filmfile': 'PS_192k_4',
        'baretrange': [0, 680],
        'filmtrange': [0, 1000],
        'Temp': [30, 60, 80, 90, 100, 110, 120, 130, 140, 150],  
        'nhplot': [3, 5],
        'nhcalc': ['355']
        }

    return sample
