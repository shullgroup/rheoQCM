def sample_dict():
    sample = {}  # individual sample dictionaries get added to this

    samplename = 'PS_3k_cool'
    sample[samplename] = {
        'samplename': samplename,
        'barefile': 'Urma_bare_cool',
        'filmfile': 'Wilfred_3k_cool',
        'baretrange': [0, 900],  # default is [0, 0]
        'filmtrange': [0, 900],  # default is [0, 0]
        'Temp': [120, 110, 100, 90, 80, 70, 60, 40],  # default is [22]
        'nhplot': [3, 5],  # default is [1, 3, 5]
        'nhcalc': ['355'],  # default is ['353']
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
