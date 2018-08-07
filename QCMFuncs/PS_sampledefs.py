def sample_dict():
    sample = {}  # individual sample dictionaries get added to this

    samplename = 'PS_3k_cool'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'data_PS/',
        'barefile': 'Urma_bare_cool',
        'filmfile': 'Wilfred_3k_cool',
        'baretrange': [0, 900],  # default is [0, 0]
        'filmtrange': [0, 900],  # default is [0, 0]
        'Temp': [120, 110, 100, 90, 80, 70, 60, 40],  # default is [22]
        'nhplot': [3, 5],  # default is [1, 3, 5]
        'nhcalc': ['355'],  # default is ['353']
        }

    samplename = 'PS_30k_cool'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'data_PS/PS_3k/',
        'barefile': 'Urma_bare_cool',
        'filmfile': 'Wilfred_3k_cool',
        'baretrange': [0, 900],  # default is [0, 0]
        'filmtrange': [0, 900],  # default is [0, 0]
        'Temp': [120, 110, 100, 90, 80, 70, 60, 40],  # default is [22]
        'nhplot': [3, 5],  # default is [1, 3, 5]
        'nhcalc': ['355'],  # default is ['353']
        }

    samplename = 'PS_192k_cool'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'data_PS/PS_3k/',
        'barefile': 'Urma_bare_cool',
        'filmfile': 'Wilfred_3k_cool',
        'baretrange': [0, 900],  # default is [0, 0]
        'filmtrange': [0, 900],  # default is [0, 0]
        'Temp': [120, 110, 100, 90, 80, 70, 60, 40],  # default is [22]
        'nhplot': [3, 5],  # default is [1, 3, 5]
        'nhcalc': ['355'],  # default is ['353']
        }

    return sample
