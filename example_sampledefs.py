# note that the values of fref used for the springpot models are for the
# reference temperature from the original DMA data, which is a bit
# arbitrary


def sample_dict():
    sample = {}  # individual sample dictionaries get added to this

    samplename = 'PS_3k_cool'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'example_data/PS_3k/',
        'barefile': 'Urma_bare_cool',
        'filmfile': 'Wilfred_3k_cool',
        'baretrange': [0, 900],  # default is [0, 0]
        'filmtrange': [0, 900],  # default is [0, 0]
        'Temp': [120, 110, 100, 90, 80, 70, 60, 40],  # default is [22]
        'nhplot': [3, 5],  # default is [1, 3, 5]
        'nhcalc': ['355'],  # default is ['353']
        }

    #  PMMA sample - Meredith
    samplename = 'PMMA_75k_T01'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'example_data/PMMA/',
        'barefile': 'Angela_bare',
        'filmfile': 'Angela_PMMA_75k_film2',
        'Temp': [120, 110, 100, 90, 80, 70, 60, 50, 40],
        'nhcalc': ['355'],
        'nhplot': [3, 5]
        }

    #  PMMA sample - Tom sample 4
    samplename = 'PMMA_75k_S04'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'example_data/PMMA/',
        'barefile': 'QCMS04_bare',
        'filmfile': 'QCMS04_75kPMMA',
        'nhplot': [3, 5]
        }

    #  PMMA sample - Tom sample 5
    samplename = 'PMMA_75k_S05'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'example_data/PMMA/',
        'barefile': 'QCMS05_bare',
        'filmfile': 'QCMS05_75kPMMA',
        'nhplot': [3, 5]
        }

    return sample
