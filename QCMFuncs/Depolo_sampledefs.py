# note that the values of fref used for the springpot models are for the
# reference temperature from the original DMA data, which is a bit
# arbitrary

def sample_dict():
    sample = {}  # individual sample dictionaries get added to this

    # Linseed oil on gold electrode - PEP with THF
    samplename = 'linseed_bulk'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': '',
        'barefile': 'bare',
        'filmfile': 'linseed_film',
        
        'nhplot': [3, 5],
        'nhcalc' : ['353']
        
        }
    
    # linseed oil on chrome electrode - PEP with THF
    samplename = 'linseed_chrome'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': '',
        'barefile': 'bare_chrome',
        'filmfile': 'linseed_film_chrome',
        
        'nhplot': [3, 5],
        'nhcalc': ['353', '355']
        
        }
    
    # Linseed oil with ultrathin PS layer
    samplename = 'linseed_PS'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': '',
        'barefile': 'bare_withPS',
        'filmfile': 'film_withPS',
        
        'nhplot': [3, 5],
        'nhcalc': ['353', '355']
        
        }
    return sample