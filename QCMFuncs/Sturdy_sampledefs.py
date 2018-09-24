# note that the values of fref used for the springpot models are for the
# reference temperature from the original DMA data, which is a bit
# arbitrary

def sample_dict():
    sample = {}  # individual sample dictionaries get added to this

    samplename = 'w00_1'
    sample[samplename] = {
        'samplename': samplename,
        'barefile': 'none',
        'filmfile': 'LFS_GZO0_01_data',
        'nhplot': [1, 3, 5],
        'nhcalc' : ['353']
        }

    samplename = 'w00_ref'
    sample[samplename] = {
        'samplename': samplename,
        'barefile': 'none',
        'filmfile': 'LFS_G_08_data',
        'nhplot': [1, 3, 5],
        'nhcalc' : ['353']
        }
    
    
    return sample
    

#w05_1 = './data/QCM/LFS_GZO5_01_data'
#w10_1 = './data/QCM/LFS_GZO10_01_data'
#w10_2 = './data/QCM/LFS_GZO10_02_data'
#w20_1 = './data/QCM/LFS_GZO20_01_data'
#w20_2 = './data/QCM/LFS_GZO20_02_data'
#w20_3 = './data/QCM/LFS_GZO20_03_data'
#w40_1 = './data/QCM/LFS_GZO40_01_data'
#w50_1 = './data/QCM/LFS_GZO50_01_data'  # no calc file for this one - no actual solutions
#w50_2 = './data/QCM/LFS_GZO50_02_data'
#w50_3 = './data/QCM/LFS_GZO50_03_data'