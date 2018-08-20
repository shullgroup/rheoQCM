
# note that the values of fref used for the springpot models are for the
# reference temperature from the original DMA data, which is a bit
# arbitrary

def sample_dict():
    sample = {}  # individual sample dictionaries get added to this

    #  PMMA sample - Tom sample 4
    samplename = 'PMMA_75k_S04'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'PMMA/',
        'barefile': 'QCMS04_bare',
        'filmfile': 'QCMS04_75kPMMA',
        
        'nhplot': [3, 5],
        'nhcalc' : ['355']
        
        }
#  PMMA sample - Tom sample 4 post heat treatment
    samplename = 'PMMA_75k_S04_postanneal'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'PMMA/',
        'barefile': 'QCMS04_bare',
        'filmfile': 'QCMS04_75kPMMA_postanneal',
        'nhplot': [3, 5],
        'nhcalc' : ['355']
        
        }
    #  PMMA sample - Tom sample 4 post TiO2 coating and calcination
    samplename = 'PMMA_75k_S04_TiO2coating'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'PMMA/',
        'barefile': 'QCMS04_bare',
        'filmfile': 'QCMS04_75kPMMA_TiO2coating',
        'nhplot': [3, 5],
        'nhcalc' : ['355']
        
        }
    #  PMMA sample - Tom sample 4 post 2000 J/cm^2 exposure
    samplename = 'PMMA_75k_S04_TiO2_2000I'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'PMMA/',
        'barefile': 'QCMS04_bare',
        'filmfile': 'QCMS04_75kPMMA_TiO2_2000I',
        'nhplot': [3, 5],
        'nhcalc' : ['355']
        
        }
     #  PMMA sample - Tom sample 4 post 4000 J/cm^2 exposure
    samplename = 'PMMA_75k_S04_TiO2_4000I'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'PMMA/',
        'barefile': 'QCMS04_bare',
        'filmfile': 'QCMS04_75kPMMA_TiO2_4000I',
        'nhplot': [3, 5],
        'nhcalc' : ['355']
        
        }
         #  PMMA sample - Tom sample 4 post 4000 J/cm^2 exposure
    samplename = 'PMMA_75k_S04_TiO2_6000I'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS04/',
        'barefile': 'QCMS04_bare',
        'filmfile': 'QCMS04_75kPMMA_TiO2_6000I',
        'nhplot': [3, 5],
        'nhcalc' : ['355']
        
        }
           #  PMMA sample - Tom sample 4 post 4000 J/cm^2 exposure
    samplename = 'PMMA_75k_S04_TiO2_36min'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS04/',
        'barefile': 'QCMS04_bare',
        'filmfile': 'QCMS04_75kPMMA_TiO2_36min',
        'nhplot': [3, 5],
        'nhcalc' : ['355']
        
        }
    #  PMMA sample - Tom sample 4 post 46 min exposure
    samplename = 'PMMA_75k_S04_TiO2_46min'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS04/',
        'barefile': 'QCMS04_bare',
        'filmfile': 'QCMS04_75kPMMA_TiO2_46min',
        'nhplot': [3, 5],
        'nhcalc' : ['355']
        
        }
        #  PMMA sample - Tom sample 4 post 46 min exposure
    samplename = 'PMMA_75k_S04_TiO2_56min'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS04/',
        'barefile': 'QCMS04_bare',
        'filmfile': 'QCMS04_75kPMMA_TiO2_56min',
        'nhplot': [3, 5],
        'nhcalc' : ['355']
        
        }    #  PMMA sample - Tom sample 4 post 46 min exposure
    samplename = 'PMMA_75k_S04_TiO2_66min'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS04/',
        'barefile': 'QCMS04_bare',
        'filmfile': 'QCMS04_75kPMMA_TiO2_66min',
        'nhplot': [3, 5],
        'nhcalc' : ['355']
        
        }
    #  PMMA sample - Tom sample 5
    samplename = 'PMMA_75k_S05'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS05/',
        'barefile': 'QCMS05_bare',
        'filmfile': 'QCMS05_75kPMMA',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }
        #  PMMA sample - Tom sample 6
    samplename = 'PMMA_75k_S06'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS06/',
        'barefile': 'QCMS06_bare_2',
        'filmfile': 'QCMS06_75kPMMA',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }
        #  PMMA sample - Tom sample 6 testing crystal
    samplename = 'PMMA_75k_S06_test'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS06/',
        'barefile': 'QCMS06_bare',
        'filmfile': 'QCMS06_bare_2',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }
 
            #  PMMA sample - Tom sample 6 testing crystal
    samplename = 'PMMA_75k_S08'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS08/',
        'barefile': 'QCMS08_bare',
        'filmfile': 'QCMS08_75kPMMA',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }
         #  PMMA sample - Tom sample 6 testing crystal
    samplename = 'PMMA_75k_S08_TiO2'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS08/',
        'barefile': 'QCMS08_bare',
        'filmfile': 'QCMS08_75kPMMA_TiO2',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }
          #  PMMA sample - Tom sample 6 testing crystal
    samplename = 'PMMA_75k_S08_TiO2_1day'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS08/',
        'barefile': 'QCMS08_bare',
        'filmfile': 'QCMS08_75kPMMA_TiO2_1day',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }
           #  PMMA sample - Tom sample 8 testing crystal
    samplename = 'PMMA_75k_S08_TiO2_2day'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS08/',
        'barefile': 'QCMS08_bare',
        'filmfile': 'QCMS08_75kPMMA_TiO2_2day',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }
            #  PMMA sample - Tom sample 8 testing crystal
    samplename = 'PMMA_75k_S08_TiO2_3day'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS08/',
        'barefile': 'QCMS08_bare',
        'filmfile': 'QCMS08_75kPMMA_TiO2_3day',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }
             #  PMMA sample - Tom sample 8 testing crystal
    samplename = 'PMMA_75k_S08_TiO2_5day'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS08/',
        'barefile': 'QCMS08_bare',
        'filmfile': 'QCMS08_75kPMMA_TiO2_5day_2',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }
                 #  PMMA sample - Tom sample 8 testing crystal
    samplename = 'PMMA_75k_S08_TiO2_6day'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS08/',
        'barefile': 'QCMS08_bare',
        'filmfile': 'QCMS08_75kPMMA_TiO2_6day',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }
                     #  PMMA sample - Tom sample 8 testing crystal
    samplename = 'PMMA_75k_S08_TiO2_6day_2'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS08/',
        'barefile': 'QCMS08_bare',
        'filmfile': 'QCMS08_75kPMMA_TiO2_6day_2',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }
                         #  PMMA sample - Tom sample 8 testing crystal
    samplename = 'PMMA_75k_S08_TiO2_7day'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS08/',
        'barefile': 'QCMS08_bare',
        'filmfile': 'QCMS08_75kPMMA_TiO2_7day',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }                         #  PMMA sample - Tom sample 8 testing crystal
    samplename = 'PMMA_75k_S08_TiO2_8day'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'QCMS08/',
        'barefile': 'QCMS08_bare',
        'filmfile': 'QCMS08_75kPMMA_TiO2_8day',
        'nhcalc' : ['355'],
        'nhplot': [3, 5]
        }
    return sample