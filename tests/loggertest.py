import numpy as np
import pandas as pd
import json

import logging
import logging.config
import json

from QCM_main import setup_logging
import loggertest_module

setup_longing
i = 1
l = [1, '3', 'aa' , {'c': 'abc'}]
d = {
    'a': [ 1, 3, 5],
    2:  'cc'
}

n = np.random.rand(10)
df = pd.DataFrame.from_dict(d)
logger.debug('infor here: %s', i)
logger.debug('infor here: %s', l)
logger.debug('infor here: %s', d)
logger.debug('infor here: %s', n)
logger.debug('infor here: %s', df)
logger.debug('ass {}'.format(3))

try:
    dfasd
except Exception as err:
    pass
    logger.exception('Exception occurred')
finally:
    logger.exception('finally')
# loggertest_module.module_test()