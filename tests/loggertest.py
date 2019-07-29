import numpy as np
import pandas as pd
import json

import logging
import logging.config

with open('logger_config.json', 'r') as f:
    logger_config = json.load(f)
    logger_config['handlers']['file_handler']['filename'] = 'err.log'
logging.config.dictConfig(logger_config)
del logger_config
# Get the logger specified in the file
# logger = logging.getLogger(__name__)
logger = logging.getLogger('errorLogger')

import loggertest_module

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

loggertest_module.module_test()