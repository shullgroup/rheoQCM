import logging
logger = logging.getLogger(__name__)

logger.info('module is imported.')
def module_test():
    logger.info('module info')
    logger.debug('module debug')
    logger.error('module error')
    logger.warning('module file')