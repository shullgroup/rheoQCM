from rheoQCM import *

if __name__ == '__main__':
    # import sys
    # import traceback
    # import logging
    # import logging.config

    # set logger
    setup_logging()

    # Get the logger specified in the file
    logger = logging.getLogger(__name__)

    # replace system exception hook
    if QT_VERSION >= 0x50501:
        sys._excepthook = sys.excepthook 
        def exception_hook(exctype, value, traceback):
            # logger.exception('Exception occurred')
            logger.error('Exceptiion error', exc_info=(exctype, value, traceback))
            qFatal('UI error occured.')
            sys._excepthook(exctype, value, traceback) 
            sys.exit(1) 
        sys.excepthook = exception_hook 

    app = QApplication(sys.argv)
    qcm_app = QCMApp()
    qcm_app.show()
    sys.exit(app.exec_())
