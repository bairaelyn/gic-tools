#!/usr/bin/env python
"""
--------------------------------------------------------------------
gictools.support

General tools for scripting and data handling when using GIC/Efield
data in real-time.

Created 2023 by R Bailey, Conrad Observatory, GeoSphere Austria
Last updated May 2023.
--------------------------------------------------------------------
"""

import os
import logging
import logging.config


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(name)s - %(levelname)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'predstorm.log',
            'mode': 'w+',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['file'],
            'level': 'INFO'
        },
    }
}

def init_logging(verbose=False):

    # DEFINE LOGGING MODULE:
    #logging.config.fileConfig(os.path.join(os.path.dirname(__file__), 'config/logging.ini'), disable_existing_loggers=False)
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    # Add handler for logging to shell:
    sh = logging.StreamHandler()
    if verbose:
        sh.setLevel(logging.INFO)
    else:
        sh.setLevel(logging.ERROR)
    shformatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    sh.setFormatter(shformatter)
    logger.addHandler(sh)

    return logger