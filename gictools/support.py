#!/usr/bin/env python
"""
--------------------------------------------------------------------
gictools.support

General tools for scripting and data handling when using GIC/Efield
data in real-time. Also includes tools for calculating different
measures of accuracy.

Created 2023 by R Bailey, Conrad Observatory, GeoSphere Austria
Last updated May 2023.
--------------------------------------------------------------------
"""

import os
import logging
import logging.config
import numpy as np


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


# --------------------------------------------------
#   FIND MEASURES OF ACCURACY
# --------------------------------------------------


def find_ppmcc(arr1, arr2):
    '''
    Calculates the Pearson Product-Moment Correlation Coefficient for a sample
    between two arrays according to a fitting function.
    For error "minpack.error: Result from function call is not a proper array of floats.",
    make sure arrays are defined as arr1 = stream1.ndarray[ind_f].astype(float)

    Parameters:
    -----------
    arr1 : List or np.array
        DataFrame of time series solar wind data with time as the index.
    i_samples : np.array
        Indices of samples.
        
    Returns:
    --------
    r : float
        The PPMCC as a float.
    '''
    
    if type(arr1) == list or type(arr1) == list:
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        
    arr1 = arr1.astype(float)
    arr2 = arr2.astype(float)

    # Coefficient of determination, plot text
    x_mean = np.mean(arr1)
    y_mean = np.mean(arr2)
    r_up = np.sum([(x-x_mean)*(y-y_mean) for x, y in zip(arr1, arr2)])
    r_down = np.sqrt(np.sum([(x-x_mean)*(x-x_mean) for x in arr1]))*np.sqrt(np.sum([(y-y_mean)*(y-y_mean) for y in arr2]))
    if r_down == 0.:
        r_down = 0.0001
    r = r_up/r_down

    return r


def find_performance_parameter(mea, mod):
    '''
    Calculates the performance parameter of the modelled (mod) and measured
    (mea) values. Method taken from Torta et al. 2017 (Space Weather).

    Parameters:
    -----------
    mea : List or np.array
        Measurements that are to be compared to model.
    mod : List or np.array
        Model results that will be compared to measurements.
        
    Returns:
    --------
    PP : float
        The performance parameter as a float.
    '''
        
    if type(mea) == list or type(mod) == list:
        mea = np.array(mea)
        mod = np.array(mod)
        
    mea = mea.astype(float)
    mod = mod.astype(float)
    
    m_mea = mea.mean()
    m_mod = mod.mean()
    std_mea = np.std(mea)
    
    PP = 1. - (1./std_mea) * np.sqrt( np.sum( ((mea-m_mea) - (mod-m_mod))**2. ) / len(mea) )
    
    return PP
