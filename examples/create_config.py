#!/usr/bin/env python
"""
--------------------------------------------------------------------
Creates config file for use in program.
https://docs.python.org/3/library/configparser.html
--------------------------------------------------------------------
"""

import configparser

config = configparser.ConfigParser()

# Path to where gic-tools code is stored
config['DEFAULT'] = {'CodePath': '/path/to/code'}

# Path to measurement data and station .json files
config['Measurements'] = {'StationDataPath': '/path/to/json/files',
                          'DataArchivePath': '/path/to/archive'}

with open('config_example.ini', 'w') as configfile:
    config.write(configfile)