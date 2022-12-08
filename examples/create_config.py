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
config['DEFAULT'] =            {'Location': 'Server',
                                'CodePath': '/path/to/code'}

# Path to measurement data and station .json files
config['Measurements'] =       {'StationDataPath': '/path/to/json/files',
                                'DataArchivePath': '/path/to/archive'}

# Parameters for real-time computations
config['RealTimeGIC'] =        {'DailyDataPath': '/path/to/dailyfiles',
                                # Where magnetic field data comes from (web or db):
                                'MagDataSource': 'web',
                                # Where to save plots to:
                                'PlotPath':      '/path/to/plots',
                                # Where to copy plots to:
                                'PlotArchive':   '/srv/products/graphs/gic/'}

# Conductivity model used for modelling
config['ConductivityModel'] =  {'ModelNumber'  : '39',
                                'Source'       : 'EURHOM',
                                'Resistivities': '\t'.join(['1000', '300', '1000']),
                                'Thicknesses'  : '\t'.join(['55000', '45000'])}

# Power grid parameters
config['PowerGrid'] =          {'NetworkPath'    : 'examples/Horton_Network.txt',
                                'TransformerPath': 'examples/Horton_TransRes.txt',
                                'ConnectionsPath': 'examples/Horton_Connections.txt',
                                'HV-LV-Levels'   : '\t'.join(['500', '345'])}

# Bounding box for country
config['BoundingBox'] =        {'NCells-X'       : '42',
                                'NCells-Y'       : '65',
                                'CellSpacing-X'  : '5.37',
                                'CellSpacing-Y'  : '8.08',
                                'NorthBoundary'  : '49.81',
                                'SouthBoundary'  : '46.05',
                                'EastBoundary'   : '17.85',
                                'WestBoundary'   : '9.10'}

with open('config_example.ini', 'w') as configfile:
    config.write(configfile)