#!/usr/bin/env python
"""
--------------------------------------------------------------------
gictools.grid

Contains the PowerGrid object and functions for handling the network
and grid data.

Created 2015 by R Bailey, ZAMG Conrad Observatory (Vienna).
This is a further development of the code in https://github.com/bairaelyn/GEOMAGICA .
Last updated Aug 2022.
--------------------------------------------------------------------
"""

import sys
from math import radians, tan, atan2, atan, cos, sin, asin, sqrt, pi, log, acos
import numpy as np

#####################################################################
#      A.  CLASSES
#####################################################################

class PowerGrid:
    """
    A class containing all information on the power grid.
    Load standard grid data using Grid.load_grid()
    
    Attributes:
        Default:
        - location:     String containing location of grid
        - nnodes:       Number of nodes in the network
        - nconnects:    Number of connections in the network
        - sitenum:      List of station indices
        - sitecodes:    List of station names
        - latitudes:    List of station latitudes
        - longitudes:   List of station longitudes
        - countrycodes: List of station country code ('AT', 'CZ', 'SK', 'HU'...)
        - r_earth:      List of station earthing resistances
        - r_tranformers: List of station transformer resistances
        - r_lines:      List of connection line resistances
        - transformer_type: List of transformer type at station
        - nodefrom:     List of starting nodes for connections
        - nodeto:       List of ending nodes for connections
        - st_to_idx:    Dictionary for getting station index from code
        - idx_to_st:    Dictionary for getting station code from index
        - voltagelevels: For each voltage level (380, 220, 110), list of booleans
                        of which nodes are present in the station.
        - volt_lines:   List of line voltages
    
    Applications:
        Grid = PowerGrid()            # (__init__) Initiate class
        print(Grid)                   # (__str__) Print grid data
        Grid.load_grid(config)        # Load power grid defined in config
    """

    def __init__(self):
        self.stations = None
        self.location = None
        self.nnodes = 0
        self.nconnects = 0
        self.sitenum = []
        self.sitecodes = []
        self.latitudes = []
        self.longitudes = []
        self.countrycodes = []
        self.r_earth = []
        self.r_transformers = []
        self.r_lines = []
        self.nodefrom = []
        self.nodeto = []
        self.st_to_idx = {}
        self.idx_to_st = {}
        self.volt_lines = []
        self.res_levels = {'': 0., 'HV': 0., 'LV': 0.}


    def __str__(self):
        """
        Prints human-readable table of basic grid data.
        """
        printstr = ''
        printstr += ("----------------------------------------------------------------------\n")
        printstr += ("DATA FOR POWER GRID CONFIGURATION\n")
        printstr += ("---------------------------------\n")
        printstr += ("Number of nodes:\t\t%d\n" % self.nnodes)
        printstr += ("Number of connections:\t\t%d\n" % self.nconnects)
        printstr += ("Outermost nodes:\n")
        printstr += ("\tN\t%.2f N (%s)\n" % (max(self.latitudes), self.idx_to_st[np.where(self.latitudes==max(self.latitudes))[0][0]]))
        printstr += ("\tE\t%.2f E (%s)\n" % (max(self.longitudes), self.idx_to_st[np.where(self.longitudes==max(self.longitudes))[0][0]]))
        printstr += ("\tS\t%.2f N (%s)\n" % (min(self.latitudes), self.idx_to_st[np.where(self.latitudes==min(self.latitudes))[0][0]]))
        printstr += ("\tW\t%.2f E (%s)\n" % (min(self.longitudes), self.idx_to_st[np.where(self.longitudes==min(self.longitudes))[0][0]]))
        printstr += ("Node with most connections:\t\t%d\n" % self.nconnects)
        printstr += ("----------------------------------------------------------------------\n")
            
        return printstr


    # ------------------------------------------------
    #     GRID-LOADING FUNCTIONS
    # ------------------------------------------------

    def load_Grid(self, config, per_phase=True):
        """
        Reads in network data from correctly formatted files in the paths.
        Returns long list of parameters.
        
        Parameters
        ----------
        config :: configparser.ConfigParser() object
            Config containing the parameters for this Grid, created using create_config.py
        per_phase :: bool (default=True)
            GICs are calculated per transformer phase (i.e. divided by 3) if True.
        """
        
        valInf = 1.e12
        
        self.networkpath = config['PowerGrid']['NetworkPath']
        self.transformerpath = config['PowerGrid']['TransformerPath']
        self.connectionspath = config['PowerGrid']['ConnectionsPath']
        HV_LV = config['PowerGrid']['HV-LV-Levels'].split('\t')
        self.res_levels['HV'] = int(HV_LV[0])
        self.res_levels['LV'] = int(HV_LV[1])
        
        # ------------------------------
        # Read in general network data:
        # ------------------------------
        with open(self.networkpath, 'r') as network:
            netdata = network.readlines()

        # Geographic latitude and longitude of node:
        geolat, geolon = [], []
        # Node names and codes:
        sitenums, sitenames = [], [] 
        # Earthing resistance of each node:
        res_earth = []
        # Dictionaries for index to station and vice versa:
        i2s, s2i = {}, {}
        # Country code:
        ccodes = []
        # Dictionary for the voltage levels in each station:
        levels = {}
        voltlevel = []
        # List of substations:
        stations, transformers = [], []

        for i in range(len(netdata)):
            data = netdata[i].replace(',','.').replace("'",'').split("\t") 

            # Add grounding node:
            name, number = data[2], int(data[11])
            sitenames.append(name) 
            sitenums.append(number)
            geolat.append(float(data[4]))
            geolon.append(float(data[5]))
            res_earth.append(float(data[6]))
            ccodes.append(data[3])

            levels[name] = [int(data[7]), int(data[8])]
            stations.append(name)
            voltlevel.append(0)

            # Add HV node:
            if float(data[7]) == self.res_levels['HV']: 
                name, number = data[2]+'_HV', int(data[9])
                sitenames.append(name) 
                sitenums.append(number) 
                geolat.append(float(data[4]))
                geolon.append(float(data[5]))
                res_earth.append(valInf)
                ccodes.append(data[3])
                transformers.append(name)
                voltlevel.append(int(data[7]))

            # Add LV node:
            for itest, ext in zip([0,1], ['_HV', '_LV']):      # Could be position 7 or 8
                if float(data[7+itest]) == self.res_levels['LV']: 
                    name, number = data[2]+ext, int(data[9+itest])
                    sitenames.append(name) 
                    sitenums.append(number)
                    geolat.append(float(data[4]))
                    geolon.append(float(data[5]))
                    res_earth.append(valInf)
                    ccodes.append(data[3])
                    transformers.append(name)
                    voltlevel.append(int(data[7+itest]))

        sitenums_new = np.arange(0,len(sitenums))
        old2new_nums = dict(zip(sitenums, sitenums_new))
        i2s = dict(zip(sitenums_new, sitenames))
        s2i = dict(zip(sitenames, sitenums_new))
        n_nodes = len(sitenums_new)

        # --------------------------
        # Read in transformer data:
        # --------------------------
        with open(self.transformerpath, 'r') as transformerfile:
            transdata = transformerfile.readlines()

        if per_phase:
            scale = 1./3.
        else:
            scale = 1.

        # Connections depicted by which nodes they go to and from:
        nodefrom, nodeto = [], []
        # Resistances of lines (transformers included as lines):
        res_line = []
        # Description of what line is:
        type_line = [] # 380 or 220 = HV or LV line, 0 = transformer
        for i in range(len(transdata)):
            trans = transdata[i].strip("\n").replace(',','.').replace("'",'').split("\t")
            station = trans[1]
            res_trans_HV, res_trans_LV = float(trans[3])*scale, float(trans[4])*scale
            trans_type = trans[2]

            nodefrom_HV = s2i[station+'_HV']
            nodeto_HV = s2i[station]
            if trans_type == 'GY-GY':
                nodefrom_LV = s2i[station+'_LV']
                nodeto_LV = s2i[station]

            # Ín Autotransformers, connections run HV --> LV --> Ground:
            if trans_type == 'Auto':
                # HV-to-LV connection:
                nodefrom.append(s2i[station+'_HV'])
                nodeto.append(s2i[station+'_LV'])
                # LV-to-Ground connection:
                nodefrom.append(s2i[station+'_LV'])
                nodeto.append(s2i[station])
                # Extend arrays:
                res_line.append(res_trans_HV)
                res_line.append(res_trans_LV)
                type_line.append(0)
                type_line.append(0)
            # In other transformers, connections are HV --> Ground and LV --> Ground
            elif trans_type == 'GSU':
                # HV-to-Ground connection:
                nodefrom.append(nodefrom_HV)
                nodeto.append(nodeto_HV)
                res_line.append(res_trans_HV)
                type_line.append(0)
            elif trans_type == 'GY-GY':
                # HV-to-Ground connection:
                nodefrom.append(nodefrom_HV)
                nodeto.append(nodeto_HV)
                res_line.append(res_trans_HV)
                type_line.append(0)
                # LV-to-Ground connection:
                nodefrom.append(nodefrom_LV)
                nodeto.append(nodeto_LV)
                res_line.append(res_trans_LV)
                type_line.append(0)

        # -------------------------
        # Read in connection data:
        # -------------------------

        # Distances between nodes (if available):
        len_lines = np.zeros((n_nodes, n_nodes))

        with open(self.connectionspath, 'r') as connectionsfile:
            conndata = connectionsfile.readlines()

        for i in range(len(conndata)):
            conns = conndata[i].strip("\n").replace(',','.').replace("'",'').split("\t")
            linevoltage = int(conns[6])

            if linevoltage == self.res_levels['HV']:
                nodefrom.append(s2i[conns[1]+'_HV'])
                nodeto.append(s2i[conns[2]+'_HV'])
            else:
                st1, st2 = conns[1], conns[2]
                if self.res_levels['HV'] in levels[st1]:
                    nodefrom.append(s2i[st1+'_LV'])
                else:
                    nodefrom.append(s2i[st1+'_HV'])
                if self.res_levels['HV'] in levels[st2]:
                    nodeto.append(s2i[st2+'_LV'])
                else:
                    nodeto.append(s2i[st2+'_HV'])

            res_line.append(float(conns[5])*scale)
            type_line.append(linevoltage)
            try:
                if float(conns[3]) > 0.:
                    len_lines[nodefrom[-1], nodeto[-1]] = float(conns[3])
                    len_lines[nodeto[-1], nodefrom[-1]] = float(conns[3])
            except:
                pass

        # Number of connections:
        nconnects = len(res_line)
                
        if ccodes[0] == 'AT':
            self.location = 'Austria'
        elif ccodes[0] == 'US':
            self.location = 'US'
            
        # Fix infs to a real value:
        res_earth, res_line = np.array(res_earth), np.array(res_line)
        res_earth[res_earth==np.inf] = valInf
        res_line[res_line==np.inf] = valInf
        res_earth[res_earth>=9999999.0] = valInf
        res_line[res_line>=9999999.0] = valInf
        
        # Split into LV (220kV) and HV (380kV) nodes
        self.stations = stations # this is eventually replaced
        self.stationnames = stations
        self.transformers = transformers
        self.nnodes = n_nodes
        self.nconnects = len(res_line)
        self.sitenum = np.array(sitenums_new)
        self.sitecodes = sitenames
        self.latitudes = np.array(geolat)
        self.longitudes = np.array(geolon)
        self.countrycodes = ccodes
        self.r_earth = res_earth
        self.r_transformers = np.zeros(len(res_earth))
        #self.transformer_type = transtype
        self.r_lines = res_line
        self.volt_lines = type_line
        self.nodefrom = np.array(nodefrom)
        self.nodeto = np.array(nodeto)
        self.st_to_idx = s2i
        self.idx_to_st = i2s
        self.voltlevel = voltlevel
        self.dists_lines = len_lines

        # Boundingbox config:
        self.x_inc = float(config['BoundingBox']['CellSpacing-X'])
        self.y_inc = float(config['BoundingBox']['CellSpacing-Y'])
        self.x_ncells = int(config['BoundingBox']['NCells-X'])
        self.y_ncells = int(config['BoundingBox']['NCells-Y'])
        self.nbound, self.sbound = float(config['BoundingBox']['NorthBoundary']), float(config['BoundingBox']['SouthBoundary'])
        self.ebound, self.wbound = float(config['BoundingBox']['EastBoundary']), float(config['BoundingBox']['WestBoundary'])

        return self


    # ------------------------------------------------
    #     FUNCTIONS FOR CALCULATING GIC
    # ------------------------------------------------

    def calc_gic_in_grid(self, En, Ee, lmods=['39'], n_steps=200, use_interp2d=False, threading=True, verbose=False):
        '''
        Provide a list of (EN, EE) tuples, where each EN/EE is a 
        dictionary containing different layer models, and each layer
        model has the electric field over all time stamps.

        Parameters
        ----------
        ne :: int
            Current iteration number
        Grid :: gictools.grid.PowerGrid object
            Contains all the grid data for the calculation.
        En, Ee :: dict('layermodel': np.array), shape=(ne, Grid.lon_cells, Grid.loat_cells)
            Dictionaries containing E-field at certain timesteps.
        n_loops :: int
            Total number of iterations.
        lmods :: list(str), default=['39']
            If using more than one layer model for the E-field, list here by model name.
        n_steps :: int, default=200
            Number of steps to iterate along the power lines.
        use_interp2d :: bool, default=False
            If True, E-field will be interpolated over (much slower)
        threading :: bool, default=True
            Use threading in multiprocessing package to speed up computations on some CPUs.
        verbose :: bool (default=False)
            If True, print steps.
        '''

        self.resis = np.zeros((self.nnodes, self.nnodes))
        self.connections = np.zeros((self.nnodes,self.nnodes))

        for i in range(0,self.nconnects):
            x, y = int(self.nodefrom[i]), int(self.nodeto[i])
            if (self.resis[x,y] > 0. and self.r_lines[i] > 0.):
                # If res already added to this line, add another line in parallel:
                self.resis[x,y] = 1./(1./self.resis[x,y] + 1./self.r_lines[i])
                self.resis[y,x] = self.resis[x,y]
            else:
                self.resis[x,y] = self.r_lines[i]
                self.resis[y,x] = self.resis[x,y]
            if self.resis[x,y] > 0.:
                self.connections[x,y] = 1./self.resis[x,y]
                self.connections[y,x] = 1./self.resis[x,y]

        # Calculate matrix of distance between each point:
        self.dists = np.zeros((self.nnodes, self.nnodes))
        self.azi = np.zeros((self.nnodes, self.nnodes))
        for i in range(0,self.nnodes):
            for j in range(0,self.nnodes):
                lati, loni = self.latitudes[i], self.longitudes[i]
                latj, lonj = self.latitudes[j], self.longitudes[j]
                if self.resis[i, j] != 0.:
                    self.dists[i, j] = grc_distance(lati, loni, latj, lonj, result='km')
                    self.dists[j, i] = self.dists[i, j]
                    try:
                        self.azi[i, j] = grc_azimuth([loni, lati], [lonj, latj])
                        if self.azi[i, j] >= 0. and self.azi[i, j] <= np.pi:
                            self.azi[j, i] = self.azi[i, j] + np.pi
                        if self.azi[i, j] > np.pi and self.azi[i, j] <= 2*np.pi:
                            self.azi[j, i] = self.azi[i, j] - np.pi
                    except ZeroDivisionError:
                        self.azi[i, j] = np.nan
                        self.azi[j, i] = np.nan

        # Create earth impedance matrix:
        # LP1984 eq. (3): **Z**
        # (assuming that nodes are spaced far enough apart to not affect one another)
        self.earthimp = np.diag(self.r_earth + self.r_transformers)

        # Calculate network admittance matrix:
        # LP1984 eq. (10): **Y**
        self.netadmit = -1.*self.connections + np.diag(sum(self.connections))

        # Create system matrix (1+YZ), which needs to be inverted:
        # LP1984 eq. (part of 12): **1 + YZ**
        self.systemmat = np.dot(self.netadmit, self.earthimp) + np.identity(self.nnodes)

        # Latitude and longitude range:
        self.lat_cells = np.asarray([self.sbound + n*self.x_inc for n in range(0, self.x_ncells)])
        self.lon_cells = np.asarray([self.wbound + n*self.y_inc for n in range(0, self.y_ncells)])

        # RUN CALCULATIONS HERE
        # ---------------------------------------------------------------
        n_loops = En[list(En.keys())[0]].shape[0]
        n_efields = range(n_loops)
        if threading == True:
            import multiprocessing
            from itertools import repeat

            pool = multiprocessing.Pool()
            other_args = [repeat(x) for x in [self, En, Ee, n_loops, lmods, n_steps, use_interp2d, verbose]]
            pool_results = pool.starmap(_calc_gic_in_grid, zip(n_efields, *other_args))
        else:
            pool_results = []
            for ne in n_efields:
                pool_results.append(_calc_gic_in_grid(ne, self, En, Ee, n_loops, lmods, n_steps, use_interp2d, verbose))
        # ---------------------------------------------------------------

        return pool_results


    def calc_gic_in_transformers(self, gic_mod_results, gicfilepath='', print_results=False):
        '''
        Takes input from the function PowerGrid.calc_gic_in_grid.
        '''

        # Dictionary of transformer location as a connection
        transID = {}
        for station in self.stations:
            # Normal transformers:
            transID[station+'-'+station+'_HV'] = station+'_HV'
            transID[station+'-'+station+'_LV'] = station+'_LV'
            # Autotransformers:
            transID[station+'_LV'+'-'+station] = station+'_LV'
            transID[station+'-'+station+'_LV'] = station+'_LV'
            transID[station+'_HV'+'-'+station+'_LV'] = station+'_HV'
            transID[station+'_LV'+'-'+station+'_HV'] = station+'_HV'

        # Arrays for final results writing:
        trans_gic = { s+c : [] for s in transID.values() for c in ['', '_GICX', '_GICY'] }
        scale = 1. # in the transformer neutral
        for ne in range(len(gic_mod_results)):

            netconN, netconE, netconT, IlineN, IlineE, _, _ = gic_mod_results[ne]

            # TRANFORMER CURRENTS
            # --------------------
            for i in range(self.nconnects):
                codes = self.idx_to_st[self.nodefrom[i]]+'-'+self.idx_to_st[self.nodeto[i]], 
                codes_r = self.idx_to_st[self.nodeto[i]]+'-'+self.idx_to_st[self.nodefrom[i]]
                if codes in transID or codes_r in transID:
                    if codes in transID: 
                        usecode = transID[codes]
                    elif codes_r in transID: 
                        usecode = transID[codes_r]
                    gic_x, gic_y = IlineN[i]*scale, IlineE[i]*scale
                    gic_tot = gic_x + gic_y
                    trans_gic[usecode].append(gic_tot)
                    trans_gic[usecode+'_GICX'].append(gic_x)
                    trans_gic[usecode+'_GICY'].append(gic_y)
                    output = "{}\t{:10.3f}\t{:10.3f}\t{:10.3f}".format(usecode, gic_x, gic_y, gic_tot)
                    if print_results and ne == 0:
                        if i == 0:
                            print('\n' + 'TRANSFORMER CURRENTS\n' + '---------------------')
                            print("{}\t{:>10}\t{:>10}\t{:>10}".format('NAME', 'GIC_N', 'GIC_E', 'GIC_TOT'))
                        print(output)

        return trans_gic

    def calc_gic_in_substations(self, gic_mod_results, gicfilepath='', print_results=False):
        '''
        Takes input from the function PowerGrid.calc_gic_in_grid.
        '''

        # SUBSTATION CURRENTS
        # --------------------
        netconN, netconE, netconT, IlineN, IlineE, _, _ = gic_mod_results[0]
        for i in range(0, len(netconN)):
            if self.idx_to_st[i] in self.stations:
                output = "{}\t{:10.3f}\t{:10.3f}\t{:10.3f}".format(self.idx_to_st[i], netconN[i], netconE[i], netconT[i])
                if print_results:
                    if i == 0:
                        print('\n' + 'SUBSTATION CURRENTS\n' + '--------------------')
                        print("{}\t{:>10}\t{:>10}\t{:>10}".format('NAME', 'GIC_N', 'GIC_E', 'GIC_TOT'))
                    print(output)

        print("")


    # ------------------------------------------------
    #    ANALYTICS FUNCTIONS
    # ------------------------------------------------

    def list_transformers(self):
        """Provides a list of transformers rather than nodes in the network."""
        
        s2i = self.st_to_idx
        transformers = []
        for station in self.sitecodes:
            # Two levels present:
            if (station in s2i and station+'HV' in s2i and station+'LV' in s2i):
                transformers.append('T'+station+'HV')
                transformers.append('T'+station+'LV')
            # Only one level present:
            elif (station in s2i and station+'HV' in s2i and station+'LV' not in s2i):
                transformers.append('T'+station+'HV')
            elif (station in s2i and station+'LV' in s2i and station+'HV' not in s2i):
                transformers.append('T'+station+'LV')
            
        return transformers
        

    def select_country(self, country="AT"):
        """
        Function used to select all stations from a specific country
        and return the Grid object with all stations from other
        countries removed. Only affects self.stations dictionary.
        
        INPUT:
            - country:          (str)
                                Should be given as ISO-name, e.g. AT
                                
        USAGE:
            >>> Grid.select_country('AT')
        """
        
        origlen = len(self.sitecodes)
        removelist = []
        #cc = countries.CountryChecker('src/support/TM_WORLD_BORDERS-0.3/TM_WORLD_BORDERS-0.3.shp')
        stations = self.stations
        for i in range(origlen):
            station = self.sitecodes[i]
            #lat, lon = station.latitude, station.longitude
            #ciso = cc.getCountry(countries.Point(lat, lon)).iso
            if self.countrycodes[i] != country:
                removelist.append(station)
                    
        for s in removelist:
            del stations[s]
        self.stations = stations
        
        print("Removed {n} stations. Only {m} stations in {c} remain.".format(
            n=(origlen-len(self.stations)), m=len(self.stations), c=country))
                        
        return self


# -------------------------------------------------------------------
#      VI.  FUNCTIONS FOR CALCULATING GICS
# -------------------------------------------------------------------

def _calc_gic_in_grid(ne, Grid, enday, eeday, n_loops, lmods, n_steps, use_interp2d, verbose):
    '''
    Needs lat, lon, systemmat, earthimp.

    Parameters
    ----------
    ne :: int
        Current iteration number
    Grid :: gictools.grid.PowerGrid object
        Contains all the grid data for the calculation.
    enday, eeday :: dict('layermodel': np.array), shape=(ne, Grid.lon_cells, Grid.loat_cells)
        Dictionaries containing E-field at certain timesteps.
    n_loops :: int
        Total number of iterations.
    lmods :: list(str)
        If using more than one layer model for the E-field, list here by model name.
    n_steps :: int
        Number of steps to iterate along the power lines.
    use_interp2d :: bool
        If True, E-field will be interpolated over (much slower)
    verbose :: bool
        If True, print steps.
    '''

    if verbose:
        print("On loop #{} of {} for calculating GIC...".format(ne+1, n_loops))
    if use_interp2d:
        e_int = {}
        for lm in lmods:
            en_int = interpolate.interp2d(Grid.lon_cells, Grid.lat_cells, enday[lm][ne])
            ee_int = interpolate.interp2d(Grid.lon_cells, Grid.lat_cells, eeday[lm][ne])
            e_int[lm] = (en_int, ee_int)

    # ===============================================================
    # PART III: Integrate over E for potential V along lines
    # ===============================================================

    n_conns, n_nodes = Grid.nconnects, Grid.nnodes
    nto, nfrom = Grid.nodeto, Grid.nodefrom
    #n_steps = 200
    slat, slon, flat, flon = np.zeros((n_conns)), np.zeros((n_conns)), np.zeros((n_conns)), np.zeros((n_conns))
    intline, intazi, eucline = np.zeros((n_conns)), np.zeros((n_conns)), np.zeros((n_conns))
    pathlatsteps, pathlonsteps = np.zeros((n_conns,n_steps)), np.zeros((n_conns,n_steps))
    steplat, steplon = np.zeros((n_conns)), np.zeros((n_conns))
    vnseg, veseg = np.zeros((n_conns,n_steps)), np.zeros((n_conns,n_steps))
    Vn_tot, Ve_tot = np.zeros((n_conns)), np.zeros((n_conns))
    E_e, E_n = np.zeros((n_conns,n_steps)), np.zeros((n_conns,n_steps))

    for i in range(0,n_conns):
        # Starting and finishing coordinates:
        slat = Grid.latitudes[nfrom[i]]
        slon = Grid.longitudes[nfrom[i]]
        flat = Grid.latitudes[nto[i]]
        flon = Grid.longitudes[nto[i]]

        # Distances and azimuths of lines:
        intline[i] = Grid.dists[nfrom[i]][nto[i]] # TODO
        #intline[i] = Grid.dists_lines[nfrom[i]][nto[i]]
        intazi[i] =  Grid.azi[nfrom[i]][nto[i]]

        # Set number of n_steps in path integration:
        steplat[i] = (flat - slat)/float(n_steps)
        steplon[i] = (flon - slon)/float(n_steps)

        if steplat[i] != 0.:
            pathlatsteps[i,:] = np.arange(slat, flat-steplat[i]*0.1, steplat[i])
        else:
            pathlatsteps[i,:] = slat

        if steplon[i] != 0.:
            pathlonsteps[i,:] = np.arange(slon, flon-steplon[i]*0.1, steplon[i])
        else:
            pathlonsteps[i,:] = slon

        # Define values of geoelectric field:
        if use_interp2d:
            for p in range(0,len(pathlonsteps[i,:])):
                if layermodel == '00' or layermodel == '88':
                    localmod = get_layermodel_for_00(pathlatsteps[i,p], pathlonsteps[i,p])
                elif layermodel == '11':
                    localmod = get_layermodel_for_11(pathlatsteps[i,p], pathlonsteps[i,p])
                else:
                    localmod = layermodel
                E_n[i,p] = e_int[localmod][0](pathlonsteps[i,p], pathlatsteps[i,p]) / 1000. # Convert from mV to V
                E_e[i,p] = e_int[localmod][1](pathlonsteps[i,p], pathlatsteps[i,p]) / 1000. # Convert from mV to V
        else:
            layermodel = lmods[0]
            E_n[i] = np.full(n_steps, enday[layermodel][ne,0,0]) / 1000. # Convert from mV to V
            E_e[i] = np.full(n_steps, eeday[layermodel][ne,0,0]) / 1000. # Convert from mV to V

        # Integrate to get V = int(E*dL), use cylindrical coordinates:
        # Difference here to matlab version on how many steps are done.
        for j in range(0, n_steps-1):
            # For a North field: #
            vnseg[i,j] = (0.5 * ( E_n[i,j] + E_n[i,j+1] ) * cos(intazi[i])*(intline[i]/n_steps) )
            # For an East field:
            veseg[i,j] = (0.5 * ( E_e[i,j] + E_e[i,j+1] ) * sin(intazi[i])*(intline[i]/n_steps) )

            Vn_tot[i] += vnseg[i,j]
            Ve_tot[i] += veseg[i,j]
            
    # ===============================================================
    # PART IV: Use V and system matrix to determine current (J)
    # ===============================================================

    # Create nvoltage matrix for North field:
    Nvoltage = np.zeros((n_nodes, n_nodes))
    for l in range(0,n_conns):
        Nvoltage[nfrom[l],nto[l]] = Vn_tot[l]
        Nvoltage[nto[l],nfrom[l]] = -Vn_tot[l]

    # Create Evoltage matrix for an East field:
    Evoltage = np.zeros((n_nodes, n_nodes))
    for l in range(0,n_conns):
        Evoltage[nfrom[l],nto[l]] = Ve_tot[l]
        Evoltage[nto[l],nfrom[l]] = -Ve_tot[l]
    
    # Use Ohm's law to calculate the current on each used pathlength:
    # LP1985 eq. (14): J = V/R
    # NORTH
    Nlinecurr = np.zeros((n_nodes, n_nodes))
    for m in range(0,n_nodes):
        for n in range(0,n_nodes):
            if Grid.resis[m,n] > 0.:
                newval = Nvoltage[m,n]/Grid.resis[m,n]
                if not np.isnan(newval):
                    Nlinecurr[m,n] = newval
                else:
                    Nlinecurr[m,n] = 0.

    # EAST
    Elinecurr = np.zeros((n_nodes, n_nodes))
    for m in range(0,n_nodes):
        for n in range(0,n_nodes):
            if Grid.resis[m,n] > 0.:
                newval = Evoltage[m,n]/Grid.resis[m,n]
                if not np.isnan(newval):
                    Elinecurr[m,n] = newval
                else:
                    Elinecurr[m,n] = 0.
    Tlinecurr = Nlinecurr + Elinecurr

    # Total current at each node due to the E field:
    Nsourcevec = np.sum(Nlinecurr, axis=0) 
    Esourcevec = np.sum(Elinecurr, axis=0)

    # GIC formed at each node due to the E field:
    # LP1985 eq. (part of 12): **(1 + YZ)^(-1)*J**
    # systemmat\Esourcevec'
    netconN, resid, rank, s = np.linalg.lstsq(Grid.systemmat, np.transpose(Nsourcevec), rcond=-1)
    netconE, resid, rank, s = np.linalg.lstsq(Grid.systemmat, np.transpose(Esourcevec), rcond=-1)
    #netconN = np.linalg.solve(Grid.systemmat, Nsourcevec.conj().T)
    #netconE = np.linalg.solve(Grid.systemmat, Esourcevec.conj().T)
    netconT = netconN + netconE

    # Currents per line, valid for transformers:
    Vn_trans = np.multiply(netconN, (Grid.r_earth + Grid.r_transformers))
    Ve_trans = np.multiply(netconE, (Grid.r_earth + Grid.r_transformers))
    IlineN, IlineE = np.zeros(n_conns), np.zeros(n_conns)
    for l in range(0,Grid.nconnects):
        IlineN[l] = (Vn_trans[nfrom[l]]-Vn_trans[nto[l]]) / Grid.r_lines[l]
        IlineE[l] = (Ve_trans[nfrom[l]]-Ve_trans[nto[l]]) / Grid.r_lines[l]

    results = [netconN, netconE, netconT, IlineN, IlineE, Nvoltage, Nsourcevec]
    return results


# -------------------------------------------------------------------
#      VI.  USEFUL FUNCTIONS
# -------------------------------------------------------------------

def grc_azimuth(lonlat1, lonlat2):
    """
    Function to compute the geographic distance on a prolate ellipsiod such
    as a planet or moon. This computes the distance accurately - not that it
    makes much difference in most cases, as the location of the points of interest is
    generally relatively poorly known.
    
    This function is based on the formula from the Wikipedia page, verified
    against the Geoscience Australia website calculator
    
    Author: Ciaran Beggan
    Rewritten from matlab into python by Rachel Bailey
    
    Parameters
    ----------
    lonlat1 :: tuple/array/list of floats
        Longitude and latitude of first point in degrees.
    lonlat2 :: tuple/array/list of floats
        Longitude and latitude of second point in degrees.

    Returns
    -------
    a :: float
        Azimuth between point 1 and 2 in radians.
    """

    a, b = 6378.137, 6356.752
    f = (a-b)/a

    u1 = atan((1.-f)*tan(pi/180.*(lonlat1[1])))
    u2 = atan((1.-f)*tan(pi/180.*(lonlat2[1])))

    L = pi/180.*(lonlat2[0] - lonlat1[0])
    Lambda, converge, iters = L, False, 0

    while not converge and iters < 20:
        sinsig = sqrt((cos(u2)*sin(Lambda))**2. + (cos(u1)*sin(u2) - sin(u1)*cos(u2)*cos(Lambda))**2.)
        cossig = sin(u1)*sin(u2) + cos(u1)*cos(u2)*cos(Lambda)
        sig = atan2(sinsig, cossig)
    
        sinalpha = (cos(u1)*cos(u2)*sin(Lambda))/sinsig
        cossqalpha = 1. - sinalpha**2.
        cos2sigm = cossig - (2.*sin(u1)*sin(u2))/cossqalpha

        C = (f/16.) * cossqalpha*(4. + f*(4.-3.*cossqalpha))
    
        calclambda = L + (1.-C)*f*sinalpha*(sig + C*sinalpha*(cos2sigm + C*cossig*(-1. + 2.*cos2sigm) ))
    
        if (abs(Lambda - calclambda) < 10.**(-12.)):
            converge = True
            Lambda = calclambda
        else:
            iters = iters + 1
            Lambda = calclambda

    usq = cossqalpha * ((a**2. - b**2.)/b**2.)
    A = 1. + usq/16384. * (4096. + usq*(-768. + usq*(320. - 175.*usq)))
    B = usq/1024.* (256. + usq*(-128. + usq*(74. - 47.*usq)))
    delsig = B * sinsig * (cos2sigm + 0.25 * B *(cossig *(-1. + 2.*cos2sigm) -(1./6.)*B * cos2sigm*(-3. + 4.*sinalpha**2.)*(-3.+4.*cos2sigm**2.)   ))
    s = b*A*(sig - delsig)
    a1 = atan2(cos(u2)*sin(Lambda), cos(u1)*sin(u2) - sin(u1)*cos(u2)*cos(Lambda) )
    #a2 = atan2(cos(u1)*sin(Lambda), -sin(u1)*cos(u2) + cos(u1)*sin(u2)*cos(Lambda) )    
    #print usq, A, B, delsig, s, a1
 
    if np.isnan(a1):
        a1 = 0.

    #a1 = -a1
    if a1 < 0.:
        a1 = 2.*pi + a1

    return a1


def grc_distance(lat1, lon1, lat2, lon2, result='km'):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) using the Haversine method.
    Combination of:
    http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    &
    http://gis.stackexchange.com/questions/29239/calculate-bearing-between-two-decimal-gps-coordinates
    """

    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula for distance:
    # Source: Wiki https://en.wikipedia.org/wiki/Haversine_formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    #c =  2.* asin( sqrt( sin((dlat)/2.)**2. + cos(lat1)*cos(lat2)* sin((dlon)/2.)**2. ) )
    r = 6371. # Radius of earth in kilometers
    
    if dlat == 0. and dlon == 0.:
        return 0.

    # Great circle distance:
    c = acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(dlon))

    if result == 'km':
        return c * r
    elif result == 'rad':
        return c
