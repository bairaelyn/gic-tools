#!/usr/bin/env python
"""
--------------------------------------------------------------------
gictools.meas

Tools for handling GIC measurement data and stations.

Created 2015-2021 by R Bailey, ZAMG Conrad Observatory (Vienna).
Last updated October 2021.
--------------------------------------------------------------------
"""

import os
import sys
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd

#####################################################################
#                   MeasurementStation Class
#####################################################################

class MeasurementStation:
    """
    Provides data on each measurement station currently available.
    Will load an individual station from a .json file (for example file,
    see examples/SUB1_test.json).

    Attributes:
    -----------
        name :: str
            Full station name.
        station :: str
            Power grid substation network code.
        code :: str
            Substation transformer code.
        clientnum :: str
            Measurement station number (if applicable).
        latitude, longitude :: float
            Latitude and longitude of the station.
        startdate, enddate :: datetime.datetime objects
            Startdate of measurements and enddate (if applicable).
        notes :: str
            Anything worth noting about the station.

    Functions:
    ----------
        SUB1 = MeasurementStation("examples/SUB1_test.json")   # load object
        print(SUB1)                                            # print details
        SUB1.has_data_on("2020-11-01")                         # if data available on date
    """

    def __init__(self, json_path):
        """Load object details from json file at json_path (str).

        Parameters:
        -----------
        json_path :: str
            Path to JSON file containing station details.

        Returns:
        --------
        MeasurementStation object.
        """

        with open(json_path) as f:
            station_dict = json.load(f)
        self.name = station_dict['name']
        self.station = station_dict['network_code']
        self.code = station_dict['transformer_code']
        self.clientnum = station_dict['number']
        self.latitude = float(station_dict['lat'])
        self.longitude = float(station_dict['lon'])
        self.startdate = datetime.strptime(station_dict['start_date'], "%Y-%m-%d")
        self.enddate = (datetime.strptime(station_dict['end_date'], "%Y-%m-%d") if station_dict['end_date'] != "None" else None)
        self.notes = station_dict['notes']


    def __str__(self):
        """Prints human-readable string of measurement station parameters."""
        printstr = ''
        printstr += ("----------------------------------------------------------------------\n")
        printstr += ("DATA FOR MEASUREMENT STATION\n")
        printstr += ("----------------------------\n")
        printstr += ("Name:\t\t%s\n" % self.name)
        printstr += ("Code:\t\t%s\n" % self.code)
        printstr += ("Client #:\t{}\n".format(self.clientnum))
        printstr += ("Data range:\t{} - {}\n".format(self.startdate.strftime("%Y-%m-%d"),
            (self.enddate.strftime("%Y-%m-%d") if self.enddate != None else "now")))
        printstr += ("Coordinates:\t{:.4f} N, {:.4f} E\n".format(self.latitude, self.longitude))
        printstr += ("----------------------------------------------------------------------\n")

        return printstr


    def has_data_on(self, testdate):
        """Returns True if data is available from station on date provided.

        Parameters:
        -----------
        testdate :: str (YYYY-mm-dd) or datetime.datetime object
            Date to check.

        Returns:
        --------
        data_available :: bool
            True if data is there, False if not.
        """

        if type(testdate) == str:
            testdate = datetime.strptime(testdate, "%Y-%m-%d")

        after_start = testdate >= self.startdate
        before_end  = (testdate < self.enddate if self.enddate != None else True)

        if self.enddate == None and after_start:
            return True
        elif self.enddate != None and after_start and before_end:
            return True
        else:
            return False


# *******************************************************************
#                           FUNCTIONS
# *******************************************************************

def calc_ab_for_gic_from_E(Ex, Ey, GIC):
    """Taken from Pulkkinen et al. (2007), EPS, Eqs. (4-5):
    Determination of ground conductivity and system parameters for optimal modeling of
    geomagnetically induced current flow in technological systems.

    Returns the parameters a and b for the equation GIC = a*Ex + b*Ey. It is
    essentially a least-squares fit.

    Parameters:
    -----------
    Ex : np.array
        Northward geoelectric field component.
    Ey : np.array
        Eastward geoelectric field component.
    GIC : np.array
        Array of GIC measurements for same time range and sampling rate as Ex & Ey.

    Returns:
    --------
    (a, b) : (float, float)
        The components a and b to satisfy the equation GIC = a*Ex + b*Ey.
    """

    assert len(Ex) == len(GIC), "Array lengths do not match."

    ExEy = np.nanmean(Ex*Ey)
    ExEx = np.nanmean(Ex*Ex)
    EyEy = np.nanmean(Ey*Ey)
    denominator = (ExEy**2. - ExEx * EyEy)
    a = (np.nanmean(GIC*Ey) * ExEy - np.nanmean(GIC*Ex) * EyEy) / denominator
    b = (np.nanmean(GIC*Ex) * ExEy - np.nanmean(GIC*Ey) * ExEx) / denominator
    return (a, b)


def list_stations_with_data_on(testdate, json_dir):
    """Returns a list of the GIC measurement stations with data for any given date.

    Parameters:
    -----------
    testdate :: str (YYYY-mm-dd) or datetime.datetime object
        Date to check for data.
    json_dir :: str
        Directory containing json files for all stations with measurements.

    Returns:
    --------
    avail_list :: list
        List of available stations.
    """

    if type(testdate) == str:
        testdate = datetime.strptime(testdate, "%Y-%m-%d")

    # Get list of stations with measurements:
    all_st = list_all_measurement_stations(json_dir)

    # Find those with data on testdata:
    avail_st = []
    for st in all_st:
        MSt = MeasurementStation(os.path.join(json_dir, st+".json"))
        if MSt.has_data_on(testdate):
            avail_st.append(st)

    return avail_st


def list_all_measurement_stations(json_dir):
    """Returns a list of the GIC measurement stations available.

    Parameters:
    -----------
    json_dir :: str
        Directory containing json files for all stations with measurements.

    Returns:
    --------
    all_st :: list
        List of all stations in json_dir.
    """

    files = os.listdir(json_dir)
    json_files = [x for x in files if '.json' in x]

    all_st = []
    for json_file in json_files:
        all_st.append(json_file.strip(".json"))

    return all_st


def read_gic_file(filepath, skiplines=11):
    """For reading original .txt-format GIC files from Austria.

    Parameters:
    -----------
    filepath :: str
        Path to file to be read.
    skiplines :: int (default=11)
        Header lines to skip when reading raw data file.

    Returns:
    --------
    gic_data :: np.ndarray
        Structured array of GIC data with cols 'time', 'temp', 'dc'.
    """
    datestr = os.path.split(filepath)[-1].strip('.txt').split('_')[-1]
    datestr = datestr[:4]+'-'+datestr[4:6]+'-'+datestr[6:]
    datafile = open(filepath, 'rb')
    datastr = datafile.readlines()
    gic_data = np.genfromtxt(datastr[skiplines:], delimiter=';', usecols=(1,2,3),
                             dtype={'names':('time', 'temp', 'dc'), 'formats':('<M8[s]', '<f8', '<f8')},
                             converters={1: lambda x: np.datetime64(datestr+'T'+x.decode().strip(' ')),
                                         2: lambda x: float(x.decode()),               # temperature in C
                                         3: lambda x: float(x.decode())/1000.*(-1)},)  # GICs in mA --> to A
    return gic_data


def read_minute_data(trange, data_path, verbose=False):
    """
    Reads a single file in field format with field values per lat & lon.
    If interpol==True, field will be returned after interpolation.

    Parameters:
    -----------
    trange :: [start time, end time], list of datetime objs
        Start time and end time of range to read.
    data_path :: str
        Path to minute data directiory.
    verbose :: bool (default=False)
        If True, prints debugging messages.

    Returns:
    --------
    df_gic :: pandas.DataFrame
        DF with all the data within the time range.
    """

    starttime, endtime = trange[0], trange[1]
    dates = [starttime + timedelta(days=n) for n in range((endtime-starttime).days)]
    prefix = 'GICMEAS'

    gic_days, missing_files = {}, []
    for i_day, date in enumerate(dates):
        datestr = date.strftime("%Y-%m-%d")
        filepath = os.path.join(data_path, '{}_{}.csv'.format(prefix, datestr))
        if os.path.exists(filepath):
            gic_data = pd.read_csv(filepath)
        else:
            missing_files.append(datestr)
            gic_data = pd.DataFrame()

        df_gic = gic_data if i_day==0 else df_gic.append(gic_data, ignore_index=True)

    if len(missing_files) > 0 and verbose:
        print("WARNING: Missing files for the following days: {}".format(missing_files))

    return df_gic


def write_gic_minute_data(trange, data_path, save_path="mindata", st_json_dir="stations", verbose=True):
    """Writes daily minute data files (.csv) containing all measurements for that day.

    Parameters:
    -----------
    trange :: list of datetime.datetime objects
        Start and end as datetime objects.
    data_path :: str
        Path for directory containing GIC data.
    save_path :: str
        Path to save new CSV-formatted daily files of minute data.
    st_json_dir :: str (default="stations")
        Directory containing all station details for list_all_measurement_stations()

    Returns:
    --------
    None
    """

    all_st = list_all_measurement_stations(st_json_dir)

    # Create path for files if it does not exist:
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # List of days covered by trange:
    daylist = [trange[0] + timedelta(days=n) for n in range(0, (trange[1]-trange[0]).days)]

    # Loop through all days:
    for day in daylist:
        dayf = datetime.strftime(day, "%Y-%m-%d")
        dayf_ = dayf.replace("-",'')
        st_codes, st_nums = [], []

        # Check which stations have measurements:
        for station in all_st:
            MSt = MeasurementStation(os.path.join(st_json_dir, station+".json"))

            path_to_data = os.path.join(data_path, 'GIC0{}'.format(MSt.clientnum),'GIC0{}_{}.txt').format(MSt.clientnum, dayf_)
            if os.path.exists(path_to_data):
                st_codes.append(MSt.code)
                st_nums.append(MSt.clientnum)

        # Read existing files:
        if len(st_codes) > 0:
            if verbose:
                print("Writing file for {} with {}...".format(dayf, st_codes))

            df = pd.DataFrame(columns=['time']+st_codes)

            # Read in GIC and time data:
            for code, num in zip(st_codes, st_nums):
                gic_data = read_gic_file(os.path.join(data_path, 'GIC0{}'.format(num),
                                                      'GIC0{}_{}.txt').format(num, dayf_))

                if len(gic_data) != 0.:
                    try:
                        df[code] = gic_data['dc']
                        df['time'] = gic_data['time']
                    except:
                        if verbose:
                            print("Problem writing day {} for {}. Probably missing data.".format(dayf, code))
                        st_codes.remove(code)
                        st_nums.remove(num)

            # Set time as index and resample to 1 minute:
            df = df.set_index('time')
            # Note that this is centered to avoid an offset of 30s wrt measurements:
            df.index = df.index + pd.Timedelta(30, unit='s')
            df_min = df.iloc[:-30].resample('1min').median()

            # Save to CSV:
            df_min.to_csv(os.path.join(save_path, "GICMEAS_{}.csv".format(dayf)))

    if verbose:
        print("Writing complete.")

    return
