#!/usr/bin/env python
"""
--------------------------------------------------------------------
gictools.meas

Tools for handling GIC measurement data and stations.

Created 2015-2021 by R Bailey, ZAMG Conrad Observatory (Vienna).
Last updated July 2021.
--------------------------------------------------------------------
"""

import os
import sys
from datetime import datetime
import json

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
    """
    Returns a list of the GIC measurement stations available.

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
