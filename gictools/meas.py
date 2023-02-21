#!/usr/bin/env python
"""
--------------------------------------------------------------------
gictools.meas

Tools for handling GIC measurement data and stations.
Most of these functions are specific to files containing 
measurements in Austria.

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
import warnings

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
        bad_data :: str
            List of ranges containing bad/contaminated measurements.
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
        self.gic_coeffs = [float(station_dict['gic_coefficient_a']), float(station_dict['gic_coefficient_b'])]
        if len(station_dict['bad_data']) > 0.:
            self.bad_data = [[datetime.strptime(x, '%Y-%m-%d') for x in y.split(':')] for y in station_dict['bad_data'].split(',')]
        else:
            self.bad_data = []
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


    def calc_gic(self, En, Ee):

        return self.gic_coeffs[0]*En + self.gic_coeffs[1]*Ee


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


def list_all_measurement_stations(json_dir, return_nums=False):
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
    json_files.sort()

    all_st, all_cl = [], []
    for json_file in json_files:
        all_st.append(json_file.strip(".json").split('-')[1])
        all_cl.append(json_file.split('-')[0])

    if return_nums:
        return all_st, all_cl
    else:
        return all_st


def return_all_measurement_stations(json_dir):
    """Returns a list of the GIC measurement stations available.

    Parameters:
    -----------
    json_dir :: str
        Directory containing json files for all stations with measurements.

    Returns:
    --------
    MStations :: dict
        All available MeasurementStation objects in one dictionary.
    """

    files = os.listdir(json_dir)
    json_files = [x for x in files if '.json' in x]
    json_files.sort()

    MStations = {}
    for json_file in json_files:
        MStations[json_file.strip(".json").split('-')[1]] = MeasurementStation(os.path.join(json_dir,json_file))

    return MStations


def plot_gic_measurements(gic_mea, station_path, gic_mod=[], plotdir='', ylim=0.58, skip=1, verbose=False):
    """Plots the GIC measurements at all available stations for a specific period.

    Parameters:
    -----------
    gic_mea :: pd.DataFrame
        Array containing GIC data (use read_minute_data())
    station_path :: str
        Path detailing where the current list of measurement stations are.
    plotdir :: str (default='')
        Path to where the plots should be saved, if wanted.
    ylim :: float (default=0.5)
        Default value for +/- y limits on the plot.

    Returns:
    --------
    - :: -
        ...
    """

    import matplotlib
    import matplotlib.cm
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    try: import seaborn as sns
    except: pass

    if 'seaborn' in sys.modules:
        gic_colours = list(sns.color_palette())
    else:
        cmap = matplotlib.cm.get_cmap('tab10', 10)
        gic_colours = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

    # Get list of all measurement stations:
    all_stations = list_all_measurement_stations(station_path)

    daystr_start = gic_mea['time'].iloc[0][:10]
    daystr_end =   gic_mea['time'].iloc[-1][:10]
    n_days = (datetime.strptime(daystr_end, '%Y-%m-%d') - datetime.strptime(daystr_start, '%Y-%m-%d')).days + 1

    clients = list(gic_mea.columns)
    clients.remove('time')
    clients = [c for c in all_stations if c in clients]
    n_clients = len(clients)

    t_mea_dn = np.array([mdates.datestr2num(x) for x in gic_mea['time'][::skip]])
    if len(gic_mod) > 0:
        t_mod_dn = np.array([x for x in gic_mod['time'][::skip]])

    fig, axes = plt.subplots(n_clients,1, figsize=(6,1*n_clients+0.5), sharex=True)
    if n_clients == 1:
        axes = [axes]

    # Loop for each subplot:
    print("Plotting data for {} days...".format(n_days))
    for icl, cl in enumerate(clients):
        # Plot:
        axes[icl].plot_date(t_mea_dn, gic_mea[cl][::skip], '-', c=gic_colours[all_stations.index(cl)], zorder=2, lw=1)
        #axes[icl].plot_date(t_mea_dn, gic_mea[cl][::skip], '-', c='k', zorder=2, lw=1)
        y_pos = 0.75 + 0.25/n_clients
        if len(gic_mod) > 0:
            axes[icl].plot_date(t_mod_dn, gic_mod[cl][::skip], '-', c=gic_colours[all_stations.index(cl)], 
                                zorder=0, alpha=0.3, lw=1)
        axes[icl].text(0.01, y_pos, "{}".format(cl), transform=axes[icl].transAxes, fontsize=12)
        axes[icl].set_ylabel("DC [A]")

        # Set axis limits:
        max_val = np.max(np.abs(gic_mea[cl][::skip]))*1.1
        if np.max(np.abs(gic_mea[cl][::skip])) > ylim:
            axes[icl].set_ylim(-max_val, max_val)
        else:
            axes[icl].set_ylim(-ylim, ylim)

    print("Editing labels...")
    if n_days == 1:
        axes[0].set_title("GICs on {}".format(daystr_start))
    else:
        axes[0].set_title("GICs from {} till {}".format(daystr_start, daystr_end))

    axes[-1].set_xlim((t_mea_dn[0], t_mea_dn[-1]))
    axes[-1].set_xlabel("Time (UTC)")
    plt.subplots_adjust(hspace=0)

    locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)

    if len(plotdir) != 0:
        plotpath = os.path.join(plotdir,"GICs_{}.png".format(daystr))
        print("Saving plot to {}.".format(plotpath))
        plt.savefig(plotpath)
    else:
        plt.show()

    plt.clf()


def read_gic_file(data_path, skiplines=11):
    """For reading original .txt-format GIC files from Austria.

    Parameters:
    -----------
    data_path :: str
        Path to file to be read.
    skiplines :: int (default=11)
        Header lines to skip when reading raw data file.

    Returns:
    --------
    gic_data :: np.ndarray
        Structured array of GIC data with cols 'time', 'temp', 'dc'.
    """
    datestr = os.path.split(data_path)[-1].strip('.txt').split('_')[-1]
    datestr = datestr[:4]+'-'+datestr[4:6]+'-'+datestr[6:]
    datafile = open(data_path, 'rb')
    datastr = datafile.readlines()
    gic_data = np.genfromtxt(datastr[skiplines:], delimiter=';', usecols=(1,2,3),
                             dtype={'names':('time', 'temp', 'dc'), 'formats':('<M8[s]', '<f8', '<f8')},
                             converters={1: lambda x: np.datetime64(datestr+'T'+x.decode().strip(' ')),
                                         2: lambda x: float(x.decode()),               # temperature in C
                                         3: lambda x: float(x.decode())/1000.*(-1)},)  # GICs in mA --> to A
    return gic_data


def read_minute_data(trange, data_path, prefix='GICMEAS', verbose=False):
    """
    Reads a single file with GIC measurements. Replies on pd.DataFrames.

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
    The 1-second data is resampled using a simple 1-minute median function shifted by
    thirty seconds to be centred.

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

    all_st, all_cl = list_all_measurement_stations(st_json_dir, return_nums=True)

    MSt = {}
    for station, client in zip(all_st, all_cl):
        MSt[station] = MeasurementStation(os.path.join(st_json_dir, client+'-'+station+".json"))

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

            path_to_data = os.path.join(data_path, 'GIC0{}'.format(MSt[station].clientnum),
                                        'GIC0{}_{}.txt').format(MSt[station].clientnum, dayf_)
            if os.path.exists(path_to_data):
                st_codes.append(MSt[station].code)
                st_nums.append(MSt[station].clientnum)

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
                        # Pack into DataFrame for easy resample:
                        df_resample = pd.DataFrame(gic_data)
                        # Fill missing timesteps with nans:
                        df_resample = df_resample.set_index('time').resample('1S').sum().replace(0.00, np.nan)
                        # Check there are no missing timesteps at start:
                        if len(df_resample) < 86400:
                            start_points_missing = int((df_resample.index[0] - day).total_seconds())
                            if start_points_missing > 0.:
                                if verbose:
                                    print(" - Missing data (n={}) at start of file. Filling with NaNs...".format(start_points_missing))
                                n_points = range(0, start_points_missing)
                                t_missing = [day + timedelta(seconds=x) for x in n_points]
                                df_missing = pd.DataFrame({'temp': [np.nan for x in n_points],
                                                           'dc':   [np.nan for x in n_points]}, index=t_missing)
                                df_resample = pd.concat([df_missing, df_resample])
                            # ...and no missing timesteps at end:
                            end_points_missing = int(((day+timedelta(days=1)) - df_resample.index[-1]).total_seconds() - 1)
                            if end_points_missing > 0:
                                if verbose:
                                    print(" - Missing data (n={}) at end of file. Filling with NaNs...".format(end_points_missing))
                                n_points = range(86400-end_points_missing, 86400)
                                t_missing = [day + timedelta(seconds=x) for x in n_points]
                                df_missing = pd.DataFrame({'temp': [np.nan for x in n_points],
                                                           'dc':   [np.nan for x in n_points]}, index=t_missing)
                                df_resample = pd.concat([df_resample, df_missing])
                        # Pack into final DataFrame:
                        df[code] = df_resample['dc'].astype(float)
                        df['time'] = df_resample.index
                    except Exception as e:
                        if verbose:
                            print(" - ERROR: Problem writing day {} for {}. Exited with error: {}.".format(dayf, code, e))
                        st_codes.remove(code)
                        st_nums.remove(num)

            # Set time as index and resample to 1 minute:
            df = df.set_index('time')
            # Note that this is centered to avoid an offset of 30s wrt measurements:
            df.index = df.index + pd.Timedelta(30, unit='s')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning) # suppress nan-only warnings
                df_min = df.iloc[:-30].resample('1min').median()

            # Save to CSV:
            df_min.to_csv(os.path.join(save_path, "GICMEAS_{}.csv".format(dayf)))

    if verbose:
        print("Writing complete.")

    return


