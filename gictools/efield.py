#!/usr/bin/env python
"""
--------------------------------------------------------------------
gictools.efield

Tools for handling geomagnetic data in preparation for geoelectric 
field modelling.

Created 2020-2021 by R Bailey, Conrad Observatory, GeoSphere Austria
Some functions taken or adapted from Greg Lucas (USGS):
	https://github.com/greglucas/bezpy/blob/master/bezpy/mt/site.py
Last updated May 2023.
--------------------------------------------------------------------
"""

from datetime import datetime, timedelta
import numpy as np
from matplotlib.dates import date2num, num2date, datestr2num

# *******************************************************************
#                           FUNCTIONS
# *******************************************************************

def calc_E_using_plane_wave_method(mag_x, mag_y, resistivities, thicknesses, dt=60):
    """
    Calculation of geoelectric field for a 1D layered half-space going into the Earth.
    Takes minute data of horizontal magnetic field measurements and outputs minute values of the
    geoelectric field magnitude. Vertical fields (z-direction) are ignored.
    Adapted from https://github.com/greglucas/bezpy/blob/master/bezpy/mt/site.py

    Parameters:
    -----------
    mag_x :: np.array
        Contains Bx (northward) geomagnetic field measurements. Should not contain nans or data gaps.
    mag_y :: np.array
        Contains By (eastward) geomagnetic field measurements. Should not contain nans or data gaps.
    resistivities :: np.array (len=n)
        A list of resistivities in Ohm-m. The last value represents resis. into the Earth (infinity),
        which doesn't have a corresponding layer thickness.
    thicknesses :: np.array (len=n-1)
        A list of the thicknesses (in m) of each resistive layer.
    dt :: int (default=60)
        Sampling rate for the FFT in seconds.

    Returns:
    --------
    (Ex_t, Ey_t) :: (np.array, np.array)
        Array of northward (x) and eastward (y) geoelectric field variables for the db/dt
        values provided over the same time range.
    """

    # Add buffer to reduce edge effects:
    # TODO: should use a proper window for this
    buffer_len = 100000
    mag_x = np.hstack((np.full(buffer_len, mag_x[0]), mag_x, np.full(buffer_len, mag_x[-1])))
    mag_y = np.hstack((np.full(buffer_len, mag_y[0]), mag_y, np.full(buffer_len, mag_y[-1])))

    N0 = len(mag_x)
    # Round N to the next highest power of 2 (+1 (makes it 2) to prevent circular convolution)
    N = 2**(int(np.log2(N0))+2)

    freqs = np.fft.rfftfreq(N, d=dt)    # d is sample spacing in s

    # Z needs to be organized as: xx, xy, yx, yy
    Z_interp = _calc_Z(freqs, resistivities, thicknesses)

    # Take Fourier Transform of function:
    mag_x_fft = np.fft.rfft(mag_x, n=N)
    mag_y_fft = np.fft.rfft(mag_y, n=N)

    # Multiply each frequency component by the transfer function:
    En_fft = Z_interp[0, :]*mag_x_fft + Z_interp[1, :]*mag_y_fft
    Ee_fft = Z_interp[2, :]*mag_x_fft + Z_interp[3, :]*mag_y_fft

    # Inverse Fourier transform:
    En_t = np.real(np.fft.irfft(En_fft)[:N0])
    Ee_t = np.real(np.fft.irfft(Ee_fft)[:N0])

    # Remove buffers around edges:
    En_t = En_t[buffer_len:-buffer_len]
    Ee_t = Ee_t[buffer_len:-buffer_len]

    return En_t, Ee_t


def _calc_Z(freqs, resistivities, thicknesses):
    '''
    Calculates the Z array for contributions of the subsurface resistive layers
    and geomagnetic field components to the geoelectric field.

    Called by calc_E_using_plane_wave_method().
    Taken from https://github.com/greglucas/bezpy/blob/master/bezpy/mt/site.py
    '''

    MU = 1.2566370614*1e-6

    freqs = np.asarray(freqs)

    n = len(resistivities)
    nfreq = len(freqs)

    omega = 2*np.pi*freqs
    complex_factor = 1j*omega*MU

    # eq. 5
    k = np.sqrt(1j*omega[np.newaxis, :]*MU/resistivities[:, np.newaxis])

    # eq. 6
    Z = np.zeros(shape=(n, nfreq), dtype=complex)
    # DC frequency produces divide by zero errors
    with np.errstate(divide='ignore', invalid='ignore'):
        Z[-1, :] = complex_factor/k[-1, :]

        # eq. 7 (reflection coefficient at interface)
        r = np.zeros(shape=(n, nfreq), dtype=complex)

        for i in range(n-2, -1, -1):
            r[i, :] = ((1-k[i, :]*Z[i+1, :]/complex_factor) /
                       (1+k[i, :]*Z[i+1, :]/complex_factor))
            Z[i, :] = (complex_factor*(1-r[i, :]*np.exp(-2*k[i, :]*thicknesses[i])) /
                       (k[i, :]*(1+r[i, :]*np.exp(-2*k[i, :]*thicknesses[i]))))

    # Fill in the DC impedance as zero
    if freqs[0] == 0.:
        Z[:, 0] = 0.

    # Return a 3d impedance [0, Z; -Z, 0]
    Z_output = np.zeros(shape=(4, nfreq), dtype=complex)
    # Only return the top layer impedance
    # Z_factor is conversion from H->B, 1.e-3/MU
    Z_output[1, :] = Z[0, :]*(1.e-3/MU)
    Z_output[2, :] = -Z_output[1, :]

    return Z_output


def load_WIC_data_from_web(starttime):
    '''Loads data from observatory web service and returns X, Y and time array.

    Parameters:
    -----------
    starttime :: string
        starttime can be in many formats, among others: timef_wic = "%Y-%m-%dT%H:%M:%SZ"
    '''

    import json
    try:
        from urllib.request import urlopen
    except:
        from urllib2 import urlopen

    timef_wic = "%Y-%m-%dT%H:%M:%SZ"

    # of=output format, starttime=time to take measurements from
    url_wic = "https://cobs.zamg.ac.at/data/webservice/query.php?id=WIC&of=json&starttime={}".format(starttime)
    try:
        with urlopen(url_wic) as url:
            wic_data = json.loads (url.read().decode("utf-8").strip('<pre>').strip('</pre>'))
    except json.decoder.JSONDecodeError:
        raise Exception(" - No new WIC data. Exiting. (Run code with --skipdownload to run anyway.)")
        sys.exit()

    wic_X, wic_Y = wic_data['ranges']['X']['values'], wic_data['ranges']['Y']['values']
    wic_time = wic_data['domain']['axes']['t']['values']

    return datestr2num(wic_time), np.array(wic_X), np.array(wic_Y)


def make_test_efield(Grid, en_val, ee_val, condmodel='39'):
    '''Return a geoelectric field for testing in the same format as calc_E functions.

    Parameters:
    -----------
    Grid :: gictools.grid.PowerGrid object
        Contains the grid spacing details.
    en_val, ee_val :: floats
        Single values for geoelectric field northward/eastward components in mV.
    condmodel :: str (default='39')
        Model to use for the dictionary.

    Returns:
    --------
    En, Ee :: dicts[condmodel] = np.zeros((1, x_ncells, y_ncells))
        Geoelectric field arrays accessible by dict key condmodel.
    '''
    En, Ee = {}, {}
    En[condmodel] = np.zeros((1, Grid.x_ncells, Grid.y_ncells)) + float(en_val)
    Ee[condmodel] = np.zeros((1, Grid.x_ncells, Grid.y_ncells)) + float(ee_val)
    return (En, Ee)


def prepare_B_for_E_calc(mag_x_raw, mag_y_raw, mag_time, subtract_means=True, return_time=False, timestep='min'):
    '''Takes x and y variations in the geomagnetic field and prepares the
    arrays for use in the plane wave calculation of E. This includes
    subtracting the mean, interpolating to regular timesteps and removing
    any nan (using linear interpolation).

    NOTE: If there are missing/double data points, the new X/Y arrays can
    have different lengths than the original time series. Use return_time
    and take the equally spaced times if this is the case.

    Parameters:
    -----------
    mag_x_raw, mag_y_raw :: np.arrays
        Arrays of geomagnetic x- and y-variations (raw data).
    mag_time :: np.array of time in numerical matplotlib.dates date2num format
        Array containing the time steps of mag_x_raw and mag_y_raw.
    return_time :: bool (default=False)
        If True, the new timesteps are also returned.
    timestep :: str (default='min')
        Defines the timestep used. Can be 'min' or 'sec' for minutes/seconds.

    Returns:
    --------
    (mag_x_raw, mag_y_raw) :: np.arrays (floats)
        New, regular geomagnetic field measurements.
    (mag_x_raw, mag_y_raw, time) if return_time==True ::
        time is returned in numerical matplotlib.dates date2num format.
    '''

    assert timestep in ['min', 'sec'], "'timestep' must be either 'min' or 'sec'!"
    mag_x_raw = np.array(mag_x_raw)
    mag_y_raw = np.array(mag_y_raw)

    # Remove nans and linearly interpolate over them:
    try:
        nan_inds = np.isnan(mag_x_raw)
        mag_x_raw[nan_inds] = np.interp(nan_inds.nonzero()[0], (~nan_inds).nonzero()[0], mag_x_raw[~nan_inds])
    except: # no nans present
        pass
    try:
        nan_inds = np.isnan(mag_y_raw)
        mag_y_raw[nan_inds] = np.interp(nan_inds.nonzero()[0], (~nan_inds).nonzero()[0], mag_y_raw[~nan_inds])
    except: # no nans present
        pass

    # Subtract the mean
    if subtract_means:
        mag_x_raw, mag_y_raw = mag_x_raw - np.mean(mag_x_raw), mag_y_raw - np.mean(mag_y_raw)

    # Interpolate so that every point in time is covered:
    mag_start = num2date(mag_time[0]).replace(tzinfo=None)
    if timestep == 'min':
        n_steps = int((mag_time[-1] - mag_time[0])*24*60) + 1
        new_time = date2num(np.arange(mag_start, mag_start+timedelta(minutes=n_steps), timedelta(minutes=1)).astype(datetime))
    elif timestep == 'sec':
        n_steps = int((mag_time[-1] - mag_time[0])*24*60*60) + 1
        new_time = date2num(np.arange(mag_start, mag_start+timedelta(seconds=n_steps), timedelta(seconds=1)).astype(datetime))
    mag_x_raw, mag_y_raw = np.interp(new_time, mag_time, mag_x_raw), np.interp(new_time, mag_time, mag_y_raw)

    if return_time:
        return (mag_x_raw, mag_y_raw, new_time)
    else:
        return (mag_x_raw, mag_y_raw)
