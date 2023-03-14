#!/usr/bin/env python
"""
--------------------------------------------------------------------
gictools.pred

Tools for forecasting GICs from solar wind data using the neural
network (LSTM) models developed in the 2022 paper.
Paper: https://doi.org/10.1029/2021SW002907
Original code: https://github.com/bairaelyn/SOLARWIND2GIC

Created 2023 by R Bailey, GeoSphere Austria, Conrad Observatory.
Last updated February 2023.
--------------------------------------------------------------------
"""

import os
import calendar
import copy
from datetime import datetime, timedelta, timezone
from dateutil import tz
import json
from matplotlib.dates import num2date, date2num, DateFormatter
import numpy as np
import pandas as pd
import urllib.request

from keras.layers import Layer
from tensorflow.keras import backend as K

# Solar wind and local time variables as model input:
sw_vars = ['speed', 'bz', 'by', 'btot', 'density']
lt_vars = ['sin_DOY', 'cos_DOY', 'sin_LT', 'cos_LT']

# Standard values for SolarWindFeatureEncoder:
max_vals = {'bz' : 50, 'by' : 50, 'btot': 60, 'speed': 1100, 'density': 60}
min_vals = {'bz' : -50, 'by' : -50, 'btot': 0., 'speed': 250., 'density': 0.}

# Standard output, input and offset # of minutes for LSTM:
op_tr, ip_tr, os_tr = 40, 120, 10


# #########################################################################
#     ENCODER FOR FEATURE ENCODING                                        #
# #########################################################################

class SolarWindFeatureEncoder:
    '''Creates an object for encoding all SW variables in preparation for model training.
    
    Attributes:
    -----------
    self.Encoder_bz, self.Encoder_by, self.Encoder_btot, self.Encoder_speed, self.Encoder_density
        : TriangularValueEncoding objects for each SW variable
    self.encode_nodes : int (default=4)
        Number of overlapping triangles to encode variables in.
    '''
    
    def __init__(self, max_vals, min_vals, encode_nodes=4):
        '''Creates an object for encoding all SW variables into overlapping triangles
        (n of triangles=encode_nodes) - each triangle goes from 0 to 1.
    
        Parameters:
        -----------
        max_vals : Dict 
            Contains maximum values for each variable for scaling
        min_vals : Dict 
            Contains minimum values for each variable for scaling
        encode_nodes : int (default=4)
            Number of overlapping triangles - splits one var into four.
        '''
        # IMF Bz, By, Btot
        self.Encoder_bz = TriangularValueEncoding(min_value=min_vals['bz'], max_value=max_vals['bz'], 
                                                  n_nodes=encode_nodes)
        self.Encoder_by = TriangularValueEncoding(min_value=min_vals['by'], max_value=max_vals['by'], 
                                                  n_nodes=encode_nodes)
        self.Encoder_btot = TriangularValueEncoding(min_value=min_vals['btot'], max_value=max_vals['btot'], 
                                                    n_nodes=encode_nodes)
        # Solar wind speed
        self.Encoder_speed = TriangularValueEncoding(min_value=min_vals['speed'], max_value=max_vals['speed'], 
                                                     n_nodes=encode_nodes)
        # Density
        self.Encoder_density = TriangularValueEncoding(min_value=min_vals['density'], max_value=max_vals['density'], 
                                                       n_nodes=encode_nodes)
        # Number of encoding nodes
        self.encode_nodes = encode_nodes
        
    def encode_all(self, df, verbose=False):
        '''Uses the Encoder objects created in initialisation to encode all the SW
        variables in the DataFrame object provided.
        
        Local time variables are also returned, but these are already encoded in sine/cosine
        functions and don't undergo triangular encoding.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Should contain all SW variables under: ['bx', 'by', 'btot', 'speed', 'density']
            Should also contain local time variables: ['sin_DOY', 'cos_DOY', 'sin_LT', 'cos_LT']
        verbose : bool (default=False)
            Print the steps if True.
            
        Returns:
        --------
        omni_feat : np.ndarray, dtype=np.float32
            Returns an array of shape (len(df), 24). SW vars x encode_nodes + local-time vars = 24.
        '''
        bz_encoded = self.Encoder_bz.encode_values(df['bz'].to_numpy())
        by_encoded = self.Encoder_by.encode_values(df['by'].to_numpy())
        btot_encoded = self.Encoder_btot.encode_values(df['btot'].to_numpy())
        speed_encoded = self.Encoder_speed.encode_values(df['speed'].to_numpy())
        density_encoded = self.Encoder_density.encode_values(df['density'].to_numpy())
        # Don't encode these since sin/cos is already an encoding:
        lt_vars = df[['sin_DOY', 'cos_DOY', 'sin_LT', 'cos_LT']].to_numpy()
        
        # Stack them into a feature array:
        omni_feat = np.hstack((bz_encoded, by_encoded, btot_encoded, speed_encoded, density_encoded, lt_vars)).astype(np.float32)
        var_list = ['Bz', 'By', 'Btot', 'Speed', 'Density', 'LT']
            
        # Print locations in array:
        if verbose:
            pos_str = 'Positions in feature array: '
            for i_var, varname in enumerate(var_list):
                if varname != 'LT':
                    pos_str += ("{}=[{}:{}], ".format(varname, i_var*self.encode_nodes, (i_var+1)*self.encode_nodes))
                else:
                    pos_str += ("{}=[{}:{}],".format(varname, i_var*self.encode_nodes, omni_feat.shape[1]-1))
            print(pos_str)

        return omni_feat


class TriangularValueEncoding(object):
    def __init__(self, max_value, min_value, n_nodes: int, normalize: bool = False):
        """
        Code originally from M. Widrich, Python package widis-lstm-tools (Feb 2021):
        https://github.com/widmi/widis-lstm-tools/blob/master/widis_lstm_tools/preprocessing.py
        
        Encodes values in range [min_value, max_value] as array of shape (len(values), n_nodes)
        
        LSTM profits from having a numerical input with large range split into multiple input nodes; This class encodes
        a numerical input as n_nodes nodes with activations of range [0,1]; Each node represents a triangle of width
        triangle_span; These triangles are distributed equidistantly over the input value range such that 2 triangles
        overlap by 1/2 width and the whole input value range is covered; For each value to encode, the height of the
        triangle at this value is taken as node activation, i.e. max. 2 nodes have an activation > 0 for each input
        value, where both activations sum up to 1.
        
        Values are encoded via self.encode_value(value) and returned as float32 numpy array of length self.n_nodes;
        
        Parameters
        ----------
        max_value : float or int
            Maximum value to encode
        min_value : float or int
            Minimum value to encode
        n_nodes : int
            Number of nodes to use for encoding; n_nodes has to be larger than 1;
        normalize : bool
            Normalize encoded values? (default: False)
        """
        if n_nodes < 2:
            raise ValueError("n_nodes has to be > 1")
        
        # Set max_value to max value when starting from min value = 0
        max_value -= min_value
        
        # Calculate triangle_span (triangles overlap -> * 2)
        triangle_span = (max_value / (n_nodes - 1)) * 2
        
        self.n_nodes = int(n_nodes)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.triangle_span = float(triangle_span)
        self.normalize = normalize
    
    def encode_values(self, values):
        """Encode values as multiple triangle node activations
        Parameters
        ----------
        values : numpy.ndarray
            Values to encode as numpy array of shape (len(values),)
        
        Returns
        ----------
        float32 numpy array
            Encoded value as float32 numpy array of shape (len(values), n_nodes)
        """
        values = np.array(values, dtype=np.float32)
        values[:] -= self.min_value
        values[:] *= ((self.n_nodes - 1) / self.max_value)
        encoding = np.zeros((len(values), self.n_nodes), np.float32)
        value_inds = np.arange(len(values))
        
        # low node
        node_ind = np.asarray(np.floor(values), dtype=np.int32)
        node_activation = 1 - (values - node_ind)
        node_ind[:] = np.clip(node_ind, 0, self.n_nodes - 1)
        encoding[value_inds, node_ind] = node_activation
        
        # high node
        node_ind[:] += 1
        node_activation[:] = 1 - node_activation
        node_ind[:] = np.mod(node_ind, self.n_nodes)
        encoding[value_inds, node_ind] = node_activation
        
        # normalize encoding
        if self.normalize:
            encoding[:] -= (1 / self.n_nodes)
        
        return encoding


# #########################################################################
#     ATTENTION MECHANISM USED AS LSTM LAYER                              #
# #########################################################################

class BasicAttention(Layer):
    '''Basic Self-Attention Layer built using this resource:
    https://towardsdatascience.com/create-your-own-custom-attention-layer-understand-all-flavours-2201b5e8be9e
    '''
    
    def __init__(self, return_sequences=True, n_units=1, w_init='normal', b_init='zeros', **kwargs):
        self.return_sequences = return_sequences
        self.n_units = n_units
        self.w_init = w_init
        self.b_init = b_init
        super(BasicAttention,self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.n_features = input_shape[-1]
        self.seq_len = input_shape[-2]
        
        self.W=self.add_weight(name="att_weight", shape=(self.n_features,self.n_units),
                               initializer=self.w_init)
        self.b=self.add_weight(name="att_bias", shape=(self.seq_len,self.n_units),
                               initializer=self.b_init)
        
        super(BasicAttention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)
    
    def get_config(self):
        config = super(BasicAttention, self).get_config()
        config["return_sequences"] = self.return_sequences
        config["n_units"] = self.n_units
        config["w_init"] = self.w_init
        config["b_init"] = self.b_init
        #config["name"] = self.name
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# *******************************************************************
#               FUNCTIONS FOR SAMPLING DATA SETS
# *******************************************************************

def extract_feature_samples(df, i_samples, SWEncoder, output_timerange, input_timerange, offset_timerange, nan_window=15, use_rand_offset=True, verbose=False):
    '''Extracts samples from DataFrame with list of indices provided.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame of time series solar wind data with time as the index.
    i_samples : np.array
        Indices of samples.
    SWEncoder : sw2gic.sampling.SolarWindEncoder
        Object that encodes all variables in overlapping triangles - splits one var into four.
    output_timerange : int
        Window over which to take the max. value of variable var_name for the target.
    input_timerange : int
        Window over which to extract the input solar wind data to make the features.
    offset_timerange : int
        Offset to apply to sample extraction.
    nan_window : int (default=15)
        Max. allowable number of consecutive nans in input_timerange.
        If exceeded, sample is excluded from analysis and training.
    use_rand_offset : bool (default=True)
        If True, applies a random offset in the range -10:10
        
    Returns:
    --------
    (X, i_clean) : (np.array, np.array)
        X : max(df[var_name]) for each sample in i_samples over output_timerange.
        i_clean : i_samples with nan-filled samples removed
    '''
    
    # Variables to extract:
    variables = sw_vars + lt_vars
    np.random.seed(123)
    
    # Define random offsets:
    if use_rand_offset:
        rand_offset = np.random.randint(-10, 10, i_samples.shape)
    else:
        rand_offset = np.zeros(i_samples.shape).astype(int)
    i_clean = []
    omni_seq = []
    nan_count = 0
    for i_i, i in enumerate(i_samples):
        # Subtract the offset and add a random offset so not all maxima are at omni_t=input_timerange
        ioff = i + rand_offset[i_i] - offset_timerange
        df_excerpt = df.iloc[ioff - input_timerange:ioff][variables]
        if len(df_excerpt) == 0:
            raise Exception("Empty DataFrame! Something wrong with indexing: i={}, ioff={}, offset_timerange={}".format(
                            i, ioff, offset_timerange))
        # If there is a period of nans longer than nan_window, ignore it:
        df_excerpt = df_excerpt.interpolate(method='linear', limit=int(nan_window/2), limit_direction='both')
        test_for_nans = np.count_nonzero(np.isnan(df_excerpt.values))
        if test_for_nans > 0:
            nan_count += 1
        else:
            omni_feat_excerpt = SWEncoder.encode_all(df_excerpt)
            omni_seq.append(omni_feat_excerpt)
            i_clean.append(i)
    if verbose:
	    print("Ignored {:.1f}% of data contaminated by NaNs in data set.".format(nan_count/len(i_samples)*100.))
    X = np.array(omni_seq).astype(np.float32)
    
    return X, np.array(i_clean)


def extract_local_time_variables(dtime):
    """Takes the UTC time in numpy date format and 
    returns local time and day of year variables, cos/sin.

    Parameters:
    -----------
    dtime : np.array
        Contains UTC timestamps in datetime format.

    Returns:
    --------
    sin_DOY, cos_DOY, sin_LT, cos_LT : np.arrays
        Sine and cosine of day-of-yeat and local-time.
    """

    utczone = tz.gettz('UTC')
    cetzone = tz.gettz('CET')
    # Original data is in UTC:
    dtimeUTC = [dt.replace(tzinfo=utczone) for dt in dtime]
    # Correct to local time zone (CET) for local time:
    dtimeCET = [dt.astimezone(cetzone) for dt in dtime]
    dtlocaltime = np.array([(dt.hour + dt.minute/60. + dt.second/3600.) for dt in dtimeCET]) / 24.
    # add_day calculation checks for leap years
    add_day = np.array([calendar.isleap(dt.year) for dt in dtimeCET]).astype(int)
    dtdayofyear = (np.array([dt.timetuple().tm_yday for dt in dtimeCET]) + dtlocaltime) / (365. + add_day)
    
    sin_DOY, cos_DOY = np.sin(2.*np.pi*dtdayofyear), np.cos(2.*np.pi*dtdayofyear)
    sin_LT, cos_LT = np.sin(2.*np.pi*dtlocaltime), np.cos(2.*np.pi*dtlocaltime)

    return sin_DOY, cos_DOY, sin_LT, cos_LT


def min_max_loss(y_true, y_pred, loss_adjustment=0.1):
    N = tf.dtypes.cast(len(y_true), tf.float32)
    abs_diff = tf.abs(y_true - y_pred)
    error = tf.reduce_sum(abs_diff, axis=-1) / N
    adjustment = ((tf.reduce_max(y_true, axis=-1) - tf.reduce_min(y_true, axis=-1)) - 
                  (tf.reduce_max(y_pred, axis=-1) - tf.reduce_min(y_pred, axis=-1))) / N
    return error + 0.1*adjustment


# *******************************************************************
#               FUNCTIONS FOR LOADING SATELLITE DATA
# *******************************************************************

def get_noaa_realtime_data(interpolate_gaps=False):
    """
    Downloads and returns real-time solar wind data
    7-day data downloaded from http://services.swpc.noaa.gov/products/solar-wind/

    Parameters
    ==========
    interpolate_gaps :: bool, default=False
        If True, will linearly interpolate the data to minute values.

    Returns
    =======
    sw_data : pandas.DataFrame object
        DF containing NOAA real-time solar wind data under standard keys.
    """

    # Data sources
    url_plasma='http://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json'
    url_mag='http://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'

    # Read plasma data:
    with urllib.request.urlopen(url_plasma) as url:
        dp = json.loads (url.read().decode())
        dpn = [[np.nan if x == None else x for x in d] for d in dp]     # Replace None w NaN
        dtype=[(x, 'float') for x in dp[0]]
        dates = [date2num(datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S.%f")) for x in dpn[1:]]
        dp_ = [tuple([d]+[float(y) for y in x[1:]]) for d, x in zip(dates, dpn[1:])]
        plasma = np.array(dp_, dtype=dtype)
    # Read magnetic field data:
    with urllib.request.urlopen(url_mag) as url:
        dm = json.loads(url.read().decode())
        dmn = [[np.nan if x == None else x for x in d] for d in dm]     # Replace None w NaN
        dtype=[(x, 'float') for x in dmn[0]]
        dates = [date2num(datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S.%f")) for x in dmn[1:]]
        dm_ = [tuple([d]+[float(y) for y in x[1:]]) for d, x in zip(dates, dm[1:])]
        magfield = np.array(dm_, dtype=dtype)

    btot = magfield['bt']
    bxgsm = magfield['bx_gsm']
    bygsm = magfield['by_gsm']
    bzgsm = magfield['bz_gsm']
    pv = plasma['speed']
    pn = plasma['density']
    pt = plasma['temperature']
    itime_dt = magfield['time_tag']
    itime_pla = plasma['time_tag']

    # Linearly interpolate over gaps
    if interpolate_gaps:
        # Make sure plasma/mag data have matching timesteps:
        last_timestep = np.min([magfield['time_tag'][-1], plasma['time_tag'][-1]])
        first_timestep = np.max([magfield['time_tag'][0], plasma['time_tag'][0]])

        nminutes = int((num2date(last_timestep)-num2date(first_timestep)).total_seconds()/60.)
        itime_dt = np.asarray([num2date(first_timestep).replace(tzinfo=timezone.utc) + 
                               timedelta(minutes=i) for i in range(nminutes)])
        itime = date2num(itime_dt).astype(np.float64)
        itime_pla = itime_dt
        btot = np.interp(itime, magfield['time_tag'], btot)
        bxgsm = np.interp(itime, magfield['time_tag'], bxgsm)
        bygsm = np.interp(itime, magfield['time_tag'], bygsm)
        bzgsm = np.interp(itime, magfield['time_tag'], bzgsm)
        pv = np.interp(itime, plasma['time_tag'], pv)
        pn = np.interp(itime, plasma['time_tag'], pn)
        pt = np.interp(itime, plasma['time_tag'], pt)

    # Pack into DataFrame
    sw_mag = pd.DataFrame({'datetime': itime_dt,
                           'btot': btot, 'bx': bxgsm, 'by': bygsm, 'bz': bzgsm})
    sw_pla = pd.DataFrame({'datetime': itime_pla,
                           'speed': pv, 'density': pn, 'temp': pt})

    #logger.info('get_noaa_realtime_data: NOAA RTSW data read completed.')

    return sw_pla, sw_mag


def get_predstorm_realtime_data(urlpath):
    """Reads data from PREDSTORM real-time output.

    Parameters
    ==========
    resolution : str ['hour'(=default) or 'minute']
        Data resolution, only two available.

    Returns
    =======
    pred_data : pandas.Dataframe
        Object containing all data.
    """

    dtype = [('time', 'float'), ('btot', 'float'), ('bx', 'float'), ('by', 'float'), ('bz', 'float'),
            ('density', 'float'), ('speed', 'float'), ('dst', 'float'), ('kp', 'float')]
    data = np.loadtxt(urlpath, usecols=[6,7,8,9,10,11,12,13,14], dtype=dtype)
    datetimes = np.loadtxt(urlpath, usecols=[0,1,2,3,4], dtype=int)
    timedata = []
    for x in datetimes: 
        timedata.append(datetime(x[0],x[1],x[2],x[3],x[4]))
    timedata = np.array([x.replace(tzinfo=timezone.utc) for x in timedata])

    pred_data = pd.DataFrame({var[0]: data[var[0]] for var in dtype})
    pred_data['datetime'] = timedata

    # np.loadtxt will download the file once and won't re-download, so remove it after use:
    os.remove("helioforecast.space/static/sync/predstorm_real_1m.txt")

    return pred_data


def propagate_to_bow_shock(df_SW):

    # Radius of the Earth
    R_Earth = 6371. # km
    # Distance of bow shock (estimated)
    R_BOW = 14. * R_Earth # km
    # Astronomical units
    AU = 149597870.700 # km
    # Distance Earth to L1 (technically not a constant)
    dist_to_L1 = 1496547.603 # km

    sw_variables = ['bx', 'by', 'bz', 'btot', 'speed', 'density', 'temp']
    speed = df_SW['speed'].interpolate(limit_direction='both').to_numpy()
    r_bow = AU - np.full(speed.shape, R_BOW)
    r_L1 = AU - np.full(speed.shape, dist_to_L1)
    timeshift = (r_bow - r_L1)/speed / 60. # in minutes

    # Apply new timesteps to index:
    timeshift_mins = np.array([pd.offsets.DateOffset(minutes=x) for x in timeshift])
    df_SW_timenew = copy.copy(df_SW)
    df_SW_timenew.index += np.array(timeshift_mins)

    ts1 = df_SW_timenew.iloc[0].name
    first_timestep = datetime(ts1.year, ts1.month, ts1.day, ts1.hour, ts1.minute)
    n_mins = int((df_SW_timenew.iloc[-1].name.replace(tzinfo=None) - first_timestep.replace(tzinfo=None)).total_seconds() / 60.)
    df_index_mins = [first_timestep.replace(tzinfo=timezone.utc) + timedelta(minutes=x) for x in range(n_mins)]

    # Change index to numbers to make interpolation easier:
    df_SW_timenew['newtimes'] = date2num(df_SW_timenew.index.to_numpy())
    df_SW_timenew = df_SW_timenew.set_index('newtimes')
    df_index_mins_num = date2num(df_index_mins)

    # Reindexing to a higher sampling rate would be more accurate but much slower
    df_SW_prop = (df_SW_timenew
                             .reindex(df_SW_timenew.index.union(df_index_mins_num))
                             .interpolate(method='index', limit=2)
                             .reindex(df_index_mins_num)
                            )

    df_SW_prop['datetime'] = num2date(df_SW_prop.index.to_numpy())
    df_SW_prop['time'] = df_SW_prop.index.to_numpy()
    df_SW_prop = df_SW_prop.set_index('datetime')

    print("Number of NaNs should stay roughly the same:")
    print("# of NaNs in propagated data:    ", np.count_nonzero(np.isnan(df_SW_prop['speed'])))
    print("# of NaNs in original data:      ", np.count_nonzero(np.isnan(df_SW['speed'])))

    return df_SW_prop



