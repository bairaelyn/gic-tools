#!/usr/bin/env python
"""
--------------------------------------------------------------------
gictools.plot

Functions for plotting geomagnetic, geoelectric and GIC data.

Created 2022 by R Bailey, Conrad Observatory, GeoSphere Austria
Last updated July 2022.

TODO: All of these functions should work with the same dataframe
format.
--------------------------------------------------------------------
"""

import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter

try:
    import seaborn as sns
except:
    print("No seaborn support.")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt
except:
    print("No cartopy support.")

c_En = "green" #"#e69f00"
c_Ee = "purple" #"#56b3e9"
l_min = 1. / (24.*60.) # minute as a fraction of day


# =======================================================================================
# --------------------------------- CREATING MAPS ---------------------------------------
# =======================================================================================

def create_map_of_austria(fig=None, ax=None, figsize=(20,13), use_terrain=True, Bfield_data=True, terrain_alpha=0.7, shpfile=''):
    '''Plots the geomagnetic field variations (H) and the modelled geoelectric field
    in both components.
    !! Requires cartopy !!

    Parameters:
    -----------
    fig :: matplotlib.Figure object (default=None)
        Define Figure object if desired.
    figsize :: tuple (default=(20,13))
        Standard tuple of WxH for matplotlib plot size.
    use_terrain :: bool (default=True)
        Determines whether or not to load terrain image.
    terrain_alpha :: float (default=0.7)
        Alpha value determining how much fade to apply to the terrain (if loaded).

    Returns:
    --------
    ax

    Example:
    --------
    >>> gic = np.random.uniform(low=-30, high=30, size=(len(PowerGrid.latitudes),))
    >>> ax = gicplot.plot_map_with_gics(gic, PowerGrid, En_val=0, Ee_val=-30)
    >>> plt.show()
    '''

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    if fig == None:
        fig = plt.figure(figsize=figsize)

    proj = ccrs.TransverseMercator(13.4, approx=True)

    if len(Bfield_data) > 0.:
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 1, height_ratios=[4, 1])
        ax = fig.add_subplot(gs[0], projection=proj)
        ax_B = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(111, projection=proj)

    # Define projection:
    ax.set_extent([9, 17.5, 46, 49.3], crs=ccrs.PlateCarree())
    
    # Load image tile for background:
    if use_terrain:
        import cartopy.io.img_tiles as cimgt
        stamen_terrain = cimgt.Stamen('terrain-background')
        ax.add_image(stamen_terrain, 8)

        # Make image paler:
        if terrain_alpha > 0.:
            ax.plot(13.5, 47.5, markersize=2000, marker='o', color='white', zorder=1,
                    alpha=terrain_alpha, lw=0, transform=ccrs.Geodetic())

    # Add borders:
    ax.add_feature(cfeature.BORDERS, zorder=3)

    # Add shading to country:
    if len(shpfile) > 0:
        import cartopy.io.shapereader as shpreader
        adm1_shapes = list(shpreader.Reader(shpfile).geometries())
        ax.add_geometries(adm1_shapes, ccrs.PlateCarree(), facecolor='darkolivegreen', alpha=0.1)

    # Remove axis and add gridlines
    ax.gridlines(color='lightgrey', linestyle='-', draw_labels=True, zorder=2)
    ax.axis("off")
    
    if len(Bfield_data) > 0.:
        return fig, [ax, ax_B]
    else:
        return fig, [ax]


def add_network_to_map(ax, Grid, c_lines='slategray', ms=5):
    '''Adds the full power network to a prepared map.

    Parameters:
    -----------
    ax :: matplotlix.axis object
        Plot containing map.
    Grid :: gictools.grid.PowerGrid object
        Object describing the grid.
    c_lines :: str (default='slategrey')
        Matplotlib colour for network lines.
    ms :: float (default=5)
        Markersize for network nodes (circles).

    Returns:
    --------
    ax

    Example:
    --------
    >>> ax = create_map_of_austria()
    >>> ax = add_network_to_map(ax, Grid)
    '''
    
    nfrom, nto = Grid.nodefrom, Grid.nodeto    
    lats, lons = Grid.latitudes, Grid.longitudes

    # Plot connections between stations:
    for i in range(0,len(nfrom)):
        ax.plot([lons[nfrom[i]], lons[nto[i]]], [lats[nfrom[i]], lats[nto[i]]],
                color=c_lines, linewidth=1, marker='o', markersize=ms,
                transform=ccrs.Geodetic(), zorder=4)
    return ax


def add_dc_meas_substations_to_map(ax, st_dict, c_st='red', c_text='black', fs=15, lat_off=-0.04, lon_off=0.11, no_text='4'):
    '''Plots the geomagnetic field variations (H) and the modelled geoelectric field
    in both components.

    Parameters:
    -----------
    ax :: matplotlix.axis object
        Plot containing map.
    st_dict :: dict containing gictools.meas.MeasurementStation objects
        Object describing the measurement stations.
    c_st :: str (default='slategrey')
        Matplotlib colour for station markers (circles).
    c_text :: str (default='slategrey')
        Matplotlib colour for station naming (by clientnum) text.
    fs :: float (default=15)
        Font size for station naming (by clientnum) text.
    lat_off, lon_off :: float (default=-0.04, 0.11)
        Offsets for the text inserts in the map
    no_text :: str (default='4')
        Define a station that shouldn't be labelled on the map (useful for doubles).

    Returns:
    --------
    ax

    Example:
    --------
    >>> ax = create_map_of_austria()
    >>> all_stations, all_nums = gictools.meas.list_all_measurement_stations(station_path, return_nums=True)
    >>> st_dict = {}
    >>> for station, num in zip(all_stations, all_nums):
            st_dict[station] = gictools.meas.MeasurementStation(os.path.join(
                station_path,"{}-{}.json".format(num, station)))
    >>> ax = add_dc_meas_substations_to_map(ax, st_dict)
    '''

    for st in st_dict.values():
        if st.clientnum != no_text:
            ax.plot(st.longitude, st.latitude, markersize=18, marker='o', color=c_st, zorder=10,
                    fillstyle='none', lw=5, markeredgecolor='r', transform=ccrs.Geodetic())
            ax.text(st.longitude+lon_off, st.latitude+lat_off, '#{:02d}'.format(int(st.clientnum)),
                    fontsize=fs, transform=ccrs.Geodetic(), color=c_text, zorder=10)
    return ax


# =======================================================================================
# ------------------------------- PLOTTING FUNCTIONS ------------------------------------
# =======================================================================================

def plot_map_with_gics(gic, Grid, En_val=0, Ee_val=0, point_range=[], Bfield_data=[], figsize=(16,13), scaling=2, shpfile=''):
    '''Plots a map with circles depicting the GICs at each location.

    Parameters:
    -----------
    gic :: pandas.DataFrame
        A pandas DF containing the stations as columns and single GIC values.
    PowerGrid :: gictools.grid.PowerGrid object
        Object describing the grid.
    En_val, Eeval :: floats (default=0)
        Values for the geoelectric field to plot an arrow showing direction.
    figsize :: tuple (default=(20,13))
        Standard tuple of WxH for matplotlib plot size.
    scaling :: float (default=2)
        Scale the size of the GIC values to the map.

    Returns:
    --------
    None
    '''

    fig = plt.figure(figsize=figsize)
    
    fig, axes = create_map_of_austria(fig=fig, use_terrain=False, shpfile=shpfile, Bfield_data=Bfield_data)
    ax_map = axes[0]

    line_voltages = Grid.volt_lines

    # Plot the power grid substations and network lines:
    ax_map = add_network_to_map(ax_map, Grid, c_lines='lightgray')

    # Plot GICs as circles:
    # ---------------------
    x, y = Grid.longitudes, Grid.latitudes
    gic_abs = np.abs(gic)
    gic_pos = np.maximum(gic, 0)
    gic_neg = np.minimum(gic, 0)
    gic_pos, gic_neg = gic_pos*scaling, abs(gic_neg)*scaling
    coord_list = {}

    for gval, gx, gy in zip(gic_abs, x, y):
        coords = (gx, gy)
        if coords not in coord_list:
            coord_list[coords] = gval
        else:
            if gval > coord_list[coords]:
                coord_list[coords] = gval

    for gx, gy, gp, gn, hvlv in zip(x, y, gic_pos, gic_neg, line_voltages):
        alpha = 0.4 # 0.25
        if (gx, gy) in coord_list:
            ax_map.plot(gx, gy, markersize=coord_list[(gx, gy)]*scaling, marker='o', zorder=11,
                    color='darkred', alpha=alpha, lw=0, transform=ccrs.Geodetic())
            coord_list.pop((gx, gy))
            
    # Plot arrow for geoelectric field direction:
    # -------------------------------------------
    # Look here for another arrow option: https://matplotlib.org/stable/gallery/shapes_and_collections/arrow_guide.html
    # Styling here: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html#matplotlib.patches.FancyArrowPatch
    E_tot = np.sqrt(En_val**2. + Ee_val**2.)
    arrow_len = 0.3
    x_pos, y_pos = 3.9-arrow_len*Ee_val/E_tot/2., 3.9-arrow_len*En_val/E_tot/2.
    plt.arrow(x=x_pos, y=y_pos, dx=arrow_len*Ee_val/E_tot, dy=arrow_len*En_val/E_tot, 
              width=0.05, head_length=0.2, fc='k', ec='k',
              transform=fig.dpi_scale_trans)

    # Plot the magnetic field data
    # -----------------------------
    if len(Bfield_data) > 0.:
        ax_H =   axes[1]
        ax_H.plot_date(Bfield_data[0], Bfield_data[1], 'k-', lw=1)
        ax_H.axvspan(Bfield_data[0][point_range[0]], Bfield_data[0][point_range[-1]], facecolor='darkred', lw=3, alpha=alpha)
        ax_H.set_xlim(Bfield_data[0][0], Bfield_data[0][-1]+1/24.)
        ax_H.set_xlabel("Time [UTC]")
        ax_H.set_ylabel("H [nT]")

        # FORMATTING
        # -----------
        for ax_side in ['bottom', 'top', 'right', 'left']:
            ax_H.spines[ax_side].set_color('darkgrey')

        ax_H.set_title("Horizontal ground magnetic field variations (H) over the past 3 days")


    import matplotlib.patches as mpatches

    bbox = ax_map.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    adj_w, adj_h = width/(width+height), height/(width+height)
    base_length_x, base_length_y = 0.3/adj_w, 0.3/adj_h # at max_E_exp
    max_E_exp = 100.
    adj_En, adj_Ee = np.sqrt(np.abs(En_val)/max_E_exp), np.sqrt(np.abs(Ee_val)/max_E_exp)

    x_tail_N, x_tail_E = 0.05, 0.046
    y_tail_N, y_tail_E = 0.042, 0.05
    x_head_N, x_head_E = 0.05, 0.046 + base_length_x*adj_Ee
    y_head_N, y_head_E = 0.042 + base_length_y*adj_En, 0.05
    arrow_N = mpatches.FancyArrowPatch((x_tail_N, y_tail_N), (x_head_N, y_head_N), facecolor='darkolivegreen',
                                       lw=0, mutation_scale=20, transform=ax_map.transAxes)
    arrow_E = mpatches.FancyArrowPatch((x_tail_E, y_tail_E), (x_head_E, y_head_E), facecolor='darkolivegreen',
                                       lw=0, mutation_scale=20, transform=ax_map.transAxes)
    ax_map.add_patch(arrow_N)
    ax_map.add_patch(arrow_E)

    plt.subplots_adjust(hspace=0.1)

    return fig


def plot_B_E_time_series(t, H, En, Ee, min_dbdt=5., max_dbdt=30., max_elim=210., past_days=3, savepath=''):
    '''Plots the geomagnetic field variations (H) and the modelled geoelectric field
    in both components.

    Parameters:
    -----------
    t :: np.array
        Array of timesteps for use in plot_date.
    H :: np.array
        Geomagnetic field measurements in the horizontal component.
    En / Ee :: np.arrays
        Geoelectric field northward and eastward components.
    min_dbdt :: float (default=5)
        Minimum value of dB/min to show as geomagnetic activity.
    max_dbdt :: float (default=30)
        Value at which geomag. activity is saturated. Values above this will have the
        same colour.
    past_days :: int (default=3)
        Number of past days to include in plot (t[-1] - past_days)
    savepath :: str (default='')
        Leaving savepath empty will show the plot. Setting a filename will save to that.

    Returns:
    --------
    None
    '''

    try: sns.set_style("whitegrid")
    except: pass
    lw = 0.75
    today = datetime.strftime(mdates.num2date(t[-1]), "%d.%m.%Y")

    fig, axes = plt.subplots(2, 1, figsize=(10,6))

    # Take the gradient for dH per min:
    dHdt = np.abs(np.diff(H, prepend=H[0]))
    # Remove all values below 10.
    dHdt[dHdt < min_dbdt] = 0.
    for ix, x in enumerate(dHdt):
        for ax in axes:
            ax.axvspan(t[ix]-0.5*l_min, t[ix]+0.5*l_min, fc='gold',
                       alpha=np.min((x,max_dbdt))/max_dbdt)
    axes[0].axvspan(t[ix]+2, t[ix]+2, fc='gold', alpha=0.5, label="Geomagnetic activity")
    axes[0].plot_date(t, H, '-', lw=lw, label="Horizontal B-field (measured)")
    axes[1].plot_date(t, np.abs(En), '-', lw=lw, c=c_En, label="Northward E-field (modelled)")
    axes[1].plot_date(t, np.abs(Ee), '-', lw=lw, c=c_Ee, label="Eastward E-field (modelled)")

    for ax in axes:
        ax.grid(color='lightgrey', alpha=0.5)
        locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim([t[-1]-past_days, t[-1]])

        # Add legend
        ax.legend(loc='upper left')

    # Line over zero for E-fields/GIC:
    axes[1].axhline(y=0., color='darkgrey', ls='--', lw=lw)

    # Line over zero:
    for level, c_level in zip([100., 200., 400.], ['gold', 'orange', 'red']):
        axes[1].axhline(y=level, color=c_level, ls='--', lw=lw)

    # Set axis limits
    e_ylim = np.nanmax( [np.nanmax(np.abs(En)), np.nanmax(np.abs(Ee))] )*1.1
    axes[1].set_ylim( [0, np.nanmax( [max_elim, e_ylim*1.1])] )
    axes[1].set_xlabel("Time [UTC]")
    axes[0].set_ylabel("H [nT]")
    axes[1].set_ylabel("E [mV/km]")
    axes[0].set_title("Past 7 days ({}) of geomagnetic field (H) and geoelectric field (E) in Austria".format(today))

    # Adjust spacing:
    plt.subplots_adjust(hspace=0.28)
    plt.tight_layout()

    # Add run time:
    _add_run_time_text()

    if savepath != '':
        plt.savefig(savepath)
    else:
        plt.show()
    plt.clf()


def plot_gic_time_series(t, gics, past_days=3, savepath=''):
    '''Plots the geomagnetic field variations (H) and the modelled geoelectric field
    in both components.

    Parameters:
    -----------
    t :: np.array
        Array of timesteps for use in plot_date.
    gics :: Dict of np.arrays, variable length
        Arrays containing GIC values for any number of nodes. Figure will be expanded
        to fit. The keys list the station names.
    past_days :: int (default=3)
        Number of past days to include in plot (t[-1] - past_days)
    savepath :: str (default='')
        Leaving savepath empty will show the plot. Setting a filename will save to that.

    Returns:
    --------
    None
    '''

    try: sns.set_style("whitegrid")
    except: pass
    lw=0.75
    today = datetime.strftime(mdates.num2date(t[-1]), "%d.%m.%Y")

    fig, axes = plt.subplots(len(gics), 1, figsize=(10,6))

    for i, node in enumerate(list(gics.keys())):
        axes[i].plot_date(t, np.abs(gics[node]), '-', lw=1, label="Estimated GICs at {}".format(node))

        # Line over zero:
        axes[i].axhline(y=5., color='gold', ls='--', lw=lw)
        axes[i].axhline(y=10., color='orange', ls='--', lw=lw)
        axes[i].axhline(y=20., color='red', ls='--', lw=lw)

        # Limits
        axes[i].set_ylim( [0, np.max( [21., np.max(np.abs(gics[node]))*1.1])] )
        axes[i].set_ylabel("DC [A]")

        # Add textbox with max past values
        textstr = "Max GIC in past 72h: {:.2f} A\n".format(np.max(np.abs(gics[node][-1440*3:])))
        textstr += "Max GIC in past 24h: {:.2f} A\n".format(np.max(np.abs(gics[node][-1440:])))
        textstr += "Max GIC in past 2h: {:.2f} A".format(np.max(np.abs(gics[node][-120:])))
        props = dict(boxstyle='round', facecolor='gold', alpha=0.2)
        axes[i].text(0.02, 0.9, textstr, transform=axes[i].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        axes[i].set_title("Past 72 hours ({}) of GICs at the {} transformer in Austria".format(today, node))

    for ax in axes:
        ax.grid(color='lightgrey', alpha=0.5)
        locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim([t[-1]-past_days, t[-1]])

        # Add legend
        #ax.legend(loc='upper right')

    # Set title
    axes[-1].set_xlabel("Time [UTC]")

    # Adjust spacing:
    plt.subplots_adjust(hspace=0.28)
    plt.tight_layout()

    # Add run time:
    _add_run_time_text()

    if savepath != '':
        plt.savefig(savepath)
    else:
        plt.show()
    plt.clf()


def plot_gic_time_series_zoom(t, gics, station='station', past_days=3, savepath=''):
    '''Plots the geomagnetic field variations (H) and the modelled geoelectric field
    in both components.

    Parameters:
    -----------
    t :: np.array
        Array of timesteps for use in plot_date.
    gics :: Dict of np.arrays, variable length
        Arrays containing GIC values for any number of nodes. Figure will be expanded
        to fit. The keys list the station names.
    station :: str (default='station')
        String with name of power station used for plotting.
    past_days :: int (default=3)
        Number of past days to include in plot (t[-1] - past_days)
    savepath :: str (default='')
        Leaving savepath empty will show the plot. Setting a filename will save to that.

    Returns:
    --------
    None
    '''

    try: sns.set_style("whitegrid")
    except: pass
    lw=0.75
    now = datetime.utcnow()
    today = datetime.strftime(mdates.num2date(t[-1]), "%d.%m.%Y")
    plot_past_days = [1, 7] # [7, 3, 1] #
    skip = 5

    fig, axes = plt.subplots(len(plot_past_days), 1, figsize=(10,6))

    for i, ndays in enumerate(plot_past_days):
        axes[i].plot_date(t[t > t[-1]-ndays], np.abs(gics[t > t[-1]-ndays]), '-', lw=1, label="Estimated GICs at TRANS")

        # Line over zero:
        axes[i].axhline(y=5., color='gold', ls='--', lw=lw)
        axes[i].axhline(y=10., color='orange', ls='--', lw=lw)
        axes[i].axhline(y=20., color='red', ls='--', lw=lw)

        # Limits
        ylim = np.max( [21., np.max(np.abs(gics))*1.1])
        axes[i].set_ylim( [0, ylim] )
        axes[i].set_ylabel("DC [A]")

        axes[i].grid(color='lightgrey', alpha=0.5)
        locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
        formatter = mdates.ConciseDateFormatter(locator)
        axes[i].xaxis.set_major_locator(locator)
        axes[i].xaxis.set_major_formatter(formatter)
        axes[i].set_xlim([t[-1]-ndays, t[-1]])

    # Add textbox with max past values
    textstr = "Max GIC in past 24h: {:.2f} A\n".format(np.max(np.abs(gics[-1440:])))
    textstr += "Max GIC in past 2h: {:.2f} A".format(np.max(np.abs(gics[-120:])))
    props = dict(boxstyle='round', facecolor='gold', alpha=0.2)
    axes[0].text(0.02, 0.9, textstr, transform=axes[0].transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)

    textstr = "Max GIC in past 7 days: {:.2f} A\n".format(np.max(np.abs(gics[-1440*7:])))
    textstr += "Max GIC in past 3 days: {:.2f} A".format(np.max(np.abs(gics[-1440*3:])))
    props = dict(boxstyle='round', facecolor='gold', alpha=0.2)
    axes[1].text(0.02, 0.9, textstr, transform=axes[1].transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)

    zoom_effect(axes[0], axes[1], direction='out')

    # Add axes for 'now' and midnight:
    midnight = mdates.date2num(datetime(now.year, now.month, now.day))
    axes[0].annotate('midnight', xy=(midnight,ylim*0.88), xytext=(midnight+0.007,ylim*0.88), color='k', fontsize=10)
    axes[0].plot_date([midnight,midnight],[-10,1000],'--k', alpha=0.5, linewidth=0.5)

    # Set title
    axes[0].set_title("Past 24 hours ({}) of GICs at the {} transformer".format(today, station))
    axes[1].set_title("Past 7 days ({}) of GICs at the {} transformer".format(today, station))
    axes[-1].set_xlabel("Time [UTC]")

    # Adjust spacing:
    plt.subplots_adjust(hspace=0.01)
    plt.tight_layout()

    # Add run time:
    _add_run_time_text()

    if savepath != '':
        plt.savefig(savepath)
    else:
        plt.show()
    plt.clf()


def plot_gic_bars(gic, stations, voltages, En_val=0, Ee_val=0):
    '''

    Parameters:
    -----------
    gic :: pandas.DataFrame
        A pandas DF containing the stations as columns and single GIC values.
    stations :: list
        List of stations to plot.
    voltages :: dict (str: float)
        Dict of voltages by station name.
    En_val / Ee_val :: floats (default=0)
        These values will be plotted into the title.
    '''

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(5,10))

    labels = []
    pos = 0
    height = 0.4
    for st in stations:
        if st+'_HV' in gic:
            st_HV, st_LV = st+'_HV', st+'_LV'
            if voltages[st_HV] == 380: fc = 'steelblue'
            elif voltages[st_HV] == 220: fc = 'lightsteelblue'
            ax.barh(pos, np.abs(gic[st_HV]), height=height, align='center', fc=fc)
        if st_LV in gic:
            ax.barh(pos+height, np.abs(gic[st_LV]), height=height, align='center', fc='lightsteelblue')

        labels.append(st)
        pos += 1

    # Remove spines (except for the lower x-axis):
    sns.despine(top=True)

    ax.set_yticks(np.arange(pos), labels=labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('GICs [A]')
    ax.set_title(r"GICs in Austrian Power Grid")

    plt.savefig("bar_gic_Ex{:04d}_Ey{:04d}.png".format(int(En_val), int(Ee_val)), dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()


# =======================================================================================
# ------------------------------ FORMATTING FUNCTIONS -----------------------------------
# =======================================================================================

def _add_run_time_text():
    now = datetime.utcnow()
    plt.figtext(0.01,0.020,'Plot created {} UTC.'.format(now.strftime("%d %B %Y, %H:%M")), fontsize=10, ha='left')


def zoom_effect(ax1, ax2, direction='in', **kwargs):
    """
    ax2 : the big main axes
    ax1 : the zoomed axes
    The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    prop_patches = kwargs.copy()
    prop_patches["ec"] = "none"
    prop_patches["alpha"] = 0.1

    if direction == 'in':
        loc1a, loc2a, loc1b, loc2b = 2, 3, 1, 4
    elif direction == 'out':
        loc1a, loc2a, loc1b, loc2b = 3, 2, 4, 1

    c1, c2, bbox_patch1, bbox_patch2, p = \
        _connect_bbox(mybbox1, mybbox2,
                     loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                     prop_lines=kwargs, prop_patches=prop_patches)

    if direction == 'in':
        ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


def _connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):

    from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector, BboxConnectorPatch
    if prop_patches is None:
        prop_patches = prop_lines.copy()
        prop_patches["alpha"] = prop_patches.get("alpha", 1)*0.2

    c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, color='k', lw=0.5, **prop_lines)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, color='k', lw=0.5, **prop_lines)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           **prop_patches)
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p
