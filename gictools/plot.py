#!/usr/bin/env python
"""
--------------------------------------------------------------------
gictools.plot

Functions for plotting geomagnetic, geoelectric and GIC data.

Created 2022 by R Bailey, ZAMG Conrad Observatory (Vienna).
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

c_En = "#e69f00"
c_Ee = "#56b3e9"
l_min = 1. / (24.*60.) # minute as a fraction of day

def plot_map_with_gics(gic, nodes, latlons, line_voltages, En_val=0, Ee_val=0):
    '''Plots a map with circles depicting the GICs at each location in Austria.

    Parameters:
    -----------
    gic : pandas.DataFrame
        A pandas DF containing the stations as columns and single GIC values.
    nodes : [Grid.nodefrom, Grid.nodeto] arrays
        Arrays describing how the grid nodes are connected.
    latlons : [Grid.latitudes, Grid.longitudes] arrays
        Arrays describing where the grid nodes are located.
    line_voltages : ...
    '''

    fig = plt.figure(figsize=(20,13))

    # Create plot object with edges:
    proj = ccrs.TransverseMercator(13.4, approx=True)
    ax = fig.add_subplot(111, projection=proj)
    ax1.set_extent([9, 17.5, 46, 49.3], crs=ccrs.PlateCarree())

    # Make image paler:
    ax1.plot(13.5, 47.5, markersize=2000, marker='o', color='white', alpha=0.5, lw=0, transform=ccrs.Geodetic())

    # Add borders:
    ax1.add_feature(cfeature.BORDERS, zorder=10)
    #ax1.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
    ax1.axis("off")

    nfrom = nodes[0]
    nto = nodes[1]
    lats = latlons[0]
    lons = latlons[1]

    # Plot connections between stations:
    for i in range(0,len(nfrom)):
        plt.plot([lons[nfrom[i]], lons[nto[i]]], [lats[nfrom[i]], lats[nto[i]]],
            color='lightgrey', linewidth=1, marker='o', markersize=5,
            transform=ccrs.Geodetic())

    # Plot GICs as circles:
    x, y = lons, lats
    gic_abs = np.abs(gic)
    gic_pos = np.maximum(gic, 0)
    gic_neg = np.minimum(gic, 0)
    scaling = 0.5
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
        alpha = 0.25
        if (gx, gy) in coord_list:
            ax1.plot(gx, gy, markersize=coord_list[(gx, gy)]*scaling, marker='o', color='darkred', alpha=alpha, lw=0, transform=ccrs.Geodetic())
            coord_list.pop((gx, gy))

    #plt.savefig("map_gic_Ex{:04d}_Ey{:04d}.png".format(int(En_val), int(Ee_val)), dpi=200, bbox_inches='tight')
    plt.show()


def plot_B_E_time_series(t, H, En, Ee, min_dbdt=5., max_dbdt=30., savepath=''):
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

    Returns:
    --------
    None
    '''

    sns.set_style("whitegrid")
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
    axes[1].plot_date(t, En, '-', lw=lw, c=c_En, label="Northward E-field (modelled)")
    axes[1].plot_date(t, Ee, '-', lw=lw, c=c_Ee, label="Eastward E-field (modelled)")

    for ax in axes:
        ax.grid(color='lightgrey', alpha=0.5)
        locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim([t[-1]-3, t[-1]])

        # Add legend
        ax.legend(loc='lower right')

    # Line over zero for E-fields/GIC:
    axes[1].axhline(y=0., color='darkgrey', ls='--', lw=0.75)

    # Set axis limits
    e_ylim = np.max( [np.max(np.abs(En)), np.max(np.abs(Ee))] )*1.1
    axes[1].set_ylim((-e_ylim, e_ylim))
    axes[1].set_xlabel("Time [UTC]")
    axes[0].set_ylabel("H [nT]")
    axes[1].set_ylabel("E [mV/km]")
    axes[0].set_title("Past 24 hours ({}) of geomagnetic field (H) and geoelectric field (E) in Austria".format(today))

    if savepath != '':
        plt.savefig(savepath)
    else:
        plt.show()


def plot_gic_time_series(t, gics, savepath=''):
    '''Plots the geomagnetic field variations (H) and the modelled geoelectric field
    in both components.

    Parameters:
    -----------
    t :: np.array
        Array of timesteps for use in plot_date.
    gics :: Dict of np.arrays, variable length
        Arrays containing GIC values for any number of nodes. Figure will be expanded
        to fit. The keys list the station names.

    Returns:
    --------
    None
    '''

    sns.set_style("whitegrid")
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
        ax.set_xlim([t[-1]-3, t[-1]])

        # Add legend
        ax.legend(loc='upper right')

    # Set title
    axes[-1].set_xlabel("Time [UTC]")

    # Adjust spacing:
    plt.subplots_adjust(hspace=0.28)

    if savepath != '':
        plt.savefig(savepath)
    else:
        plt.show()


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
