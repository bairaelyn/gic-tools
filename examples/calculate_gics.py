#!/usr/bin/env python
"""
--------------------------------------------------------------------
This script gives examples of how to use the code to calculate GICs
using the Horton grid and the software package gictools.

https://github.com/bairaelyn/gic-tools

Run with this for testing:
$ python -m examples.calculate_gics

Expected output (compare Table VIII in Horton 2012 paper):

TRANSFORMER CURRENTS — 2020-01-01 00:00:00.000000000 UTC
--------------------
ID                             GIC_N [A]       GIC_E [A]     GIC_TOT [A]
T1_HV                              -0.00           -0.00           -0.00
T2_HV                               1.75           -6.94           -5.20
T2_LV                               0.59           -5.18           -4.59
T3_HV                              19.27          -31.55          -12.28
T4_HV                              19.27          -31.55          -12.28
T5_Series                          18.09          -34.90          -16.81
T5_Common                          23.31          -18.26            5.05
T6_HV                              -9.55           59.02           49.47
T7_HV                              -9.55           59.02           49.47
T8_HV                             -27.67          -17.90          -45.58
T8_LV                             -18.84            6.98          -11.86
T9_HV                             -27.67          -17.90          -45.58
T9_LV                             -18.84            6.98          -11.86
T10_HV                             10.15           22.48           32.63
T11_HV                             10.15           22.48           32.63
T12_Series                          7.24          -21.75          -14.51
T12_Common                          0.99           -8.64           -7.65
T13_HV                              1.75           -6.94           -5.20
T13_LV                              0.59           -5.18           -4.59
T14_Series                          7.24          -21.75          -14.51
T14_Common                          0.99           -8.64           -7.65
T15_Series                         18.09          -34.90          -16.81
T15_Common                         23.31          -18.26            5.05

Created 2026 by R Bailey, Conrad Observatory, GeoSphere Austria
Last updated Oct 2026.
--------------------------------------------------------------------
"""

from datetime import datetime, timedelta, timezone
import time
import json
import numpy as np
import pandas as pd

import configparser
config = configparser.ConfigParser()
config.read('examples/config_example.ini')

import sys
sys.path.insert(0, config['DEFAULT']['CodePath'])
import gictools.efield
import gictools.grid
import gictools.results


def main():

    # Read in Horton grid model model:
    # --------------------------------
    PowerGrid = gictools.grid.PowerGrid()
    PowerGrid.load_Grid(config)

    # Make a test geoelectric field (En=1 V/km, Ee=1 V/km):
    # -----------------------------------------------------
    uniform_E = True

    if uniform_E:
        n_timesteps = 5
        En, Ee = gictools.efield.make_test_efield(PowerGrid, 1000, 1000, n_timesteps=n_timesteps)

    else:
        # Read geomagnetic field data:
        df_mag = pd.read_csv("examples/wic_2021-05-12.csv")
        mag_x, mag_y = df_mag['Bx'].to_numpy(), df_mag['By'].to_numpy()
        mag_time = np.array(df_mag['time'])
        mag_x, mag_y = gictools.efield.prepare_B_for_E_calc(mag_x, mag_y, mag_time)

        # Read in details for 1D conductivity model:
        resistivities = np.array([int(x) for x in config['ConductivityModel']['Resistivities'].split('\t')])
        thicknesses   = np.array([int(x) for x in config['ConductivityModel']['Thicknesses'].split('\t')])

        # Compute geoelectric field:
        En_t, Ee_t = gictools.efield.calc_E_using_plane_wave_method(mag_x, mag_y, resistivities, thicknesses)
        En, Ee = {}, {}
        En['39'] = np.array([np.full((PowerGrid.x_ncells, PowerGrid.y_ncells), x) for x in En_t])
        Ee['39'] = np.array([np.full((PowerGrid.x_ncells, PowerGrid.y_ncells), x) for x in Ee_t])

    # Run GIC calculation:
    # --------------------
    tic = time.perf_counter()
    results = PowerGrid.calc_gic_in_grid(En, Ee, use_horton_geom=True)
    toc = time.perf_counter()
    gic_trans, id_map = PowerGrid.calc_gic_in_transformers(results, scale=1./3., print_results=True, return_id_map=True)
    gic_lines = PowerGrid.calc_gic_in_power_lines(results)
    gic_subs =  PowerGrid.calc_gic_in_substations(results, print_results=True)
    print(f"\nGICs calculated in {(toc - tic)*1000./10.:0.2f} milliseconds.\n")

    # Rename transformers with Horton et al. codes:
    transformer_codes = {
        "Sub1_HV": "T1_HV",
        "Sub4_HV": "T2_HV",
        "Sub4_LV": "T2_LV",
        "Sub2_HV": "T3_HV",
        "Sub2_HV_1": "T4_HV",
        "Sub3_HV": "T5_Series",
        "Sub3_LV": "T5_Common",
        "Sub6_HV": "T6_HV",
        "Sub6_HV_1": "T7_HV",
        "Sub5_HV": "T8_HV",
        "Sub5_LV": "T8_LV",
        "Sub5_HV_1": "T9_HV",
        "Sub5_LV_1": "T9_LV",
        "Sub8_HV": "T10_HV",
        "Sub8_HV_1": "T11_HV",
        "Sub4_HV_1": "T12_Series",
        "Sub4_LV_1": "T12_Common",
        "Sub4_HV_2": "T13_HV",
        "Sub4_LV_2": "T13_LV",
        "Sub4_HV_3": "T14_Series",
        "Sub4_LV_3": "T14_Common",
        "Sub3_HV_1": "T15_Series",
        "Sub3_LV_1": "T15_Common",
    }

    gic_trans_horton = gictools.results.rename_transformers(gic_trans, transformer_codes)

    # Pack results into xarray:
    # -------------------------
    start = datetime(2020,1,1)
    times = [start + timedelta(minutes=i) for i in range(n_timesteps)]

    test = gictools.results.gic_results_to_xarray(gic_trans_horton, gic_lines, times, PowerGrid,
        metadata={
            "grid_version": "2026-10-09",
            "code_version": "0.1.0",
            "code_commit": "your_git_commit",
            "conductivity_model": "39",
            "electric_field_mode": "spatially_uniform",
            "input_source": "Uniform geoelectric field of En=1000 mV and Ee=1000 mV.",
            "current_definition": "Transformer currents are provided in A per phase.",
            "transformer_positive_current_direction": "positive into ground",
            "line_positive_current_direction": "line_from_node to line_to_node",
            "timestamp_convention": "Fake timestamps for testing purposes.",
        })

    gictools.results.print_gic_results(test, data='transformer')


if __name__ == '__main__':
	main()
