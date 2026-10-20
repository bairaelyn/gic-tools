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
T1_HV                             -0.000          -0.000          -0.000
T2_HV                              1.770          -6.931          -5.161
T2_LV                              0.673          -5.688          -5.015
T3_HV                             19.654         -31.301         -11.646
T4_HV                             19.654         -31.301         -11.646
T5_Series                         19.259         -35.550         -16.291
T5_Common                         23.419         -17.946           5.473
T6_HV                            -10.008          58.567          48.559
T7_HV                            -10.008          58.567          48.559
T8_HV                            -27.114         -18.299         -45.412
T8_LV                            -18.812           6.860         -11.952
T9_HV                            -27.114         -18.299         -45.412
T9_LV                            -18.812           6.860         -11.952
T10_HV                             9.733          22.621          32.354
T11_HV                             9.733          22.621          32.354
T12_Series                         7.165         -20.434         -13.268
T12_Common                         1.122          -9.480          -8.358
T13_HV                             1.770          -6.931          -5.161
T13_LV                             0.673          -5.688          -5.015
T14_Series                         7.165         -20.434         -13.268
T14_Common                         1.122          -9.480          -8.358
T15_Series                        19.259         -35.550         -16.291
T15_Common                        23.419         -17.946           5.473

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
    results = PowerGrid.calc_gic_in_grid(En, Ee)
    toc = time.perf_counter()
    gic_trans, id_map = PowerGrid.calc_gic_in_transformers(results, scale=1./3., print_results=True, return_id_map=True)
    gic_lines = PowerGrid.calc_gic_in_power_lines(results)
    gic_subs =  PowerGrid.calc_gic_in_substations(results)
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
