# python -m unittest -v test_gictools.py

import configparser
import numpy as np
import pandas as pd
import unittest

import gictools.meas
import gictools.efield
import gictools.grid
import gictools.pred

class TestGICTOOLS(unittest.TestCase):

    def test_meas(self):
        self.assertEqual(gictools.meas.list_all_measurement_stations("examples"), ['SUB1_test'])
        SUB1 = gictools.meas.MeasurementStation("examples/01-SUB1_test.json")
        self.assertTrue(SUB1.has_data_on("2021-03-05"))
        self.assertFalse(SUB1.has_data_on("2011-03-05"))

    def test_efield(self):
        # Get geomagnetic data:
        df_mag = pd.read_csv("examples/wic_2021-05-12.csv")
        mag_x, mag_y = df_mag['Bx'].to_numpy(), df_mag['By'].to_numpy()
        mag_time = np.array(df_mag.index)
        mag_x, mag_y = gictools.efield.prepare_B_for_E_calc(mag_x, mag_y, mag_time, timestep='min')

        # Read in details for 1D conductivity model:
        config = configparser.ConfigParser()
        config.read('examples/config_example.ini')
        resistivities = np.array([int(x) for x in config['ConductivityModel']['Resistivities'].split('\t')])
        thicknesses   = np.array([int(x) for x in config['ConductivityModel']['Thicknesses'].split('\t')])

        # Compute geoelectric field:
        Ex_t, Ey_t = gictools.efield.calc_E_using_plane_wave_method(mag_x, mag_y, resistivities, thicknesses)

    def test_calc_gic(self):
        # Read in Horton grid model model:
        config = configparser.ConfigParser()
        config.read('examples/config_example.ini')
        PowerGrid = gictools.grid.PowerGrid()
        PowerGrid.load_Grid(config)

        # Make a test geoelectric field (1 V/km, 0 V/km)
        En, Ee = gictools.efield.make_test_efield(PowerGrid, 1000, 0)

        # Run GIC calculation
        results = PowerGrid.calc_gic_in_grid(En, Ee)
        gic_trans = PowerGrid.calc_gic_in_transformers(results)

    def test_pred(self):
        # Read in example solar wind data:
        df_SW_pred = pd.read_csv('examples/predstorm_example_sw.csv')
        df_SW_pred['datetime'] = pd.to_datetime(df_SW_pred['datetime'])

        # Output, input and offset values in minutes:
        op_tr, ip_tr, os_tr = gictools.pred.op_tr, gictools.pred.ip_tr, gictools.pred.os_tr
        run_model_every = 15 # minutes
        n_sections_pred = int(len(df_SW_pred)/run_model_every)
        i_SW_pred = np.arange(int(np.ceil((ip_tr+os_tr)/run_model_every)), n_sections_pred) * run_model_every

        # Create SolarWindEncoder object:
        SWEncoder = gictools.pred.SolarWindFeatureEncoder(gictools.pred.max_vals, gictools.pred.min_vals, encode_nodes=4)

        # Get arrays of local time variables
        sin_DOY_pred, cos_DOY_pred, sin_LT_pred, cos_LT_pred = gictools.pred.extract_local_time_variables(df_SW_pred['datetime'])

        # Add to feature dataframe
        # (x + 1) / 2 changes range from [-1,1] to [0,1]
        df_SW_pred['sin_DOY'] = (sin_DOY_pred+1)/2
        df_SW_pred['cos_DOY'] = (cos_DOY_pred+1)/2
        df_SW_pred['sin_LT'] = (sin_LT_pred+1)/2
        df_SW_pred['cos_LT'] = (cos_LT_pred+1)/2

        # Extract features for LSTM algorithm
        X_SW_pred_all, i_SW_pred = gictools.pred.extract_feature_samples(df_SW_pred, i_SW_pred, SWEncoder, 
                                                                         op_tr, ip_tr, os_tr, use_rand_offset=False)



if __name__ == '__main__':
    unittest.main()
