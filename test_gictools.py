# python -m unittest -v test_gictools.py

import configparser
import numpy as np
import pandas as pd
import unittest

import gictools.meas
import gictools.efield
import gictools.grid

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
        return

    def test_grid(self):
        # Read in Horton grid model model:
        config = configparser.ConfigParser()
        config.read('examples/config_example.ini')
        PowerGrid = gictools.grid.PowerGrid()
        PowerGrid.load_Grid(config)

        print(PowerGrid)

        return

if __name__ == '__main__':
    unittest.main()
