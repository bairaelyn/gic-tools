# python -m unittest -v test_gictools.py

import gictools.meas
import unittest

class TestGICTOOLS(unittest.TestCase):

    def test_meas(self):
        self.assertEqual(gictools.meas.list_all_measurement_stations("examples"), ['SUB1_test'])
        SUB1 = gictools.meas.MeasurementStation("examples/SUB1_test.json")
        self.assertTrue(SUB1.has_data_on("2021-03-05"))
        self.assertFalse(SUB1.has_data_on("2011-03-05"))

if __name__ == '__main__':
    unittest.main()
