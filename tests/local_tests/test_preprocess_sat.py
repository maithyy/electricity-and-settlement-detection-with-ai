""" Tests for the preprocessing utilities. """

import sys
import unittest
import numpy as np
from pathlib import Path

# import local modules
sys.path.append(".")
from src.preprocessing.preprocess_sat import *
from src.utilities import SatelliteType

ROOT = Path.cwd()


def test_data_array(
    self: unittest.TestCase,
    data_array: xr.DataArray,
    shape: tuple,
    satellite_type: SatelliteType,
    tile_dir: Path,
    parent_tile_dir: str,
):
    # check object and shape
    self.assertIsInstance(data_array, xr.DataArray)
    self.assertEqual(data_array.shape, shape)

    # check dimensions
    self.assertEqual(len(data_array.dims), 4)
    self.assertEqual(data_array.dims, ("date", "band", "height", "width"))

    # check attributes
    self.assertEqual(len(data_array.attrs.keys()), 3)
    for attr in ["satellite_type", "tile_dir", "parent_tile_id"]:
        self.assertIn(attr, data_array.attrs)

    # check satellite type
    self.assertIsInstance(data_array.attrs["satellite_type"], SatelliteType)
    self.assertEqual(data_array.attrs["satellite_type"], satellite_type)

    # check tile_dir
    self.assertIsInstance(data_array.attrs["tile_dir"], Path)
    self.assertEqual(data_array.attrs["tile_dir"], tile_dir)

    # check parent_tile_id
    self.assertIsInstance(data_array.attrs["parent_tile_id"], str)
    self.assertEqual(data_array.attrs["parent_tile_id"], parent_tile_dir)


class TestPreprocessSat(unittest.TestCase):
    """
    Class to test the preprocessing utilities.
    """

    @classmethod
    def setUpClass(self):
        np.random.seed(123456789)
        self.shape = (4, 3, 800, 800)
        self.satellite_type = SatelliteType.S1
        self.tile_dir = ROOT / "foo"
        self.parent_tile_dir = "Tile1"

    def setUp(self):
        self.blank_data_array = xr.DataArray(
            data=np.ones(shape=self.shape),
            dims=("date", "band", "height", "width"),
            coords={
                "date": 4,
                "band": 3,
                "height": 800,
                "width": 800,
            },
        )
        self.blank_data_array.attrs["satellite_type"] = self.satellite_type
        self.blank_data_array.attrs["tile_dir"] = self.tile_dir
        self.blank_data_array.attrs["parent_tile_id"] = self.parent_tile_dir

    def test_gaussian_filter(self):
        """
        Test the gaussian_filter function.
        """
        data_array = gaussian_filter(self.blank_data_array)
        test_data_array(
            self,
            data_array,
            self.shape,
            self.satellite_type,
            self.tile_dir,
            self.parent_tile_dir,
        )
        self.assertEqual(data_array.shape, self.blank_data_array.shape)

    def test_quantile_clip_gbt_false(self):
        """
        Test the quantile_clip function with group_by_time=False
        """
        self.blank_data_array.values = np.zeros(
            self.blank_data_array.shape
        ) + np.multiply(np.random.rand(*self.blank_data_array.shape), 10)

        # add outliers (only first band will have them)
        for j in range(3):
            for x in range(100):
                for y in range(100):
                    cutoff = 20
                    if x <= cutoff and y <= cutoff:
                        self.blank_data_array[0][j][x][y] = 100
                    elif j == 0 and x > cutoff and y > cutoff:
                        self.blank_data_array[0][j][x][y] = 1000

        data_array = quantile_clip(
            self.blank_data_array, clip_quantile=0.01, group_by_time=False
        )
        test_data_array(
            self,
            data_array,
            self.shape,
            self.satellite_type,
            self.tile_dir,
            self.parent_tile_dir,
        )

        for i in range(4):
            for j in range(3):
                if i == 0 and j == 0:
                    self.assertGreaterEqual(data_array[i][j].min(), 0)
                    self.assertLessEqual(data_array[i][j].max(), 100)
                else:
                    self.assertGreaterEqual(data_array[i][j].min(), 0)
                    self.assertLessEqual(data_array[i][j].max(), 10)

    def test_quantile_clip_gbt_true(self):
        """
        Test the quantile_clip function group_by_time=True
        """
        self.blank_data_array.values = np.zeros(
            self.blank_data_array.shape
        ) + np.multiply(np.random.rand(*self.blank_data_array.shape), 10)

        # add outliers
        for band in range(3):
            for x in range(250):
                for y in range(250):
                    cutoff = 50
                    if band == 0:
                        if x > cutoff and y > cutoff:
                            self.blank_data_array[0][band][x][y] = 200
                    elif band == 1:
                        if x <= cutoff and y <= cutoff:
                            self.blank_data_array[0][band][x][y] = 1000
                    elif band == 2:
                        if x <= cutoff and y <= cutoff:
                            self.blank_data_array[0][band][x][y] = 100

        # check outliers are set
        self.assertEqual(self.blank_data_array[0][0].max(), 200)
        self.assertEqual(self.blank_data_array[0][1].max(), 1000)
        self.assertEqual(self.blank_data_array[0][2].max(), 100)
        self.assertEqual(self.blank_data_array.max(), 1000)

        data_array = quantile_clip(
            self.blank_data_array, clip_quantile=0.01, group_by_time=True
        )
        test_data_array(
            self,
            data_array,
            self.shape,
            self.satellite_type,
            self.tile_dir,
            self.parent_tile_dir,
        )

        for date in range(4):
            for band in range(3):
                if band == 0:
                    self.assertGreaterEqual(data_array[date][band].values.min(), 0)
                    self.assertLessEqual(data_array[date][band].values.max(), 200)

                    self.assertIn(200, data_array[0][band].values)
                    self.assertNotIn(1000, data_array[0][band].values)
                else:
                    self.assertGreaterEqual(data_array[date][band].values.min(), 0)
                    self.assertLessEqual(data_array[date][band].values.max(), 10)

    def test_minmax_scale_blank_imgs(self):
        """
        Test the minmax_scale function.
        """
        self.blank_data_array.values = np.zeros(self.blank_data_array.shape)
        data_array = minmax_scale(self.blank_data_array, group_by_time=False)
        test_data_array(
            self,
            data_array,
            self.shape,
            self.satellite_type,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertAlmostEqual(data_array.values.min(), 1, 3)
        self.assertAlmostEqual(data_array.values.max(), 1, 3)
        for i in range(4):
            for j in range(3):
                self.assertAlmostEqual(data_array[i][j].values.min(), 1, 3)
                self.assertAlmostEqual(data_array[i][j].values.max(), 1, 3)

        self.blank_data_array.values = np.zeros(self.blank_data_array.shape)
        data_array = minmax_scale(self.blank_data_array, group_by_time=True)
        test_data_array(
            self,
            data_array,
            self.shape,
            self.satellite_type,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertAlmostEqual(data_array.values.min(), 1, 3)
        self.assertAlmostEqual(data_array.values.max(), 1, 3)
        for i in range(4):
            for j in range(3):
                self.assertAlmostEqual(data_array[i][j].values.min(), 1, 3)
                self.assertAlmostEqual(data_array[i][j].values.max(), 1, 3)

    def test_minmax_scale_gpt_false(self):
        """
        Test the minmax_scale function.
        """
        self.blank_data_array.values = np.zeros(
            self.blank_data_array.shape
        ) + np.multiply(np.random.rand(*self.blank_data_array.shape), 10)

        data_array = minmax_scale(self.blank_data_array, group_by_time=False)
        test_data_array(
            self,
            data_array,
            self.shape,
            self.satellite_type,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertAlmostEqual(data_array.values.min(), 0, 3)
        self.assertAlmostEqual(data_array.values.max(), 1, 3)
        for i in range(4):
            for j in range(3):
                self.assertAlmostEqual(data_array[i][j].values.min(), 0, 3)
                self.assertAlmostEqual(data_array[i][j].values.max(), 1, 3)

    def test_minmax_scale_gpt_true(self):
        """
        Test the minmax_scale function.
        """
        self.blank_data_array.values = np.zeros(
            self.blank_data_array.shape
        ) + np.multiply(np.random.rand(*self.blank_data_array.shape), 10)

        # add outlier
        self.blank_data_array[0][0][0][0] = 100

        data_array = minmax_scale(self.blank_data_array, group_by_time=True)
        test_data_array(
            self,
            data_array,
            self.shape,
            self.satellite_type,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertAlmostEqual(data_array.values.min(), 0, 3)
        self.assertAlmostEqual(data_array.values.max(), 1, 3)
        for i in range(4):
            for j in range(3):
                if j == 0:
                    if i == 0:
                        # assert there is only a single 1 (the outlier)
                        self.assertEqual(np.count_nonzero(data_array[i][j] == 1), 1)
                        self.assertAlmostEqual(data_array[i][j].values.min(), 0, 3)
                        self.assertAlmostEqual(data_array[i][j].values.max(), 1, 3)
                    else:
                        # because they are grouped by time, all values should be between 0 and 0.1 (no outlier)
                        self.assertEqual(np.count_nonzero(data_array[i][j] == 1), 0)
                        self.assertAlmostEqual(data_array[i][j].values.min(), 0, 3)
                        self.assertAlmostEqual(data_array[i][j].values.max(), 0.1, 3)
                else:
                    self.assertAlmostEqual(data_array[i][j].values.min(), 0, 3)
                    self.assertAlmostEqual(data_array[i][j].values.max(), 1, 3)

    def test_brighten(self):
        """
        Test the brighten function.
        """
        self.blank_data_array.values = np.ones(self.blank_data_array.shape)

        data_array = brighten(self.blank_data_array, 2, 12)
        test_data_array(
            self,
            data_array,
            self.shape,
            self.satellite_type,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertEqual(data_array.values.min(), data_array.values.max())
        self.assertEqual(data_array.values.min(), 14)
        self.assertEqual(data_array.values.max(), 14)

        self.blank_data_array.values = np.zeros(self.blank_data_array.shape)

        data_array = brighten(self.blank_data_array, 100, 10)
        test_data_array(
            self,
            data_array,
            self.shape,
            self.satellite_type,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertEqual(data_array.values.min(), data_array.values.max())
        self.assertEqual(data_array.values.min(), 10)
        self.assertEqual(data_array.values.max(), 10)

    def test_gamma_corr(self):
        """
        Test the gammacorr function.
        """
        self.blank_data_array.values = np.add(np.ones(self.blank_data_array.shape), 1)

        data_array = gammacorr(self.blank_data_array)
        test_data_array(
            self,
            data_array,
            self.shape,
            self.satellite_type,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertEqual(data_array.values.min(), data_array.values.max())
        self.assertAlmostEqual(data_array.values.min(), 1.414, 3)
        self.assertAlmostEqual(data_array.values.max(), 1.414, 3)

        self.blank_data_array.values = np.zeros(self.blank_data_array.shape)

        data_array = gammacorr(self.blank_data_array, 12)
        test_data_array(
            self,
            data_array,
            self.shape,
            self.satellite_type,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertEqual(data_array.values.min(), data_array.values.max())
        self.assertEqual(data_array.values.min(), 0)
        self.assertEqual(data_array.values.max(), 0)

    def test_convert_data_to_db(self):
        """
        Test the convert_data_to_db function.
        """
        self.blank_data_array.values = np.add(np.ones(self.blank_data_array.shape), 1)

        data_array = convert_data_to_db(self.blank_data_array)
        test_data_array(
            self,
            data_array,
            self.shape,
            self.satellite_type,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertEqual(data_array.values.min(), data_array.values.max())
        self.assertAlmostEqual(data_array.values.min(), 0.301, 3)
        self.assertAlmostEqual(data_array.values.max(), 0.301, 3)

    def test_maxprojection_viirs(self):
        """
        Test the maxprojection_VIIRS function.
        """
        # 18 dates 1 band of 800x800 imgs ascending from 0's to 9's to 0's
        alpha = list()
        for x in range(10):
            alpha.append(
                np.stack(
                    [
                        np.add(np.zeros((800, 800)), x),
                    ]
                )
            )
        for x in range(9, -1):
            alpha.append(
                np.stack(
                    [
                        np.add(np.zeros((800, 800)), x),
                    ]
                )
            )
        data = np.stack(alpha)

        blank_viirs_array = xr.DataArray(
            data=data,
            dims=("date", "band", "height", "width"),
            coords={
                "date": 18,
                "band": 1,
                "height": 800,
                "width": 800,
            },
        )

        blank_viirs_array.attrs["satellite_type"] = SatelliteType.VIIRS
        blank_viirs_array.attrs["tile_dir"] = self.tile_dir
        blank_viirs_array.attrs["parent_tile_id"] = self.parent_tile_dir

        # ensure min is 0 and max is 9
        self.assertEqual(blank_viirs_array.values.min(), 0)
        self.assertEqual(blank_viirs_array.values.max(), 9)

        data_array = maxprojection_viirs(blank_viirs_array)
        test_data_array(
            self,
            data_array,
            (1, 1, 800, 800),
            SatelliteType.VIIRS_MAX_PROJ,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertEqual(data_array.values.min(), data_array.values.max())
        self.assertEqual(data_array.values.min(), 9)
        self.assertEqual(data_array.values.max(), 9)

    def test_preprocess_sentinel1(self):
        """
        Test the preprocess_sentinel1 function.
        """
        self.blank_data_array.values = np.zeros(
            self.blank_data_array.shape
        ) + np.multiply(np.random.rand(*self.blank_data_array.shape), 10)

        self.blank_data_array.attrs["satellite_type"] = SatelliteType.S1

        data_array = preprocess_sentinel1(self.blank_data_array)
        test_data_array(
            self,
            data_array,
            self.shape,
            SatelliteType.S1,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertEqual(data_array.values.min(), 0)
        self.assertEqual(data_array.values.max(), 1)
        self.assertEqual(data_array.shape, self.blank_data_array.shape)

    def test_preprocess_sentinel2(self):
        """
        Test the preprocess_sentinel2 function.
        """
        self.blank_data_array.values = np.zeros(
            self.blank_data_array.shape
        ) + np.multiply(np.random.rand(*self.blank_data_array.shape), 10)

        self.blank_data_array.attrs["satellite_type"] = SatelliteType.S2

        data_array = preprocess_sentinel2(self.blank_data_array)
        test_data_array(
            self,
            data_array,
            self.shape,
            SatelliteType.S2,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertEqual(data_array.values.min(), 0)
        self.assertEqual(data_array.values.max(), 1)
        self.assertEqual(data_array.shape, self.blank_data_array.shape)

    def test_preprocess_landsat(self):
        """
        Test the preprocess_landsat function.
        """
        self.blank_data_array.values = np.zeros(
            self.blank_data_array.shape
        ) + np.multiply(np.random.rand(*self.blank_data_array.shape), 10)

        self.blank_data_array.attrs["satellite_type"] = SatelliteType.LANDSAT

        data_array = preprocess_landsat(self.blank_data_array)
        test_data_array(
            self,
            data_array,
            self.shape,
            SatelliteType.LANDSAT,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertEqual(data_array.values.min(), 0)
        self.assertEqual(data_array.values.max(), 1)
        self.assertEqual(data_array.shape, self.blank_data_array.shape)

    def test_preprocess_viirs(self):
        """
        Test the preprocess_viirs function.
        """
        self.blank_data_array.values = np.zeros(
            self.blank_data_array.shape
        ) + np.multiply(np.random.rand(*self.blank_data_array.shape), 10)

        self.blank_data_array.attrs["satellite_type"] = SatelliteType.VIIRS

        data_array = preprocess_viirs(self.blank_data_array)
        test_data_array(
            self,
            data_array,
            self.shape,
            SatelliteType.VIIRS,
            self.tile_dir,
            self.parent_tile_dir,
        )

        self.assertEqual(data_array.values.min(), 0)
        self.assertEqual(data_array.values.max(), 1)
        self.assertEqual(data_array.shape, self.blank_data_array.shape)


if __name__ == "__main__":
    unittest.main()
