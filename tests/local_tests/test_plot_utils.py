""" Tests for the preprocessing utilities. """

import sys
import unittest
import numpy as np
from pathlib import Path

# import local modules
sys.path.append(".")
from src.visualization.plot_utils import *
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
    Class to test the plot utilities.
    """

    @classmethod
    def setUpClass(self):
        np.random.seed(123456789)
        self.shape = (4, 3, 800, 800)

    def setUp(self):
        data_dict = dict()
        for x in range(10):
            data_array = xr.DataArray(
                data=np.ones(shape=self.shape),
                dims=("date", "band", "height", "width"),
            )
            data_dict[f"Tile{x}"] = data_array
        self.blank_data_set = xr.Dataset(data_dict)

    def test_flatten_dataset(self):
        flattened_data_set = flatten_dataset(self.blank_data_set)

        self.assertIsInstance(flattened_data_set, np.ndarray)
        # new shape should be (tile * date * band * height * width)
        self.assertEqual(flattened_data_set.shape, ((10 * 4 * 3 * 800 * 800),))
        self.assertEqual(flattened_data_set.min(), flattened_data_set.max())

    def test_flatten_dataset_by_band(self):
        flattened_data_set = flatten_dataset_by_band(self.blank_data_set)

        self.assertIsInstance(flattened_data_set, np.ndarray)
        # new shape should be (band, tile * date * height * width)
        self.assertEqual(
            flattened_data_set.shape,
            (3, (10 * 4 * 800 * 800)),
        )
        self.assertEqual(flattened_data_set.min(), flattened_data_set.max())


if __name__ == "__main__":
    unittest.main()
