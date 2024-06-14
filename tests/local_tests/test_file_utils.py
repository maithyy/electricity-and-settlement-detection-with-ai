import sys
import unittest
from pathlib import Path
from datetime import date


import xarray as xr

# local modules
sys.path.append(".")
from src.preprocessing.file_utils import *
from src.utilities import SatelliteType, get_satellite_dataset_size

ROOT = Path.cwd()


# --HELPERS--


def check_data_array_is_valid(
    self: unittest.TestCase,
    data_array: xr.DataArray,
    satellite_type: SatelliteType,
    paths: List[Path],
    shape: tuple,
):
    """
    This function is used as a helper for the test_load_satellite tests
    """
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
    self.assertIsInstance(data_array.attrs["satellite_type"], str)
    self.assertEqual(data_array.attrs["satellite_type"], satellite_type.value)

    # check tile_dir
    self.assertIsInstance(data_array.attrs["tile_dir"], str)
    self.assertEqual(data_array.attrs["tile_dir"], str(self.tile_dir))

    # check parent_tile_id
    self.assertIsInstance(data_array.attrs["parent_tile_id"], str)
    self.assertEqual(data_array.attrs["parent_tile_id"], "Tile1")

    # check access by date
    self.assertEqual(len(data_array["date"]), shape[0])
    # check access by band
    self.assertEqual(len(data_array["band"]), shape[1])
    # ensure dates and bands line up correctly
    for file in paths:
        date_string, band = get_grouping_function(satellite_type)(file)
        self.assertEqual(data_array["date"].sel(date=date_string).values, date_string)
        self.assertEqual(data_array["band"].sel(band=band).values, band)


def check_data_array_is_valid_for_list(
    self: unittest.TestCase,
    data_array: xr.DataArray,
    paths: List[Path],
    shape: tuple,
):
    """
    This function is used as a helper for the test_load_satellite_list tests
    """
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

    # get satellite type
    satellite_type = None
    for type in SatelliteType:
        if type.value == data_array.attrs["satellite_type"]:
            satellite_type = type

    # ensure we found a valid satellite_type
    self.assertIsNotNone(
        satellite_type,
        "The satellite_type attr of the data_array was not a valid SatelliteType",
    )

    # check satellite type
    self.assertIsInstance(data_array.attrs["satellite_type"], str)
    self.assertEqual(data_array.attrs["satellite_type"], satellite_type.value)

    # check tile_dir
    self.assertIsInstance(data_array.attrs["tile_dir"], str)
    self.assertEqual(data_array.attrs["tile_dir"], str(self.tile_dir))

    # check parent_tile_id
    self.assertIsInstance(data_array.attrs["parent_tile_id"], str)
    self.assertEqual(data_array.attrs["parent_tile_id"], "Tile1")

    # check access by date
    self.assertEqual(len(data_array["date"]), shape[0])
    # check access by band
    self.assertEqual(len(data_array["band"]), shape[1])
    # ensure dates and bands line up correctly
    for file in paths:
        date_string, band = get_grouping_function(satellite_type)(file)
        self.assertEqual(data_array["date"].sel(date=date_string).values, date_string)
        self.assertEqual(data_array["band"].sel(band=band).values, band)


# -----------


class TestFilenames(unittest.TestCase):
    def test_viirs_filenames(self):
        func = process_viirs_filename
        alpha = [
            Path("DNB_VNP46A1_A2020221.tif"),
            Path("Train") / "Tile1" / "DNB_VNP46A1_A2020221.tif",
            Path("C:")
            / "Users"
            / "foo"
            / "data"
            / "raw"
            / "Train"
            / "Tile1"
            / "DNB_VNP46A1_A2020221.tif",
        ]
        beta = [
            Path("DNB_VNP46A1_A2020237.tif"),
            Path("Train") / "Tile1" / "DNB_VNP46A1_A2020237.tif",
            Path("C:")
            / "Users"
            / "foo"
            / "data"
            / "raw"
            / "Train"
            / "Tile1"
            / "DNB_VNP46A1_A2020237.tif",
        ]
        alpha_answer, beta_answer = (str(date(2020, 8, 8)), "0"), (
            str(date(2020, 8, 24)),
            "0",
        )

        answer = alpha_answer
        for item in alpha:
            filename = func(item)
            self.assertIsInstance(filename[0], str)
            self.assertIsInstance(filename[1], str)
            self.assertEqual(answer[0], filename[0])
            self.assertEqual(answer[1], filename[1])

        answer = beta_answer
        for item in beta:
            filename = func(item)
            self.assertIsInstance(filename[0], str)
            self.assertIsInstance(filename[1], str)
            self.assertEqual(answer[0], filename[0])
            self.assertEqual(answer[1], filename[1])

    def test_s1_filenames(self):
        func = process_s1_filename
        alpha = [
            Path("S1A_IW_GRDH_20200804_VV.tif"),
            Path("Train") / "Tile1" / "S1A_IW_GRDH_20200804_VV.tif",
            Path("C:")
            / "Users"
            / "foo"
            / "data"
            / "raw"
            / "Train"
            / "Tile1"
            / "S1A_IW_GRDH_20200804_VV.tif",
        ]
        beta = [
            Path("S1A_IW_GRDH_20200723_VH.tif"),
            Path("Train") / "Tile1" / "S1A_IW_GRDH_20200723_VH.tif",
            Path("C:")
            / "Users"
            / "foo"
            / "data"
            / "raw"
            / "Train"
            / "Tile1"
            / "S1A_IW_GRDH_20200723_VH.tif",
        ]
        alpha_answer, beta_answer = (str(date(2020, 8, 4)), "VV"), (
            str(date(2020, 7, 23)),
            "VH",
        )

        answer = alpha_answer
        for item in alpha:
            filename = func(item)
            self.assertIsInstance(filename[0], str)
            self.assertIsInstance(filename[1], str)
            self.assertEqual(answer[0], filename[0])
            self.assertEqual(answer[1], filename[1])

        answer = beta_answer
        for item in beta:
            filename = func(item)
            self.assertIsInstance(filename[0], str)
            self.assertIsInstance(filename[1], str)
            self.assertEqual(answer[0], filename[0])
            self.assertEqual(answer[1], filename[1])

    def test_s2_filenames(self):
        func = process_s2_filename
        alpha = [
            Path("L2A_20200816_B01.tif"),
            Path("Train") / "Tile1" / "L2A_20200816_B01.tif",
            Path("C:")
            / "Users"
            / "foo"
            / "data"
            / "raw"
            / "Train"
            / "Tile1"
            / "L2A_20200816_B01.tif",
        ]
        beta = [
            Path("L2A_20200826_B8A.tif"),
            Path("Train") / "Tile1" / "L2A_20200826_B8A.tif",
            Path("C:")
            / "Users"
            / "foo"
            / "data"
            / "raw"
            / "Train"
            / "Tile1"
            / "L2A_20200826_B8A.tif",
        ]
        alpha_answer, beta_answer = (str(date(2020, 8, 16)), "01"), (
            str(date(2020, 8, 26)),
            "8A",
        )

        answer = alpha_answer
        for item in alpha:
            filename = func(item)
            self.assertIsInstance(filename[0], str)
            self.assertIsInstance(filename[1], str)
            self.assertEqual(answer[0], filename[0])
            self.assertEqual(answer[1], filename[1])

        answer = beta_answer
        for item in beta:
            filename = func(item)
            self.assertIsInstance(filename[0], str)
            self.assertIsInstance(filename[1], str)
            self.assertEqual(answer[0], filename[0])
            self.assertEqual(answer[1], filename[1])

    def test_landsat_filename(self):
        func = process_landsat_filename
        alpha = [
            Path("LC08_L1TP_2020-07-29_B1.tif"),
            Path("Train") / "Tile1" / "LC08_L1TP_2020-07-29_B1.tif",
            Path("C:")
            / "Users"
            / "foo"
            / "data"
            / "raw"
            / "Train"
            / "Tile1"
            / "LC08_L1TP_2020-07-29_B1.tif",
        ]
        beta = [
            Path("LC08_L1TP_2020-08-30_B9.tif"),
            Path("Train") / "Tile1" / "LC08_L1TP_2020-08-30_B9.tif",
            Path("C:")
            / "Users"
            / "foo"
            / "data"
            / "raw"
            / "Train"
            / "Tile1"
            / "LC08_L1TP_2020-08-30_B9.tif",
        ]
        alpha_answer, beta_answer = (str(date(2020, 7, 29)), "1"), (
            str(date(2020, 8, 30)),
            "9",
        )

        answer = alpha_answer
        for item in alpha:
            filename = func(item)
            self.assertIsInstance(filename[0], str)
            self.assertIsInstance(filename[1], str)
            self.assertEqual(answer[0], filename[0])
            self.assertEqual(answer[1], filename[1])

        answer = beta_answer
        for item in beta:
            filename = func(item)
            self.assertIsInstance(filename[0], str)
            self.assertIsInstance(filename[1], str)
            self.assertEqual(answer[0], filename[0])
            self.assertEqual(answer[1], filename[1])

    def test_gt_filename(self):
        for x in range(100):
            self.assertEqual(
                process_ground_truth_filename(str(x)), (str(date(1, 1, 1)), "0")
            )


class TestInfoRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.train_dir = ROOT / "data" / "raw" / "Train"
        self.tile_dir = ROOT / "data" / "raw" / "Train" / "Tile1"
        self.falsePath = ROOT / "tests"

    def test_get_filename_pattern(self):
        patterns = {
            SatelliteType.VIIRS: "DNB_VNP46A1_",
            SatelliteType.S1: "S1A_IW_GRDH_",
            SatelliteType.S2: "L2A_",
            SatelliteType.LANDSAT: "LC08_L1TP_",
            SatelliteType.GT: "groundTruth.tif",
        }
        [
            self.assertEqual(get_filename_pattern(satellite_type), pattern)
            for satellite_type, pattern in patterns.items()
        ]

    def test_get_viirs_files(self):
        satellite_type, correct_amount = SatelliteType.VIIRS, 9

        self.assertEqual(len(get_satellite_files(self.falsePath, satellite_type)), 0)

        files = get_satellite_files(self.tile_dir, satellite_type)
        self.assertEqual(len(files), correct_amount)
        for file in files:
            self.assertIsInstance(file, Path)

    def test_get_s1_files(self):
        satellite_type, correct_amount = SatelliteType.S1, 8

        self.assertEqual(len(get_satellite_files(self.falsePath, satellite_type)), 0)

        files = get_satellite_files(self.tile_dir, satellite_type)
        self.assertEqual(len(files), correct_amount)
        for file in files:
            self.assertIsInstance(file, Path)

    def test_get_s1_files(self):
        satellite_type, correct_amount = SatelliteType.S2, 48

        self.assertEqual(len(get_satellite_files(self.falsePath, satellite_type)), 0)

        files = get_satellite_files(self.tile_dir, satellite_type)
        self.assertEqual(len(files), correct_amount)
        for file in files:
            self.assertIsInstance(file, Path)

    def test_get_l8_files(self):
        satellite_type, correct_amount = SatelliteType.LANDSAT, 33

        self.assertEqual(len(get_satellite_files(self.falsePath, satellite_type)), 0)

        files = get_satellite_files(self.tile_dir, satellite_type)
        self.assertEqual(len(files), correct_amount)
        for file in files:
            self.assertIsInstance(file, Path)

    def test_get_gt_files(self):
        satellite_type, correct_amount = SatelliteType.GT, 1

        self.assertEqual(len(get_satellite_files(self.falsePath, satellite_type)), 0)

        files = get_satellite_files(self.tile_dir, satellite_type)
        self.assertEqual(len(files), correct_amount)
        for file in files:
            self.assertIsInstance(file, Path)

    def test_get_grouping_function(self):
        self.assertEqual(
            get_grouping_function(SatelliteType.VIIRS), process_viirs_filename
        )
        self.assertEqual(get_grouping_function(SatelliteType.S1), process_s1_filename)
        self.assertEqual(get_grouping_function(SatelliteType.S2), process_s2_filename)
        self.assertEqual(
            get_grouping_function(SatelliteType.LANDSAT), process_landsat_filename
        )
        self.assertEqual(
            get_grouping_function(SatelliteType.GT), process_ground_truth_filename
        )

    def test_get_unique_dates_and_bands(self):
        alpha = (
            [
                str(date(2020, 7, 23)),
                str(date(2020, 8, 4)),
                str(date(2020, 8, 16)),
                str(date(2020, 8, 28)),
            ],
            ["VH", "VV"],
        )
        beta = (
            [str(date(2020, 7, 29)), str(date(2020, 8, 14)), str(date(2020, 8, 30))],
            ["1", "10", "11", "2", "3", "4", "5", "6", "7", "8", "9"],
        )

        answer = alpha
        dates, bands = get_unique_dates_and_bands(self.tile_dir, SatelliteType.S1)
        self.assertEqual(len(answer[0]), len(dates))
        self.assertEqual(len(answer[1]), len(bands))
        for index, date_string in enumerate(dates):
            self.assertEqual(date_string, answer[0][index])
        for index, band in enumerate(bands):
            self.assertEqual(band, answer[1][index])

        answer = beta
        dates, bands = get_unique_dates_and_bands(self.tile_dir, SatelliteType.LANDSAT)
        self.assertEqual(len(answer[0]), len(dates))
        self.assertEqual(len(answer[1]), len(bands))
        for index, date_string in enumerate(dates):
            self.assertEqual(date_string, answer[0][index])
        for index, band in enumerate(bands):
            self.assertEqual(band, answer[1][index])


class TestDataProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.train_dir = ROOT / "data" / "raw" / "Train"
        self.tile_dir = ROOT / "data" / "raw" / "Train" / "Tile1"
        self.falsePath = ROOT / "tests"

        self.viirs_paths = [
            self.tile_dir / "DNB_VNP46A1_A2020221.tif",
            self.tile_dir / "DNB_VNP46A1_A2020224.tif",
            self.tile_dir / "DNB_VNP46A1_A2020225.tif",
            self.tile_dir / "DNB_VNP46A1_A2020226.tif",
            self.tile_dir / "DNB_VNP46A1_A2020227.tif",
            self.tile_dir / "DNB_VNP46A1_A2020231.tif",
            self.tile_dir / "DNB_VNP46A1_A2020235.tif",
            self.tile_dir / "DNB_VNP46A1_A2020236.tif",
            self.tile_dir / "DNB_VNP46A1_A2020237.tif",
        ]
        self.s1_paths = [
            self.tile_dir / "S1A_IW_GRDH_20200723_VH.tif",
            self.tile_dir / "S1A_IW_GRDH_20200723_VV.tif",
            self.tile_dir / "S1A_IW_GRDH_20200804_VH.tif",
            self.tile_dir / "S1A_IW_GRDH_20200804_VV.tif",
            self.tile_dir / "S1A_IW_GRDH_20200816_VH.tif",
            self.tile_dir / "S1A_IW_GRDH_20200816_VV.tif",
            self.tile_dir / "S1A_IW_GRDH_20200828_VH.tif",
            self.tile_dir / "S1A_IW_GRDH_20200828_VV.tif",
        ]
        self.s2_paths = [
            self.tile_dir / "L2A_20200811_B01.tif",
            self.tile_dir / "L2A_20200811_B02.tif",
            self.tile_dir / "L2A_20200811_B03.tif",
            self.tile_dir / "L2A_20200811_B04.tif",
            self.tile_dir / "L2A_20200811_B05.tif",
            self.tile_dir / "L2A_20200811_B06.tif",
            self.tile_dir / "L2A_20200811_B07.tif",
            self.tile_dir / "L2A_20200811_B08.tif",
            self.tile_dir / "L2A_20200811_B09.tif",
            self.tile_dir / "L2A_20200811_B11.tif",
            self.tile_dir / "L2A_20200811_B12.tif",
            self.tile_dir / "L2A_20200811_B8A.tif",
            self.tile_dir / "L2A_20200816_B01.tif",
            self.tile_dir / "L2A_20200816_B02.tif",
            self.tile_dir / "L2A_20200816_B03.tif",
            self.tile_dir / "L2A_20200816_B04.tif",
            self.tile_dir / "L2A_20200816_B05.tif",
            self.tile_dir / "L2A_20200816_B06.tif",
            self.tile_dir / "L2A_20200816_B07.tif",
            self.tile_dir / "L2A_20200816_B08.tif",
            self.tile_dir / "L2A_20200816_B09.tif",
            self.tile_dir / "L2A_20200816_B11.tif",
            self.tile_dir / "L2A_20200816_B12.tif",
            self.tile_dir / "L2A_20200816_B8A.tif",
            self.tile_dir / "L2A_20200826_B01.tif",
            self.tile_dir / "L2A_20200826_B02.tif",
            self.tile_dir / "L2A_20200826_B03.tif",
            self.tile_dir / "L2A_20200826_B04.tif",
            self.tile_dir / "L2A_20200826_B05.tif",
            self.tile_dir / "L2A_20200826_B06.tif",
            self.tile_dir / "L2A_20200826_B07.tif",
            self.tile_dir / "L2A_20200826_B08.tif",
            self.tile_dir / "L2A_20200826_B09.tif",
            self.tile_dir / "L2A_20200826_B11.tif",
            self.tile_dir / "L2A_20200826_B12.tif",
            self.tile_dir / "L2A_20200826_B8A.tif",
            self.tile_dir / "L2A_20200831_B01.tif",
            self.tile_dir / "L2A_20200831_B02.tif",
            self.tile_dir / "L2A_20200831_B03.tif",
            self.tile_dir / "L2A_20200831_B04.tif",
            self.tile_dir / "L2A_20200831_B05.tif",
            self.tile_dir / "L2A_20200831_B06.tif",
            self.tile_dir / "L2A_20200831_B07.tif",
            self.tile_dir / "L2A_20200831_B08.tif",
            self.tile_dir / "L2A_20200831_B09.tif",
            self.tile_dir / "L2A_20200831_B11.tif",
            self.tile_dir / "L2A_20200831_B12.tif",
            self.tile_dir / "L2A_20200831_B8A.tif",
        ]
        self.landsat_paths = [
            self.tile_dir / "LC08_L1TP_2020-07-29_B1.tif",
            self.tile_dir / "LC08_L1TP_2020-07-29_B10.tif",
            self.tile_dir / "LC08_L1TP_2020-07-29_B11.tif",
            self.tile_dir / "LC08_L1TP_2020-07-29_B2.tif",
            self.tile_dir / "LC08_L1TP_2020-07-29_B3.tif",
            self.tile_dir / "LC08_L1TP_2020-07-29_B4.tif",
            self.tile_dir / "LC08_L1TP_2020-07-29_B5.tif",
            self.tile_dir / "LC08_L1TP_2020-07-29_B6.tif",
            self.tile_dir / "LC08_L1TP_2020-07-29_B7.tif",
            self.tile_dir / "LC08_L1TP_2020-07-29_B8.tif",
            self.tile_dir / "LC08_L1TP_2020-07-29_B9.tif",
            self.tile_dir / "LC08_L1TP_2020-08-14_B1.tif",
            self.tile_dir / "LC08_L1TP_2020-08-14_B10.tif",
            self.tile_dir / "LC08_L1TP_2020-08-14_B11.tif",
            self.tile_dir / "LC08_L1TP_2020-08-14_B2.tif",
            self.tile_dir / "LC08_L1TP_2020-08-14_B3.tif",
            self.tile_dir / "LC08_L1TP_2020-08-14_B4.tif",
            self.tile_dir / "LC08_L1TP_2020-08-14_B5.tif",
            self.tile_dir / "LC08_L1TP_2020-08-14_B6.tif",
            self.tile_dir / "LC08_L1TP_2020-08-14_B7.tif",
            self.tile_dir / "LC08_L1TP_2020-08-14_B8.tif",
            self.tile_dir / "LC08_L1TP_2020-08-14_B9.tif",
            self.tile_dir / "LC08_L1TP_2020-08-30_B1.tif",
            self.tile_dir / "LC08_L1TP_2020-08-30_B10.tif",
            self.tile_dir / "LC08_L1TP_2020-08-30_B11.tif",
            self.tile_dir / "LC08_L1TP_2020-08-30_B2.tif",
            self.tile_dir / "LC08_L1TP_2020-08-30_B3.tif",
            self.tile_dir / "LC08_L1TP_2020-08-30_B4.tif",
            self.tile_dir / "LC08_L1TP_2020-08-30_B5.tif",
            self.tile_dir / "LC08_L1TP_2020-08-30_B6.tif",
            self.tile_dir / "LC08_L1TP_2020-08-30_B7.tif",
            self.tile_dir / "LC08_L1TP_2020-08-30_B8.tif",
            self.tile_dir / "LC08_L1TP_2020-08-30_B9.tif",
        ]

    def test_get_parent_tile_id(self):
        for x in range(100):
            parent_tile_id = get_parent_tile_id(self.train_dir / f"Tile{x}")
            self.assertIsInstance(parent_tile_id, str)
            self.assertEqual(parent_tile_id, f"Tile{x}")

    def test_read_satellite_file(self):
        paths = [
            self.s1_paths,
            self.viirs_paths,
        ]
        for file in self.viirs_paths:
            data = read_satellite_file(file)

            self.assertIsInstance(data, np.ndarray)
            self.assertEqual(len(data.shape), 2)
            self.assertEqual((data.shape)[0], 800)
            self.assertEqual((data.shape)[1], 800)

    def test_load_satellite(self):
        satellite_type = SatelliteType.VIIRS
        data_array = load_satellite(self.tile_dir, satellite_type)
        check_data_array_is_valid(
            self, data_array, satellite_type, self.viirs_paths, (9, 1, 800, 800)
        )

        satellite_type = SatelliteType.S1
        data_array = load_satellite(self.tile_dir, satellite_type)
        check_data_array_is_valid(
            self, data_array, satellite_type, self.s1_paths, (4, 2, 800, 800)
        )

        satellite_type = SatelliteType.S2
        data_array = load_satellite(self.tile_dir, satellite_type)
        check_data_array_is_valid(
            self, data_array, satellite_type, self.s2_paths, (4, 12, 800, 800)
        )

        satellite_type = SatelliteType.LANDSAT
        data_array = load_satellite(self.tile_dir, satellite_type)
        check_data_array_is_valid(
            self, data_array, satellite_type, self.landsat_paths, (3, 11, 800, 800)
        )

    def test_load_satellite_list_one(self):
        satellite_type_list = [SatelliteType.VIIRS]
        satellite_list = load_satellite_list(self.tile_dir, satellite_type_list)
        self.assertIsInstance(satellite_list, list)
        self.assertEqual(len(satellite_list), 1)

        data_array = satellite_list[0]
        check_data_array_is_valid_for_list(
            self, data_array, self.viirs_paths, (9, 1, 800, 800)
        )

        satellite_type_list = [SatelliteType.S1]
        satellite_list = load_satellite_list(self.tile_dir, satellite_type_list)
        self.assertIsInstance(satellite_list, list)
        self.assertEqual(len(satellite_list), 1)

        data_array = satellite_list[0]
        check_data_array_is_valid_for_list(
            self, data_array, self.s1_paths, (4, 2, 800, 800)
        )

        satellite_type_list = [SatelliteType.S2]
        satellite_list = load_satellite_list(self.tile_dir, satellite_type_list)
        self.assertIsInstance(satellite_list, list)
        self.assertEqual(len(satellite_list), 1)

        data_array = satellite_list[0]
        check_data_array_is_valid_for_list(
            self, data_array, self.s2_paths, (4, 12, 800, 800)
        )

        satellite_type_list = [SatelliteType.LANDSAT]
        satellite_list = load_satellite_list(self.tile_dir, satellite_type_list)
        self.assertIsInstance(satellite_list, list)
        self.assertEqual(len(satellite_list), 1)

        data_array = satellite_list[0]
        check_data_array_is_valid_for_list(
            self, data_array, self.landsat_paths, (3, 11, 800, 800)
        )

    def test_load_satellite_list_many(self):
        # test a subset of types
        paths = [
            self.s1_paths,
            self.viirs_paths,
            self.landsat_paths,
        ]
        satellite_type_list = [
            SatelliteType.S1,
            SatelliteType.VIIRS,
            SatelliteType.LANDSAT,
        ]
        shapes = [(4, 2, 800, 800), (9, 1, 800, 800), (3, 11, 800, 800)]
        satellite_list = load_satellite_list(self.tile_dir, satellite_type_list)
        self.assertIsInstance(satellite_list, list)
        self.assertEqual(len(satellite_list), 3)

        for index in range(len(satellite_type_list)):
            data_array = satellite_list[index]
            check_data_array_is_valid_for_list(
                self, data_array, paths[index], shapes[index]
            )

        # test all types
        paths = [self.viirs_paths, self.s1_paths, self.s2_paths, self.landsat_paths]
        satellite_type_list = [
            SatelliteType.VIIRS,
            SatelliteType.S1,
            SatelliteType.S2,
            SatelliteType.LANDSAT,
        ]
        shapes = [
            (9, 1, 800, 800),
            (4, 2, 800, 800),
            (4, 12, 800, 800),
            (3, 11, 800, 800),
        ]
        satellite_list = load_satellite_list(self.tile_dir, satellite_type_list)
        self.assertIsInstance(satellite_list, list)
        self.assertEqual(len(satellite_list), 4)

        for index in range(len(satellite_type_list)):
            data_array = satellite_list[index]
            check_data_array_is_valid_for_list(
                self, data_array, paths[index], shapes[index]
            )

    def test_load_satellite_dir(self):
        """
        This test has been disabled by default because it loads
        and tests all the data possible. To ensure you did everything right,
        you can change the false below to a true, but be warned it will take a while
        to run.
        """
        if False:
            # test a subset of types
            paths = [
                self.s1_paths,
                self.viirs_paths,
                self.landsat_paths,
            ]
            satellite_type_list = [
                SatelliteType.S1,
                SatelliteType.VIIRS,
                SatelliteType.LANDSAT,
            ]
            shapes = [(4, 2, 800, 800), (9, 1, 800, 800), (3, 11, 800, 800)]
            list_of_data_array_list = load_satellite_dir(
                self.train_dir, satellite_type_list
            )
            self.assertIsInstance(list_of_data_array_list, list)
            self.assertEqual(len(list_of_data_array_list), 60)

            for satellite_list in list_of_data_array_list:
                self.assertIsInstance(satellite_list, list)
                self.assertEqual(len(satellite_list), 3)

                satellite_list = load_satellite_list(self.tile_dir, satellite_type_list)
                for index in range(len(satellite_type_list)):
                    data_array = satellite_list[index]
                    check_data_array_is_valid_for_list(
                        self, data_array, paths[index], shapes[index]
                    )

            # test all types
            paths = [self.viirs_paths, self.s1_paths, self.s2_paths, self.landsat_paths]
            satellite_type_list = [
                SatelliteType.VIIRS,
                SatelliteType.S1,
                SatelliteType.S2,
                SatelliteType.LANDSAT,
            ]
            shapes = [
                (9, 1, 800, 800),
                (4, 2, 800, 800),
                (4, 12, 800, 800),
                (3, 11, 800, 800),
            ]
            list_of_data_array_list = load_satellite_dir(
                self.train_dir, satellite_type_list
            )
            self.assertIsInstance(list_of_data_array_list, list)
            self.assertEqual(len(list_of_data_array_list), 60)

            for satellite_list in list_of_data_array_list:
                self.assertIsInstance(satellite_list, list)
                self.assertEqual(len(satellite_list), 4)

                satellite_list = load_satellite_list(self.tile_dir, satellite_type_list)
                for index in range(len(satellite_type_list)):
                    data_array = satellite_list[index]
                    check_data_array_is_valid_for_list(
                        self, data_array, paths[index], shapes[index]
                    )

    def test_create_satellite_dataset(self):
        satellite_type_list = [
            SatelliteType.VIIRS,
            SatelliteType.S1,
            SatelliteType.S2,
            SatelliteType.LANDSAT,
        ]
        shapes = [
            (9, 1, 800, 800),
            (4, 2, 800, 800),
            (4, 12, 800, 800),
            (3, 11, 800, 800),
        ]
        list_of_data_array_list = load_satellite_dir(
            self.train_dir, satellite_type_list
        )
        self.assertIsInstance(list_of_data_array_list, list)
        self.assertEqual(len(list_of_data_array_list), 60)

        data_set_list = create_satellite_dataset_list(
            list_of_data_array_list, satellite_type_list
        )

        self.assertIsInstance(data_set_list, list)
        self.assertEqual(len(data_set_list), 4)

        for index, data_set in enumerate(data_set_list):
            self.assertIsInstance(data_set, xr.Dataset)
            self.assertEqual(
                data_set.attrs["satellite_type"], (satellite_type_list[index]).value
            )
            self.assertEqual(get_satellite_dataset_size(data_set), shapes[index])
            self.assertEqual(len(data_set), 60)

            for x in range(1, 61):
                self.assertEqual(data_set[f"Tile{x}"].shape, shapes[index])


if __name__ == "__main__":
    unittest.main()
