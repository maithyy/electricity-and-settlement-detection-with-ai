from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union

import xarray as xr


# given
class SatelliteType(Enum):
    VIIRS_MAX_PROJ = "viirs_max_projection"
    VIIRS = "viirs"
    S1 = "sentinel1"
    S2 = "sentinel2"
    LANDSAT = "landsat"
    GT = "gt"
    GT_COMBINED = "gt_combined"


ROOT = Path.cwd()
PROJ_NAME = "CS175-spring-2024"
MODEL = "RandomForestClassifier"  # valid values are ["UNet", "SegmentationCNN", "FCNResnetTransfer", "RandomForestClassifier"]

@dataclass
class ESDConfig:
    processed_dir: Path = ROOT / "data" / "processed"
    raw_dir: Path = ROOT / "data" / "raw" / "Train"
    results_dir: Path = ROOT / "data" / "predictions" / MODEL
    intermediate_tiffs_dir: Path = ROOT / "data" / "intermediate_tiffs" / MODEL
    final_tiffs_dir: Path = ROOT / "data" / "final_tiffs" / MODEL
    viirs_dir: Path = ROOT / "data" / "processed" / "Train" / "subtiles"
    
    '''
    Selected Bands for the Settlement Detection Task:

        Sentinel 1:
            VH Band: differentiate between urban areas and natural terrain

        Sentinel 2:
            Visible Bands (2,3,4): blue, green, red
            Near-Infrared Band (8): helpful for differentiating between vegetation and man-made surfaces
            Shortwave Infrared Bands (11,12): identifying materials like concrete and asphalt

        Landsat 8:
            Visible Bands (2,3,4): blue, green, red
            Near-Infrared Band (5): healthy plants reflect it (lower value for man-made surfaces)
            Thermal Infrared (10,11): heat retention from human activity and infrastructure

    '''
    selected_bands = {
        # SatelliteType.VIIRS: ["0"],
        SatelliteType.S1: ["VH"], #["VV", "VH"],
        SatelliteType.S2: [
            "12",
            "11",
            # "09",
            # "8A",
            "08",
            # "07",
            # "06",
            # "05",
            "04",
            "03",
            "02",
            # "01",
        ],
        SatelliteType.LANDSAT: [
            "11",
            "10",
        #     "9",
        #     "8",
        #     "7",
        #     "6",
            "5",
            "4",
            "3",
            "2",
        #     "1",
        ],
        # SatelliteType.VIIRS_MAX_PROJ: ["0"],
    }

    accelerator: str = "cpu"
    batch_size: int = 2
    depth: int = 2
    devices: int = 1
    embedding_size: int = 64
    in_channels: int = 46  # num_dates * num_bands (99 for all bands)
    kernel_size: int = 3
    learning_rate: float = 0.002
    max_epochs: int = 1
    model_path: Path = ROOT / "models" / MODEL / "last.ckpt"
    model_type: str = MODEL
    n_encoders: int = 4
    num_workers: int = 8
    out_channels: int = 2  # 2 channels for binary classification task, 4 for original task
    pool_sizes: str = "5,5,2"
    seed: int = 12378921
    slice_size: tuple = (4,4)
    wandb_run_name: Union[str, None] = None

    # random forest classifier parameters
    bootstrap: bool = True
    criterion: str = "gini"
    fit_frequency: int = 50
    max_depth: Union[int, None] = None
    min_samples_leaf: int = 2
    min_samples_split: int = 3
    n_estimators: int = 150


# given
def get_satellite_dataset_size(
    data_set: xr.Dataset, dims: List[str] = ["date", "band", "height", "width"]
):
    """
    Gets the shape of a dataset

    Parameters
    ----------
    data_set : xr.Dataset
        A satellite dataset
    dims: List[str]
        A list of dimensions of the data_set data_arrays
    Returns
    -------
    Tuple:
        Shape of the data_set, default is (date, band, height, width)
    """
    return tuple(data_set.sizes[d] for d in dims)
