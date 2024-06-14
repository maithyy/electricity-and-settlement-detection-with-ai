import sys
from argparse import ArgumentParser
from pathlib import Path
import math
import matplotlib.pyplot as plt

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig
from src.visualization.restitch_plot import restitch_and_plot, restitch_and_save_to_tiff
from src.preprocessing.file_utils import get_parent_tile_id, load_satellite_dir
import tifffile as tiff
from src.preprocessing.file_utils import load_satellite
from src.utilities import SatelliteType
from src.preprocessing.preprocess_sat import maxprojection_viirs
import numpy as np

HAS_ELECTRICITY = 3
HAS_NO_ELECTRICITY = 1

def print_chunk(chunk):
    for row in chunk:
        print(row)

def plot_histogram(ax, title, x_label, y_label, data=[], bins=[]):
    ax.hist(data, bins=bins, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def calculate_bins(data1, data2):
    combined_data = np.concatenate([data1, data2])
    max_value = np.percentile(combined_data, 98) # ignore right end 2% of outliers
    bins = np.linspace(min(combined_data), max_value, num=30)
    return bins

def plot_data_analysis(options):
    all_gt_arrays = load_satellite_dir(options.raw_dir, [SatelliteType.GT])
    downsampling_size = np.prod([int(x) for x in options.pool_sizes.split(",")])
    
    electricity_data = {"mean": [], "std": [], "max": []}
    no_electricity_data = {"mean": [], "std": [], "max": []}

    for gt_arr in all_gt_arrays:
        gt_arr = gt_arr[0][0][0]

        raw_viirs_path = "data/raw/Train/"+gt_arr.attrs['parent_tile_id']
        raw_viirs_file = load_satellite(Path(raw_viirs_path), SatelliteType.VIIRS)
        max_projection_viirs_arr = maxprojection_viirs(raw_viirs_file).values[0][0]

        gt_dim_1, gt_dim_2 = gt_arr.shape

        for i in range(gt_dim_1):
            for j in range(gt_dim_2):
                chunk_gt = gt_arr[i][j]

                start_i = downsampling_size*i
                end_i = downsampling_size*(i+1)
                start_j = downsampling_size*j
                end_j = downsampling_size*(j+1)

                max_viirs_chunk = max_projection_viirs_arr[start_i:end_i, start_j:end_j]

                mean_brightness = np.mean(max_viirs_chunk)
                std_deviation = np.std(max_viirs_chunk)
                brightness_max = np.max(max_viirs_chunk)

                if chunk_gt == HAS_NO_ELECTRICITY:
                    no_electricity_data['mean'].append(mean_brightness)
                    no_electricity_data['std'].append(std_deviation)
                    no_electricity_data['max'].append(brightness_max)
                elif chunk_gt == HAS_ELECTRICITY:
                    electricity_data['mean'].append(mean_brightness)
                    electricity_data['std'].append(std_deviation)
                    electricity_data['max'].append(brightness_max)
    
    bins_mean = calculate_bins(no_electricity_data['mean'], electricity_data['mean'])
    bins_std = calculate_bins(no_electricity_data['std'], electricity_data['std'])
    bins_max = calculate_bins(no_electricity_data['max'], electricity_data['max'])

    # Mean
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    plot_histogram(axs[0, 0], "No Electricity", "Pixel Brightness Mean", "Frequency", no_electricity_data['mean'], bins_mean)
    plot_histogram(axs[0, 1], "Electricity", "Pixel Brightness Mean", "Frequency", electricity_data['mean'], bins_mean)

    # Standard Deviation
    plot_histogram(axs[1, 0], "No Electricity", "Pixel Brightness Standard Deviation", "Frequency", no_electricity_data['std'], bins_std)
    plot_histogram(axs[1, 1], "Electricity", "Pixel Brightness Standard Deviation", "Frequency", electricity_data['std'], bins_std)

    # Brightest Pixel
    plot_histogram(axs[2, 0], "No Electricity", "Pixel Brightness Max", "Frequency", no_electricity_data['max'], bins_max)
    plot_histogram(axs[2, 1], "Electricity", "Pixel Brightness Max", "Frequency", electricity_data['max'], bins_max)

    plt.tight_layout()
    plt.show()

def main(options):
    # selecting threshold algorithm

    all_gt_arrays = load_satellite_dir(options.raw_dir, [SatelliteType.GT])
    downsampling_size = np.prod([int(x) for x in options.pool_sizes.split(",")])

    accuracy_results = {
        12.0: {"no_electricity": [], "electricity": [], "combined": []},
        12.5: {"no_electricity": [], "electricity": [], "combined": []},
        13.0: {"no_electricity": [], "electricity": [], "combined": []},
        13.5: {"no_electricity": [], "electricity": [], "combined": []},
        14.0: {"no_electricity": [], "electricity": [], "combined": []},
        14.5: {"no_electricity": [], "electricity": [], "combined": []},
        15.0: {"no_electricity": [], "electricity": [], "combined": []},
        15.5: {"no_electricity": [], "electricity": [], "combined": []},
    
    }
    pixel_brightness_thresholds = np.arange(12, 16, 0.5)

    print(pixel_brightness_thresholds)

    brightness_percentage_thresholds = [x/20 for x in range(1, 20)]

    for pixel_brightness_threshold in pixel_brightness_thresholds:
        for brightness_percentage_threshold in brightness_percentage_thresholds:
            # Note:
            # settlement with electricity chunks are very capable of having dim pixels
            # although settlement without electricity chunks are much less capable of having bright pixels

            # 50x50 pixels in a chunk
            # 16x16 chunks in a tile

            electricity_data = {
                "classified_correctly": [], # so far not using classified correctly/incorrectly (can store/use data later if needed)
                "classified_incorrectly": [], 
                "num_bright_pixels": [], 
                "num_dim_pixels": [], 
                "bright_pixel_percentage": []
            }
            no_electricity_data = {
                "classified_correctly": [],
                "classified_incorrectly": [], 
                "num_bright_pixels": [], 
                "num_dim_pixels": [], 
                "bright_pixel_percentage": []
            }

            num_classified_correctly_electricity = 0
            num_classified_correctly_no_electricity = 0

            for gt_arr in all_gt_arrays:
                gt_arr = gt_arr[0][0][0]

                raw_viirs_path = "data/raw/Train/"+gt_arr.attrs['parent_tile_id']
                raw_viirs_file = load_satellite(Path(raw_viirs_path), SatelliteType.VIIRS)
                max_projection_viirs_arr = maxprojection_viirs(raw_viirs_file).values[0][0]

                gt_dim_1, gt_dim_2 = gt_arr.shape

                for i in range(gt_dim_1):
                    for j in range(gt_dim_2):
                        chunk_gt = gt_arr[i][j]
                        
                        start_i = downsampling_size*i
                        end_i = downsampling_size*(i+1)
                        start_j = downsampling_size*j
                        end_j = downsampling_size*(j+1)

                        max_viirs_chunk = max_projection_viirs_arr[start_i:end_i, start_j:end_j]
                        
                        num_bright_pixels = (max_viirs_chunk >= pixel_brightness_threshold).sum()
                        num_dim_pixels = (max_viirs_chunk < pixel_brightness_threshold).sum()
                        bright_pixel_percentage = num_bright_pixels/(num_dim_pixels+num_bright_pixels)

                        if chunk_gt == HAS_NO_ELECTRICITY:
                            no_electricity_data["num_bright_pixels"].append(num_bright_pixels)
                            no_electricity_data["num_dim_pixels"].append(num_dim_pixels)
                            no_electricity_data["bright_pixel_percentage"].append(bright_pixel_percentage)
                        elif chunk_gt == HAS_ELECTRICITY:
                            electricity_data["num_bright_pixels"].append(num_bright_pixels)
                            electricity_data["num_dim_pixels"].append(num_dim_pixels)
                            electricity_data["bright_pixel_percentage"].append(bright_pixel_percentage)

            electricity_data["bright_pixel_percentage"] = np.array(electricity_data["bright_pixel_percentage"])
            no_electricity_data["bright_pixel_percentage"] = np.array(no_electricity_data["bright_pixel_percentage"])

            no_electricity_data["num_chunks"] = (no_electricity_data["bright_pixel_percentage"]).size
            electricity_data["num_chunks"] = (electricity_data["bright_pixel_percentage"]).size

            no_electricity_data["total_num_bright_pixels"] = sum(no_electricity_data["num_bright_pixels"])
            electricity_data["total_num_bright_pixels"] = sum(electricity_data["num_bright_pixels"])

            no_electricity_data["total_num_dim_pixels"] = sum(no_electricity_data["num_dim_pixels"])
            electricity_data["total_num_dim_pixels"] = sum(electricity_data["num_dim_pixels"])
            
            no_electricity_majority_dim = (no_electricity_data["bright_pixel_percentage"] < 0.5).sum()
            electricity_majority_bright = (electricity_data["bright_pixel_percentage"] >= 0.5).sum()

            num_classified_correctly_no_electricity = (no_electricity_data["bright_pixel_percentage"] < brightness_percentage_threshold).sum()
            num_classified_correctly_electricity = (electricity_data["bright_pixel_percentage"] >= brightness_percentage_threshold).sum()

            accuracy_no_electricity = num_classified_correctly_no_electricity/no_electricity_data["num_chunks"]
            accuracy_electricity = num_classified_correctly_electricity/electricity_data["num_chunks"]

            majority_accuracy_no_electricity = no_electricity_majority_dim/no_electricity_data["num_chunks"]
            majority_accuracy_electricity = electricity_majority_bright/electricity_data["num_chunks"]

            accuracy_results[pixel_brightness_threshold]["no_electricity"].append(accuracy_no_electricity)
            accuracy_results[pixel_brightness_threshold]["electricity"].append(accuracy_electricity)

            num_accurate = accuracy_no_electricity*no_electricity_data["num_chunks"]+accuracy_electricity*electricity_data["num_chunks"]
            combined_accuracy = num_accurate/(no_electricity_data["num_chunks"]+electricity_data["num_chunks"])

            accuracy_results[pixel_brightness_threshold]["combined"].append(combined_accuracy)

            print()
            print('Pixel brightness threshold:', pixel_brightness_threshold)
            print('Brightness percentage threshold:', brightness_percentage_threshold)
            if combined_accuracy > 0.7:
                print()
                print('Settlements with electricity')
                print('Accuracy using threshold:', accuracy_electricity)
                print('Accuracy using majority:', majority_accuracy_electricity)

                print()

                print('Settlements with no electricity')
                print('Accuracy using threshold:', accuracy_no_electricity)
                print('Accuracy using majority:', majority_accuracy_no_electricity)
                print()
                print('Total accuracy', combined_accuracy)
            else:
                print('bad set of parameters')

   
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))

    for idx, pixel_brightness_threshold in enumerate(pixel_brightness_thresholds):
        row = idx // 3
        col = idx % 3
        axs[row, col].plot(brightness_percentage_thresholds, accuracy_results[pixel_brightness_threshold]["combined"], marker='o', linestyle='-', color='g', label='Combined')
        axs[row, col].plot(brightness_percentage_thresholds, accuracy_results[pixel_brightness_threshold]["no_electricity"], marker='o', linestyle='-', color='r', label='No Electricity')
        axs[row, col].plot(brightness_percentage_thresholds, accuracy_results[pixel_brightness_threshold]["electricity"], marker='o', linestyle='-', color='b', label='Electricity')
        axs[row, col].set_title(f'Pixel Brightness Threshold {pixel_brightness_threshold}')
        axs[row, col].set_xlabel('Brightness Percentage Threshold')
        axs[row, col].set_ylabel('Accuracy')
        axs[row, col].set_ylim(0, 1)
        axs[row, col].legend()

    for idx in range(len(pixel_brightness_thresholds), 9):
        fig.delaxes(axs.flat[idx])

    plt.tight_layout()
    plt.show()

    # bins_raw = calculate_bins(all_no_electricity_pixels, all_electricity_pixels
    # plot_histogram(axs[2, 0], "No Electricity Bright Pixel Count", "Pixel Counts", "Frequency", all_no_electricity_pixels, bins_raw)
    # plot_histogram(axs[2, 1], "Electricity Bright Pixel Count", "Pixel Counts", "Frequency", all_electricity_pixels, bins_raw)


if __name__ == "__main__":
    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, help="Model path.", default=config.model_path
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.intermediate_tiffs_dir, help="Results dir"
    )
    main(ESDConfig(**parser.parse_args().__dict__))
