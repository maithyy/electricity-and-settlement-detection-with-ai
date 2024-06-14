import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig
from src.visualization.restitch_plot import restitch_and_save_to_tiff
from src.preprocessing.file_utils import get_parent_tile_id
from src.preprocessing.file_utils import load_satellite
from src.utilities import SatelliteType
from src.preprocessing.preprocess_sat import maxprojection_viirs
from src.models.supervised.random_forest_module import RandomForest

import tifffile as tiff
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

ROOT = Path.cwd()

TRAIN_OR_VAL = "Val" # Change to "Train" or "Val" based on which you want to evaluate on
IS_RANDOM_FOREST= True  # Change to False if running SegCNN, FCNResnet, or UNet

def evaluate_final_tiffs(options):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    final_tiles = list(options.final_tiffs_dir.glob("Tile*"))
    print(len(final_tiles))
    for final_tile_path in final_tiles:
        curr_tile = str(final_tile_path).split('\\')[-1].split('/')[-1].split('_')[0]
        curr_ground_truth_path = f"data/raw/Train/{curr_tile}/groundTruth.tif"

        # Create a new ground truth with two labels
        #   class 1: settlements with no electricty
        #   class 0: everything else
        binary_gt_img = tiff.imread(curr_ground_truth_path)
        binary_gt_img[binary_gt_img != 1] = 0

        final_img = tiff.imread(final_tile_path)
        print('ground truth')
        print(binary_gt_img)
        print('final predictions')
        print(final_img)

        y_np = binary_gt_img.flatten()
        y_pred_np = final_img.flatten()

        cm = confusion_matrix(y_np, y_pred_np)
        true_positives = cm[1,1]
        true_negatives = cm[0,0]
        false_positives = cm[0,1]
        false_negatives = cm[1,0]

        accuracy = accuracy_score(y_np, y_pred_np)
        precision = precision_score(y_np, y_pred_np)
        recall = recall_score(y_np, y_pred_np)
        f1 = f1_score(y_np, y_pred_np)

        print(f"Metrics for {curr_tile}:\n")
        print(f"    accuracy: {accuracy:.4f}")
        print(f"    precision: {precision:.4f}")
        print(f"    recall: {recall:.4f}")
        print(f"    f1 score: {f1:.4f}\n")

        print(f"    true positives: {true_positives}")
        print(f"    true negatives: {true_negatives}")
        print(f"    false positives: {false_positives}")
        print(f"    false negatives: {false_negatives}\n")
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    print(f'Final Average Metrics:')
    print(f' - Accuracy: {np.mean(accuracies):.4f}')
    print(f' - Precision: {np.mean(precisions):.4f}')
    print(f' - Recall: {np.mean(recalls):.4f}')
    print(f' - F1 Score: {np.mean(f1_scores):.4f}\n')



def plot_results(options):
    '''
    Plot Electricity Ground Truth and Predictions & Final Combined Ground Truth and Predictions
    for validation tiles to be saved to the data/final_tiffs folder
    '''

    settlement_tiles = list(options.intermediate_tiffs_dir.glob("Tile*"))
    PIXEL_BRIGHTNESS_THRESHOLD = 14
    BRIGHTNESS_PERCENTAGE_THRESHOLD = 0.2

    for tiff_file in settlement_tiles:
        predicted_img = tiff.imread(tiff_file)      # combined settlement and electricity predictions
        electricity_prediction = tiff.imread(tiff_file) # keep track just the electricity predictions

        curr_tile = get_parent_tile_id(tiff_file).split('.tif')[0]
        curr_ground_truth_path = f"data/raw/Train/{curr_tile}/groundTruth.tif"

        # get max_viirs projection file using curr_tile
        raw_viirs_path = "data/raw/Train/"+curr_tile
        raw_viirs_file = load_satellite(Path(raw_viirs_path), SatelliteType.VIIRS)
        max_projection_viirs_arr = maxprojection_viirs(raw_viirs_file).values[0][0]

        predict_dim_1, predict_dim_2 = predicted_img.shape
        for i in range(predict_dim_1):
            for j in range(predict_dim_2):
                prediction_pixel = predicted_img[i][j] # 0 or 1
                start_i = 50*i
                end_i = 50*(i+1)
                start_j = 50*j
                end_j = 50*(j+1)
                max_viirs_chunk = max_projection_viirs_arr[start_i:end_i, start_j:end_j]

                num_bright_pixels = (max_viirs_chunk >= PIXEL_BRIGHTNESS_THRESHOLD).sum()
                num_dim_pixels = (max_viirs_chunk < PIXEL_BRIGHTNESS_THRESHOLD).sum()
                bright_pixel_percentage = num_bright_pixels/(num_dim_pixels+num_bright_pixels)

                # if model predicts no electricity
                if bright_pixel_percentage < BRIGHTNESS_PERCENTAGE_THRESHOLD:
                    electricity_prediction[i][j] = 1

                    # if settlement was predicted for that pixel
                    if prediction_pixel == 1:
                        predicted_img[i][j] = 1
                    else: 
                        predicted_img[i][j] = 0
                else:
                    electricity_prediction[i][j] = 0
                    predicted_img[i][j] = 0

        # Create a new ground truth with two labels
        #   class 1: settlements with no electricty
        #   class 0: everything else
        combined_gt_img = tiff.imread(curr_ground_truth_path)
        combined_gt_img[combined_gt_img != 1] = 0

        # create a ground truth image for electricity
        #   class 1: no electricity
        #   class 0: has electricity
        electricity_gt_img = tiff.imread(curr_ground_truth_path)
        electricity_gt_img[electricity_gt_img == 1] = 1
        electricity_gt_img[electricity_gt_img == 2] = 1
        electricity_gt_img[electricity_gt_img == 3] = 0
        electricity_gt_img[electricity_gt_img == 4] = 0

        elec_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "Electricity", np.array(["#f0e371", "#bd6666"]), N=2
            # yellow for electricity (0), pink for no electricity (1)
        )
        # Normalize the data to ensure values are between 0 and 1
        norm = matplotlib.colors.BoundaryNorm([0, 0.5, 1], elec_cmap.N)

        fig, axs = plt.subplots(nrows=2, ncols=2)
        axs[0,0].set_title("Electricity GT")
        axs[0,0].imshow(electricity_gt_img, cmap=elec_cmap, norm=norm)
        axs[0,1].set_title("Electricity Predictions")
        axs[0,1].imshow(electricity_prediction, cmap=elec_cmap, norm=norm)

        combined_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "Combined", np.array(["#aba782", "#690101"]), N=2
            # yellow-gray for everything else (0), burgundy for settlements w/o electricity (1)
        )

        axs[1,0].set_title("Combined GT")
        axs[1,0].imshow(combined_gt_img, cmap=combined_cmap)
        axs[1,1].set_title("Combined Predictions")
        axs[1,1].imshow(predicted_img, cmap=combined_cmap)

        # save the images to the final tiffs folder
        Path(options.final_tiffs_dir.parent).mkdir(exist_ok=True)
        Path(options.final_tiffs_dir).mkdir(exist_ok=True)
        Path(options.final_tiffs_dir / f"Plot_{curr_tile}.png").touch(exist_ok=True)
        fig.savefig(options.final_tiffs_dir / f"Plot_{curr_tile}.png", format="png")
        plt.close()
    


def main(options):
    # initialize datamodule
    datamodule = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        batch_size=options.batch_size,
        seed=options.seed,
        selected_bands=options.selected_bands,
        slice_size=options.slice_size
    )
    # prepare data
    datamodule.prepare_data()
    datamodule.setup("fit")

    # load model from checkpoint
    if options.model_type == "RandomForestClassifier":
        model = RandomForest.load_from_checkpoint(options.model_path)
        model.load_model(ROOT / "models" / options.model_type / "random_forest_model.pkl")
    else:
        model = ESDSegmentation.load_from_checkpoint(options.model_path)
    # set model to eval mode
    model.eval()

    # get a list of all processed tiles
    processed_tiles = list(Path(options.processed_dir/f"{TRAIN_OR_VAL}/subtiles").glob("Tile*"))

    if not (ROOT / "data" / "intermediate_tiffs" / options.model_type).exists():
        # for each tile
        for tile in processed_tiles:
            # restitch the plots and save the predictions to tiff files
            restitch_and_save_to_tiff(
                options=options, datamodule=datamodule, model=model, parent_tile_id=get_parent_tile_id(tile), 
                results_dir=options.intermediate_tiffs_dir, accelerator=options.accelerator, train_or_val=TRAIN_OR_VAL,
                is_random_forest=IS_RANDOM_FOREST
            )

        print('finished restitch and save to tiff')
    else:
        print('already have tiff files')

    # get a list of all predictions
    prediction_tiles = list(options.intermediate_tiffs_dir.glob("Tile*"))

    PIXEL_BRIGHTNESS_THRESHOLD = 14
    BRIGHTNESS_PERCENTAGE_THRESHOLD = 0.2

    for tiff_file in prediction_tiles:
        predicted_img = tiff.imread(tiff_file)

        curr_tile = get_parent_tile_id(tiff_file).split('.tif')[0]

        # get max_viirs projection file using curr_tile
        raw_viirs_path = "data/raw/Train/"+curr_tile
        raw_viirs_file = load_satellite(Path(raw_viirs_path), SatelliteType.VIIRS)
        max_projection_viirs_arr = maxprojection_viirs(raw_viirs_file).values[0][0]

        predict_dim_1, predict_dim_2 = predicted_img.shape
        for i in range(predict_dim_1):
            for j in range(predict_dim_2):
                prediction_pixel = predicted_img[i][j] # 0 or 1
                start_i = 50*i
                end_i = 50*(i+1)
                start_j = 50*j
                end_j = 50*(j+1)
                max_viirs_chunk = max_projection_viirs_arr[start_i:end_i, start_j:end_j]

                # if model predicted that a settlement exists for the current chunk
                if prediction_pixel == 1:

                    num_bright_pixels = (max_viirs_chunk >= PIXEL_BRIGHTNESS_THRESHOLD).sum()
                    num_dim_pixels = (max_viirs_chunk < PIXEL_BRIGHTNESS_THRESHOLD).sum()
                    bright_pixel_percentage = num_bright_pixels/(num_dim_pixels+num_bright_pixels)
                    
                    if bright_pixel_percentage < BRIGHTNESS_PERCENTAGE_THRESHOLD:
                        predicted_img[i][j] = 1
                    else:
                        predicted_img[i][j] = 0
                else:
                    predicted_img[i][j] = 0

        # predicted_img has finished processing
        Path(options.final_tiffs_dir.parent).mkdir(exist_ok=True)
        Path(options.final_tiffs_dir).mkdir(exist_ok=True)
        tiff.imwrite(options.final_tiffs_dir / f"{curr_tile}_final.tif", predicted_img)
    
    evaluate_final_tiffs(options)
    # plot_results(options)     # uncomment to plot results



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
