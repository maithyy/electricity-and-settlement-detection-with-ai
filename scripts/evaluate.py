import sys
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

sys.path.append(".")
from src.models.supervised.random_forest_module import RandomForest
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig
from src.visualization.restitch_plot import restitch_and_plot, restitch_and_plot_randforest, restitch_eval
from src.preprocessing.file_utils import get_parent_tile_id
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path.cwd()

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
        model = ESDSegmentation.load_from_checkpoint(options.model_path, strict=False)
    # set model to eval mode
    model.eval()

    # get a list of all processed tiles
    # processed_tiles = list(options.processed_dir.rglob("Tile*"))
    processed_tiles = list(Path(options.processed_dir/"Val/subtiles").glob("Tile*"))

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    # for each tile
    for tile in processed_tiles:
        if options.model_type == "RandomForestClassifier":
            restitch_and_plot_randforest(
                options=options, datamodule=datamodule, model=model, parent_tile_id=get_parent_tile_id(tile), 
                results_dir=options.results_dir, accelerator=options.accelerator
            )
        else:
            # run restitch and plot
            restitch_and_plot(
                options=options, datamodule=datamodule, model=model, parent_tile_id=get_parent_tile_id(tile), 
                results_dir=options.results_dir, accelerator=options.accelerator
            )

        subtile, prediction = restitch_eval(
            processed_dir=options.processed_dir / "Val",
            parent_tile_id=get_parent_tile_id(tile),
            accelerator=options.accelerator,
            datamodule=datamodule,
            model=model,
        )
        
        # calculate accuracy metrics
        ground_truth = subtile.ground_truth.values[0][0] - 1
        if options.model_type == "RandomForestClassifier":
            predictions = prediction[0][0]
        else:
            predictions = prediction[0].argmax(axis=0)
        
        y_np = ground_truth.astype(int).flatten()
        y_pred_np = predictions.astype(int).flatten()

        accuracies.append(accuracy_score(y_np, y_pred_np))
        precisions.append(precision_score(y_np, y_pred_np))
        recalls.append(recall_score(y_np, y_pred_np))
        f1_scores.append(f1_score(y_np, y_pred_np))


    print("Final Average Metrics for Settlement Detection")
    print(f' - Accuracy: {np.mean(accuracies):.4f}')
    print(f' - Precision: {np.mean(precisions):.4f}')
    print(f' - Recall: {np.mean(recalls):.4f}')
    print(f' - F1 Score: {np.mean(f1_scores):.4f}\n')


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
        "--results_dir", type=str, default=config.results_dir, help="Results dir"
    )
    main(ESDConfig(**parser.parse_args().__dict__))
