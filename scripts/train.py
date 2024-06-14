import sys
from argparse import ArgumentParser

from pathlib import Path
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig, PROJ_NAME
from src.models.supervised.random_forest_module import RandomForest

ROOT = Path.cwd()


def train(options: ESDConfig):
    wandb.finish()
    # initialize wandb
    wandb.init(project=PROJ_NAME, config=options)
    # setup the wandb logger
    logger = pl.loggers.WandbLogger(project=PROJ_NAME)

    # initialize the datamodule
    datamodule = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        seed=options.seed,
        selected_bands=options.selected_bands,
        slice_size=options.slice_size
    )

    # prepare the data
    datamodule.prepare_data()
    
    datamodule.setup("fit")

    # create a model params dict to initialize ESDSegmentation
    # note: different models have different parameters
    
    model_params = {}
    if options.model_type == "SegmentationCNN":
        model_params = {
            "depth": options.depth,
            "embedding_size": options.embedding_size,
            "pool_sizes": [int(x) for x in options.pool_sizes.split(",")],
            "kernel_size": options.kernel_size
        }
    elif options.model_type == "FCNResnetTransfer":
        pass
    elif options.model_type == "UNet":
        model_params = {
            "n_encoders": options.n_encoders,
            "embedding_size": options.embedding_size
        }

    if options.model_type == "RandomForestClassifier":
        model = RandomForest(
            n_estimators = options.n_estimators,
            max_depth = options.max_depth,
            min_samples_split = options.min_samples_split,
            min_samples_leaf = options.min_samples_leaf,
            bootstrap = options.bootstrap,
            criterion = options.criterion,
            random_state = options.seed,
            fit_frequency=options.fit_frequency
        )
    else:
        # initialize the ESDSegmentation model
        model = ESDSegmentation(
            model_type=options.model_type,
            in_channels=options.in_channels,
            out_channels=options.out_channels,
            learning_rate=options.learning_rate,
            model_params=model_params
        )

    # Use the following callbacks, they're provided for you,
    # but you may change some of the settings
    # ModelCheckpoint: saves intermediate results for the neural network
    # in case it crashes
    # LearningRateMonitor: logs the current learning rate on weights and biases
    # RichProgressBar: nicer looking progress bar (requires the rich package)
    # RichModelSummary: shows a summary of the model before training (requires rich)
    callbacks = [
        ModelCheckpoint(
            dirpath=ROOT / "models" / options.model_type,
            filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
            save_top_k=0,
            save_last=True,
            verbose=True,
            monitor="val_loss",
            mode="min",
            every_n_train_steps=1000,
        ),
        LearningRateMonitor(),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]
    
    # initialize trainer, set accelerator, devices, number of nodes, logger
    # max epochs and callbacks
    trainer = pl.Trainer(accelerator=options.accelerator,
                         devices=options.devices,
                         logger=logger,
                         max_epochs=options.max_epochs,
                         callbacks=callbacks
                        )

    # run trainer.fit
    trainer.fit(model=model, datamodule=datamodule)

    # if running random forest, save the model and checkpoint and print validation metrics
    if options.model_type == "RandomForestClassifier":
        model.save_model(ROOT / "models" / options.model_type / 'random_forest_model.pkl')
        trainer.save_checkpoint(ROOT / "models" / options.model_type / "last.ckpt")
        print(f"Average Validation Metrics:")
        print(f"Accuracy: {model.avg_val_accuracy():.4f}")
        print(f"F1 Score: {model.avg_f1_score():.4f}")
        print(f"Precision: {model.avg_precision():.4f}")
        print(f"Recall: {model.avg_recall():.4f}")

if __name__ == "__main__":
    # load dataclass arguments from yml file

    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=str,
        help="The model to initialize.",
        default=config.model_type,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="The learning rate for training model",
        default=config.learning_rate,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Number of epochs to train for.",
        default=config.max_epochs,
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )

    parser.add_argument(
        "--in_channels",
        type=int,
        default=config.in_channels,
        help="Number of input channels",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=config.out_channels,
        help="Number of output channels",
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="Depth of the encoders (CNN only)",
        default=config.depth,
    )
    parser.add_argument(
        "--n_encoders",
        type=int,
        help="Number of encoders (Unet only)",
        default=config.n_encoders,
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        help="Embedding size of the neural network (CNN/Unet)",
        default=config.embedding_size,
    )
    parser.add_argument(
        "--pool_sizes",
        help="A comma separated list of pool_sizes (CNN only)",
        type=str,
        default=config.pool_sizes,
    )
    parser.add_argument(
        "--kernel_size",
        help="Kernel size of the convolutions",
        type=int,
        default=config.kernel_size,
    )

    parser.add_argument(
        "--fit_frequency",
        help="How often random forest is fit",
        type=int,
        default=config.fit_frequency,
    )

    parser.add_argument(
        "--min_samples_leaf",
        help="Random Forest min samples leaf",
        type=int,
        default=config.min_samples_leaf,
    )

    parser.add_argument(
        "--min_samples_split",
        help="Min number of samples to split in random forest",
        type=int,
        default=config.min_samples_split,
    )

    parser.add_argument(
        "--n_estimators",
        help="Number of estimators in random forest",
        type=int,
        default=config.n_estimators,
    )

    parse_args = parser.parse_args()

    config = ESDConfig(**parse_args.__dict__)
    train(config)
