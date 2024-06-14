import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics

from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer
from src.models.supervised.random_forest_module import RandomForestClassifier

class ESDSegmentation(pl.LightningModule):
    def __init__(self, model_type, in_channels, out_channels, 
                 learning_rate=1e-3, model_params: dict = {}):
        '''
        Constructor for ESDSegmentation class.
        '''
        # call the constructor of the parent class
        super(ESDSegmentation, self).__init__()
        # use self.save_hyperparameters to ensure that the module will load
        self.save_hyperparameters()
        # store in_channels and out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate

        # if the model type is segmentation_cnn, initalize a unet as self.model
        if model_type =='SegmentationCNN':
            self.model = SegmentationCNN(in_channels,
                                         out_channels,
                                         **model_params)
        # if the model type is unet, initialize a unet as self.model
        elif model_type == "UNet":
            self.model = UNet(in_channels=in_channels, 
                              out_channels=out_channels, 
                              n_encoders=model_params["n_encoders"], 
                              embedding_size=model_params["embedding_size"])
        # if the model type is fcn_resnet_transfer, initialize a fcn_resnet_transfer as self.model
        elif model_type == "FCNResnetTransfer":
            self.model = FCNResnetTransfer(in_channels=in_channels, out_channels=out_channels, **model_params)
        elif model_type == "RandomForestClassifier":
            self.model = RandomForestClassifier(**model_params)
        
        # initialize the accuracy metrics for the semantic segmentation task
        weights = torch.tensor([1, 2], dtype=torch.float)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=out_channels)
        self.jaccard_index = torchmetrics.JaccardIndex(task="multiclass", num_classes=out_channels)
        self.auc = torchmetrics.classification.MulticlassAUROC(num_classes=out_channels)
    
    def forward(self, X):
        # evaluate self.model
        X = torch.nan_to_num(X)
        X = X.to(torch.float32)
        return self.model.forward(X)
    
    def training_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch

        # evaluate batch
        y_pred = self.forward(sat_img)
        mask = mask.to(torch.int64)

        # calculate cross entropy loss
        loss = self.loss_fn(y_pred, mask)

        self.log("training/loss", loss)

        # return loss
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch

        # evaluate batch for validation
        y_pred = self.forward(sat_img)
        mask = mask.to(torch.int64)
        
        # get the class with the highest probability
        final_class = torch.argmax(y_pred, dim=1)
        # evaluate each accuracy metric and log it in wandb
        
        # Evaulate and log jaccard index, accuracy, AUC, f1-score
        # self.log("validation/jaccard_index", self.jaccard_index(final_class, mask))
        # self.log("validation/accuracy", self.accuracy(final_class, mask))
        # self.log("validation/f1", self.f1(final_class, mask))
        # self.log("validation/auc", self.auc(y_pred, mask))
        
        loss = self.loss_fn(y_pred, mask)
        self.log("validation/loss", loss)

        # return validation loss 
        return loss
    
    def configure_optimizers(self):
        # initialize optimizer - Adam
        optimizer = Adam(params=self.parameters(), lr=self.hparams.learning_rate)
        # return optimizer
        return optimizer
