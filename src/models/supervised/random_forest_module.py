import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torchmetrics
import pytorch_lightning as pl
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, matthews_corrcoef

class RandomForest(pl.LightningModule):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 random_state=None, max_features=None, bootstrap=True, fit_frequency=25, criterion="gini", 
                 **kwargs):
        
        super(RandomForest, self).__init__()

        class_weights = [{0: 1, 1: 2} for _ in range(16)]  # slice_size ** 2

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            criterion=criterion,
            class_weight=class_weights,
            random_state=random_state,
            warm_start=True
        )
        self.train_X = None
        self.train_y = None

        self.automatic_optimization=False
        self.n_estimators = n_estimators
        self.total_estimators = n_estimators    # incrementally update number of estimators at each fit
        self.fit_frequency = fit_frequency

        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.agg_val_accuracy = 0
        self.agg_f1_score = 0
        self.agg_precision = 0
        self.agg_recall = 0
        self.agg_specificity = 0
        self.val_step_count = 0
        self.training_step_count = 0
        self.count = 0
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y = y.to(torch.float32)
        if self.train_X is None:
            self.train_X = X.numpy()
            self.train_y = y.numpy()
        else:
            self.train_X = np.concatenate((self.train_X, X.numpy()), axis=0)
            self.train_y = np.concatenate((self.train_y, y.numpy()), axis=0)

        self.training_step_count += 1
        if self.training_step_count % self.fit_frequency == 0:
            print('fitting after', self.fit_frequency, 'steps')

            batch_size = self.train_X.shape[0]
            flattened_train_X = self.train_X.reshape(batch_size, -1)
            flattened_train_y = self.train_y.reshape(batch_size, -1)

            # increase n_estimators
            self.total_estimators += self.n_estimators    # can experiment with how many additional estimators
            self.model.set_params(n_estimators = self.total_estimators)

            self.model.fit(flattened_train_X, flattened_train_y)
            self.train_X = None
            self.train_y = None

            # calculate training accuracy
            X_flattened = X.numpy().reshape(X.shape[0], -1)
            y_pred_train = torch.tensor(self.model.predict(X_flattened), dtype=torch.float32, requires_grad=True)
            y_pred_train = y_pred_train.reshape(X.shape[0], y.shape[1], y.shape[2])

            accuracy_train = self.accuracy(y_pred_train, y)
            print(f'Training Accuracy: {accuracy_train:.4f}')
            self.log("training/accuracy", accuracy_train, on_step=True)

            self.print_metrics(y, y_pred_train)

            if False:
                # Predict probabilities rather than classes
                # Using this method has lower accuracy than just directly predicting classes
                probabilities = self.model.predict_proba(X_flattened)
                probabilities_class_0 = np.array([array[:, 0] for array in probabilities])

                probabilities_sample_1 = probabilities_class_0[:, 0].reshape(8,8)
                probabilities_sample_2 = probabilities_class_0[:, 1].reshape(8,8)

                # classify something as a settlement if it has over 0.8 probability of being that class
                binary_sample_1 = (probabilities_sample_1 < 0.8).astype(int)
                binary_sample_2 = (probabilities_sample_2 < 0.8).astype(int)

                predictions = np.array([binary_sample_1, binary_sample_2])
                print("predictions", predictions)

        return {'loss': torch.tensor(0.0)}

    def validation_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch
        mask = mask.to(torch.float32)
    
        # Flatten or reshape the input images
        batch_size = sat_img.size(0)
        sat_img = sat_img.numpy()
        mask_numpy = mask.numpy()

        flattened_img = sat_img.reshape(batch_size, -1)
        flattened_mask = mask_numpy.reshape(batch_size, -1)

        if not hasattr(self.model, 'estimators_'):
            self.model.fit(flattened_img, flattened_mask)

        y_pred = torch.tensor(self.model.predict(flattened_img), dtype=torch.float32, requires_grad=True)

        # reshape y_pred to match the shape of mask
        y_pred = y_pred.reshape(batch_size, mask.shape[1], mask.shape[2])

        accuracy = self.accuracy(y_pred, mask)
        print(f'Validation Accuracy: {accuracy:.4f}')
        self.log("validation/accuracy", accuracy, on_step=True)

        self.print_metrics(mask, y_pred, is_validation=True)

        # to keep track of number of validation steps to calculate averages
        self.val_step_count += 1

        return {'loss': torch.tensor(0.0)}


    def configure_optimizers(self):
        return None

    def forward(self, X):
        # needed to add this to get predictions for evaluate in restitch_plot

        X = torch.nan_to_num(X)
        X = X.to(torch.float32)

        # Flatten the input X
        batch_size = X.shape[0]
        X_flat = X.reshape(batch_size, -1)
        
        # Get predictions
        predictions = self.model.predict(X_flat)
        n = int(predictions.shape[1] ** 0.5)
        
        # Convert predictions to a tensor and reshape appropriately
        predictions_tensor = torch.tensor(predictions, dtype=torch.float32).view(1,1,n,n)
        
        return predictions_tensor
    

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)

    def avg_val_accuracy(self):
        # ignore first 2 sanity check validation steps
        return self.agg_val_accuracy / (self.val_step_count-2)
    
    def avg_f1_score(self):
        return self.agg_f1_score / (self.count)
    
    def avg_precision(self):
        return self.agg_precision / (self.count)
    
    def avg_recall(self):
        return self.agg_recall / (self.count)
    
    def avg_specificity(self):
        return self.agg_specificity / (self.val_step_count-2)


    def print_metrics(self, y_true, y_pred, is_validation=False):
        '''
        Accuracy: correctly classified samples / total number of samples
        
        Precision: num true positives / num positive predictions made
        
        Recall (Sensitivity): proportion of true positive predictions out of all actual positive samples
        
        F1 Score: the harmonic mean of precision and recall
        
        Specificity: proportion of true negative predictions out of all actual negative samples
        
        ROC-AUC: measure of model's ability to distinguish between positive and negative samples 
            across all possible threshold values
            0.5 - predictions are random
            1.0 - perfect classification

        Matthews Correlation Coefficient: measure the quality of binary classifications
            considering all four confusion matrix values
             1 - perfect predictions
             0 - random predictions
            -1 - all incorrect predictions
        '''

        # convert to numpy and flatten tensors to work with sklearn metrics
        y_np = y_true.detach().cpu().numpy().astype(int).flatten()
        y_pred_np = y_pred.detach().cpu().numpy().astype(int).flatten()

        cm = confusion_matrix(y_np, y_pred_np)
        if cm.shape[0] == 1:  # Handle single class case
            true_positives = 0
            true_negatives = cm[0, 0]
            false_positives = 0
            false_negatives = 0
        else:
            true_positives = cm[1,1]
            true_negatives = cm[0,0]
            false_positives = cm[0,1]
            false_negatives = cm[1,0]

        '''
        Class 0: non-settlement
        Class 1: settlement

        True Positive: correctly predicts settlement
        True Negative: correctly predicts non-settlement
        False Positive: incorrectly predicts a non-settlement as a settlement
        False Negative: incorrectly predicts a settlement as a non-settlement
        '''

        print("true_positives:", true_positives)
        print("true_negatives:", true_negatives)
        print("false_positives:", false_positives)
        print("false_negatives:", false_negatives)

        accuracy = accuracy_score(y_np, y_pred_np)
        precision = precision_score(y_np, y_pred_np)
        recall = recall_score(y_np, y_pred_np)
        f1 = f1_score(y_np, y_pred_np)
        specificity = true_negatives / (true_negatives + false_positives)

        # Uncomment to print out more accuracy metrics

        # print(f"accuracy: {accuracy:.4f}")
        print(f"precision: {precision:.4f}")
        print(f"recall: {recall:.4f}")
        print(f"f1: {f1:.4f}")
        print(f"specificity: {specificity:.4f}")

        # if len(np.unique(y_np)) > 1 and len(np.unique(y_pred_np)) > 1:
        #     # make sure both classes are present to calculate
        #     print(f"roc_auc: {roc_auc_score(y_np, y_pred_np):.4f}")

        # print(f"matthews_corrcoef: {matthews_corrcoef(y_np, y_pred_np):.4f}")
        print("")

        if is_validation and self.val_step_count >= 2:
            # ignore first two sanity check validation steps
            self.agg_val_accuracy += accuracy
            self.agg_specificity += specificity if not np.isnan(specificity) else 0.5

            if true_positives + false_negatives != 0:
                # only add if there are true samples
                self.count += 1
                self.agg_f1_score += f1
                self.agg_precision += precision
                self.agg_recall += recall