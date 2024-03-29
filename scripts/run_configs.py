import sys
sys.path.append('../src')

import pandas as pd
import torch
import sklearn
from torch.utils.data import DataLoader

from training import TrainingHandler
from evaluation import EvaluationHandler
from custom_dataset import CustomDataset
from data_preprocessing import ANOMALY_LABELS, ANOMALY_SUBCATEGORIES, load_data
from custom_dataset import CustomDataset
from model import SequenceClassificationModel
from metrics import custom_classification_report
import losses
from model_management import ModelManager

def read_model_configurations(file_path):
    return pd.read_csv(file_path)

def main():
    config_df = read_model_configurations("model-configs.csv")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for _, row in config_df.iterrows():
        abbreviated = row['Abbreviation']
        loss_type = row['Loss Type']
        balanced = row['Balanced']
        layer_nums_str = row['Layers to Unfreeze']
        encoder_name = row['Encoder Name']
        batch_size = row['Batch Size']
        max_epochs = row['Epochs']
        learning_rate = row['Learning Rate']
        max_len = row['Max Length']
        is_subcategory = row['Subcategory']

        layer_nums = eval(layer_nums_str) if isinstance(layer_nums_str, str) else None        
        
        # Load data
        if is_subcategory:
            labels = ANOMALY_SUBCATEGORIES
        else:
            labels = ANOMALY_LABELS 

        if abbreviated:
            train_df = load_data("../data/train_data_final.pkl", labels, pp_path="../data/train_data_processed2.pkl")
            test_df = load_data("../data/test_data_final.pkl", labels, pp_path="../data/test_data_processed2.pkl")
        else: 
            train_df = load_data("../data/train_data_final.pkl", labels)
            # test_df = load_data("../data/test_data_final.pkl", labels)
            test_df = pd.read_csv("../data/subset_test_data.csv")
            # Define a function to convert lists to numpy arrays
            def str_to_list(string):
                # Remove the square brackets and split the string by spaces
                elements = string.strip('[]').split()
                # Convert each element to an integer and create a list
                return [int(x) for x in elements]

            # Apply the function to the column containing lists
            test_df['labels'] = test_df['labels'].apply(str_to_list)

        tokenizer = SequenceClassificationModel.get_tokenizer(encoder_name)

        training_set = CustomDataset(train_df, tokenizer, max_len)
        testing_set = CustomDataset(test_df, tokenizer, max_len)

        train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2}
        test_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2}

        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)

        # Initialize model, loss, optimizer, and metrics
        num_labels = len(test_df.labels.iloc[0])
        model = SequenceClassificationModel(encoder_name, num_labels=num_labels)
        model.set_trainable_layers(layer_nums)
        model.to(device)

        if abbreviated:
            model.model_name += '_Abbreviated'

        if is_subcategory:
            model.model_name += '_Subcategory'

        loss_fn = losses.loss(model, loss_type, balanced, training_set, training_loader, device)
        optimizer = torch.optim.Adam(params =  model.parameters(), lr=learning_rate)
        metrics_dict = {
            "Custom Classification Report": lambda y_true, y_pred: custom_classification_report(y_true, y_pred, labels),
            "Optimization Metric": lambda y_true, y_pred: sklearn.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
        }


        # Train and evaluate the model
        trainer = TrainingHandler(model, optimizer, loss_fn, device, labels, metrics_dict, directory="model_save")
        evaluator = EvaluationHandler(model, device, labels, metrics_dict)    
        trainer.train(training_loader, testing_loader, save=True, epochs=max_epochs)
        # _, _, thresholds = evaluator.evaluate(training_loader, loss_fn, optimize=True)
        # hyperparameters={"thresholds": thresholds}
        # loss, metrics_results, thresholds = evaluator.evaluate(testing_loader, loss_fn, optimize=False, hyperparameters=hyperparameters)

        # # Save results, model, etc.
        # ModelManager.save_results(model.model_name, metrics_results, thresholds, "../out/results-sub.csv")

        pred = evaluator.evaluate(testing_loader, loss_fn, optimize=True, return_pred=True)
        pred_df = pd.read_csv("../out/pred.csv")
        pred_df[model.model_name] = pred.tolist()
        pred_df.to_csv("../out/pred.csv")


if __name__ == "__main__":
    main()


