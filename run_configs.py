import pandas as pd
import torch
import sklearn
from torch.utils.data import DataLoader

from training import TrainingHandler
from evaluation import EvaluationHandler
from custom_dataset import CustomDataset
from data_preprocessing import ANOMALY_LABELS, load_data
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
        abbreviated = row['Abbreviation'] == 'True'
        loss_type = row['Loss Type']
        balanced = row['Balanced'] == 'True'
        layer_nums_str = row['Layers to Unfreeze']
        encoder_name = row['Encoder Name']
        batch_size = row['Batch Size']
        max_epochs = row['Epochs']
        learning_rate = row['Learning Rate']
        max_len = row['Max Length']

        layer_nums = eval(layer_nums_str) if isinstance(layer_nums_str, str) else None        
        
        # Load data
        if abbreviated:
            train_df = load_data("./data/train_data_final.pkl", ANOMALY_LABELS, pp_path="./data/train_data_processed2.pkl")
            test_df = load_data("./data/test_data_final.pkl", ANOMALY_LABELS, pp_path="./data/test_data_processed2.pkl")
        else: 
            train_df = load_data("./data/train_data_final.pkl", ANOMALY_LABELS)
            test_df = load_data("./data/test_data_final.pkl", ANOMALY_LABELS)

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

        loss_fn = losses.loss(model, loss_type, balanced, training_set, training_loader, device)
        optimizer = torch.optim.Adam(params =  model.parameters(), lr=learning_rate)
        metrics_dict = {
            "Custom Classification Report": lambda y_true, y_pred: custom_classification_report(y_true, y_pred),
            "Optimization Metric": lambda y_true, y_pred: sklearn.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
        }
        

        # Train and evaluate the model
        trainer = TrainingHandler(model, optimizer, loss_fn, device, metrics_dict, directory="model_save")
        evaluator = EvaluationHandler(model, device, metrics_dict)    
        trainer.train(training_loader, testing_loader, save=True, epochs=max_epochs)
        loss, metrics_results, thresholds = evaluator.evaluate(testing_loader, loss_fn, optimize=True)

        # Save results, model, etc.
        ModelManager.save_results(model, metrics_results, thresholds, "results.csv")

if __name__ == "__main__":
    main()


