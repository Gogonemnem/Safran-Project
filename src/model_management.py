import os
import torch
import pandas as pd
from pandas.errors import EmptyDataError

class ModelManager:
    @staticmethod
    def save_model(model, epoch, directory='model_save', base_path=None):
        """
        Saves the model state.

        Args:
        model (torch.nn.Module): The model to save.
        epoch (int): The current epoch number.
        file_path (str): Base directory to save the models.
        """
        if base_path is None:
            base_path = model.model_name # assumption that it has a name

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        file_path = os.path.join(directory, f"{base_path}_epoch_{epoch}.pth")

        torch.save(model.state_dict(), file_path)
        print(f'Model saved at {file_path}')

    @staticmethod
    def load_model(model, directory='model_save', base_path=None, epoch=None):
        """
        Loads the model state.

        Args:
        model (torch.nn.Module): The model to load state into.
        file_path (str): Path to the saved model file.
        """
        if base_path is None:
            base_path = model.model_name

        if epoch is None:
            epoch = ModelManager.find_last_saved_epoch(model, directory, base_path)
            if epoch == -1:
                print("No saved model found.")
                return
        
        file_path = os.path.join(directory, f"{base_path}_epoch_{epoch}.pth")
        if not os.path.exists(file_path):
            print(f"No model file found at {file_path}")
            return

        model.load_state_dict(torch.load(file_path))
        print(f'Model loaded from {file_path}')
        return file_path, epoch
    
    @staticmethod
    def find_last_saved_epoch(model, directory='model_save', base_path=None):
        """
        Finds the last saved epoch number in the specified directory.

        Args:
        file_path (str): The directory where models are saved.

        Returns:
        int: The last saved epoch number. Returns -1 if no saved model is found.
        """
        last_epoch = -1

        if base_path is None:
            base_path = model.model_name

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory):
            return last_epoch

        
        for filename in os.listdir(directory):
            split_filename = filename.split('_epoch_')
            model_filename = split_filename[0]
            if base_path == model_filename:
                epoch_number = int(split_filename[1].split('.')[0])
                if epoch_number > last_epoch:
                    last_epoch = epoch_number
        
        return last_epoch
    
    @staticmethod
    def create_records(report, model_name, metrics_results, thresholds):
        records = []
        categories = list(report.keys())
        num_threshold_categories = len(thresholds)

        for i, category in enumerate(categories):
            record = {
                'Model Name': model_name,
                'Category': category,
                **report[category],  # Unpack metrics into the record
                'Optimization Metric': metrics_results['Optimization Metric']
            }
            if i < num_threshold_categories:
                record['Threshold'] = thresholds[i]

            records.append(record)

        return records

    @staticmethod
    def read_or_initialize_df(file_path, sample_record):
        try:
            return pd.read_csv(file_path)
        except (EmptyDataError, FileNotFoundError):
            return pd.DataFrame(columns=sample_record.keys())

    @staticmethod
    def update_or_append_dataframe(existing_df, new_df, model_name):
        if model_name in existing_df['Model Name'].values:
            existing_df = existing_df[existing_df['Model Name'] != model_name]
            return pd.concat([existing_df, new_df])
        else:
            return pd.concat([existing_df, new_df])

    @staticmethod
    def save_to_csv(df, file_path):
        df.to_csv(file_path, index=False)

    @staticmethod
    def save_results(model_name, metrics_results, thresholds, file_path):
        report = metrics_results.get('Custom Classification Report', {})

        new_records = ModelManager.create_records(report, model_name, metrics_results, thresholds)
        sample_record = new_records[0] if new_records else {}
        existing_df = ModelManager.read_or_initialize_df(file_path, sample_record)
        updated_df = ModelManager.update_or_append_dataframe(existing_df, pd.DataFrame(new_records), model_name)
        ModelManager.save_to_csv(updated_df, file_path)

