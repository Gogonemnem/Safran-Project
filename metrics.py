import torch
from sklearn import metrics
import numpy as np


def binary_accuracy_per_label(y_true, y_pred):
    correct = y_pred == y_true
    accuracy_per_label = correct.float().mean(axis=0)
    return accuracy_per_label

def binary_accuracy_averaged(y_true, y_pred):
    accuracy_per_label = binary_accuracy_per_label(y_true, y_pred)
    accuracy_averaged = accuracy_per_label.mean()
    return accuracy_averaged

def custom_classification_report(y_true, y_pred, labels):
    report = metrics.classification_report(y_true, y_pred, output_dict=True, target_names=labels, zero_division=0)
    accuracy = binary_accuracy_per_label(y_true, y_pred)
    extended_accuracy_new = np.append(accuracy, [accuracy.mean()] * (len(report) - len(accuracy)))

    updated_report = {}
    for i, class_label in enumerate(report.keys()):
        # Create a new dictionary for the class with binary accuracy
        class_dict = {'binary_accuracy': extended_accuracy_new[i]}
        
        # Merge this dictionary with the existing metrics for the class
        class_dict.update(report[class_label])

        # Update the main report dictionary
        updated_report[class_label] = class_dict

    return updated_report

class MetricsManager:
    def __init__(self, model, device, metrics_dict, labels):
        self.model = model
        self.device = device
        self.metrics_dict = metrics_dict
        self.labels = labels
    
    def calculate_metrics(self, targets, outputs, is_logit=True, thresholds=0.5, percentile=None, return_thresholds=False):
        results = {}
        if is_logit:
            outputs = torch.sigmoid(outputs)

        if thresholds is None:
            thresholds = 0.5

        # Calculate percentile is specified
        if percentile is not None:
            thresholds = []
            for i in range(outputs.shape[1]):  # Iterate over each label
                label_scores = outputs[:, i].detach().cpu().numpy()
                threshold = np.percentile(label_scores, percentile)
                thresholds.append(threshold)
            thresholds = np.array(thresholds)

        # Apply thresholds to outputs
        outputs = (outputs >= torch.tensor(thresholds, device=outputs.device)).float()

        for metric_name, metric_fn in self.metrics_dict.items():
            if metric_name in ["F1 Scores per Class", "Binary Accuracy per Class"]:
                metric_scores = metric_fn(targets.cpu(), outputs.cpu())  # Assuming targets and outputs are tensors
                for i, score in enumerate(metric_scores):
                    label = self.labels[i] if i < len(self.labels) else f"Class {i}"
                    results[f"{metric_name} - {label}"] = score
            else:
                results[metric_name] = metric_fn(targets.cpu(), outputs.cpu())

        if return_thresholds:
            return results, thresholds
        return results
    
    def format_value(self, val):
        """Helper function to format the value for printing."""
        if isinstance(val, (float, np.float16, np.float32, np.float64)):
            return f"{val:.4f}"
        elif isinstance(val, torch.Tensor) and val.dtype in [torch.float16, torch.float32, torch.float64]:
            return f"{val.item():.4f}"
        else:
            return val

    def print_metrics_results(self, metrics_results):
        # First, print scalar values and simple dictionaries
        for metric, value in metrics_results.items():
            if isinstance(value, dict) and not any(isinstance(v, dict) for v in value.values()):
                # Print simple dictionaries on a single line
                dict_values = ", ".join([f"{k}: {self.format_value(v)}" for k, v in value.items()])
                print(f"{metric}: {dict_values}")
            elif not isinstance(value, dict):
                # Print scalar values
                print(f"{metric}: {self.format_value(value)}")

        # Then, print nested dictionaries
        for metric, value in metrics_results.items():
            if isinstance(value, dict) and any(isinstance(v, dict) for v in value.values()):
                # Print nested dictionaries
                print(f"\n{metric}:")
                # Find the longest key length for formatting
                max_key_length = max(len(str(k)) for k in value.keys())
                for sub_key, sub_dict in value.items():
                    formatted_key = f"{sub_key}:".ljust(max_key_length + 2)
                    dict_values = ", ".join([f"{k}: {self.format_value(v)}" for k, v in sub_dict.items()])
                    print(f"  {formatted_key} {dict_values}")

