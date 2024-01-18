from batch_processor import BatchProcessor
from metrics import MetricsManager
import numpy as np
import torch
import traceback
from torch.utils.data import DataLoader


class EvaluationHandler:
    def __init__(self, model, device, metrics_dict=None):
        self.batch_processor = BatchProcessor(model, device)
        if metrics_dict is not None:
            self.metrics_manager = MetricsManager(model, device, metrics_dict)

    def adjust_batch_size_for_evaluation(self, dataloader, desired_batch_size, max_attempts=5):
        attempt = 0
        current_batch_size = desired_batch_size

        while attempt < max_attempts:
            try:
                # Try processing one batch
                data_iter = iter(dataloader)
                sample_batch = next(data_iter)
                self.batch_processor.process_evaluation_batch(sample_batch, lambda x, y: 0)  # Dummy loss function
                break  # If successful, break out of the loop
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Reduce batch size
                    current_batch_size //= 2
                    dataloader = DataLoader(dataloader.dataset, batch_size=current_batch_size, shuffle=False, num_workers=2)
                    torch.cuda.empty_cache()
                    attempt += 1
                else:
                    raise e  # Re-raise exception if it's not a memory error
            except Exception:
                traceback.print_exc()
                break  # Break on other exceptions

        return dataloader

    def evaluate(self, eval_loader, loss_fn, hyperparameters=None, optimize=False):
        eval_loader = self.adjust_batch_size_for_evaluation(eval_loader, eval_loader.batch_size)
        loss, logits, targets = self.batch_processor.process_evaluation_batches(eval_loader, loss_fn)

        if optimize:
            print("Optimizing Thresholds")
            thresholds, metrics_results = self.optimize_thresholds(logits, targets)
        else:
            thresholds, metrics_results = self.predetermined_thresholds(logits, targets, hyperparameters)

        print(f"Evaluation Results:")
        print(f"Average Loss: {loss:.4f}")
        self.metrics_manager.print_metrics_results(metrics_results)

        return loss, metrics_results, thresholds
    
    def predetermined_thresholds(self, logits, targets, hyperparameters):
        # Set default values
        thresholds = None
        percentile = None

        # Update values based on hyperparameters if provided
        if hyperparameters:
            thresholds = hyperparameters.get("thresholds", thresholds)
            percentile = hyperparameters.get("percentile", percentile)

        metrics_results, thresholds = self.metrics_manager.calculate_metrics(targets, logits, thresholds=thresholds, percentile=percentile, return_thresholds=True)
        return thresholds, metrics_results
    
    def optimize_thresholds(self, logits, targets):
        best_global_metric = -np.inf
        num_labels = len(logits[0])
        best_thresholds = [0.5] * num_labels

        # Iterate over a range of thresholds for each label
        for label in range(num_labels):
            for threshold in np.linspace(0, 1, 101):  # Example range and step size
                temp_thresholds = best_thresholds.copy()
                temp_thresholds[label] = threshold
                metrics_results = self.metrics_manager.calculate_metrics(targets, logits, thresholds=temp_thresholds)
                
                current_metric = metrics_results["Optimization Metric"]

                if current_metric > best_global_metric:
                    best_global_metric = current_metric
                    best_thresholds = temp_thresholds

        metrics_results = self.metrics_manager.calculate_metrics(targets, logits, thresholds=temp_thresholds)
        return best_thresholds, metrics_results

