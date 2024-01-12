import torch
import traceback
from torch.utils.data import DataLoader

from batch_processor import BatchProcessor
from metrics import MetricsManager
from evaluation import EvaluationHandler
from model_management import ModelManager

class TrainingHandler:
    def __init__(self, model, optimizer, loss_fn, device, metrics_dict=None, directory=None, base_path=None):
        self.batch_processor = BatchProcessor(model, device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model = model
        self.device = device
        self.metrics_dict = metrics_dict
        self.directory = directory
        self.base_path = base_path
        self.start_epoch = 0
        self.accumulation_steps = 1

        if metrics_dict is not None:
            self.metrics_manager = MetricsManager(model, device, metrics_dict)
        
        if directory is not None:
            loaded_model = ModelManager.load_model(model, directory, base_path)
            if loaded_model is not None:
                file_path, loaded_epoch = loaded_model
                self.start_epoch = loaded_epoch + 1

    def train_epoch(self, train_loader, epoch, validation_loader=None, save=True):
        loss, outputs, targets = self.batch_processor.process_training_batches(train_loader, self.loss_fn, self.optimizer, epoch, self.accumulation_steps)
        
        if save:
            ModelManager.save_model(self.model, epoch, directory=self.directory, base_path=self.base_path)

        metrics_results = self.metrics_manager.calculate_metrics(targets, outputs)

        print(f"Train Results:")
        print(f"Average Loss: {loss:.4f}")
        self.metrics_manager.print_metrics_results(metrics_results)

        # Validation phase
        if validation_loader is not None:
            evaluation_handler = EvaluationHandler(self.model, self.device, self.metrics_dict)
            avg_val_loss, val_metrics_results, val_thresholds = evaluation_handler.evaluate(validation_loader, self.loss_fn, optimize=False)
        else:
            avg_val_loss = None
            val_metrics_results = {}

        return loss, avg_val_loss, val_metrics_results
    
    def adjust_batch_size_and_accumulation(self, dataloader, desired_batch_size, max_attempts=10):
        attempt = 0
        current_batch_size = desired_batch_size
        accumulation_steps = 1

        while attempt < max_attempts:
            try:
                # Try processing one batch
                data_iter = iter(dataloader)
                sample_batch = next(data_iter)
                self.batch_processor.process_training_batch(sample_batch, self.loss_fn)
                break  # If successful, break out of the loop
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Reduce batch size and increase accumulation steps
                    current_batch_size //= 2
                    accumulation_steps = desired_batch_size // current_batch_size
                    dataloader = DataLoader(dataloader.dataset, batch_size=current_batch_size, shuffle=True, num_workers=2)
                    torch.cuda.empty_cache()
                    attempt += 1
                else:
                    raise e  # Re-raise exception if it's not a memory error
            except Exception:
                traceback.print_exc()
                break  # Break on other exceptions
        return dataloader, accumulation_steps
    
    def train(self, train_loader, validation_loader=None, save=True, epochs=5):
        if self.start_epoch < epochs:
            print(f"Resuming training from epoch {self.start_epoch}")
        
        train_loader, self.accumulation_steps = self.adjust_batch_size_and_accumulation(train_loader, train_loader.batch_size)

        for epoch in range(self.start_epoch, epochs):
            self.train_epoch(train_loader, epoch, validation_loader, save)

