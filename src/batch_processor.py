import torch
import time

class BatchProcessor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.start_time = time.time()
        self.batch = 0

    def print_batch_results(self, mode, dataset_size, loss, batch_start_time, batch_size, epoch=None):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        batch_time_ms = (current_time - batch_start_time) * 1000

        current = (self.batch + 1) * batch_size
        epoch_str = f"Epoch: {epoch+1}, " if epoch is not None else ""
        
        print(f"\r{mode} - {epoch_str}Batch: {self.batch+1} [{current:>5d}/{dataset_size:>5d}], "
            f"Time: {elapsed_time:.0f}s {batch_time_ms:.0f}ms/step, Loss: {loss:>7f}", end="")


    def process_training_batch(self, batch_data, loss_fn):
        batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
        outputs = self.model(batch_data)
        targets = batch_data['targets']
        loss = loss_fn(outputs, targets)
        loss.backward()
        return outputs, targets, loss

    def process_evaluation_batch(self, batch_data, loss_fn):
        with torch.no_grad():
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            outputs = self.model(batch_data)
            targets = batch_data['targets']
            loss = loss_fn(outputs, targets)
            return outputs, targets, loss

    def process_training_batches(self, loader, loss_fn, optimizer, epoch, accumulation_steps=1):
        self.batch = 0
        total_loss = 0.0
        all_targets = []
        all_outputs = []
        self.start_time = time.time()
        self.model.train()
        optimizer.zero_grad()

        for data in loader:
            batch_start_time = time.time()
            
            logits, targets, loss = self.process_training_batch(data, loss_fn)
            total_loss += loss.item()

            fully_accumulated = (self.batch + 1) % accumulation_steps == 0
            if fully_accumulated:
                optimizer.step()
                optimizer.zero_grad()

            # Detach from the (gradient) computation graph to save on memory
            all_outputs.append(logits.detach())
            all_targets.append(targets.detach())

            batch_size = targets.shape[0]
            self.print_batch_results("Training", len(loader.dataset), loss.item(), batch_start_time, batch_size, epoch)
            self.batch += 1

        if not fully_accumulated:
            # Ensure any remaining gradients are applied
            optimizer.step()
            optimizer.zero_grad()
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        print()
        avg_loss = total_loss / len(loader)
        return avg_loss, all_outputs, all_targets

    def process_evaluation_batches(self, loader, loss_fn):
        self.batch = 0
        total_loss = 0.0
        all_targets = []
        all_outputs = []
        self.start_time = time.time()
        self.model.eval()

        for data in loader:
            batch_start_time = time.time()
            
            logits, targets, loss = self.process_evaluation_batch(data, loss_fn)
            total_loss += loss.item()

            # Detach from the (gradient) computation graph to save on memory
            all_outputs.append(logits.detach())
            all_targets.append(targets.detach())

            batch_size = targets.shape[0]
            self.print_batch_results("Evaluation", len(loader.dataset), loss.item(), batch_start_time, batch_size)
            self.batch += 1
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        print()
        avg_loss = total_loss / len(loader)
        return avg_loss, all_outputs, all_targets
