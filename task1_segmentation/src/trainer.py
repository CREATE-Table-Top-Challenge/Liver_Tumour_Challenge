import os
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Callable
from tqdm import tqdm
from .metric_tracker import MetricTracker
import logging
import torch.nn as nn


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "output",
        max_epochs: int = 100,
        val_interval: int = 1,
        patience: int = 10
    ):
        """
        A trainer class to handle the training and validation loops.

        Args:
            model: The model to train (must inherit from BaseModel)
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to use for training ('cuda' or 'cpu')
            output_dir: Directory to save checkpoints and logs
            max_epochs: Maximum number of epochs to train
            val_interval: How often to run validation (in epochs)
            patience: Number of epochs to wait for improvement before early stopping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.patience = patience
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get optimizer and scheduler
        self.optimizer, self.scheduler = model.configure_optimizers()
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_metric = -1
        self.best_metric_epoch = -1
        self.no_improvement = 0
        
        # Initialize metric history
        self.train_metrics_history = []
        self.val_metrics_history = []
        
        # Initialize metric tracker
        self.metric_tracker = MetricTracker(os.path.join(self.output_dir, 'metrics'))
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self) -> Dict[str, float]:
        """Run one epoch of training."""
        self.model.train()
        epoch_metrics = {}
        total_samples = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.max_epochs}') as pbar:
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Training step
                self.optimizer.zero_grad()
                loss, metrics = self.model.training_step(batch)
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                batch_size = batch['image'].size(0)
                total_samples += batch_size
                
                for k, v in metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = 0
                    epoch_metrics[k] += v * batch_size
                
                # Update progress bar
                pbar.set_postfix(loss=metrics.get('train_loss', 0))
        
        # Calculate epoch averages
        for k in epoch_metrics:
            epoch_metrics[k] /= total_samples
            
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_metrics = {}
        total_samples = 0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for batch in pbar:
                    # Move batch to device
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    # Validation step
                    metrics = self.model.validation_step(batch)
                    
                    # Update metrics
                    batch_size = batch['image'].size(0)
                    total_samples += batch_size
                    
                    for k, v in metrics.items():
                        if k not in val_metrics:
                            val_metrics[k] = 0
                        val_metrics[k] += v * batch_size
                    
                    # Update progress bar
                    pbar.set_postfix(val_loss=metrics['val_loss'])
        
        # Calculate averages
        for k in val_metrics:
            val_metrics[k] /= total_samples
            
        # Get end of epoch validation metrics
        epoch_metrics = self.model.on_validation_epoch_end()
        val_metrics.update(epoch_metrics)
        
        return val_metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save a checkpoint of the model."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'best_metric_epoch': self.best_metric_epoch,
            'metrics': metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best performance
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f'Saved new best model to {best_path}')
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.best_metric_epoch = checkpoint['best_metric_epoch']
        
        self.logger.info(f'Loaded checkpoint from epoch {self.current_epoch}')
    
    def train(self, early_stopping_metric: str = 'val_loss', 
              higher_is_better: bool = False):
        """
        Run the training loop.
        
        Args:
            early_stopping_metric: Metric to use for early stopping
            higher_is_better: Whether higher values of the metric are better
        """
        self.logger.info(f'Starting training on device: {self.device}')
        self.logger.info(f'Training for {self.max_epochs} epochs')
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            self.train_metrics_history.append(train_metrics)
            
            # Log training metrics
            train_log = f'Epoch {epoch + 1}/{self.max_epochs} - '
            train_log += ' - '.join(f'{k}: {v:.4f}' for k, v in train_metrics.items())
            self.logger.info(train_log)

            # Validation phase
            epoch_metrics = train_metrics.copy()
            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self.validate()
                self.val_metrics_history.append(val_metrics)
                epoch_metrics.update(val_metrics)
                
                # Log validation metrics
                val_log = 'Validation - '
                val_log += ' - '.join(f'{k}: {v:.4f}' for k, v in val_metrics.items())
                self.logger.info(val_log)
                
                # Check for improvement
                current_metric = val_metrics[early_stopping_metric]
                is_best = False
                
                if higher_is_better:
                    is_better = current_metric > self.best_metric
                else:
                    is_better = current_metric < self.best_metric
                    
                if is_better or self.best_metric == -1:
                    self.best_metric = current_metric
                    self.best_metric_epoch = epoch + 1
                    self.no_improvement = 0
                    is_best = True
                else:
                    self.no_improvement += 1
                
                # Save checkpoint
                self.save_checkpoint({**train_metrics, **val_metrics}, is_best)
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    else:
                        self.scheduler.step()
                
                # Early stopping check
                if self.no_improvement >= self.patience:
                    self.logger.info(
                        f'Early stopping triggered: no improvement in {self.patience} epochs'
                    )
                    break
            
            # Update metrics tracker with all available metrics for this epoch (only once)
            self.metric_tracker.update(epoch, epoch_metrics)
            
            # Update plots if validation ran
            if (epoch + 1) % self.val_interval == 0:
                self.metric_tracker.update_plots()
            
            # Print epoch summary
            self.logger.info(
                f'Current best {early_stopping_metric}: {self.best_metric:.4f} '
                f'at epoch {self.best_metric_epoch}'
            )
        
        self.logger.info('Training completed!')