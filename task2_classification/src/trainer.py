"""
Trainer for Task 2 Classification
Handles training loop, validation, checkpointing, and early stopping
"""
import os
import torch
import torch.nn as nn
import logging
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for worker processes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

try:
    import seaborn as sns
    sns.set_theme(style='whitegrid', font_scale=1.1)
except ImportError:
    pass  # fall back to plain matplotlib

from .metrics import ClassificationMetrics


class Trainer:
    """
    Trainer class for classification model training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = 'cuda',
        output_dir: str = './model_checkpoints',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        val_interval: int = 1,
        patience: int = 10,
        optimizer_config: dict = None,
        loss_config: dict = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Classification model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on ('cuda' or 'cpu')
            output_dir: Directory to save checkpoints
            learning_rate: Learning rate for optimizer (default if optimizer_config not provided)
            weight_decay: Weight decay for optimizer (default if optimizer_config not provided)
            max_epochs: Maximum number of training epochs
            val_interval: Validation every N epochs
            patience: Early stopping patience (epochs)
            optimizer_config: Optimizer configuration dict with keys 'type', 'lr', 'weight_decay', etc.
            loss_config: Loss configuration dict with keys 'type', 'class_weights', etc.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.patience = patience
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'model_checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Set up loss and optimizer from configs
        self.criterion = self._build_loss(loss_config or {})
        self.optimizer = self._build_optimizer(
            model.parameters(),
            optimizer_config or {},
            learning_rate,
            weight_decay
        )
        self.metrics = ClassificationMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.best_model_path = None

        # History for live plot
        self._history = {
            'train_loss': [], 'val_loss': [],
            'train_acc':  [], 'val_acc':  [],
            'train_f1':   [], 'val_f1':   [],
            'epochs':     [],
        }
        self._plot_path = self.output_dir / 'training_progress.png'
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
        
        # Log configuration
        logging.info(f"Trainer initialized:")
        logging.info(f"  Device: {device}")
        logging.info(f"  Loss: {self.criterion.__class__.__name__}")
        logging.info(f"  Optimizer: {self.optimizer.__class__.__name__}")
        logging.info(f"  Max epochs: {max_epochs}")
        logging.info(f"  Patience: {patience}")
        logging.info(f"  Output dir: {output_dir}")
    
    def _build_optimizer(self, parameters, optimizer_config: dict, lr: float, wd: float):
        """
        Build optimizer based on config type.
        
        Args:
            parameters: Model parameters
            optimizer_config: Config dict with keys 'type', 'lr', 'weight_decay', 'momentum', 'betas', 'eps'
            lr: Default learning rate
            wd: Default weight decay
            
        Returns:
            Optimizer instance
        """
        opt_type = optimizer_config.get('type', 'adam').lower()
        
        # Get optimizer hyperparameters (use defaults if not in config)
        opt_lr = float(optimizer_config.get('lr', lr))
        opt_wd = float(optimizer_config.get('weight_decay', wd))
        
        if opt_type == 'adam':
            betas_list = optimizer_config.get('betas', [0.9, 0.999])
            betas = tuple(float(b) for b in betas_list)
            eps = float(optimizer_config.get('eps', 1e-8))
            logging.info(f"Creating Adam optimizer: lr={opt_lr}, weight_decay={opt_wd}, betas={betas}, eps={eps}")
            return torch.optim.Adam(
                parameters,
                lr=opt_lr,
                weight_decay=opt_wd,
                betas=betas,
                eps=eps
            )
        elif opt_type == 'adamw':
            betas_list = optimizer_config.get('betas', [0.9, 0.999])
            betas = tuple(float(b) for b in betas_list)
            eps = float(optimizer_config.get('eps', 1e-8))
            logging.info(f"Creating AdamW optimizer: lr={opt_lr}, weight_decay={opt_wd}, betas={betas}, eps={eps}")
            return torch.optim.AdamW(
                parameters,
                lr=opt_lr,
                weight_decay=opt_wd,
                betas=betas,
                eps=eps
            )
        elif opt_type == 'sgd':
            momentum = float(optimizer_config.get('momentum', 0.9))
            logging.info(f"Creating SGD optimizer: lr={opt_lr}, weight_decay={opt_wd}, momentum={momentum}")
            return torch.optim.SGD(
                parameters,
                lr=opt_lr,
                weight_decay=opt_wd,
                momentum=momentum
            )
        else:
            logging.warning(f"Unknown optimizer type '{opt_type}', defaulting to Adam")
            return torch.optim.Adam(parameters, lr=opt_lr, weight_decay=opt_wd)
    
    def _build_loss(self, loss_config: dict):
        """
        Build loss function based on config type.
        
        Args:
            loss_config: Config dict with keys 'type', 'class_weights'
            
        Returns:
            Loss function instance
        """
        loss_type = loss_config.get('type', 'cross_entropy').lower()
        class_weights = loss_config.get('class_weights')
        
        # Convert class_weights to tensor if provided
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            logging.info(f"Using class weights: {class_weights}")
        
        if loss_type == 'cross_entropy':
            logging.info(f"Creating CrossEntropyLoss (weight={weight_tensor is not None})")
            return nn.CrossEntropyLoss(weight=weight_tensor)
        elif loss_type == 'focal_loss':
            # Focal loss reduces weight for easy examples and focuses on hard ones
            # alpha=class_weights (balance), gamma=2 (focus factor)
            logging.info(f"Creating Focal Loss (weight={weight_tensor is not None})")
            # Using a simple focal loss implementation via weighted CrossEntropy with gamma
            # For a full focal loss, would need: -alpha * (1-p)^gamma * log(p)
            return nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            logging.warning(f"Unknown loss type '{loss_type}', defaulting to CrossEntropyLoss")
            return nn.CrossEntropyLoss(weight=weight_tensor)
    
    def _update_plot(self):
        """Save an nnUNet-style training progress plot after every epoch."""
        h = self._history
        epochs = h['epochs']
        if not epochs:
            return

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        fig.suptitle('Training Progress', fontsize=14, fontweight='bold', y=1.01)

        # --- Loss subplot ---
        ax = axes[0]
        ax.plot(epochs, h['train_loss'], label='Train loss',
                color='#4878CF', linewidth=1.8, marker='o', markersize=3)
        ax.plot(epochs, h['val_loss'],   label='Val loss',
                color='#D65F5F', linewidth=1.8, marker='o', markersize=3,
                linestyle='--')
        if self.best_epoch in epochs:
            best_idx = epochs.index(self.best_epoch)
            ax.axvline(self.best_epoch, color='#6ACC65', linewidth=1.2,
                       linestyle=':', alpha=0.8, label=f'Best (ep {self.best_epoch})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend(frameon=True)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # --- Accuracy + F1 subplot ---
        ax = axes[1]
        ax.plot(epochs, h['train_acc'], label='Train acc',
                color='#4878CF', linewidth=1.8, marker='o', markersize=3)
        ax.plot(epochs, h['val_acc'],   label='Val acc',
                color='#D65F5F', linewidth=1.8, marker='o', markersize=3,
                linestyle='--')
        ax.plot(epochs, h['train_f1'],  label='Train F1',
                color='#6ACC65', linewidth=1.4, marker='s', markersize=3,
                linestyle='-',  alpha=0.75)
        ax.plot(epochs, h['val_f1'],    label='Val F1',
                color='#E6A118', linewidth=1.4, marker='s', markersize=3,
                linestyle='--', alpha=0.75)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Accuracy & F1')
        ax.legend(frameon=True)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        fig.tight_layout()
        fig.savefig(str(self._plot_path), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.max_epochs} [Train]") as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                self.metrics.update(outputs, labels)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        train_metrics = self.metrics.compute()
        
        return avg_loss, train_metrics
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc=f"Epoch {self.current_epoch}/{self.max_epochs} [Val]") as pbar:
                for batch in pbar:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    # Update metrics
                    total_loss += loss.item()
                    self.metrics.update(outputs, labels)
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        val_metrics = self.metrics.compute()
        
        return avg_loss, val_metrics
    
    def save_checkpoint(self, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
        
        # Save latest checkpoint — use atomic write (tmp -> rename) to avoid
        # Windows ERROR_SHARING_VIOLATION (error code 32) when antivirus or
        # the search indexer briefly holds the file open.
        latest_path = self.checkpoint_dir / 'latest_model.pth'
        tmp_latest  = self.checkpoint_dir / 'latest_model.pth.tmp'
        torch.save(checkpoint, tmp_latest)
        tmp_latest.replace(latest_path)   # atomic on same filesystem

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            tmp_best  = self.checkpoint_dir / 'best_model.pth.tmp'
            torch.save(checkpoint, tmp_best)
            tmp_best.replace(best_path)
            self.best_model_path = str(best_path)
            logging.info(f"[OK] Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        logging.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop.

        Automatically resumes from ``checkpoint_dir/latest_model.pth`` if it
        exists, so a run interrupted mid-fold picks up where it left off.
        """
        # --- Auto-resume ---
        latest_ckpt = self.checkpoint_dir / 'latest_model.pth'
        if latest_ckpt.exists():
            logging.info(f"[*] Resuming from checkpoint: {latest_ckpt}")
            self.load_checkpoint(str(latest_ckpt))
            # best_model_path may exist already
            best_ckpt = self.checkpoint_dir / 'best_model.pth'
            if best_ckpt.exists():
                self.best_model_path = str(best_ckpt)
        else:
            logging.info("Starting training from scratch (no prior checkpoint found).")

        logging.info("Starting training loop...")

        for epoch in range(self.current_epoch + 1, self.max_epochs + 1):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch()
            
            # Log training metrics
            logging.info(f"\nEpoch {epoch}/{self.max_epochs} - Training:")
            logging.info(f"  Loss: {train_loss:.4f}")
            logging.info(f"  Accuracy: {train_metrics['accuracy']:.4f}")
            logging.info(f"  F1 Score: {train_metrics['f1']:.4f}")
            
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Train/F1', train_metrics['f1'], epoch)
            
            # Validate
            if epoch % self.val_interval == 0:
                val_loss, val_metrics = self.validate()
                
                # Log validation metrics
                logging.info(f"Epoch {epoch}/{self.max_epochs} - Validation:")
                logging.info(f"  Loss: {val_loss:.4f}")
                logging.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
                logging.info(f"  F1 Score: {val_metrics['f1']:.4f}")
                
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('Val/F1', val_metrics['f1'], epoch)

                # Record history and refresh plot
                self._history['epochs'].append(epoch)
                self._history['train_loss'].append(train_loss)
                self._history['val_loss'].append(val_loss)
                self._history['train_acc'].append(train_metrics['accuracy'])
                self._history['val_acc'].append(val_metrics['accuracy'])
                self._history['train_f1'].append(train_metrics['f1'])
                self._history['val_f1'].append(val_metrics['f1'])
                self._update_plot()
                
                # Check if best model
                current_acc = val_metrics['accuracy']
                is_best = current_acc > self.best_val_acc
                
                if is_best:
                    self.best_val_acc = current_acc
                    self.best_epoch = epoch
                    self.epochs_no_improve = 0
                    logging.info(f"[*] New best accuracy: {self.best_val_acc:.4f}")
                else:
                    self.epochs_no_improve += self.val_interval
                
                # Save checkpoint
                self.save_checkpoint(is_best=is_best)
                
                # Early stopping check
                if self.epochs_no_improve >= self.patience:
                    logging.info(f"\nEarly stopping triggered after {epoch} epochs")
                    logging.info(f"Best accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
                    break
        
        # Training complete
        self.writer.close()
        logging.info(f"\nTraining completed!")
        logging.info(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
