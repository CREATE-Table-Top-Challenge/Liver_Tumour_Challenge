from typing import Dict, Any, List, Optional
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime


class MetricTracker:
    """Class for tracking and visualizing training metrics."""
    
    def __init__(self, save_dir: str):
        """
        Initialize the metric tracker.
        
        Args:
            save_dir: Directory to save metric logs and plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics: Dict[str, List[float]] = {}
        self.epochs: List[int] = []
        
    def update(self, epoch: int, metrics: Dict[str, float]):
        """
        Update metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric values
        """
        self.epochs.append(epoch)
        
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
            
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return self.metrics[metric_name][-1]
        return None
        
    def get_best(self, metric_name: str, higher_is_better: bool = True) -> tuple[float, int]:
        """
        Get the best value for a metric and the epoch it occurred.
        
        Args:
            metric_name: Name of the metric
            higher_is_better: Whether higher values are better
            
        Returns:
            tuple: (best_value, best_epoch)
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return float('-inf' if higher_is_better else 'inf'), -1
            
        values = np.array(self.metrics[metric_name])
        epochs = np.array(self.epochs)
        
        if higher_is_better:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
            
        return values[best_idx], epochs[best_idx]
    
    def plot_metrics(self, metric_groups: Optional[Dict[str, List[str]]] = None):
        """
        Plot metrics over time with handling for metrics that are not logged every epoch.
        
        Args:
            metric_groups: Dictionary mapping plot titles to lists of metrics to include.
                           If None, all metrics will be plotted separately.
        """
        if metric_groups is None:
            # Plot each metric separately
            metric_groups = {name: [name] for name in self.metrics}

        for title, metrics in metric_groups.items():
            plt.figure(figsize=(10, 6))
            for metric in metrics:
                if metric not in self.metrics:
                    print(f"[MetricTracker] Skipping '{metric}' (not found in tracked metrics).")
                    continue

                values = self.metrics[metric]
                num_epochs = len(self.epochs)
                num_values = len(values)

                if num_values < num_epochs:
                    # Pad with NaN to match epoch count
                    padded_values = values + [np.nan] * (num_epochs - num_values)
                elif num_values > num_epochs:
                    # Extra values? Truncate them (shouldn't happen, but just in case)
                    padded_values = values[:num_epochs]
                else:
                    padded_values = values

                plt.plot(self.epochs, padded_values, label=metric)

            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title(title)
            plt.legend()
            plt.grid(True)

            # Save plot with clean name (no timestamp)
            plot_path = os.path.join(self.save_dir, f'{title.lower().replace(" ", "_")}.png')
            plt.savefig(plot_path)
            plt.close()

    def update_plots(self):
        """
        Re-render training plots and overwrite fixed-name files so they stay
        current after every validation step.  Groups are inferred automatically
        from which metric names are currently being tracked.
        """
        if not self.metrics:
            return

        # Build groups automatically from metric names
        metric_groups = {
            "Loss":        [m for m in self.metrics if "loss" in m.lower()],
            "Dice Scores": [m for m in self.metrics if "dice" in m.lower()],
            "HD95 Scores": [m for m in self.metrics if "hd95" in m.lower()],
        }
        # Drop empty groups
        metric_groups = {k: v for k, v in metric_groups.items() if v}
        # Anything that didn't fit into the groups above
        covered = {m for v in metric_groups.values() for m in v}
        other = [m for m in self.metrics if m not in covered]
        if other:
            metric_groups["Other"] = other

        for title, metrics in metric_groups.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            has_data = False
            for metric in metrics:
                if metric not in self.metrics:
                    continue
                values = self.metrics[metric]
                num_epochs = len(self.epochs)
                num_values = len(values)
                if num_values < num_epochs:
                    padded = values + [np.nan] * (num_epochs - num_values)
                else:
                    padded = values[:num_epochs]
                ax.plot(self.epochs, padded, label=metric)
                has_data = True

            if not has_data:
                plt.close(fig)
                continue

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_title(title)
            ax.legend()
            ax.grid(True)

            # Fixed filename – overwritten on every call
            plot_path = os.path.join(
                self.save_dir, f'{title.lower().replace(" ", "_")}.png'
            )
            fig.savefig(plot_path)
            plt.close(fig)
            
    def save(self):
        """Save metrics to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(self.save_dir, f'metrics_{timestamp}.json')
        
        data = {
            'epochs': self.epochs,
            'metrics': self.metrics
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load(self, metrics_path: str):
        """Load metrics from a JSON file."""
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            
        self.epochs = data['epochs']
        self.metrics = data['metrics']
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the tracked metrics."""
        summary = {}
        
        for metric_name in self.metrics:
            values = np.array(self.metrics[metric_name])
            
            summary[metric_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'last': float(values[-1]) if len(values) > 0 else None
            }
            
        return summary