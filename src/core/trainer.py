"""
QREX-FL Training Engine
Handles federated learning training with quantum-resistant security
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from datetime import datetime
import copy

logger = logging.getLogger(__name__)

class QREXTrainer:
    """
    Quantum-resistant federated learning trainer for cryptocurrency risk assessment
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = "cpu",
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        Initialize QREX trainer
        
        Args:
            model: PyTorch model to train
            device: Training device (cpu/cuda)
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training history
        self.training_history = []
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"QREXTrainer initialized with {self._count_parameters()} parameters")
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, 
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   epoch: int) -> Dict[str, float]:
        """
        Train model for one epoch
        
        Args:
            dataloader: Training data loader
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        for batch_idx, (features, labels) in enumerate(dataloader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            
            # Handle different output formats
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                # Multi-class output
                loss = criterion(outputs, labels.long())
                predictions = torch.argmax(outputs, dim=1)
            else:
                # Binary classification
                loss = criterion(outputs.squeeze(), labels.float())
                predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
            correct_predictions += (predictions == labels).sum().item()
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        
        metrics = {
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "samples": total_samples
        }
        
        self.training_history.append(metrics)
        
        self.logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        
        return metrics
    
    def evaluate(self, 
                dataloader: DataLoader,
                criterion: nn.Module) -> Dict[str, float]:
        """
        Evaluate model on validation/test data
        
        Args:
            dataloader: Evaluation data loader
            criterion: Loss function
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                
                # Handle different output formats
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    # Multi-class output
                    loss = criterion(outputs, labels.long())
                    predictions = torch.argmax(outputs, dim=1)
                    probabilities = torch.softmax(outputs, dim=1)
                else:
                    # Binary classification
                    loss = criterion(outputs.squeeze(), labels.float())
                    probabilities = torch.sigmoid(outputs.squeeze())
                    predictions = (probabilities > 0.5).float()
                
                total_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
                correct_predictions += (predictions == labels).sum().item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        
        # Calculate additional metrics for binary classification
        if len(set(all_labels)) == 2:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
            precision = precision_score(all_labels, all_predictions, zero_division=0)
            recall = recall_score(all_labels, all_predictions, zero_division=0)
            f1 = f1_score(all_labels, all_predictions, zero_division=0)
            
            # For AUC, we need probabilities
            try:
                auc = roc_auc_score(all_labels, all_predictions)
            except:
                auc = 0.5  # Random baseline if AUC can't be calculated
            
            return {
                "loss": avg_loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc_roc": auc,
                "samples": total_samples
            }
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "samples": total_samples
        }
    
    def train(self,
             train_dataloader: DataLoader,
             val_dataloader: Optional[DataLoader] = None,
             epochs: int = 10,
             criterion: Optional[nn.Module] = None) -> Dict[str, List[float]]:
        """
        Complete training loop
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            epochs: Number of training epochs
            criterion: Loss function (default: BCEWithLogitsLoss)
            
        Returns:
            Training history
        """
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()
        
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        best_val_accuracy = 0.0
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_dataloader, criterion, epoch)
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            
            # Validation
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader, criterion)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                
                # Save best model
                if val_metrics["accuracy"] > best_val_accuracy:
                    best_val_accuracy = val_metrics["accuracy"]
                    best_model_state = copy.deepcopy(self.model.state_dict())
                
                self.logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                               f"Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Restore best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"Restored best model with validation accuracy: {best_val_accuracy:.4f}")
        
        return history
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters for federated learning"""
        return {name: param.clone() for name, param in self.model.named_parameters()}
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters from federated aggregation"""
        model_dict = self.model.state_dict()
        model_dict.update(parameters)
        self.model.load_state_dict(model_dict)
    
    def save_model(self, filepath: str) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay
            }
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        
        self.logger.info(f"Model loaded from: {filepath}")
    
    def compute_parameter_update(self, 
                                initial_parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute parameter update (difference) for federated learning
        
        Args:
            initial_parameters: Parameters before training
            
        Returns:
            Parameter updates (difference)
        """
        current_parameters = self.get_model_parameters()
        
        updates = {}
        for name, current_param in current_parameters.items():
            if name in initial_parameters:
                updates[name] = current_param - initial_parameters[name]
            else:
                updates[name] = current_param
        
        return updates
    
    def apply_parameter_update(self, 
                              updates: Dict[str, torch.Tensor],
                              learning_rate: float = 1.0) -> None:
        """
        Apply parameter updates to model
        
        Args:
            updates: Parameter updates to apply
            learning_rate: Scaling factor for updates
        """
        current_parameters = self.get_model_parameters()
        
        updated_parameters = {}
        for name, param in current_parameters.items():
            if name in updates:
                updated_parameters[name] = param + learning_rate * updates[name]
            else:
                updated_parameters[name] = param
        
        self.set_model_parameters(updated_parameters)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress"""
        if not self.training_history:
            return {"status": "No training completed"}
        
        latest_metrics = self.training_history[-1]
        
        return {
            "total_epochs": len(self.training_history),
            "latest_epoch": latest_metrics["epoch"],
            "latest_loss": latest_metrics["loss"],
            "latest_accuracy": latest_metrics["accuracy"],
            "best_accuracy": max(m["accuracy"] for m in self.training_history),
            "model_parameters": self._count_parameters(),
            "device": self.device
        }
