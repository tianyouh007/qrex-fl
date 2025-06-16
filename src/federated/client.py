"""
QREX-FL Federated Learning Client
Based on Flower framework with quantum-resistant security
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import numpy as np
from collections import OrderedDict

import flwr as fl
from flwr.common import NDArrays, Scalar

from ..quantum.crypto_manager import QuantumCryptoManager

class QREXFederatedClient(fl.client.NumPyClient):
    """
    QREX-FL Federated Learning Client with quantum-resistant security
    Implements Flower NumPyClient interface
    """
    
    def __init__(self, 
                 client_id: str,
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 device: str = 'cpu'):
        """
        Initialize QREX-FL client
        
        Args:
            client_id: Unique client identifier
            model: PyTorch model for training
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Computing device (cpu/cuda)
        """
        super().__init__()
        
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize quantum-resistant cryptography
        self.crypto_manager = QuantumCryptoManager()
        self.public_keys = self.crypto_manager.initialize_keys()
        
        # Training configuration
        self.learning_rate = 0.001
        self.local_epochs = 1
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.logger.info(f"Initialized QREX-FL client {client_id} with quantum-resistant crypto")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Get model parameters for federated aggregation
        
        Args:
            config: Configuration from server
            
        Returns:
            Model parameters as numpy arrays
        """
        try:
            # Extract parameters as numpy arrays
            parameters = []
            for param in self.model.parameters():
                parameters.append(param.detach().cpu().numpy())
            
            self.logger.debug(f"Client {self.client_id} returning {len(parameters)} parameter arrays")
            return parameters
            
        except Exception as e:
            self.logger.error(f"Failed to get parameters: {e}")
            raise
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model parameters from federated aggregation
        
        Args:
            parameters: Model parameters as numpy arrays
        """
        try:
            params_dict = zip(self.model.parameters(), parameters)
            for param, new_param in params_dict:
                param.data = torch.tensor(new_param, device=self.device, dtype=param.dtype)
            
            self.logger.debug(f"Client {self.client_id} updated parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to set parameters: {e}")
            raise
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train model locally with quantum-resistant security
        
        Args:
            parameters: Global model parameters
            config: Training configuration
            
        Returns:
            (updated_parameters, num_examples, metrics)
        """
        try:
            # Set global parameters
            self.set_parameters(parameters)
            
            # Configure training
            self.local_epochs = int(config.get("local_epochs", 1))
            self.learning_rate = float(config.get("learning_rate", 0.001))
            
            # Initialize optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            
            # Perform local training
            train_loss, num_examples = self._train_local()
            
            # Get updated parameters
            updated_parameters = self.get_parameters({})
            
            # Create secure metrics
            metrics = {
                "client_id": self.client_id,
                "train_loss": float(train_loss),
                "local_epochs": self.local_epochs,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Client {self.client_id} completed training: loss={train_loss:.4f}")
            
            return updated_parameters, num_examples, metrics
            
        except Exception as e:
            self.logger.error(f"Training failed for client {self.client_id}: {e}")
            raise
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate model locally
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
            
        Returns:
            (loss, num_examples, metrics)
        """
        try:
            # Set parameters
            self.set_parameters(parameters)
            
            # Evaluate model
            val_loss, accuracy, num_examples = self._evaluate_local()
            
            metrics = {
                "client_id": self.client_id,
                "accuracy": float(accuracy),
                "num_examples": num_examples,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Client {self.client_id} evaluation: loss={val_loss:.4f}, acc={accuracy:.4f}")
            
            return float(val_loss), num_examples, metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for client {self.client_id}: {e}")
            raise
    
    def _train_local(self) -> Tuple[float, int]:
        """
        Perform local training
        
        Returns:
            (average_loss, num_examples)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        num_examples = 0
        
        for epoch in range(self.local_epochs):
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Handle class imbalance with pos_weight (based on Elliptic dataset ~2% illicit)
                pos_weight = torch.tensor([49.0], device=self.device)  # 98%/2% ratio
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets.float())
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                num_examples += data.size(0)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, num_examples
    
    def _evaluate_local(self) -> Tuple[float, float, int]:
        """
        Perform local evaluation
        
        Returns:
            (loss, accuracy, num_examples)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets.float())
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy, total
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client information for registration"""
        return {
            "client_id": self.client_id,
            "public_keys": self.public_keys,
            "model_info": {
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "architecture": str(type(self.model).__name__)
            },
            "capabilities": {
                "quantum_resistant": True,
                "ml_dsa": True,
                "ml_kem": True
            }
        }
