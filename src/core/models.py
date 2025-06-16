"""
QREX-FL Core Models
Neural network architectures for cryptocurrency risk assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import logging

class TemporalBitcoinRiskModel(nn.Module):
    """
    Temporal Bitcoin Risk Assessment Model
    Based on research findings for Elliptic dataset structure
    """
    
    def __init__(self, 
                 input_size: int = 165,  # Elliptic dataset feature count
                 hidden_sizes: tuple = (256, 128, 64, 32),
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True):
        """
        Initialize temporal Bitcoin risk model
        
        Args:
            input_size: Number of input features (165 for Elliptic)
            hidden_sizes: Tuple of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout (except for last layer)
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer for binary classification
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized TemporalBitcoinRiskModel with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output logits of shape (batch_size,)
        """
        logits = self.network(x)
        return logits.squeeze(1)  # Remove last dimension for BCEWithLogitsLoss
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities
        
        Args:
            x: Input tensor
            
        Returns:
            Probabilities between 0 and 1
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        return probabilities
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        return {
            "model_type": "TemporalBitcoinRiskModel",
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "total_parameters": self._count_parameters(),
            "dropout_rate": self.dropout_rate,
            "batch_norm": self.use_batch_norm
        }

class ComplianceModel(nn.Module):
    """
    Multi-task compliance model for regulatory assessment
    """
    
    def __init__(self, 
                 input_size: int = 165,
                 num_jurisdictions: int = 6,
                 shared_hidden_size: int = 128,
                 task_hidden_size: int = 64):
        """
        Initialize compliance model
        
        Args:
            input_size: Number of input features
            num_jurisdictions: Number of regulatory jurisdictions
            shared_hidden_size: Size of shared representation layer
            task_hidden_size: Size of task-specific layers
        """
        super().__init__()
        
        self.input_size = input_size
        self.num_jurisdictions = num_jurisdictions
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, shared_hidden_size),
            nn.BatchNorm1d(shared_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Risk assessment head
        self.risk_head = nn.Sequential(
            nn.Linear(shared_hidden_size, task_hidden_size),
            nn.ReLU(),
            nn.Linear(task_hidden_size, 1)
        )
        
        # Compliance classification heads (one per jurisdiction)
        self.compliance_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_hidden_size, task_hidden_size),
                nn.ReLU(),
                nn.Linear(task_hidden_size, 1)
            ) for _ in range(num_jurisdictions)
        ])
        
        # Violation type classification
        self.violation_head = nn.Sequential(
            nn.Linear(shared_hidden_size, task_hidden_size),
            nn.ReLU(),
            nn.Linear(task_hidden_size, 10)  # 10 common violation types
        )
        
        self._initialize_weights()
        self.logger = logging.getLogger(__name__)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task outputs
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with different task outputs
        """
        # Shared representation
        shared_features = self.shared_layers(x)
        
        # Task-specific outputs
        outputs = {
            'risk_score': self.risk_head(shared_features).squeeze(1),
            'compliance': [head(shared_features).squeeze(1) for head in self.compliance_heads],
            'violations': self.violation_head(shared_features)
        }
        
        return outputs
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

class GraphBitcoinRiskModel(nn.Module):
    """
    Graph-based Bitcoin risk model using transaction relationships
    """
    
    def __init__(self, 
                 node_features: int = 165,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        """
        Initialize graph-based model
        
        Args:
            node_features: Number of node features
            hidden_dim: Hidden dimension size
            num_layers: Number of graph convolution layers
            dropout: Dropout rate
        """
        super().__init__()
        
        try:
            from torch_geometric.nn import GCNConv, global_mean_pool
            self.gcn_available = True
        except ImportError:
            self.gcn_available = False
            self.logger = logging.getLogger(__name__)
            self.logger.warning("torch_geometric not available, using MLP fallback")
        
        if self.gcn_available:
            # Graph convolution layers
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(node_features, hidden_dim))
            
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.global_pool = global_mean_pool
        else:
            # Fallback to MLP if torch_geometric not available
            self.convs = nn.Sequential(
                nn.Linear(node_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass for graph data
        
        Args:
            x: Node features
            edge_index: Graph edge indices (if using PyG)
            batch: Batch indices (if using PyG)
            
        Returns:
            Prediction logits
        """
        if self.gcn_available and edge_index is not None:
            # Graph convolution
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Global pooling
            if batch is not None:
                x = self.global_pool(x, batch)
            else:
                x = x.mean(dim=0, keepdim=True)
        else:
            # Fallback MLP
            x = self.convs(x)
        
        # Classification
        output = self.classifier(x)
        return output.squeeze(1)

class EnsembleBitcoinModel(nn.Module):
    """
    Ensemble model combining multiple approaches
    """
    
    def __init__(self, 
                 input_size: int = 165,
                 num_models: int = 3):
        """
        Initialize ensemble model
        
        Args:
            input_size: Number of input features
            num_models: Number of base models in ensemble
        """
        super().__init__()
        
        # Create diverse base models
        self.models = nn.ModuleList([
            TemporalBitcoinRiskModel(input_size, (256, 128, 64), 0.3),
            TemporalBitcoinRiskModel(input_size, (512, 256, 128, 64), 0.2),
            TemporalBitcoinRiskModel(input_size, (128, 64, 32), 0.4)
        ])
        
        # Ensemble aggregation
        self.ensemble_weight = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble prediction
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Weighted ensemble
        predictions = torch.stack(predictions, dim=1)
        weights = F.softmax(self.ensemble_weight, dim=0)
        ensemble_pred = torch.sum(predictions * weights.unsqueeze(0), dim=1)
        
        return ensemble_pred
