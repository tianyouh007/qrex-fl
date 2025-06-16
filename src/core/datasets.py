"""
QREX-FL Dataset Classes
Handles Bitcoin and cryptocurrency datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class EllipticDataset(Dataset):
    """
    Elliptic Bitcoin Dataset handler
    Based on research findings for temporal evaluation
    """
    
    def __init__(self, 
                 features_path: str,
                 classes_path: str,
                 edgelist_path: Optional[str] = None,
                 time_split: int = 34,
                 split: str = 'train',
                 normalize: bool = True,
                 binary_classification: bool = True):
        """
        Initialize Elliptic dataset
        
        Args:
            features_path: Path to transaction features CSV
            classes_path: Path to transaction classes CSV  
            edgelist_path: Optional path to edge list CSV
            time_split: Time step for temporal split (34 for train, 35+ for test)
            split: 'train', 'test', or 'all'
            normalize: Whether to normalize features
            binary_classification: Whether to use binary (vs multiclass) classification
        """
        self.features_path = Path(features_path)
        self.classes_path = Path(classes_path)
        self.edgelist_path = Path(edgelist_path) if edgelist_path else None
        self.time_split = time_split
        self.split = split
        self.normalize = normalize
        self.binary_classification = binary_classification
        
        self.logger = logging.getLogger(__name__)
        
        # Load and process data
        self._load_data()
        self._process_data()
        
        self.logger.info(f"Loaded Elliptic dataset: {len(self)} samples in {split} split")
    
    def _load_data(self):
        """Load raw data files"""
        try:
            # Load features (no header in original Elliptic dataset)
            if self.features_path.suffix == '.csv':
                # Check if file has header
                with open(self.features_path, 'r') as f:
                    first_line = f.readline()
                    has_header = not first_line.split(',')[0].isdigit()
                
                self.features_df = pd.read_csv(
                    self.features_path, 
                    header=0 if has_header else None
                )
            else:
                raise ValueError(f"Unsupported file format: {self.features_path.suffix}")
            
            # Load classes
            self.classes_df = pd.read_csv(self.classes_path)
            
            # Load edges if provided
            if self.edgelist_path and self.edgelist_path.exists():
                self.edges_df = pd.read_csv(self.edgelist_path)
            else:
                self.edges_df = None
                
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def _process_data(self):
        """Process and filter data based on split"""
        try:
            # Ensure consistent column naming
            if self.features_df.columns[0] != 'txId':
                # Original Elliptic format: first column is txId, second is time_step
                feature_cols = [f'feature_{i}' for i in range(len(self.features_df.columns) - 2)]
                self.features_df.columns = ['txId', 'time_step'] + feature_cols
            
            # Merge features with classes
            self.data = pd.merge(self.features_df, self.classes_df, on='txId', how='left')
            
            # Filter out unknown transactions for supervised learning
            if self.binary_classification:
                # Keep only known transactions (class 1=illicit, 2=licit)
                self.data = self.data[self.data['class'].isin([1, 2])].copy()
                # Convert to binary: 1=illicit->1, 2=licit->0
                self.data['binary_class'] = (self.data['class'] == 1).astype(int)
            
            # Apply temporal split based on research methodology
            if self.split == 'train':
                self.data = self.data[self.data['time_step'] <= self.time_split]
            elif self.split == 'test':
                self.data = self.data[self.data['time_step'] > self.time_split]
            # 'all' uses complete dataset
            
            # Extract features and labels
            feature_columns = [col for col in self.data.columns 
                             if col.startswith('feature_') or 
                             (col not in ['txId', 'time_step', 'class', 'binary_class'])]
            
            self.features = self.data[feature_columns].values.astype(np.float32)
            
            if self.binary_classification:
                self.labels = self.data['binary_class'].values.astype(np.float32)
            else:
                # For multiclass, map classes to 0-based indexing
                unique_classes = sorted(self.data['class'].dropna().unique())
                class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
                self.labels = self.data['class'].map(class_mapping).values.astype(np.long)
            
            # Normalize features if requested
            if self.normalize:
                self.scaler = StandardScaler()
                self.features = self.scaler.fit_transform(self.features)
            
            # Store metadata
            self.num_features = self.features.shape[1]
            self.num_samples = len(self.features)
            
            # Class distribution for debugging
            unique, counts = np.unique(self.labels, return_counts=True)
            class_dist = dict(zip(unique, counts))
            self.logger.info(f"Class distribution in {self.split}: {class_dist}")
            
        except Exception as e:
            self.logger.error(f"Failed to process data: {e}")
            raise
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index
        
        Args:
            idx: Sample index
            
        Returns:
            (features, label) tuple
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return features, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset
        
        Returns:
            Class weights tensor
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        weights = total / (len(unique) * counts)
        
        # For binary classification, return positive class weight
        if self.binary_classification:
            pos_weight = weights[1] if len(weights) > 1 else 1.0
            return torch.tensor([pos_weight], dtype=torch.float32)
        else:
            return torch.tensor(weights, dtype=torch.float32)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            "num_samples": self.num_samples,
            "num_features": self.num_features,
            "split": self.split,
            "class_distribution": dict(zip(*np.unique(self.labels, return_counts=True))),
            "time_split": self.time_split,
            "feature_range": {
                "min": float(self.features.min()),
                "max": float(self.features.max()),
                "mean": float(self.features.mean()),
                "std": float(self.features.std())
            }
        }

class BitcoinDataset(Dataset):
    """
    Generic Bitcoin dataset handler for various formats
    """
    
    def __init__(self, 
                 data_path: str,
                 target_column: str = 'label',
                 feature_columns: Optional[List[str]] = None,
                 test_size: float = 0.2,
                 split: str = 'train',
                 random_state: int = 42):
        """
        Initialize Bitcoin dataset
        
        Args:
            data_path: Path to dataset file
            target_column: Name of target column
            feature_columns: List of feature column names
            test_size: Fraction for test split
            split: 'train' or 'test'
            random_state: Random seed
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.test_size = test_size
        self.split = split
        self.random_state = random_state
        
        self.logger = logging.getLogger(__name__)
        
        self._load_and_split_data()
    
    def _load_and_split_data(self):
        """Load and split data"""
        # Load data
        if self.data_path.suffix == '.csv':
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.suffix == '.parquet':
            self.data = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Select features
        if self.feature_columns is None:
            self.feature_columns = [col for col in self.data.columns 
                                   if col != self.target_column]
        
        X = self.data[self.feature_columns].values.astype(np.float32)
        y = self.data[self.target_column].values.astype(np.float32)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        if self.split == 'train':
            self.features = X_train
            self.labels = y_train
        else:
            self.features = X_test
            self.labels = y_test
        
        self.logger.info(f"Loaded Bitcoin dataset: {len(self)} samples in {self.split} split")
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

def create_federated_dataloaders(dataset: Dataset, 
                                num_clients: int,
                                batch_size: int = 32,
                                shuffle: bool = True) -> List[DataLoader]:
    """
    Create federated data loaders for multiple clients
    
    Args:
        dataset: Source dataset
        num_clients: Number of federated clients
        batch_size: Batch size for each client
        shuffle: Whether to shuffle data
        
    Returns:
        List of DataLoaders for each client
    """
    # Split dataset into num_clients partitions
    dataset_size = len(dataset)
    partition_size = dataset_size // num_clients
    
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * partition_size
        if i == num_clients - 1:  # Last client gets remaining data
            end_idx = dataset_size
        else:
            end_idx = (i + 1) * partition_size
        
        indices = list(range(start_idx, end_idx))
        client_subset = torch.utils.data.Subset(dataset, indices)
        client_datasets.append(client_subset)
    
    # Create DataLoaders
    client_loaders = []
    for client_dataset in client_datasets:
        loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Avoid multiprocessing issues
        )
        client_loaders.append(loader)
    
    return client_loaders
