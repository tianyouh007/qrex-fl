"""Configuration management for QREX-FL"""
import os
from typing import Dict, Any

class Config:
    def __init__(self):
        self.values = {
            'quantum_resistant': True,
            'federated_learning': {'min_clients': 2, 'rounds': 10},
            'compliance': {'jurisdictions': ['US_FINCEN', 'EU_MICA']}
        }
    
    def get(self, key: str, default=None):
        return self.values.get(key, default)
