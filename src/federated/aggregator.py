"""
QREX-FL Secure Aggregator
Implements secure aggregation with quantum-resistant cryptography
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

from ..quantum.crypto_manager import QuantumCryptoManager

class SecureAggregator:
    """
    Secure aggregation for federated learning with quantum-resistant protection
    """
    
    def __init__(self, crypto_manager: Optional[QuantumCryptoManager] = None):
        self.crypto_manager = crypto_manager or QuantumCryptoManager()
        self.logger = logging.getLogger(__name__)
        
        # Aggregation statistics
        self.aggregation_history = []
        self.verified_updates = 0
        self.failed_verifications = 0
        
    def secure_aggregate(
        self,
        client_updates: List[Dict[str, Any]],
        aggregation_weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Perform secure aggregation of client updates
        
        Args:
            client_updates: List of client model updates with signatures
            aggregation_weights: Optional weights for weighted aggregation
            
        Returns:
            Aggregated model parameters
        """
        try:
            if not client_updates:
                raise ValueError("No client updates provided for aggregation")
            
            # Step 1: Verify quantum-resistant signatures
            verified_updates = self._verify_client_signatures(client_updates)
            
            if not verified_updates:
                raise ValueError("No updates passed signature verification")
            
            # Step 2: Extract and validate parameters
            parameter_arrays = self._extract_parameters(verified_updates)
            
            # Step 3: Perform secure aggregation
            aggregated_params = self._aggregate_parameters(parameter_arrays, aggregation_weights)
            
            # Step 4: Create aggregation metadata
            aggregation_metadata = {
                "total_clients": len(client_updates),
                "verified_clients": len(verified_updates),
                "failed_verifications": len(client_updates) - len(verified_updates),
                "aggregation_method": "federated_averaging",
                "quantum_resistant": True
            }
            
            # Store aggregation history
            self.aggregation_history.append(aggregation_metadata)
            
            self.logger.info(f"Secure aggregation completed: {len(verified_updates)}/{len(client_updates)} verified")
            
            return {
                "parameters": aggregated_params,
                "metadata": aggregation_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Secure aggregation failed: {e}")
            raise
    
    def _verify_client_signatures(self, client_updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verify ML-DSA signatures on client updates"""
        verified_updates = []
        
        for i, update in enumerate(client_updates):
            try:
                # Extract signature components
                if "signature" not in update or "public_key" not in update:
                    self.logger.warning(f"Client {i}: Missing signature or public key")
                    self.failed_verifications += 1
                    continue
                
                signature = bytes.fromhex(update["signature"])
                public_key = bytes.fromhex(update["public_key"])
                
                # Extract parameters for verification
                params_to_verify = {k: v for k, v in update.items() 
                                  if k not in ["signature", "public_key"]}
                
                # Verify ML-DSA signature
                is_valid = self.crypto_manager.ml_dsa.verify_federated_update(
                    params_to_verify, signature, public_key
                )
                
                if is_valid:
                    verified_updates.append(update)
                    self.verified_updates += 1
                    self.logger.debug(f"Client {i}: Signature verified")
                else:
                    self.logger.warning(f"Client {i}: Signature verification failed")
                    self.failed_verifications += 1
                    
            except Exception as e:
                self.logger.error(f"Client {i}: Signature verification error: {e}")
                self.failed_verifications += 1
        
        return verified_updates
    
    def _extract_parameters(self, verified_updates: List[Dict[str, Any]]) -> List[Dict[str, np.ndarray]]:
        """Extract model parameters from verified updates"""
        parameter_arrays = []
        
        for update in verified_updates:
            if "parameters" in update:
                params = {}
                for name, param_list in update["parameters"].items():
                    params[name] = np.array(param_list)
                parameter_arrays.append(params)
            else:
                self.logger.warning("Update missing parameters field")
        
        return parameter_arrays
    
    def _aggregate_parameters(
        self,
        parameter_arrays: List[Dict[str, np.ndarray]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, List[float]]:
        """Aggregate model parameters using federated averaging"""
        
        if not parameter_arrays:
            raise ValueError("No parameter arrays to aggregate")
        
        # Use uniform weights if none provided
        if weights is None:
            weights = [1.0 / len(parameter_arrays)] * len(parameter_arrays)
        elif len(weights) != len(parameter_arrays):
            raise ValueError("Number of weights must match number of parameter arrays")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Get parameter names from first client
        param_names = list(parameter_arrays[0].keys())
        
        # Aggregate each parameter
        aggregated = {}
        for param_name in param_names:
            # Collect all arrays for this parameter
            param_arrays = []
            for i, params in enumerate(parameter_arrays):
                if param_name in params:
                    param_arrays.append(params[param_name])
                else:
                    self.logger.warning(f"Parameter {param_name} missing from client {i}")
            
            if param_arrays:
                # Weighted average
                weighted_sum = np.zeros_like(param_arrays[0])
                for param_array, weight in zip(param_arrays, weights[:len(param_arrays)]):
                    weighted_sum += weight * param_array
                
                # Convert back to list for JSON serialization
                aggregated[param_name] = weighted_sum.tolist()
        
        return aggregated
    
    def differential_privacy_noise(
        self,
        parameters: Dict[str, List[float]],
        noise_multiplier: float = 0.1,
        l2_norm_clip: float = 1.0
    ) -> Dict[str, List[float]]:
        """
        Add differential privacy noise to aggregated parameters
        
        Args:
            parameters: Aggregated parameters
            noise_multiplier: Noise scale for differential privacy
            l2_norm_clip: L2 norm clipping threshold
            
        Returns:
            Parameters with differential privacy noise
        """
        try:
            noisy_parameters = {}
            
            for param_name, param_values in parameters.items():
                param_array = np.array(param_values)
                
                # Clip L2 norm
                param_norm = np.linalg.norm(param_array)
                if param_norm > l2_norm_clip:
                    param_array = param_array * (l2_norm_clip / param_norm)
                
                # Add Gaussian noise
                noise = np.random.normal(0, noise_multiplier * l2_norm_clip, param_array.shape)
                noisy_param = param_array + noise
                
                noisy_parameters[param_name] = noisy_param.tolist()
            
            self.logger.info(f"Applied differential privacy with noise multiplier {noise_multiplier}")
            
            return noisy_parameters
            
        except Exception as e:
            self.logger.error(f"Failed to apply differential privacy: {e}")
            return parameters
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        return {
            "total_aggregations": len(self.aggregation_history),
            "verified_updates": self.verified_updates,
            "failed_verifications": self.failed_verifications,
            "verification_rate": self.verified_updates / (self.verified_updates + self.failed_verifications)
                                if (self.verified_updates + self.failed_verifications) > 0 else 0,
            "recent_aggregations": self.aggregation_history[-5:] if self.aggregation_history else []
        }
