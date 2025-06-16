"""
QREX-FL Federated Learning Server
Coordinates quantum-resistant federated learning
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import numpy as np

import flwr as fl
from flwr.common import NDArrays, Scalar, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy

from ..quantum.crypto_manager import QuantumCryptoManager
from .strategy import QREXStrategy

class QREXFederatedServer:
    """
    QREX-FL Federated Learning Server with quantum-resistant security
    """
    
    def __init__(self, 
                 model_fn: Callable,
                 num_rounds: int = 10,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2):
        """
        Initialize QREX-FL server
        
        Args:
            model_fn: Function that returns initial model
            num_rounds: Number of federated learning rounds
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients
        """
        self.model_fn = model_fn
        self.num_rounds = num_rounds
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize quantum-resistant cryptography
        self.crypto_manager = QuantumCryptoManager()
        self.server_keys = self.crypto_manager.initialize_keys()
        
        # Initialize strategy
        self.strategy = QREXStrategy(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=self._get_initial_parameters(),
            crypto_manager=self.crypto_manager
        )
        
        # Track client information
        self.registered_clients = {}
        self.round_history = []
        
        self.logger.info("Initialized QREX-FL server with quantum-resistant security")
    
    def start_federated_learning(self, server_address: str = "localhost:8080") -> Dict[str, Any]:
        """
        Start federated learning process
        
        Args:
            server_address: Server address for clients to connect
            
        Returns:
            Training history and results
        """
        try:
            self.logger.info(f"Starting QREX-FL federated learning on {server_address}")
            
            # Configure Flower server
            config = fl.server.ServerConfig(num_rounds=self.num_rounds)
            
            # Start server
            history = fl.server.start_server(
                server_address=server_address,
                config=config,
                strategy=self.strategy
            )
            
            self.logger.info("Federated learning completed successfully")
            
            # Process and return results
            return self._process_results(history)
            
        except Exception as e:
            self.logger.error(f"Federated learning failed: {e}")
            raise
    
    def register_client(self, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register client with quantum-resistant key exchange
        
        Args:
            client_info: Client information including public keys
            
        Returns:
            Registration response with server public keys
        """
        try:
            client_id = client_info["client_id"]
            
            # Verify client quantum capabilities
            if not client_info.get("capabilities", {}).get("quantum_resistant", False):
                raise ValueError(f"Client {client_id} does not support quantum-resistant crypto")
            
            # Store client information
            self.registered_clients[client_id] = {
                "info": client_info,
                "registered_at": datetime.now(),
                "public_keys": client_info.get("public_keys", {})
            }
            
            self.logger.info(f"Registered quantum-resistant client: {client_id}")
            
            # Return server public keys
            return {
                "status": "registered",
                "server_public_keys": self.crypto_manager.get_public_keys(),
                "server_capabilities": {
                    "quantum_resistant": True,
                    "ml_dsa": True,
                    "ml_kem": True,
                    "algorithms": {
                        "signature": "ML-DSA-44 (FIPS 204)",
                        "key_encapsulation": "ML-KEM-768 (FIPS 203)"
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Client registration failed: {e}")
            raise
    
    def _get_initial_parameters(self) -> NDArrays:
        """Get initial model parameters"""
        try:
            model = self.model_fn()
            parameters = []
            for param in model.parameters():
                parameters.append(param.detach().cpu().numpy())
            return parameters
            
        except Exception as e:
            self.logger.error(f"Failed to get initial parameters: {e}")
            raise
    
    def _process_results(self, history) -> Dict[str, Any]:
        """Process federated learning results"""
        try:
            results = {
                "num_rounds": self.num_rounds,
                "registered_clients": len(self.registered_clients),
                "completion_time": datetime.now().isoformat(),
                "quantum_resistant": True,
                "history": {
                    "rounds": len(history.losses_distributed),
                    "final_loss": history.losses_distributed[-1][1] if history.losses_distributed else None,
                    "metrics": history.metrics_distributed
                }
            }
            
            # Add quantum crypto statistics
            results["crypto_stats"] = {
                "signatures_verified": getattr(self.crypto_manager, 'signatures_verified', 0),
                "secure_communications": getattr(self.crypto_manager, 'secure_comms', 0)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process results: {e}")
            return {"error": str(e)}
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status"""
        return {
            "registered_clients": len(self.registered_clients),
            "quantum_resistant": True,
            "server_keys": self.crypto_manager.get_public_keys(),
            "configuration": {
                "num_rounds": self.num_rounds,
                "min_fit_clients": self.min_fit_clients,
                "min_evaluate_clients": self.min_evaluate_clients
            }
        }
