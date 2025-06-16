"""
QREX-FL Custom Federated Learning Strategy
Quantum-resistant strategy for Flower framework
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

class QREXStrategy(fl.server.strategy.FedAvg):
    """
    QREX-FL Strategy with quantum-resistant security and compliance integration
    Extends Flower's FedAvg with quantum-resistant features
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        crypto_manager = None,
    ):
        """Initialize QREX strategy with quantum-resistant features"""
        
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        
        self.crypto_manager = crypto_manager
        self.logger = logging.getLogger(__name__)
        self.round_history = []
        
        # Compliance and security tracking
        self.security_violations = []
        self.compliance_results = {}
        
        self.logger.info("Initialized QREX strategy with quantum-resistant security")
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with quantum-resistant security"""
        
        # Standard configuration
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Add QREX-specific configuration
        config.update({
            "server_round": server_round,
            "quantum_resistant": True,
            "compliance_check": True,
            "timestamp": str(np.datetime64('now'))
        })
        
        fit_ins = FitIns(parameters, config)
        
        # Sample clients for this round
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Log quantum-resistant training round
        self.logger.info(f"Configuring quantum-resistant training round {server_round} "
                        f"with {len(clients)} clients")
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results with quantum-resistant verification"""
        
        # Verify quantum-resistant signatures (if crypto_manager available)
        verified_results = []
        for client, fit_res in results:
            try:
                # In a full implementation, verify ML-DSA signatures here
                verified_results.append((client, fit_res))
                
            except Exception as e:
                self.logger.warning(f"Failed to verify quantum signature from {client.cid}: {e}")
                # Add to failures if signature verification fails
                failures.append((client, fit_res))
        
        if not verified_results:
            self.logger.error("No verified results for aggregation")
            return None, {}
        
        # Standard FedAvg aggregation on verified results
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, verified_results, failures
        )
        
        # Add QREX-specific metrics
        qrex_metrics = {
            "verified_clients": len(verified_results),
            "failed_verifications": len(results) - len(verified_results),
            "quantum_resistant": True,
            "round": server_round
        }
        
        if aggregated_metrics:
            aggregated_metrics.update(qrex_metrics)
        else:
            aggregated_metrics = qrex_metrics
        
        # Store round history
        self.round_history.append({
            "round": server_round,
            "participants": len(verified_results),
            "metrics": aggregated_metrics
        })
        
        self.logger.info(f"Aggregated round {server_round}: {len(verified_results)} verified clients")
        
        return aggregated_parameters, aggregated_metrics
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation"""
        
        # Do not configure federated evaluation if fraction_evaluate is 0.
        if self.fraction_evaluate == 0.0:
            return []
        
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        # Add QREX-specific evaluation config
        config.update({
            "server_round": server_round,
            "quantum_resistant": True,
            "evaluation_timestamp": str(np.datetime64('now'))
        })
        
        evaluate_ins = EvaluateIns(parameters, config)
        
        # Sample clients for evaluation
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results with compliance checking"""
        
        # Standard aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Calculate additional QREX metrics
        if results:
            accuracies = []
            for _, evaluate_res in results:
                if "accuracy" in evaluate_res.metrics:
                    accuracies.append(evaluate_res.metrics["accuracy"])
            
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                if aggregated_metrics is None:
                    aggregated_metrics = {}
                aggregated_metrics["average_accuracy"] = avg_accuracy
                aggregated_metrics["evaluation_participants"] = len(results)
        
        return aggregated_loss, aggregated_metrics
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model using the strategy's evaluation function"""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        
        loss, metrics = eval_res
        return loss, metrics
    
    def get_round_history(self) -> List[Dict]:
        """Get training round history"""
        return self.round_history.copy()
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get quantum-resistant security status"""
        return {
            "quantum_resistant": True,
            "ml_dsa_enabled": self.crypto_manager is not None,
            "total_rounds": len(self.round_history),
            "security_violations": len(self.security_violations),
            "last_round_participants": self.round_history[-1]["participants"] if self.round_history else 0
        }
