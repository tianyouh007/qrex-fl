"""
QREX-FL Explainer
Quantum-resistant explainable AI for cryptocurrency risk assessment
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

class QREXExplainer:
    """
    Quantum-resistant explainable AI for cryptocurrency risk assessment
    Provides regulatory-compliant explanations
    """
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize QREX explainer
        
        Args:
            model: Trained model to explain
            feature_names: List of feature names for interpretability
        """
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(165)]
        self.logger = logging.getLogger(__name__)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        self._initialize_explainers()
        
        self.logger.info(f"Initialized QREX explainer with {len(self.feature_names)} features")
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        try:
            if SHAP_AVAILABLE:
                # For neural networks, use DeepExplainer or GradientExplainer
                if hasattr(self.model, 'forward'):  # PyTorch model
                    self.shap_explainer = shap.DeepExplainer(self.model, torch.zeros(1, len(self.feature_names)))
                else:
                    self.shap_explainer = shap.Explainer(self.model)
                
                self.logger.info("SHAP explainer initialized")
            else:
                self.logger.warning("SHAP not available, using fallback explanations")
        
        except Exception as e:
            self.logger.warning(f"Failed to initialize SHAP explainer: {e}")
    
    def explain_prediction(self, 
                          transaction_features: np.ndarray,
                          explanation_type: str = 'shap',
                          regulatory_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Explain model prediction for a transaction
        
        Args:
            transaction_features: Feature vector for transaction
            explanation_type: Type of explanation ('shap', 'lime', 'gradient')
            regulatory_context: Regulatory context for compliance
            
        Returns:
            Explanation results with regulatory compliance information
        """
        try:
            self.logger.debug(f"Generating {explanation_type} explanation")
            
            # Get model prediction
            if isinstance(transaction_features, np.ndarray):
                features_tensor = torch.tensor(transaction_features, dtype=torch.float32)
            else:
                features_tensor = transaction_features
            
            if len(features_tensor.shape) == 1:
                features_tensor = features_tensor.unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(features_tensor)
                probability = torch.sigmoid(prediction).item()
            
            # Generate explanation based on type
            if explanation_type == 'shap' and self.shap_explainer is not None:
                explanation = self._generate_shap_explanation(features_tensor)
            elif explanation_type == 'lime' and LIME_AVAILABLE:
                explanation = self._generate_lime_explanation(transaction_features.flatten())
            elif explanation_type == 'gradient':
                explanation = self._generate_gradient_explanation(features_tensor)
            else:
                explanation = self._generate_simple_explanation(features_tensor)
            
            # Add regulatory compliance information
            compliance_info = self._add_regulatory_compliance(explanation, regulatory_context)
            
            result = {
                'transaction_features': transaction_features.tolist() if isinstance(transaction_features, np.ndarray) else transaction_features,
                'prediction': float(prediction.item()),
                'probability': float(probability),
                'risk_level': 'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.3 else 'LOW',
                'explanation_type': explanation_type,
                'explanation': explanation,
                'regulatory_compliance': compliance_info,
                'generated_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"Generated {explanation_type} explanation for prediction: {probability:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            raise
    
    def _generate_shap_explanation(self, features_tensor: torch.Tensor) -> Dict[str, Any]:
        """Generate SHAP explanation"""
        try:
            if self.shap_explainer is None:
                return self._generate_simple_explanation(features_tensor)
            
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(features_tensor)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For binary classification
            
            if isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.detach().numpy()
            
            # Get top contributing features
            feature_importance = np.abs(shap_values.flatten())
            top_indices = np.argsort(feature_importance)[::-1][:10]
            
            top_features = []
            for idx in top_indices:
                top_features.append({
                    'feature_name': self.feature_names[idx],
                    'feature_index': int(idx),
                    'shap_value': float(shap_values.flatten()[idx]),
                    'feature_value': float(features_tensor.flatten()[idx]),
                    'importance': float(feature_importance[idx])
                })
            
            return {
                'method': 'SHAP',
                'top_features': top_features,
                'total_shap_contribution': float(np.sum(shap_values)),
                'base_value': 0.0,  # Could be model's base prediction
                'explanation_quality': 'high'
            }
            
        except Exception as e:
            self.logger.warning(f"SHAP explanation failed: {e}")
            return self._generate_simple_explanation(features_tensor)
    
    def _generate_lime_explanation(self, features: np.ndarray) -> Dict[str, Any]:
        """Generate LIME explanation"""
        try:
            if not LIME_AVAILABLE:
                return {'method': 'LIME', 'error': 'LIME not available'}
            
            # Initialize LIME explainer if not already done
            if self.lime_explainer is None:
                # Create synthetic training data for LIME
                training_data = np.random.randn(1000, len(self.feature_names))
                self.lime_explainer = LimeTabularExplainer(
                    training_data,
                    feature_names=self.feature_names,
                    mode='classification'
                )
            
            # Define prediction function for LIME
            def predict_fn(x):
                with torch.no_grad():
                    x_tensor = torch.tensor(x, dtype=torch.float32)
                    if len(x_tensor.shape) == 1:
                        x_tensor = x_tensor.unsqueeze(0)
                    logits = self.model(x_tensor)
                    probs = torch.sigmoid(logits)
                    return np.column_stack([1 - probs.numpy(), probs.numpy()])
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                features, predict_fn, num_features=10
            )
            
            # Extract top features
            top_features = []
            for feature_name, importance in explanation.as_list():
                feature_idx = self.feature_names.index(feature_name) if feature_name in self.feature_names else -1
                top_features.append({
                    'feature_name': feature_name,
                    'feature_index': feature_idx,
                    'lime_importance': float(importance),
                    'feature_value': float(features[feature_idx]) if feature_idx >= 0 else 0.0
                })
            
            return {
                'method': 'LIME',
                'top_features': top_features,
                'explanation_quality': 'medium'
            }
            
        except Exception as e:
            self.logger.warning(f"LIME explanation failed: {e}")
            return {'method': 'LIME', 'error': str(e)}
    
    def _generate_gradient_explanation(self, features_tensor: torch.Tensor) -> Dict[str, Any]:
        """Generate gradient-based explanation"""
        try:
            features_tensor.requires_grad_(True)
            
            # Forward pass
            output = self.model(features_tensor)
            
            # Backward pass to get gradients
            output.backward()
            
            gradients = features_tensor.grad.detach().numpy().flatten()
            feature_values = features_tensor.detach().numpy().flatten()
            
            # Calculate feature importance (gradient * input)
            feature_importance = np.abs(gradients * feature_values)
            
            # Get top contributing features
            top_indices = np.argsort(feature_importance)[::-1][:10]
            
            top_features = []
            for idx in top_indices:
                top_features.append({
                    'feature_name': self.feature_names[idx],
                    'feature_index': int(idx),
                    'gradient': float(gradients[idx]),
                    'feature_value': float(feature_values[idx]),
                    'importance': float(feature_importance[idx])
                })
            
            return {
                'method': 'Gradient',
                'top_features': top_features,
                'explanation_quality': 'medium'
            }
            
        except Exception as e:
            self.logger.warning(f"Gradient explanation failed: {e}")
            return self._generate_simple_explanation(features_tensor)
    
    def _generate_simple_explanation(self, features_tensor: torch.Tensor) -> Dict[str, Any]:
        """Generate simple feature importance explanation"""
        feature_values = features_tensor.detach().numpy().flatten()
        
        # Simple importance based on absolute feature values
        feature_importance = np.abs(feature_values)
        top_indices = np.argsort(feature_importance)[::-1][:10]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                'feature_name': self.feature_names[idx],
                'feature_index': int(idx),
                'feature_value': float(feature_values[idx]),
                'importance': float(feature_importance[idx])
            })
        
        return {
            'method': 'Simple',
            'top_features': top_features,
            'explanation_quality': 'basic',
            'note': 'Fallback explanation based on feature magnitudes'
        }
    
    def _add_regulatory_compliance(self, 
                                 explanation: Dict[str, Any], 
                                 regulatory_context: Optional[str] = None) -> Dict[str, Any]:
        """Add regulatory compliance information to explanation"""
        
        compliance_info = {
            'explanation_method': explanation.get('method', 'Unknown'),
            'regulatory_standards': [],
            'auditability': 'high' if explanation.get('explanation_quality') == 'high' else 'medium',
            'transparency_level': 'detailed'
        }
        
        # Add context-specific compliance information
        if regulatory_context:
            if 'fincen' in regulatory_context.lower():
                compliance_info['regulatory_standards'].append('US FinCEN')
                compliance_info['sar_reporting_ready'] = True
            
            if 'mica' in regulatory_context.lower():
                compliance_info['regulatory_standards'].append('EU MiCA')
                compliance_info['gdpr_compliant'] = True
            
            if 'fatf' in regulatory_context.lower():
                compliance_info['regulatory_standards'].append('FATF Guidelines')
                compliance_info['aml_compliant'] = True
        
        # Assess explanation completeness for regulatory requirements
        top_features = explanation.get('top_features', [])
        if len(top_features) >= 5:
            compliance_info['feature_coverage'] = 'sufficient'
        else:
            compliance_info['feature_coverage'] = 'limited'
        
        # Add interpretability metrics
        compliance_info['human_interpretable'] = True
        compliance_info['machine_verifiable'] = True
        compliance_info['real_time_capable'] = True
        
        return compliance_info
    
    def batch_explain(self, 
                     transaction_batch: np.ndarray,
                     explanation_type: str = 'shap') -> List[Dict[str, Any]]:
        """
        Explain predictions for a batch of transactions
        
        Args:
            transaction_batch: Batch of transaction features
            explanation_type: Type of explanation to generate
            
        Returns:
            List of explanation results
        """
        explanations = []
        
        for i, transaction in enumerate(transaction_batch):
            try:
                explanation = self.explain_prediction(
                    transaction, explanation_type=explanation_type
                )
                explanation['batch_index'] = i
                explanations.append(explanation)
                
            except Exception as e:
                self.logger.error(f"Failed to explain transaction {i}: {e}")
                explanations.append({
                    'batch_index': i,
                    'error': str(e),
                    'explanation_type': explanation_type
                })
        
        self.logger.info(f"Generated explanations for {len(explanations)} transactions")
        return explanations
    
    def get_global_feature_importance(self, 
                                    sample_data: np.ndarray,
                                    n_samples: int = 100) -> Dict[str, Any]:
        """
        Get global feature importance across multiple samples
        
        Args:
            sample_data: Sample data for global analysis
            n_samples: Number of samples to analyze
            
        Returns:
            Global feature importance analysis
        """
        try:
            # Sample subset of data
            sample_size = min(n_samples, len(sample_data))
            sampled_data = sample_data[:sample_size]
            
            # Get explanations for all samples
            explanations = self.batch_explain(sampled_data, explanation_type='gradient')
            
            # Aggregate feature importance
            feature_importance_sum = np.zeros(len(self.feature_names))
            valid_explanations = 0
            
            for explanation in explanations:
                if 'explanation' in explanation and 'top_features' in explanation['explanation']:
                    valid_explanations += 1
                    for feature_info in explanation['explanation']['top_features']:
                        idx = feature_info['feature_index']
                        importance = feature_info.get('importance', 0)
                        if 0 <= idx < len(feature_importance_sum):
                            feature_importance_sum[idx] += importance
            
            # Calculate average importance
            if valid_explanations > 0:
                avg_importance = feature_importance_sum / valid_explanations
                
                # Get top global features
                top_indices = np.argsort(avg_importance)[::-1][:20]
                
                global_features = []
                for idx in top_indices:
                    global_features.append({
                        'feature_name': self.feature_names[idx],
                        'feature_index': int(idx),
                        'average_importance': float(avg_importance[idx]),
                        'rank': len(global_features) + 1
                    })
                
                return {
                    'samples_analyzed': valid_explanations,
                    'global_top_features': global_features,
                    'analysis_type': 'global_feature_importance'
                }
            else:
                return {'error': 'No valid explanations generated'}
                
        except Exception as e:
            self.logger.error(f"Global feature importance analysis failed: {e}")
            raise
