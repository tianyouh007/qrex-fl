"""
QREX-FL Transaction Validator
Real-time transaction validation with ML integration
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from .engine import ComplianceEngine, Transaction, JurisdictionType, ComplianceResult

class TransactionValidator:
    """
    Real-time transaction validator with ML-based risk assessment
    """
    
    def __init__(self, compliance_engine: Optional[ComplianceEngine] = None):
        self.compliance_engine = compliance_engine or ComplianceEngine()
        self.logger = logging.getLogger(__name__)
        
        # Validation statistics
        self.validation_history = []
        self.blocked_transactions = []
        self.flagged_transactions = []
        
    def validate_real_time(self, 
                          transaction_data: Dict[str, Any],
                          jurisdictions: List[JurisdictionType],
                          ml_risk_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform real-time transaction validation
        
        Args:
            transaction_data: Raw transaction data
            jurisdictions: Applicable jurisdictions
            ml_risk_score: Optional ML model risk score
            
        Returns:
            Validation result with action recommendations
        """
        try:
            # Parse transaction data
            transaction = self._parse_transaction_data(transaction_data)
            
            # Add ML risk score if provided
            if ml_risk_score is not None:
                transaction.features["ml_risk_score"] = ml_risk_score
            
            # Perform compliance validation
            compliance_result = self.compliance_engine.validate_transaction(
                transaction, jurisdictions
            )
            
            # Determine action based on risk level
            action = self._determine_action(compliance_result, ml_risk_score)
            
            # Create validation response
            validation_result = {
                "transaction_id": transaction.tx_id,
                "action": action["type"],
                "risk_level": self._categorize_risk(compliance_result.risk_score),
                "compliance_result": {
                    "is_compliant": compliance_result.is_compliant,
                    "risk_score": compliance_result.risk_score,
                    "violations": compliance_result.violations,
                    "recommendations": compliance_result.recommendations,
                    "confidence": compliance_result.confidence
                },
                "ml_assessment": {
                    "risk_score": ml_risk_score,
                    "features_analyzed": len(transaction.features)
                } if ml_risk_score is not None else None,
                "action_details": action,
                "validated_at": datetime.now().isoformat()
            }
            
            # Update validation history
            self._update_validation_history(validation_result)
            
            self.logger.info(f"Validated transaction {transaction.tx_id}: {action['type']}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Real-time validation failed: {e}")
            raise
    
    def _parse_transaction_data(self, data: Dict[str, Any]) -> Transaction:
        """Parse raw transaction data into Transaction object"""
        return Transaction(
            tx_id=data.get("tx_id", ""),
            from_address=data.get("from_address", ""),
            to_address=data.get("to_address", ""),
            amount=float(data.get("amount", 0)),
            currency=data.get("currency", "BTC"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            features=data.get("features", {}),
            blockchain=data.get("blockchain", "bitcoin"),
            confirmations=int(data.get("confirmations", 0))
        )
    
    def _determine_action(self, 
                         compliance_result: ComplianceResult, 
                         ml_risk_score: Optional[float]) -> Dict[str, Any]:
        """Determine action based on compliance and ML results"""
        
        risk_score = compliance_result.risk_score
        is_compliant = compliance_result.is_compliant
        
        # Combine compliance and ML risk scores
        if ml_risk_score is not None:
            # Weighted combination (compliance 60%, ML 40%)
            combined_risk = 0.6 * risk_score + 0.4 * ml_risk_score
        else:
            combined_risk = risk_score
        
        # Determine action based on risk level and compliance
        if not is_compliant or combined_risk >= 0.8:
            action_type = "BLOCK"
            priority = "HIGH"
            reason = "High risk transaction with compliance violations"
        elif combined_risk >= 0.6:
            action_type = "FLAG"
            priority = "MEDIUM"
            reason = "Medium risk transaction requiring review"
        elif combined_risk >= 0.3:
            action_type = "MONITOR"
            priority = "LOW"
            reason = "Low-medium risk transaction for monitoring"
        else:
            action_type = "APPROVE"
            priority = "LOW"
            reason = "Low risk compliant transaction"
        
        return {
            "type": action_type,
            "priority": priority,
            "reason": reason,
            "combined_risk_score": combined_risk,
            "requires_manual_review": action_type in ["BLOCK", "FLAG"],
            "auto_approval": action_type == "APPROVE"
        }
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into human-readable levels"""
        if risk_score >= 0.7:
            return "HIGH"
        elif risk_score >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _update_validation_history(self, validation_result: Dict[str, Any]):
        """Update validation history and statistics"""
        self.validation_history.append(validation_result)
        
        # Track blocked and flagged transactions
        action = validation_result["action"]
        if action == "BLOCK":
            self.blocked_transactions.append(validation_result["transaction_id"])
        elif action == "FLAG":
            self.flagged_transactions.append(validation_result["transaction_id"])
        
        # Limit history size
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self.validation_history:
            return {"error": "No validation history available"}
        
        total_validations = len(self.validation_history)
        action_counts = {}
        risk_level_counts = {}
        
        for validation in self.validation_history:
            action = validation["action"]
            risk_level = validation["risk_level"]
            
            action_counts[action] = action_counts.get(action, 0) + 1
            risk_level_counts[risk_level] = risk_level_counts.get(risk_level, 0) + 1
        
        return {
            "total_validations": total_validations,
            "action_distribution": action_counts,
            "risk_distribution": risk_level_counts,
            "blocked_transactions": len(self.blocked_transactions),
            "flagged_transactions": len(self.flagged_transactions),
            "approval_rate": action_counts.get("APPROVE", 0) / total_validations,
            "block_rate": action_counts.get("BLOCK", 0) / total_validations,
            "recent_validations": self.validation_history[-10:]
        }
    
    def batch_validate(self, 
                      transactions: List[Dict[str, Any]],
                      jurisdictions: List[JurisdictionType]) -> List[Dict[str, Any]]:
        """
        Validate multiple transactions in batch
        
        Args:
            transactions: List of transaction data
            jurisdictions: Applicable jurisdictions
            
        Returns:
            List of validation results
        """
        results = []
        
        for tx_data in transactions:
            try:
                result = self.validate_real_time(tx_data, jurisdictions)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch validation failed for transaction {tx_data.get('tx_id')}: {e}")
                results.append({
                    "transaction_id": tx_data.get("tx_id"),
                    "action": "ERROR",
                    "error": str(e)
                })
        
        return results
