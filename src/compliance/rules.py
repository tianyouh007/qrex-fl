"""
QREX-FL Regulatory Rule Engine
Implements multi-jurisdictional compliance rules for cryptocurrency transactions
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class JurisdictionType(Enum):
    """Supported regulatory jurisdictions"""
    US_FINCEN = "us_fincen"
    EU_MICA = "eu_mica"
    SINGAPORE_MAS = "singapore_mas"
    FATF_GLOBAL = "fatf_global"

class RuleViolation:
    """Represents a regulatory rule violation"""
    
    def __init__(self, rule_id: str, jurisdiction: JurisdictionType, 
                 description: str, severity: str, risk_score: float):
        self.rule_id = rule_id
        self.jurisdiction = jurisdiction
        self.description = description
        self.severity = severity  # "low", "medium", "high", "critical"
        self.risk_score = risk_score

class RegulatoryRuleEngine:
    """Multi-jurisdictional regulatory rule engine for QREX-FL"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict[JurisdictionType, Dict[str, Any]]:
        """Load regulatory rules for each jurisdiction"""
        return {
            JurisdictionType.US_FINCEN: {
                "large_transaction_threshold": 10000.0,
                "structuring_threshold": 9000.0,
                "suspicious_activity_threshold": 5000.0,
                "kyc_required_threshold": 3000.0,
                "daily_limit": 25000.0,
                "sanctioned_countries": ["IR", "KP", "CU"],
                "mixer_interaction_penalty": 0.8,
                "high_risk_jurisdiction_penalty": 0.6
            },
            JurisdictionType.EU_MICA: {
                "large_transaction_threshold": 1000.0,  # EUR equivalent
                "structuring_threshold": 900.0,
                "suspicious_activity_threshold": 500.0,
                "kyc_required_threshold": 1000.0,
                "daily_limit": 15000.0,
                "sanctioned_countries": ["IR", "KP", "CU", "RU", "BY"],
                "mixer_interaction_penalty": 0.9,
                "high_risk_jurisdiction_penalty": 0.7
            },
            JurisdictionType.SINGAPORE_MAS: {
                "large_transaction_threshold": 13000.0,  # SGD equivalent
                "structuring_threshold": 12000.0,
                "suspicious_activity_threshold": 6000.0,
                "kyc_required_threshold": 5000.0,
                "daily_limit": 50000.0,
                "sanctioned_countries": ["IR", "KP"],
                "mixer_interaction_penalty": 0.7,
                "high_risk_jurisdiction_penalty": 0.5
            },
            JurisdictionType.FATF_GLOBAL: {
                "large_transaction_threshold": 15000.0,
                "structuring_threshold": 14000.0,
                "suspicious_activity_threshold": 10000.0,
                "kyc_required_threshold": 1000.0,
                "daily_limit": 100000.0,
                "sanctioned_countries": ["IR", "KP", "AF"],
                "mixer_interaction_penalty": 0.8,
                "high_risk_jurisdiction_penalty": 0.6
            }
        }
    
    def evaluate_transaction(self, transaction: Dict[str, Any], 
                           jurisdiction: JurisdictionType) -> List[RuleViolation]:
        """
        Evaluate a transaction against regulatory rules
        
        Args:
            transaction: Transaction data
            jurisdiction: Regulatory jurisdiction to check against
            
        Returns:
            List of rule violations
        """
        violations = []
        rules = self.rules.get(jurisdiction, {})
        
        if not rules:
            self.logger.warning(f"No rules found for jurisdiction: {jurisdiction}")
            return violations
        
        # Check large transaction reporting
        if transaction.get("amount", 0) >= rules.get("large_transaction_threshold", 10000):
            violations.append(RuleViolation(
                rule_id="LARGE_TRANSACTION",
                jurisdiction=jurisdiction,
                description=f"Transaction exceeds reporting threshold of {rules['large_transaction_threshold']}",
                severity="medium",
                risk_score=0.4
            ))
        
        # Check for potential structuring
        if (rules.get("structuring_threshold", 9000) <= 
            transaction.get("amount", 0) < rules.get("large_transaction_threshold", 10000)):
            violations.append(RuleViolation(
                rule_id="POTENTIAL_STRUCTURING",
                jurisdiction=jurisdiction,
                description="Transaction amount suggests potential structuring",
                severity="high",
                risk_score=0.7
            ))
        
        # Check mixer interaction
        if transaction.get("features", {}).get("mixer_interaction", False):
            penalty = rules.get("mixer_interaction_penalty", 0.8)
            violations.append(RuleViolation(
                rule_id="MIXER_INTERACTION",
                jurisdiction=jurisdiction,
                description="Transaction involves cryptocurrency mixer",
                severity="high",
                risk_score=penalty
            ))
        
        # Check sanctioned entities
        if transaction.get("features", {}).get("sanctioned_entity", False):
            violations.append(RuleViolation(
                rule_id="SANCTIONED_ENTITY",
                jurisdiction=jurisdiction,
                description="Transaction involves sanctioned entity",
                severity="critical",
                risk_score=0.95
            ))
        
        # Check KYC requirements
        if (transaction.get("amount", 0) >= rules.get("kyc_required_threshold", 3000) and
            not transaction.get("features", {}).get("kyc_verified", False)):
            violations.append(RuleViolation(
                rule_id="KYC_REQUIRED",
                jurisdiction=jurisdiction,
                description="Transaction requires KYC verification",
                severity="medium",
                risk_score=0.5
            ))
        
        # Check for high-frequency patterns (simplified)
        if transaction.get("features", {}).get("high_frequency_pattern", False):
            violations.append(RuleViolation(
                rule_id="HIGH_FREQUENCY_PATTERN",
                jurisdiction=jurisdiction,
                description="Transaction part of high-frequency pattern",
                severity="medium",
                risk_score=0.4
            ))
        
        return violations
    
    def calculate_compliance_score(self, violations: List[RuleViolation]) -> float:
        """
        Calculate overall compliance score based on violations
        
        Args:
            violations: List of rule violations
            
        Returns:
            Compliance score (0.0 = non-compliant, 1.0 = fully compliant)
        """
        if not violations:
            return 1.0
        
        # Calculate weighted penalty based on severity
        total_penalty = 0.0
        severity_weights = {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.6,
            "critical": 1.0
        }
        
        for violation in violations:
            weight = severity_weights.get(violation.severity, 0.5)
            total_penalty += violation.risk_score * weight
        
        # Normalize to 0-1 scale
        compliance_score = max(0.0, 1.0 - min(1.0, total_penalty))
        return compliance_score
    
    def generate_compliance_report(self, transaction: Dict[str, Any],
                                 jurisdictions: List[JurisdictionType]) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report for multiple jurisdictions
        
        Args:
            transaction: Transaction data
            jurisdictions: List of jurisdictions to check
            
        Returns:
            Compliance report
        """
        report = {
            "transaction_id": transaction.get("tx_id", "unknown"),
            "evaluated_at": datetime.now().isoformat(),
            "jurisdictions": {},
            "overall_compliance": True,
            "overall_risk_score": 0.0,
            "recommendations": []
        }
        
        all_violations = []
        jurisdiction_scores = []
        
        for jurisdiction in jurisdictions:
            violations = self.evaluate_transaction(transaction, jurisdiction)
            compliance_score = self.calculate_compliance_score(violations)
            
            jurisdiction_scores.append(compliance_score)
            all_violations.extend(violations)
            
            report["jurisdictions"][jurisdiction.value] = {
                "compliance_score": compliance_score,
                "violations": [
                    {
                        "rule_id": v.rule_id,
                        "description": v.description,
                        "severity": v.severity,
                        "risk_score": v.risk_score
                    }
                    for v in violations
                ],
                "is_compliant": compliance_score >= 0.7
            }
        
        # Calculate overall metrics
        if jurisdiction_scores:
            report["overall_risk_score"] = 1.0 - min(jurisdiction_scores)
            report["overall_compliance"] = all(score >= 0.7 for score in jurisdiction_scores)
        
        # Generate recommendations
        if not report["overall_compliance"]:
            report["recommendations"].append("Transaction requires manual review")
            
            if any(v.severity == "critical" for v in all_violations):
                report["recommendations"].append("IMMEDIATE ACTION REQUIRED: Critical violation detected")
            
            if any(v.rule_id == "KYC_REQUIRED" for v in all_violations):
                report["recommendations"].append("Complete KYC verification before processing")
                
            if any(v.rule_id == "MIXER_INTERACTION" for v in all_violations):
                report["recommendations"].append("Enhanced due diligence required for mixer interaction")
        
        return report
    
    def update_rules(self, jurisdiction: JurisdictionType, 
                    rule_updates: Dict[str, Any]) -> bool:
        """
        Update rules for a specific jurisdiction
        
        Args:
            jurisdiction: Jurisdiction to update
            rule_updates: Rule updates to apply
            
        Returns:
            Success status
        """
        try:
            if jurisdiction not in self.rules:
                self.rules[jurisdiction] = {}
            
            self.rules[jurisdiction].update(rule_updates)
            self.logger.info(f"Updated rules for jurisdiction: {jurisdiction}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update rules for {jurisdiction}: {e}")
            return False
    
    def get_jurisdiction_rules(self, jurisdiction: JurisdictionType) -> Dict[str, Any]:
        """Get rules for a specific jurisdiction"""
        return self.rules.get(jurisdiction, {})
    
    def list_supported_jurisdictions(self) -> List[JurisdictionType]:
        """Get list of supported jurisdictions"""
        return list(self.rules.keys())
