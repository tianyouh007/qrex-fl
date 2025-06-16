"""
QREX-FL Compliance Engine
Multi-jurisdictional regulatory compliance automation
Based on FinCEN, FATF, MiCA standards
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

class JurisdictionType(Enum):
    """Supported regulatory jurisdictions"""
    US_FINCEN = "us_fincen"
    EU_MICA = "eu_mica"
    SINGAPORE_MAS = "singapore_mas"
    FATF_GLOBAL = "fatf_global"
    UK_FCA = "uk_fca"
    JAPAN_FSA = "japan_fsa"

@dataclass
class Transaction:
    """Cryptocurrency transaction representation"""
    tx_id: str
    from_address: str
    to_address: str
    amount: float
    currency: str
    timestamp: datetime
    features: Dict[str, Any] = field(default_factory=dict)
    blockchain: str = "bitcoin"
    confirmations: int = 0

@dataclass
class ComplianceResult:
    """Compliance assessment result"""
    transaction_id: str
    jurisdiction: JurisdictionType
    is_compliant: bool
    risk_score: float
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

class RegulatoryRuleEngine:
    """Base regulatory rule engine"""
    
    def __init__(self, jurisdiction: JurisdictionType):
        self.jurisdiction = jurisdiction
        self.rules = self._load_jurisdiction_rules()
        self.logger = logging.getLogger(__name__)
    
    def _load_jurisdiction_rules(self) -> Dict[str, Any]:
        """Load jurisdiction-specific rules and thresholds"""
        base_rules = {
            "aml_threshold": 10000.0,
            "kyc_required": True,
            "suspicious_patterns": [
                "rapid_succession_transactions",
                "round_amount_structuring", 
                "high_risk_jurisdiction",
                "mixer_interaction",
                "darkmarket_association"
            ],
            "reporting_requirements": {
                "sar_threshold": 5000.0,
                "ctr_threshold": 10000.0,
                "travel_rule_threshold": 3000.0
            },
            "risk_factors": {
                "high_frequency": 0.3,
                "round_amounts": 0.2,
                "mixer_usage": 0.8,
                "sanctioned_entity": 1.0,
                "unusual_pattern": 0.4
            }
        }
        
        # Jurisdiction-specific adjustments based on research
        if self.jurisdiction == JurisdictionType.EU_MICA:
            base_rules["aml_threshold"] = 1000.0  # EUR threshold per MiCA
            base_rules["reporting_requirements"]["travel_rule_threshold"] = 1000.0
            base_rules["kyc_enhanced"] = True
        elif self.jurisdiction == JurisdictionType.SINGAPORE_MAS:
            base_rules["aml_threshold"] = 5000.0  # SGD threshold  
            base_rules["licensing_required"] = True
        elif self.jurisdiction == JurisdictionType.JAPAN_FSA:
            base_rules["aml_threshold"] = 1000000.0  # JPY threshold
            base_rules["exchange_registration"] = True
        elif self.jurisdiction == JurisdictionType.UK_FCA:
            base_rules["aml_threshold"] = 10000.0  # GBP threshold
            base_rules["cryptoasset_registration"] = True
        
        return base_rules
    
    def evaluate_transaction(self, transaction: Transaction) -> ComplianceResult:
        """
        Evaluate transaction against regulatory rules
        
        Args:
            transaction: Transaction to evaluate
            
        Returns:
            Compliance assessment result
        """
        violations = []
        risk_score = 0.0
        
        try:
            # 1. AML threshold check
            if transaction.amount > self.rules["aml_threshold"]:
                violations.append(f"Amount {transaction.amount} exceeds AML threshold of {self.rules['aml_threshold']}")
                risk_score += self.rules["risk_factors"]["high_frequency"]
            
            # 2. Suspicious pattern detection
            suspicious_patterns = self._detect_suspicious_patterns(transaction)
            for pattern in suspicious_patterns:
                violations.append(f"Suspicious pattern detected: {pattern}")
                risk_score += self.rules["risk_factors"].get(pattern, 0.3)
            
            # 3. Round amount structuring
            if self._is_round_amount_structuring(transaction):
                violations.append("Potential amount structuring detected")
                risk_score += self.rules["risk_factors"]["round_amounts"]
            
            # 4. High-risk features analysis
            risk_score += self._analyze_transaction_features(transaction)
            
            # 5. Travel Rule compliance
            if transaction.amount >= self.rules["reporting_requirements"]["travel_rule_threshold"]:
                if not self._check_travel_rule_compliance(transaction):
                    violations.append("Travel Rule compliance required")
                    risk_score += 0.2
            
            # 6. Sanctions screening
            if self._check_sanctions_lists(transaction):
                violations.append("Sanctioned entity detected")
                risk_score += self.rules["risk_factors"]["sanctioned_entity"]
            
            # Normalize risk score
            risk_score = min(risk_score, 1.0)
            is_compliant = len(violations) == 0 and risk_score < 0.5
            
            result = ComplianceResult(
                transaction_id=transaction.tx_id,
                jurisdiction=self.jurisdiction,
                is_compliant=is_compliant,
                risk_score=risk_score,
                violations=violations,
                recommendations=self._generate_recommendations(violations, risk_score),
                confidence=self._calculate_confidence(transaction)
            )
            
            self.logger.debug(f"Evaluated transaction {transaction.tx_id}: compliant={is_compliant}, risk={risk_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Transaction evaluation failed: {e}")
            raise
    
    def _detect_suspicious_patterns(self, transaction: Transaction) -> List[str]:
        """Detect suspicious transaction patterns"""
        patterns = []
        
        # High-frequency trading pattern
        if transaction.features.get("tx_frequency_1h", 0) > 10:
            patterns.append("high_frequency")
        
        # Mixer interaction
        if transaction.features.get("mixer_interaction", False):
            patterns.append("mixer_usage")
        
        # Sanctioned entity interaction
        if transaction.features.get("sanctioned_entity", False):
            patterns.append("sanctioned_entity")
        
        # Unusual timing patterns
        if transaction.features.get("unusual_timing", False):
            patterns.append("unusual_pattern")
        
        # Cross-chain laundering
        if transaction.features.get("cross_chain_activity", False):
            patterns.append("cross_chain_laundering")
        
        return patterns
    
    def _is_round_amount_structuring(self, transaction: Transaction) -> bool:
        """Check for round amount structuring"""
        # Round amounts under threshold (potential structuring)
        return (transaction.amount % 1000 == 0 and 
                transaction.amount < self.rules["aml_threshold"])
    
    def _analyze_transaction_features(self, transaction: Transaction) -> float:
        """Analyze transaction features for risk assessment"""
        risk = 0.0
        
        # Address reuse risk
        if transaction.features.get("address_reuse_count", 0) > 5:
            risk += 0.1
        
        # Transaction value patterns
        if transaction.features.get("value_similarity_score", 0) > 0.8:
            risk += 0.2
        
        # Geographic risk
        if transaction.features.get("high_risk_geography", False):
            risk += 0.3
        
        # Privacy coin usage
        if transaction.currency.lower() in ["monero", "zcash", "dash"]:
            risk += 0.4
        
        return min(risk, 0.5)
    
    def _check_travel_rule_compliance(self, transaction: Transaction) -> bool:
        """Check Travel Rule compliance requirements"""
        # Check if required information is present
        required_fields = ["originator_info", "beneficiary_info"]
        return all(transaction.features.get(field) for field in required_fields)
    
    def _check_sanctions_lists(self, transaction: Transaction) -> bool:
        """Check against OFAC and other sanctions lists"""
        # In production, this would check against real sanctions lists
        sanctioned_addresses = transaction.features.get("sanctioned_addresses", [])
        return (transaction.from_address in sanctioned_addresses or 
                transaction.to_address in sanctioned_addresses)
    
    def _generate_recommendations(self, violations: List[str], risk_score: float) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if violations:
            recommendations.append("Enhanced due diligence required")
            
            if risk_score > 0.7:
                recommendations.append("File Suspicious Activity Report (SAR)")
                recommendations.append("Consider transaction blocking")
                recommendations.append("Notify law enforcement if required")
            elif risk_score > 0.5:
                recommendations.append("Additional monitoring required")
                recommendations.append("Customer verification recommended")
                recommendations.append("Review transaction patterns")
        
        if "Travel Rule" in str(violations):
            recommendations.append("Obtain beneficiary information per Travel Rule")
        
        if "sanctioned" in str(violations).lower():
            recommendations.append("Immediately block transaction")
            recommendations.append("Report to OFAC")
        
        return recommendations
    
    def _calculate_confidence(self, transaction: Transaction) -> float:
        """Calculate confidence in assessment"""
        # Higher confidence with more transaction features
        feature_count = len([v for v in transaction.features.values() if v is not None])
        base_confidence = min(0.5 + (feature_count * 0.05), 1.0)
        
        # Reduce confidence for new addresses
        if transaction.features.get("address_age_days", 365) < 30:
            base_confidence *= 0.9
        
        # Reduce confidence for low confirmation count
        if transaction.confirmations < 3:
            base_confidence *= 0.8
        
        return base_confidence

class ComplianceEngine:
    """Main compliance engine coordinating multiple jurisdictions"""
    
    def __init__(self):
        self.rule_engines = {
            jurisdiction: RegulatoryRuleEngine(jurisdiction) 
            for jurisdiction in JurisdictionType
        }
        self.logger = logging.getLogger(__name__)
        
        # Compliance statistics
        self.total_evaluations = 0
        self.compliance_rate = 0.0
        self.risk_distribution = {"low": 0, "medium": 0, "high": 0}
        
    def validate_transaction(self, 
                           transaction: Transaction,
                           jurisdictions: List[JurisdictionType]) -> ComplianceResult:
        """
        Validate transaction across multiple jurisdictions
        
        Args:
            transaction: Transaction to validate
            jurisdictions: List of jurisdictions to check
            
        Returns:
            Unified compliance result
        """
        try:
            results = []
            
            # Evaluate against each jurisdiction
            for jurisdiction in jurisdictions:
                if jurisdiction in self.rule_engines:
                    result = self.rule_engines[jurisdiction].evaluate_transaction(transaction)
                    results.append(result)
            
            # Resolve conflicts using most restrictive approach
            unified_result = self._resolve_jurisdictional_conflicts(results)
            
            # Update statistics
            self._update_statistics(unified_result)
            
            self.logger.info(f"Validated transaction {transaction.tx_id} across {len(jurisdictions)} jurisdictions")
            return unified_result
            
        except Exception as e:
            self.logger.error(f"Multi-jurisdictional validation failed: {e}")
            raise
    
    def _resolve_jurisdictional_conflicts(self, results: List[ComplianceResult]) -> ComplianceResult:
        """Resolve conflicts between jurisdictional requirements"""
        if not results:
            raise ValueError("No compliance results to resolve")
        
        # Use most restrictive approach
        overall_compliant = all(r.is_compliant for r in results)
        max_risk_score = max(r.risk_score for r in results)
        min_confidence = min(r.confidence for r in results)
        
        # Combine all violations and recommendations
        all_violations = []
        all_recommendations = []
        
        for result in results:
            all_violations.extend(result.violations)
            all_recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        unique_violations = list(dict.fromkeys(all_violations))
        unique_recommendations = list(dict.fromkeys(all_recommendations))
        
        return ComplianceResult(
            transaction_id=results[0].transaction_id,
            jurisdiction=JurisdictionType.FATF_GLOBAL,  # Global standard
            is_compliant=overall_compliant,
            risk_score=max_risk_score,
            violations=unique_violations,
            recommendations=unique_recommendations,
            confidence=min_confidence,
            timestamp=datetime.now()
        )
    
    def _update_statistics(self, result: ComplianceResult):
        """Update compliance statistics"""
        self.total_evaluations += 1
        
        # Update compliance rate
        compliant_count = sum(1 for _ in range(self.total_evaluations) if result.is_compliant)
        self.compliance_rate = compliant_count / self.total_evaluations
        
        # Update risk distribution
        if result.risk_score < 0.3:
            self.risk_distribution["low"] += 1
        elif result.risk_score < 0.7:
            self.risk_distribution["medium"] += 1
        else:
            self.risk_distribution["high"] += 1
    
    def generate_compliance_report(self, results: List[ComplianceResult]) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        if not results:
            return {"error": "No compliance results to report"}
        
        total_transactions = len(results)
        compliant_transactions = sum(1 for r in results if r.is_compliant)
        avg_risk_score = sum(r.risk_score for r in results) / total_transactions
        avg_confidence = sum(r.confidence for r in results) / total_transactions
        
        # Analyze violations
        violation_counts = {}
        recommendation_counts = {}
        
        for result in results:
            for violation in result.violations:
                violation_counts[violation] = violation_counts.get(violation, 0) + 1
            for recommendation in result.recommendations:
                recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1
        
        # Risk distribution
        risk_distribution = {
            "low_risk": sum(1 for r in results if r.risk_score < 0.3),
            "medium_risk": sum(1 for r in results if 0.3 <= r.risk_score < 0.7),
            "high_risk": sum(1 for r in results if r.risk_score >= 0.7)
        }
        
        report = {
            "summary": {
                "total_transactions": total_transactions,
                "compliant_transactions": compliant_transactions,
                "non_compliant_transactions": total_transactions - compliant_transactions,
                "compliance_rate": compliant_transactions / total_transactions,
                "average_risk_score": avg_risk_score,
                "average_confidence": avg_confidence
            },
            "violations": {
                "most_common": sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                "total_unique": len(violation_counts)
            },
            "recommendations": {
                "most_frequent": sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                "total_unique": len(recommendation_counts)
            },
            "risk_distribution": risk_distribution,
            "jurisdictional_analysis": self._analyze_jurisdictional_patterns(results),
            "generated_at": datetime.now().isoformat(),
            "report_version": "1.0"
        }
        
        return report
    
    def _analyze_jurisdictional_patterns(self, results: List[ComplianceResult]) -> Dict[str, Any]:
        """Analyze patterns across jurisdictions"""
        jurisdiction_stats = {}
        
        for result in results:
            jurisdiction = result.jurisdiction.value
            if jurisdiction not in jurisdiction_stats:
                jurisdiction_stats[jurisdiction] = {
                    "total": 0,
                    "compliant": 0,
                    "avg_risk": 0.0,
                    "common_violations": {}
                }
            
            stats = jurisdiction_stats[jurisdiction]
            stats["total"] += 1
            if result.is_compliant:
                stats["compliant"] += 1
            stats["avg_risk"] = (stats["avg_risk"] * (stats["total"] - 1) + result.risk_score) / stats["total"]
            
            for violation in result.violations:
                stats["common_violations"][violation] = stats["common_violations"].get(violation, 0) + 1
        
        # Calculate compliance rates
        for jurisdiction, stats in jurisdiction_stats.items():
            stats["compliance_rate"] = stats["compliant"] / stats["total"] if stats["total"] > 0 else 0
        
        return jurisdiction_stats
