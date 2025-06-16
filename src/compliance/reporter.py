"""
QREX-FL Compliance Reporter
Generates automated regulatory reports and filing documents
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ComplianceReporter:
    """Automated compliance reporting for QREX-FL"""
    
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def generate_sar_report(self, suspicious_transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate Suspicious Activity Report (SAR) for FinCEN
        
        Args:
            suspicious_transactions: List of flagged transactions
            
        Returns:
            SAR report data
        """
        report = {
            "report_type": "SAR",
            "filing_institution": "QREX-FL System",
            "report_date": datetime.now().isoformat(),
            "reporting_period": {
                "start": (datetime.now() - timedelta(days=30)).isoformat(),
                "end": datetime.now().isoformat()
            },
            "suspicious_activities": [],
            "total_amount": 0.0,
            "transaction_count": len(suspicious_transactions)
        }
        
        for tx in suspicious_transactions:
            activity = {
                "transaction_id": tx.get("tx_id"),
                "date": tx.get("timestamp"),
                "amount": tx.get("amount", 0),
                "currency": tx.get("currency", "BTC"),
                "from_address": tx.get("from_address"),
                "to_address": tx.get("to_address"),
                "suspicious_indicators": tx.get("violations", []),
                "risk_score": tx.get("risk_score", 0.0),
                "narrative": self._generate_sar_narrative(tx)
            }
            
            report["suspicious_activities"].append(activity)
            report["total_amount"] += activity["amount"]
        
        return report
    
    def _generate_sar_narrative(self, transaction: Dict[str, Any]) -> str:
        """Generate narrative description for SAR filing"""
        violations = transaction.get("violations", [])
        amount = transaction.get("amount", 0)
        
        narrative = f"Transaction of {amount} {transaction.get('currency', 'BTC')} "
        narrative += f"from {transaction.get('from_address', 'unknown')} "
        narrative += f"to {transaction.get('to_address', 'unknown')} "
        narrative += f"flagged for: {', '.join(violations)}."
        
        if "mixer_interaction" in str(violations):
            narrative += " Transaction involved cryptocurrency mixing service."
        
        if "structuring" in str(violations):
            narrative += " Transaction amount suggests potential structuring to avoid reporting requirements."
        
        return narrative
    
    def generate_ctr_report(self, large_transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate Currency Transaction Report (CTR) for transactions over $10,000
        
        Args:
            large_transactions: List of large transactions
            
        Returns:
            CTR report data
        """
        report = {
            "report_type": "CTR",
            "filing_institution": "QREX-FL System",
            "report_date": datetime.now().isoformat(),
            "transactions": [],
            "total_amount": 0.0,
            "transaction_count": len(large_transactions)
        }
        
        for tx in large_transactions:
            ctr_transaction = {
                "transaction_id": tx.get("tx_id"),
                "date": tx.get("timestamp"),
                "amount": tx.get("amount", 0),
                "currency": tx.get("currency", "BTC"),
                "transaction_type": "cryptocurrency_transfer",
                "from_address": tx.get("from_address"),
                "to_address": tx.get("to_address"),
                "reporting_threshold": "10000_USD_equivalent"
            }
            
            report["transactions"].append(ctr_transaction)
            report["total_amount"] += ctr_transaction["amount"]
        
        return report
    
    def generate_compliance_summary(self, transactions: List[Dict[str, Any]], 
                                  period_days: int = 30) -> Dict[str, Any]:
        """
        Generate periodic compliance summary report
        
        Args:
            transactions: List of all transactions in period
            period_days: Reporting period in days
            
        Returns:
            Compliance summary report
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Categorize transactions
        compliant = [tx for tx in transactions if tx.get("is_compliant", True)]
        non_compliant = [tx for tx in transactions if not tx.get("is_compliant", True)]
        
        large_transactions = [tx for tx in transactions if tx.get("amount", 0) >= 10000]
        suspicious_transactions = [tx for tx in transactions if tx.get("risk_score", 0) > 0.7]
        
        summary = {
            "report_type": "COMPLIANCE_SUMMARY",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": period_days
            },
            "statistics": {
                "total_transactions": len(transactions),
                "compliant_transactions": len(compliant),
                "non_compliant_transactions": len(non_compliant),
                "large_transactions": len(large_transactions),
                "suspicious_transactions": len(suspicious_transactions),
                "compliance_rate": len(compliant) / len(transactions) if transactions else 1.0,
                "total_volume": sum(tx.get("amount", 0) for tx in transactions),
                "average_risk_score": sum(tx.get("risk_score", 0) for tx in transactions) / len(transactions) if transactions else 0.0
            },
            "regulatory_filings_required": {
                "sar_filings": len(suspicious_transactions),
                "ctr_filings": len(large_transactions)
            },
            "top_violations": self._get_top_violations(non_compliant),
            "risk_distribution": self._get_risk_distribution(transactions)
        }
        
        return summary
    
    def _get_top_violations(self, non_compliant_transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get most common violation types"""
        violation_counts = {}
        
        for tx in non_compliant_transactions:
            violations = tx.get("violations", [])
            for violation in violations:
                violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        # Sort by frequency
        sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"violation": violation, "count": count, "percentage": count / len(non_compliant_transactions)}
            for violation, count in sorted_violations[:10]
        ]
    
    def _get_risk_distribution(self, transactions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of transactions by risk level"""
        distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for tx in transactions:
            risk_score = tx.get("risk_score", 0.0)
            
            if risk_score < 0.3:
                distribution["low"] += 1
            elif risk_score < 0.6:
                distribution["medium"] += 1
            elif risk_score < 0.8:
                distribution["high"] += 1
            else:
                distribution["critical"] += 1
        
        return distribution
    
    def export_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export report to file
        
        Args:
            report: Report data
            filename: Optional filename, auto-generated if not provided
            
        Returns:
            Path to exported file
        """
        if not filename:
            report_type = report.get("report_type", "REPORT")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_type}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Exported report to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            raise
    
    def generate_all_reports(self, transactions: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate all required compliance reports
        
        Args:
            transactions: List of transactions to analyze
            
        Returns:
            Dictionary of report type to file path
        """
        reports = {}
        
        # Filter transactions for different report types
        suspicious_transactions = [tx for tx in transactions if tx.get("risk_score", 0) > 0.7]
        large_transactions = [tx for tx in transactions if tx.get("amount", 0) >= 10000]
        
        # Generate SAR if needed
        if suspicious_transactions:
            sar_report = self.generate_sar_report(suspicious_transactions)
            reports["SAR"] = self.export_report(sar_report)
        
        # Generate CTR if needed
        if large_transactions:
            ctr_report = self.generate_ctr_report(large_transactions)
            reports["CTR"] = self.export_report(ctr_report)
        
        # Always generate compliance summary
        summary_report = self.generate_compliance_summary(transactions)
        reports["COMPLIANCE_SUMMARY"] = self.export_report(summary_report)
        
        return reports
    
    def schedule_periodic_reporting(self, interval_days: int = 30) -> bool:
        """
        Set up periodic automated reporting
        
        Args:
            interval_days: Reporting interval in days
            
        Returns:
            Success status
        """
        # This would integrate with a job scheduler in a real implementation
        self.logger.info(f"Scheduled periodic reporting every {interval_days} days")
        return True
