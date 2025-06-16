"""
QREX-FL Compliance Demo
Demonstrates multi-jurisdictional cryptocurrency compliance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
from datetime import datetime, timedelta
import random
from typing import List

from compliance.engine import ComplianceEngine, Transaction, JurisdictionType
from compliance.validator import TransactionValidator

def generate_test_transactions(num_transactions: int = 20) -> List[Transaction]:
    """Generate diverse test transactions for compliance demonstration"""
    
    transactions = []
    
    # Predefined address examples (Bitcoin testnet style)
    addresses = [
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # Genesis block
        "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",  # Regular address
        "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",  # Bech32 address
        "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",  # P2SH address
        "1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF",  # Known exchange
        "1dice8EMZmqKvrGE4Qc9bUFf9PX3xaYDp",   # Known service
    ]
    
    # Transaction scenarios
    scenarios = [
        # Low risk scenarios
        {
            "amount_range": (100, 1000),
            "mixer_interaction": False,
            "sanctioned_entity": False,
            "high_risk_geography": False,
            "description": "Regular small transaction"
        },
        # Medium risk scenarios
        {
            "amount_range": (5000, 9999),
            "mixer_interaction": False,
            "sanctioned_entity": False,
            "high_risk_geography": True,
            "description": "Large transaction from high-risk region"
        },
        # High risk scenarios
        {
            "amount_range": (15000, 50000),
            "mixer_interaction": True,
            "sanctioned_entity": False,
            "high_risk_geography": False,
            "description": "Large transaction with mixer interaction"
        },
        # Very high risk scenarios
        {
            "amount_range": (8000, 12000),
            "mixer_interaction": True,
            "sanctioned_entity": True,
            "high_risk_geography": True,
            "description": "Sanctioned entity with mixer"
        }
    ]
    
    for i in range(num_transactions):
        scenario = random.choice(scenarios)
        
        # Generate transaction details
        amount = random.uniform(*scenario["amount_range"])
        
        # Round amounts for structuring detection
        if random.random() < 0.3:  # 30% chance of round amounts
            amount = round(amount / 1000) * 1000
        
        transaction = Transaction(
            tx_id=f"demo_tx_{i+1:03d}",
            from_address=random.choice(addresses),
            to_address=random.choice(addresses),
            amount=amount,
            currency="BTC",
            timestamp=datetime.now() - timedelta(hours=random.randint(0, 168)),  # Last week
            features={
                "mixer_interaction": scenario["mixer_interaction"],
                "sanctioned_entity": scenario["sanctioned_entity"],
                "high_risk_geography": scenario["high_risk_geography"],
                "tx_frequency_1h": random.randint(0, 20),
                "address_reuse_count": random.randint(0, 10),
                "value_similarity_score": random.uniform(0, 1),
                "address_age_days": random.randint(1, 1000),
                "confirmations": random.randint(1, 10),
                "originator_info": random.choice([True, False]) if amount > 3000 else False,
                "beneficiary_info": random.choice([True, False]) if amount > 3000 else False,
                "description": scenario["description"]
            },
            confirmations=random.randint(1, 10)
        )
        
        # Ensure from and to addresses are different
        while transaction.from_address == transaction.to_address:
            transaction.to_address = random.choice(addresses)
        
        transactions.append(transaction)
    
    return transactions

def demo_single_jurisdiction():
    """Demonstrate compliance checking for single jurisdiction"""
    print("🇺🇸 Single Jurisdiction Demo (US FinCEN)")
    print("-" * 40)
    
    engine = ComplianceEngine()
    test_transactions = generate_test_transactions(5)
    
    for tx in test_transactions:
        result = engine.validate_transaction(tx, [JurisdictionType.US_FINCEN])
        
        print(f"\n📋 Transaction: {tx.tx_id}")
        print(f"   Amount: ${tx.amount:,.2f} {tx.currency}")
        print(f"   Description: {tx.features.get('description', 'N/A')}")
        print(f"   Compliant: {'✅' if result.is_compliant else '❌'}")
        print(f"   Risk Score: {result.risk_score:.3f}")
        
        if result.violations:
            print(f"   ⚠️  Violations:")
            for violation in result.violations:
                print(f"      • {violation}")
        
        if result.recommendations:
            print(f"   💡 Recommendations:")
            for rec in result.recommendations[:2]:  # Show first 2
                print(f"      • {rec}")

def demo_multi_jurisdiction():
    """Demonstrate multi-jurisdictional compliance"""
    print("\n🌍 Multi-Jurisdictional Compliance Demo")
    print("-" * 40)
    
    engine = ComplianceEngine()
    jurisdictions = [
        JurisdictionType.US_FINCEN,
        JurisdictionType.EU_MICA,
        JurisdictionType.SINGAPORE_MAS
    ]
    
    test_transactions = generate_test_transactions(3)
    
    for tx in test_transactions:
        result = engine.validate_transaction(tx, jurisdictions)
        
        print(f"\n🌐 Transaction: {tx.tx_id}")
        print(f"   Amount: ${tx.amount:,.2f} {tx.currency}")
        print(f"   Jurisdictions: {len(jurisdictions)} (US, EU, Singapore)")
        print(f"   Overall Compliant: {'✅' if result.is_compliant else '❌'}")
        print(f"   Max Risk Score: {result.risk_score:.3f}")
        
        if result.violations:
            print(f"   🚨 Combined Violations: {len(result.violations)}")
            for violation in result.violations[:3]:  # Show first 3
                print(f"      • {violation}")

def demo_real_time_validation():
    """Demonstrate real-time transaction validation"""
    print("\n⚡ Real-Time Validation Demo")
    print("-" * 40)
    
    validator = TransactionValidator()
    test_transactions = generate_test_transactions(8)
    
    validation_results = []
    
    for tx in test_transactions:
        # Convert to dictionary format
        tx_data = {
            "tx_id": tx.tx_id,
            "from_address": tx.from_address,
            "to_address": tx.to_address,
            "amount": tx.amount,
            "currency": tx.currency,
            "timestamp": tx.timestamp.isoformat(),
            "features": tx.features,
            "confirmations": tx.confirmations
        }
        
        # Add simulated ML risk score
        ml_risk_score = random.uniform(0.0, 1.0)
        
        # Validate in real-time
        result = validator.validate_real_time(
            tx_data,
            [JurisdictionType.US_FINCEN, JurisdictionType.EU_MICA],
            ml_risk_score
        )
        
        validation_results.append(result)
        
        action_emoji = {
            "APPROVE": "✅",
            "MONITOR": "👁️",
            "FLAG": "🔍",
            "BLOCK": "🚫"
        }
        
        print(f"\n{action_emoji.get(result['action'], '❓')} {result['transaction_id']}")
        print(f"   Action: {result['action']} ({result['risk_level']} risk)")
        print(f"   Compliance: {result['compliance_result']['risk_score']:.3f}")
        print(f"   ML Score: {ml_risk_score:.3f}")
        print(f"   Combined: {result['action_details']['combined_risk_score']:.3f}")
    
    # Show validation statistics
    stats = validator.get_validation_statistics()
    print(f"\n📊 Validation Statistics:")
    print(f"   Total Processed: {stats['total_validations']}")
    print(f"   Approved: {stats['action_distribution'].get('APPROVE', 0)}")
    print(f"   Flagged: {stats['action_distribution'].get('FLAG', 0)}")
    print(f"   Blocked: {stats['action_distribution'].get('BLOCK', 0)}")
    print(f"   Approval Rate: {stats['approval_rate']:.1%}")

def demo_compliance_reporting():
    """Demonstrate compliance reporting"""
    print("\n📈 Compliance Reporting Demo")
    print("-" * 40)
    
    engine = ComplianceEngine()
    
    # Generate larger dataset for reporting
    test_transactions = generate_test_transactions(50)
    jurisdictions = [JurisdictionType.US_FINCEN, JurisdictionType.EU_MICA]
    
    # Process all transactions
    results = []
    for tx in test_transactions:
        result = engine.validate_transaction(tx, jurisdictions)
        results.append(result)
    
    # Generate comprehensive report
    report = engine.generate_compliance_report(results)
    
    print(f"📋 Compliance Report Summary:")
    print(f"   Transactions Analyzed: {report['summary']['total_transactions']}")
    print(f"   Compliance Rate: {report['summary']['compliance_rate']:.1%}")
    print(f"   Average Risk Score: {report['summary']['average_risk_score']:.3f}")
    
    print(f"\n🎯 Risk Distribution:")
    risk_dist = report['risk_distribution']
    print(f"   Low Risk: {risk_dist['low_risk']} ({risk_dist['low_risk']/report['summary']['total_transactions']:.1%})")
    print(f"   Medium Risk: {risk_dist['medium_risk']} ({risk_dist['medium_risk']/report['summary']['total_transactions']:.1%})")
    print(f"   High Risk: {risk_dist['high_risk']} ({risk_dist['high_risk']/report['summary']['total_transactions']:.1%})")
    
    print(f"\n⚠️  Top Violations:")
    for violation, count in report['violations']['most_common'][:3]:
        print(f"   • {violation}: {count} occurrences")
    
    print(f"\n💡 Top Recommendations:")
    for rec, count in report['recommendations']['most_frequent'][:3]:
        print(f"   • {rec}: {count} times")
    
    # Save report
    report_path = "data/outputs/compliance_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved to: {report_path}")

def main():
    """Run all compliance demos"""
    print("🏛️  QREX-FL Compliance System Demonstration")
    print("=" * 50)
    print("Multi-Jurisdictional Cryptocurrency Compliance Automation")
    print()
    
    try:
        demo_single_jurisdiction()
        demo_multi_jurisdiction()
        demo_real_time_validation()
        demo_compliance_reporting()
        
        print("\n" + "=" * 50)
        print("🎉 COMPLIANCE DEMO COMPLETED")
        print("=" * 50)
        print("✅ Single jurisdiction validation")
        print("✅ Multi-jurisdictional compliance")
        print("✅ Real-time transaction validation")
        print("✅ Comprehensive compliance reporting")
        
        print("\n🚀 The QREX-FL compliance system is ready for:")
        print("   • Financial institution deployment")
        print("   • Multi-jurisdictional regulatory compliance")
        print("   • Real-time transaction monitoring")
        print("   • Automated regulatory reporting")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Compliance demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
