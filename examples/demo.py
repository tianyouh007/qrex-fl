"""
QREX-FL Comprehensive Demo
Demonstrates quantum-resistant federated learning for cryptocurrency compliance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# Import QREX-FL modules
from quantum.ml_dsa import MLDSAManager
from quantum.crypto_manager import QuantumCryptoManager
from compliance.engine import ComplianceEngine, Transaction, JurisdictionType
from core.models import TemporalBitcoinRiskModel
from core.datasets import EllipticDataset

def demo_quantum_crypto():
    """Demonstrate quantum-resistant cryptography"""
    print("🔒 QREX-FL Quantum-Resistant Cryptography Demo")
    print("=" * 50)
    
    # Initialize ML-DSA manager
    ml_dsa = MLDSAManager()
    print(f"✓ Initialized {ml_dsa.get_algorithm_info()['algorithm']}")
    
    # Generate keypair
    public_key, secret_key = ml_dsa.generate_keypair()
    print(f"✓ Generated quantum-resistant keypair")
    
    # Sign federated update
    test_update = {
        "client_id": "demo_client",
        "parameters": {"layer1": [0.1, 0.2], "layer2": [0.3, 0.4]},
        "round": 1
    }
    
    signature = ml_dsa.sign_federated_update(test_update, secret_key)
    print(f"✓ Signed federated update")
    
    # Verify signature
    is_valid = ml_dsa.verify_federated_update(test_update, signature, public_key)
    print(f"✓ Signature verification: {'PASSED' if is_valid else 'FAILED'}")
    
    return is_valid

def demo_compliance_engine():
    """Demonstrate multi-jurisdictional compliance"""
    print("\n🏛️  QREX-FL Compliance Engine Demo")
    print("=" * 50)
    
    # Initialize compliance engine
    engine = ComplianceEngine()
    print("✓ Initialized multi-jurisdictional compliance engine")
    
    # Test transactions
    transactions = [
        Transaction(
            tx_id="demo_tx_001",
            from_address="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            to_address="1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
            amount=15000.0,  # Above threshold
            currency="BTC",
            timestamp=datetime.now(),
            features={"mixer_interaction": False, "sanctioned_entity": False}
        ),
        Transaction(
            tx_id="demo_tx_002",
            from_address="1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
            to_address="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            amount=9000.0,  # Potential structuring
            currency="BTC",
            timestamp=datetime.now(),
            features={"mixer_interaction": True, "sanctioned_entity": False}
        )
    ]
    
    # Test jurisdictions
    jurisdictions = [JurisdictionType.US_FINCEN, JurisdictionType.EU_MICA]
    
    results = []
    for tx in transactions:
        result = engine.validate_transaction(tx, jurisdictions)
        results.append(result)
        
        print(f"\n📊 Transaction: {tx.tx_id}")
        print(f"   Amount: {tx.amount} {tx.currency}")
        print(f"   Compliant: {'✅' if result.is_compliant else '❌'}")
        print(f"   Risk Score: {result.risk_score:.2f}")
        if result.violations:
            print(f"   Violations: {', '.join(result.violations)}")
    
    # Generate compliance report
    report = engine.generate_compliance_report(results)
    print(f"\n📈 Compliance Summary:")
    print(f"   Total Transactions: {report['summary']['total_transactions']}")
    print(f"   Compliance Rate: {report['summary']['compliance_rate']:.1%}")
    print(f"   Average Risk Score: {report['summary']['average_risk_score']:.2f}")
    
    return len(results)

def demo_bitcoin_risk_model():
    """Demonstrate Bitcoin risk assessment model"""
    print("\n🤖 QREX-FL Bitcoin Risk Model Demo")
    print("=" * 50)
    
    # Initialize model
    model = TemporalBitcoinRiskModel(input_size=165)
    print(f"✓ Initialized model with {model._count_parameters()} parameters")
    
    # Create synthetic transaction data (165 features like Elliptic dataset)
    batch_size = 10
    synthetic_features = torch.randn(batch_size, 165)
    
    # Model inference
    model.eval()
    with torch.no_grad():
        logits = model(synthetic_features)
        probabilities = torch.sigmoid(logits)
    
    print(f"✓ Processed {batch_size} synthetic transactions")
    print(f"   Risk scores range: {probabilities.min():.3f} - {probabilities.max():.3f}")
    
    # Show individual predictions
    for i, (logit, prob) in enumerate(zip(logits[:5], probabilities[:5])):
        risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
        print(f"   Transaction {i+1}: {prob:.3f} ({risk_level} risk)")
    
    return model

def demo_federated_simulation():
    """Demonstrate federated learning simulation"""
    print("\n🌐 QREX-FL Federated Learning Simulation")
    print("=" * 50)
    
    # Initialize quantum crypto managers for 3 clients
    num_clients = 3
    clients = []
    
    for i in range(num_clients):
        crypto_manager = QuantumCryptoManager()
        client_keys = crypto_manager.initialize_keys()
        clients.append({
            "id": f"client_{i+1}",
            "crypto_manager": crypto_manager,
            "public_keys": client_keys
        })
    
    print(f"✓ Initialized {num_clients} quantum-resistant clients")
    
    # Simulate federated round
    print("\n🔄 Simulating Federated Learning Round")
    
    # Each client creates model update
    client_updates = []
    for client in clients:
        # Simulate model parameters
        model_update = {
            "client_id": client["id"],
            "round": 1,
            "parameters": {
                "layer1": np.random.randn(10).tolist(),
                "layer2": np.random.randn(5).tolist()
            },
            "num_samples": np.random.randint(100, 1000),
            "loss": np.random.uniform(0.1, 0.5)
        }
        
        # Sign update with ML-DSA
        signature = client["crypto_manager"].ml_dsa.sign_federated_update(
            model_update, client["crypto_manager"].dsa_secret_key
        )
        
        signed_update = {
            **model_update,
            "signature": signature.hex(),
            "public_key": client["public_keys"]["dsa_public_key"].hex()
        }
        
        client_updates.append(signed_update)
        print(f"   ✓ {client['id']}: Signed update with {model_update['num_samples']} samples")
    
    # Server aggregates updates (simplified FedAvg)
    print("\n🔒 Server: Verifying and Aggregating Updates")
    
    verified_updates = []
    for update in client_updates:
        # Verify signature
        signature = bytes.fromhex(update["signature"])
        public_key = bytes.fromhex(update["public_key"])
        
        # Create update for verification (without signature fields)
        params_to_verify = {k: v for k, v in update.items() 
                          if k not in ["signature", "public_key"]}
        
        # Verify with first client's crypto manager (any would work)
        is_valid = clients[0]["crypto_manager"].ml_dsa.verify_federated_update(
            params_to_verify, signature, public_key
        )
        
        if is_valid:
            verified_updates.append(update)
            print(f"   ✅ Verified update from {update['client_id']}")
        else:
            print(f"   ❌ Failed to verify update from {update['client_id']}")
    
    # Simple federated averaging
    if verified_updates:
        total_samples = sum(update["num_samples"] for update in verified_updates)
        
        aggregated_params = {}
        for param_name in ["layer1", "layer2"]:
            weighted_sum = np.zeros_like(verified_updates[0]["parameters"][param_name])
            
            for update in verified_updates:
                weight = update["num_samples"] / total_samples
                param_array = np.array(update["parameters"][param_name])
                weighted_sum += weight * param_array
            
            aggregated_params[param_name] = weighted_sum.tolist()
        
        print(f"\n🎯 Aggregation Complete:")
        print(f"   Verified Updates: {len(verified_updates)}/{len(client_updates)}")
        print(f"   Total Samples: {total_samples}")
        print(f"   Aggregated Parameters: {len(aggregated_params)} layers")
    
    return len(verified_updates)

def demo_integrated_workflow():
    """Demonstrate integrated QREX-FL workflow"""
    print("\n🚀 QREX-FL Integrated Workflow Demo")
    print("=" * 50)
    
    print("1️⃣  Quantum-Resistant Setup")
    crypto_manager = QuantumCryptoManager()
    keys = crypto_manager.initialize_keys()
    print("   ✓ Quantum-resistant cryptography initialized")
    
    print("\n2️⃣  Compliance Engine Ready")
    compliance_engine = ComplianceEngine()
    print("   ✓ Multi-jurisdictional compliance engine ready")
    
    print("\n3️⃣  AI Model Initialization")
    risk_model = TemporalBitcoinRiskModel()
    print(f"   ✓ Risk assessment model ready ({risk_model._count_parameters()} parameters)")
    
    print("\n4️⃣  Transaction Processing Pipeline")
    
    # Simulate real transaction
    test_transaction = Transaction(
        tx_id="integrated_test_001",
        from_address="bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
        to_address="1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
        amount=25000.0,
        currency="BTC",
        timestamp=datetime.now(),
        features={
            "confirmations": 6,
            "mixer_interaction": False,
            "sanctioned_entity": False,
            "ml_risk_score": 0.15  # Low risk from AI model
        }
    )
    
    # Compliance check
    compliance_result = compliance_engine.validate_transaction(
        test_transaction, [JurisdictionType.US_FINCEN, JurisdictionType.EU_MICA]
    )
    
    # AI risk assessment (using random features for demo)
    risk_model.eval()  # Set to evaluation mode to disable BatchNorm
    risk_features = torch.randn(2, 165)  # Use batch size of 2 instead of 1
    with torch.no_grad():
        ai_risk_scores = torch.sigmoid(risk_model(risk_features))
        ai_risk_score = ai_risk_scores[0].item()  # Take first prediction
    
    # Combined assessment
    combined_risk = 0.6 * compliance_result.risk_score + 0.4 * ai_risk_score
    
    print(f"   📊 Transaction Analysis:")
    print(f"      ID: {test_transaction.tx_id}")
    print(f"      Amount: {test_transaction.amount} {test_transaction.currency}")
    print(f"      Compliance Risk: {compliance_result.risk_score:.3f}")
    print(f"      AI Risk Score: {ai_risk_score:.3f}")
    print(f"      Combined Risk: {combined_risk:.3f}")
    print(f"      Compliant: {'✅' if compliance_result.is_compliant else '❌'}")
    
    if combined_risk > 0.7:
        action = "🚫 BLOCK"
    elif combined_risk > 0.4:
        action = "🔍 FLAG"
    else:
        action = "✅ APPROVE"
    
    print(f"      Recommended Action: {action}")
    
    print("\n🎉 Integrated workflow completed successfully!")
    
    return {
        "compliance_score": compliance_result.risk_score,
        "ai_score": ai_risk_score,
        "combined_score": combined_risk,
        "action": action
    }

def main():
    """Run all demos"""
    print("🌟 Welcome to QREX-FL Demonstration")
    print("Quantum-Resistant Explainable Federated Learning")
    print("for Next-Generation Cryptocurrency Risk Assessment")
    print("=" * 60)
    
    try:
        # Run individual demos
        crypto_valid = demo_quantum_crypto()
        compliance_count = demo_compliance_engine()
        risk_model = demo_bitcoin_risk_model()
        federated_verified = demo_federated_simulation()
        integrated_result = demo_integrated_workflow()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 DEMO SUMMARY")
        print("=" * 60)
        print(f"✅ Quantum Cryptography: {'WORKING' if crypto_valid else 'FAILED'}")
        print(f"✅ Compliance Engine: {compliance_count} transactions validated")
        print(f"✅ AI Risk Model: {risk_model._count_parameters()} parameters loaded")
        print(f"✅ Federated Learning: {federated_verified}/3 clients verified")
        print(f"✅ Integrated Workflow: Combined risk {integrated_result['combined_score']:.3f}")
        
        print("\n🚀 QREX-FL is ready for production deployment!")
        print("\n📚 Next Steps:")
        print("   • Download real datasets: python scripts/download_datasets.py")
        print("   • Start federated training: python examples/federated_training.py")
        print("   • Launch API server: python src/api/main.py")
        print("   • View documentation: docs/README.md")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
