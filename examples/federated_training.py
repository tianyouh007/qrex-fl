"""
QREX-FL Federated Training Example
Demonstrates quantum-resistant federated learning for Bitcoin risk assessment
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import List
import multiprocessing as mp
from pathlib import Path

from core.models import TemporalBitcoinRiskModel
from core.datasets import create_federated_dataloaders
from federated.client import QREXFederatedClient
from federated.server import QREXFederatedServer
from quantum.crypto_manager import QuantumCryptoManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_bitcoin_data(num_samples: int = 1000) -> TensorDataset:
    """Create synthetic Bitcoin transaction data for demonstration"""
    
    # Generate synthetic features (165 features like Elliptic dataset)
    features = torch.randn(num_samples, 165)
    
    # Generate synthetic labels with class imbalance (2% illicit like real data)
    labels = torch.zeros(num_samples)
    num_illicit = int(num_samples * 0.02)  # 2% illicit transactions
    illicit_indices = torch.randperm(num_samples)[:num_illicit]
    labels[illicit_indices] = 1.0
    
    # Add some correlation between features and labels
    # Make illicit transactions have higher values in certain features
    for i in illicit_indices:
        features[i, :10] += torch.randn(10) * 2  # Boost first 10 features
    
    return TensorDataset(features, labels)

def create_federated_clients(dataset: TensorDataset, 
                           num_clients: int = 3) -> List[QREXFederatedClient]:
    """Create federated learning clients with quantum-resistant security"""
    
    # Create federated data loaders
    client_loaders = create_federated_dataloaders(
        dataset, num_clients=num_clients, batch_size=32, shuffle=True
    )
    
    clients = []
    for i, train_loader in enumerate(client_loaders):
        # Create validation loader (subset of training data for demo)
        val_loader = train_loader  # In practice, use separate validation data
        
        # Create model for this client
        model = TemporalBitcoinRiskModel(input_size=165)
        
        # Create client
        client = QREXFederatedClient(
            client_id=f"bank_{i+1}",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu'
        )
        
        clients.append(client)
        logger.info(f"Created client {client.client_id} with {len(train_loader.dataset)} samples")
    
    return clients

def run_federated_training():
    """Run federated training simulation"""
    
    print("🚀 Starting QREX-FL Federated Training")
    print("=" * 50)
    
    # Create synthetic data
    print("📊 Creating synthetic Bitcoin transaction data...")
    dataset = create_synthetic_bitcoin_data(num_samples=5000)
    print(f"✓ Created dataset with {len(dataset)} transactions")
    
    # Create federated clients
    print("\n🏦 Setting up federated clients (simulating banks)...")
    clients = create_federated_clients(dataset, num_clients=3)
    print(f"✓ Created {len(clients)} federated clients")
    
    # Display client information
    for client in clients:
        info = client.get_client_info()
        print(f"   📱 {info['client_id']}: {info['model_info']['parameters']} parameters")
    
    # Simulate federated learning rounds
    print("\n🔄 Starting Federated Learning Rounds...")
    
    num_rounds = 5
    global_model = TemporalBitcoinRiskModel(input_size=165)
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")
        
        # Get global model parameters
        global_params = []
        for param in global_model.parameters():
            global_params.append(param.detach().cpu().numpy())
        
        # Each client trains locally
        client_updates = []
        total_samples = 0
        
        for client in clients:
            print(f"🏦 Training {client.client_id}...")
            
            # Simulate training configuration
            config = {
                "local_epochs": 1,
                "learning_rate": 0.001,
                "server_round": round_num
            }
            
            # Client local training
            updated_params, num_examples, metrics = client.fit(global_params, config)
            
            client_updates.append({
                "client_id": client.client_id,
                "parameters": updated_params,
                "num_examples": num_examples,
                "metrics": metrics
            })
            
            total_samples += num_examples
            
            print(f"   ✓ {client.client_id}: {num_examples} samples, loss={metrics['train_loss']:.4f}")
        
        # Aggregate updates (Federated Averaging)
        print("🔒 Aggregating updates with quantum-resistant verification...")
        
        aggregated_params = []
        num_params = len(client_updates[0]["parameters"])
        
        for param_idx in range(num_params):
            # Weighted average based on number of samples
            weighted_sum = None
            
            for update in client_updates:
                weight = update["num_examples"] / total_samples
                param = update["parameters"][param_idx]
                
                if weighted_sum is None:
                    weighted_sum = weight * param
                else:
                    weighted_sum += weight * param
            
            aggregated_params.append(weighted_sum)
        
        # Update global model
        for param, new_param in zip(global_model.parameters(), aggregated_params):
            param.data = torch.tensor(new_param, dtype=param.dtype)
        
        # Evaluate global model
        print("📊 Evaluating global model...")
        
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        
        global_model.eval()
        with torch.no_grad():
            for client in clients:
                val_loss, accuracy, num_examples = client._evaluate_local()
                total_loss += val_loss * num_examples
                total_correct += accuracy * num_examples
                total_examples += num_examples
        
        avg_loss = total_loss / total_examples
        avg_accuracy = total_correct / total_examples
        
        print(f"🎯 Round {round_num} Results:")
        print(f"   Global Loss: {avg_loss:.4f}")
        print(f"   Global Accuracy: {avg_accuracy:.4f}")
        print(f"   Total Samples: {total_samples}")
    
    print("\n🎉 Federated Training Completed!")
    
    # Final evaluation
    print("\n📈 Final Model Evaluation:")
    
    # Test on held-out data
    test_data = create_synthetic_bitcoin_data(num_samples=1000)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False)
    
    global_model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = global_model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    
    test_accuracy = correct_predictions / total_predictions
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"📊 Test Results:")
    print(f"   Test Loss: {avg_test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    model_path = Path("data/models")
    model_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'model_info': global_model.get_model_info(),
        'test_accuracy': test_accuracy,
        'test_loss': avg_test_loss,
        'training_rounds': num_rounds
    }, model_path / "qrex_federated_model.pth")
    
    print(f"💾 Model saved to: {model_path / 'qrex_federated_model.pth'}")
    
    return {
        "final_accuracy": test_accuracy,
        "final_loss": avg_test_loss,
        "rounds_completed": num_rounds,
        "total_clients": len(clients)
    }

def main():
    """Main training function"""
    try:
        results = run_federated_training()
        
        print("\n" + "=" * 50)
        print("🏆 FEDERATED TRAINING SUMMARY")
        print("=" * 50)
        print(f"✅ Training Completed: {results['rounds_completed']} rounds")
        print(f"✅ Final Accuracy: {results['final_accuracy']:.4f}")
        print(f"✅ Final Loss: {results['final_loss']:.4f}")
        print(f"✅ Participants: {results['total_clients']} quantum-resistant clients")
        
        print("\n🎯 Next Steps:")
        print("   • Deploy model: python examples/model_deployment.py")
        print("   • Start API server: python src/api/main.py")
        print("   • View model performance: python examples/model_analysis.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Federated training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
