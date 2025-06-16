#!/usr/bin/env python3
"""
QREX-FL Dataset Download Helper
Automated download and setup for cryptocurrency datasets
"""

import os
import sys
import subprocess
import requests
import zipfile
import tarfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

class QREXDatasetDownloader:
    """Download and prepare datasets for QREX-FL research."""
    
    def __init__(self, base_path: str = "datasets"):
        """Initialize downloader with base path."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
    def log(self, message: str) -> None:
        """Log message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """Download file with progress indicator."""
        try:
            self.log(f"Downloading {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end="", flush=True)
            
            print()  # New line after progress
            self.log(f"Downloaded: {filepath}")
            return True
            
        except Exception as e:
            self.log(f"Error downloading {url}: {e}")
            return False
    
    def clone_repository(self, repo_url: str, target_dir: Path) -> bool:
        """Clone Git repository."""
        try:
            if target_dir.exists():
                self.log(f"Repository already exists: {target_dir}")
                return True
                
            self.log(f"Cloning repository: {repo_url}")
            result = subprocess.run(
                ["git", "clone", repo_url, str(target_dir)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log(f"Successfully cloned to: {target_dir}")
                return True
            else:
                self.log(f"Error cloning repository: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"Error cloning repository: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract zip or tar archive."""
        try:
            self.log(f"Extracting {archive_path}")
            
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                self.log(f"Unsupported archive format: {archive_path.suffix}")
                return False
                
            self.log(f"Extracted to: {extract_to}")
            return True
            
        except Exception as e:
            self.log(f"Error extracting archive: {e}")
            return False
    
    def download_elliptic_plus_dataset(self) -> bool:
        """Download Elliptic++ dataset from GitHub."""
        repo_url = "https://github.com/git-disl/EllipticPlusPlus.git"
        target_dir = self.base_path / "elliptic-plus"
        
        success = self.clone_repository(repo_url, target_dir)
        
        if success:
            # Verify dataset files
            expected_files = [
                "Transactions Dataset/txs_features.csv",
                "Transactions Dataset/txs_classes.csv", 
                "Transactions Dataset/txs_edgelist.csv",
                "Actors Dataset/wallets_features.csv",
                "Actors Dataset/wallets_classes.csv"
            ]
            
            for file_path in expected_files:
                full_path = target_dir / file_path
                if full_path.exists():
                    self.log(f"✓ Found: {file_path}")
                else:
                    self.log(f"⚠ Missing: {file_path}")
        
        return success
    
    def create_elliptic_download_instructions(self) -> None:
        """Create instructions for manual Elliptic dataset download."""
        elliptic_dir = self.base_path / "elliptic"
        elliptic_dir.mkdir(exist_ok=True)
        
        instructions = """
# Elliptic Bitcoin Dataset Download Instructions

The original Elliptic dataset requires manual download from Kaggle due to authentication requirements.

## Steps:
1. Create a Kaggle account at https://www.kaggle.com/
2. Go to https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
3. Click "Download" button
4. Extract the downloaded zip file
5. Copy the following files to this directory:
   - elliptic_txs_features.csv
   - elliptic_txs_classes.csv
   - elliptic_txs_edgelist.csv

## Expected Files:
- elliptic_txs_features.csv (203,769 transactions × 167 features)
- elliptic_txs_classes.csv (Transaction labels: 1=illicit, 2=licit)
- elliptic_txs_edgelist.csv (Transaction graph edges)

## Dataset Statistics:
- Total transactions: 203,769
- Labeled transactions: ~46,000
- Time span: 49 time steps
- Graph edges: ~234,000

After downloading, run: python verify_datasets.py
"""
        
        with open(elliptic_dir / "DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
            f.write(instructions)
        
        self.log(f"Created download instructions: {elliptic_dir}/DOWNLOAD_INSTRUCTIONS.txt")
    
    def generate_synthetic_data(self) -> bool:
        """Generate synthetic cryptocurrency data for testing."""
        synthetic_dir = self.base_path / "synthetic"
        synthetic_dir.mkdir(exist_ok=True)
        
        try:
            self.log("Generating synthetic cryptocurrency data...")
            
            # Generate synthetic transaction data
            n_transactions = 10000
            n_addresses = 1000
            
            # Create addresses
            addresses = [f"addr_{i:06d}" for i in range(n_addresses)]
            
            # Generate transactions
            np.random.seed(42)  # For reproducibility
            
            transactions = []
            for i in range(n_transactions):
                tx_id = f"tx_{i:06d}"
                from_addr = np.random.choice(addresses)
                to_addr = np.random.choice(addresses)
                while to_addr == from_addr:
                    to_addr = np.random.choice(addresses)
                
                # Transaction features (similar to Elliptic dataset structure)
                time_step = np.random.randint(1, 50)
                local_features = np.random.randn(93)  # 93 local features
                aggregated_features = np.random.randn(72)  # 72 aggregated features
                
                # Label (illicit with 5% probability)
                is_illicit = np.random.random() < 0.05
                label = 1 if is_illicit else (2 if np.random.random() < 0.2 else 0)  # 0=unknown
                
                transactions.append({
                    'txId': tx_id,
                    'time_step': time_step,
                    'from_address': from_addr,
                    'to_address': to_addr,
                    'class': label,
                    **{f'local_feature_{j}': local_features[j] for j in range(93)},
                    **{f'agg_feature_{j}': aggregated_features[j] for j in range(72)}
                })
            
            # Save transaction features
            df_features = pd.DataFrame(transactions)
            feature_cols = ['txId', 'time_step'] + [f'local_feature_{j}' for j in range(93)] + [f'agg_feature_{j}' for j in range(72)]
            df_features[feature_cols].to_csv(synthetic_dir / "synthetic_txs_features.csv", index=False)
            
            # Save transaction classes
            df_classes = df_features[['txId', 'class']]
            df_classes.to_csv(synthetic_dir / "synthetic_txs_classes.csv", index=False)
            
            # Generate synthetic edges
            edges = []
            for _ in range(n_transactions // 2):  # Roughly half as many edges as transactions
                tx1 = f"tx_{np.random.randint(0, n_transactions):06d}"
                tx2 = f"tx_{np.random.randint(0, n_transactions):06d}"
                if tx1 != tx2:
                    edges.append({'txId1': tx1, 'txId2': tx2})
            
            df_edges = pd.DataFrame(edges)
            df_edges.to_csv(synthetic_dir / "synthetic_txs_edgelist.csv", index=False)
            
            # Generate address features (for actor-level analysis)
            address_features = []
            for addr in addresses:
                # Address features
                features = np.random.randn(56)  # 56 features per address
                is_illicit_addr = np.random.random() < 0.03  # 3% illicit addresses
                
                address_features.append({
                    'address': addr,
                    'class': 1 if is_illicit_addr else 0,
                    **{f'feature_{j}': features[j] for j in range(56)}
                })
            
            df_addresses = pd.DataFrame(address_features)
            df_addresses.to_csv(synthetic_dir / "synthetic_wallets_features.csv", index=False)
            
            # Create dataset statistics
            stats = {
                "name": "Synthetic Cryptocurrency Dataset",
                "transactions": n_transactions,
                "addresses": n_addresses,
                "edges": len(edges),
                "illicit_transactions": sum(1 for t in transactions if t['class'] == 1),
                "licit_transactions": sum(1 for t in transactions if t['class'] == 2),
                "unknown_transactions": sum(1 for t in transactions if t['class'] == 0),
                "time_steps": 49,
                "features_per_transaction": 165,
                "features_per_address": 56
            }
            
            with open(synthetic_dir / "dataset_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            self.log(f"✓ Generated synthetic dataset with {n_transactions} transactions and {n_addresses} addresses")
            return True
            
        except Exception as e:
            self.log(f"Error generating synthetic data: {e}")
            return False
    
    def download_additional_datasets(self) -> bool:
        """Download additional cryptocurrency datasets."""
        try:
            # Bitcoin dataset from Nature Scientific Data
            nature_dir = self.base_path / "bitcoin-nature"
            nature_dir.mkdir(exist_ok=True)
            
            self.log("Creating placeholder for Bitcoin Nature dataset...")
            
            # Create instructions for Nature dataset
            instructions = """
# Bitcoin Research Dataset (Nature Scientific Data 2025)

Paper: "Bitcoin research with a transaction graph dataset"
Authors: H. Schnoering, M. Vazirgiannis
Source: https://www.nature.com/articles/s41597-025-04684-8

## Dataset Specifications:
- 252 million nodes (transactions)
- 785 million edges
- Temporal annotations
- 34,000 labeled nodes with entity types
- 100,000 Bitcoin addresses with entity names

## Download Instructions:
This dataset is very large (several TB) and requires special access.
Contact the authors for download instructions.

For development, use the synthetic dataset or Elliptic++ dataset.
"""
            
            with open(nature_dir / "DATASET_INFO.txt", "w") as f:
                f.write(instructions)
            
            self.log("✓ Created Bitcoin Nature dataset information")
            return True
            
        except Exception as e:
            self.log(f"Error setting up additional datasets: {e}")
            return False
    
    def create_dataset_verification_script(self) -> None:
        """Create script to verify downloaded datasets."""
        verification_script = '''#!/usr/bin/env python3
"""
Dataset Verification Script for QREX-FL
Checks integrity and structure of downloaded datasets.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def verify_elliptic_dataset():
    """Verify original Elliptic dataset."""
    elliptic_dir = Path("elliptic")
    
    print("\\nVerifying Elliptic Bitcoin Dataset...")
    
    expected_files = [
        "elliptic_txs_features.csv",
        "elliptic_txs_classes.csv", 
        "elliptic_txs_edgelist.csv"
    ]
    
    for filename in expected_files:
        filepath = elliptic_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"✓ {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"✗ {filename}: Not found")
    
    # Detailed verification if files exist
    if (elliptic_dir / "elliptic_txs_features.csv").exists():
        features_df = pd.read_csv(elliptic_dir / "elliptic_txs_features.csv", header=None)
        print(f"  - Features shape: {features_df.shape}")
        print(f"  - Expected: (203769, 167)")
        
    if (elliptic_dir / "elliptic_txs_classes.csv").exists():
        classes_df = pd.read_csv(elliptic_dir / "elliptic_txs_classes.csv")
        print(f"  - Classes distribution:")
        print(f"    {classes_df['class'].value_counts().to_dict()}")

def verify_elliptic_plus_dataset():
    """Verify Elliptic++ dataset."""
    elliptic_plus_dir = Path("elliptic-plus")
    
    print("\\nVerifying Elliptic++ Dataset...")
    
    if not elliptic_plus_dir.exists():
        print("✗ Elliptic++ directory not found")
        return
    
    # Check transaction dataset
    tx_dir = elliptic_plus_dir / "Transactions Dataset"
    if tx_dir.exists():
        tx_files = ["txs_features.csv", "txs_classes.csv", "txs_edgelist.csv"]
        for filename in tx_files:
            filepath = tx_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                print(f"✓ Transactions/{filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            else:
                print(f"✗ Transactions/{filename}: Not found")
    
    # Check actors dataset
    actors_dir = elliptic_plus_dir / "Actors Dataset"
    if actors_dir.exists():
        actor_files = ["wallets_features.csv", "wallets_classes.csv"]
        for filename in actor_files:
            filepath = actors_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                print(f"✓ Actors/{filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            else:
                print(f"✗ Actors/{filename}: Not found")

def verify_synthetic_dataset():
    """Verify synthetic dataset."""
    synthetic_dir = Path("synthetic")
    
    print("\\nVerifying Synthetic Dataset...")
    
    if not synthetic_dir.exists():
        print("✗ Synthetic directory not found")
        return
    
    files_to_check = [
        "synthetic_txs_features.csv",
        "synthetic_txs_classes.csv",
        "synthetic_txs_edgelist.csv",
        "synthetic_wallets_features.csv"
    ]
    
    for filename in files_to_check:
        filepath = synthetic_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"✓ {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"✗ {filename}: Not found")
    
    # Load and display statistics
    stats_file = synthetic_dir / "dataset_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print("  Statistics:")
        for key, value in stats.items():
            print(f"    {key}: {value}")

def main():
    """Run all dataset verifications."""
    print("QREX-FL Dataset Verification")
    print("=" * 40)
    
    verify_elliptic_dataset()
    verify_elliptic_plus_dataset()
    verify_synthetic_dataset()
    
    print("\\n" + "=" * 40)
    print("Dataset verification complete!")

if __name__ == "__main__":
    main()
'''
        
        with open(self.base_path / "verify_datasets.py", "w") as f:
            f.write(verification_script)
        
        # Make it executable
        import stat
        (self.base_path / "verify_datasets.py").chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        self.log("Created dataset verification script")
    
    def run_full_download(self) -> None:
        """Run complete dataset download and setup."""
        self.log("Starting QREX-FL dataset download and setup...")
        
        # Download Elliptic++ dataset
        self.download_elliptic_plus_dataset()
        
        # Create Elliptic download instructions
        self.create_elliptic_download_instructions()
        
        # Generate synthetic data
        self.generate_synthetic_data()
        
        # Setup additional datasets
        self.download_additional_datasets()
        
        # Create verification script
        self.create_dataset_verification_script()
        
        self.log("Dataset setup completed!")
        self.log("\nNext steps:")
        self.log("1. Run: python datasets/verify_datasets.py")
        self.log("2. Follow instructions in datasets/elliptic/DOWNLOAD_INSTRUCTIONS.txt")
        self.log("3. Datasets ready for QREX-FL research!")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QREX-FL Dataset Downloader")
    parser.add_argument("--path", default="datasets", help="Base path for datasets")
    parser.add_argument("--elliptic-plus", action="store_true", help="Download Elliptic++ only")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data only")
    parser.add_argument("--verify", action="store_true", help="Verify datasets only")
    
    args = parser.parse_args()
    
    downloader = QREXDatasetDownloader(args.path)
    
    if args.elliptic_plus:
        downloader.download_elliptic_plus_dataset()
    elif args.synthetic:
        downloader.generate_synthetic_data()
    elif args.verify:
        # Run verification if it exists
        verify_script = Path(args.path) / "verify_datasets.py"
        if verify_script.exists():
            subprocess.run([sys.executable, str(verify_script)])
        else:
            print("Verification script not found. Run full download first.")
    else:
        downloader.run_full_download()

if __name__ == "__main__":
    main()
