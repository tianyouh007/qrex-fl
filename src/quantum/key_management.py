"""
Quantum-Resistant Key Management for QREX-FL
Handles key storage, rotation, and distribution
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3

class KeyManager:
    """
    Quantum-resistant key management system
    Handles storage, rotation, and secure distribution of ML-DSA and ML-KEM keys
    """
    
    def __init__(self, storage_path: str = "data/keys"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize key database
        self.db_path = self.storage_path / "keys.db"
        self._init_key_database()
        
    def _init_key_database(self):
        """Initialize SQLite database for key storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS keys (
                    key_id TEXT PRIMARY KEY,
                    key_type TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    public_key BLOB,
                    secret_key BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_key_type ON keys(key_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON keys(status)
            """)
    
    def store_keypair(self, 
                     key_id: str,
                     key_type: str,
                     algorithm: str,
                     public_key: bytes,
                     secret_key: bytes,
                     expires_in_days: Optional[int] = 365,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a quantum-resistant keypair
        
        Args:
            key_id: Unique identifier for the keypair
            key_type: Type of key (dsa, kem, hybrid)
            algorithm: Cryptographic algorithm (ML-DSA-44, ML-KEM-768)
            public_key: Public key bytes
            secret_key: Secret key bytes
            expires_in_days: Key expiration in days
            metadata: Additional key metadata
            
        Returns:
            Success status
        """
        try:
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now() + timedelta(days=expires_in_days)
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO keys 
                    (key_id, key_type, algorithm, public_key, secret_key, expires_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (key_id, key_type, algorithm, public_key, secret_key, expires_at, metadata_json))
            
            self.logger.info(f"Stored {algorithm} keypair with ID: {key_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store keypair {key_id}: {e}")
            return False
    
    def get_keypair(self, key_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a keypair by ID
        
        Args:
            key_id: Key identifier
            
        Returns:
            Keypair information or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM keys WHERE key_id = ? AND status = 'active'
                """, (key_id,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'key_id': row['key_id'],
                        'key_type': row['key_type'],
                        'algorithm': row['algorithm'],
                        'public_key': row['public_key'],
                        'secret_key': row['secret_key'],
                        'created_at': row['created_at'],
                        'expires_at': row['expires_at'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                    }
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve keypair {key_id}: {e}")
            return None
    
    def get_public_key(self, key_id: str) -> Optional[bytes]:
        """Get only the public key for a keypair"""
        keypair = self.get_keypair(key_id)
        return keypair['public_key'] if keypair else None
    
    def list_keys(self, key_type: Optional[str] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        List stored keys
        
        Args:
            key_type: Filter by key type
            active_only: Only return active keys
            
        Returns:
            List of key information
        """
        try:
            query = "SELECT key_id, key_type, algorithm, created_at, expires_at, status FROM keys"
            params = []
            
            conditions = []
            if key_type:
                conditions.append("key_type = ?")
                params.append(key_type)
            if active_only:
                conditions.append("status = 'active'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Failed to list keys: {e}")
            return []
    
    def rotate_key(self, key_id: str, new_key_data: Dict[str, Any]) -> bool:
        """
        Rotate a key (mark old as expired, create new)
        
        Args:
            key_id: Original key ID
            new_key_data: New key information
            
        Returns:
            Success status
        """
        try:
            # Mark old key as expired
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE keys SET status = 'expired' WHERE key_id = ?
                """, (key_id,))
            
            # Create new key with rotated ID
            new_key_id = f"{key_id}_rotated_{int(datetime.now().timestamp())}"
            
            success = self.store_keypair(
                key_id=new_key_id,
                key_type=new_key_data['key_type'],
                algorithm=new_key_data['algorithm'],
                public_key=new_key_data['public_key'],
                secret_key=new_key_data['secret_key'],
                expires_in_days=new_key_data.get('expires_in_days', 365),
                metadata=new_key_data.get('metadata', {})
            )
            
            if success:
                self.logger.info(f"Successfully rotated key {key_id} to {new_key_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to rotate key {key_id}: {e}")
            return False
    
    def cleanup_expired_keys(self) -> int:
        """
        Remove expired keys from storage
        
        Returns:
            Number of keys cleaned up
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM keys 
                    WHERE expires_at < datetime('now') OR status = 'expired'
                """)
                cleaned_count = cursor.rowcount
            
            self.logger.info(f"Cleaned up {cleaned_count} expired keys")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired keys: {e}")
            return 0
    
    def export_public_keys(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export all public keys for sharing
        
        Args:
            output_path: Optional file path to save export
            
        Returns:
            Public key export data
        """
        try:
            keys = self.list_keys(active_only=True)
            
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'public_keys': []
            }
            
            for key_info in keys:
                keypair = self.get_keypair(key_info['key_id'])
                if keypair:
                    export_data['public_keys'].append({
                        'key_id': keypair['key_id'],
                        'key_type': keypair['key_type'],
                        'algorithm': keypair['algorithm'],
                        'public_key': keypair['public_key'].hex(),
                        'created_at': keypair['created_at'],
                        'expires_at': keypair['expires_at']
                    })
            
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                self.logger.info(f"Exported public keys to {output_path}")
            
            return export_data
            
        except Exception as e:
            self.logger.error(f"Failed to export public keys: {e}")
            return {}
