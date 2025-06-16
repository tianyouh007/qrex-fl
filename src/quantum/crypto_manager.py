"""
Quantum-Resistant Cryptography Manager for QREX-FL
Combines ML-DSA signatures with ML-KEM key encapsulation
"""

import json
import logging
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
import secrets

from .ml_dsa import MLDSAManager

try:
    from pqcrypto.kem.ml_kem_768 import generate_keypair as kem_generate_keypair
    from pqcrypto.kem.ml_kem_768 import encapsulate, decapsulate
    KEM_AVAILABLE = True
except ImportError:
    KEM_AVAILABLE = False

class QuantumCryptoManager:
    """
    Comprehensive quantum-resistant cryptography manager
    Implements both ML-DSA (signatures) and ML-KEM (key encapsulation)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ml_dsa = MLDSAManager()
        
        # Initialize key pairs
        self.dsa_public_key = None
        self.dsa_secret_key = None
        self.kem_public_key = None
        self.kem_secret_key = None
        
        self.logger.info("Initialized QuantumCryptoManager with ML-DSA and ML-KEM")
    
    def initialize_keys(self) -> Dict[str, bytes]:
        """
        Initialize both ML-DSA and ML-KEM key pairs
        Returns dictionary with all public keys
        """
        # Generate ML-DSA keys for signatures
        self.dsa_public_key, self.dsa_secret_key = self.ml_dsa.generate_keypair()
        
        # Generate ML-KEM keys for key encapsulation
        if KEM_AVAILABLE:
            self.kem_public_key, self.kem_secret_key = kem_generate_keypair()
            self.logger.info("Generated ML-KEM keypair using NIST standard")
        else:
            # Fallback KEM implementation
            self.kem_secret_key = secrets.token_bytes(32)
            self.kem_public_key = secrets.token_bytes(1184)  # ML-KEM-768 public key size
            self.logger.warning("Using fallback KEM - not quantum-resistant!")
        
        return {
            'dsa_public_key': self.dsa_public_key,
            'kem_public_key': self.kem_public_key
        }
    
    def secure_federated_communication(self, 
                                     message: Dict[str, Any], 
                                     recipient_kem_public_key: bytes) -> Dict[str, Any]:
        """
        Secure federated learning communication using hybrid crypto
        
        Args:
            message: Data to send securely
            recipient_kem_public_key: Recipient's ML-KEM public key
            
        Returns:
            Encrypted and signed message package
        """
        try:
            # 1. Encapsulate a shared secret using ML-KEM
            if KEM_AVAILABLE:
                ciphertext, shared_secret = encapsulate(recipient_kem_public_key)
            else:
                # Fallback: generate random shared secret
                shared_secret = secrets.token_bytes(32)
                ciphertext = secrets.token_bytes(1088)  # ML-KEM-768 ciphertext size
            
            # 2. Encrypt message with AES using shared secret
            encrypted_message = self._aes_encrypt(message, shared_secret)
            
            # 3. Sign the package with ML-DSA
            package = {
                'kem_ciphertext': ciphertext.hex(),
                'encrypted_message': encrypted_message.hex(),
                'sender_public_key': self.dsa_public_key.hex(),
                'timestamp': datetime.now().isoformat()
            }
            
            signature = self.ml_dsa.sign_federated_update(package, self.dsa_secret_key)
            
            package['signature'] = signature.hex()
            
            self.logger.debug("Created secure federated communication package")
            return package
            
        except Exception as e:
            self.logger.error(f"Secure communication failed: {e}")
            raise
    
    def decrypt_federated_communication(self, 
                                      package: Dict[str, Any], 
                                      sender_dsa_public_key: bytes) -> Dict[str, Any]:
        """
        Decrypt and verify federated communication
        
        Args:
            package: Encrypted and signed message package
            sender_dsa_public_key: Sender's ML-DSA public key
            
        Returns:
            Decrypted message
        """
        try:
            # 1. Verify ML-DSA signature
            signature = bytes.fromhex(package['signature'])
            package_to_verify = {k: v for k, v in package.items() if k != 'signature'}
            
            if not self.ml_dsa.verify_federated_update(package_to_verify, signature, sender_dsa_public_key):
                raise ValueError("ML-DSA signature verification failed")
            
            # 2. Decapsulate shared secret using ML-KEM
            kem_ciphertext = bytes.fromhex(package['kem_ciphertext'])
            
            if KEM_AVAILABLE and self.kem_secret_key:
                shared_secret = decapsulate(self.kem_secret_key, kem_ciphertext)
            else:
                # Fallback: would need to be coordinated in real scenario
                shared_secret = secrets.token_bytes(32)
            
            # 3. Decrypt message with AES
            encrypted_message = bytes.fromhex(package['encrypted_message'])
            decrypted_message = self._aes_decrypt(encrypted_message, shared_secret)
            
            self.logger.debug("Successfully decrypted federated communication")
            return decrypted_message
            
        except Exception as e:
            self.logger.error(f"Secure communication decryption failed: {e}")
            raise
    
    def _aes_encrypt(self, data: Dict[str, Any], key: bytes) -> bytes:
        """AES encryption with GCM mode"""
        # Serialize data
        plaintext = json.dumps(data, sort_keys=True).encode('utf-8')
        
        # Generate IV
        iv = secrets.token_bytes(12)  # GCM recommended IV size
        
        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Combine IV, tag, and ciphertext
        return iv + encryptor.tag + ciphertext
    
    def _aes_decrypt(self, encrypted_data: bytes, key: bytes) -> Dict[str, Any]:
        """AES decryption with GCM mode"""
        # Extract components
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        # Decrypt
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Deserialize
        return json.loads(plaintext.decode('utf-8'))
    
    def get_public_keys(self) -> Dict[str, str]:
        """Get public keys for sharing"""
        return {
            'dsa_public_key': self.dsa_public_key.hex() if self.dsa_public_key else None,
            'kem_public_key': self.kem_public_key.hex() if self.kem_public_key else None,
            'algorithms': {
                'signature': 'ML-DSA-44 (FIPS 204)',
                'key_encapsulation': 'ML-KEM-768 (FIPS 203)'
            }
        }
