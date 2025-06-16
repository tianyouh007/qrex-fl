"""
QREX-FL ML-DSA Implementation
Correct FIPS 204 ML-DSA implementation using dilithium-py
"""

import json
import logging
from typing import Dict, Any, Tuple, Optional
import hashlib

# Import the official NIST FIPS 204 ML-DSA implementation
try:
    from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
    DILITHIUM_AVAILABLE = True
except ImportError:
    DILITHIUM_AVAILABLE = False
    print("Warning: dilithium-py not installed. Install with: pip install dilithium-py")

logger = logging.getLogger(__name__)

class MLDSAManager:
    """
    Quantum-resistant ML-DSA (FIPS 204) digital signature manager for QREX-FL
    """
    
    def __init__(self, security_level: int = 44):
        """
        Initialize ML-DSA manager
        
        Args:
            security_level: ML-DSA security level (44, 65, or 87)
        """
        self.security_level = security_level
        self.logger = logging.getLogger(__name__)
        
        if not DILITHIUM_AVAILABLE:
            self.logger.error("dilithium-py not available. Using fallback implementation.")
            self._use_fallback = True
        else:
            self._use_fallback = False
            
            # Select the appropriate ML-DSA variant
            if security_level == 44:
                self.ml_dsa = ML_DSA_44
                self.algorithm_name = "ML-DSA-44"
            elif security_level == 65:
                self.ml_dsa = ML_DSA_65
                self.algorithm_name = "ML-DSA-65"
            elif security_level == 87:
                self.ml_dsa = ML_DSA_87
                self.algorithm_name = "ML-DSA-87"
            else:
                raise ValueError(f"Unsupported security level: {security_level}")
        
        self.logger.info(f"ML-DSA Manager initialized with {self.algorithm_name}")
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information"""
        return {
            "algorithm": self.algorithm_name,
            "security_level": self.security_level,
            "type": "post_quantum_signature",
            "standard": "NIST FIPS 204",
            "available": not self._use_fallback
        }
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate ML-DSA keypair
        
        Returns:
            Tuple of (public_key, secret_key) as bytes
        """
        if self._use_fallback:
            return self._fallback_generate_keypair()
        
        try:
            # Use the official FIPS 204 implementation
            public_key, secret_key = self.ml_dsa.keygen()
            
            self.logger.info(f"Generated {self.algorithm_name} keypair")
            return public_key, secret_key
            
        except Exception as e:
            self.logger.error(f"Keypair generation failed: {e}")
            return self._fallback_generate_keypair()
    
    def sign_federated_update(self, update_data: Dict[str, Any], secret_key: bytes) -> bytes:
        """
        Sign federated learning model update using ML-DSA
        
        Args:
            update_data: Model update data to sign
            secret_key: ML-DSA secret key
            
        Returns:
            ML-DSA signature bytes
        """
        if self._use_fallback:
            return self._fallback_sign(update_data, secret_key)
        
        try:
            # Serialize update data consistently
            message = self._serialize_update_data(update_data)
            
            # Sign using official ML-DSA implementation
            signature = self.ml_dsa.sign(secret_key, message)
            
            self.logger.debug(f"Signed federated update with {self.algorithm_name}")
            return signature
            
        except Exception as e:
            self.logger.error(f"Signing failed: {e}")
            return self._fallback_sign(update_data, secret_key)
    
    def verify_federated_update(self, update_data: Dict[str, Any], 
                               signature: bytes, public_key: bytes) -> bool:
        """
        Verify federated learning model update signature
        
        Args:
            update_data: Model update data that was signed
            signature: ML-DSA signature to verify
            public_key: ML-DSA public key
            
        Returns:
            True if signature is valid, False otherwise
        """
        if self._use_fallback:
            return self._fallback_verify(update_data, signature, public_key)
        
        try:
            # Serialize update data consistently (same as signing)
            message = self._serialize_update_data(update_data)
            
            # Verify using official ML-DSA implementation
            is_valid = self.ml_dsa.verify(public_key, message, signature)
            
            self.logger.debug(f"Signature verification: {'PASSED' if is_valid else 'FAILED'}")
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            return False
    
    def _serialize_update_data(self, update_data: Dict[str, Any]) -> bytes:
        """
        Serialize update data consistently for signing/verification
        
        Args:
            update_data: Data to serialize
            
        Returns:
            Serialized data as bytes
        """
        try:
            # Use consistent JSON serialization
            json_str = json.dumps(update_data, sort_keys=True, separators=(',', ':'))
            return json_str.encode('utf-8')
        except Exception as e:
            self.logger.error(f"Serialization failed: {e}")
            # Fallback to string representation
            return str(update_data).encode('utf-8')
    
    def _fallback_generate_keypair(self) -> Tuple[bytes, bytes]:
        """Fallback keypair generation when ML-DSA not available"""
        import secrets
        
        # Generate deterministic keys for demo purposes
        seed = secrets.token_bytes(32)
        secret_key = hashlib.sha256(seed + b"secret").digest() * 2  # 64 bytes
        public_key = hashlib.sha256(seed + b"public").digest() * 2  # 64 bytes
        
        self.logger.warning("Using fallback keypair generation - NOT quantum-resistant!")
        return public_key, secret_key
    
    def _fallback_sign(self, update_data: Dict[str, Any], secret_key: bytes) -> bytes:
        """Fallback signing when ML-DSA not available"""
        try:
            import hmac
            
            message = self._serialize_update_data(update_data)
            signature = hmac.new(secret_key[:32], message, hashlib.sha256).digest()
            
            self.logger.warning("Using fallback signature - NOT quantum-resistant!")
            return signature
            
        except Exception as e:
            self.logger.error(f"Fallback signing failed: {e}")
            return b"fallback_signature"
    
    def _fallback_verify(self, update_data: Dict[str, Any], 
                        signature: bytes, public_key: bytes) -> bool:
        """Fallback verification when ML-DSA not available"""
        try:
            import hmac
            
            message = self._serialize_update_data(update_data)
            
            # For demo, derive secret from public key
            derived_secret = hashlib.sha256(public_key).digest()
            expected_signature = hmac.new(derived_secret[:32], message, hashlib.sha256).digest()
            
            is_valid = hmac.compare_digest(signature, expected_signature)
            self.logger.warning("Using fallback verification - NOT quantum-resistant!")
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Fallback verification failed: {e}")
            return False
    
    def benchmark_performance(self, iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark ML-DSA performance
        
        Args:
            iterations: Number of iterations to average
            
        Returns:
            Performance metrics
        """
        if self._use_fallback:
            return {"error": "ML-DSA not available for benchmarking"}
        
        import time
        
        # Generate test data
        test_message = b"QREX-FL benchmark test message"
        
        # Benchmark keygen
        start_time = time.time()
        for _ in range(iterations):
            pk, sk = self.ml_dsa.keygen()
        keygen_time = (time.time() - start_time) / iterations
        
        # Benchmark signing
        pk, sk = self.ml_dsa.keygen()
        start_time = time.time()
        for _ in range(iterations):
            sig = self.ml_dsa.sign(sk, test_message)
        sign_time = (time.time() - start_time) / iterations
        
        # Benchmark verification
        sig = self.ml_dsa.sign(sk, test_message)
        start_time = time.time()
        for _ in range(iterations):
            valid = self.ml_dsa.verify(pk, test_message, sig)
        verify_time = (time.time() - start_time) / iterations
        
        return {
            "algorithm": self.algorithm_name,
            "iterations": iterations,
            "keygen_ms": keygen_time * 1000,
            "sign_ms": sign_time * 1000,
            "verify_ms": verify_time * 1000
        }
    
    def get_signature_size(self) -> int:
        """Get signature size in bytes"""
        if self._use_fallback:
            return 32  # Fallback HMAC size
        
        # Generate a test signature to get size
        pk, sk = self.ml_dsa.keygen()
        sig = self.ml_dsa.sign(sk, b"test")
        return len(sig)
    
    def get_key_sizes(self) -> Dict[str, int]:
        """Get key sizes in bytes"""
        if self._use_fallback:
            return {"public_key": 64, "secret_key": 64}
        
        # Generate test keys to get sizes
        pk, sk = self.ml_dsa.keygen()
        return {
            "public_key": len(pk),
            "secret_key": len(sk)
        }

# Export the manager
__all__ = ['MLDSAManager']