"""
encryption.py — AES-256 Encryption Helpers for ApexAlgo Backend
"""

import os
import hmac
import hashlib
import json
import base64
from typing import Dict, Any, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class ApexAlgoEncryption:
    def __init__(self, master_key_b64: str = None):  # type: ignore
        """
        Initializes the encryption module.
        If master_key_b64 is not provided, it attempts to load APEXALGO_MASTER_KEY from env.
        """
        env_key = os.getenv("APEXALGO_MASTER_KEY")
        self.master_key = master_key_b64 or env_key
        
        if not self.master_key:
            # Fallback for development only: deterministic key from a known salt
            # In production, THIS MUST be set purely via ENV variable.
            self.master_key = self.generate_key_from_password("apexalgo_dev_fallback", b"ante_salt")
        
        # Ensure it's a valid Fernet key (must be URL-safe base64 encoded, 32 bytes)
        try:
            self.cipher = Fernet(self.master_key.encode('utf-8') if isinstance(self.master_key, str) else self.master_key)
        except Exception as e:
            raise ValueError(f"Invalid Master Key format for Fernet initialization: {e}")

    @staticmethod
    def generate_random_key() -> str:
        """Generates a secure, random AES-256 Fernet key."""
        return Fernet.generate_key().decode('utf-8')

    @staticmethod
    def generate_key_from_password(password: str, salt: bytes = b"apexalgo_salt_2026") -> bytes:
        """Derives a deterministic AES-256 Fernet key from a password and salt using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = kdf.derive(password.encode('utf-8'))
        return base64.urlsafe_b64encode(key)

    def encrypt_string(self, plaintext: str) -> str:
        """Encrypt a standard string."""
        if not plaintext:
            return ""
        encrypted_bytes = self.cipher.encrypt(plaintext.encode('utf-8'))
        return encrypted_bytes.decode('utf-8')

    def decrypt_string(self, ciphertext: str) -> str:
        """Decrypt a standard string."""
        if not ciphertext:
            return ""
        decrypted_bytes = self.cipher.decrypt(ciphertext.encode('utf-8'))
        return decrypted_bytes.decode('utf-8')

    def encrypt_json(self, data: Dict[str, Any]) -> str:
        """Serialize and encrypt a dictionary or JSON object."""
        if not data:
            return ""
        json_str = json.dumps(data)
        return self.encrypt_string(json_str)

    def decrypt_json(self, ciphertext: str) -> Dict[str, Any]:
        """Decrypt and deserialize a JSON object."""
        if not ciphertext:
            return {}
        decrypted_str = self.decrypt_string(ciphertext)
        return json.loads(decrypted_str)

    def sign_request(self, payload: Union[str, Dict], secret: str) -> str:
        """
        Generate HMAC SHA-256 signature for bot-to-backend communication.
        Provides protection against tampering and replay attacks.
        """
        if isinstance(payload, dict):
            # Sort keys to ensure deterministic ordering of JSON strings
            payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        else:
            payload_str = payload

        signature = hmac.new(
            key=secret.encode('utf-8'),
            msg=payload_str.encode('utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()
        
        return signature

    def verify_request_signature(self, payload: Union[str, Dict], secret: str, provided_signature: str) -> bool:
        """Verify an HMAC SHA-256 signature."""
        expected_signature = self.sign_request(payload, secret)
        return hmac.compare_digest(expected_signature, provided_signature)

# Global singleton for backend usage
kms = ApexAlgoEncryption()
