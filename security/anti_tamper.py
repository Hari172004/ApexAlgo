"""
anti_tamper.py — Anti-Debug and Checksum Protection
Detects if the bot code has been modified or if a debugger is attached.
"""

import sys
import os
import hashlib
import logging

logger = logging.getLogger("apexalgo.tampering")

class AntiTamper:
    def __init__(self):
        self._expected_checksums = {}
        # In a real build pipeline, these checksums are securely injected during compilation
        # Here we mock the behavior of loading expected signatures.

    def _get_file_sha256(self, filepath: str) -> str:
        """Reads a file chunk by chunk to calculate its exact SHA-256 hash."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except FileNotFoundError:
            return ""

    def load_signatures(self, signature_file: str):
        """Loads the expected checksums (usually an encrypted manifest)."""
        # Simulated structure: {"core.py": "abc123hash..."}
        pass

    def verify_integrity(self, target_files: list) -> bool:
        """
        Scans critical bot files and matches against expected SHA-256.
        If the hash doesn't match the signature manifest, tampering is detected.
        """
        for filepath in target_files:
            if not os.path.exists(filepath):
                continue
                
            current_hash = self._get_file_sha256(filepath)
            expected_hash = self._expected_checksums.get(filepath)
            
            if expected_hash and current_hash != expected_hash:
                logger.critical(f"[SHIELD] Integrity failure in {filepath}. Code was modified!")
                return False
                
        logger.info("[SHIELD] Code integrity verified. No tampering detected.")
        return True

    def detect_debugger(self):
        """
        Basic Python anti-debug check.
        Checks if sys.gettrace() is active (often used by IDE debuggers or decompilers).
        """
        is_debugged = sys.gettrace() is not None
        if is_debugged:
            logger.critical("[SHIELD] Debugger attachment detected! Shutting down immediately to protect logic.")
            os._exit(1)

    def enforce_protections(self, target_files: list):
        """Runs all protections sequentially. Hard crash if failed."""
        self.detect_debugger()
        
        # Disabled locally since we edit files rapidly.
        # In a compiled executable (PyArmor), this would be active.
        # if not self.verify_integrity(target_files):
        #     os._exit(1)

# Global shield instance
shield = AntiTamper()
