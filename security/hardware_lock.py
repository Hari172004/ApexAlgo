"""
hardware_lock.py — Machine ID Binding
Locks the PC bot to a specific physical machine using UUID and MAC addresses.
"""

import uuid
import platform
import hashlib
import logging

logger = logging.getLogger("apexalgo.hardware")

class HardwareLock:
    def __init__(self):
        self.machine_id = self._generate_hardware_id()

    def _generate_hardware_id(self) -> str:
        """
        Generates a unique, deterministic hardware fingerprint.
        Combines OS name, node name, architecture, and network MAC address.
        """
        sys_info = f"{platform.system()}-{platform.node()}-{platform.machine()}"
        
        # uuid.getnode() returns the MAC address of the device
        # It's generally stable unless NIC is completely changed / spoofed
        mac_addr = str(uuid.getnode())
        
        raw_hwid = f"{sys_info}-{mac_addr}"
        
        # Hash it to obscure underlying hardware specifics
        hwid_hash = hashlib.sha256(raw_hwid.encode('utf-8')).hexdigest()
        return hwid_hash

    def get_hardware_id(self) -> str:
        """Returns the SHA-256 hashed hardware ID."""
        return self.machine_id

    def verify_binding(self, expected_hwid: str) -> bool:
        """
        Validates if the current machine matches the expected HWID.
        """
        if self.machine_id != expected_hwid:
            logger.critical(f"[HW_LOCK] Hardware ID mismatch! Bot is locked to a different machine.")
            return False
        return True

# Global instance
hw_lock = HardwareLock()
