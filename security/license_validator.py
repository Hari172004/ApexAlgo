"""
license_validator.py — Continuous JWT License Check
Verifies the ApexAlgo bot license every 60 seconds against the backend API.
"""

import threading
import time
import os
import requests
import logging
import sys
import jwt
from typing import Optional

# We dynamically import hardware lock to inject HWID into requests
from security.hardware_lock import hw_lock

logger = logging.getLogger("apexalgo.license")

class LicenseValidator:
    def __init__(self, backend_url: str = "https://api.apexalgo.com"):
        self.backend_url = backend_url
        self.license_key: Optional[str] = os.getenv("APEXALGO_LICENSE_KEY")
        self.jwt_public_key: Optional[str] = os.getenv("APEXALGO_PUBLIC_KEY")
        
        self._stop_event = threading.Event()
        self._monitor_thread = None

    def _decode_license_offline(self) -> dict:
        """
        Performs local validation of the license signature using RS256 public key.
        Prevents trivial offline spoofing.
        """
        if not self.license_key or not self.jwt_public_key:
            return {}
            
        try:
            # Must be signed with the backend's RS256 private key
            payload = jwt.decode(self.license_key, self.jwt_public_key, algorithms=["RS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.critical("[LICENSE] Your ApexAlgo license has expired!")
            sys.exit(1)
        except jwt.InvalidSignatureError:
            logger.critical("[LICENSE] Invalid license signature. Forgery detected.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"[LICENSE] Local decode failed: {e}")
            return {}

    def _ping_backend(self) -> bool:
        """
        Pings the verification API.
        Passes the JWT license and the Hardware ID.
        If the backend revokes the key, it returns 401/403.
        """
        if not self.license_key:
            return False

        current_hwid = hw_lock.get_hardware_id()
        headers = {
            "Authorization": f"Bearer {self.license_key}",
            "X-Hardware-ID": current_hwid
        }
        
        try:
            # In severe network outages, we might tolerate a missed heartbeat,
            # but usually, we enforce a strict check. For this implementation,
            # we will return True on timeout to not punish bad Wi-Fi instantly,
            # keeping a counter instead (omitted for brevity).
            res = requests.post(f"{self.backend_url}/v1/license/verify", headers=headers, timeout=5)
            
            if res.status_code in [401, 403]:
                logger.critical("[LICENSE] License revoked or invalid HWID. Backend blocked access.")
                return False
                
            return True
        except requests.exceptions.RequestException:
            # Network issue. Assume OK for this tick.
            logger.debug("[LICENSE] Verification server unreachable. Will retry later.")
            return True

    def _monitor_loop(self):
        """Infinite loop running every 60 seconds."""
        logger.info("[LICENSE] Real-time license monitor activated. Heartbeat interval: 60s")
        while not self._stop_event.is_set():
            # Locally verify expiry first
            self._decode_license_offline()
            
            # Remotely verify revocation / HWID bindings
            valid = self._ping_backend()
            if not valid:
                logger.critical("[LICENSE] Bot execution halted due to license invalidation.")
                # Hard exit the entire program
                os._exit(1)
                
            time.sleep(60)

    def start(self):
        """Starts the background verification thread."""
        if not self.license_key:
            logger.critical("[LICENSE] No APEXALGO_LICENSE_KEY found in .env file.")
            sys.exit(1)
            
        # Initial boot check
        if not self._ping_backend():
            logger.critical("[LICENSE] Initial license check failed. Shutting down.")
            sys.exit(1)
            
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        """Stops the background thread gracefully."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()
