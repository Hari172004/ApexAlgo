import os
import time
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional

# Third-party securely vetted libraries
import jwt
import pyotp

logger = logging.getLogger("apexalgo.auth")

# ── Global Constants ─────────────────────────────────────────────────────────

ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7
MAX_DEVICES_PER_USER = 3
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 30
INACTIVITY_TIMEOUT_MINUTES = 30

# In-memory stores for security session tracking (in production, use Redis)
# To keep this script 100% self-contained and free, we use dicts.
FAILED_LOGINS: Dict[str, Dict] = {}        # { email: {"attempts": int, "locked_until": float} }
USER_SESSIONS: Dict[str, Dict] = {}        # { user_id: { session_id: { ... } } }
DEVICE_REGISTRY: Dict[str, list] = {}      # { user_id: [device_id_1, device_id_2] }

class ApexAlgoAuth:
    def __init__(self, jwt_secret: Optional[str] = None):
        """
        Initializes the authentication module with a secure JWT signing secret.
        """
        self.jwt_secret = jwt_secret or os.getenv("JWT_SECRET_KEY", "fallback_dev_secret_2026_xyz!@#")
        self.jwt_algorithm = "HS256"
        self.firebase_api_key = os.getenv("FIREBASE_API_KEY")

    # ── 1. Firebase Mock / Wrapper ──────────────────────────────────────────

    def verify_firebase_token(self, id_token: str) -> Optional[Dict]:
        """
        Verifies a Firebase ID token.
        In a real app, uses firebase-admin.auth.verify_id_token().
        For this implementation, we decode standard JWT payloads simulating Firebase.
        """
        try:
            # Example decode structure for a Firebase-like token
            decoded = jwt.decode(id_token, options={"verify_signature": False})
            if "uid" not in decoded and "sub" not in decoded:
                raise ValueError("Invalid Firebase payload")
            user_id = decoded.get("uid", decoded.get("sub"))
            return {"user_id": user_id, "email": decoded.get("email")}
        except Exception as e:
            logger.error(f"[Auth] Firebase token error: {e}")
            return None

    # ── 2. Brute Force & Lockout Protection ──────────────────────────────────

    def check_brute_force(self, email: str) -> bool:
        """Returns True if the account is currently locked out."""
        record = FAILED_LOGINS.get(email)
        if not record:
            return False
        
        if record["locked_until"] and time.time() < record["locked_until"]:
            logger.warning(f"[Auth] Account {email} is locked until {datetime.fromtimestamp(record['locked_until'])}")
            return True
            
        # If lock expired, reset
        if record["locked_until"] and time.time() >= record["locked_until"]:
            FAILED_LOGINS[email] = {"attempts": 0, "locked_until": None}
            
        return False

    def record_failed_login(self, email: str):
        """Records a failed attempt. Locks if threshold reached."""
        record = FAILED_LOGINS.get(email, {"attempts": 0, "locked_until": None})
        if record["locked_until"]:
            return # Already locked
            
        record["attempts"] += 1
        logger.warning(f"[Auth] Failed login for {email}. Attempt {record['attempts']}/{MAX_FAILED_ATTEMPTS}")
        
        if record["attempts"] >= MAX_FAILED_ATTEMPTS:
            lock_time = time.time() + (LOCKOUT_MINUTES * 60)
            record["locked_until"] = lock_time
            logger.error(f"[Auth] Account {email} LOCKED due to '{MAX_FAILED_ATTEMPTS}' failed attempts.")
            # Trigger: suspicious login alert (Telegram)
        
        FAILED_LOGINS[email] = record

    def clear_failed_login(self, email: str):
        """Clears failures upon a successful 2FA login."""
        if email in FAILED_LOGINS:
            FAILED_LOGINS.pop(email, None)

    # ── 3. Device Fingerprinting ─────────────────────────────────────────────

    def register_device(self, user_id: str, device_id: str) -> bool:
        """Enforces a hard limit of MAX_DEVICES_PER_USER (3)."""
        devices = DEVICE_REGISTRY.get(user_id, [])
        if device_id in devices:
            return True # Already registered
            
        if len(devices) >= MAX_DEVICES_PER_USER:
            logger.error(f"[Auth] Max devices ({MAX_DEVICES_PER_USER}) reached for {user_id}. Blocking new device {device_id}.")
            return False
            
        devices.append(device_id)
        DEVICE_REGISTRY[user_id] = devices
        logger.info(f"[Auth] New device registered for {user_id}. Total: {len(devices)}/{MAX_DEVICES_PER_USER}")
        # Trigger: Telegram alert (New Device)
        return True

    # ── 4. JWT Token Generation (`15 min access`, `7 day refresh`) ───────────

    def generate_tokens(self, user_id: str, device_id: str, session_id: str) -> Tuple[str, str]:
        """Generates the secure Access and Refresh JWT tokens."""
        now = datetime.now(timezone.utc)
        
        # Access Token payload
        access_payload = {
            "sub": user_id,
            "device": device_id,
            "sess": session_id,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        }
        access_token = jwt.encode(access_payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        # Refresh Token payload
        refresh_payload = {
            "sub": user_id,
            "device": device_id,
            "sess": session_id,
            "type": "refresh",
            "iat": now,
            "exp": now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        }
        refresh_token = jwt.encode(refresh_payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        return access_token, refresh_token

    # ── 5. Standard Two Factor Auth (Authenticator App) ──────────────────────

    def generate_2fa_secret(self) -> str:
        """Generates a secure Base32 secret for Google Authenticator / Authy."""
        return pyotp.random_base32()

    def get_2fa_uri(self, secret: str, email: str) -> str:
        """Provides the URI to generate the QR Code."""
        return pyotp.totp.TOTP(secret).provisioning_uri(name=email, issuer_name="ApexAlgo Trading")

    def verify_2fa_code(self, secret: str, token: str) -> bool:
        """Verifies the user's 6-digit TOTP code."""
        totp = pyotp.TOTP(secret)
        return totp.verify(token)

    # ── 6. OTP via Telegram / Email ──────────────────────────────────────────

    def send_telegram_otp(self, telegram_id: str, user_id: str) -> str:
        """Generates a transient OTP and sends it via Telegram bot (mocked push)."""
        otp = pyotp.random_base32()[:6].upper() # 6 letter/digit OTP
        logger.info(f"[Auth] (Simulated) Sending Telegram OTP '{otp}' to {telegram_id}")
        return otp # Return for the caller to store in cache temporarily

    def send_email_otp(self, email: str, user_id: str) -> str:
        """Generates a transient OTP and sends it via Email (mocked push)."""
        otp = pyotp.random_base32()[:6].upper()
        logger.info(f"[Auth] (Simulated) Sending Email OTP '{otp}' to {email}")
        return otp

    # ── 7. Session Validation (IP Logging & Auto Logout) ─────────────────────

    def validate_session(self, access_token: str, current_ip: str) -> Optional[Dict]:
        """Decodes JWT, checks expiry, tracks activity for auto-logout."""
        try:
            payload = jwt.decode(access_token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            if payload.get("type") != "access":
                raise ValueError("Invalid token type")
                
            user_id = payload["sub"]
            session_id = payload["sess"]
            
            # Check Active Session Dict
            user_dict = USER_SESSIONS.get(user_id, {})
            session_data = user_dict.get(session_id)
            
            if not session_data:
                raise ValueError("Session terminated or logged out")
                
            # Verify Inactivity Auto-Logout
            last_active = session_data["last_active"]
            if time.time() - last_active > (INACTIVITY_TIMEOUT_MINUTES * 60):
                USER_SESSIONS[user_id].pop(session_id, None)
                raise ValueError("Session expired due to inactivity (>30 mins)")

            # Log IP Address and Timestamp
            session_data["last_active"] = time.time()
            if current_ip and current_ip not in session_data["ips"]:
                session_data["ips"].append(current_ip)
                logger.info(f"[Auth] New IP {current_ip} detected for {user_id}")
                # Trigger: Threat detector 'new IP' alert
                
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("[Auth] Access token expired.")
            return None
        except Exception as e:
            logger.error(f"[Auth] Token validation failed: {str(e)}")
            return None

    # ── 8. Login Flow Execution ──────────────────────────────────────────────

    def execute_login(self, email: str, user_id: str, device_id: str, ip_address: str, country: str) -> dict:
        """
        The master flow:
        1. Checks lockouts.
        2. Enforces max 3 devices limit.
        3. Provisions the session, JWTs, and logs the IP + Country.
        """
        if self.check_brute_force(email):
            return {"error": "Account is locked due to multiple failed login attempts. Try again in 30 mins."}
            
        if not self.register_device(user_id, device_id):
            return {"error": "Device limit reached. Max 3 devices allowed per account. Please deregister an old device."}
            
        self.clear_failed_login(email)
        
        session_id = str(uuid.uuid4())
        
        # Save session data for inactivity auto logout and IP logging
        if user_id not in USER_SESSIONS:
            USER_SESSIONS[user_id] = {}
            
        USER_SESSIONS[user_id][session_id] = {
            "device": device_id,
            "created_at": time.time(),
            "last_active": time.time(),
            "ips": [ip_address],
            "country": country
        }
        
        access, refresh = self.generate_tokens(user_id, device_id, session_id)
        
        logger.info(f"[Auth] Successful login for {email} from {ip_address} ({country}). Session: {session_id}")
        
        return {
            "access_token": access,
            "refresh_token": refresh,
            "session_id": session_id,
            "user_id": user_id
        }

    def execute_logout(self, user_id: str, session_id: str):
        """Immediately destroys the active session on the backend."""
        if user_id in USER_SESSIONS and session_id in USER_SESSIONS[user_id]:
            USER_SESSIONS[user_id].pop(session_id, None)
            logger.info(f"[Auth] Session {session_id} manually logged out.")

# Global instance
auth_manager = ApexAlgoAuth()
