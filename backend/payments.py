"""
payments.py — Stripe Subscription Integration for ApexAlgo
=============================================================
Handles: checkout session creation, webhook processing, license generation.
Plans:
  STARTER  — $29/month — XAUUSD only, Demo + Real
  PRO      — $59/month — XAUUSD + BTC, Demo + Real + Funded
  ELITE    — $99/month — Everything + priority signals
"""

import os
import secrets
import logging
import stripe
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException, Request
from backend.database import create_license, deactivate_license, upsert_user

logger = logging.getLogger("apexalgo.payments")

PLAN_PRICES = {
    "STARTER": os.getenv("STRIPE_PRICE_STARTER", "price_starter_id"),
    "PRO":     os.getenv("STRIPE_PRICE_PRO",     "price_pro_id"),
    "ELITE":   os.getenv("STRIPE_PRICE_ELITE",   "price_elite_id"),
}

PLAN_FEATURES = {
    "STARTER": {"modes": ["DEMO", "REAL"],               "assets": ["XAUUSD"]},
    "PRO":     {"modes": ["DEMO", "REAL", "FUNDED"],     "assets": ["XAUUSD", "BTCUSD"]},
    "ELITE":   {"modes": ["DEMO", "REAL", "FUNDED"],     "assets": ["XAUUSD", "BTCUSD"], "priority": True},
}


def init_stripe():
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
    if not stripe.api_key:
        logger.warning("[Payments] STRIPE_SECRET_KEY not set.")


def create_checkout_session(user_id: str, plan: str, email: str,
                            success_url: str, cancel_url: str) -> dict:
    """Create a Stripe Checkout session for a subscription plan."""
    init_stripe()
    price_id = PLAN_PRICES.get(plan.upper())
    if not price_id:
        raise HTTPException(status_code=400, detail=f"Unknown plan: {plan}")
    try:
        session = stripe.checkout.Session.create(
            customer_email=email,
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
            metadata={"user_id": user_id, "plan": plan.upper()},
            subscription_data={
                "metadata": {"user_id": user_id, "plan": plan.upper()}
            },
        )
        logger.info(f"[Payments] Checkout created | User={user_id} Plan={plan}")
        return {"checkout_url": session.url, "session_id": session.id}
    except stripe.error.StripeError as e:  # type: ignore
        logger.error(f"[Payments] Stripe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_license_key() -> str:
    """Generate a unique ApexAlgo license key."""
    chars: list = list(secrets.token_hex(16).upper())
    a = "".join(chars[0:4])   # type: ignore[misc]
    b = "".join(chars[4:8])   # type: ignore[misc]
    c = "".join(chars[8:12])  # type: ignore[misc]
    d = "".join(chars[12:16]) # type: ignore[misc]
    return f"AG-{a}-{b}-{c}-{d}"


def handle_webhook(request_body: bytes, stripe_signature: str) -> dict:
    """
    Process Stripe webhooks.
    Handles: checkout.session.completed, customer.subscription.deleted
    """
    init_stripe()
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    try:
        event = stripe.Webhook.construct_event(
            request_body, stripe_signature, webhook_secret
        )
    except stripe.error.SignatureVerificationError:  # type: ignore
        raise HTTPException(status_code=400, detail="Invalid Stripe signature.")

    event_type = event["type"]
    data       = event["data"]["object"]

    if event_type == "checkout.session.completed":
        _on_subscription_created(data)

    elif event_type in ("customer.subscription.deleted", "customer.subscription.paused"):
        _on_subscription_cancelled(data)

    elif event_type == "invoice.payment_failed":
        _on_payment_failed(data)

    return {"status": "ok", "event": event_type}


def _on_subscription_created(session: dict):
    """Activate license when payment succeeds."""
    meta    = session.get("metadata", {})
    user_id = meta.get("user_id")
    plan    = meta.get("plan", "STARTER")

    if not user_id:
        logger.error("[Payments] No user_id in metadata!")
        return

    key        = generate_license_key()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=32)).isoformat()

    create_license(user_id, plan, key, expires_at)
    upsert_user(user_id, {"plan": plan, "license_key": key, "active": True})

    logger.info(f"[Payments] ✅ License activated | User={user_id} Plan={plan} Key={key}")


def _on_subscription_cancelled(subscription: dict):
    """Deactivate license on cancellation."""
    meta    = subscription.get("metadata", {})
    user_id = meta.get("user_id")
    if user_id:
        deactivate_license(user_id)
        upsert_user(user_id, {"active": False, "plan": None})
        logger.info(f"[Payments] ❌ License deactivated | User={user_id}")


def _on_payment_failed(invoice: dict):
    customer_id = invoice.get("customer")
    logger.warning(f"[Payments] ⚠️ Payment failed for customer={customer_id} — consider sending reminder email.")
