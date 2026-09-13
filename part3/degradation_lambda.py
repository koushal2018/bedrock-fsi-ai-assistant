"""
Graceful Degradation Lambda — returns FSI-appropriate canned responses
when both primary and fallback models are unavailable.

Always logs a structured degradation event to CloudWatch.
"""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CONTACT_NUMBER = os.environ.get("CONTACT_NUMBER", "1-800-BANK-HELP")
DOCS_URL = os.environ.get("DOCS_URL", "docs.example-bank.com")

DEGRADED_RESPONSES = {
    "product_faq": (
        f"Our product information is temporarily unavailable. "
        f"Please visit {DOCS_URL} or call {CONTACT_NUMBER}."
    ),
    "customer_service": (
        f"Our AI assistant is currently unavailable. "
        f"For immediate assistance, please call {CONTACT_NUMBER} "
        f"(Mon–Fri 8am–8pm EST)."
    ),
    "compliance_check": (
        "Automated compliance checking is temporarily offline. "
        "Please route this request to your compliance officer for manual review. "
        f"For urgent matters, call {CONTACT_NUMBER}."
    ),
    "complex_query": (
        "Our advanced query service is temporarily unavailable. "
        "A relationship manager will follow up within 1 business day. "
        f"For urgent matters, please call {CONTACT_NUMBER}."
    ),
    "account_inquiry": (
        "We are unable to process account inquiries at the moment. "
        f"For urgent matters, please call {CONTACT_NUMBER} "
        "or visit your nearest branch."
    ),
    "fraud_report": (
        "IMPORTANT: If you suspect fraud, please call our 24/7 fraud line "
        f"immediately at {CONTACT_NUMBER}. Do not delay — your account security matters."
    ),
}

DEFAULT_RESPONSE = (
    "Our service is temporarily unavailable. "
    "Please try again in a few minutes or call "
    f"{CONTACT_NUMBER} for immediate assistance."
)


def lambda_handler(event, context):
    use_case = event.get("use_case", "general")
    # Do NOT log prompt content — financial prompts may contain PII (account numbers,
    # personal details). Log only non-PII metadata: use_case and prompt length.
    prompt_length = len(str(event.get("prompt", "")))
    reason = event.get("reason", "all_models_unavailable")
    timestamp = datetime.now(timezone.utc).isoformat()

    response_text = DEGRADED_RESPONSES.get(use_case, DEFAULT_RESPONSE)

    logger.warning(
        json.dumps(
            {
                "event": "graceful_degradation",
                "use_case": use_case,
                "reason": reason,
                "prompt_length": prompt_length,
                "timestamp": timestamp,
                "request_id": context.aws_request_id if context else "local",
                "contact_number": CONTACT_NUMBER,
            }
        )
    )

    return {
        "model_used": "DEGRADED_SERVICE",
        "response": response_text,
        "use_case": use_case,
        "degraded": True,
        "timestamp": timestamp,
        "contact": CONTACT_NUMBER,
        "latency_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
