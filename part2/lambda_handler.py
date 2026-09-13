"""
Part 2 – Model Abstraction Lambda
Financial Services AI Assistant on Amazon Bedrock

Environment variables (set by appconfig_setup.sh / CloudFormation):
  APPCONFIG_APP_ID         AppConfig application ID
  APPCONFIG_ENV_ID         AppConfig environment ID
  APPCONFIG_PROFILE_ID     AppConfig configuration profile ID
  APPCONFIG_REGION         Region where AppConfig is hosted (default: us-east-1)
  AWS_REGION               AWS region (auto-set by Lambda runtime)

COMPLIANCE NOTICE — Operator Responsibility (Shared Responsibility Model):
  Deployers of this Lambda for EU customers must satisfy GDPR obligations for any
  personal data flowing through the API (including prompt content and model responses).
  If payment-related data is processed, PCI-DSS scoping applies to this function and
  its surrounding infrastructure. Operators are also responsible for applicable local
  financial regulations (e.g., UAE CBUAE, UK FCA, US GLBA/CFPB requirements).
  See https://aws.amazon.com/compliance/ for AWS compliance resources and shared
  responsibility guidance.
"""

import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

# ── Structured logging ────────────────────────────────────────────────────────
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def _log(level: str, msg: str, **extra):
    record = {"level": level, "message": msg, **extra}
    getattr(logger, level)(json.dumps(record))


# ── Default model routing (overridden by AppConfig) ───────────────────────────
MODEL_ROUTING: dict = {
    "economy": "amazon.nova-micro-v1:0",
    "standard": "amazon.nova-lite-v1:0",
    "premium": "us.anthropic.claude-sonnet-4-6",
    "use_case_overrides": {
        "compliance_check": "us.anthropic.claude-sonnet-4-6",
        "complex_query": "us.anthropic.claude-opus-4-8",
        "product_faq": "amazon.nova-micro-v1:0",
        "customer_service": "amazon.nova-lite-v1:0",
    },
    "fallback_chain": ["amazon.nova-lite-v1:0", "amazon.nova-micro-v1:0"],
}

TOKEN_COSTS: dict = {
    "amazon.nova-micro-v1:0":               {"input": 0.000035, "output": 0.000140},
    "amazon.nova-lite-v1:0":                {"input": 0.000060, "output": 0.000240},
    "amazon.nova-pro-v1:0":                 {"input": 0.000800, "output": 0.003200},
    "us.anthropic.claude-sonnet-4-6":       {"input": 0.003000, "output": 0.015000},
    "us.anthropic.claude-opus-4-8":         {"input": 0.015000, "output": 0.075000},
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 0.000800, "output": 0.004000},
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a financial services assistant for a regulated institution. "
    "Always recommend consulting a qualified financial advisor for personalized advice. "
    "Never provide specific investment recommendations. "
    "Comply with applicable financial regulations including KYC, AML, and consumer protection rules. "
    "If a question involves illegal activity, decline and explain why."
)

# ── AppConfig cache (module-level — survives warm invocations) ────────────────
_appconfig_client = None
_config_cache: dict      = {"data": None, "token": None, "fetched_at": 0.0}
_prompt_cache: dict      = {"data": None, "token": None, "fetched_at": 0.0}
_CONFIG_TTL_SECONDS = 60
_BEDROCK_CLIENT = None


def _get_appconfig_client():
    global _appconfig_client
    if _appconfig_client is None:
        # AppConfig is deployed in APPCONFIG_REGION (us-east-1 by default).
        # This is intentionally different from the Lambda's own region —
        # both regions share one AppConfig deployment for consistency.
        appconfig_region = os.environ.get("APPCONFIG_REGION", "us-east-1")
        _appconfig_client = boto3.client("appconfigdata", region_name=appconfig_region)
    return _appconfig_client


def _get_bedrock_client():
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is None:
        _BEDROCK_CLIENT = boto3.client("bedrock-runtime",
                                       region_name=os.environ.get("AWS_REGION", "us-east-1"))
    return _BEDROCK_CLIENT


def _fetch_appconfig() -> dict:
    """Return merged routing config (AppConfig overrides defaults when available)."""
    app_id = os.environ.get("APPCONFIG_APP_ID", "")
    env_id = os.environ.get("APPCONFIG_ENV_ID", "")
    profile_id = os.environ.get("APPCONFIG_PROFILE_ID", "")

    if not (app_id and env_id and profile_id):
        _log("warning", "AppConfig env vars not set; using hard-coded defaults")
        return MODEL_ROUTING

    now = time.time()
    cache = _config_cache

    # Within TTL — use cached data if we have it
    if cache["data"] is not None and (now - cache["fetched_at"]) < _CONFIG_TTL_SECONDS:
        return cache["data"]

    client = _get_appconfig_client()
    try:
        # Start a new session if we don't have a token yet
        if cache["token"] is None:
            resp = client.start_configuration_session(
                ApplicationIdentifier=app_id,
                EnvironmentIdentifier=env_id,
                ConfigurationProfileIdentifier=profile_id,
                RequiredMinimumPollIntervalInSeconds=15,
            )
            cache["token"] = resp["InitialConfigurationToken"]

        resp = client.get_latest_configuration(ConfigurationToken=cache["token"])
        cache["token"] = resp["NextPollConfigurationToken"]

        raw = resp["Configuration"].read()
        if raw:                                       # empty means "no change"
            merged = {**MODEL_ROUTING, **json.loads(raw)}
            cache["data"] = merged
            cache["fetched_at"] = now
            _log("info", "AppConfig refreshed", bytes_received=len(raw))
        elif cache["data"] is None:
            cache["data"] = MODEL_ROUTING             # first call, no data yet
            cache["fetched_at"] = now

    except Exception as exc:
        _log("warning", "AppConfig fetch failed; using defaults", error=str(exc))
        if cache["data"] is None:
            cache["data"] = MODEL_ROUTING

    return cache["data"]


def _fetch_prompts() -> dict:
    """Return prompt templates from AppConfig (second profile), with TTL cache."""
    app_id   = os.environ.get("APPCONFIG_APP_ID", "")
    env_id   = os.environ.get("APPCONFIG_ENV_ID", "")
    profile_id = os.environ.get("APPCONFIG_PROMPT_PROFILE_ID", "")

    if not (app_id and env_id and profile_id):
        return {}   # no prompt profile configured — caller uses DEFAULT_SYSTEM_PROMPT

    now   = time.time()
    cache = _prompt_cache

    if cache["data"] is not None and (now - cache["fetched_at"]) < _CONFIG_TTL_SECONDS:
        return cache["data"]

    client = _get_appconfig_client()
    try:
        if cache["token"] is None:
            resp = client.start_configuration_session(
                ApplicationIdentifier=app_id,
                EnvironmentIdentifier=env_id,
                ConfigurationProfileIdentifier=profile_id,
                RequiredMinimumPollIntervalInSeconds=15,
            )
            cache["token"] = resp["InitialConfigurationToken"]

        resp = client.get_latest_configuration(ConfigurationToken=cache["token"])
        cache["token"] = resp["NextPollConfigurationToken"]
        raw = resp["Configuration"].read()
        if raw:
            cache["data"] = json.loads(raw)
            cache["fetched_at"] = now
            _log("info", "Prompt templates refreshed", bytes_received=len(raw))
        elif cache["data"] is None:
            cache["data"] = {}
            cache["fetched_at"] = now
    except Exception as exc:
        _log("warning", "Prompt fetch failed; using defaults", error=str(exc))
        if cache["data"] is None:
            cache["data"] = {}

    return cache["data"]


def _apply_model_formatter(model_id: str, system: str, prompt: str, templates: dict) -> tuple[str, str]:
    """
    Apply per-model prompt formatting rules from AppConfig _model_formatters.
    Returns (formatted_system, formatted_prompt).
    The Converse API handles system separately, so we only wrap when needed.
    """
    formatters = templates.get("_model_formatters", {})
    fmt = formatters.get(model_id) or {}
    style = fmt.get("style", "prose")

    if style == "prose":
        return system, prompt   # no wrapping; Converse API handles system natively

    if style == "xml_tags":
        sys_tpl  = fmt.get("system_wrapper", "{system}")
        user_tpl = fmt.get("user_wrapper", "{prompt}")
        return sys_tpl.format(system=system), user_tpl.format(prompt=prompt)

    # instruct / llama3 styles: system wrapper prepended to user turn
    # (Converse API doesn't use a separate system block for these providers)
    sys_tpl  = fmt.get("system_wrapper", "{system}")
    user_tpl = fmt.get("user_wrapper", "{prompt}")
    wrapped_system = sys_tpl.format(system=system)
    wrapped_prompt = user_tpl.format(prompt=prompt)
    # Return empty system (handled inline) + wrapped user prompt
    return "", f"{wrapped_system}\n{wrapped_prompt}"


def _build_prompt_config(use_case: str, model_id: str = "") -> dict:
    """
    Return {system_prompt, few_shot, temperature, max_tokens} for the use_case,
    with prompt text formatted for the specific model.
    """
    templates = _fetch_prompts()
    template  = templates.get(use_case) or templates.get("default") or {}

    system   = template.get("system", DEFAULT_SYSTEM_PROMPT)
    few_shot = template.get("few_shot", [])
    temp     = template.get("temperature", 0.1)
    max_tok  = template.get("max_tokens", 512)

    # Apply per-model formatter if we know the model at this point
    if model_id:
        system, _ = _apply_model_formatter(model_id, system, "", templates)

    return {
        "system_prompt": system,
        "few_shot":      few_shot,
        "temperature":   temp,
        "max_tokens":    max_tok,
        "templates":     templates,   # pass through for per-model prompt wrapping at invoke time
    }


# ── Model selection ───────────────────────────────────────────────────────────

def _region_routing(routing: dict) -> dict:
    """
    Merge top-level routing with region_overrides for the current Lambda region.
    Region-specific keys win over the global defaults.
    """
    current_region = os.environ.get("AWS_REGION", os.environ.get("AWS_REGION_NAME", "us-east-1"))
    region_overrides = routing.get("region_overrides", {}).get(current_region, {})
    if region_overrides:
        _log("info", "Applying region overrides", region=current_region)
        return {**routing, **region_overrides}
    return routing


def _select_model(use_case: str, constraints: dict, routing: dict) -> str:
    """Return primary model ID based on routing rules (region-aware)."""
    effective = _region_routing(routing)
    overrides = effective.get("use_case_overrides", {})
    if use_case in overrides:
        return overrides[use_case]

    cost_tier = (constraints or {}).get("cost_tier", "standard")
    return effective.get(cost_tier, effective.get("standard", "amazon.nova-lite-v1:0"))


def _estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    costs = TOKEN_COSTS.get(model_id, {"input": 0.001, "output": 0.003})
    return (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])


# ── Bedrock invocation with retry + fallback ──────────────────────────────────

def _invoke_with_retry(model_id: str, prompt: str, max_tokens: int = 512,
                       system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                       few_shot: list = None) -> dict:
    """
    Call Bedrock Converse API with up to 3 retries (exponential back-off).
    Injects AppConfig system prompt and optional few-shot examples.
    Raises on final failure so the caller can try the next fallback model.
    """
    client = _get_bedrock_client()

    # Build messages: few-shot examples first, then the actual user query
    messages = []
    for ex in (few_shot or []):
        messages.append({"role": "user",      "content": [{"text": ex["user"]}]})
        messages.append({"role": "assistant", "content": [{"text": ex["assistant"]}]})
    messages.append({"role": "user", "content": [{"text": prompt}]})

    system = [{"text": system_prompt}]
    inference_cfg: dict = {"maxTokens": max_tokens}

    # Newer Claude 5/Fable models reject temperature — omit it universally
    # to keep routing logic simple; models use sensible defaults.

    backoff = 1.0
    last_exc = None
    for attempt in range(3):
        try:
            t0 = time.time()
            try:
                response = client.converse(
                    modelId=model_id,
                    messages=messages,
                    system=system,
                    inferenceConfig=inference_cfg,
                )
            except ClientError as sys_exc:
                # Older Mistral/Llama models don't support separate system messages —
                # retry by prepending system prompt to the first user turn.
                if "system" in str(sys_exc).lower() and "doesn't support" in str(sys_exc).lower():
                    _log("info", "Model lacks system-message support; prepending to user turn",
                         model=model_id)
                    patched = list(messages)
                    if patched and patched[0]["role"] == "user":
                        first_text = patched[0]["content"][0]["text"]
                        patched[0] = {"role": "user", "content": [
                            {"text": f"[System instructions: {system_prompt}]\n\n{first_text}"}
                        ]}
                    response = client.converse(
                        modelId=model_id,
                        messages=patched,
                        inferenceConfig=inference_cfg,
                    )
                else:
                    raise
            latency_ms = (time.time() - t0) * 1000

            content_blocks = response["output"]["message"]["content"]
            text_blocks = [b["text"] for b in content_blocks if "text" in b]
            output_text = text_blocks[0] if text_blocks else ""

            usage = response.get("usage", {})
            return {
                "model_id": model_id,
                "response": output_text,
                "latency_ms": round(latency_ms, 1),
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
            }

        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("ThrottlingException", "ServiceUnavailableException", "TooManyRequestsException"):
                _log("warning", "Throttled; retrying", model=model_id, attempt=attempt, backoff=backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                last_exc = exc
            else:
                raise       # non-retryable — propagate immediately

    raise last_exc          # exhausted retries


def _invoke_with_fallback(primary_model: str, prompt: str, routing: dict,
                          prompt_cfg: dict = None) -> dict:
    """Try primary model, then up to 2 models from fallback_chain (region-aware)."""
    effective = _region_routing(routing)
    fallback_chain = effective.get("fallback_chain", ["amazon.nova-lite-v1:0", "amazon.nova-micro-v1:0"])
    candidates = [primary_model] + [m for m in fallback_chain if m != primary_model][:2]
    cfg = prompt_cfg or {}

    for model_id in candidates:
        try:
            # Apply per-model prompt formatter from AppConfig
            templates = cfg.get("templates", {})
            sys_fmt, _ = _apply_model_formatter(
                model_id, cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT), prompt, templates
            )
            result = _invoke_with_retry(
                model_id, prompt,
                max_tokens    = cfg.get("max_tokens", 512),
                system_prompt = sys_fmt or cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                few_shot      = cfg.get("few_shot", []),
            )
            if model_id != primary_model:
                _log("warning", "Used fallback model", primary=primary_model, used=model_id)
            return result
        except Exception as exc:
            _log("error", "Model invocation failed", model=model_id, error=str(exc))

    raise RuntimeError(f"All models exhausted. Primary={primary_model}, tried={candidates}")


# ── Lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    request_id = context.aws_request_id if context else "local"
    _log("info", "Request received", request_id=request_id)

    # Parse body — supports two invocation styles:
    #   1. API Gateway proxy: event has a "body" string
    #   2. Direct / Step Functions: event fields are top-level
    if "body" in event and isinstance(event.get("body"), str):
        try:
            body = json.loads(event["body"])
        except json.JSONDecodeError as exc:
            return _error(400, f"Invalid JSON body: {exc}")
        is_apigw = True
    else:
        body = event   # Step Functions / direct Lambda invoke
        is_apigw = False

    prompt = body.get("prompt", "").strip()
    if not prompt:
        err = _error(400, "Missing required field: prompt")
        return err if is_apigw else {"statusCode": 400, "error": "Missing required field: prompt"}

    use_case    = body.get("use_case", "general")
    constraints = body.get("constraints", {})
    max_tokens  = min(int(constraints.get("max_tokens", 512)), 2048)

    # Load routing config (cached)
    routing = _fetch_appconfig()

    # Select model
    primary_model = _select_model(use_case, constraints, routing)

    # Load prompt config from AppConfig (system prompt, few-shot, temperature)
    prompt_cfg = _build_prompt_config(use_case)
    if constraints.get("max_tokens"):
        prompt_cfg["max_tokens"] = max_tokens   # request-level override wins

    _log("info", "Model selected", model=primary_model, use_case=use_case,
         cost_tier=constraints.get("cost_tier", "standard"),
         prompt_source="appconfig" if _fetch_prompts() else "default")

    # Invoke
    try:
        result = _invoke_with_fallback(primary_model, prompt, routing, prompt_cfg)
    except Exception as exc:
        _log("error", "All models failed", error=str(exc), request_id=request_id)
        return _error(503, "Service temporarily unavailable. Please try again shortly.")

    cost = _estimate_cost(result["model_id"], result["input_tokens"], result["output_tokens"])
    _log("info", "Request completed",
         model=result["model_id"],
         latency_ms=result["latency_ms"],
         input_tokens=result["input_tokens"],
         output_tokens=result["output_tokens"],
         estimated_cost_usd=round(cost, 6),
         request_id=request_id)

    payload = {
        "model_used": result["model_id"],
        "response": result["response"],
        "latency_ms": result["latency_ms"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "estimated_cost_usd": round(cost, 6),
    }

    # Step Functions / direct invoke → return dict directly
    if not is_apigw:
        return payload

    # API Gateway → wrap in HTTP response
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def _error(status: int, message: str) -> dict:
    _log("error", message, status=status)
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}),
    }
