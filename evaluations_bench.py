"""
Bedrock Model Evaluation Benchmark
Academic exercise — evaluates all accessible on-demand text models in us-east-1
across multiple categories with ground-truth scoring.

Metrics per response:
  - ROUGE-1 F1   (unigram overlap with ground truth)
  - ROUGE-2 F1   (bigram overlap)
  - Jaccard       (token-level set similarity)
  - Keyword Coverage (fraction of expected key concepts present)
  - Latency (seconds)
  - Token usage (input / output from Converse metadata)

Modes:
  python evaluations_bench.py                        # run all models, all custom test cases
  python evaluations_bench.py --filter claude        # only models whose ID contains "claude"
  python evaluations_bench.py --dry-run              # print model/test list, no API calls
  python evaluations_bench.py --fsi-mmlu             # FSI knowledge benchmark (MMLU financial MCQ)
  python evaluations_bench.py --fsi-mmlu --n-per-subject 30  # 30 questions per subject (150 total)
  python evaluations_bench.py --fsi-mmlu --mantle    # FSI benchmark on Mantle-exclusive models

FSI MMLU mode:
  Uses MMLU financial subsets as a proxy for FinBench (FinBench's dataset script is incompatible
  with datasets>=3.0). Subjects: professional_accounting, business_ethics, econometrics,
  high_school_macroeconomics, high_school_microeconomics.
  Scoring: exact-match accuracy on A/B/C/D (much stricter than ROUGE).
  Requires: pip install datasets
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

# Optional: HuggingFace datasets (required only for --fsi-mmlu mode)
try:
    from datasets import load_dataset as hf_load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Mantle support: requests + botocore SigV4
try:
    import requests as _requests
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest as _AWSRequest
    MANTLE_AVAILABLE = True
except ImportError:
    MANTLE_AVAILABLE = False

MANTLE_BASE = "https://bedrock-mantle.us-east-1.api.aws"

# Mantle-exclusive text models (not reachable via standard bedrock-runtime Converse)
# Skip: voxtral (audio), palmyra-vision, qwen3-vl (vision), safeguard models
MANTLE_MODELS = {
    "OpenAI GPT-5": [
        "openai.gpt-5.6-sol",
        "openai.gpt-5.6-luna",
        "openai.gpt-5.6-terra",
        "openai.gpt-5.5",
        "openai.gpt-5.4",
    ],
    "xAI Grok": [
        "xai.grok-4.3",
    ],
    "Google Gemma 4": [
        "google.gemma-4-31b",
        "google.gemma-4-26b-a4b",
        "google.gemma-4-e2b",
    ],
    "Qwen3 (Mantle)": [
        "qwen.qwen3-235b-a22b-2507",
        "qwen.qwen3-coder-480b-a35b-instruct",
        "qwen.qwen3-coder-next",
    ],
    "DeepSeek (Mantle)": [
        "deepseek.v3.1",
    ],
    "Z.AI GLM (Mantle)": [
        "zai.glm-4.6",
    ],
    "Moonshot Kimi (Mantle)": [
        "moonshotai.kimi-k2-thinking",
        "moonshotai.kimi-k2.5",
    ],
    "Anthropic Claude (Mantle)": [
        "anthropic.claude-sonnet-5",
        "anthropic.claude-opus-5",
        "anthropic.claude-fable-5",
        "anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-7",
        "anthropic.claude-haiku-4-5",
    ],
}
ALL_MANTLE_MODELS = [m for models in MANTLE_MODELS.values() for m in models]


def _mantle_path(model_id: str) -> str:
    """Route model to the correct Mantle inference path."""
    if model_id.startswith("anthropic."):
        return "/anthropic/v1/messages"
    # GPT-5.6+ and Gemma 4 require /openai/v1 path
    if model_id.startswith("google.gemma-4") or (
        model_id.startswith("openai.gpt-5.") and
        any(model_id.startswith(f"openai.gpt-5.{v}") for v in ["6"])
    ):
        return "/openai/v1/chat/completions"
    return "/v1/chat/completions"


def _mantle_payload(model_id: str, prompt: str, max_tokens: int,
                    include_temperature: bool = True) -> dict:
    if model_id.startswith("anthropic."):
        return {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
    # GPT-5.6+ uses max_completion_tokens instead of max_tokens
    is_gpt56 = model_id.startswith("openai.gpt-5.6")
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    if is_gpt56:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    if include_temperature and not is_gpt56:
        payload["temperature"] = 0.1
    return payload


def _mantle_extract(model_id: str, resp_json: dict) -> str:
    if model_id.startswith("anthropic."):
        blocks = resp_json.get("content", [])
        text_blocks = [b["text"] for b in blocks if b.get("type") == "text" or "text" in b]
        return text_blocks[0] if text_blocks else ""
    choices = resp_json.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    # Handle thinking-style responses (content may be list or string or None)
    content = msg.get("content") or ""
    if isinstance(content, list):
        text_parts = [b.get("text", "") for b in content if b.get("type") == "text" or "text" in b]
        return " ".join(text_parts)
    return str(content) if content is not None else ""


def invoke_mantle(model_id: str, prompt: str, max_tokens: int = 400,
                  region: str = "us-east-1") -> InvokeResult:
    """Call Bedrock Mantle endpoint via SigV4 (no separate API key needed)."""
    if not MANTLE_AVAILABLE:
        return InvokeResult(success=False, error="requests/botocore not installed")

    start = time.time()
    path = _mantle_path(model_id)
    url = f"{MANTLE_BASE}{path}"
    payload = _mantle_payload(model_id, prompt, max_tokens)
    body = json.dumps(payload).encode()

    try:
        session = boto3.Session()
        creds = session.get_credentials().get_frozen_credentials()

        def _signed_post(target_url: str, pay: dict, extra_headers: dict = None):
            b = json.dumps(pay).encode()
            h = {"Content-Type": "application/json"}
            if extra_headers:
                h.update(extra_headers)
            r = _AWSRequest(method="POST", url=target_url, data=b, headers=h)
            SigV4Auth(creds, "bedrock", region).add_auth(r)
            return _requests.post(target_url, headers=dict(r.headers), data=b, timeout=60)

        # Anthropic /anthropic/v1/messages requires anthropic-version header
        extra_hdrs = {"anthropic-version": "2023-06-01"} if model_id.startswith("anthropic.") else None

        payload = _mantle_payload(model_id, prompt, max_tokens, include_temperature=True)
        resp = _signed_post(url, payload, extra_headers=extra_hdrs)

        # Retry without temperature if model rejects it
        if resp.status_code == 400 and "unsupported_parameter" in resp.text and "temperature" in resp.text:
            payload = _mantle_payload(model_id, prompt, max_tokens, include_temperature=False)
            resp = _signed_post(url, payload, extra_headers=extra_hdrs)

        # Retry with max_completion_tokens if model rejects max_tokens (GPT-5.6+ style)
        if resp.status_code == 400 and "max_tokens" in resp.text and "unsupported_parameter" in resp.text:
            alt_payload = {k: v for k, v in payload.items() if k != "max_tokens"}
            alt_payload["max_completion_tokens"] = max_tokens
            resp = _signed_post(url, alt_payload, extra_headers=extra_hdrs)

        latency = time.time() - start

        if not resp.ok:
            # Retry: if /v1/chat/completions failed, try /openai/v1/chat/completions
            if path == "/v1/chat/completions":
                alt_url = f"{MANTLE_BASE}/openai/v1/chat/completions"
                resp = _signed_post(alt_url, payload, extra_headers=extra_hdrs)
                latency = time.time() - start

        if not resp.ok:
            return InvokeResult(success=False, latency=round(latency, 3),
                                error=f"HTTP {resp.status_code}: {resp.text[:150]}")

        rj = resp.json()
        output = _mantle_extract(model_id, rj)
        usage = rj.get("usage", {})
        return InvokeResult(
            success=True,
            output=output,
            latency=round(latency, 3),
            input_tokens=usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            output_tokens=usage.get("completion_tokens", usage.get("output_tokens", 0)),
        )
    except Exception as e:
        return InvokeResult(success=False, latency=round(time.time() - start, 3),
                            error=str(e)[:150])


# ─────────────────────────────────────────────
# Model registry (all ACTIVE text-generation models in us-east-1)
# Skip: audio (Voxtral, Nova Sonic), video (Pegasus), rerank (Cohere),
#        safeguard (gpt-oss-safeguard-*), vision-only (Palmyra Vision, Nemotron VL)
# ─────────────────────────────────────────────
MODEL_REGISTRY = {
    "Anthropic Claude": [
        # Cross-region inference profiles required for these models in this account
        "us.anthropic.claude-sonnet-5",
        "us.anthropic.claude-opus-5",
        "us.anthropic.claude-fable-5-1",
        "us.anthropic.claude-fable-5",
        "us.anthropic.claude-opus-4-8",
        "us.anthropic.claude-opus-4-7",
        "us.anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ],
    "Amazon Nova": [
        "amazon.nova-pro-v1:0",
        "amazon.nova-lite-v1:0",
        "amazon.nova-micro-v1:0",
    ],
    "Meta Llama": [
        "meta.llama3-70b-instruct-v1:0",
        "meta.llama3-8b-instruct-v1:0",
    ],
    "Mistral": [
        "mistral.mistral-large-3-675b-instruct",
        "mistral.magistral-small-2509",
        "mistral.devstral-2-123b",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral.ministral-3-14b-instruct",
        "mistral.ministral-3-8b-instruct",
        "mistral.mistral-large-2402-v1:0",
        "mistral.mistral-small-2402-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.ministral-3-3b-instruct",
    ],
    "DeepSeek": [
        "deepseek.v3.2",
    ],
    "Qwen": [
        "qwen.qwen3-next-80b-a3b",
        "qwen.qwen3-32b-v1:0",
        "qwen.qwen3-coder-30b-a3b-v1:0",
    ],
    "Google Gemma": [
        "google.gemma-3-27b-it",
        "google.gemma-3-12b-it",
        "google.gemma-3-4b-it",
    ],
    "NVIDIA Nemotron": [
        "nvidia.nemotron-super-3-120b",
        "nvidia.nemotron-nano-3-30b",
        "nvidia.nemotron-nano-9b-v2",
    ],
    "OpenAI OSS": [
        "openai.gpt-oss-120b-1:0",
        "openai.gpt-oss-20b-1:0",
    ],
    "Moonshot Kimi": [
        "moonshotai.kimi-k2.5",
        "moonshot.kimi-k2-thinking",
    ],
    "MiniMax": [
        "minimax.minimax-m2.5",
        "minimax.minimax-m2.1",
        "minimax.minimax-m2",
    ],
    "Z.AI GLM": [
        "zai.glm-5",
        "zai.glm-4.7",
        "zai.glm-4.7-flash",
    ],
    # AI21 Jamba excluded: Legacy models with access denied ("not been actively using")
    # "AI21 Jamba": ["ai21.jamba-1-5-large-v1:0", "ai21.jamba-1-5-mini-v1:0"],
}

# Flat list used for iteration
ALL_MODELS = [m for models in MODEL_REGISTRY.values() for m in models]


# ─────────────────────────────────────────────
# Test cases
# Each has: question, context passage (optional), ground_truth (reference),
#           key_concepts (list of tokens that MUST appear for full keyword score),
#           category
# ─────────────────────────────────────────────
TEST_CASES = [
    # ── FINANCE ──────────────────────────────────────────────────────────────
    {
        "id": "FIN-01",
        "category": "Finance",
        "question": "What is a 401(k) retirement plan and what is the 2024 contribution limit for employees under 50?",
        "context": "",
        "ground_truth": (
            "A 401(k) is a tax-advantaged employer-sponsored retirement savings plan. "
            "Employees contribute pre-tax dollars, reducing taxable income now, and pay taxes on withdrawal. "
            "The 2024 IRS contribution limit for employees under 50 is $23,000."
        ),
        "key_concepts": ["401", "tax", "retirement", "23000", "employer"],
    },
    {
        "id": "FIN-02",
        "category": "Finance",
        "question": (
            "If you invest $10,000 at 8% annual interest compounded monthly for 5 years, "
            "what is the approximate final balance? Show the formula used."
        ),
        "context": "",
        "ground_truth": (
            "Using A = P(1 + r/n)^(nt): A = 10000 * (1 + 0.08/12)^(12*5) ≈ $14,898. "
            "The formula is A = P(1 + r/n)^(nt) where P=principal, r=annual rate, n=compounding periods, t=years."
        ),
        "key_concepts": ["14898", "14,898", "compound", "formula", "monthly"],
    },
    {
        "id": "FIN-03",
        "category": "Banking / Regulation",
        "question": "What is Basel III and what are its three pillars?",
        "context": "",
        "ground_truth": (
            "Basel III is an international regulatory framework for banks developed by the Basel Committee on Banking Supervision "
            "after the 2008 financial crisis. Its three pillars are: "
            "Pillar 1 – Minimum Capital Requirements (CET1, Tier 1, Total Capital ratios); "
            "Pillar 2 – Supervisory Review Process; "
            "Pillar 3 – Market Discipline through public disclosure."
        ),
        "key_concepts": ["capital", "pillar", "supervisory", "disclosure", "liquidity"],
    },
    {
        "id": "FIN-04",
        "category": "Banking / Compliance",
        "question": "What does KYC stand for, and why is it critical in banking?",
        "context": "",
        "ground_truth": (
            "KYC stands for Know Your Customer. It is critical in banking to verify customer identities, "
            "prevent money laundering (AML), counter terrorist financing (CFT), reduce fraud, and comply with "
            "regulatory requirements such as FATF guidelines and local central bank rules."
        ),
        "key_concepts": ["Know Your Customer", "money laundering", "AML", "compliance", "identity"],
    },
    # ── CLOUD / TECHNOLOGY ────────────────────────────────────────────────────
    {
        "id": "CLOUD-01",
        "category": "Cloud Computing",
        "question": "What are the three main cloud service models? Give a one-sentence definition of each.",
        "context": "",
        "ground_truth": (
            "IaaS (Infrastructure as a Service): provides virtualized compute, storage, and networking. "
            "PaaS (Platform as a Service): provides a managed platform for developers to build and deploy apps. "
            "SaaS (Software as a Service): delivers fully managed applications over the internet."
        ),
        "key_concepts": ["IaaS", "PaaS", "SaaS", "infrastructure", "platform", "software"],
    },
    {
        "id": "CLOUD-02",
        "category": "Cloud Computing",
        "question": (
            "Amazon launched AWS in 2006. According to the passage below, what were the first three services offered? "
            "\n\nPassage: 'Amazon Web Services launched in 2006 with three foundational services: "
            "Amazon S3 for object storage, Amazon SQS for message queuing, and Amazon EC2 for compute.'"
        ),
        "context": "",
        "ground_truth": "Amazon S3, Amazon SQS, and Amazon EC2.",
        "key_concepts": ["S3", "SQS", "EC2"],
    },
    # ── CODING ────────────────────────────────────────────────────────────────
    {
        "id": "CODE-01",
        "category": "Coding",
        "question": "Write a Python function called is_palindrome that returns True if a string is a palindrome and False otherwise. Include a docstring.",
        "context": "",
        "ground_truth": (
            "def is_palindrome(s: str) -> bool:\n"
            '    """Return True if s is a palindrome."""\n'
            "    return s == s[::-1]"
        ),
        "key_concepts": ["def is_palindrome", "return", "[::-1]", "bool"],
    },
    {
        "id": "CODE-02",
        "category": "Coding / SQL",
        "question": (
            "The following SQL query is intended to find all employees with salary > 50000 "
            "but it returns an error. Identify and fix the bug:\n\n"
            "SELECT name, salary FROM employees WHERE salary > '50000';"
        ),
        "context": "",
        "ground_truth": (
            "The bug is comparing a numeric salary column to a string literal '50000'. "
            "Fix: remove the quotes. Correct query: SELECT name, salary FROM employees WHERE salary > 50000;"
        ),
        "key_concepts": ["string", "50000", "quotes", "numeric"],
    },
    # ── MATH / REASONING ─────────────────────────────────────────────────────
    {
        "id": "MATH-01",
        "category": "Mathematics",
        "question": "A train travels 300 miles in 4 hours. What is its average speed in mph?",
        "context": "",
        "ground_truth": "75 mph. Average speed = distance / time = 300 / 4 = 75 miles per hour.",
        "key_concepts": ["75"],
    },
    {
        "id": "MATH-02",
        "category": "Mathematics / Probability",
        "question": "If you flip a fair coin 3 times, what is the probability of getting exactly 2 heads? Express as a fraction and a percentage.",
        "context": "",
        "ground_truth": "3/8 = 37.5%. The combinations giving exactly 2 heads are HHT, HTH, THH (3 out of 8 total outcomes).",
        "key_concepts": ["3/8", "37.5", "3", "8"],
    },
    {
        "id": "MATH-03",
        "category": "Logical Reasoning",
        "question": (
            "Consider this argument: 'All roses are flowers. Some flowers fade quickly. "
            "Therefore, some roses fade quickly.' Is this argument logically valid? Explain why."
        ),
        "context": "",
        "ground_truth": (
            "No, this argument is NOT logically valid. It commits the fallacy of the undistributed middle. "
            "From 'All roses are flowers' and 'Some flowers fade quickly', we cannot conclude that those "
            "fading flowers are roses. A valid conclusion requires the middle term to be distributed."
        ),
        "key_concepts": ["not valid", "fallacy", "cannot conclude", "undistributed"],
    },
    # ── SCIENCE / GENERAL KNOWLEDGE ──────────────────────────────────────────
    {
        "id": "SCI-01",
        "category": "Science",
        "question": "Explain the greenhouse effect in 2–3 sentences, including at least two greenhouse gases.",
        "context": "",
        "ground_truth": (
            "The greenhouse effect is a natural process where atmospheric gases trap heat from the sun, "
            "warming the Earth's surface. Key greenhouse gases include carbon dioxide (CO2) and methane (CH4). "
            "Human activities have amplified this effect, driving global warming and climate change."
        ),
        "key_concepts": ["CO2", "carbon dioxide", "methane", "atmosphere", "heat", "warming"],
    },
    {
        "id": "HIST-01",
        "category": "History",
        "question": "In what year did World War II end, and what were the two primary theaters of war?",
        "context": "",
        "ground_truth": "World War II ended in 1945. The two primary theaters were the European Theater and the Pacific Theater.",
        "key_concepts": ["1945", "Europe", "Pacific"],
    },
    # ── INSTRUCTION FOLLOWING ────────────────────────────────────────────────
    {
        "id": "INST-01",
        "category": "Instruction Following",
        "question": (
            "List exactly 5 programming languages. Format your answer as a numbered list. "
            "The first item MUST be Python."
        ),
        "context": "",
        "ground_truth": "1. Python\n2. Java\n3. JavaScript\n4. C++\n5. Go",
        "key_concepts": ["1", "Python", "2", "3", "4", "5"],
    },
    # ── SUMMARIZATION ────────────────────────────────────────────────────────
    {
        "id": "SUM-01",
        "category": "Summarization",
        "question": (
            "Summarize the following passage in exactly one sentence:\n\n"
            "'Generative AI models like large language models are trained on vast datasets of text and can produce "
            "human-like text, answer questions, write code, and perform a wide range of language tasks. "
            "They are built using transformer architectures and trained with techniques such as RLHF. "
            "Their capabilities have led to rapid adoption across industries including healthcare, finance, and software development.'"
        ),
        "context": "",
        "ground_truth": (
            "Generative AI large language models, built on transformer architectures and trained with RLHF, "
            "can perform diverse language tasks and are being rapidly adopted across industries."
        ),
        "key_concepts": ["language model", "transformer", "text", "tasks"],
    },
]


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def rouge_n(hypothesis: str, reference: str, n: int) -> float:
    """Compute ROUGE-N F1 score."""
    def ngrams(tokens, n):
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)
    hyp_ng = ngrams(hyp_tokens, n)
    ref_ng = ngrams(ref_tokens, n)

    if not ref_ng or not hyp_ng:
        return 0.0

    ref_count = {}
    for g in ref_ng:
        ref_count[g] = ref_count.get(g, 0) + 1

    overlap = 0
    hyp_count = {}
    for g in hyp_ng:
        hyp_count[g] = hyp_count.get(g, 0) + 1

    for g, count in hyp_count.items():
        overlap += min(count, ref_count.get(g, 0))

    precision = overlap / len(hyp_ng) if hyp_ng else 0.0
    recall = overlap / len(ref_ng) if ref_ng else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def jaccard(hypothesis: str, reference: str) -> float:
    h = set(tokenize(hypothesis))
    r = set(tokenize(reference))
    if not r and not h:
        return 1.0
    return len(h & r) / len(h | r) if (h | r) else 0.0


def keyword_coverage(hypothesis: str, key_concepts: list[str]) -> float:
    """Fraction of key concepts present in the hypothesis (case-insensitive)."""
    if not key_concepts:
        return 1.0
    hyp_lower = hypothesis.lower()
    hits = sum(1 for kw in key_concepts if kw.lower() in hyp_lower)
    return hits / len(key_concepts)


def score_response(output: str, ground_truth: str, key_concepts: list[str]) -> dict:
    return {
        "rouge1": round(rouge_n(output, ground_truth, 1), 4),
        "rouge2": round(rouge_n(output, ground_truth, 2), 4),
        "jaccard": round(jaccard(output, ground_truth), 4),
        "keyword_coverage": round(keyword_coverage(output, key_concepts), 4),
    }


# ─────────────────────────────────────────────
# Model invocation
# ─────────────────────────────────────────────

@dataclass
class InvokeResult:
    success: bool
    output: str = ""
    latency: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


def invoke_converse(client, model_id: str, prompt: str, max_tokens: int = 400,
                    temperature: Optional[float] = 0.1) -> InvokeResult:
    """
    Call Bedrock Converse API. Works uniformly across all providers.
    Newer Claude models (Sonnet 5, Opus 5, Fable 5+) reject temperature — falls back
    automatically by retrying without it.
    """
    start = time.time()

    def _call(include_temp: bool):
        cfg = {"maxTokens": max_tokens}
        if include_temp and temperature is not None:
            cfg["temperature"] = temperature
        return client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=cfg,
        )

    try:
        try:
            response = _call(include_temp=True)
        except ClientError as e:
            if "temperature" in str(e).lower() and "deprecated" in str(e).lower():
                response = _call(include_temp=False)
            else:
                raise
        latency = time.time() - start
        # Some models (thinking models like kimi-k2-thinking) return a "thinking"
        # block before the actual text block — find the first "text" type block.
        content_blocks = response["output"]["message"]["content"]
        text_blocks = [b for b in content_blocks if b.get("type") == "text" or "text" in b]
        output = text_blocks[0]["text"] if text_blocks else ""
        usage = response.get("usage", {})
        return InvokeResult(
            success=True,
            output=output,
            latency=round(latency, 3),
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )
    except ClientError as e:
        return InvokeResult(
            success=False,
            latency=round(time.time() - start, 3),
            error=f"{e.response['Error']['Code']}: {e.response['Error']['Message'][:120]}",
        )
    except Exception as e:
        return InvokeResult(
            success=False,
            latency=round(time.time() - start, 3),
            error=str(e)[:150],
        )


def _worker(args):
    """Thread worker: (client, model_id, test_case) → result dict."""
    client, model_id, tc = args
    prompt = tc["question"]
    result = invoke_converse(client, model_id, prompt)
    row = {
        "model_id": model_id,
        "test_id": tc["id"],
        "category": tc["category"],
        "question": tc["question"][:80] + "..." if len(tc["question"]) > 80 else tc["question"],
        "success": result.success,
        "latency_s": result.latency,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "output_snippet": result.output[:200] if result.success else "",
        "error": result.error,
    }
    if result.success:
        scores = score_response(result.output, tc["ground_truth"], tc["key_concepts"])
        row.update(scores)
    else:
        row.update({"rouge1": None, "rouge2": None, "jaccard": None, "keyword_coverage": None})
    return row


# ─────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────

def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-model metrics and rank by composite score."""
    success_df = df[df["success"] == True].copy()
    if success_df.empty:
        return pd.DataFrame()

    agg = (
        success_df.groupby("model_id")
        .agg(
            tests_passed=("success", "sum"),
            avg_rouge1=("rouge1", "mean"),
            avg_rouge2=("rouge2", "mean"),
            avg_jaccard=("jaccard", "mean"),
            avg_keyword_cov=("keyword_coverage", "mean"),
            avg_latency_s=("latency_s", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
    )

    total_tests = df.groupby("model_id")["test_id"].count().reset_index(name="tests_total")
    agg = agg.merge(total_tests, on="model_id")
    agg["success_rate"] = (agg["tests_passed"] / agg["tests_total"]).round(3)

    # Composite score: equal weight on the four quality metrics (failures contribute 0)
    agg["composite_score"] = (
        0.25 * agg["avg_rouge1"]
        + 0.25 * agg["avg_rouge2"]
        + 0.25 * agg["avg_jaccard"]
        + 0.25 * agg["avg_keyword_cov"]
    ).round(4)

    agg = agg.sort_values("composite_score", ascending=False).reset_index(drop=True)
    agg.index += 1  # 1-based rank

    return agg


def print_leaderboard(lb: pd.DataFrame):
    cols = ["model_id", "composite_score", "avg_keyword_cov", "avg_rouge1", "avg_rouge2",
            "avg_latency_s", "success_rate", "tests_passed", "tests_total"]
    print("\n" + "=" * 100)
    print("  LEADERBOARD  (ranked by composite score = mean of ROUGE-1, ROUGE-2, Jaccard, Keyword Coverage)")
    print("=" * 100)
    display = lb[cols].copy()
    display.columns = ["Model", "Composite", "KW-Cov", "ROUGE-1", "ROUGE-2", "Latency(s)", "SuccRate", "Pass", "Total"]
    print(display.to_string(index=True))
    print("=" * 100 + "\n")


def print_category_breakdown(df: pd.DataFrame):
    success_df = df[df["success"] == True]
    if success_df.empty:
        return
    cat_agg = (
        success_df.groupby(["model_id", "category"])["composite_score"]
        .mean()
        .unstack(fill_value=0)
        .round(3)
    )
    # Add composite_score column to success_df for this to work
    if "composite_score" not in success_df.columns:
        success_df = success_df.copy()
        success_df["composite_score"] = (
            success_df["rouge1"].fillna(0)
            + success_df["rouge2"].fillna(0)
            + success_df["jaccard"].fillna(0)
            + success_df["keyword_coverage"].fillna(0)
        ) / 4

    cat_agg = (
        success_df.groupby(["model_id", "category"])["composite_score"]
        .mean()
        .unstack(fill_value=0)
        .round(3)
    )
    print("\n" + "=" * 100)
    print("  CATEGORY BREAKDOWN  (mean composite score per model per category)")
    print("=" * 100)
    print(cat_agg.to_string())
    print("=" * 100 + "\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# FSI MMLU benchmark (financial MCQ, exact-match accuracy)
# ─────────────────────────────────────────────

FSI_MMLU_SUBJECTS = {
    "professional_accounting": "Professional Accounting",
    "business_ethics": "Business Ethics",
    "econometrics": "Econometrics",
    "high_school_macroeconomics": "Macroeconomics",
    "high_school_microeconomics": "Microeconomics",
}
ANSWER_LETTERS = ["A", "B", "C", "D"]


def load_fsi_mmlu_questions(n_per_subject: int = 20, seed: int = 42) -> list[dict]:
    """
    Load n_per_subject questions from each FSI MMLU subject.
    Returns list of dicts with: subject, subject_label, question, choices, answer_letter.
    """
    if not HF_AVAILABLE:
        raise ImportError("Run: pip install datasets")

    import random
    random.seed(seed)
    questions = []
    for subject_id, subject_label in FSI_MMLU_SUBJECTS.items():
        ds = hf_load_dataset("cais/mmlu", subject_id, split="test")
        indices = random.sample(range(len(ds)), min(n_per_subject, len(ds)))
        for i in indices:
            ex = ds[i]
            questions.append({
                "subject": subject_id,
                "subject_label": subject_label,
                "question": ex["question"],
                "choices": ex["choices"],           # list of 4 strings
                "answer_letter": ANSWER_LETTERS[ex["answer"]],  # int → A/B/C/D
            })
    return questions


def format_mcq_prompt(question: str, choices: list[str]) -> str:
    choice_text = "\n".join(f"{ANSWER_LETTERS[i]}: {c}" for i, c in enumerate(choices))
    return (
        "Answer the following multiple-choice question from a financial services knowledge test. "
        "Respond with ONLY the letter of the correct answer (A, B, C, or D). "
        "Do not explain.\n\n"
        f"Question: {question}\n{choice_text}\n\nAnswer:"
    )


def extract_answer_letter(response_text: str) -> str:
    """
    Extract A/B/C/D from model response.
    Handles both direct answers and chain-of-thought reasoning models
    (e.g. nemotron, thinking models) that put the final answer at the end.
    Priority:
      1. "Answer: X" / "answer is X" / "The answer is X" patterns
      2. Standalone letter on its own line (common final answer format)
      3. First standalone letter at very start of response
      4. Last standalone letter in response (chain-of-thought puts answer at end)
    """
    if not response_text:
        return ""
    text = str(response_text).strip()

    # 1. Explicit answer declaration patterns
    for pat in [
        r"(?:the\s+)?answer\s+is[:\s]+([ABCD])\b",
        r"\bAnswer[:\s]+([ABCD])\b",
        r"^([ABCD])[.):]\s",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    # 2. Standalone letter on its own line (e.g. final line "B")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        if re.fullmatch(r"[ABCD][.):]?", line, re.IGNORECASE):
            return line[0].upper()

    # 3. First standalone letter at start
    m = re.match(r"^\s*([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 4. Last standalone letter anywhere (chain-of-thought conclusion)
    matches = re.findall(r"\b([ABCD])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    return ""


def _fsi_worker(args):
    """Thread worker for FSI MCQ: returns result row."""
    client, model_id, q, max_tokens, use_mantle, region = args
    prompt = format_mcq_prompt(q["question"], q["choices"])
    if use_mantle:
        result = invoke_mantle(model_id, prompt, max_tokens=max_tokens, region=region)
    else:
        result = invoke_converse(client, model_id, prompt, max_tokens=max_tokens, temperature=None)
    predicted = extract_answer_letter(result.output) if result.success else ""
    correct = predicted == q["answer_letter"]
    return {
        "model_id": model_id,
        "subject": q["subject_label"],
        "question_snippet": q["question"][:80] + "..." if len(q["question"]) > 80 else q["question"],
        "answer_letter": q["answer_letter"],
        "predicted_letter": predicted,
        "correct": correct,
        "success": result.success,
        "latency_s": result.latency,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "error": result.error,
    }


def fsi_mmlu_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-model MCQ accuracy and rank."""
    agg = (
        df.groupby("model_id")
        .agg(
            correct=("correct", "sum"),
            total=("correct", "count"),
            success_rate=(
                "success",
                lambda x: round(x.sum() / len(x), 3),
            ),
            avg_latency_s=("latency_s", "mean"),
        )
        .reset_index()
    )
    agg["accuracy"] = (agg["correct"] / agg["total"]).round(4)
    agg = agg.sort_values("accuracy", ascending=False).reset_index(drop=True)
    agg.index += 1

    # Per-subject accuracy
    subject_acc = (
        df.groupby(["model_id", "subject"])["correct"]
        .mean()
        .unstack(fill_value=0)
        .round(3)
    )
    agg = agg.merge(subject_acc.reset_index(), on="model_id", how="left")
    return agg


def print_fsi_leaderboard(lb: pd.DataFrame, n_questions: int):
    cols = ["model_id", "accuracy", "correct", "total", "avg_latency_s"] + list(FSI_MMLU_SUBJECTS.values())
    available_cols = [c for c in cols if c in lb.columns]
    print("\n" + "=" * 120)
    print(f"  FSI KNOWLEDGE LEADERBOARD  (MMLU Financial Subsets — {n_questions} questions, exact-match accuracy)")
    print("=" * 120)
    display = lb[available_cols].copy()
    print(display.to_string(index=True))
    print("=" * 120 + "\n")


def run_fsi_mmlu_bench(models: list[str], n_per_subject: int, max_workers: int,
                       region: str, dry_run: bool = False, use_mantle: bool = False):
    questions = load_fsi_mmlu_questions(n_per_subject)
    total_q = len(questions)
    total_calls = len(models) * total_q

    print(f"\n{'='*70}")
    print(f"  FSI Knowledge Benchmark (MMLU Financial Subsets)")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Region    : {region}")
    print(f"  Models    : {len(models)}")
    print(f"  Questions : {total_q} ({n_per_subject}/subject × {len(FSI_MMLU_SUBJECTS)} subjects)")
    print(f"  API calls : {total_calls}")
    print(f"  Scoring   : Exact-match accuracy on A/B/C/D")
    print(f"{'='*70}\n")

    for s, label in FSI_MMLU_SUBJECTS.items():
        n = sum(1 for q in questions if q["subject_label"] == label)
        print(f"  {label}: {n} questions")

    if dry_run:
        return

    client = boto3.client("bedrock-runtime", region_name=region)
    # 300 tokens: enough for chain-of-thought models (nemotron, GLM) to reason then answer
    work = [(client, model_id, q, 300, use_mantle, region) for model_id in models for q in questions]
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fsi_worker, item): item for item in work}
        for fut in as_completed(futures):
            row = fut.result()
            results.append(row)
            completed += 1
            status = "✓" if row["correct"] else ("✗" if row["success"] else "⚠")
            if completed % 50 == 0 or not row["success"]:
                print(
                    f"  [{completed:4d}/{total_calls}] {status} {row['model_id']:<55} "
                    f"{row['subject']:<28} pred={row['predicted_letter'] or '?'} "
                    f"ans={row['answer_letter']} {row['latency_s']:.2f}s"
                    + (f"  ERR: {row['error'][:50]}" if not row["success"] else "")
                )

    df = pd.DataFrame(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path = f"fsi_mmlu_raw_{ts}.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results → {raw_path}")

    lb = fsi_mmlu_leaderboard(df)
    lb_path = f"fsi_mmlu_leaderboard_{ts}.csv"
    lb.to_csv(lb_path, index=True)
    print(f"Leaderboard  → {lb_path}")

    print_fsi_leaderboard(lb, total_q)

    failures = df[~df["success"]]
    if not failures.empty:
        print(f"⚠  {len(failures)} failed calls (see raw CSV for details)")

    print(f"Done. {completed} calls, {df['correct'].sum()} correct out of {len(df[df['success']])} successful.")


# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Bedrock model evaluation benchmark")
    p.add_argument("--filter", type=str, default="", help="Only run models whose ID contains this string")
    p.add_argument("--dry-run", action="store_true", help="Print model/test list without calling APIs")
    p.add_argument("--max-workers", type=int, default=5, help="Concurrent API workers (default 5)")
    p.add_argument("--max-tokens", type=int, default=400, help="Max output tokens per call (default 400)")
    p.add_argument("--region", type=str, default="us-east-1", help="AWS region (default us-east-1)")
    # FSI MMLU mode
    p.add_argument("--fsi-mmlu", action="store_true", help="Run FSI knowledge benchmark (MMLU financial MCQ)")
    p.add_argument("--n-per-subject", type=int, default=20,
                   help="Questions per subject in FSI MMLU mode (default 20, max varies per subject)")
    # Mantle mode
    p.add_argument("--mantle", action="store_true",
                   help="Use Bedrock Mantle endpoint (SigV4, no API key needed); runs Mantle-exclusive models")
    return p.parse_args()


def main():
    args = parse_args()

    models = ALL_MODELS
    if args.filter:
        models = [m for m in models if args.filter.lower() in m.lower()]
        print(f"\nFilter '{args.filter}' matched {len(models)} model(s).")

    # ── FSI MMLU mode ────────────────────────────────────────────────────────
    if args.fsi_mmlu:
        if not HF_AVAILABLE:
            print("ERROR: Install HuggingFace datasets first:  pip install datasets")
            sys.exit(1)
        if args.mantle:
            # Use Mantle-exclusive models instead of standard registry
            models = ALL_MANTLE_MODELS
            if args.filter:
                models = [m for m in models if args.filter.lower() in m.lower()]
            print(f"\nMantle mode: {len(models)} Mantle-exclusive models selected.")
        run_fsi_mmlu_bench(
            models=models,
            n_per_subject=args.n_per_subject,
            max_workers=args.max_workers,
            region=args.region,
            dry_run=args.dry_run,
            use_mantle=args.mantle,
        )
        return
    # ────────────────────────────────────────────────────────────────────────

    if args.dry_run:
        print(f"\n{'─'*60}")
        print(f"DRY RUN — would evaluate {len(models)} models × {len(TEST_CASES)} test cases "
              f"= {len(models) * len(TEST_CASES)} API calls\n")
        for provider, mlist in MODEL_REGISTRY.items():
            filtered = [m for m in mlist if m in models]
            if filtered:
                print(f"  {provider}:")
                for m in filtered:
                    print(f"    {m}")
        print(f"\nTest cases:")
        for tc in TEST_CASES:
            print(f"  [{tc['id']}] {tc['category']}: {tc['question'][:70]}...")
        return

    total_calls = len(models) * len(TEST_CASES)
    print(f"\n{'='*70}")
    print(f"  Bedrock Model Evaluation Benchmark")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Region  : {args.region}")
    print(f"  Models  : {len(models)}")
    print(f"  Tests   : {len(TEST_CASES)}")
    print(f"  Calls   : {total_calls}")
    print(f"  Workers : {args.max_workers}")
    print(f"{'='*70}\n")

    client = boto3.client("bedrock-runtime", region_name=args.region)

    work = [(client, model_id, tc) for model_id in models for tc in TEST_CASES]
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_worker, item): item for item in work}
        for fut in as_completed(futures):
            row = fut.result()
            results.append(row)
            completed += 1
            status = "✓" if row["success"] else "✗"
            print(
                f"  [{completed:3d}/{total_calls}] {status} {row['model_id']:<55} "
                f"{row['test_id']:<10} latency={row['latency_s']:.2f}s"
                + (f"  ERR: {row['error'][:60]}" if not row["success"] else "")
            )

    df = pd.DataFrame(results)

    # Compute per-row composite for category breakdown
    df["composite_score"] = (
        df["rouge1"].fillna(0)
        + df["rouge2"].fillna(0)
        + df["jaccard"].fillna(0)
        + df["keyword_coverage"].fillna(0)
    ) / 4

    # Save raw results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = f"eval_results_raw_{ts}.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results → {raw_path}")

    # Leaderboard
    lb = leaderboard(df)
    lb_path = f"eval_leaderboard_{ts}.csv"
    lb.to_csv(lb_path, index=True)
    print(f"Leaderboard  → {lb_path}")

    print_leaderboard(lb)
    print_category_breakdown(df)

    # Failures summary
    failures = df[df["success"] == False]
    if not failures.empty:
        print(f"\n⚠  {len(failures)} failed calls:")
        for _, row in failures.iterrows():
            print(f"   {row['model_id']:<55} {row['test_id']:<10} {row['error']}")

    print(f"\nDone. {completed} calls completed, {len(failures)} errors.")


if __name__ == "__main__":
    main()
