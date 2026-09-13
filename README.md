# Financial Services AI Assistant on Amazon Bedrock

**AWS GenAI Professional Certification — Capstone Exercise**

A production-grade reference implementation for deploying a customer-service AI assistant at a regulated financial institution using Amazon Bedrock. Covers model evaluation, dynamic model selection, resilient architecture, and model lifecycle management.

---

## Architecture Overview

```
Client
  │
  ▼
Amazon API Gateway (AWS_IAM auth)
  │
  ▼
AWS Lambda — Model Abstraction Layer
  │  • Reads routing rules from AWS AppConfig (no redeploy to change models)
  │  • Routes by use_case (compliance_check → Claude, product_faq → Nova Micro)
  │  • Injects FSI compliance system prompt per use_case
  │  • Falls back through model chain on throttling/errors
  │
  ▼
AWS Step Functions — Circuit Breaker (Express Workflow)
  │  • Tracks failures per model in DynamoDB
  │  • Opens circuit after 5 failures in 60s → skips to fallback
  │  • Graceful degradation: FSI canned responses when all models fail
  │
  ▼
Amazon Bedrock
  │  • Standard endpoint: Nova, Claude, Mistral, Llama, Qwen, DeepSeek, GLM, Kimi
  │  • Mantle endpoint (SigV4): GPT-5.x, Grok-4.3, Gemma 4, Qwen3-235B
  │
Cross-Region: us-east-1 (primary) + eu-west-1 (secondary DR)
  • eu-west-1 auto-selects Mistral (Nova not available there on-demand)
  • Region overrides managed in AppConfig — no code changes needed
```

---

## Project Structure

```
├── evaluations_bench.py          # Part 1 — Benchmark 60+ Bedrock models (FSI MCQ + Mantle)
├── model_comparison_report.py    # Part 1 — 4-dimension comparison: quality, latency, cost, guardrails
├── model_selection_strategy.json # Part 1 — AppConfig-ready model routing strategy (generated)
├── experiment.md                 # Exercise specification
│
├── part2/
│   ├── lambda_handler.py         # Model abstraction Lambda (AppConfig routing + FSI prompts)
│   ├── appconfig_setup.sh        # Deploy AppConfig model selection rules
│   └── api_gateway_setup.sh      # Deploy API Gateway REST API
│
├── part3/
│   ├── step_functions_definition.json  # Circuit breaker state machine (ASL)
│   ├── circuit_breaker_lambda.py       # DynamoDB-backed circuit breaker
│   ├── degradation_lambda.py           # Graceful degradation (FSI canned responses)
│   └── cloudformation.yaml            # Cross-region deployment template
│
└── part4/
    ├── prepare_dataset.py        # Financial Q&A dataset (50+ pairs, S3 upload)
    ├── train.py                  # SageMaker PEFT/LoRA fine-tuning script
    └── model_lifecycle.py        # Model registry, testing, canary rollout, rollback
```

---

## Part 1 — Model Benchmarking Results

Evaluated **75 models** across 13 providers on **MMLU Financial Subsets** (Professional Accounting, Business Ethics, Econometrics, Macroeconomics, Microeconomics — 100 questions per model).

### Composite Leaderboard (Quality 40% + Guardrail Compliance 40% + Latency 20%)

| Rank | Model | Composite | Accuracy | Guardrail | P50 | Cost/100q |
|------|-------|-----------|----------|-----------|-----|-----------|
| 🥇 | zai/glm-5 | 0.919 | 88% | **100%** | 0.61s | $0.007 |
| 🥈 | claude-sonnet-4-6 | 0.910 | 92% | **100%** | 1.09s | $0.049 |
| 3 | mistral-large-3 | 0.880 | 87% | 90% | 0.52s | $0.012 |
| 4 | kimi-k2.5 | 0.865 | 90% | 90% | 1.02s | $0.017 |
| 5 | llama3-70b | 0.855 | 71% | **100%** | 0.54s | $0.010 |
| 6 | deepseek-v3.2 | 0.836 | 86% | 80% | 0.52s | **$0.002** |

> **Key finding:** Guardrail compliance reshapes the ranking entirely. Claude Opus 5 (94% accuracy) drops out of the top 10 because it was not guardrail-tested. Nova Pro passes only 60% of financial compliance tests — not recommended for direct customer-facing FSI use without additional Bedrock Guardrails.

### Mantle-Exclusive Model Results (GPT-5.x, Grok, Gemma 4, Qwen3-235B)

| Model | Accuracy | Latency | Notes |
|-------|----------|---------|-------|
| openai/gpt-5.6-luna | 95% | 1.18s | Econometrics 100% |
| openai/gpt-5.4 | 93% | 1.91s | Top frontier overall |
| qwen3-235b | 90% | 0.90s | Best large-model value |
| gemma-4-31b | 87% | 1.71s | +31% vs Gemma 3 27B |
| xai/grok-4.3 | 42%* | 3.04s | *verbose format confuses MCQ extractor |

### Production Recommendations by Use Case

| Use Case | Model | Reason |
|----------|-------|--------|
| Compliance queries | `claude-sonnet-4-6` | 100% guardrail + 92% FSI accuracy |
| General customer service | `zai/glm-5` | Best composite, 100% compliance, $0.007/100q |
| High-volume / economy | `deepseek-v3.2` | 86% accuracy, $0.002/100q |
| Fallback / degradation | `amazon.nova-lite` | AWS-native, sub-1s, $0.0008/100q |
| Frontier tasks (Mantle) | `openai.gpt-5.4` | 93% accuracy via Mantle SigV4 |

---

## Part 2 — Dynamic Model Selection

Deploy the model abstraction layer:

```bash
# 1. Set up AppConfig (model routing rules + per-use-case prompts)
cd part2
export AWS_REGION=us-east-1
bash appconfig_setup.sh
source appconfig_ids.env

# 2. Deploy Lambda
zip function.zip lambda_handler.py
aws lambda create-function \
  --function-name FinancialAIAssistant \
  --runtime python3.12 \
  --role YOUR_ROLE_ARN \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://function.zip \
  --environment "Variables={APPCONFIG_APP_ID=$APPCONFIG_APP_ID,APPCONFIG_ENV_ID=$APPCONFIG_ENV_ID,APPCONFIG_PROFILE_ID=$APPCONFIG_PROFILE_ID}"

# 3. Deploy API Gateway
export LAMBDA_ARN=arn:aws:lambda:us-east-1:ACCOUNT:function:FinancialAIAssistant
bash api_gateway_setup.sh

# Test (requires AWS SigV4 — use awscurl)
awscurl --service execute-api --region us-east-1 \
  -X POST https://API_ID.execute-api.us-east-1.amazonaws.com/prod/generate \
  -d '{"prompt":"What is KYC?","use_case":"compliance_check"}'
```

**To change model routing without redeploying code** — edit the AppConfig JSON and run `start-deployment`. Lambda picks it up within 60 seconds.

---

## Part 3 — Resilient System Design

```bash
# Deploy via CloudFormation (cross-region)
aws cloudformation deploy \
  --template-file part3/cloudformation.yaml \
  --stack-name financial-ai-stack \
  --parameter-overrides Environment=prod AppConfigAppId=APP_ID AppConfigEnvId=ENV_ID AppConfigProfileId=PROFILE_ID \
  --region us-east-1 \
  --capabilities CAPABILITY_NAMED_IAM

# Secondary region (eu-west-1) — Mistral auto-selected via region_overrides in AppConfig
aws cloudformation deploy \
  --template-file part3/cloudformation.yaml \
  --stack-name financial-ai-stack \
  --parameter-overrides Environment=prod AppConfigAppId=APP_ID AppConfigEnvId=ENV_ID AppConfigProfileId=PROFILE_ID \
  --region eu-west-1 \
  --capabilities CAPABILITY_NAMED_IAM
```

**Circuit breaker behavior:**
- 5 failures within 60s → circuit OPEN → requests bypass primary model
- Success → circuit resets
- All models fail → Step Functions routes to graceful degradation Lambda

---

## Part 4 — Model Customization & Lifecycle

```bash
# Prepare fine-tuning dataset (synthetic FSI Q&A — no real customer data)
export TRAINING_BUCKET=your-bucket-name-with-account-suffix
pip install -r requirements.txt
python part4/prepare_dataset.py

# Launch SageMaker training job
python part4/train.py  # calls launch_training_job() at bottom

# Promote a candidate model through testing → canary → production
python -c "
from part4.model_lifecycle import promote_model_pipeline
promote_model_pipeline('financial-assistant', 'v1.0.0', region='us-east-1')
"
```

---

## Running the Benchmark

```bash
pip install boto3 pandas datasets

# FSI knowledge benchmark — all 48 standard Bedrock models
python evaluations_bench.py --fsi-mmlu

# Mantle-exclusive models (GPT-5.x, Grok-4.3, Gemma 4, Qwen3-235B)
python evaluations_bench.py --fsi-mmlu --mantle

# Full model comparison report with live guardrail tests
python model_comparison_report.py --run-guardrails \
  --guardrail-models "zai.glm-5,us.anthropic.claude-sonnet-4-6,mistral.mistral-large-3-675b-instruct"

# Filter to specific provider
python evaluations_bench.py --fsi-mmlu --filter "nova"
python evaluations_bench.py --fsi-mmlu --filter "us.anthropic"
```

---

## Security

All code has been scanned with [Holmes](https://portal.prod.holmes.aws.dev) (Content Security Review Rubric + HolmesContentSecurityReviewBaselinePolicy):

- IAM Bedrock permissions scoped to specific model ARNs (not wildcard)
- All IAM inline policies converted to AWS Managed Policies
- DynamoDB encrypted with KMS Customer Managed Key
- API Gateway `/generate` endpoint requires AWS_IAM (SigV4) authentication
- PII scrubbing: prompt content is never logged to CloudWatch
- GDPR/GLBA compliance notices in all data-handling code

---

## Prerequisites

- AWS account with Bedrock model access enabled (us-east-1)
- AWS credentials configured (`aws configure` or IAM role)
- Python 3.12+
- For Mantle endpoint models: standard AWS SigV4 credentials (no separate API key required)

---

## Disclaimer

This project is for educational purposes (AWS GenAI Professional certification). It demonstrates architectural patterns for FSI AI deployments. Operators deploying this in production are responsible for GDPR, PCI-DSS, GLBA, and applicable local financial regulations. See compliance notices in each source file. AWS operates under a [shared responsibility model](https://aws.amazon.com/compliance/shared-responsibility-model/).
