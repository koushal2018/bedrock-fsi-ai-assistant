#!/usr/bin/env bash
# Part 2 – AppConfig setup for Financial AI Assistant model selection rules
# Idempotent: re-running updates config and re-deploys without error.
#
# Usage:
#   export AWS_REGION=us-east-1
#   bash appconfig_setup.sh
#
# Outputs (also written to appconfig_ids.env):
#   APPCONFIG_APP_ID, APPCONFIG_ENV_ID, APPCONFIG_PROFILE_ID

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
APP_NAME="FinancialAIAssistant"
ENV_NAME="Production"
PROFILE_NAME="ModelSelectionRules"
STRATEGY_NAME="FinancialAI-AllAtOnce"
IDS_FILE="$(dirname "$0")/appconfig_ids.env"

echo "==> Setting up AppConfig in region: $REGION"

# ── Application ────────────────────────────────────────────────────────────────
APP_ID=$(aws appconfig list-applications \
    --region "$REGION" \
    --query "Items[?Name=='${APP_NAME}'].Id" \
    --output text 2>/dev/null | tr -d '[:space:]')

if [[ -z "$APP_ID" ]]; then
    APP_ID=$(aws appconfig create-application \
        --name "$APP_NAME" \
        --description "Financial AI Assistant — model selection and guardrail configuration" \
        --region "$REGION" \
        --query "Id" --output text)
    echo "  Created application: $APP_ID"
else
    echo "  Using existing application: $APP_ID"
fi

# ── Environment ────────────────────────────────────────────────────────────────
ENV_ID=$(aws appconfig list-environments \
    --application-id "$APP_ID" \
    --region "$REGION" \
    --query "Items[?Name=='${ENV_NAME}'].Id" \
    --output text 2>/dev/null | tr -d '[:space:]')

if [[ -z "$ENV_ID" ]]; then
    ENV_ID=$(aws appconfig create-environment \
        --application-id "$APP_ID" \
        --name "$ENV_NAME" \
        --description "Production environment" \
        --region "$REGION" \
        --query "Id" --output text)
    echo "  Created environment: $ENV_ID"
else
    echo "  Using existing environment: $ENV_ID"
fi

# ── Configuration Profile ──────────────────────────────────────────────────────
PROFILE_ID=$(aws appconfig list-configuration-profiles \
    --application-id "$APP_ID" \
    --region "$REGION" \
    --query "Items[?Name=='${PROFILE_NAME}'].Id" \
    --output text 2>/dev/null | tr -d '[:space:]')

if [[ -z "$PROFILE_ID" ]]; then
    PROFILE_ID=$(aws appconfig create-configuration-profile \
        --application-id "$APP_ID" \
        --name "$PROFILE_NAME" \
        --description "Model routing rules, cost tiers, and guardrail settings" \
        --location-uri "hosted" \
        --type "AWS.Freeform" \
        --region "$REGION" \
        --query "Id" --output text)
    echo "  Created configuration profile: $PROFILE_ID"
else
    echo "  Using existing configuration profile: $PROFILE_ID"
fi

# ── Configuration content ──────────────────────────────────────────────────────
CONFIG_FILE=$(mktemp /tmp/financial-ai-config-XXXXXX.json)
trap "rm -f $CONFIG_FILE" EXIT

cat > "$CONFIG_FILE" <<'JSON'
{
  "economy":   "amazon.nova-micro-v1:0",
  "standard":  "amazon.nova-lite-v1:0",
  "premium":   "us.anthropic.claude-sonnet-4-6",
  "use_case_overrides": {
    "compliance_check": "us.anthropic.claude-sonnet-4-6",
    "complex_query":    "us.anthropic.claude-opus-4-8",
    "product_faq":      "amazon.nova-micro-v1:0",
    "customer_service": "amazon.nova-lite-v1:0"
  },
  "fallback_chain": [
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0"
  ],
  "guardrail_settings": {
    "inject_system_prompt": true,
    "require_financial_disclaimer": true,
    "blocked_topics": [
      "specific_investment_recommendations",
      "tax_evasion",
      "insider_trading",
      "money_laundering"
    ],
    "max_output_tokens": 1024
  },
  "latency_thresholds_ms": {
    "economy":  500,
    "standard": 2000,
    "premium":  5000
  }
}
JSON

CONFIG_OUT=$(mktemp /tmp/financial-ai-config-out-XXXXXX)
aws appconfig create-hosted-configuration-version \
    --application-id "$APP_ID" \
    --configuration-profile-id "$PROFILE_ID" \
    --content-type "application/json" \
    --content "fileb://$CONFIG_FILE" \
    --region "$REGION" \
    "$CONFIG_OUT" > /dev/null
CONFIG_VERSION=$(aws appconfig list-hosted-configuration-versions \
    --application-id "$APP_ID" \
    --configuration-profile-id "$PROFILE_ID" \
    --region "$REGION" \
    --query "Items[0].VersionNumber" --output text)
echo "  Created config version: $CONFIG_VERSION"

# ── Deployment strategy ────────────────────────────────────────────────────────
STRATEGY_ID=$(aws appconfig list-deployment-strategies \
    --region "$REGION" \
    --query "Items[?Name=='${STRATEGY_NAME}'].Id" \
    --output text 2>/dev/null | tr -d '[:space:]')

if [[ -z "$STRATEGY_ID" ]]; then
    STRATEGY_ID=$(aws appconfig create-deployment-strategy \
        --name "$STRATEGY_NAME" \
        --description "Immediate full deployment — suitable for non-traffic-shifting config" \
        --deployment-duration-in-minutes 0 \
        --growth-type "LINEAR" \
        --growth-factor 100 \
        --final-bake-time-in-minutes 0 \
        --replicate-to "NONE" \
        --region "$REGION" \
        --query "Id" --output text)
    echo "  Created deployment strategy: $STRATEGY_ID"
else
    echo "  Using existing deployment strategy: $STRATEGY_ID"
fi

# ── Deploy ─────────────────────────────────────────────────────────────────────
aws appconfig start-deployment \
    --application-id "$APP_ID" \
    --environment-id "$ENV_ID" \
    --configuration-profile-id "$PROFILE_ID" \
    --configuration-version "$CONFIG_VERSION" \
    --deployment-strategy-id "$STRATEGY_ID" \
    --description "Deploy model selection rules v${CONFIG_VERSION}" \
    --region "$REGION" \
    --output text > /dev/null

echo "  Deployment started for config version $CONFIG_VERSION"

# ── Write IDs file ─────────────────────────────────────────────────────────────
cat > "$IDS_FILE" <<EOF
# AppConfig IDs — source this file or set as Lambda env vars
export APPCONFIG_APP_ID="${APP_ID}"
export APPCONFIG_ENV_ID="${ENV_ID}"
export APPCONFIG_PROFILE_ID="${PROFILE_ID}"
export AWS_REGION="${REGION}"
EOF

echo ""
echo "==> Done. IDs written to: $IDS_FILE"
echo "    APPCONFIG_APP_ID     = $APP_ID"
echo "    APPCONFIG_ENV_ID     = $ENV_ID"
echo "    APPCONFIG_PROFILE_ID = $PROFILE_ID"
echo ""
echo "    Set these as Lambda environment variables:"
echo "    source $IDS_FILE"
