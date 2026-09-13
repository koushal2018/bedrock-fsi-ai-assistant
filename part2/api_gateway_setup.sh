#!/usr/bin/env bash
# Part 2 – API Gateway setup for Financial AI Assistant
# Creates a REST API with /generate (POST → Lambda) and /health (GET → MOCK).
#
# Prerequisites:
#   export LAMBDA_ARN=arn:aws:lambda:us-east-1:ACCOUNT:function:FUNCTION_NAME
#   export AWS_REGION=us-east-1   (optional, defaults to us-east-1)
#   export CLOUDWATCH_ROLE_ARN=arn:aws:iam::ACCOUNT:role/APIGatewayCloudWatchRole  (optional)
#
# Usage:
#   bash api_gateway_setup.sh

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
API_NAME="FinancialAIAssistantAPI"
STAGE_NAME="prod"

# Validate required env var
if [[ -z "${LAMBDA_ARN:-}" ]]; then
    echo "ERROR: LAMBDA_ARN is not set."
    echo "  Export it first:  export LAMBDA_ARN=arn:aws:lambda:${REGION}:ACCOUNT_ID:function:FUNCTION_NAME"
    exit 1
fi

# Derive account ID from Lambda ARN  (arn:aws:lambda:REGION:ACCOUNT:function:NAME)
ACCOUNT_ID=$(echo "$LAMBDA_ARN" | cut -d: -f5)

echo "==> Creating API Gateway REST API: $API_NAME (region=$REGION)"

# ── Create REST API ────────────────────────────────────────────────────────────
API_ID=$(aws apigateway create-rest-api \
    --name "$API_NAME" \
    --description "Financial AI Assistant — Bedrock model abstraction API" \
    --endpoint-configuration '{"types":["REGIONAL"]}' \
    --region "$REGION" \
    --query "id" --output text)
echo "  API ID: $API_ID"

# Root resource ID
ROOT_ID=$(aws apigateway get-resources \
    --rest-api-id "$API_ID" \
    --region "$REGION" \
    --query "items[?path=='/'].id" --output text)

# ── /generate resource ─────────────────────────────────────────────────────────
GENERATE_ID=$(aws apigateway create-resource \
    --rest-api-id "$API_ID" \
    --parent-id "$ROOT_ID" \
    --path-part "generate" \
    --region "$REGION" \
    --query "id" --output text)
echo "  /generate resource: $GENERATE_ID"

# POST method — AWS_IAM requires callers to sign requests with SigV4 credentials.
# This prevents unauthenticated internet access to the Bedrock-backed endpoint.
aws apigateway put-method \
    --rest-api-id "$API_ID" \
    --resource-id "$GENERATE_ID" \
    --http-method POST \
    --authorization-type "AWS_IAM" \
    --region "$REGION" \
    --output text > /dev/null

# Lambda proxy integration
LAMBDA_URI="arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/${LAMBDA_ARN}/invocations"
aws apigateway put-integration \
    --rest-api-id "$API_ID" \
    --resource-id "$GENERATE_ID" \
    --http-method POST \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri "$LAMBDA_URI" \
    --region "$REGION" \
    --output text > /dev/null

# Method response
aws apigateway put-method-response \
    --rest-api-id "$API_ID" \
    --resource-id "$GENERATE_ID" \
    --http-method POST \
    --status-code 200 \
    --response-models '{"application/json":"Empty"}' \
    --region "$REGION" \
    --output text > /dev/null

echo "  POST /generate → Lambda proxy configured"

# ── /health resource ───────────────────────────────────────────────────────────
HEALTH_ID=$(aws apigateway create-resource \
    --rest-api-id "$API_ID" \
    --parent-id "$ROOT_ID" \
    --path-part "health" \
    --region "$REGION" \
    --query "id" --output text)
echo "  /health resource: $HEALTH_ID"

aws apigateway put-method \
    --rest-api-id "$API_ID" \
    --resource-id "$HEALTH_ID" \
    --http-method GET \
    --authorization-type "NONE" \
    --region "$REGION" \
    --output text > /dev/null

# MOCK integration returning 200 {"status":"ok"}
aws apigateway put-integration \
    --rest-api-id "$API_ID" \
    --resource-id "$HEALTH_ID" \
    --http-method GET \
    --type MOCK \
    --request-templates '{"application/json":"{\"statusCode\": 200}"}' \
    --region "$REGION" \
    --output text > /dev/null

aws apigateway put-method-response \
    --rest-api-id "$API_ID" \
    --resource-id "$HEALTH_ID" \
    --http-method GET \
    --status-code 200 \
    --response-models '{"application/json":"Empty"}' \
    --region "$REGION" \
    --output text > /dev/null

aws apigateway put-integration-response \
    --rest-api-id "$API_ID" \
    --resource-id "$HEALTH_ID" \
    --http-method GET \
    --status-code 200 \
    --response-templates '{"application/json":"{\"status\":\"ok\",\"service\":\"FinancialAIAssistant\"}"}' \
    --region "$REGION" \
    --output text > /dev/null

echo "  GET /health → MOCK 200 configured"

# ── Lambda invoke permission ───────────────────────────────────────────────────
aws lambda add-permission \
    --function-name "$LAMBDA_ARN" \
    --statement-id "APIGatewayInvoke-${API_ID}" \
    --action "lambda:InvokeFunction" \
    --principal "apigateway.amazonaws.com" \
    --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*/*" \
    --region "$REGION" \
    --output text > /dev/null
echo "  Lambda invoke permission granted"

# ── CloudWatch logging (optional — requires account-level role) ────────────────
if [[ -n "${CLOUDWATCH_ROLE_ARN:-}" ]]; then
    aws apigateway update-account \
        --patch-operations "[{\"op\":\"replace\",\"path\":\"/cloudwatchRoleArn\",\"value\":\"${CLOUDWATCH_ROLE_ARN}\"}]" \
        --region "$REGION" \
        --output text > /dev/null
    echo "  CloudWatch role set on account"
fi

# ── Deploy to stage ────────────────────────────────────────────────────────────
DEPLOYMENT_ID=$(aws apigateway create-deployment \
    --rest-api-id "$API_ID" \
    --stage-name "$STAGE_NAME" \
    --stage-description "Production stage" \
    --description "Initial deployment" \
    --region "$REGION" \
    --query "id" --output text)
echo "  Deployed to stage '$STAGE_NAME': $DEPLOYMENT_ID"

# Enable detailed CloudWatch metrics + logging on the stage
aws apigateway update-stage \
    --rest-api-id "$API_ID" \
    --stage-name "$STAGE_NAME" \
    --patch-operations \
        "op=replace,path=/*/*/metrics/enabled,value=true" \
        "op=replace,path=/*/*/logging/loglevel,value=INFO" \
        "op=replace,path=/tracingEnabled,value=true" \
    --region "$REGION" \
    --output text > /dev/null
echo "  Stage metrics, logging, and X-Ray tracing enabled"

# ── Output ────────────────────────────────────────────────────────────────────
INVOKE_URL="https://${API_ID}.execute-api.${REGION}.amazonaws.com/${STAGE_NAME}"

echo ""
echo "==> API Gateway setup complete"
echo "    API ID       : $API_ID"
echo "    Invoke URL   : $INVOKE_URL"
echo "    Generate     : POST ${INVOKE_URL}/generate"
echo "    Health check : GET  ${INVOKE_URL}/health"
echo ""
echo "    Test with (requires AWS SigV4 — use awscurl or AWS SDK):"
echo "    awscurl --service execute-api --region ${REGION} -X POST '${INVOKE_URL}/generate' \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"prompt\":\"What is a 401k?\",\"use_case\":\"product_faq\",\"constraints\":{\"cost_tier\":\"economy\"}}'"
echo "    # Health check (no auth): curl '${INVOKE_URL}/health'"

# Write endpoint file for use in other scripts
cat > "$(dirname "$0")/api_endpoint.env" <<EOF
export API_ID="${API_ID}"
export API_INVOKE_URL="${INVOKE_URL}"
export AWS_REGION="${REGION}"
EOF
echo "    Endpoint vars written to: $(dirname "$0")/api_endpoint.env"
