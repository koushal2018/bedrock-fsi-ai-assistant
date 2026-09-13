"""
Circuit Breaker Lambda — manages per-model failure state in DynamoDB.

Operations:
  check          → returns circuit_open bool, failure_count, last_failure ts
  record_failure → atomically increments failure_count, sets last_failure
  record_success → resets failure_count to 0

Circuit opens when: failure_count >= OPEN_THRESHOLD AND
                    last_failure within RECOVERY_WINDOW_SECONDS
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TABLE_NAME = os.environ.get("CIRCUIT_BREAKER_TABLE", "ModelCircuitBreaker")
OPEN_THRESHOLD = int(os.environ.get("CIRCUIT_OPEN_THRESHOLD", "5"))
RECOVERY_WINDOW_SECONDS = int(os.environ.get("RECOVERY_WINDOW_SECONDS", "60"))

dynamodb = boto3.resource("dynamodb")


def _ensure_table_exists() -> object:
    """Create DynamoDB table if it doesn't exist; return table resource."""
    client = boto3.client("dynamodb")
    try:
        client.create_table(
            TableName=TABLE_NAME,
            AttributeDefinitions=[{"AttributeName": "model_id", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "model_id", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
            SSESpecification={"Enabled": True},
            Tags=[
                {"Key": "Project", "Value": "FinancialAIAssistant"},
                {"Key": "ManagedBy", "Value": "CircuitBreakerLambda"},
            ],
        )
        logger.info("DynamoDB table %s created", TABLE_NAME)
        waiter = client.get_waiter("table_exists")
        waiter.wait(TableName=TABLE_NAME)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceInUseException":
            raise
    return dynamodb.Table(TABLE_NAME)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_epoch() -> int:
    return int(time.time())


def check_circuit(table, model_id: str) -> dict:
    try:
        response = table.get_item(Key={"model_id": model_id})
    except ClientError as e:
        logger.error(
            json.dumps({"event": "ddb_get_error", "model_id": model_id, "error": str(e)})
        )
        return {"circuit_open": False, "failure_count": 0, "last_failure": ""}

    item = response.get("Item", {})
    failure_count = int(item.get("failure_count", 0))
    last_failure_epoch = int(item.get("last_failure_epoch", 0))
    last_failure_iso = item.get("last_failure", "")

    seconds_since_failure = _now_epoch() - last_failure_epoch
    circuit_open = (
        failure_count >= OPEN_THRESHOLD
        and seconds_since_failure <= RECOVERY_WINDOW_SECONDS
    )

    logger.info(
        json.dumps(
            {
                "event": "circuit_check",
                "model_id": model_id,
                "failure_count": failure_count,
                "seconds_since_failure": seconds_since_failure,
                "circuit_open": circuit_open,
            }
        )
    )

    return {
        "circuit_open": circuit_open,
        "failure_count": failure_count,
        "last_failure": last_failure_iso,
        "seconds_since_failure": seconds_since_failure,
    }


def record_failure(table, model_id: str, error: dict | None = None) -> dict:
    now_iso = _now_iso()
    now_epoch = _now_epoch()
    try:
        response = table.update_item(
            Key={"model_id": model_id},
            UpdateExpression=(
                "ADD failure_count :inc "
                "SET last_failure = :ts, last_failure_epoch = :ep, last_error = :err"
            ),
            ExpressionAttributeValues={
                ":inc": 1,
                ":ts": now_iso,
                ":ep": now_epoch,
                ":err": json.dumps(error) if error else "unknown",
            },
            ReturnValues="ALL_NEW",
        )
        new_count = int(response["Attributes"].get("failure_count", 0))
        circuit_open = new_count >= OPEN_THRESHOLD

        logger.warning(
            json.dumps(
                {
                    "event": "failure_recorded",
                    "model_id": model_id,
                    "new_failure_count": new_count,
                    "circuit_open": circuit_open,
                    "timestamp": now_iso,
                }
            )
        )
        return {"failure_count": new_count, "circuit_open": circuit_open}
    except ClientError as e:
        logger.error(
            json.dumps({"event": "ddb_update_error", "model_id": model_id, "error": str(e)})
        )
        raise


def record_success(table, model_id: str) -> dict:
    try:
        table.update_item(
            Key={"model_id": model_id},
            UpdateExpression=(
                "SET failure_count = :zero, last_success = :ts"
            ),
            ExpressionAttributeValues={
                ":zero": 0,
                ":ts": _now_iso(),
            },
        )
        logger.info(
            json.dumps({"event": "success_recorded", "model_id": model_id})
        )
        return {"reset": True}
    except ClientError as e:
        logger.error(
            json.dumps({"event": "ddb_reset_error", "model_id": model_id, "error": str(e)})
        )
        raise


def lambda_handler(event, context):
    operation = event.get("operation", "check")
    model_id = event.get("model_id", "unknown")

    logger.info(
        json.dumps(
            {
                "event": "circuit_breaker_invoked",
                "operation": operation,
                "model_id": model_id,
                "request_id": context.aws_request_id if context else "local",
            }
        )
    )

    table = _ensure_table_exists()

    if operation == "check":
        return check_circuit(table, model_id)
    elif operation == "record_failure":
        return record_failure(table, model_id, event.get("error"))
    elif operation == "record_success":
        return record_success(table, model_id)
    else:
        raise ValueError(f"Unknown operation: {operation}")
