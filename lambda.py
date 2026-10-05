import os
import json
import re
import boto3
from botocore.eventstream import EventStream
from botocore.exceptions import ClientError, EventStreamError

REGION = os.environ.get("AWS_REGION", "us-east-1")
KB_ID = os.environ["KB_ID"]
GR_ID = os.environ.get("GUARDRAIL_ID")
GR_VER = os.environ.get("GUARDRAIL_VERSION", "DRAFT")

# IMPORTANT: set this to a real model ARN you have access to
# Example (check with list-foundation-models):
# arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0
MODEL_ARN = os.environ.get("FOUNDATION_MODEL_ARN")

client = boto3.client("bedrock-agent-runtime", region_name=REGION)

INSTRUCTION = (
    "Answer only with facts from the sources. "
    "If a price or detail is not stated, say so and point the user to https://www.cloudboosta.com/enroll/. "
    "Question: "
)

def reply(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }

def ask(question: str) -> str:
    agentic_cfg = {
        "rerankingModelType": "MANAGED",
    }

    if GR_ID:
        if not MODEL_ARN:
            raise ValueError(
                "GUARDRAIL_ID is set but FOUNDATION_MODEL_ARN is missing. "
                "Set FOUNDATION_MODEL_ARN to a valid Bedrock model ARN."
            )

        agentic_cfg["foundationModelType"] = "CUSTOM"
        agentic_cfg["foundationModelConfiguration"] = {
            "type": "BEDROCK_FOUNDATION_MODEL",
            "bedrockFoundationModelConfiguration": {
                "modelConfiguration": {
                    "modelArn": MODEL_ARN,
                }
            },
        }
        policy_cfg = {
            "bedrockGuardrailConfiguration": {
                "guardrailId": GR_ID,
                "guardrailVersion": GR_VER,
            }
        }
    else:
        agentic_cfg["foundationModelType"] = "MANAGED"
        policy_cfg = None

    params = {
        "messages": [
            {
                "role": "user",
                "content": {"text": INSTRUCTION + question},
            }
        ],
        "retrievers": [
            {
                "description": "CloudBoosta website content: courses, training, enrollment and services",
                "configuration": {
                    "knowledgeBase": {"knowledgeBaseId": KB_ID}
                },
            }
        ],
        "agenticRetrieveConfiguration": agentic_cfg,
        "generateResponse": True,
    }

    if policy_cfg:
        params["policyConfiguration"] = policy_cfg

    try:
        response = client.agentic_retrieve_stream(**params)
        stream = next(
            (v for v in response.values() if isinstance(v, EventStream)),
            None,
        )

        if stream is None:
            return "Sorry, I could not retrieve an answer at this time."

        final_answer = None
        parts = []

        for event in stream:
            if "responseEvent" in event:
                parts.append(event["responseEvent"].get("text", ""))
            elif "result" in event:
                gen = event["result"].get("generatedResponse") or {}
                final_answer = gen.get("answer")

        text = final_answer or "".join(parts)
        return re.sub(r"\s*(?:\[\d+\])+", "", text).strip()

    except (ClientError, EventStreamError) as ex:
        msg = str(ex)
        print("ERROR:", repr(ex))

        if "blocked by guardrail" in msg.lower():
            return (
                "I'm sorry, I can't process that particular request. "
                "Please rephrase your question or visit https://www.cloudboosta.com/enroll/ "
                "or contact the Academy WhatsApp advisor for help."
            )

        if "model identifier is invalid" in msg.lower():
            return (
                "Configuration error: the foundation model ARN is invalid or you do not have access to it. "
                "Please set a valid FOUNDATION_MODEL_ARN environment variable."
            )

        raise

def lambda_handler(event, context):
    body = event
    if isinstance(event.get("body"), str):
        try:
            body = json.loads(event["body"])
        except ValueError:
            body = {}

    question = (body.get("question") or "").strip()

    if not question:
        return reply(400, {"error": "Please include a question."})
    if len(question) > 1000:
        return reply(400, {"error": "Question is too long."})

    try:
        answer = ask(question)
        return reply(200, {"answer": answer})
    except Exception as ex:
        print("UNHANDLED ERROR:", repr(ex))
        return reply(500, {"error": "Something went wrong. Please try again."})
