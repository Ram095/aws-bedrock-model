import json
from typing import Iterable, List

import boto3


def _extract_text_from_payload(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""

    if "output" in payload and isinstance(payload["output"], dict):
        message = payload["output"].get("message", {})
        if isinstance(message, dict):
            texts: List[str] = []
            content_parts = message.get("content", [])
            if isinstance(content_parts, list):
                for item in content_parts:
                    if isinstance(item, dict) and item.get("type") == "text":
                        value = item.get("text")
                        if isinstance(value, str):
                            texts.append(value)
            text = "".join(texts).strip()
            if text:
                return text

    choices = payload.get("choices", [])
    if isinstance(choices, list) and choices:
        first_choice = choices[0] if isinstance(choices[0], dict) else {}

        text = first_choice.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

        message = first_choice.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                text_parts: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        value = item.get("text")
                        if isinstance(value, str):
                            text_parts.append(value)
                joined = "".join(text_parts).strip()
                if joined:
                    return joined

    for key in ("outputText", "completion", "response"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def get_bedrock_client(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


def generate_answer(
    client,
    model_id: str,
    question: str,
    contexts: Iterable[str],
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    context_list = [c.strip() for c in contexts if isinstance(c, str) and c.strip()]
    has_context = len(context_list) > 0
    context_block = "\n\n".join(context_list)
    
    # Llama3 uses text completion format
    if "llama" in model_id.lower():
        if has_context:
            prompt = (
                "You are a concise assistant. Use ONLY the provided context snippets to answer. "
                "If the context is empty or insufficient, reply: 'I do not have enough information.' "
                "Respond with a short, direct answer only.\n\n"
                f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
            )
        else:
            prompt = (
                "You are a concise and helpful assistant. "
                "Respond with a short, direct answer.\n\n"
                f"Question: {question}\n\nAnswer:"
            )
        body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": ["\n\n", "\"\"\""],
        }
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        payload = json.loads(response["body"].read())
        return payload.get("generation", "").strip()
    
    # Claude and other chat models use messages format
    else:
        if has_context:
            system_prompt = (
                "You are a concise assistant. Use ONLY the provided context snippets to answer. "
                "If the context is empty or insufficient, reply: 'I do not have enough information.' "
                "Respond with a short, direct answer only."
            )
            user_text = f"Context:\n{context_block}\n\nQuestion: {question}"
        else:
            system_prompt = "You are a concise and helpful assistant. Respond with a short, direct answer."
            user_text = question

        body = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_text,
                        }
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        payload = json.loads(response["body"].read())

        text = _extract_text_from_payload(payload)
        if text:
            return text

        return "I could not parse a text response from the model output."
