"""
Ask a model for JSON and validate it against a Pydantic schema.

Prompt-and-parse is used instead of tool calling / response_format, because
OpenAI-compatible proxies don't all support those for every model. If the
output is invalid, the error is sent back to the model and it gets one retry.
"""
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the model's reply")
    return json.loads(text[start:end + 1])   # JSONDecodeError is a ValueError


def _text_of(reply) -> str:
    content = reply.content
    if isinstance(content, str):
        return content
    # some providers return a list of content parts
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def invoke_json(llm, system: str, user: str, schema: type[BaseModel], attempts: int = 2) -> BaseModel:
    messages = [SystemMessage(system), HumanMessage(user)]
    last_err = None
    for _ in range(attempts):
        text = _text_of(llm.invoke(messages))
        try:
            return schema.model_validate(extract_json(text))
        except (ValueError, ValidationError) as e:
            last_err = e
            messages += [
                AIMessage(text),
                HumanMessage(f"That reply was invalid:\n{e}\nReply with only the corrected JSON object."),
            ]
    raise RuntimeError(f"Model did not return valid JSON after {attempts} attempts. Last error: {last_err}")
