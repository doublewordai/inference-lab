"""What a simulated engine generates for one request: the tokens and when.

Output is junk tokens up to the request's token limit, unless the prompt
carries an inference-lab directive, ``<<respond:{"text": ...}>>``, in which
case the engine emits exactly that text (in the model's own format, e.g.
reasoning or tool-call markup) and stops. The directive contract is the one
``inference-lab serve`` follows: the last well-formed directive in the prompt
wins, so a chained agent loop advances on the directive in its newest tool
result."""

import json
import os
import random

MARKER = "<<respond:"
DEFAULT_MAX_TOKENS = int(os.environ.get("SIM_DEFAULT_MAX_TOKENS", "16"))
INTER_TOKEN_SECONDS = float(os.environ.get("SIM_ITL_S", "0.02"))


class Plan:
    """The token ids to emit, and whether they end in a stop (scripted) or at
    the length limit (junk)."""

    def __init__(self, token_ids, scripted):
        self.token_ids = token_ids
        self.scripted = scripted


# The worker's --dyn-tool-call-parser: which markup the frontend will parse
# tool calls out of. Set by the bootstrap from the worker's arguments.
TOOL_CALL_PARSER = None


def _value_text(value):
    return value if isinstance(value, str) else json.dumps(value)


def _hermes(call):
    return "<tool_call>\n" + json.dumps({"name": call["name"], "arguments": call.get("arguments", {})}) + "\n</tool_call>"


def _qwen3_coder(call):
    params = "".join(f"<parameter={key}>\n{_value_text(value)}\n</parameter>\n"
                     for key, value in call.get("arguments", {}).items())
    return f"<tool_call>\n<function={call['name']}>\n{params}</function>\n</tool_call>"


def _glm47(call):
    params = "".join(f"<arg_key>{key}</arg_key><arg_value>{_value_text(value)}</arg_value>"
                     for key, value in call.get("arguments", {}).items())
    return f"<tool_call>{call['name']}{params}</tool_call>"


TOOL_CALL_FORMATS = {
    "hermes": _hermes,
    "qwen25": _hermes,
    "qwen3_coder": _qwen3_coder,
    "glm47": _glm47,
}


def render_tool_calls(tool_calls):
    """A directive's tool calls in the model's own markup, or None when this
    worker's parser has no format here."""
    formatter = TOOL_CALL_FORMATS.get(TOOL_CALL_PARSER or "")
    if formatter is None:
        return None
    return "\n".join(formatter(call) for call in tool_calls)


THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def directive_text(payload, prompt_text=""):
    """The raw model output a directive asks for: its optional reasoning, its
    text, then its tool calls rendered for this worker's parser. When the chat
    template has already opened a thinking block (the prompt ends in
    ``<think>``), the output closes it first, as the real model would."""
    text = payload.get("text") or ""
    tool_calls = payload.get("tool_calls") or []
    if tool_calls:
        rendered = render_tool_calls(tool_calls)
        if rendered is None:
            from inference_lab_dynamo.records import emit

            emit("directive_tool_calls_unsupported", parser=TOOL_CALL_PARSER)
        else:
            text = f"{text}\n{rendered}" if text else rendered
    reasoning = payload.get("reasoning") or ""
    if THINK_CLOSE not in text:
        if (prompt_text or "").rstrip().endswith(THINK_OPEN):
            text = f"{reasoning}{THINK_CLOSE}{text}"
        elif reasoning:
            text = f"{THINK_OPEN}{reasoning}{THINK_CLOSE}{text}"
    return text


def _well_formed(payload):
    """The shape ``inference-lab serve`` deserializes a directive into."""
    if not isinstance(payload, dict):
        return False
    if any(not isinstance(payload.get(key), (str, type(None))) for key in ("text", "reasoning", "finish_reason")):
        return False
    calls = payload.get("tool_calls")
    if calls is None:
        return True
    return isinstance(calls, list) and all(
        isinstance(call, dict) and isinstance(call.get("name"), str)
        and isinstance(call.get("arguments", {}), dict)
        for call in calls)


def find_directive(prompt_text):
    """The last well-formed directive in the prompt, or None. Exactly one JSON
    value is read after each marker, so ``>>`` inside its strings is
    harmless."""
    decoder = json.JSONDecoder()
    found = None
    position = prompt_text.find(MARKER)
    while position != -1:
        start = position + len(MARKER)
        try:
            payload, _ = decoder.raw_decode(prompt_text, start)
        except ValueError:
            payload = None
        if _well_formed(payload):
            found = payload
        position = prompt_text.find(MARKER, start)
    return found


def plan(tokenizer, prompt_text, max_tokens):
    payload = find_directive(prompt_text or "")
    if payload is not None:
        text = directive_text(payload, prompt_text)
        return Plan(tokenizer.encode(text, add_special_tokens=False), scripted=True)
    count = max_tokens or DEFAULT_MAX_TOKENS
    upper = max(1001, min(tokenizer.vocab_size - 1, 20000))
    return Plan([random.randint(1000, upper) for _ in range(count)], scripted=False)


def plans(tokenizer, prompt_text, max_tokens, choices):
    """One plan per requested choice (``n``)."""
    return [plan(tokenizer, prompt_text, max_tokens) for _ in range(max(1, choices or 1))]


def steps(choice_plans):
    """Per generation step, the (choice index, new token ids, finished) triples
    to emit. Every choice ends with exactly one finished triple, including a
    choice whose plan is empty."""
    longest = max((len(p.token_ids) for p in choice_plans), default=0)
    for position in range(max(1, longest)):
        emitted = []
        for index, choice in enumerate(choice_plans):
            length = len(choice.token_ids)
            if position < length:
                emitted.append((index, [choice.token_ids[position]], position == length - 1))
            elif position == 0:
                emitted.append((index, [], True))
        yield emitted
