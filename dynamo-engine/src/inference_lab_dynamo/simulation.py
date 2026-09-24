"""What a simulated engine generates for one request: the tokens and when.

Output is junk tokens up to the request's token limit, unless the prompt
carries an inference-lab directive, ``<<respond:{"text": ...}>>``, in which
case the engine emits exactly that text (in the model's own format, e.g.
reasoning or tool-call markup) and stops."""

import json
import os
import random
import re

DIRECTIVE = re.compile(r"<<respond:(\{.*?\})>>", re.S)
DEFAULT_MAX_TOKENS = int(os.environ.get("SIM_DEFAULT_MAX_TOKENS", "16"))
INTER_TOKEN_SECONDS = float(os.environ.get("SIM_ITL_S", "0.02"))


class Plan:
    """The token ids to emit, and whether they end in a stop (scripted) or at
    the length limit (junk)."""

    def __init__(self, token_ids, scripted):
        self.token_ids = token_ids
        self.scripted = scripted


def plan(tokenizer, prompt_text, max_tokens):
    directive = DIRECTIVE.search(prompt_text or "")
    if directive:
        try:
            text = json.loads(directive.group(1)).get("text", "")
        except ValueError:
            text = None
        if text is not None:
            return Plan(tokenizer.encode(text, add_special_tokens=False), scripted=True)
    count = max_tokens or DEFAULT_MAX_TOKENS
    upper = max(1001, min(tokenizer.vocab_size - 1, 20000))
    return Plan([random.randint(1000, upper) for _ in range(count)], scripted=False)
