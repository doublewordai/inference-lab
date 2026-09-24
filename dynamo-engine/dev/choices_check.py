"""Two-choice and empty-scripted requests against each model: every choice
must come back finished. Usage: choices_check.py <port> <model>..."""

import sys

import requests

port, models = int(sys.argv[1]), sys.argv[2:]
for model in models:
    for label, body in (
        ("n=2", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "n": 2}),
        ("empty script", {"messages": [{"role": "user", "content": '<<respond:{"text": ""}>>'}], "max_tokens": 5}),
    ):
        response = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json={"model": model, **body}, timeout=60)
        data = response.json()
        choices = [(c["index"], c["finish_reason"]) for c in data.get("choices", [])]
        print(f"{model:32s} {label:13s} HTTP {response.status_code} choices={choices} "
              f"usage={data.get('usage', {}).get('completion_tokens')} {data.get('message', '')[:120]}")
