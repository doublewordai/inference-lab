"""Send representative requests through the frontend and print what the
simulated engine recorded for each. Usage: probe.py <port> <model> <worker service>"""

import json
import os
import pathlib
import subprocess
import sys
import time

import requests

PORT, MODEL, SERVICE = int(sys.argv[1]), sys.argv[2], sys.argv[3]
BASE = f"http://127.0.0.1:{PORT}/v1"
MARKER = "\"record\":\"inference_lab\""
COMPOSE = str(pathlib.Path(__file__).with_name("compose.yaml"))
ENV_FILE = ["--env-file", os.environ["HARNESS_ENV_FILE"]] if "HARNESS_ENV_FILE" in os.environ else []


def wait_ready(timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            names = [m["id"] for m in requests.get(f"{BASE}/models", timeout=3).json()["data"]]
            if MODEL in names:
                return names
        except requests.RequestException:
            pass
        time.sleep(3)
    raise SystemExit(f"{MODEL} not registered after {timeout}s")


def records():
    out = subprocess.run(
        ["docker", "compose", *ENV_FILE, "-f", COMPOSE, "logs", "--no-log-prefix", SERVICE],
        capture_output=True, text=True, check=True,
    ).stdout
    return [json.loads(line[line.index("{"):]) for line in out.splitlines() if MARKER in line]


def chat(name, stream=False, **extra):
    before = len(records())
    body = {"model": MODEL, "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "max_tokens": 24, "stream": stream, **extra}
    response = requests.post(f"{BASE}/chat/completions", json=body, stream=stream, timeout=120)
    if stream:
        text = "".join(line.decode() + "\n" for line in response.iter_lines() if line)
    else:
        text = response.text
    time.sleep(0.5)
    new = records()[before:]
    print(f"\n=== {name}: HTTP {response.status_code}")
    print("client:", text[:600].replace("\n", " | "))
    for record in new:
        if record["event"] == "engine_request":
            call = record["call"]
            shown = {k: v for k, v in call.items() if v not in (None, False, {}, []) and k != "input_ids"}
            print("engine_request:", json.dumps(shown)[:1500])
            print("prompt_text:", repr(record["prompt_text"])[:800])
        else:
            print(record["event"], {k: v for k, v in record.items() if k not in ("event", "ts")})


print("models:", wait_ready())
if len(sys.argv) > 4 and sys.argv[4] == "extra":
    chat("no max_tokens", max_tokens=None)
    chat("nvext priority", nvext={"agent_hints": {"priority": 7}})
    raise SystemExit
chat("plain")
chat("stream", stream=True, stream_options={"include_usage": True})
chat("reasoning off", chat_template_kwargs={"enable_thinking": False})
chat("tools", tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}}}}])
chat("json_schema", response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})
chat("priority + no max_tokens", priority=1234, max_tokens=None)
chat("directive", messages=[{"role": "user", "content":
     '<<respond:{"text": "<think>brief</think>Paris."}>>'}])
