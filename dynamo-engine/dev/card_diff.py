"""Diff a model card registered in the local etcd against a production card.
Usage: card_diff.py <prod cards json> <prod key substring> [local namespace substring]
(HARNESS_ENV_FILE: the env file holding the harness image variables.)"""

import json
import os
import pathlib
import subprocess
import sys

COMPOSE = str(pathlib.Path(__file__).with_name("compose.yaml"))
prod_cards = json.load(open(sys.argv[1]))
prod = next(v for k, v in prod_cards.items() if sys.argv[2] in k)
env_file = ["--env-file", os.environ["HARNESS_ENV_FILE"]] if "HARNESS_ENV_FILE" in os.environ else []
local_raw = subprocess.run(
    ["docker", "compose", *env_file, "-f", COMPOSE, "exec", "-T", "etcd",
     "etcdctl", "get", "--prefix", "v1/mdc/", "-w", "json"],
    capture_output=True, text=True, check=True,
).stdout
import base64

local_cards = {
    base64.b64decode(kv["key"]).decode(): json.loads(base64.b64decode(kv["value"]))
    for kv in json.loads(local_raw).get("kvs", [])
}
wanted = sys.argv[3] if len(sys.argv) > 3 else ""
local = next(v for k, v in local_cards.items() if wanted in k)


def flat(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from flat(value, f"{path}.{key}" if path else key)
    else:
        yield path, node


ignored = {"instance_id", "namespace"}
local_flat, prod_flat = dict(flat(local)), dict(flat(prod))
for key in sorted(set(local_flat) | set(prod_flat)):
    if key.split(".")[0] in ignored:
        continue
    ours, theirs = local_flat.get(key, "<absent>"), prod_flat.get(key, "<absent>")
    if ours != theirs:
        print(f"{key}\n   sim : {json.dumps(ours)[:140]}\n   prod: {json.dumps(theirs)[:140]}")
