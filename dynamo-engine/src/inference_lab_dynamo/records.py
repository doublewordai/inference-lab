"""One JSON object per line on stdout for each engine event. Every record
carries ``"record": "inference_lab"``, so a log query selects them with
``|= "\"record\":\"inference_lab\""`` and parses them with ``| json``."""

import json
import sys
import time


def emit(event, **fields):
    record = {"record": "inference_lab", "event": event, "ts": time.time(), **fields}
    sys.stdout.write(json.dumps(record, default=repr, separators=(",", ":")) + "\n")
    sys.stdout.flush()
