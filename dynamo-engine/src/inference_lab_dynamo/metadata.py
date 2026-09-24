"""Fetch the metadata (config, tokenizer, chat template; never weights) of
every model a worker's arguments name into the Hugging Face cache, so the
unchanged worker finds them offline where production finds full snapshots."""

import os

from inference_lab_dynamo.records import emit

WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.npz",
                   "original/*", "metal/*"]
REPO_FLAGS = {
    "--model": "--revision",
    "--model-path": "--revision",
    "--tokenizer": "--tokenizer-revision",
    "--tokenizer-path": "--revision",
    "--speculative-draft-model-path": "--speculative-draft-model-revision",
}


def argument(args, flag):
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def repositories(args):
    """(repo_id, revision) pairs named by a worker's arguments."""
    found = []
    for flag, revision_flag in REPO_FLAGS.items():
        repo = argument(args, flag)
        if repo and not os.path.isabs(repo) and "/" in repo:
            revision = argument(args, revision_flag) or argument(args, "--revision")
            if (repo, revision) not in found:
                found.append((repo, revision))
    return found


def fetch(args):
    import huggingface_hub
    from huggingface_hub import constants, snapshot_download

    offline = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = False
    try:
        for repo, revision in repositories(args):
            path = snapshot_download(repo, revision=revision, ignore_patterns=WEIGHT_PATTERNS)
            emit("metadata_fetched", repo=repo, revision=revision, path=path,
                 hub_version=huggingface_hub.__version__)
    finally:
        constants.HF_HUB_OFFLINE = offline
