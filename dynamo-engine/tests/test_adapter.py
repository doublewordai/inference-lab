"""Checks that need no worker image: output planning, argument scanning,
worker detection, and the adapter zip. Run: python3 -m unittest discover
dynamo-engine/tests"""

import importlib
import importlib.util
import pathlib
import sys
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference_lab_dynamo import metadata, simulation  # noqa: E402


class FakeTokenizer:
    vocab_size = 50000

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


class PlanTest(unittest.TestCase):
    def test_junk_tokens_fill_the_limit_and_end_at_length(self):
        plan = simulation.plan(FakeTokenizer(), "<|user|>hello", 7)
        self.assertEqual(len(plan.token_ids), 7)
        self.assertFalse(plan.scripted)

    def test_missing_limit_uses_the_default(self):
        plan = simulation.plan(FakeTokenizer(), "hello", None)
        self.assertEqual(len(plan.token_ids), simulation.DEFAULT_MAX_TOKENS)

    def test_directive_scripts_the_exact_text(self):
        prompt = '<|user|><<respond:{"text": "<tool_call>f</tool_call>"}>><|assistant|>'
        plan = simulation.plan(FakeTokenizer(), prompt, 3)
        self.assertTrue(plan.scripted)
        self.assertEqual(plan.token_ids, [ord(ch) for ch in "<tool_call>f</tool_call>"])

    def test_malformed_directive_falls_back_to_junk(self):
        plan = simulation.plan(FakeTokenizer(), "<<respond:{not json}>>", 4)
        self.assertFalse(plan.scripted)
        self.assertEqual(len(plan.token_ids), 4)


class RepositoriesTest(unittest.TestCase):
    def test_sglang_arguments(self):
        args = ["--model", "nvidia/GLM-5.2-NVFP4", "--revision", "abc", "--tp", "4",
                "--speculative-draft-model-path", "org/draft", "--speculative-draft-model-revision", "def"]
        self.assertEqual(metadata.repositories(args),
                         [("nvidia/GLM-5.2-NVFP4", "abc"), ("org/draft", "def")])

    def test_vllm_tokenizer_and_equals_form(self):
        args = ["--model=openai/gpt-oss-20b", "--revision=r1", "--tokenizer", "org/tok"]
        self.assertEqual(metadata.repositories(args),
                         [("openai/gpt-oss-20b", "r1"), ("org/tok", "r1")])

    def test_local_paths_are_not_fetched(self):
        self.assertEqual(metadata.repositories(["--model-path", "/models/local"]), [])


class SitecustomizeTest(unittest.TestCase):
    def test_worker_module_detection(self):
        spec = importlib.util.spec_from_file_location("adapter_sitecustomize", ROOT / "src/sitecustomize.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module._worker_module(["python3", "-m", "dynamo.sglang", "--tp", "4"]), "dynamo.sglang")
        self.assertIsNone(module._worker_module(["python3", "-S", "/opt/launch.py"]))


class ZipTest(unittest.TestCase):
    def test_zip_carries_sitecustomize_at_its_root(self):
        spec = importlib.util.spec_from_file_location("build_zip", ROOT / "build-zip.py")
        build_zip = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_zip)
        with tempfile.TemporaryDirectory() as directory:
            path = build_zip.build(pathlib.Path(directory) / "adapter.zip")
            names = zipfile.ZipFile(path).namelist()
        self.assertIn("sitecustomize.py", names)
        self.assertIn("inference_lab_dynamo/sglang_engine.py", names)
        self.assertIn("inference_lab_dynamo/vllm_engine.py", names)


if __name__ == "__main__":
    unittest.main()
