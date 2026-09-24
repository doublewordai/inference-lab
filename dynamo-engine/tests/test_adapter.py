"""Checks that need no worker image: output planning, argument scanning,
worker detection, and the adapter zip. Run: python3 -m unittest discover
dynamo-engine/tests"""

import importlib
import importlib.util
import pathlib
import sys
import tempfile
import unittest
import unittest.mock
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

    def test_the_last_directive_wins(self):
        prompt = ('<|user|>task <<respond:{"text": "first"}>><|tool|>'
                  'result <<respond:{"text": "second >> still"}>> <<respond:{broken}>>')
        plan = simulation.plan(FakeTokenizer(), prompt, 3)
        self.assertEqual(plan.token_ids, [ord(ch) for ch in "second >> still"])

    def test_malformed_directive_falls_back_to_junk(self):
        plan = simulation.plan(FakeTokenizer(), "<<respond:{not json}>>", 4)
        self.assertFalse(plan.scripted)
        self.assertEqual(len(plan.token_ids), 4)


class DirectiveTextTest(unittest.TestCase):
    CALL = {"name": "Read", "arguments": {"file_path": "/tmp/a", "limit": 4}}

    def render(self, parser, payload, prompt=""):
        with unittest.mock.patch.object(simulation, "TOOL_CALL_PARSER", parser):
            return simulation.directive_text(payload, prompt)

    def test_hermes_tool_calls_are_json_in_tool_call_tags(self):
        text = self.render("hermes", {"tool_calls": [self.CALL]})
        self.assertEqual('<tool_call>\n{"name": "Read", "arguments": {"file_path": "/tmp/a", "limit": 4}}\n</tool_call>', text)

    def test_glm47_tool_calls_use_arg_pairs(self):
        text = self.render("glm47", {"tool_calls": [self.CALL]})
        self.assertEqual("<tool_call>Read<arg_key>file_path</arg_key><arg_value>/tmp/a</arg_value>"
                         "<arg_key>limit</arg_key><arg_value>4</arg_value></tool_call>", text)

    def test_qwen3_coder_tool_calls_use_function_tags(self):
        text = self.render("qwen3_coder", {"tool_calls": [self.CALL]})
        self.assertIn("<function=Read>\n<parameter=file_path>\n/tmp/a\n</parameter>", text)

    def test_an_open_thinking_block_is_closed_before_the_answer(self):
        text = self.render("glm47", {"reasoning": "why", "text": "Paris."}, "<|assistant|><think>")
        self.assertEqual("why</think>Paris.", text)

    def test_reasoning_without_an_open_block_gets_a_whole_one(self):
        self.assertEqual("<think>why</think>Paris.", self.render("hermes", {"reasoning": "why", "text": "Paris."}))

    def test_unsupported_parser_keeps_the_text(self):
        self.assertEqual("hi", self.render("harmony", {"text": "hi", "tool_calls": [self.CALL]}))


class StepsTest(unittest.TestCase):
    def test_every_choice_streams_to_its_own_finish(self):
        choices = [simulation.Plan([1, 2, 3], False), simulation.Plan([4], True)]
        steps = list(simulation.steps(choices))
        self.assertEqual(steps[0], [(0, [1], False), (1, [4], True)])
        self.assertEqual(steps[-1], [(0, [3], True)])
        finished = [index for step in steps for index, _, done in step if done]
        self.assertEqual(sorted(finished), [0, 1])

    def test_empty_scripted_output_still_finishes(self):
        steps = list(simulation.steps([simulation.Plan([], True)]))
        self.assertEqual(steps, [[(0, [], True)]])

    def test_n_plans_one_sequence_per_choice(self):
        self.assertEqual(len(simulation.plans(FakeTokenizer(), "hi", 2, 3)), 3)
        self.assertEqual(len(simulation.plans(FakeTokenizer(), "hi", 2, None)), 1)


class GpuSpecTest(unittest.TestCase):
    def test_unknown_or_missing_product_falls_back(self):
        from inference_lab_dynamo import accelerator

        for product in ("", "MI355X"):
            with unittest.mock.patch.dict("os.environ", {"SIM_GPU_PRODUCT": product}, clear=False):
                spec = accelerator.gpu_spec()
            self.assertEqual(spec.name, accelerator.GPU_PRODUCTS[accelerator.DEFAULT_PRODUCT][0])

    def test_mig_slice_presents_its_own_memory(self):
        from inference_lab_dynamo import accelerator

        env = {"SIM_GPU_PRODUCT": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
               "SIM_GPU_RESOURCE": "nvidia.com/mig-2g.48gb", "SIM_GPU_COUNT": "1"}
        with unittest.mock.patch.dict("os.environ", env, clear=False):
            spec = accelerator.gpu_spec()
        self.assertEqual(spec.memory_bytes, int(47.5 * 2**30))
        self.assertEqual(spec.capability, (12, 0))


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
