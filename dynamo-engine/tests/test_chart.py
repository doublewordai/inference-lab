"""The bundle chart renders one ConfigMap per bound namespace, a policy and a
binding scoped to exactly those namespaces. Needs helm on PATH."""

import json
import pathlib
import shutil
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHART = ROOT / "charts/inference-lab-dynamo"

sys.path.insert(0, str(ROOT / "dynamo-engine"))


def render(*args):
    build = subprocess.run([sys.executable, str(ROOT / "dynamo-engine/build-zip.py")],
                           capture_output=True, text=True, check=True)
    assert build.returncode == 0
    out = subprocess.run(["helm", "template", "sim", str(CHART), "--namespace", "staging-workers", *args],
                         capture_output=True, text=True, check=True).stdout
    import yaml

    return [doc for doc in yaml.safe_load_all(out) if doc]


@unittest.skipIf(shutil.which("helm") is None, "helm not installed")
class ChartTest(unittest.TestCase):
    def test_default_binds_the_release_namespace(self):
        docs = render()
        kinds = sorted(doc["kind"] for doc in docs)
        self.assertEqual(kinds, ["ConfigMap", "MutatingAdmissionPolicy", "MutatingAdmissionPolicyBinding"])
        configmap = next(doc for doc in docs if doc["kind"] == "ConfigMap")
        self.assertEqual(configmap["metadata"]["namespace"], "staging-workers")
        self.assertIn("inference-lab-dynamo.zip", configmap["binaryData"])
        binding = next(doc for doc in docs if doc["kind"] == "MutatingAdmissionPolicyBinding")
        expression = binding["spec"]["matchResources"]["namespaceSelector"]["matchExpressions"][0]
        self.assertEqual(expression["values"], ["staging-workers"])

    def test_resources_and_env_render_into_the_policy(self):
        docs = render("--set", "resources.requests.cpu=250m", "--set", "env.SIM_ITL_S=0.05")
        policy = next(doc for doc in docs if doc["kind"] == "MutatingAdmissionPolicy")
        expression = policy["spec"]["mutations"][0]["jsonPatch"]["expression"]
        self.assertIn('"cpu":"250m"', expression)
        self.assertIn(json.dumps(["SIM_ITL_S", "0.05"], separators=(",", ":")), expression)

    def test_explicit_namespaces_each_get_the_adapter(self):
        docs = render("--set", "namespaces={a,b}")
        namespaces = sorted(doc["metadata"]["namespace"] for doc in docs if doc["kind"] == "ConfigMap")
        self.assertEqual(namespaces, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
