"""Build the adapter zip that simulated workers put on PYTHONPATH (it carries
sitecustomize.py at its root). Deterministic: fixed timestamps and order.
Usage: build-zip.py [output path]"""

import pathlib
import sys
import zipfile

SOURCE = pathlib.Path(__file__).parent / "src"
DEFAULT = pathlib.Path(__file__).parent.parent / "charts/inference-lab-dynamo/files/inference-lab-dynamo.zip"


def build(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SOURCE.rglob("*.py")):
            info = zipfile.ZipInfo(str(path.relative_to(SOURCE)), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return output


if __name__ == "__main__":
    print(build(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))
