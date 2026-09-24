"""The simulated accelerator: engines see the production GPU's identity
(name, compute capability, memory) so their own default resolution picks the
same values it picks in production, while no device is ever touched.

Engines' CUDA modules link against the driver library, so the process must
start with a loadable ``libcuda.so.1``. ``ensure_driver_stub()`` points the
loader at the CUDA toolkit's link stub and re-executes the process once."""

import os
import sys
from dataclasses import dataclass

STUB_DIR = "/tmp/inference-lab-dynamo/libcuda"
STUB_CANDIDATES = (
    "/usr/local/cuda/lib64/stubs/libcuda.so",
    "/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so",
    "/usr/local/cuda/targets/sbsa-linux/lib/stubs/libcuda.so",
)


@dataclass(frozen=True)
class GpuSpec:
    name: str
    capability: tuple
    memory_bytes: int
    count: int


# Keyed by the node label production workers select on (nvidia.com/gpu.product).
# Memory is nominal HBM capacity; UNVERIFIED against the totals production
# engines report at start-up.
GPU_PRODUCTS = {
    "NVIDIA-B200": ("NVIDIA B200", (10, 0), 179.0),
    "NVIDIA-B300-SXM6-AC": ("NVIDIA B300 SXM6 AC", (10, 3), 268.0),
    "NVIDIA-B300-SXM6-PC": ("NVIDIA B300 SXM6 PC", (10, 3), 268.0),
    "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition": (
        "NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0), 95.0),
    "GH200": ("NVIDIA GH200 120GB", (9, 0), 95.0),
    "H200": ("NVIDIA H200", (9, 0), 140.0),
}
# Presented when a worker names no product, or one not listed above.
DEFAULT_PRODUCT = "NVIDIA-B200"
# A MIG slice presents its own memory size.
GPU_RESOURCES = {"nvidia.com/mig-2g.48gb": 47.5}


def gpu_spec() -> GpuSpec:
    """The production GPU this worker is placed on, from SIM_GPU_PRODUCT
    (the variant's gpu.product selector), SIM_GPU_RESOURCE and SIM_GPU_COUNT;
    SIM_GPU_NAME / SIM_GPU_CAPABILITY / SIM_GPU_MEMORY_GIB override."""
    product = os.environ.get("SIM_GPU_PRODUCT", "")
    if product not in GPU_PRODUCTS:
        from inference_lab_dynamo.records import emit

        emit("gpu_product_unknown", product=product, using=DEFAULT_PRODUCT)
        product = DEFAULT_PRODUCT
    name, capability, memory_gib = GPU_PRODUCTS[product]
    memory_gib = GPU_RESOURCES.get(os.environ.get("SIM_GPU_RESOURCE", ""), memory_gib)
    if "SIM_GPU_CAPABILITY" in os.environ:
        major, minor = os.environ["SIM_GPU_CAPABILITY"].split(".")
        capability = (int(major), int(minor))
    return GpuSpec(
        name=os.environ.get("SIM_GPU_NAME", name),
        capability=capability,
        memory_bytes=int(float(os.environ.get("SIM_GPU_MEMORY_GIB", memory_gib)) * 2**30),
        count=int(os.environ.get("SIM_GPU_COUNT", "1")),
    )


def ensure_driver_stub(argv=None):
    """Re-execute this process (``argv``, default ``sys.argv``) once with the
    CUDA driver stub on the loader path."""
    if os.environ.get("INFERENCE_LAB_DRIVER_STUB") == "1":
        return
    stub = next((path for path in STUB_CANDIDATES if os.path.exists(path)), None)
    if stub is None:
        raise SystemExit("no CUDA driver stub found in this image")
    os.makedirs(STUB_DIR, exist_ok=True)
    link = os.path.join(STUB_DIR, "libcuda.so.1")
    if not os.path.lexists(link):
        os.symlink(stub, link)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [STUB_DIR, env.get("LD_LIBRARY_PATH")]))
    env["INFERENCE_LAB_DRIVER_STUB"] = "1"
    argv = list(argv) if argv else [sys.executable] + sys.argv
    os.execve(sys.executable, [sys.executable] + argv[1:], env)


def pin_vllm_platform():
    import vllm.platforms
    from vllm.platforms.cuda import NonNvmlCudaPlatform
    from vllm.platforms.interface import DeviceCapability

    spec = gpu_spec()

    class SimCudaPlatform(NonNvmlCudaPlatform):
        @classmethod
        def get_device_capability(cls, device_id=0):
            return DeviceCapability(*spec.capability)

        @classmethod
        def get_device_name(cls, device_id=0):
            return spec.name

        @classmethod
        def get_device_total_memory(cls, device_id=0):
            return spec.memory_bytes

        @classmethod
        def is_fully_connected(cls, physical_device_ids):
            return True

        @classmethod
        def get_device_numa_node(cls, device_id=0):
            return 0

        @classmethod
        def device_count(cls):
            return spec.count

    vllm.platforms.current_platform = SimCudaPlatform()
