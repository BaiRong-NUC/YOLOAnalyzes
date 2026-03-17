from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path


def check_path(path: Path, kind: str = "any") -> tuple[bool, str]:
    if kind == "file":
        ok = path.is_file()
    elif kind == "dir":
        ok = path.is_dir()
    else:
        ok = path.exists()
    return ok, f"{path} => {'OK' if ok else 'MISSING'}"


def read_data_yaml_preview(path: Path) -> list[str]:
    if not path.is_file():
        return ["data.yaml not found"]
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    keys = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("train:") or stripped.startswith("val:") or stripped.startswith("test:"):
            keys.append(stripped)
    return keys if keys else ["train/val/test entries not found"]


def install_package(*packages: str) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    try:
        subprocess.run(cmd, check=True)
        return True, "installed"
    except Exception as exc:
        return False, str(exc)


def detect_nvidia_gpu() -> tuple[bool, str]:
    cmd = ["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return False, "nvidia-smi not found"
    except Exception as exc:
        return False, str(exc)

    line = ""
    for row in result.stdout.splitlines():
        if row.strip():
            line = row.strip()
            break

    if not line:
        return False, "nvidia-smi returned no GPU rows"

    parts = [p.strip() for p in line.split(",")]
    name = parts[0] if len(parts) >= 1 else "unknown"
    driver = parts[1] if len(parts) >= 2 else "unknown"
    cuda = parts[2] if len(parts) >= 3 else "unknown"
    return True, f"name={name}, driver={driver}, cuda={cuda}"


def probe_torch_cuda_with_subprocess() -> tuple[bool, str]:
    code = (
        "import json, torch;"
        "d={'version':torch.__version__,'cuda':torch.cuda.is_available(),"
        "'count':torch.cuda.device_count() if torch.cuda.is_available() else 0,"
        "'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''};"
        "print(json.dumps(d, ensure_ascii=True))"
    )
    cmd = [sys.executable, "-c", code]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        payload = result.stdout.strip().splitlines()[-1]
        info = json.loads(payload)
        if info.get("cuda"):
            return (
                True,
                f"version={info.get('version')}, cuda=True, count={info.get('count')}, name={info.get('name')}",
            )
        return False, f"version={info.get('version')}, cuda=False"
    except Exception as exc:
        return False, str(exc)


def ensure_torch_for_gpu(nvidia_detected: bool) -> tuple[bool, str]:
    torch_ok, torch_msg = ensure_import("torch", "torch")
    if not torch_ok:
        return False, torch_msg

    import torch

    if torch.cuda.is_available():
        return True, f"{torch.__version__}, cuda=True"

    if not nvidia_detected:
        return True, f"{torch.__version__}, cuda=False (NVIDIA GPU not detected)"

    # Try common CUDA wheels in descending priority.
    cuda_indexes = [
        "https://download.pytorch.org/whl/cu128",
        "https://download.pytorch.org/whl/cu126",
        "https://download.pytorch.org/whl/cu124",
    ]

    for index_url in cuda_indexes:
        print(f"Attempting CUDA PyTorch install from: {index_url}")
        ok, msg = install_package(
            "--upgrade",
            "--force-reinstall",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            index_url,
        )
        if not ok:
            print(f"CUDA install attempt failed: {msg}")
            continue

        cuda_ok, cuda_msg = probe_torch_cuda_with_subprocess()
        if cuda_ok:
            return True, f"CUDA build installed ({cuda_msg})"
        print(f"CUDA probe after install not ready: {cuda_msg}")

    return False, "NVIDIA GPU detected, but failed to install a CUDA-enabled PyTorch build"


def ensure_import(module_name: str, package_name: str) -> tuple[bool, str]:
    try:
        __import__(module_name)
        return True, "already available"
    except Exception as exc:
        print(f"{module_name} => FAIL ({exc})")
        print(f"Attempting to install missing package: {package_name}")
        ok, msg = install_package(package_name)
        if not ok:
            return False, f"install failed: {msg}"

        try:
            __import__(module_name)
            return True, "installed and imported"
        except Exception as exc2:
            return False, f"install done, import still failed: {exc2}"


def main() -> int:
    script_path = Path(__file__).resolve()
    workspace = script_path.parent.parent

    yolo_src = workspace / "YOLO 8.3.163"
    model_file = workspace / "Model" / "YOLOv8" / "yolo8n" / "yolov8n.pt"
    data_yaml = workspace / "Data" / "Vehicles" / "data.yaml"

    checks = [
        (workspace, "dir"),
        (workspace / "Scripts", "dir"),
        (yolo_src, "dir"),
        (yolo_src / "ultralytics", "dir"),
        (model_file, "file"),
        (data_yaml, "file"),
        (workspace / "Data" / "Vehicles" / "train" / "images", "dir"),
        (workspace / "Data" / "Vehicles" / "valid" / "images", "dir"),
        (workspace / "Data" / "Vehicles" / "test" / "images", "dir"),
    ]

    print("=== Python Runtime ===")
    print(f"python_executable = {sys.executable}")
    print(f"python_version    = {sys.version.split()[0]}")
    print(f"platform          = {platform.platform()}")

    print("\n=== Project Paths ===")
    missing_paths = 0
    for path, kind in checks:
        ok, message = check_path(path, kind)
        print(message)
        if not ok:
            missing_paths += 1

    print("\n=== Python Package Checks ===")
    import_failures = 0

    print("\n=== GPU Detection ===")
    nvidia_detected, gpu_msg = detect_nvidia_gpu()
    if nvidia_detected:
        print(f"NVIDIA GPU => DETECTED ({gpu_msg})")
    else:
        print(f"NVIDIA GPU => NOT DETECTED ({gpu_msg})")

    numpy_ok, numpy_msg = ensure_import("numpy", "numpy")
    if numpy_ok:
        print(f"numpy => OK ({numpy_msg})")
    else:
        import_failures += 1
        print(f"numpy => FAIL ({numpy_msg})")

    torch_ok, torch_msg = ensure_torch_for_gpu(nvidia_detected)
    if torch_ok:
        import torch
        print(f"torch => OK ({torch.__version__})")
        cuda_available = torch.cuda.is_available()
        print(f"torch.cuda.is_available = {cuda_available}")
        if cuda_available:
            print(f"torch.cuda.device_count = {torch.cuda.device_count()}")
            print(f"torch.cuda.device_name  = {torch.cuda.get_device_name(0)}")
    else:
        import_failures += 1
        print(f"torch => FAIL ({torch_msg})")

    try:
        # Prefer local source tree for this project check.
        if str(yolo_src) not in sys.path:
            sys.path.insert(0, str(yolo_src))
        import ultralytics

        print(f"ultralytics => OK ({ultralytics.__version__})")
        print(f"ultralytics_path = {ultralytics.__file__}")
    except Exception as exc:
        print(f"ultralytics => FAIL ({exc})")
        print("Attempting to install local ultralytics package in editable mode")
        ok, msg = install_package("-e", str(yolo_src))
        if ok:
            try:
                import ultralytics

                print(f"ultralytics => OK ({ultralytics.__version__})")
                print(f"ultralytics_path = {ultralytics.__file__}")
            except Exception as exc2:
                import_failures += 1
                print(f"ultralytics => FAIL (install done, import still failed: {exc2})")
        else:
            import_failures += 1
            print(f"ultralytics => FAIL (install failed: {msg})")

    print("\n=== Data YAML Preview ===")
    for row in read_data_yaml_preview(data_yaml):
        print(row)

    print("\n=== Summary ===")
    print(f"missing_paths   = {missing_paths}")
    print(f"import_failures = {import_failures}")

    if missing_paths == 0 and import_failures == 0:
        print("STATUS: PASS")
        return 0

    print("STATUS: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
