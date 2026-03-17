"""
YOLO8 (Ultralytics) 环境安装脚本
====================================
自动检测 NVIDIA GPU 型号与驱动版本，选择最合适的 PyTorch + CUDA 组合，
并安装 YOLOv8 所有运行依赖。

用法:
    python Scripts/setup_env.py            # 正常安装
    python Scripts/setup_env.py --upgrade  # 升级已有安装
    python Scripts/setup_env.py --verify   # 仅验证当前环境
"""

import subprocess
import sys
import os
import re
import shutil
import platform
import argparse
from pathlib import Path

# ─────────────────────────────────────────────
#  控制台输出工具
# ─────────────────────────────────────────────

def _c(code: str, text: str) -> str:
    """ANSI 颜色（Windows Terminal / PowerShell 支持）"""
    if sys.stdout.isatty() and platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )
        except Exception:
            pass
    return f"\033[{code}m{text}\033[0m"

def header(text: str):
    bar = "═" * 62
    print(f"\n{_c('1;36', bar)}")
    print(f"  {_c('1;37', text)}")
    print(f"{_c('1;36', bar)}")

def step(text: str):
    print(f"\n{_c('1;34', '►')} {text}")

def ok(text: str):
    print(f"  {_c('1;32', '✔')} {text}")

def warn(text: str):
    print(f"  {_c('1;33', '⚠')} {text}")

def err(text: str):
    print(f"  {_c('1;31', '✘')} {text}")

def info(text: str):
    print(f"  {_c('0;37', '·')} {text}")

# ─────────────────────────────────────────────
#  系统检查
# ─────────────────────────────────────────────

def check_python() -> str:
    step("检查 Python 版本")
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if v.major != 3 or not (8 <= v.minor <= 12):
        err(f"Python {ver}  ←  需要 3.8 ~ 3.12")
        if v.minor > 12:
            warn("Python 3.13+ 对部分 C 扩展兼容性尚未完全验证，建议使用 3.11")
        sys.exit(1)
    ok(f"Python {ver}")
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_DEFAULT_ENV")
    if venv:
        ok(f"虚拟环境: {venv}")
    else:
        warn("当前不在虚拟环境中，建议先激活 .venv 再运行本脚本")
    return ver


def check_disk():
    step("检查磁盘空间")
    drive = Path(sys.executable).drive or "/"
    total, used, free = shutil.disk_usage(drive)
    free_gb = free / 1024 ** 3
    if free_gb < 5:
        warn(f"可用空间 {free_gb:.1f} GB（建议 ≥ 5 GB，首次安装 PyTorch CUDA 约需 4~5 GB）")
    else:
        ok(f"可用磁盘空间: {free_gb:.1f} GB")


# ─────────────────────────────────────────────
#  GPU 检测
# ─────────────────────────────────────────────

def _nvidia_smi(*extra_args) -> str | None:
    """调用 nvidia-smi，返回 stdout；失败返回 None"""
    try:
        r = subprocess.run(
            ["nvidia-smi"] + list(extra_args),
            capture_output=True, text=True, timeout=15
        )
        return r.stdout if r.returncode == 0 else None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def detect_gpu() -> list[dict] | None:
    step("检测 NVIDIA GPU")
    raw = _nvidia_smi(
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits"
    )
    if not raw:
        warn("未检测到 NVIDIA GPU 或 nvidia-smi 不可用")
        warn("将安装 CPU 版 PyTorch（训练速度较慢）")
        return None

    gpus = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append({
                "index":   parts[0],
                "name":    parts[1],
                "driver":  parts[2],
                "vram_mb": parts[3],
                "cc":      parts[4],   # compute capability, e.g. "8.9"
            })

    if not gpus:
        warn("nvidia-smi 可用但未返回有效 GPU 信息")
        return None

    for g in gpus:
        vram_gb = int(g["vram_mb"]) / 1024 if g["vram_mb"].isdigit() else 0
        ok(f"GPU {g['index']}: {_c('1;35', g['name'])}")
        info(f"驱动版本   : {g['driver']}")
        info(f"显存       : {vram_gb:.0f} GB")
        info(f"计算能力   : SM {g['cc']}")

    return gpus


# ─────────────────────────────────────────────
#  CUDA / PyTorch 版本选择
# ─────────────────────────────────────────────

# 每条记录: (cuda_label, cuda_major, cuda_minor,
#            min_drv_major, min_drv_minor, pip_index_url)
# 驱动兼容性来源: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
_CUDA_TABLE = [
    ("cu128", 12, 8, 570,  0, "https://download.pytorch.org/whl/cu128"),
    ("cu126", 12, 6, 560, 28, "https://download.pytorch.org/whl/cu126"),
    ("cu124", 12, 4, 551, 61, "https://download.pytorch.org/whl/cu124"),
    ("cu121", 12, 1, 530, 30, "https://download.pytorch.org/whl/cu121"),
    ("cu118", 11, 8, 522,  6, "https://download.pytorch.org/whl/cu118"),
]

# Ada Lovelace (SM 8.9) → RTX 40 系列
_ADA_MODELS = re.compile(r"RTX\s*40\d{2}", re.I)


def _driver_num(drv: str) -> int:
    """把 '566.14' 转为整数 566014 便于比较"""
    try:
        m, n = drv.strip().split(".")[:2]
        return int(m) * 1000 + int(n)
    except Exception:
        return 0


def select_pytorch(gpus: list[dict] | None) -> dict:
    """
    返回安装参数字典:
      {
        "cuda_label": "cu126" | "cpu",
        "cuda_ver":   "12.6"  | None,
        "index_url":  "https://..." | None,
        "packages":   ["torch", "torchvision", "torchaudio"],
      }
    """
    step("选择 PyTorch / CUDA 版本")
    packages = ["torch", "torchvision", "torchaudio"]

    if not gpus:
        warn("无 GPU → 安装 CPU 版 PyTorch")
        return {"cuda_label": "cpu", "cuda_ver": None, "index_url": None, "packages": packages}

    g         = gpus[0]
    driver    = g["driver"]
    drv_n     = _driver_num(driver)
    gpu_name  = g["name"]
    cc        = g["cc"]

    # 计算能力检查（RTX 4070 Ti = 8.9, 需要 CUDA 11.8+）
    try:
        cc_float = float(cc)
    except Exception:
        cc_float = 0.0

    if cc_float < 3.5:
        warn(f"计算能力 {cc} 过低，PyTorch 已不支持该 GPU CUDA 加速")
        return {"cuda_label": "cpu", "cuda_ver": None, "index_url": None, "packages": packages}

    # RTX 40 系列建议提示
    if _ADA_MODELS.search(gpu_name):
        info(f"{gpu_name} 为 Ada Lovelace 架构 (SM 8.9)，建议使用 CUDA 12.x")

    for label, cmaj, cmin, dmin_maj, dmin_min, url in _CUDA_TABLE:
        min_drv = dmin_maj * 1000 + dmin_min
        if drv_n >= min_drv:
            ok(f"选定 CUDA {cmaj}.{cmin}  (PyTorch wheel: {_c('1;33', label)})")
            info(f"驱动版本 {driver} ≥ 最低要求 {dmin_maj}.{dmin_min:02d}")
            return {
                "cuda_label": label,
                "cuda_ver":   f"{cmaj}.{cmin}",
                "index_url":  url,
                "packages":   packages,
            }

    err(f"驱动版本 {driver} 过低，无法支持任何已知 CUDA 版本")
    warn("请将 NVIDIA 驱动升级至 ≥ 522.06 以使用 CUDA 11.8")
    warn("下载地址: https://www.nvidia.com/Download/index.aspx")
    warn("继续安装 CPU 版 PyTorch …")
    return {"cuda_label": "cpu", "cuda_ver": None, "index_url": None, "packages": packages}


# ─────────────────────────────────────────────
#  安装工具
# ─────────────────────────────────────────────

_PIP = [sys.executable, "-m", "pip"]


def _pip(*args, check: bool = True) -> bool:
    """运行 pip 子命令，实时输出，返回是否成功"""
    cmd = _PIP + list(args)
    r = subprocess.run(cmd)
    if check and r.returncode != 0:
        return False
    return r.returncode == 0


def upgrade_pip():
    step("升级 pip / setuptools / wheel")
    if _pip("install", "--upgrade", "pip", "setuptools", "wheel", "--quiet"):
        ok("pip 已是最新")
    else:
        warn("pip 升级失败，继续安装…")


def install_pytorch(pt: dict, upgrade: bool = False) -> bool:
    step(f"安装 PyTorch ({pt['cuda_label'].upper()})")
    args = ["install"] + pt["packages"]
    if upgrade:
        args.append("--upgrade")
    if pt["index_url"]:
        args += ["--index-url", pt["index_url"]]
    if _pip(*args):
        ok("PyTorch 安装完成")
        return True
    err("PyTorch 安装失败")
    return False


def install_ultralytics(upgrade: bool = False) -> bool:
    step("安装 Ultralytics YOLOv8")
    args = ["install", "ultralytics"]
    if upgrade:
        args.append("--upgrade")
    if _pip(*args):
        ok("Ultralytics 安装完成")
        return True
    err("Ultralytics 安装失败")
    return False


def install_extras(upgrade: bool = False):
    step("安装推荐扩展包: 无打印请耐心等待")
    extras = [
        # Notebook 支持
        "jupyter", "ipykernel", "ipywidgets",
        # 可视化
        "matplotlib", "seaborn", "supervision",
        # 实验跟踪
        "tensorboard",
        # 目标跟踪（BotSort / ByteTrack 所需）
        "lapx",
        # 图像增强
        "albumentations",
    ]
    args = ["install"] + extras + ["--quiet"]
    if upgrade:
        args.append("--upgrade")
    if _pip(*args, check=False):
        ok("扩展包安装完成")
    else:
        warn("部分扩展包安装失败，不影响 YOLOv8 核心功能")


# ─────────────────────────────────────────────
#  验证安装
# ─────────────────────────────────────────────

_VERIFY_SCRIPT = r"""
import sys, importlib

results = {}

# ── PyTorch ──
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    results["torch"] = {
        "version": torch.__version__,
        "cuda_available": cuda_ok,
        "cuda_version": torch.version.cuda if cuda_ok else "N/A",
        "gpus": [],
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        results["torch"]["gpus"].append(
            f"  GPU {i}: {p.name}  |  {p.total_memory // 1024**3} GB  |  SM {p.major}.{p.minor}"
        )
except Exception as e:
    results["torch"] = {"error": str(e)}

# ── 其他包 ──
for pkg, imp in [
    ("ultralytics", "ultralytics"),
    ("opencv",      "cv2"),
    ("numpy",       "numpy"),
    ("Pillow",      "PIL"),
    ("matplotlib",  "matplotlib"),
    ("tensorboard", "tensorboard"),
    ("supervision", "supervision"),
]:
    try:
        m = importlib.import_module(imp)
        results[pkg] = getattr(m, "__version__", "ok")
    except ImportError as e:
        results[pkg] = f"MISSING ({e})"

# ── 打印报告 ──
print("\n  ┌─ 环境验证报告 ─────────────────────────────────")
t = results.get("torch", {})
if "error" in t:
    print(f"  │  PyTorch       : ✘ {t['error']}")
else:
    cuda_tag = f"CUDA {t['cuda_version']}" if t['cuda_available'] else "CPU ONLY"
    print(f"  │  PyTorch       : {t['version']}  [{cuda_tag}]")
    for line in t["gpus"]:
        print(f"  │{line}")

for pkg in ["ultralytics", "opencv", "numpy", "Pillow", "matplotlib", "tensorboard", "supervision"]:
    v = results.get(pkg, "?")
    icon = "✘" if "MISSING" in str(v) else "✔"
    print(f"  │  {pkg:<15}: {icon} {v}")
print("  └────────────────────────────────────────────────")
"""


def verify():
    header("验证安装结果")
    subprocess.run([sys.executable, "-c", _VERIFY_SCRIPT])


# ─────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO8 环境安装脚本")
    parser.add_argument("--upgrade", action="store_true", help="升级已有安装")
    parser.add_argument("--verify",  action="store_true", help="仅验证当前环境，不安装任何包")
    args = parser.parse_args()

    header("YOLO8 (Ultralytics) 环境安装脚本")
    info(f"操作系统 : {platform.system()} {platform.version()}")
    info(f"Python   : {sys.executable}")

    if args.verify:
        verify()
        return

    # ── 前置检查 ──────────────────────────────
    check_python()
    check_disk()

    # ── GPU 检测 ──────────────────────────────
    gpus = detect_gpu()

    # ── 版本选择 ──────────────────────────────
    pt = select_pytorch(gpus)

    # ── 安装流程 ──────────────────────────────
    upgrade_pip()

    if not install_pytorch(pt, upgrade=args.upgrade):
        err("PyTorch 安装失败，中止安装")
        sys.exit(1)

    if not install_ultralytics(upgrade=args.upgrade):
        err("Ultralytics 安装失败，中止安装")
        sys.exit(1)

    install_extras(upgrade=args.upgrade)

    # ── 验证 ──────────────────────────────────
    verify()

    header("🎉  安装完成！快速上手示例")
    print("""
from ultralytics import YOLO

  # 加载预训练模型（首次运行会自动下载权重）
  model = YOLO("yolov8n.pt")

  # 推理
  results = model("image.jpg")

  # 训练（使用自定义数据集）
  model.train(data="Data/Vehicles/data.yaml", epochs=100, imgsz=640)
""")


if __name__ == "__main__":
    main()
