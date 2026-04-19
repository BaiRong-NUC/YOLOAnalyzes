import os
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
LOCAL_YOLO_SRC = ROOT / "YOLO 8.3.163"
if LOCAL_YOLO_SRC.exists():
    sys.path.insert(0, str(LOCAL_YOLO_SRC))

os.environ.setdefault("YOLO_OFFLINE", "true")

from ultralytics import YOLO

# ── 路径配置（相对于项目根目录）────────────────────────────────
DATA_YAML = ROOT / "Data" / "Vehicles" / "data.yaml"
# 只保留一条生效的 WEIGHTS 路径，其他候选项保持注释，避免被 tuple 占位覆盖。
WEIGHTS = LOCAL_YOLO_SRC / "ultralytics" / "cfg" / "models" / "v8" / "yolov8.yaml"
# WEIGHTS = LOCAL_YOLO_SRC / "ultralytics" / "cfg" / "models" / "v8" / "yolov8_vif.yaml"
# WEIGHTS = ROOT / "Model" / "YOLOv8" / "yolo8n" / "yolov8n.pt"
# WEIGHTS = ROOT / "Scripts" / "Improve" / "Debug" / "mixed_yolov_vif.pt"
# WEIGHTS = LOCAL_YOLO_SRC / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-swin-t.yaml"

# mixed_yolov_vif.pt冻结参数训练,之后整体训练
# WEIGHTS = (
#     ROOT
#     / "Scripts"
#     / "Improve"
#     / "Output-mixed_yolov_vif_freeze"
#     / "vehicles_yolov8n"
#     / "weights"
#     / "best.pt"
# )
OUTPUT_DIR = ROOT / "Scripts" / "Improve" / "Output"

if not isinstance(WEIGHTS, Path):
    raise TypeError(
        f"[错误] WEIGHTS 必须是 pathlib.Path，当前类型为 {type(WEIGHTS).__name__}。"
        "请检查是否误写成了括号表达式或 tuple。"
    )


STRICT_COMPARE_TRAIN_ARGS = {
    "epochs": 30,
    "patience": 20,
    "optimizer": "AdamW",
    "lr0": 1e-3,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 5e-4,
    "warmup_epochs": 3,
    "cos_lr": True,
}

STRICT_COMPARE_AUGMENT_ARGS = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.1,
}


def parse_env_value(name: str, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if isinstance(default, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


# ── GPU 检测 ────────────────────────────────────────────────────
def get_device() -> str | int:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {name}  |  显存 {vram:.0f} GB")
        return 0  # 使用第 0 块 GPU
    print("[警告] 未检测到 CUDA GPU，将使用 CPU 训练（速度较慢）")
    return "cpu"


# ── 严格对比时使用统一的共同可跑 batch size ──────────────────────
def resolve_compare_batch(device) -> int:
    if device == "cpu":
        return 2
    vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    if vram_gb >= 24:
        return 48
    elif vram_gb >= 16:
        return 32
    elif vram_gb >= 8:
        return 12
    return 8


def resolve_model_tag() -> str:
    model_name = WEIGHTS.stem.lower()
    if "swin" in model_name:
        return "swin_t"
    if "vif" in model_name or "transformer" in model_name:
        return "vif"
    return model_name.replace("-", "_")


def resolve_train_config(device) -> dict:
    model_tag = resolve_model_tag()
    is_cpu = device == "cpu"

    defaults = {
        "imgsz": 512,
        "batch": resolve_compare_batch(device),
        "nbs": 2 if is_cpu else 16,
        "workers": 0 if is_cpu else 4,
        "amp": False,
        "epochs": STRICT_COMPARE_TRAIN_ARGS["epochs"],
        "optimizer": STRICT_COMPARE_TRAIN_ARGS["optimizer"],
        "name": f"vehicles_strict_compare_{model_tag}",
    }

    return {
        "imgsz": parse_env_value("YOLO_TRAIN_IMGSZ", defaults["imgsz"]),
        "batch": parse_env_value("YOLO_TRAIN_BATCH", defaults["batch"]),
        "nbs": parse_env_value("YOLO_TRAIN_EFFECTIVE_BATCH", defaults["nbs"]),
        "workers": parse_env_value("YOLO_TRAIN_WORKERS", defaults["workers"]),
        "amp": parse_env_value("YOLO_TRAIN_AMP", defaults["amp"]),
        "epochs": parse_env_value("YOLO_TRAIN_EPOCHS", defaults["epochs"]),
        "optimizer": parse_env_value("YOLO_TRAIN_OPTIMIZER", defaults["optimizer"]),
        "name": parse_env_value("YOLO_TRAIN_NAME", defaults["name"]),
    }


# ── 主训练流程 ──────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  YOLOv8 车辆检测训练")
    print("=" * 60)

    # 文件检查
    if not DATA_YAML.exists():
        print(f"[错误] 数据集配置文件不存在: {DATA_YAML}")
        sys.exit(1)
    if not WEIGHTS.exists():
        print(f"[错误] 模型配置或权重文件不存在: {WEIGHTS}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    train_cfg = resolve_train_config(device)
    print(
        f"[配置] device={device}  imgsz={train_cfg['imgsz']}  batch={train_cfg['batch']}  "
        f"effective_batch={train_cfg['nbs']}  workers={train_cfg['workers']}  amp={train_cfg['amp']}"
    )
    print(
        "[提示] 当前使用严格对比配置：统一 imgsz、有效 batch、amp、epochs、optimizer 和增强策略。"
    )

    # 加载预训练模型
    model = YOLO(str(WEIGHTS))
    if device != "cpu":
        torch.cuda.empty_cache()

    # ── 训练参数 ────────────────────────────────────────────────
    results = model.train(
        data=str(DATA_YAML),  # 数据集 yaml
        epochs=train_cfg["epochs"],  # 训练轮次
        imgsz=train_cfg["imgsz"],  # 输入图像尺寸
        batch=train_cfg["batch"],  # 共同可跑的实际 batch size
        nbs=train_cfg["nbs"],  # Ultralytics 会按 nbs / batch 自动做梯度累积
        device=device,  # GPU/CPU
        workers=train_cfg["workers"],  # 数据加载线程数
        patience=STRICT_COMPARE_TRAIN_ARGS["patience"],  # Early stopping 容忍轮次
        optimizer=train_cfg["optimizer"],  # 优化器
        lr0=STRICT_COMPARE_TRAIN_ARGS["lr0"],  # 初始学习率
        lrf=STRICT_COMPARE_TRAIN_ARGS["lrf"],  # 最终学习率 = lr0 * lrf
        momentum=STRICT_COMPARE_TRAIN_ARGS["momentum"],
        weight_decay=STRICT_COMPARE_TRAIN_ARGS["weight_decay"],
        warmup_epochs=STRICT_COMPARE_TRAIN_ARGS["warmup_epochs"],  # 热身轮次
        cos_lr=STRICT_COMPARE_TRAIN_ARGS["cos_lr"],  # 余弦学习率调度
        # 数据增强
        **STRICT_COMPARE_AUGMENT_ARGS,
        # 输出路径
        amp=train_cfg["amp"],
        project=str(OUTPUT_DIR),
        name=train_cfg["name"],
        exist_ok=True,  # 允许覆盖已有实验
        # 日志
        plots=True,  # 保存训练曲线图
        save=True,
        save_period=10,  # 每 10 轮保存一次 checkpoint
        verbose=True,
        # 参数冻结
        # freeze=list(range(0, 10)),  # 冻结前 10 层（根据模型结构调整）
    )

    # ── 训练结束后在验证集上评估 ────────────────────────────────
    print("\n[评估] 在验证集上测试最佳权重…")
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        best_model = YOLO(str(best_pt))
        if device != "cpu":
            torch.cuda.empty_cache()
        metrics = best_model.val(
            data=str(DATA_YAML),
            device=device,
            imgsz=train_cfg["imgsz"],
            split="val",
            project=str(OUTPUT_DIR),
            name=f"{train_cfg['name']}_eval",
            exist_ok=True,
        )
        print(f"\n[结果] mAP50    : {metrics.box.map50:.4f}")
        print(f"[结果] mAP50-95 : {metrics.box.map:.4f}")
        print(f"[结果] 最佳权重  : {best_pt}")
    else:
        print(f"[警告] 未找到最佳权重文件: {best_pt}")

    print(f"\n[完成] 所有输出已保存至: {OUTPUT_DIR / train_cfg['name']}")


if __name__ == "__main__":
    train()
