# -*- coding: utf-8 -*-
"""
车辆检测 YOLOv8 训练脚本
数据集 : Data/Vehicles   (12 类车辆)
预训练权重 : Model/YOLOv8/yolo8n/yolov8n.pt
输出目录   : Output/
"""

import sys
from pathlib import Path

import torch

from ultralytics import YOLO

# ── 路径配置（相对于项目根目录）────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]          # d:\Python\YOLOAnalyzes
DATA_YAML  = ROOT / "Data" / "Vehicles" / "data.yaml"
WEIGHTS    = ROOT / "Model" / "YOLOv8" / "yolo8n" / "yolov8n.pt"
OUTPUT_DIR = ROOT / "Output"

# ── GPU 检测 ────────────────────────────────────────────────────
def get_device() -> str | int:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[GPU] {name}  |  显存 {vram:.0f} GB")
        return 0          # 使用第 0 块 GPU
    print("[警告] 未检测到 CUDA GPU，将使用 CPU 训练（速度较慢）")
    return "cpu"

# ── 根据显存自动调整 batch size ─────────────────────────────────
def auto_batch(device) -> int:
    if device == "cpu":
        return 4
    vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    # RTX 4070 Ti = 12 GB；640px 图像下经验值
    if vram_gb >= 24:
        return 32
    elif vram_gb >= 11:   # 4070 Ti = 12282 MiB ≈ 11.99 GB
        return 16
    elif vram_gb >= 8:
        return 8
    return 4

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
        print(f"[错误] 预训练权重不存在: {WEIGHTS}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device     = get_device()
    batch_size = auto_batch(device)
    print(f"[配置] batch={batch_size}  device={device}")

    # 加载预训练模型
    model = YOLO(str(WEIGHTS))

    # ── 训练参数 ────────────────────────────────────────────────
    results = model.train(
        data      = str(DATA_YAML),   # 数据集 yaml
        epochs    = 100,               # 训练轮次
        imgsz     = 640,               # 输入图像尺寸
        batch     = batch_size,        # 批大小（自动适配显存）
        device    = device,            # GPU/CPU
        workers   = 8,                 # 数据加载线程数
        patience  = 20,                # Early stopping 容忍轮次
        optimizer = "AdamW",           # 优化器
        lr0       = 1e-3,              # 初始学习率
        lrf       = 0.01,              # 最终学习率 = lr0 * lrf
        momentum  = 0.937,
        weight_decay = 5e-4,
        warmup_epochs = 3,             # 热身轮次
        cos_lr    = True,              # 余弦学习率调度
        # 数据增强
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        flipud    = 0.0,
        fliplr    = 0.5,
        mosaic    = 1.0,
        mixup     = 0.1,
        # 输出路径
        amp       = False,      # true训练器在开始前会下载一个小模型(硬编码为 yolo11n.pt)            
        project   = str(OUTPUT_DIR),
        name      = "vehicles_yolov8n",
        exist_ok  = True,              # 允许覆盖已有实验
        # 日志
        plots     = True,              # 保存训练曲线图
        save      = True,
        save_period = 10,              # 每 10 轮保存一次 checkpoint
        verbose   = True,
    )

    # ── 训练结束后在验证集上评估 ────────────────────────────────
    print("\n[评估] 在验证集上测试最佳权重…")
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        best_model = YOLO(str(best_pt))
        metrics = best_model.val(
            data    = str(DATA_YAML),
            device  = device,
            imgsz   = 640,
            split   = "val",
            project = str(OUTPUT_DIR),
            name    = "vehicles_yolov8n_eval",
            exist_ok = True,
        )
        print(f"\n[结果] mAP50    : {metrics.box.map50:.4f}")
        print(f"[结果] mAP50-95 : {metrics.box.map:.4f}")
        print(f"[结果] 最佳权重  : {best_pt}")
    else:
        print(f"[警告] 未找到最佳权重文件: {best_pt}")

    print(f"\n[完成] 所有输出已保存至: {OUTPUT_DIR / 'vehicles_yolov8n'}")


if __name__ == "__main__":
    train()
