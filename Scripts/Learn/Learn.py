import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# 基于脚本所在目录拼接路径，避免受当前工作目录影响
script_dir = Path(__file__).resolve().parent
image_path = script_dir / "Sample.jpg"
model_path = script_dir.parents[1] / "Model" / "YOLOv8" / "yolo8n" / "yolov8n.pt"

# 打印当前工作目录和图片路径
print("当前工作目录:", os.getcwd())
print("图片路径:", str(image_path))

# 检查文件是否存在
if not image_path.exists():
    raise FileNotFoundError(f"图片文件未找到: {image_path}")
if not model_path.exists():
    raise FileNotFoundError(f"模型文件未找到: {model_path}")

# 加载 YOLO 模型
model = YOLO(str(model_path))

# 读取图片
image = Image.open(str(image_path))

# 使用 YOLO 模型进行推理
# 调用的是 YOLO类的 __call__方法,这个方法在父类Model中定义
results = model(image)

# 打印推理结果
print("推理结果:")
print(results)

# 可视化结果（保存到文件）
output_path = script_dir / "inference_result.jpg"
results[0].save(filename=str(output_path))
print(f"推理结果已保存到: {output_path}")