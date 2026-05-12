import os
import shutil
import socket
import subprocess
import sys
import tempfile
import asyncio
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd
from imageio_ffmpeg import get_ffmpeg_exe

os.environ.setdefault("YOLO_OFFLINE", "true")


def ignore_connection_reset(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    exception = context.get("exception")
    if (
        isinstance(exception, ConnectionResetError)
        and getattr(exception, "winerror", None) == 10054
    ):
        return
    loop.default_exception_handler(context)


if sys.platform.startswith("win"):
    base_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if base_policy is not None:

        class QuietWindowsSelectorEventLoopPolicy(base_policy):
            def new_event_loop(self):
                loop = super().new_event_loop()
                loop.set_exception_handler(ignore_connection_reset)
                return loop

        asyncio.set_event_loop_policy(QuietWindowsSelectorEventLoopPolicy())

ROOT = Path(__file__).resolve().parents[2]
LOCAL_YOLO_SRC = ROOT / "YOLO 8.3.163"
if LOCAL_YOLO_SRC.exists():
    sys.path.insert(0, str(LOCAL_YOLO_SRC))

from ultralytics import YOLO


def find_default_model() -> Path:
    output_dir = ROOT / "Output"
    if output_dir.exists():
        candidates = sorted(
            output_dir.glob("**/weights/best.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return ROOT / "Model" / "YOLOv8" / "yolo8n" / "yolov8n.pt"


DEFAULT_MODEL = find_default_model()
APP_TITLE = "YOLO 车辆检测最小演示版"


def find_available_port(preferred_port: int = 7860, max_tries: int = 20) -> int:
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@lru_cache(maxsize=4)
def load_model(model_path: str) -> YOLO:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    return YOLO(str(path))


def build_stats(result) -> tuple[pd.DataFrame, str]:
    boxes = result.boxes
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        empty = pd.DataFrame(columns=["类别", "数量"])
        return empty, "未检测到目标"

    names = result.names
    class_ids = [int(class_id) for class_id in boxes.cls.tolist()]
    counter = Counter(names[class_id] for class_id in class_ids)
    stats = pd.DataFrame(
        [
            {"类别": class_name, "数量": count}
            for class_name, count in counter.most_common()
        ],
        columns=["类别", "数量"],
    )
    summary = f"检测到 {len(class_ids)} 个目标，共 {len(counter)} 个类别"
    return stats, summary


def empty_stats() -> pd.DataFrame:
    return pd.DataFrame(columns=["类别", "数量"])


def annotate_image(result):
    annotated = result.plot()
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def make_browser_video(video_path: Path) -> Path:
    ffmpeg_path = shutil.which("ffmpeg") or get_ffmpeg_exe()
    if not ffmpeg_path:
        return video_path

    web_path = video_path.with_name(f"{video_path.stem}_web.mp4")
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(web_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if web_path.exists() and web_path.stat().st_size > 0:
            return web_path
    except Exception:
        return video_path
    return video_path


def run_image_detection(image, model_path: str, conf: float, imgsz: int):
    if image is None:
        return None, empty_stats(), "等待输入画面"

    model = load_model(model_path)
    result = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)[0]
    annotated = annotate_image(result)
    stats, summary = build_stats(result)
    return annotated, stats, summary


def predict_image(image, model_path: str, conf: float, imgsz: int):
    if image is None:
        raise gr.Error("请先上传一张图片")

    return run_image_detection(image, model_path, conf, imgsz)


def predict_webcam_frame(image, model_path: str, conf: float, imgsz: int):
    annotated, stats, summary = run_image_detection(image, model_path, conf, imgsz)
    if image is None:
        return None, empty_stats(), "等待摄像头画面"
    return annotated, stats, f"实时检测中: {summary}"


def reset_webcam_outputs():
    return None, empty_stats(), "摄像头检测已停止"


def build_video_summary(
    frame_count: int, total_detections: int, counter: Counter
) -> tuple[pd.DataFrame, str]:
    if not counter:
        empty = pd.DataFrame(columns=["类别", "累计检测次数"])
        return empty, f"已处理 {frame_count} 帧，未检测到目标"

    stats = pd.DataFrame(
        [
            {"类别": class_name, "累计检测次数": count}
            for class_name, count in counter.most_common()
        ],
        columns=["类别", "累计检测次数"],
    )
    summary = (
        f"已处理 {frame_count} 帧，累计检测 {total_detections} 次。"
        "视频统计为逐帧累计次数，未做跨帧去重。"
    )
    return stats, summary


def predict_video(
    video_path: str, model_path: str, conf: float, imgsz: int, progress=gr.Progress()
):
    if not video_path:
        raise gr.Error("请先上传一个视频")

    model = load_model(model_path)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise gr.Error("视频打开失败，请更换文件后重试")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    output_path = Path(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    total_detections = 0
    counter: Counter[str] = Counter()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[
                0
            ]
            annotated = result.plot()
            writer.write(annotated)

            boxes = result.boxes
            if boxes is not None and boxes.cls is not None and len(boxes) > 0:
                class_ids = [int(class_id) for class_id in boxes.cls.tolist()]
                total_detections += len(class_ids)
                counter.update(result.names[class_id] for class_id in class_ids)

            frame_index += 1
            if total_frames > 0:
                progress(
                    frame_index / total_frames,
                    desc=f"处理中: {frame_index}/{total_frames} 帧",
                )
    finally:
        capture.release()
        writer.release()

    stats, summary = build_video_summary(frame_index, total_detections, counter)
    browser_video_path = make_browser_video(output_path)
    return str(browser_video_path), stats, summary


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"""
            # {APP_TITLE}
            支持图片检测、视频检测和类别统计。默认使用 {DEFAULT_MODEL}。
            如果你已经训练出自己的 best.pt，可以直接把模型路径改成你的权重文件。
            """)

        with gr.Row():
            model_path = gr.Textbox(label="模型路径", value=str(DEFAULT_MODEL), scale=4)
            conf = gr.Slider(
                label="置信度阈值", minimum=0.1, maximum=0.9, value=0.25, step=0.05
            )
            imgsz = gr.Slider(
                label="推理尺寸", minimum=320, maximum=1280, value=640, step=32
            )

        with gr.Tab("图片检测"):
            with gr.Row():
                image_input = gr.Image(label="输入图片", type="numpy")
                image_output = gr.Image(label="检测结果")
            image_stats = gr.Dataframe(label="类别统计", interactive=False)
            image_summary = gr.Textbox(label="结果说明", interactive=False)
            image_button = gr.Button("开始图片检测", variant="primary")
            image_button.click(
                fn=predict_image,
                inputs=[image_input, model_path, conf, imgsz],
                outputs=[image_output, image_stats, image_summary],
            )

        with gr.Tab("视频检测"):
            video_input = gr.Video(label="输入视频")
            video_output = gr.Video(label="检测结果视频")
            video_stats = gr.Dataframe(label="类别统计", interactive=False)
            video_summary = gr.Textbox(label="结果说明", interactive=False)
            video_button = gr.Button("开始视频检测", variant="primary")
            video_button.click(
                fn=predict_video,
                inputs=[video_input, model_path, conf, imgsz],
                outputs=[video_output, video_stats, video_summary],
            )

        with gr.Tab("摄像头检测"):
            gr.Markdown(
                "允许浏览器调用本机摄像头，采集到的画面会逐帧送入模型做实时检测。"
            )
            with gr.Row():
                webcam_input = gr.Image(
                    label="摄像头输入",
                    sources=["webcam"],
                    type="numpy",
                    streaming=True,
                )
                webcam_output = gr.Image(label="实时检测结果")
            webcam_stats = gr.Dataframe(label="当前帧类别统计", interactive=False)
            webcam_summary = gr.Textbox(
                value="点击摄像头按钮开始采集", label="状态", interactive=False
            )
            webcam_clear = gr.Button("停止并清空结果")
            webcam_input.stream(
                fn=predict_webcam_frame,
                inputs=[webcam_input, model_path, conf, imgsz],
                outputs=[webcam_output, webcam_stats, webcam_summary],
            )
            webcam_clear.click(
                fn=reset_webcam_outputs,
                inputs=None,
                outputs=[webcam_output, webcam_stats, webcam_summary],
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    server_port = find_available_port(int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    try:
        demo.launch(server_name="127.0.0.1", server_port=server_port)
    except KeyboardInterrupt:
        print("服务已停止")
