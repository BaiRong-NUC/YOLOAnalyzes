import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

try:
    from imageio_ffmpeg import get_ffmpeg_exe
except ImportError:

    def get_ffmpeg_exe() -> str | None:
        return None


os.environ.setdefault("YOLO_OFFLINE", "true")

ROOT = Path(__file__).resolve().parents[2]
LOCAL_YOLO_SRC = ROOT / "YOLO 8.3.163"
if LOCAL_YOLO_SRC.exists():
    sys.path.insert(0, str(LOCAL_YOLO_SRC))

from ultralytics import YOLO


APP_TITLE = "YOLOv8 视频人体行为姿态预测"
DEFAULT_MODEL = ROOT / "Model" / "YOLOv8" / "yolo8n" / "yolov8n-pose.pt"
VIDEO_STATS_HEADERS = ["行为姿态", "累计帧次数"]
KEYPOINT_CONF_THRESHOLD = 0.2
TRACK_HISTORY = 18
ACTION_DISPLAY_NAMES = {
    "analyzing": "预分析中",
    "standing": "站立",
    "walking": "行走",
    "running": "跑步",
    "jumping": "跳跃",
    "crouching": "蹲伏",
}


@dataclass
class TrackState:
    track_id: int
    center_history: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=TRACK_HISTORY)
    )
    hip_history: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=TRACK_HISTORY)
    )
    ankle_y_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=TRACK_HISTORY)
    )
    height_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=TRACK_HISTORY)
    )
    last_frame: int = -1

    def update(
        self,
        center: tuple[float, float],
        hip_center: tuple[float, float] | None,
        ankle_y: float | None,
        box_height: float,
        frame_index: int,
    ) -> None:
        self.center_history.append(center)
        if hip_center is not None:
            self.hip_history.append(hip_center)
        if ankle_y is not None:
            self.ankle_y_history.append(ankle_y)
        self.height_history.append(max(box_height, 1.0))
        self.last_frame = frame_index

    @property
    def last_center(self) -> tuple[float, float] | None:
        if not self.center_history:
            return None
        return self.center_history[-1]

    @property
    def mean_height(self) -> float:
        if not self.height_history:
            return 1.0
        return float(np.mean(self.height_history))


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"


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


def keypoints_to_numpy(keypoints) -> tuple[np.ndarray | None, np.ndarray | None]:
    if keypoints is None or keypoints.xy is None or len(keypoints) == 0:
        return None, None

    xy = to_numpy(keypoints.xy)
    conf = None
    if getattr(keypoints, "conf", None) is not None:
        conf = to_numpy(keypoints.conf)
    return xy, conf


def to_numpy(values) -> np.ndarray:
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        return values.numpy()
    return np.asarray(values)


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


def visible_mask(points: np.ndarray, point_conf: np.ndarray | None) -> np.ndarray:
    if point_conf is not None:
        return point_conf > KEYPOINT_CONF_THRESHOLD
    return np.any(points > 0, axis=1)


def mean_visible_point(
    points: np.ndarray, mask: np.ndarray, indexes: tuple[int, ...]
) -> tuple[float, float] | None:
    visible_points = [
        points[index] for index in indexes if index < len(points) and mask[index]
    ]
    if not visible_points:
        return None
    mean_point = np.mean(np.array(visible_points), axis=0)
    return float(mean_point[0]), float(mean_point[1])


def detection_center(
    box: np.ndarray | None, points: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    if box is not None:
        x1, y1, x2, y2 = box
        return float((x1 + x2) / 2), float((y1 + y2) / 2)

    visible_points = points[mask]
    if len(visible_points) == 0:
        return 0.0, 0.0
    mean_point = visible_points.mean(axis=0)
    return float(mean_point[0]), float(mean_point[1])


def detection_height(
    box: np.ndarray | None, points: np.ndarray, mask: np.ndarray
) -> float:
    if box is not None:
        return max(float(box[3] - box[1]), 1.0)

    visible_points = points[mask]
    if len(visible_points) == 0:
        return 1.0
    return max(float(visible_points[:, 1].max() - visible_points[:, 1].min()), 1.0)


def classify_action(track: TrackState, points: np.ndarray, mask: np.ndarray) -> str:
    if len(track.center_history) < 5:
        return "analyzing"

    height = track.mean_height
    first_center = np.array(track.center_history[0])
    last_center = np.array(track.center_history[-1])
    center_motion = np.linalg.norm(last_center - first_center) / height

    centers = np.array(track.center_history)
    frame_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) / height
    mean_speed = float(frame_steps.mean()) if len(frame_steps) else 0.0
    max_speed = float(frame_steps.max()) if len(frame_steps) else 0.0
    moving_ratio = float((frame_steps > 0.008).mean()) if len(frame_steps) else 0.0

    vertical_range = 0.0
    if len(track.hip_history) >= 5:
        hips = np.array(track.hip_history)
        vertical_range = float((hips[:, 1].max() - hips[:, 1].min()) / height)

    ankle_jump = 0.0
    if len(track.ankle_y_history) >= 5:
        ankles = np.array(track.ankle_y_history)
        ankle_jump = float((ankles.max() - ankles.min()) / height)

    shoulder_center = mean_visible_point(points, mask, (5, 6))
    hip_center = mean_visible_point(points, mask, (11, 12))
    knee_center = mean_visible_point(points, mask, (13, 14))
    crouch_score = 0.0
    if (
        shoulder_center is not None
        and hip_center is not None
        and knee_center is not None
    ):
        torso = abs(hip_center[1] - shoulder_center[1]) / height
        hip_to_knee = abs(knee_center[1] - hip_center[1]) / height
        crouch_score = torso / max(hip_to_knee, 0.01)

    if vertical_range > 0.20 and ankle_jump > 0.16 and center_motion < 0.45:
        return "jumping"
    if crouch_score > 1.25 and center_motion < 0.20:
        return "crouching"
    running_speed = mean_speed > 0.10 or max_speed > 0.24
    running_pose = vertical_range > 0.10 or ankle_jump > 0.10
    if running_speed and running_pose and moving_ratio > 0.55:
        return "running"
    if (
        center_motion > 0.08
        or mean_speed > 0.015
        or max_speed > 0.12
        or moving_ratio > 0.45
    ):
        return "walking"
    return "standing"


def assign_track(
    tracks: dict[int, TrackState],
    used_track_ids: set[int],
    center: tuple[float, float],
    box_height: float,
    next_track_id: int,
) -> tuple[int, int]:
    best_track_id = None
    best_distance = float("inf")

    for track_id, track in tracks.items():
        if track_id in used_track_ids or track.last_center is None:
            continue

        distance = float(np.linalg.norm(np.array(center) - np.array(track.last_center)))
        max_distance = max(60.0, box_height * 0.8, track.mean_height * 0.8)
        if distance < max_distance and distance < best_distance:
            best_track_id = track_id
            best_distance = distance

    if best_track_id is not None:
        return best_track_id, next_track_id

    track_id = next_track_id
    tracks[track_id] = TrackState(track_id=track_id)
    return track_id, next_track_id + 1


def draw_action_label(
    frame: np.ndarray, box: np.ndarray | None, track_id: int, action: str
) -> None:
    if box is None:
        return

    x1, y1, _, _ = [int(value) for value in box]
    label = f"ID {track_id} | {action}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_y = max(y1 - 10, text_size[1] + 10)
    cv2.rectangle(
        frame,
        (x1, label_y - text_size[1] - baseline - 6),
        (x1 + text_size[0] + 8, label_y + baseline),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 4, label_y - 4),
        font,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def cleanup_tracks(
    tracks: dict[int, TrackState], frame_index: int, stale_after_frames: int
) -> None:
    stale_track_ids = [
        track_id
        for track_id, track in tracks.items()
        if frame_index - track.last_frame > stale_after_frames
    ]
    for track_id in stale_track_ids:
        del tracks[track_id]


def build_video_summary(
    frame_count: int, pose_count: int, action_counter: Counter[str]
) -> tuple[list[list[object]], str]:
    rows = [
        [ACTION_DISPLAY_NAMES.get(action, action), count]
        for action, count in action_counter.most_common()
    ]
    if not rows:
        rows = [["未检测到人体姿态", 0]]

    dominant_action = rows[0][0]
    summary = (
        f"已处理 {frame_count} 帧，累计检测人体姿态 {pose_count} 次。"
        f"出现最多的行为姿态为: {dominant_action}。"
    )
    return rows, summary


def predict_video_actions(
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

    tracks: dict[int, TrackState] = {}
    next_track_id = 1
    frame_index = 0
    pose_count = 0
    action_counter: Counter[str] = Counter()
    stale_after_frames = max(int(fps), 10)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[
                0
            ]
            annotated = result.plot()
            xy, keypoint_conf = keypoints_to_numpy(result.keypoints)
            boxes = None
            if result.boxes is not None and result.boxes.xyxy is not None:
                boxes = to_numpy(result.boxes.xyxy)

            used_track_ids: set[int] = set()
            if xy is not None:
                pose_count += len(xy)
                for person_index, points in enumerate(xy):
                    point_conf = None
                    if keypoint_conf is not None:
                        point_conf = keypoint_conf[person_index]

                    mask = visible_mask(points, point_conf)
                    box = (
                        boxes[person_index]
                        if boxes is not None and person_index < len(boxes)
                        else None
                    )
                    center = detection_center(box, points, mask)
                    box_height = detection_height(box, points, mask)
                    hip_center = mean_visible_point(points, mask, (11, 12))
                    ankle_center = mean_visible_point(points, mask, (15, 16))
                    ankle_y = ankle_center[1] if ankle_center is not None else None
                    track_id, next_track_id = assign_track(
                        tracks, used_track_ids, center, box_height, next_track_id
                    )
                    used_track_ids.add(track_id)

                    track = tracks[track_id]
                    track.update(center, hip_center, ankle_y, box_height, frame_index)
                    action = classify_action(track, points, mask)
                    action_counter[action] += 1
                    draw_action_label(annotated, box, track_id, action)

            cleanup_tracks(tracks, frame_index, stale_after_frames)
            writer.write(annotated)
            frame_index += 1

            if total_frames > 0:
                progress(
                    frame_index / total_frames,
                    desc=f"处理中: {frame_index}/{total_frames} 帧",
                )
    finally:
        capture.release()
        writer.release()

    stats, summary = build_video_summary(frame_index, pose_count, action_counter)
    browser_video_path = make_browser_video(output_path)
    return str(browser_video_path), stats, summary


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}\n默认模型: `{DEFAULT_MODEL}`")

        with gr.Row():
            model_path = gr.Textbox(label="模型路径", value=str(DEFAULT_MODEL), scale=4)
            conf = gr.Slider(
                label="置信度阈值", minimum=0.1, maximum=0.9, value=0.25, step=0.05
            )
            imgsz = gr.Slider(
                label="推理尺寸", minimum=320, maximum=1280, value=640, step=32
            )

        with gr.Row():
            video_input = gr.Video(label="输入视频")
            video_output = gr.Video(label="行为姿态预测视频")

        video_stats = gr.Dataframe(
            headers=VIDEO_STATS_HEADERS,
            datatype=["str", "number"],
            label="行为姿态统计",
            interactive=False,
        )
        video_summary = gr.Textbox(label="结果说明", interactive=False)
        video_button = gr.Button("开始视频行为姿态预测", variant="primary")
        video_button.click(
            fn=predict_video_actions,
            inputs=[video_input, model_path, conf, imgsz],
            outputs=[video_output, video_stats, video_summary],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = find_available_port(int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    access_host = get_local_ip() if server_name in {"0.0.0.0", "::"} else server_name

    print(f"服务启动地址: http://{access_host}:{server_port}")
    print(f"监听 IP: {server_name}")
    print(f"监听端口: {server_port}")

    try:
        app.launch(server_name=server_name, server_port=server_port)
    except KeyboardInterrupt:
        print("服务已停止")
