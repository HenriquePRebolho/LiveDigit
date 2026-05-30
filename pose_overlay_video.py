from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, DFineForObjectDetection, VitPoseForPoseEstimation


COCO_KEYPOINT_EDGES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)

DEFAULT_STAGE_ROI = "0.18,0.04,0.94,0.62"
DEFAULT_CONCERT_VIDEO_BOXES = (
    "0.30,0.31,0.13,0.31",  # pianist
    "0.55,0.23,0.08,0.36",  # double bass player
    "0.66,0.21,0.11,0.42",  # saxophonist
    "0.76,0.25,0.15,0.35",  # drummer, partially occluded by kit
)

KEYPOINT_COLORS = (
    (0, 255, 255),
    (0, 220, 255),
    (0, 220, 255),
    (0, 180, 255),
    (0, 180, 255),
    (0, 255, 120),
    (0, 255, 120),
    (80, 220, 0),
    (80, 220, 0),
    (160, 180, 0),
    (160, 180, 0),
    (255, 180, 0),
    (255, 180, 0),
    (255, 90, 0),
    (255, 90, 0),
    (255, 0, 80),
    (255, 0, 80),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a copy of a concert video with multi-person pose skeleton overlays."
    )
    parser.add_argument("--input", type=Path, default=Path("concertVideo.mov"), help="Input concert video path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "concertVideo_skeletons.mp4",
        help="Output annotated video path.",
    )
    parser.add_argument(
        "--detector-model",
        default="ustc-community/dfine-nano-coco",
        help="Hugging Face object detector. D-FINE COCO checkpoints are Apache-2.0.",
    )
    parser.add_argument(
        "--pose-model",
        default="usyd-community/vitpose-base-simple",
        help="Hugging Face ViTPose checkpoint. The selected default is Apache-2.0.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Inference device.")
    parser.add_argument("--person-threshold", type=float, default=0.25, help="Person detection score threshold.")
    parser.add_argument("--keypoint-threshold", type=float, default=0.35, help="Keypoint score threshold.")
    parser.add_argument(
        "--inference-width",
        type=int,
        default=1280,
        help="Resize frames to this width for model inference, then scale skeletons back to the original video.",
    )
    parser.add_argument(
        "--pose-stride",
        type=int,
        default=2,
        help="Run pose estimation every N frames and reuse the previous skeletons on skipped frames.",
    )
    parser.add_argument(
        "--detector-stride",
        type=int,
        default=6,
        help="Run person detection every N frames and reuse boxes between detections. 1 is most accurate.",
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=4,
        help="Maximum number of people to pose-estimate per frame, sorted by detection confidence.",
    )
    parser.add_argument(
        "--stage-roi",
        default=DEFAULT_STAGE_ROI,
        help=(
            "Stage region as x1,y1,x2,y2. Values in 0..1 are relative to the inference frame. "
            "Use an empty string to disable. Default focuses on the musicians and removes most audience detections."
        ),
    )
    parser.add_argument(
        "--extra-person-box",
        action="append",
        default=[],
        help=(
            "Extra calibrated person box as x,y,w,h. Values in 0..1 are relative to the inference frame. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--box-preset",
        choices=("concert-video", "none"),
        default="concert-video",
        help="Use calibrated performer boxes for the provided concertVideo.mov, or rely only on detection.",
    )
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="Use only calibrated boxes. This is fastest and most stable for fixed-camera live streams.",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=0,
        help="Optional debug limit. 0 processes the full video.",
    )
    parser.add_argument("--draw-boxes", action="store_true", help="Also draw detected person bounding boxes.")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def parse_box_arg(value: str, width: int, height: int, xyxy: bool) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected four comma-separated values, got: {value}")

    if all(0.0 <= part <= 1.0 for part in parts):
        scale = np.array([width, height, width, height], dtype=np.float32)
        parts_array = np.array(parts, dtype=np.float32) * scale
    else:
        parts_array = np.array(parts, dtype=np.float32)

    if xyxy:
        x1, y1, x2, y2 = parts_array
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid xyxy region: {value}")
    else:
        x1, y1, box_width, box_height = parts_array
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"Invalid xywh box: {value}")

    return parts_array.astype(np.float32)


def resize_for_inference(frame_bgr: np.ndarray, inference_width: int) -> tuple[np.ndarray, float, float]:
    height, width = frame_bgr.shape[:2]
    if inference_width <= 0 or inference_width >= width:
        return frame_bgr, 1.0, 1.0

    scale = inference_width / width
    inference_height = int(round(height * scale))
    resized = cv2.resize(frame_bgr, (inference_width, inference_height), interpolation=cv2.INTER_AREA)
    return resized, width / inference_width, height / inference_height


def to_coco_boxes(voc_boxes: torch.Tensor, scores: torch.Tensor, max_people: int) -> np.ndarray:
    if len(voc_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    order = torch.argsort(scores, descending=True)[:max_people]
    boxes = voc_boxes[order].detach().cpu().numpy().astype(np.float32)
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    boxes[:, 2:] = np.maximum(boxes[:, 2:], 1.0)
    return boxes


def filter_boxes_to_roi(boxes_coco: np.ndarray, roi_xyxy: np.ndarray | None) -> np.ndarray:
    if roi_xyxy is None or len(boxes_coco) == 0:
        return boxes_coco

    x1, y1, x2, y2 = roi_xyxy
    centers_x = boxes_coco[:, 0] + boxes_coco[:, 2] * 0.5
    centers_y = boxes_coco[:, 1] + boxes_coco[:, 3] * 0.5
    in_roi = (centers_x >= x1) & (centers_x <= x2) & (centers_y >= y1) & (centers_y <= y2)
    return boxes_coco[in_roi]


def add_extra_boxes(boxes_coco: np.ndarray, extra_boxes: list[np.ndarray], max_people: int) -> np.ndarray:
    if extra_boxes:
        boxes_coco = np.vstack([np.vstack(extra_boxes).astype(np.float32), boxes_coco])
    if len(boxes_coco) > max_people:
        boxes_coco = boxes_coco[:max_people]
    return boxes_coco


def scale_boxes(boxes_coco: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    if len(boxes_coco) == 0:
        return boxes_coco
    scaled = boxes_coco.copy()
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y
    return scaled


def scale_poses(
    poses: list[dict[str, torch.Tensor]],
    scale_x: float,
    scale_y: float,
) -> list[dict[str, torch.Tensor]]:
    if scale_x == 1.0 and scale_y == 1.0:
        return poses

    scaled_poses = []
    for pose in poses:
        scaled_pose = dict(pose)
        keypoints = pose["keypoints"].clone()
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y
        scaled_pose["keypoints"] = keypoints
        scaled_poses.append(scaled_pose)
    return scaled_poses


def detect_people(
    frame_rgb: Image.Image,
    detector_processor: AutoImageProcessor,
    detector: DFineForObjectDetection,
    device: torch.device,
    threshold: float,
    max_people: int,
) -> np.ndarray:
    inputs = detector_processor(images=frame_rgb, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = detector(**inputs)

    results = detector_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(frame_rgb.height, frame_rgb.width)], device=device),
        threshold=threshold,
    )[0]

    labels = results["labels"]
    scores = results["scores"]
    label_names = [detector.config.id2label[int(label)].lower() for label in labels]
    person_mask = torch.tensor([name == "person" or int(label) == 0 for name, label in zip(label_names, labels)])
    person_mask = person_mask.to(labels.device)
    return to_coco_boxes(results["boxes"][person_mask], scores[person_mask], max_people)


def estimate_pose(
    frame_rgb: Image.Image,
    boxes_coco: np.ndarray,
    pose_processor: AutoProcessor,
    pose_model: VitPoseForPoseEstimation,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    if len(boxes_coco) == 0:
        return []

    inputs = pose_processor(frame_rgb, boxes=[boxes_coco], return_tensors="pt").to(device)
    if "dataset_index" not in inputs and getattr(pose_model.config, "backbone_config", None):
        num_experts = getattr(pose_model.config.backbone_config, "num_experts", 1)
        if num_experts > 1:
            batch_size = inputs["pixel_values"].shape[0]
            inputs["dataset_index"] = torch.zeros(batch_size, dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = pose_model(**inputs)
    return pose_processor.post_process_pose_estimation(outputs, boxes=[boxes_coco])[0]


def draw_poses(
    frame_bgr: np.ndarray,
    boxes_coco: np.ndarray,
    poses: list[dict[str, torch.Tensor]],
    keypoint_threshold: float,
    draw_boxes: bool,
) -> np.ndarray:
    annotated = frame_bgr.copy()

    for person_index, pose in enumerate(poses):
        keypoints = pose["keypoints"].detach().cpu().numpy()
        scores = pose["scores"].detach().cpu().numpy()

        for edge_start, edge_end in COCO_KEYPOINT_EDGES:
            if scores[edge_start] < keypoint_threshold or scores[edge_end] < keypoint_threshold:
                continue
            x1, y1 = keypoints[edge_start]
            x2, y2 = keypoints[edge_end]
            cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (60, 255, 60), 3, cv2.LINE_AA)

        for keypoint_index, ((x_coord, y_coord), score) in enumerate(zip(keypoints, scores)):
            if score < keypoint_threshold:
                continue
            cv2.circle(
                annotated,
                (int(x_coord), int(y_coord)),
                4,
                KEYPOINT_COLORS[keypoint_index % len(KEYPOINT_COLORS)],
                -1,
                cv2.LINE_AA,
            )

        if draw_boxes and person_index < len(boxes_coco):
            x_coord, y_coord, width, height = boxes_coco[person_index]
            cv2.rectangle(
                annotated,
                (int(x_coord), int(y_coord)),
                (int(x_coord + width), int(y_coord + height)),
                (255, 180, 30),
                2,
                cv2.LINE_AA,
            )

    return annotated


def format_duration(seconds: float) -> str:
    whole_seconds = int(round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if args.detector_stride < 1:
        raise ValueError("--detector-stride must be >= 1")
    if args.pose_stride < 1:
        raise ValueError("--pose-stride must be >= 1")
    if args.box_preset == "concert-video":
        args.extra_person_box = list(DEFAULT_CONCERT_VIDEO_BOXES) + args.extra_person_box

    device = select_device(args.device)
    detector_processor = None
    detector = None
    if args.skip_detector:
        print("Skipping detector and using calibrated performer boxes.")
    else:
        print(f"Loading detector {args.detector_model} on {device}...")
        detector_processor = AutoImageProcessor.from_pretrained(args.detector_model)
        detector = DFineForObjectDetection.from_pretrained(args.detector_model).to(device).eval()

    print(f"Loading pose model {args.pose_model} on {device}...")
    pose_processor = AutoProcessor.from_pretrained(args.pose_model)
    pose_model = VitPoseForPoseEstimation.from_pretrained(args.pose_model).to(device).eval()

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {args.output}")

    last_boxes = np.empty((0, 4), dtype=np.float32)
    last_boxes_original = np.empty((0, 4), dtype=np.float32)
    last_poses_original: list[dict[str, torch.Tensor]] = []
    frame_index = 0
    started_at = time.time()

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        if args.limit_frames and frame_index >= args.limit_frames:
            break

        inference_bgr, scale_x, scale_y = resize_for_inference(frame_bgr, args.inference_width)
        inference_height, inference_width = inference_bgr.shape[:2]
        frame_rgb_array = cv2.cvtColor(inference_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = Image.fromarray(frame_rgb_array)
        roi = parse_box_arg(args.stage_roi, inference_width, inference_height, xyxy=True) if args.stage_roi else None
        extra_boxes = [
            parse_box_arg(box_arg, inference_width, inference_height, xyxy=False)
            for box_arg in args.extra_person_box
        ]

        should_update_pose = frame_index % args.pose_stride == 0
        if should_update_pose and frame_index % args.detector_stride == 0:
            if args.skip_detector:
                last_boxes = add_extra_boxes(np.empty((0, 4), dtype=np.float32), extra_boxes, args.max_people)
            else:
                if detector_processor is None or detector is None:
                    raise RuntimeError("Detector is not loaded.")
                last_boxes = detect_people(
                    frame_rgb,
                    detector_processor,
                    detector,
                    device,
                    args.person_threshold,
                    args.max_people,
                )
                last_boxes = filter_boxes_to_roi(last_boxes, roi)
                last_boxes = add_extra_boxes(last_boxes, extra_boxes, args.max_people)

        if should_update_pose:
            poses = estimate_pose(frame_rgb, last_boxes, pose_processor, pose_model, device)
            last_boxes_original = scale_boxes(last_boxes, scale_x, scale_y)
            last_poses_original = scale_poses(poses, scale_x, scale_y)

        annotated = draw_poses(
            frame_bgr,
            last_boxes_original,
            last_poses_original,
            args.keypoint_threshold,
            args.draw_boxes,
        )
        writer.write(annotated)

        frame_index += 1
        if frame_index % 25 == 0:
            elapsed = max(time.time() - started_at, 1e-6)
            print(f"Processed {frame_index}/{total_frames or '?'} frames ({frame_index / elapsed:.2f} fps).")

    capture.release()
    writer.release()
    total_elapsed = time.time() - started_at
    print(f"Done. Wrote annotated video to {args.output}")
    print(
        "Timing summary: "
        f"{frame_index} frames in {format_duration(total_elapsed)} "
        f"({total_elapsed:.2f}s, {frame_index / max(total_elapsed, 1e-6):.2f} fps average)."
    )


if __name__ == "__main__":
    main()
