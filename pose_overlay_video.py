from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, DFineForObjectDetection, VitPoseForPoseEstimation

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


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

# Leave empty by default because the camera may move or zoom. If a future camera
# is fixed, you can still pass --stage-roi to ignore audience areas.
DEFAULT_STAGE_ROI = ""

YOLO_MODEL_PATH = Path("bounding_boxes") / "testing" / "yoloe-26x-seg-pf.pt"
YOLO_PERSON_CLASS_ID = 2163
YOLO_PERFORMER_CLASS_IDS = (
    322,  # bassist
    1424,  # drummer
    1977,  # guitarist
    2758,  # musician
    3526,  # saxophonist
    4369,  # violinist
    4370,  # violist
)
YOLO_INSTRUMENT_CONTEXT_CLASS_IDS = (
    9,  # accordion
    263,  # banjo
    319,  # bass
    320,  # bass guitar
    321,  # bass horn
    519,  # boom microphone
    808,  # cello
    947,  # clarinet
    1026,  # keyboard
    1125,  # trumpet
    1244,  # cymbal
    1423,  # drum
    1425,  # drumstick
    1476,  # electric guitar
    1711,  # flute
    1976,  # guitar
    2632,  # microphone
    2757,  # musical keyboard
    3038,  # piano
    3525,  # sax
    4270,  # trombone
    4309,  # ukulele
    4368,  # violin
)
YOLO_RELEVANT_CLASS_IDS = (
    2163,  # person
    *YOLO_PERFORMER_CLASS_IDS,
    *YOLO_INSTRUMENT_CONTEXT_CLASS_IDS,
)
YOLO_BODY_CLASS_IDS = (
    YOLO_PERSON_CLASS_ID,
    *YOLO_PERFORMER_CLASS_IDS,
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
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input") / "concertVideo.mov",
        help="Input concert video path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "concertVideo_skeletons.mp4",
        help="Output annotated video path.",
    )
    parser.add_argument(
        "--detector-model",
        default="ustc-community/dfine-small-coco",
        help="Hugging Face object detector. D-FINE COCO checkpoints are Apache-2.0.",
    )
    parser.add_argument(
        "--box-source",
        choices=("dfine", "yolo", "compare"),
        default="dfine",
        help=(
            "Bounding-box source for ViTPose. 'compare' runs D-FINE and YOLO separately and prints metrics for both."
        ),
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=YOLO_MODEL_PATH,
        help="YOLO model path used when --box-source is yolo or compare.",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=1088,
        help="YOLO inference image size. Smaller is faster; larger can improve small-person detection.",
    )
    parser.add_argument(
        "--yolo-threshold",
        type=float,
        default=0.2,
        help="YOLO confidence threshold for musician/person boxes.",
    )
    parser.add_argument(
        "--comparison-report",
        type=Path,
        default=Path("output") / "box_source_comparison.csv",
        help="CSV report path written when --box-source compare is used.",
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
        default=2,
        help="Run person detection every N frames and reuse boxes between detections. 1 follows camera motion best.",
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=8,
        help="Maximum number of people to pose-estimate per frame, sorted by detection confidence.",
    )
    parser.add_argument(
        "--box-smoothing",
        type=float,
        default=0.25,
        help=(
            "Smooth detected person boxes over time. 0 uses raw detections; higher values reduce jitter but react "
            "slower to fast camera moves."
        ),
    )
    parser.add_argument(
        "--stage-roi",
        default=DEFAULT_STAGE_ROI,
        help=(
            "Stage region as x1,y1,x2,y2. Values in 0..1 are relative to the inference frame. "
            "Empty by default so detections can adapt to camera movement and zoom."
        ),
    )
    parser.add_argument(
        "--extra-person-box",
        action="append",
        default=[],
        help=(
            "Extra calibrated person box as x,y,w,h. Values in 0..1 are relative to the inference frame. "
            "Can be repeated. Optional fallback only; no boxes are hard-coded by default."
        ),
    )
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="Use only boxes passed with --extra-person-box. Useful only for fixed-camera debugging.",
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


def box_iou_xywh(first_box: np.ndarray, second_box: np.ndarray) -> float:
    """Calculate overlap between two boxes. Look up: Intersection over Union."""
    first_x1, first_y1, first_w, first_h = first_box
    second_x1, second_y1, second_w, second_h = second_box
    first_x2 = first_x1 + first_w
    first_y2 = first_y1 + first_h
    second_x2 = second_x1 + second_w
    second_y2 = second_y1 + second_h

    overlap_x1 = max(first_x1, second_x1)
    overlap_y1 = max(first_y1, second_y1)
    overlap_x2 = min(first_x2, second_x2)
    overlap_y2 = min(first_y2, second_y2)
    overlap_w = max(0.0, overlap_x2 - overlap_x1)
    overlap_h = max(0.0, overlap_y2 - overlap_y1)
    overlap_area = overlap_w * overlap_h

    first_area = max(0.0, first_w) * max(0.0, first_h)
    second_area = max(0.0, second_w) * max(0.0, second_h)
    union_area = first_area + second_area - overlap_area
    if union_area <= 0.0:
        return 0.0
    return float(overlap_area / union_area)


def expand_box_xywh(box: np.ndarray, image_width: int, image_height: int, scale: float) -> np.ndarray:
    """Expand a box around its center while keeping it inside the image."""
    x_coord, y_coord, width, height = box
    center_x = x_coord + width * 0.5
    center_y = y_coord + height * 0.5
    new_width = width * scale
    new_height = height * scale
    new_x = max(0.0, center_x - new_width * 0.5)
    new_y = max(0.0, center_y - new_height * 0.5)
    new_width = min(new_width, image_width - new_x)
    new_height = min(new_height, image_height - new_y)
    return np.array([new_x, new_y, max(new_width, 1.0), max(new_height, 1.0)], dtype=np.float32)


def instrument_box_to_person_hint(box: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Estimate a loose person crop from an instrument/context box when YOLO misses the body class."""
    x_coord, y_coord, width, height = box
    center_x = x_coord + width * 0.5
    center_y = y_coord + height * 0.35
    new_width = max(width * 2.4, height * 1.2)
    new_height = max(height * 3.2, width * 1.7)
    new_x = min(max(0.0, center_x - new_width * 0.5), max(0.0, image_width - 1.0))
    new_y = min(max(0.0, center_y - new_height * 0.35), max(0.0, image_height - 1.0))
    new_width = min(new_width, image_width - new_x)
    new_height = min(new_height, image_height - new_y)
    return np.array([new_x, new_y, max(new_width, 1.0), max(new_height, 1.0)], dtype=np.float32)


def best_context_overlap(person_box: np.ndarray, context_boxes: np.ndarray) -> float:
    """Return the strongest overlap between a person box and any instrument-context box."""
    if len(context_boxes) == 0:
        return 0.0
    return max(box_iou_xywh(person_box, context_box) for context_box in context_boxes)


def box_center_xy(box: np.ndarray) -> tuple[float, float]:
    x_coord, y_coord, width, height = box
    return float(x_coord + width * 0.5), float(y_coord + height * 0.5)


def point_inside_box(point_x: float, point_y: float, box: np.ndarray) -> bool:
    x_coord, y_coord, width, height = box
    return x_coord <= point_x <= x_coord + width and y_coord <= point_y <= y_coord + height


def best_context_association(person_box: np.ndarray, context_boxes: np.ndarray, image_width: int, image_height: int) -> float:
    """Score how likely a body box belongs to nearby instrument/microphone detections."""
    if len(context_boxes) == 0:
        return 0.0

    expanded_person = expand_box_xywh(person_box, image_width, image_height, 1.45)
    person_cx, person_cy = box_center_xy(person_box)
    person_scale = max(np.sqrt(max(person_box[2] * person_box[3], 1.0)), 1.0)
    best_score = 0.0

    for context_box in context_boxes:
        context_cx, context_cy = box_center_xy(context_box)
        context_scale = max(np.sqrt(max(context_box[2] * context_box[3], 1.0)), 1.0)
        distance = np.hypot(person_cx - context_cx, person_cy - context_cy)
        distance_limit = max(person_scale * 0.85 + context_scale * 0.85, 1.0)
        distance_score = max(0.0, 1.0 - distance / distance_limit)
        overlap_score = min(box_iou_xywh(expanded_person, context_box) * 4.0, 1.0)
        inside_score = 1.0 if point_inside_box(context_cx, context_cy, expanded_person) else 0.0
        best_score = max(best_score, overlap_score, distance_score, inside_score)

    return float(best_score)


def dedupe_ranked_boxes(boxes: np.ndarray, scores: np.ndarray, max_boxes: int, iou_threshold: float = 0.45) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    selected_boxes: list[np.ndarray] = []
    for index in np.argsort(scores)[::-1]:
        box = boxes[index]
        if any(box_iou_xywh(box, selected_box) >= iou_threshold for selected_box in selected_boxes):
            continue
        selected_boxes.append(box)
        if len(selected_boxes) >= max_boxes:
            break

    return np.array(selected_boxes, dtype=np.float32) if selected_boxes else np.empty((0, 4), dtype=np.float32)


def smooth_boxes(
    previous_boxes: np.ndarray,
    detected_boxes: np.ndarray,
    smoothing: float,
    min_iou: float = 0.15,
) -> np.ndarray:
    """Smooth matching boxes from frame to frame so skeletons jump less."""
    if smoothing <= 0.0 or len(previous_boxes) == 0 or len(detected_boxes) == 0:
        return detected_boxes

    smoothing = min(max(smoothing, 0.0), 0.95)
    smoothed_boxes = detected_boxes.copy()
    used_previous_indexes: set[int] = set()

    for detected_index, detected_box in enumerate(detected_boxes):
        best_iou = 0.0
        best_previous_index = -1
        for previous_index, previous_box in enumerate(previous_boxes):
            if previous_index in used_previous_indexes:
                continue
            iou = box_iou_xywh(previous_box, detected_box)
            if iou > best_iou:
                best_iou = iou
                best_previous_index = previous_index

        if best_previous_index >= 0 and best_iou >= min_iou:
            used_previous_indexes.add(best_previous_index)
            previous_box = previous_boxes[best_previous_index]
            smoothed_boxes[detected_index] = smoothing * previous_box + (1.0 - smoothing) * detected_box

    return smoothed_boxes


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


def detect_yolo_musicians(
    frame_rgb: Image.Image,
    yolo_model: object,
    threshold: float,
    max_people: int,
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    """Use the YOLO model from bounding_boxes/testing to find musician/person boxes."""
    # The reference bounding_boxes/testing scripts pass OpenCV BGR frames to Ultralytics.
    frame_array = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR)
    results = yolo_model(
        frame_array,
        imgsz=image_size,
        classes=list(YOLO_RELEVANT_CLASS_IDS),
        conf=threshold,
        device=str(device),
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    boxes_xyxy = results[0].boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = results[0].boxes.conf.detach().cpu().numpy()
    class_ids = results[0].boxes.cls.detach().cpu().numpy().astype(np.int32)

    boxes_coco = boxes_xyxy.copy()
    boxes_coco[:, 2] = boxes_coco[:, 2] - boxes_coco[:, 0]
    boxes_coco[:, 3] = boxes_coco[:, 3] - boxes_coco[:, 1]
    boxes_coco[:, 2:] = np.maximum(boxes_coco[:, 2:], 1.0)

    body_mask = np.isin(class_ids, YOLO_BODY_CLASS_IDS)
    context_mask = np.isin(class_ids, YOLO_INSTRUMENT_CONTEXT_CLASS_IDS)
    body_boxes = boxes_coco[body_mask]
    body_scores = scores[body_mask]
    body_class_ids = class_ids[body_mask]
    context_boxes = boxes_coco[context_mask]
    context_scores = scores[context_mask]

    candidate_boxes: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []

    if len(body_boxes) > 0:
        context_associations = np.array(
            [
                best_context_association(box, context_boxes, frame_rgb.width, frame_rgb.height)
                for box in body_boxes
            ],
            dtype=np.float32,
        )
        performer_boost = np.isin(body_class_ids, YOLO_PERFORMER_CLASS_IDS).astype(np.float32) * 1.5
        y_bottom = body_boxes[:, 1] + body_boxes[:, 3]
        foreground_audience_penalty = (
            (context_associations < 0.05)
            & (body_class_ids == YOLO_PERSON_CLASS_ID)
            & (y_bottom > frame_rgb.height * 0.82)
        ).astype(np.float32) * 0.75
        musician_scores = body_scores + performer_boost + context_associations * 2.0 - foreground_audience_penalty

        if len(context_boxes) > 0:
            musician_mask = (context_associations >= 0.05) | np.isin(body_class_ids, YOLO_PERFORMER_CLASS_IDS)
            if np.any(musician_mask):
                body_boxes = body_boxes[musician_mask]
                musician_scores = musician_scores[musician_mask]

        candidate_boxes.append(body_boxes)
        candidate_scores.append(musician_scores)

    if len(context_boxes) > 0:
        hinted_boxes = np.array(
            [instrument_box_to_person_hint(box, frame_rgb.width, frame_rgb.height) for box in context_boxes],
            dtype=np.float32,
        )
        hinted_scores = context_scores + 1.0
        candidate_boxes.append(hinted_boxes)
        candidate_scores.append(hinted_scores)

    if not candidate_boxes:
        return np.empty((0, 4), dtype=np.float32)

    all_candidate_boxes = np.vstack(candidate_boxes).astype(np.float32)
    all_candidate_scores = np.concatenate(candidate_scores).astype(np.float32)
    return dedupe_ranked_boxes(all_candidate_boxes, all_candidate_scores, max_people)


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


def run_pipeline(args: argparse.Namespace, box_source: str, output_path: Path) -> dict[str, float]:
    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if args.detector_stride < 1:
        raise ValueError("--detector-stride must be >= 1")
    if args.pose_stride < 1:
        raise ValueError("--pose-stride must be >= 1")
    if args.limit_frames < 0:
        raise ValueError("--limit-frames must be >= 0")
    if not 0.0 <= args.box_smoothing <= 0.95:
        raise ValueError("--box-smoothing must be between 0 and 0.95")
    if box_source == "yolo" and YOLO is None:
        raise ImportError("ultralytics is required for --box-source yolo. Install it with: pip install ultralytics")
    if box_source == "yolo" and not args.yolo_model.exists():
        raise FileNotFoundError(f"YOLO model not found: {args.yolo_model}")

    device = select_device(args.device)
    detector_processor = None
    detector = None
    yolo_model = None
    if args.skip_detector:
        print("Skipping detector and using only boxes passed with --extra-person-box.")
    elif box_source == "yolo":
        print(f"Loading YOLO box model {args.yolo_model}...")
        yolo_model = YOLO(str(args.yolo_model))
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
    frame_goal = min(args.limit_frames, total_frames) if args.limit_frames and total_frames else args.limit_frames or total_frames
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    last_boxes = np.empty((0, 4), dtype=np.float32)
    last_boxes_original = np.empty((0, 4), dtype=np.float32)
    last_poses_original: list[dict[str, torch.Tensor]] = []
    frame_times: list[float] = []
    skeleton_stage_times: list[float] = []
    detection_times: list[float] = []
    pose_update_counts: list[int] = []
    valid_keypoint_counts: list[int] = []
    keypoint_confidences: list[float] = []
    frame_index = 0
    started_at = time.perf_counter()
    print(f"Running pipeline with {box_source.upper()} bounding boxes.")

    while True:
        if args.limit_frames and frame_index >= args.limit_frames:
            break

        ok, frame_bgr = capture.read()
        if not ok:
            break

        frame_started_at = time.perf_counter()
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
                detection_started_at = time.perf_counter()
                if box_source == "yolo":
                    if yolo_model is None:
                        raise RuntimeError("YOLO model is not loaded.")
                    detected_boxes = detect_yolo_musicians(
                        frame_rgb,
                        yolo_model,
                        args.yolo_threshold,
                        args.max_people,
                        args.yolo_imgsz,
                        device,
                    )
                else:
                    if detector_processor is None or detector is None:
                        raise RuntimeError("Detector is not loaded.")
                    detected_boxes = detect_people(
                        frame_rgb,
                        detector_processor,
                        detector,
                        device,
                        args.person_threshold,
                        args.max_people,
                    )
                detection_times.append(time.perf_counter() - detection_started_at)
                detected_boxes = filter_boxes_to_roi(detected_boxes, roi)
                detected_boxes = add_extra_boxes(detected_boxes, extra_boxes, args.max_people)
                last_boxes = smooth_boxes(last_boxes, detected_boxes, args.box_smoothing)

        # This second timer starts after the person/musician box-detection section.
        # It measures only the skeleton-related part: pose estimation, scaling,
        # and drawing. On skipped frames it measures drawing the reused skeleton.
        skeleton_stage_started_at = time.perf_counter()
        if should_update_pose:
            poses = estimate_pose(frame_rgb, last_boxes, pose_processor, pose_model, device)
            pose_update_counts.append(len(poses))
            for pose in poses:
                scores = pose["scores"].detach().cpu().numpy()
                confident_scores = scores[scores >= args.keypoint_threshold]
                valid_keypoint_counts.append(int(len(confident_scores)))
                if len(confident_scores) > 0:
                    keypoint_confidences.append(float(np.mean(confident_scores)))
            last_boxes_original = scale_boxes(last_boxes, scale_x, scale_y)
            last_poses_original = scale_poses(poses, scale_x, scale_y)

        annotated = draw_poses(
            frame_bgr,
            last_boxes_original,
            last_poses_original,
            args.keypoint_threshold,
            args.draw_boxes,
        )
        skeleton_stage_elapsed = time.perf_counter() - skeleton_stage_started_at
        skeleton_stage_times.append(skeleton_stage_elapsed)
        writer.write(annotated)

        frame_index += 1
        frame_elapsed = time.perf_counter() - frame_started_at
        frame_times.append(frame_elapsed)
        print(f"Frame {frame_index}/{frame_goal or '?'} generated in {frame_elapsed:.3f}s.")
        print(
            f"  Skeleton stage after box detection: {skeleton_stage_elapsed:.3f}s "
            f"({'updated pose' if should_update_pose else 'reused previous pose'})."
        )
        if frame_index % 25 == 0:
            elapsed = max(time.perf_counter() - started_at, 1e-6)
            print(f"Processed {frame_index}/{frame_goal or '?'} frames ({frame_index / elapsed:.2f} fps).")

    capture.release()
    writer.release()
    total_elapsed = time.perf_counter() - started_at
    average_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0.0
    min_frame_time = min(frame_times) if frame_times else 0.0
    max_frame_time = max(frame_times) if frame_times else 0.0
    average_skeleton_stage_time = sum(skeleton_stage_times) / len(skeleton_stage_times) if skeleton_stage_times else 0.0
    min_skeleton_stage_time = min(skeleton_stage_times) if skeleton_stage_times else 0.0
    max_skeleton_stage_time = max(skeleton_stage_times) if skeleton_stage_times else 0.0
    average_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0.0
    average_people_per_pose_update = sum(pose_update_counts) / len(pose_update_counts) if pose_update_counts else 0.0
    average_valid_keypoints_per_person = (
        sum(valid_keypoint_counts) / len(valid_keypoint_counts) if valid_keypoint_counts else 0.0
    )
    average_keypoint_confidence = sum(keypoint_confidences) / len(keypoint_confidences) if keypoint_confidences else 0.0
    print(f"Done. Wrote annotated video to {output_path}")
    print(
        "Timing summary: "
        f"{frame_index} frames in {format_duration(total_elapsed)} "
        f"({total_elapsed:.2f}s, {frame_index / max(total_elapsed, 1e-6):.2f} fps average)."
    )
    print(
        "Per-frame timing: "
        f"average {average_frame_time:.3f}s, "
        f"min {min_frame_time:.3f}s, "
        f"max {max_frame_time:.3f}s."
    )
    print(
        "Skeleton-stage timing after box detection: "
        f"average {average_skeleton_stage_time:.3f}s, "
        f"min {min_skeleton_stage_time:.3f}s, "
        f"max {max_skeleton_stage_time:.3f}s."
    )
    print(f"Box-detection timing ({box_source}): average {average_detection_time:.3f}s.")
    print(
        "Pose proxy metrics: "
        f"average people per pose update {average_people_per_pose_update:.2f}, "
        f"average valid keypoints per person {average_valid_keypoints_per_person:.2f}, "
        f"average confident-keypoint score {average_keypoint_confidence:.3f}."
    )

    return {
        "frames": float(frame_index),
        "total_elapsed": total_elapsed,
        "average_fps": frame_index / max(total_elapsed, 1e-6),
        "average_frame_time": average_frame_time,
        "min_frame_time": min_frame_time,
        "max_frame_time": max_frame_time,
        "average_skeleton_stage_time": average_skeleton_stage_time,
        "min_skeleton_stage_time": min_skeleton_stage_time,
        "max_skeleton_stage_time": max_skeleton_stage_time,
        "average_detection_time": average_detection_time,
        "average_people_per_pose_update": average_people_per_pose_update,
        "average_valid_keypoints_per_person": average_valid_keypoints_per_person,
        "average_keypoint_confidence": average_keypoint_confidence,
    }


def comparison_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def print_comparison(dfine_stats: dict[str, float], yolo_stats: dict[str, float]) -> None:
    print("\nComparison summary")
    print("------------------")
    metrics = (
        ("Average FPS", "average_fps"),
        ("Average frame time", "average_frame_time"),
        ("Average box-detection time", "average_detection_time"),
        ("Average skeleton-stage time", "average_skeleton_stage_time"),
        ("Average people per pose update", "average_people_per_pose_update"),
        ("Average valid keypoints per person", "average_valid_keypoints_per_person"),
        ("Average confident-keypoint score", "average_keypoint_confidence"),
    )
    for label, key in metrics:
        print(f"{label}: D-FINE={dfine_stats[key]:.3f} | YOLO={yolo_stats[key]:.3f}")


def write_comparison_report(report_path: Path, dfine_stats: dict[str, float], yolo_stats: dict[str, float]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["box_source", *dfine_stats.keys()]
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"box_source": "dfine", **dfine_stats})
        writer.writerow({"box_source": "yolo", **yolo_stats})
    print(f"Wrote comparison report to {report_path}")


def main() -> None:
    args = parse_args()
    if args.box_source == "compare":
        dfine_stats = run_pipeline(args, "dfine", comparison_output_path(args.output, "dfine"))
        yolo_stats = run_pipeline(args, "yolo", comparison_output_path(args.output, "yolo"))
        print_comparison(dfine_stats, yolo_stats)
        write_comparison_report(args.comparison_report, dfine_stats, yolo_stats)
    else:
        run_pipeline(args, args.box_source, args.output)


if __name__ == "__main__":
    main()
