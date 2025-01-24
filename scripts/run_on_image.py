"""
Run pose estimation on the image. To run this script you need to install as follows:

```bash
uv pip install rt-pose[demo]
```
"""

import os
import torch
import argparse
import requests
import numpy as np
import supervision as sv

from PIL import Image
from loguru import logger
from rt_pose import PoseEstimationPipeline, PoseEstimationOutput


def load_image(path_or_url: str) -> np.ndarray:
    """
    Load image from path or URL.
    """
    if os.path.exists(path_or_url):
        image = Image.open(path_or_url)
    elif requests.get(path_or_url).status_code == 200:
        image = Image.open(requests.get(path_or_url, stream=True).raw)
    else:
        raise ValueError(f"Failed to load image from `{path_or_url}`")

    return np.array(image)


def visualize_output(image: np.ndarray, output: PoseEstimationOutput, confidence: float = 0.3) -> np.ndarray:
    """
    Visualize pose estimation output.
    """
    keypoints_xy = output.keypoints_xy.float().cpu().numpy()
    scores = output.scores.float().cpu().numpy()

    # Supervision will not draw vertices with `0` score
    # and coordinates with `(0, 0)` value
    invisible_keypoints = scores < confidence
    scores[invisible_keypoints] = 0
    keypoints_xy[invisible_keypoints] = 0

    keypoints = sv.KeyPoints(xy=keypoints_xy, confidence=scores)

    _, y_min, _, y_max = output.person_boxes_xyxy.T
    height = int((y_max - y_min).mean().item())
    radius = max(height // 100, 4)
    thickness = max(height // 200, 2)
    edge_annotator = sv.EdgeAnnotator(color=sv.Color.YELLOW, thickness=thickness)
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.ROBOFLOW, radius=radius)

    annotated_frame = image.copy()
    annotated_frame = edge_annotator.annotate(annotated_frame, keypoints)
    annotated_frame = vertex_annotator.annotate(annotated_frame, keypoints)

    return annotated_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to image or URL to image")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save directory")
    parser.add_argument(
        "-od",
        "--object-detection",
        type=str,
        default="PekingU/rtdetr_r50vd_coco_o365",
        help="Hugging face repository name to object detection checkpoint",
    )
    parser.add_argument(
        "-pe",
        "--pose-estimation",
        type=str,
        default="usyd-community/vitpose-plus-small",
        help="Hugging face repository name to pose estimation checkpoint",
    )
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run pipeline on")
    parser.add_argument("-t", "--dtype", type=str, default="bfloat16", help="Data type to run pipeline on")
    args = parser.parse_args()

    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if args.dtype not in dtypes:
        raise ValueError(f"Invalid data type: {args.dtype}. Valid types are: {dtypes.keys()}")

    dtype = dtypes[args.dtype]

    # Load image
    image = load_image(args.input)

    # Load pose estimation pipeline
    pipeline = PoseEstimationPipeline(
        object_detection_checkpoint=args.object_detection,
        pose_estimation_checkpoint=args.pose_estimation,
        device=args.device,
        dtype=dtype,
        compile=False,
    )

    # Run pose estimation on image
    output = pipeline(image)

    # Visualize output
    logger.info("Visualizing output...")
    annotated_image = visualize_output(image, output, confidence=0.3)

    logger.info(f"Saving output to `{args.output}`...")
    dst_dir = os.path.dirname(args.output)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    Image.fromarray(annotated_image).save(args.output)

    logger.info("Done!")
