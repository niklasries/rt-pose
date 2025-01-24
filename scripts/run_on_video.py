import torch
import moviepy
import argparse
import numpy as np
import supervision as sv

from tqdm import tqdm
from rt_pose import PoseEstimationPipeline, PoseEstimationOutput


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
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input video")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output video")
    parser.add_argument(
        "-od",
        "--object-detection",
        type=str,
        default="PekingU/rtdetr_r50vd_coco_o365",
        help="Object detection checkpoint",
    )
    parser.add_argument(
        "-pe",
        "--pose-estimation",
        type=str,
        default="usyd-community/vitpose-plus-small",
        help="Pose estimation checkpoint",
    )
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("-t", "--dtype", type=str, default="bfloat16", help="Data type to run on")
    parser.add_argument("--compile", action="store_true", help="Compile models in the pipeline")
    args = parser.parse_args()

    # Validate data type
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if args.dtype not in dtypes:
        raise ValueError(f"Invalid data type: {args.dtype}. Valid types are: {dtypes.keys()}")

    dtype = dtypes[args.dtype]

    # Load video
    clip = moviepy.VideoFileClip(args.input)

    # Load pose estimation pipeline
    pipeline = PoseEstimationPipeline(
        object_detection_checkpoint=args.object_detection,
        pose_estimation_checkpoint=args.pose_estimation,
        device=args.device,
        dtype=dtype,
        compile=args.compile,
    )
    if args.compile:
        # This will warmup the model for batch_size 0..10
        pipeline.warmup(max_num_persons=10)

    # Run pose estimation on video
    annotated_frames = []
    for frame in tqdm(clip.iter_frames(), total=clip.n_frames):
        output = pipeline(frame)
        annotated_frame = visualize_output(frame, output, confidence=0.3)
        annotated_frames.append(annotated_frame)

    # Save annotated frames as video with the same audio from clip
    annotated_clip = moviepy.ImageSequenceClip(annotated_frames, fps=clip.fps)
    annotated_clip.audio = clip.audio

    dst_path = args.path.replace(".mp4", "_annotated.mp4")
    # dst_path = args.output
    annotated_clip.write_videofile(dst_path)
