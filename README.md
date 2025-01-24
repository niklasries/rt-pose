<div align="center">
<h1> RT-Pose </h1>

Real-time (GPU) pose estimation pipeline with 🤗 Transformers

![](https://github.com/qubvel/assets/blob/main/rt_pose_break_dance_v2_annotated.gif)
![](https://github.com/qubvel/assets/blob/main/rt_pose_american_football_annotated.gif)

</div>

## Installation

1. [Optional] It's recommended to run with `uv` for faster installation.
First, install `uv`:

```bash
pip install uv
```

2. Install `rt_pose` (you can ignore `uv` in case you want to install with pure `pip`)

```bash
uv pip install rt-pose        # with minimal dependencies
uv pip install rt-pose[demo]  # with additional dependencies to run `scripts/` and `notebooks/`
```

## Quick start

 - [Python snippet](#python-snippet)
 - [Script to run on image](#run-pose-estimation-on-image)
 - [Script to run on video](#run-pose-estimation-on-video)

### Python snippet

```python
import torch
from rt_pose import PoseEstimationPipeline

# Load pose estimation pipeline
pipeline = PoseEstimationPipeline(
    object_detection_checkpoint="PekingU/rtdetr_r50vd_coco_o365",
    pose_estimation_checkpoint="usyd-community/vitpose-plus-small",
    device="cuda",
    dtype=torch.bfloat16,
    compile=False,  # or True to get more speedup
)

# Run pose estimation on image
output = pipeline(image)

# output.person_boxes_xyxy (`torch.Tensor`): 
#   of shape `(N, 4)` with `N` boxes of detected persons on the image in (x_min, y_min, x_max, y_max) format
# output.keypoints_xy (`torch.Tensor`):
#   of shape `(N, 17, 2)` with 17 keypoints per each person
# output.scores (`torch.Tensor`): 
#   of shape (N, 17) with corresponding scores (aka confidence) for each keypoint

# Visualize with supervision/matplotlib/opencv
# see ./scripts/run_on_image.py
```

Other object detection checkpoints on the Hub:

- [RT-DETR](https://huggingface.co/PekingU)
- [DETR](https://huggingface.co/models?other=detr)
- [YOLOS](https://huggingface.co/models?other=yolos)

Other pose estimation checkpoints on the Hub:

- [ViTPose and ViTPose++](https://huggingface.co/usyd-community)

### Run pose estimation on image

 - `--input` can be URL or path

```bash
python scripts/run_on_image.py \
    --input "https://res-3.cloudinary.com/dostuff-media/image/upload//w_1200,q_75,c_limit,f_auto/v1511369692/page-image-10656-892d1842-b089-4a7a-80f1-5be99b2b3454.png" \
    --output "results/image.png" \
    --device "cuda:0"
```

### Run pose estimation on video

 - `--input` can be URL or path
 - `--dtype` it's recommended to run in `bfloat16` precision to get the best precision/speed tradeoff
 - `--compile` you can compile models in the pipeline to get even more speed up (x2), but compilation can be quite long, so it makes sense 
    to activate for long videos only.

```bash
python scripts/run_on_video.py \
    --input "https://huggingface.co/datasets/qubvel-hf/assets/blob/main/rt_pose_break_dance_v1.mp4" \
    --output "results/rt_pose_break_dance_v1_annotated.mp4" \
    --device "cuda:0" \
    --dtype bfloat16
```
