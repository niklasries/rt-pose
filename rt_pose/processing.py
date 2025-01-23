"""
Fast preprocessing and postprocessing for VitPose models.
"""

import torch
import torchvision
import torchvision.ops

from typing import Tuple, Dict


def preprocess_boxes(
    boxes_xyxy: torch.Tensor,
    crop_height: int = 256,
    crop_width: int = 192,
    padding_factor: float = 1.25,
) -> torch.Tensor:
    """
    First, align boxes aspect ratio with respect to the maximum crop dimension,
    then expand it by padding factor.

    Args:
        boxes_xyxy: Bounding boxes in xyxy format.
        crop_height: Height of the crop.
        crop_width: Width of the crop.
        padding_factor: Factor to expand the box by.

    Returns:
        Processed boxes in xyxy format.

    Note:
        Boxes `xyxy` format: [x_min, y_min, x_max, y_max]
    """

    # We will expand box to preserve aspect ratio
    # of desired crop size
    aspect_ratio = crop_width / crop_height
    x_min, y_min, x_max, y_max = boxes_xyxy.T

    # Get center coords and width/height
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Align with respect to aspect ratio
    height = torch.where(width > height * aspect_ratio, width * 1 / aspect_ratio, height)
    width = torch.where(width < height * aspect_ratio, height * aspect_ratio, width)

    # Expand by padding factor
    height = height * padding_factor
    width = width * padding_factor

    # Get new coords
    x_min = x_center - width / 2
    y_min = y_center - height / 2

    return torch.stack([x_min, y_min, x_min + width, y_min + height], dim=1)


def preprocess(
    image: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    crop_height: int = 256,
    crop_width: int = 192,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    scale: float = 1 / 255.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Prepare image and boxes for the model.

    Args:
        image: Image to preprocess.
        boxes_xyxy: Bounding boxes in xyxy format.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
        crop_height: Height of the crop.
        crop_width: Width of the crop.
        dtype: Data type for the output tensors.
        device: Device to use for the output tensors.

    Returns:
        Processed image and boxes.
    """

    if not isinstance(image, torch.Tensor):
        raise ValueError("Image must be a torch.Tensor")

    if not image.ndim == 3:
        raise ValueError("Image must be a 3D tensor with shape (H, W, C)")

    # preprocess boxes
    boxes_xyxy = preprocess_boxes(boxes_xyxy, crop_height, crop_width)
    boxes_xyxy = boxes_xyxy.round().int()

    image = image.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW

    # crop boxes from image
    # upcast to float32 because roi_align is not supported for bfloat16
    image = image.to(torch.float32)
    boxes_xyxy = boxes_xyxy.to(torch.float32)
    crops = torchvision.ops.roi_align(image, [boxes_xyxy], (crop_height, crop_width), 1)
    crops = crops.to(dtype)

    # normalize
    mean_tensor = torch.tensor(mean, dtype=crops.dtype, device=crops.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=crops.dtype, device=crops.device).view(1, 3, 1, 1)
    crops = (crops * scale - mean_tensor) / std_tensor

    model_inputs = {"pixel_values": crops}

    return model_inputs, boxes_xyxy


def post_process_pose_estimation(
    heatmaps: torch.Tensor,
    crop_height: int,
    crop_width: int,
    boxes_xyxy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Postprocess heatmaps to get keypoint coordinates and scores.

    Args:
        heatmaps: Heatmaps from the model with shape (batch_size, num_keypoints, height, width).
        crop_height: Height of the crop (image passed to the pose estimation model).
        crop_width: Width of the crop (image passed to the pose estimation model).
        boxes_xyxy: Crop bounding boxes in xyxy (x_min, y_min, x_max, y_max) format.

    Returns:
        Keypoint coordinates and scores, where:
        - keypoints: Keypoint coordinates with shape (batch_size, num_keypoints, 2) in (x, y) format
        - scores: Keypoint scores with shape (batch_size, num_keypoints)
    """

    batch_size, num_keypoints, _, _ = heatmaps.shape

    # heatmaps are low resolution, we upsample it to get better estimate of keypoint coordinates
    heatmaps = torch.nn.functional.interpolate(
        heatmaps, size=(crop_height, crop_width), mode="bilinear", align_corners=True
    )

    # get keypoint coordinates and scores
    flattened_heatmaps = heatmaps.reshape(batch_size, num_keypoints, -1)
    scores, indices = torch.max(flattened_heatmaps, dim=-1)
    keypoints_x = indices % crop_width
    keypoints_y = indices // crop_width

    # scale coordinates back to original image size
    box_x1, box_y1, box_x2, box_y2 = boxes_xyxy.split(1, dim=-1)
    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1
    keypoints_x = keypoints_x.float() * box_width / crop_width + box_x1
    keypoints_y = keypoints_y.float() * box_height / crop_height + box_y1

    keypoints_xy = torch.stack([keypoints_x, keypoints_y], dim=-1)

    return keypoints_xy, scores
