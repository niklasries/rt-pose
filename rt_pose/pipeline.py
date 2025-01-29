import time
import torch
import numpy as np

from typing import Tuple
from loguru import logger
from dataclasses import dataclass
from transformers import (
    AutoProcessor,
    AutoModelForObjectDetection,
    VitPoseForPoseEstimation,
)

from rt_pose.processing import preprocess, post_process_pose_estimation


@dataclass
class PoseEstimationOutput:
    """
    Output of the pose estimation pipeline.

    Attributes:
        person_boxes_xyxy: Detected person boxes in format (N, 4) where N is the number of detected persons.
            Boxes are in format (x_min, y_min, x_max, y_max) a.k.a. Pascal VOC format.
        keypoints_xy: Keypoints in format (N, 17, 2) where N is the number of detected persons
            and each keypoint is represented by (x, y) coordinates.
        scores: Scores in format (N, 17) where each score is a confidence score for the corresponding keypoint.
    """

    person_boxes_xyxy: torch.Tensor
    keypoints_xy: torch.Tensor
    scores: torch.Tensor


class PoseEstimationPipeline:
    def __init__(
        self,
        object_detection_checkpoint: str,
        pose_estimation_checkpoint: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        compile: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.compile = compile
        self.object_detection_checkpoint = object_detection_checkpoint
        self.pose_estimation_checkpoint = pose_estimation_checkpoint

        # Loading detection model
        model, image_processor = self._load_detector(object_detection_checkpoint, device, dtype)
        self.detector = model
        self.detector_config = model.config
        self.detector_image_processor = image_processor

        # Loading pose estimation model
        model, image_processor = self._load_pose_estimator(pose_estimation_checkpoint, device, dtype)
        self.pose_estimator = model
        self.pose_estimator_config = model.config
        self.pose_estimator_image_processor = image_processor

        # Compiling models
        if self.compile:
            self._compile_models()

        logger.info("Pipeline initialized successfully!")

    @staticmethod
    def _load_detector(checkpoint: str, device: str, dtype: torch.dtype):
        logger.info(f"Loading detector from `{checkpoint}`...")
        model = AutoModelForObjectDetection.from_pretrained(checkpoint, torch_dtype=dtype)
        image_processor = AutoProcessor.from_pretrained(checkpoint, use_fast=True)
        model = model.to(device)
        logger.info(f"Detector loaded to `{device}` with dtype `{dtype}`!")
        return model, image_processor

    @staticmethod
    def _load_pose_estimator(checkpoint: str, device: str, dtype: torch.dtype):
        logger.info(f"Loading pose estimator from `{checkpoint}`...")
        model = VitPoseForPoseEstimation.from_pretrained(checkpoint, torch_dtype=dtype)
        image_processor = AutoProcessor.from_pretrained(checkpoint)
        model = model.to(device)
        logger.info(f"Pose estimator loaded to `{device}` with dtype `{dtype}`!")
        return model, image_processor

    def _compile_models(self):
        logger.info("Applying compilation to models...")
        self.detector = torch.compile(self.detector, mode="reduce-overhead")
        self.pose_estimator = torch.compile(self.pose_estimator, mode="reduce-overhead", dynamic=True)
        logger.info("Model compilation is enabled, don't forget to call `pipeline.warmup()` method!")

    def _run_detection_step(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run the detection step of the pipeline. Detects person boxes in the image.

        Args:
            image: RGB image with shape (H, W, 3) to run detection on.

        Returns:
            Detected person boxes in format (N, 4) where N is the number of detected persons.
            Boxes are in format (x_min, y_min, x_max, y_max) a.k.a. Pascal VOC format.
        """
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Image must be a 3-channel RGB image with shape (H, W, 3)")

        # Preprocess image
        detector_inputs = self.detector_image_processor(images=image, device=self.device, return_tensors="pt")
        detector_inputs = detector_inputs.to(self.device).to(self.dtype)

        # Run detection
        with torch.no_grad():
            outputs = self.detector(**detector_inputs)

        # Postprocess detection results, extract boxes and labels from logits
        height, width = image.shape[:2]
        detection_results = self.detector_image_processor.post_process_object_detection(
            outputs, target_sizes=[(height, width)], threshold=0.3
        )
        image_detections = detection_results[0]  # take first image results

        # Human label refers 0 index in COCO dataset
        person_boxes_xyxy = image_detections["boxes"][image_detections["labels"] == 0]

        return person_boxes_xyxy

    def _run_pose_estimation_step(
        self, image: torch.Tensor, person_boxes_xyxy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run pose estimation step of the pipeline. Estimates keypoints for detected person boxes.

        Args:
            image: RGB image with shape (H, W, 3) to run pose estimation on.
            person_boxes_xyxy: Detected person boxes in format (N, 4) where N is the number of detected persons.
                Boxes are in format (x_min, y_min, x_max, y_max) a.k.a. Pascal VOC format.

        Returns:
            Keypoints and corresponding scores, where:
                - Keypoints in format (N, 17, 2) where N is the number of detected persons
                  and each keypoint is represented by (x, y) coordinates.
                - Scores in format (N, 17) where each score is a confidence score for the corresponding keypoint.
        """

        # Original preprocessing by VitPoseImagProcessor is slow for real time
        # we use custom preprocessing which is not guaranteed to be exactly the same
        # but it is faster and works well for real time applications.
        crop_height = self.pose_estimator_image_processor.size["height"]
        crop_width = self.pose_estimator_image_processor.size["width"]
        mean = self.pose_estimator_image_processor.image_mean
        std = self.pose_estimator_image_processor.image_std

        inputs, preprocessed_boxes_xyxy = preprocess(
            image=image,
            boxes_xyxy=person_boxes_xyxy,
            mean=mean,
            std=std,
            crop_height=crop_height,
            crop_width=crop_width,
            dtype=self.dtype,
        )

        # Dataset index is required for ViTPose++ models, because this architecture uses MoE layers.
        # We specify which "Expert" to use for each image in batch.
        # `0`` index indicates to use expert trained on "COCO" dataset.
        if self.pose_estimator_config.backbone_config.num_experts > 1:
            batch_size = person_boxes_xyxy.shape[0]
            inputs["dataset_index"] = torch.full((batch_size,), 0, dtype=torch.int64, device=self.device)

        # Run pose estimation
        with torch.no_grad():
            outputs = self.pose_estimator(**inputs)

        # Postprocess pose estimation results
        keypoints_xy, scores = post_process_pose_estimation(
            outputs.heatmaps,
            crop_height=crop_height,
            crop_width=crop_width,
            boxes_xyxy=preprocessed_boxes_xyxy,
        )

        return keypoints_xy, scores

    def __call__(self, image: np.ndarray) -> PoseEstimationOutput:
        """
        Run pose estimation pipeline on the image.

        Args:
            image: RGB image with shape (H, W, 3) to run pose estimation on.

        Returns:
            PipelineOutput object with keypoints and corresponding scores.
        """
        image_tensor = torch.from_numpy(image.astype(np.float32))
        image_tensor = image_tensor.to(self.device)

        person_boxes_xyxy = self._run_detection_step(image_tensor)
        keypoints_xy, scores = self._run_pose_estimation_step(image_tensor, person_boxes_xyxy)

        output = PoseEstimationOutput(
            keypoints_xy=keypoints_xy,
            person_boxes_xyxy=person_boxes_xyxy,
            scores=scores,
        )

        return output

    @torch.no_grad()
    def warmup(self, max_num_persons: int = 30):
        """
        Warmup the pipeline by running it once.
        """

        image_tensor = torch.ones((512, 512, 3), device=self.device, dtype=torch.float32) * 255

        logger.info("Running warmup for object detection step...")
        start_time = time.time()
        for _ in range(10):
            self._run_detection_step(image_tensor)
        end_time = time.time()
        logger.info(f"Object detection step warmup took {end_time - start_time:.2f} seconds")

        logger.info("Running warmup for pose estimation step...")
        start_time = time.time()
        for i in range(max_num_persons):
            boxes_xyxy = torch.tensor([[0, 0, 100, 100]] * (i + 1), device=self.device, dtype=self.dtype)
            self._run_pose_estimation_step(image_tensor, boxes_xyxy)
        end_time = time.time()
        logger.info(f"Pose estimation step warmup took {end_time - start_time:.2f} seconds")
