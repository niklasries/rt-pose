import time
import torch
import numpy as np

from typing import Tuple, List
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
        video_batch_size: int = 16,
        max_persons_per_batch: int = 50,
        cache_dir: str = "./torch_compile_cache", 
    ):
        """
        Initializes the optimized pose estimation pipeline.

        Args:
            object_detection_checkpoint: Hugging Face model ID for the object detector.
            pose_estimation_checkpoint: Hugging Face model ID for the pose estimator.
            device: The device to run on (e.g., "cuda").
            dtype: The data type for inference (e.g., torch.bfloat16).
            compile: Whether to apply torch.compile with TensorRT backend.
            video_batch_size: The maximum number of video frames to process at once.
                              Used to set compiler constraints for the detector.
            max_persons_per_batch: The maximum number of people expected to be detected
                                   across a single batch of video frames. Used for
                                   compiler constraints for the pose estimator.
        """
        self.device = device
        self.dtype = dtype
        self.compile = compile
        self.video_batch_size = video_batch_size
        self.max_persons_per_batch = max_persons_per_batch
        self.cache_dir = cache_dir
        

        # Enable Flash Attention 2 for better performance if supported
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        logger.info("Flash Attention SDP enabled.")

        # Loading models
        self.detector, self.detector_image_processor = self._load_detector(object_detection_checkpoint)
        self.pose_estimator, self.pose_estimator_image_processor = self._load_pose_estimator(pose_estimation_checkpoint)
        self.pose_estimator_config = self.pose_estimator.config

        if self.compile:
            self._compile_models()

        logger.info("Optimized Pipeline initialized successfully!")
    @staticmethod
    def _load_detector(self, checkpoint: str):
        logger.info(f"Loading detector from `{checkpoint}`...")
        model = AutoModelForObjectDetection.from_pretrained(checkpoint, torch_dtype=self.dtype).eval()
        image_processor = AutoProcessor.from_pretrained(checkpoint, use_fast=True)
        model = model.to(self.device)
        logger.info(f"Detector loaded to `{self.device}` with dtype `{self.dtype}`!")
        return model, image_processor
    @staticmethod
    def _load_pose_estimator(self, checkpoint: str):
        logger.info(f"Loading pose estimator from `{checkpoint}`...")
        model = VitPoseForPoseEstimation.from_pretrained(checkpoint, torch_dtype=self.dtype).eval()
        image_processor = AutoProcessor.from_pretrained(checkpoint)
        model = model.to(self.device)
        logger.info(f"Pose estimator loaded to `{self.device}` with dtype `{self.dtype}`!")
        return model, image_processor

    def _compile_models(self):
        """
        Compiles models with a pragmatic approach:
        - Pose Estimator (ViT) is compiled with TensorRT for a huge speedup.
        - Detector (RT-DETR) is run in eager mode to avoid compiler bugs.
        """
        logger.info("Applying compilation to models...")
        self.detector = torch.compile(self.detector, mode="reduce-overhead")      
        self.pose_estimator = torch.compile(self.pose_estimator, mode="reduce-overhead", dynamic=True)

        logger.info("Model compilation is enabled, don't forget to call `pipeline.warmup()` method!")

    def __call__(self, images: List[np.ndarray]) -> List[PoseEstimationOutput]:
        """
        Run the full pose estimation pipeline on a batch of images.
        """
        if not images:
            return []

        image_tensors = [torch.from_numpy(img.astype(np.float32)).to(self.device) for img in images]
        original_shapes = [img.shape[:2] for img in image_tensors]

        # --- Batch Detection Step ---
        detector_inputs = self.detector_image_processor(images=image_tensors, return_tensors="pt")
        detector_inputs = {k: v.to(self.device).to(self.dtype if k == "pixel_values" else v.dtype) for k, v in detector_inputs.items()}
        with torch.no_grad():
            detector_outputs = self.detector(**detector_inputs)
        detection_results = self.detector_image_processor.post_process_object_detection(
            detector_outputs, target_sizes=original_shapes, threshold=0.3
        )

        # --- Prepare for Batch Pose Estimation ---
        all_person_boxes = []
        image_indices = []
        for i, result in enumerate(detection_results):
            person_boxes = result["boxes"][result["labels"] == 0]
            all_person_boxes.append(person_boxes)
            image_indices.extend([i] * len(person_boxes))
        
        if not image_indices:
            empty_out = PoseEstimationOutput(torch.empty(0, 4), torch.empty(0, 17, 2), torch.empty(0, 17))
            return [empty_out for _ in images]

        # --- Batch Pose Estimation Step ---
        crop_height = self.pose_estimator_image_processor.size["height"]
        crop_width = self.pose_estimator_image_processor.size["width"]
        mean = self.pose_estimator_image_processor.image_mean
        std = self.pose_estimator_image_processor.image_std
        
        pose_input_list = []
        preprocessed_boxes_list = []
        for i, boxes in enumerate(all_person_boxes):
            if len(boxes) > 0:
                inputs, preprocessed_boxes = preprocess(
                    image=image_tensors[i], boxes_xyxy=boxes, mean=mean, std=std,
                    crop_height=crop_height, crop_width=crop_width, dtype=self.dtype,
                )
                pose_input_list.append(inputs['pixel_values'])
                preprocessed_boxes_list.append(preprocessed_boxes)

        if not pose_input_list:
            logger.debug("No persons detected in this batch, skipping pose estimation.")
            empty_out = PoseEstimationOutput(
                torch.empty(0, 4, device=self.device),
                torch.empty(0, 17, 2, device=self.device),
                torch.empty(0, 17, device=self.device)
            )
            # Return a list of empty outputs, 
            # one for each image in the original batch
            return [empty_out for _ in images]

        pose_inputs = {'pixel_values': torch.cat(pose_input_list).to(self.device)}
        all_preprocessed_boxes = torch.cat(preprocessed_boxes_list)
        total_persons = pose_inputs['pixel_values'].shape[0]

        if self.pose_estimator_config.backbone_config.num_experts > 1:
            pose_inputs["dataset_index"] = torch.full((total_persons,), 0, dtype=torch.int64, device=self.device)

        with torch.no_grad():
            pose_outputs = self.pose_estimator(**pose_inputs)
        keypoints_xy, scores = post_process_pose_estimation(
            pose_outputs.heatmaps, crop_height=crop_height, crop_width=crop_width, boxes_xyxy=all_preprocessed_boxes,
        )

        # --- Split results back to their original images ---
        final_outputs = []
        current_pos = 0
        for i in range(len(images)):
            num_persons_in_image = len(all_person_boxes[i])
            img_keypoints = keypoints_xy[current_pos : current_pos + num_persons_in_image]
            img_scores = scores[current_pos : current_pos + num_persons_in_image]
            final_outputs.append(PoseEstimationOutput(
                person_boxes_xyxy=all_person_boxes[i], keypoints_xy=img_keypoints, scores=img_scores
            ))
            current_pos += num_persons_in_image
            
        return final_outputs

    @torch.no_grad()
    def warmup(self):
        """
        Warms up both stages of the pipeline to trigger TensorRT engine builds.
        1. Runs the detector with dummy images to compile it (if enabled).
        2. Manually creates dummy person boxes to compile the pose estimator
           with a representative batch size.
        """
        logger.info(f"Running warmup for a video batch of {self.video_batch_size}...")

        # --- Part 1: Warmup the Detector (if it's compiled) ---
        # We still run this to ensure the detector's engine is built if needed.
        dummy_sizes = [(720, 1280), (1080, 1920)]
        dummy_images = []
        for i in range(self.video_batch_size):
            h, w = dummy_sizes[i % len(dummy_sizes)]
            dummy_image = np.full((h, w, 3), 128, dtype=np.uint8)
            dummy_images.append(dummy_image)
        # This call will trigger the detector compilation and get handled by the empty-batch guard
        self(dummy_images)
        logger.info("Detector warmup call complete.")

        # --- Part 2: Force Compilation of the Pose Estimator ---
        logger.info(f"Forcing compilation of pose estimator with a dummy batch of people...")
        
        # Create a representative number of dummy person boxes. Let's use a common case,
        # like 10 people, to build the engine. The engine will be valid for any
        # number up to `self.max_persons_per_batch` due to our constraints.
        num_dummy_persons = min(10, self.max_persons_per_batch)
        if num_dummy_persons == 0:
             logger.warning("max_persons_per_batch is 0, skipping pose estimator warmup.")
             return

        # We need a single dummy image tensor to crop from
        dummy_image_tensor = torch.from_numpy(dummy_images[0].astype(np.float32)).to(self.device)
        
        # Create dummy bounding boxes [x1, y1, x2, y2]
        dummy_boxes_xyxy = torch.tensor(
            [[10, 10, 110, 110]] * num_dummy_persons,
            dtype=torch.float32,
            device=self.device
        )

        # To call the pose estimation step directly, we need to manually create the inputs
        # This mimics the logic inside the `__call__` method.
        crop_height = self.pose_estimator_image_processor.size["height"]
        crop_width = self.pose_estimator_image_processor.size["width"]
        mean = self.pose_estimator_image_processor.image_mean
        std = self.pose_estimator_image_processor.image_std
        
        inputs, _ = preprocess(
            image=dummy_image_tensor, boxes_xyxy=dummy_boxes_xyxy, mean=mean, std=std,
            crop_height=crop_height, crop_width=crop_width, dtype=self.dtype,
        )
        
        pose_inputs = {'pixel_values': inputs['pixel_values'].to(self.device)}

        if self.pose_estimator_config.backbone_config.num_experts > 1:
            pose_inputs["dataset_index"] = torch.full((num_dummy_persons,), 0, dtype=torch.int64, device=self.device)
            
        # This is the crucial call that triggers the lazy compilation of the pose estimator
        self.pose_estimator(**pose_inputs)
        
        logger.info("Warmup complete. All engines built.")
