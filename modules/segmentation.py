"""
Person Segmentation Module

Segments each tracked person from the background with high precision,
including temporal consistency for smooth masks across frames.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import cv2
import torch


@dataclass
class SegmentationMask:
    """Segmentation mask for a single person."""
    track_id: int
    frame_idx: int
    mask: np.ndarray  # Binary mask (H, W)
    confidence_map: Optional[np.ndarray] = None  # Soft mask (H, W)
    bbox: Optional[np.ndarray] = None
    
    @property
    def area(self) -> int:
        return int(self.mask.sum())
    
    def get_cropped_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get mask cropped to bounding box."""
        if self.bbox is None:
            return self.mask, np.array([0, 0, self.mask.shape[1], self.mask.shape[0]])
        x1, y1, x2, y2 = self.bbox.astype(int)
        return self.mask[y1:y2, x1:x2], self.bbox


class PersonSegmenter:
    """
    High-quality person segmentation using SAM (Segment Anything Model).
    Includes temporal consistency for video processing.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda')
        self.model_type = config.get('model', 'sam_vit_h')
        self.refinement = config.get('refinement', True)
        self.temporal_consistency = config.get('temporal_consistency', True)
        
        self.sam_model = None
        self.sam_predictor = None
        self.previous_masks: Dict[int, np.ndarray] = {}
        
    def initialize(self):
        """Load SAM model."""
        from segment_anything import sam_model_registry, SamPredictor
        
        model_type_map = {
            'sam_vit_b': 'vit_b',
            'sam_vit_l': 'vit_l', 
            'sam_vit_h': 'vit_h'
        }
        
        checkpoint_map = {
            'vit_b': 'models/sam_vit_b_01ec64.pth',
            'vit_l': 'models/sam_vit_l_0b3195.pth',
            'vit_h': 'models/sam_vit_h_4b8939.pth'
        }
        
        model_type = model_type_map.get(self.model_type, 'vit_h')
        checkpoint = checkpoint_map[model_type]
        
        print(f"Loading SAM model: {model_type}")
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam_model.to(self.device)
        self.sam_predictor = SamPredictor(self.sam_model)
        print("SAM model loaded successfully")
        
    def segment_person(self, frame: np.ndarray, 
                       bbox: np.ndarray,
                       track_id: int,
                       frame_idx: int,
                       pose_keypoints: Optional[np.ndarray] = None) -> SegmentationMask:
        """
        Segment a single person from the frame.
        
        Args:
            frame: BGR image
            bbox: Person bounding box [x1, y1, x2, y2]
            track_id: Person track ID
            frame_idx: Frame index
            pose_keypoints: Optional pose keypoints for better prompting
            
        Returns:
            SegmentationMask for the person
        """
        if self.sam_predictor is None:
            self.initialize()
            
        # Set image for SAM
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(rgb_frame)
        
        # Prepare prompts
        input_box = bbox.reshape(1, 4)
        
        # Add point prompts from pose keypoints if available
        input_points = None
        input_labels = None
        if pose_keypoints is not None and len(pose_keypoints) > 0:
            # Use high-confidence keypoints as positive prompts
            valid_kps = pose_keypoints[pose_keypoints[:, 2] > 0.5] if pose_keypoints.shape[1] > 2 else pose_keypoints
            if len(valid_kps) > 0:
                input_points = valid_kps[:, :2].reshape(-1, 2)
                input_labels = np.ones(len(input_points))
        
        # Run SAM prediction
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=True
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.uint8)
        
        # Apply temporal consistency
        if self.temporal_consistency and track_id in self.previous_masks:
            mask = self._apply_temporal_consistency(mask, track_id)
            
        # Refine mask edges
        if self.refinement:
            mask = self._refine_mask(mask, frame)
            
        # Store for temporal consistency
        self.previous_masks[track_id] = mask.copy()
        
        return SegmentationMask(
            track_id=track_id,
            frame_idx=frame_idx,
            mask=mask,
            bbox=bbox
        )
    
    def segment_frame(self, frame: np.ndarray,
                      detections: List[Dict],
                      frame_idx: int) -> List[SegmentationMask]:
        """Segment all persons in a frame."""
        masks = []
        for det in detections:
            mask = self.segment_person(
                frame=frame,
                bbox=det['bbox'],
                track_id=det['track_id'],
                frame_idx=frame_idx,
                pose_keypoints=det.get('keypoints')
            )
            masks.append(mask)
        return masks
    
    def _apply_temporal_consistency(self, mask: np.ndarray, 
                                    track_id: int) -> np.ndarray:
        """Apply temporal smoothing between consecutive frames."""
        prev_mask = self.previous_masks[track_id]
        if prev_mask.shape != mask.shape:
            prev_mask = cv2.resize(prev_mask, (mask.shape[1], mask.shape[0]))
        # Blend with previous mask for smoother transitions
        alpha = 0.7
        blended = (alpha * mask + (1 - alpha) * prev_mask).astype(np.uint8)
        return (blended > 0.5).astype(np.uint8)
    
    def _refine_mask(self, mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Refine mask edges using GrabCut or morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    def reset(self):
        """Reset temporal state for new video."""
        self.previous_masks = {}

