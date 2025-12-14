"""
Multi-Person Tracking Module

Handles detection and tracking of multiple people across video frames,
maintaining consistent identity assignments throughout the video.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import torch


@dataclass
class TrackedPerson:
    """Represents a tracked person across frames."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    frame_idx: int
    features: Optional[np.ndarray] = None  # Re-ID features
    is_visible: bool = True
    facing_camera: bool = True  # Whether person is facing camera
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, 
                (self.bbox[1] + self.bbox[3]) / 2)
    
    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class FrameDetections:
    """All detections for a single frame."""
    frame_idx: int
    persons: List[TrackedPerson] = field(default_factory=list)
    
    def get_person(self, track_id: int) -> Optional[TrackedPerson]:
        for person in self.persons:
            if person.track_id == track_id:
                return person
        return None


class MultiPersonTracker:
    """
    Multi-person detection and tracking using YOLOv8 + ByteTrack.
    Maintains consistent person IDs across the entire video.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda')
        self.model_name = config.get('model', 'yolov8x')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.7)
        self.tracker_type = config.get('tracker', 'bytetrack')
        
        self.model = None
        self.track_history: Dict[int, List[TrackedPerson]] = {}
        self.frame_detections: List[FrameDetections] = []
        
    def initialize(self):
        """Load the detection model."""
        from ultralytics import YOLO
        
        print(f"Loading detection model: {self.model_name}")
        self.model = YOLO(f'{self.model_name}.pt')
        self.model.to(self.device)
        print("Detection model loaded successfully")
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameDetections:
        """
        Process a single frame and return tracked persons.
        
        Args:
            frame: BGR image as numpy array
            frame_idx: Frame index in video
            
        Returns:
            FrameDetections with all tracked persons
        """
        if self.model is None:
            self.initialize()
            
        # Run detection with tracking
        results = self.model.track(
            frame,
            persist=True,
            tracker=f"{self.tracker_type}.yaml",
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[0],  # Only detect persons (COCO class 0)
            verbose=False
        )
        
        frame_dets = FrameDetections(frame_idx=frame_idx)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for bbox, track_id, conf in zip(boxes, track_ids, confidences):
                person = TrackedPerson(
                    track_id=int(track_id),
                    bbox=bbox,
                    confidence=float(conf),
                    frame_idx=frame_idx
                )
                frame_dets.persons.append(person)
                
                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append(person)
        
        self.frame_detections.append(frame_dets)
        return frame_dets
    
    def process_video(self, frames: List[np.ndarray], 
                      progress_callback=None) -> List[FrameDetections]:
        """
        Process entire video and return all frame detections.
        
        Args:
            frames: List of BGR frames
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of FrameDetections for each frame
        """
        self.reset()
        
        for idx, frame in enumerate(frames):
            self.process_frame(frame, idx)
            if progress_callback:
                progress_callback(idx + 1, len(frames))
                
        return self.frame_detections
    
    def get_unique_track_ids(self) -> List[int]:
        """Get all unique person track IDs in the video."""
        return list(self.track_history.keys())
    
    def get_track_timeline(self, track_id: int) -> List[TrackedPerson]:
        """Get the complete timeline for a specific person."""
        return self.track_history.get(track_id, [])
    
    def reset(self):
        """Reset tracker state for new video."""
        self.track_history = {}
        self.frame_detections = []
        if self.model is not None:
            self.model.predictor = None  # Reset tracker state

