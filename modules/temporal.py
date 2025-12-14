"""
Temporal Consistency Module

Ensures smooth transitions between frames and reduces flickering
in the transformed video output.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Deque
from collections import deque
import cv2
import torch


@dataclass
class TemporalFrame:
    """Frame data for temporal processing."""
    frame_idx: int
    frame: np.ndarray
    flow_forward: Optional[np.ndarray] = None
    flow_backward: Optional[np.ndarray] = None


class TemporalConsistencyProcessor:
    """
    Applies temporal consistency to ensure smooth video output.
    Uses optical flow for motion-aware smoothing.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enable', True)
        self.method = config.get('method', 'optical_flow')
        self.flow_model = config.get('flow_model', 'raft')
        self.smoothing_window = config.get('smoothing_window', 5)
        self.feature_matching = config.get('feature_matching', True)
        self.color_propagation = config.get('color_propagation', True)
        
        self.device = config.get('device', 'cuda')
        self.optical_flow_model = None
        
        # Frame buffer for temporal smoothing
        self.frame_buffer: Deque[TemporalFrame] = deque(maxlen=self.smoothing_window)
        self.processed_frames: List[np.ndarray] = []
        
    def initialize(self):
        """Initialize optical flow model."""
        if self.method == 'optical_flow':
            self._init_optical_flow()
        print("Temporal consistency processor initialized")
        
    def _init_optical_flow(self):
        """Initialize RAFT optical flow model."""
        try:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            
            self.optical_flow_model = raft_large(
                weights=Raft_Large_Weights.DEFAULT
            ).to(self.device).eval()
            print("RAFT optical flow model loaded")
        except ImportError:
            print("RAFT not available, using OpenCV optical flow")
            self.optical_flow_model = None
            
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Process a single frame with temporal consistency.
        
        Args:
            frame: BGR frame to process
            frame_idx: Frame index
            
        Returns:
            Temporally consistent frame
        """
        if not self.enabled:
            return frame
            
        if self.optical_flow_model is None and self.method == 'optical_flow':
            self.initialize()
            
        # Add to buffer
        temporal_frame = TemporalFrame(frame_idx=frame_idx, frame=frame.copy())
        
        # Compute optical flow if we have previous frames
        if len(self.frame_buffer) > 0:
            prev_frame = self.frame_buffer[-1].frame
            temporal_frame.flow_backward = self._compute_flow(frame, prev_frame)
            self.frame_buffer[-1].flow_forward = self._compute_flow(prev_frame, frame)
            
        self.frame_buffer.append(temporal_frame)
        
        # Apply temporal smoothing
        if len(self.frame_buffer) >= 3:
            smoothed = self._apply_temporal_smoothing()
        else:
            smoothed = frame
            
        return smoothed
    
    def _compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Compute optical flow between two frames."""
        if self.optical_flow_model is not None:
            return self._compute_flow_raft(frame1, frame2)
        else:
            return self._compute_flow_opencv(frame1, frame2)
            
    def _compute_flow_raft(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Compute optical flow using RAFT."""
        # Prepare images
        img1 = torch.from_numpy(frame1).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img2 = torch.from_numpy(frame2).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        with torch.no_grad():
            flow = self.optical_flow_model(img1, img2)[-1]
            
        return flow.squeeze().permute(1, 2, 0).cpu().numpy()
    
    def _compute_flow_opencv(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Compute optical flow using OpenCV Farneback."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        return flow
    
    def _apply_temporal_smoothing(self) -> np.ndarray:
        """Apply temporal smoothing using optical flow."""
        center_idx = len(self.frame_buffer) // 2
        center_frame = self.frame_buffer[center_idx]
        
        # Collect warped frames
        warped_frames = [center_frame.frame.astype(float)]
        weights = [1.0]
        
        # Warp previous frames forward
        for i in range(center_idx - 1, -1, -1):
            prev_frame = self.frame_buffer[i]
            if prev_frame.flow_forward is not None:
                warped = self._warp_frame(prev_frame.frame, prev_frame.flow_forward)
                warped_frames.append(warped.astype(float))
                weights.append(0.5 ** (center_idx - i))
                
        # Warp future frames backward
        for i in range(center_idx + 1, len(self.frame_buffer)):
            next_frame = self.frame_buffer[i]
            if next_frame.flow_backward is not None:
                warped = self._warp_frame(next_frame.frame, next_frame.flow_backward)
                warped_frames.append(warped.astype(float))
                weights.append(0.5 ** (i - center_idx))
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        smoothed = np.zeros_like(center_frame.frame, dtype=float)
        
        for frame, weight in zip(warped_frames, weights):
            if frame.shape == smoothed.shape:
                smoothed += frame * weight

        return np.clip(smoothed, 0, 255).astype(np.uint8)

    def _warp_frame(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp frame using optical flow."""
        h, w = frame.shape[:2]

        # Create mesh grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Apply flow
        map_x = (x + flow[:, :, 0]).astype(np.float32)
        map_y = (y + flow[:, :, 1]).astype(np.float32)

        # Remap
        warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

        return warped

    def process_video(self, frames: List[np.ndarray],
                     progress_callback=None) -> List[np.ndarray]:
        """
        Process entire video with temporal consistency.

        Args:
            frames: List of BGR frames
            progress_callback: Optional callback(current, total)

        Returns:
            List of temporally consistent frames
        """
        self.reset()
        processed = []

        for idx, frame in enumerate(frames):
            result = self.process_frame(frame, idx)
            processed.append(result)

            if progress_callback:
                progress_callback(idx + 1, len(frames))

        # Process remaining frames in buffer
        while len(self.frame_buffer) > 1:
            self.frame_buffer.popleft()
            if len(self.frame_buffer) >= 1:
                processed.append(self.frame_buffer[0].frame)

        return processed

    def apply_color_propagation(self, frames: List[np.ndarray],
                                keyframe_indices: List[int]) -> List[np.ndarray]:
        """
        Propagate colors from keyframes to other frames.
        Useful for maintaining consistent colors across the video.
        """
        if not self.color_propagation:
            return frames

        result = frames.copy()

        for i, frame in enumerate(frames):
            if i in keyframe_indices:
                continue

            # Find nearest keyframes
            prev_key = max([k for k in keyframe_indices if k < i], default=None)
            next_key = min([k for k in keyframe_indices if k > i], default=None)

            if prev_key is not None and next_key is not None:
                # Interpolate between keyframes
                alpha = (i - prev_key) / (next_key - prev_key)
                result[i] = self._blend_colors(
                    frame, frames[prev_key], frames[next_key], alpha
                )
            elif prev_key is not None:
                result[i] = self._transfer_colors(frame, frames[prev_key])
            elif next_key is not None:
                result[i] = self._transfer_colors(frame, frames[next_key])

        return result

    def _blend_colors(self, frame: np.ndarray,
                     ref1: np.ndarray,
                     ref2: np.ndarray,
                     alpha: float) -> np.ndarray:
        """Blend colors from two reference frames."""
        # Convert to LAB
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
        ref1_lab = cv2.cvtColor(ref1, cv2.COLOR_BGR2LAB).astype(float)
        ref2_lab = cv2.cvtColor(ref2, cv2.COLOR_BGR2LAB).astype(float)

        # Interpolate color channels (a and b)
        for c in [1, 2]:
            ref_mean = (1 - alpha) * ref1_lab[:, :, c].mean() + alpha * ref2_lab[:, :, c].mean()
            frame_mean = frame_lab[:, :, c].mean()
            frame_lab[:, :, c] += (ref_mean - frame_mean)

        frame_lab = np.clip(frame_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)

    def _transfer_colors(self, frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Transfer colors from reference to frame."""
        return self._blend_colors(frame, reference, reference, 0.5)

    def reset(self):
        """Reset processor state for new video."""
        self.frame_buffer.clear()
        self.processed_frames = []

