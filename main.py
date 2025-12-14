"""
Video Identity Transformation Pipeline

Main orchestrator that coordinates all modules to transform
person identities in video while preserving expressions, lip sync,
and realistic interactions.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import cv2
from tqdm import tqdm
import yaml
from omegaconf import OmegaConf

from modules import (
    MultiPersonTracker,
    PoseExpressionExtractor,
    PersonSegmenter,
    IdentityTransformer,
    SceneCompositor,
    TemporalConsistencyProcessor
)
from modules.transformation import TargetIdentity


class VideoIdentityTransformer:
    """
    Main pipeline for video identity transformation.
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._init_modules()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return OmegaConf.create(config)
    
    def _init_modules(self):
        """Initialize all processing modules."""
        print("Initializing modules...")
        
        self.tracker = MultiPersonTracker(
            dict(self.config.get('tracking', {})) | 
            {'device': self.config.processing.device}
        )
        
        self.pose_extractor = PoseExpressionExtractor(
            dict(self.config.get('pose_estimation', {})) |
            dict(self.config.get('facial_analysis', {})) |
            {'device': self.config.processing.device}
        )
        
        self.segmenter = PersonSegmenter(
            dict(self.config.get('segmentation', {})) |
            {'device': self.config.processing.device}
        )
        
        self.transformer = IdentityTransformer(
            dict(self.config.get('transformation', {})) |
            {'device': self.config.processing.device}
        )
        
        self.compositor = SceneCompositor(
            dict(self.config.get('compositing', {}))
        )
        
        self.temporal_processor = TemporalConsistencyProcessor(
            dict(self.config.get('temporal', {})) |
            {'device': self.config.processing.device}
        )
        
        print("All modules initialized")
        
    def load_target_identities(self, targets_config: dict):
        """Load target identities from configuration."""
        method = targets_config.get('method', 'reference_images')
        
        if method == 'reference_images':
            ref_images = targets_config.get('reference_images', {})
            for person_key, image_path in ref_images.items():
                track_id = int(person_key.split('_')[1])
                
                # Load reference image
                ref_img = cv2.imread(image_path)
                if ref_img is None:
                    print(f"Warning: Could not load {image_path}")
                    continue
                    
                identity = TargetIdentity(
                    identity_id=person_key,
                    reference_images=[ref_img]
                )
                self.transformer.set_target_identity(track_id, identity)
                
        elif method == 'text_descriptions':
            descriptions = targets_config.get('text_descriptions', {})
            for person_key, description in descriptions.items():
                track_id = int(person_key.split('_')[1])
                
                identity = TargetIdentity(
                    identity_id=person_key,
                    text_description=description
                )
                self.transformer.set_target_identity(track_id, identity)
                
    def load_video(self, video_path: str) -> tuple:
        """Load video and return frames and metadata."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        start_frame = self.config.input.get('start_frame', 0)
        end_frame = self.config.input.get('end_frame') or total_frames
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"Loading video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {end_frame - start_frame}")
        
        for i in tqdm(range(end_frame - start_frame), desc="Loading frames"):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        
        return frames, {'fps': fps, 'width': width, 'height': height}
    
    def save_video(self, frames: List[np.ndarray], output_path: str, 
                   fps: float, codec: str = 'mp4v'):
        """Save frames to video file."""
        if not frames:
            raise ValueError("No frames to save")
            
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Saving video to: {output_path}")
        for frame in tqdm(frames, desc="Writing frames"):
            writer.write(frame)

        writer.release()
        print(f"Video saved successfully")

    def process_video(self, video_path: str, output_path: str = None):
        """
        Main processing pipeline.

        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
        """
        # Load video
        frames, metadata = self.load_video(video_path)

        if output_path is None:
            output_path = self.config.output.output_path

        # Load target identities
        self.load_target_identities(dict(self.config.targets))

        # Stage 1: Multi-person tracking
        print("\n=== Stage 1: Person Tracking ===")
        all_detections = self.tracker.process_video(
            frames,
            progress_callback=lambda c, t: None
        )

        unique_ids = self.tracker.get_unique_track_ids()
        print(f"Detected {len(unique_ids)} unique persons: {unique_ids}")

        # Stage 2-6: Process each frame
        print("\n=== Stage 2-6: Frame Processing ===")
        processed_frames = []

        for frame_idx, (frame, detections) in enumerate(tqdm(
            zip(frames, all_detections),
            total=len(frames),
            desc="Processing frames"
        )):
            processed = self._process_single_frame(
                frame, detections, frame_idx
            )
            processed_frames.append(processed)

        # Stage 7: Temporal consistency
        print("\n=== Stage 7: Temporal Consistency ===")
        if self.config.temporal.get('enable', True):
            final_frames = self.temporal_processor.process_video(
                processed_frames,
                progress_callback=lambda c, t: None
            )
        else:
            final_frames = processed_frames

        # Save output
        print("\n=== Saving Output ===")
        self.save_video(
            final_frames,
            output_path,
            metadata['fps'],
            self.config.output.get('codec', 'mp4v')
        )

        return output_path

    def _process_single_frame(self, frame: np.ndarray,
                              detections, frame_idx: int) -> np.ndarray:
        """Process a single frame through all stages."""
        if not detections.persons:
            return frame

        transformed_persons = []
        original_masks = []

        for person in detections.persons:
            # Stage 2: Pose and expression extraction
            pose_data = self.pose_extractor.extract_person_data(
                frame, person.bbox, person.track_id, frame_idx
            )

            # Stage 3: Segmentation
            seg_mask = self.segmenter.segment_person(
                frame, person.bbox, person.track_id, frame_idx,
                pose_keypoints=pose_data.body_pose.keypoints_2d if pose_data.body_pose else None
            )

            # Stage 4: Identity transformation
            transform_result = self.transformer.transform_person(
                frame=frame,
                mask=seg_mask.mask,
                bbox=person.bbox,
                track_id=person.track_id,
                frame_idx=frame_idx,
                pose_data=pose_data.body_pose.__dict__ if pose_data.body_pose else None,
                expression_data=pose_data.expression.__dict__ if pose_data.expression else None,
                lip_sync_data=pose_data.lip_sync.__dict__ if pose_data.lip_sync else None
            )

            transformed_persons.append({
                'rgba': transform_result.transformed_person,
                'bbox': person.bbox,
                'track_id': person.track_id
            })
            original_masks.append(seg_mask.mask)

        # Stage 5: Compositing
        composite_result = self.compositor.composite_frame(
            background=frame,
            transformed_persons=transformed_persons,
            original_masks=original_masks
        )

        return composite_result.composited_frame


def main():
    parser = argparse.ArgumentParser(
        description="Video Identity Transformation System"
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Path to input video'
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Path for output video'
    )
    parser.add_argument(
        '--config', '-c', default='configs/transform_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--targets', '-t', nargs='+', default=None,
        help='Target identity reference images (person1.jpg person2.jpg ...)'
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = VideoIdentityTransformer(args.config)

    # Override targets if provided via CLI
    if args.targets:
        for idx, target_path in enumerate(args.targets):
            ref_img = cv2.imread(target_path)
            if ref_img is not None:
                identity = TargetIdentity(
                    identity_id=f"person_{idx}",
                    reference_images=[ref_img]
                )
                pipeline.transformer.set_target_identity(idx, identity)

    # Process video
    output_path = pipeline.process_video(args.input, args.output)
    print(f"\nTransformation complete! Output saved to: {output_path}")


if __name__ == "__main__":
    main()

