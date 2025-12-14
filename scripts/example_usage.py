#!/usr/bin/env python3
"""
Example usage of the Video Identity Transformation System.

This script demonstrates how to use the system programmatically.
"""

import sys
sys.path.insert(0, '..')

import cv2
import numpy as np
from pathlib import Path

from main import VideoIdentityTransformer
from modules.transformation import TargetIdentity


def example_basic_usage():
    """Basic usage with configuration file."""
    print("=== Basic Usage Example ===\n")
    
    # Initialize pipeline with config
    pipeline = VideoIdentityTransformer('configs/transform_config.yaml')
    
    # Process video
    output = pipeline.process_video(
        video_path='input_video.mp4',
        output_path='output/transformed.mp4'
    )
    
    print(f"Output saved to: {output}")


def example_custom_targets():
    """Usage with custom target identities."""
    print("=== Custom Targets Example ===\n")
    
    # Initialize pipeline
    pipeline = VideoIdentityTransformer('configs/transform_config.yaml')
    
    # Load reference images for target identities
    target1_img = cv2.imread('targets/new_person1.jpg')
    target2_img = cv2.imread('targets/new_person2.jpg')
    
    # Create target identities
    identity1 = TargetIdentity(
        identity_id="person_0",
        reference_images=[target1_img],
        gender="female",
        age_range=(25, 35),
        ethnicity="Asian"
    )
    
    identity2 = TargetIdentity(
        identity_id="person_1", 
        reference_images=[target2_img],
        gender="male",
        age_range=(40, 50),
        ethnicity="African"
    )
    
    # Set target identities for tracked persons
    # Person with track_id=0 will be transformed to identity1
    # Person with track_id=1 will be transformed to identity2
    pipeline.transformer.set_target_identity(0, identity1)
    pipeline.transformer.set_target_identity(1, identity2)
    
    # Process video
    output = pipeline.process_video('input_video.mp4')
    print(f"Output saved to: {output}")


def example_text_based_targets():
    """Usage with text-based identity descriptions."""
    print("=== Text-Based Targets Example ===\n")
    
    pipeline = VideoIdentityTransformer('configs/transform_config.yaml')
    
    # Create identities from text descriptions
    identity1 = TargetIdentity(
        identity_id="person_0",
        text_description="A 30 year old Caucasian woman with blonde hair, blue eyes, wearing professional attire"
    )
    
    identity2 = TargetIdentity(
        identity_id="person_1",
        text_description="A 45 year old Hispanic man with short gray hair and a beard, wearing casual clothes"
    )
    
    pipeline.transformer.set_target_identity(0, identity1)
    pipeline.transformer.set_target_identity(1, identity2)
    
    output = pipeline.process_video('input_video.mp4')
    print(f"Output saved to: {output}")


def example_frame_by_frame():
    """Process video frame by frame for more control."""
    print("=== Frame-by-Frame Processing Example ===\n")
    
    from modules import (
        MultiPersonTracker,
        PoseExpressionExtractor,
        PersonSegmenter,
        IdentityTransformer,
        SceneCompositor
    )
    
    # Initialize modules individually
    tracker = MultiPersonTracker({'model': 'yolov8x', 'device': 'cuda'})
    pose_extractor = PoseExpressionExtractor({'device': 'cuda'})
    segmenter = PersonSegmenter({'model': 'sam_vit_h', 'device': 'cuda'})
    transformer = IdentityTransformer({'device': 'cuda'})
    compositor = SceneCompositor({})
    
    # Load video
    cap = cv2.VideoCapture('input_video.mp4')
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Process each frame
    processed_frames = []
    
    for idx, frame in enumerate(frames):
        print(f"Processing frame {idx + 1}/{len(frames)}")
        
        # Track persons
        detections = tracker.process_frame(frame, idx)
        
        # Process each detected person
        for person in detections.persons:
            # Extract pose and expression
            pose_data = pose_extractor.extract_person_data(
                frame, person.bbox, person.track_id, idx
            )
            
            # Segment person
            mask = segmenter.segment_person(
                frame, person.bbox, person.track_id, idx
            )
            
            # Transform identity (if target is set)
            result = transformer.transform_person(
                frame, mask.mask, person.bbox, 
                person.track_id, idx
            )
            
        processed_frames.append(frame)
    
    print(f"Processed {len(processed_frames)} frames")


if __name__ == "__main__":
    print("Video Identity Transformation - Examples\n")
    print("Choose an example to run:")
    print("1. Basic usage")
    print("2. Custom target identities")
    print("3. Text-based targets")
    print("4. Frame-by-frame processing")
    
    # For demonstration, just show the code
    print("\nSee the source code for implementation details.")

