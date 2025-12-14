"""
Video utility functions.
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import cv2
import subprocess


def load_video_frames(video_path: str, 
                      start_frame: int = 0,
                      end_frame: Optional[int] = None,
                      max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], dict]:
    """
    Load frames from a video file.
    
    Args:
        video_path: Path to video file
        start_frame: Starting frame index
        end_frame: Ending frame index (exclusive)
        max_frames: Maximum number of frames to load
        
    Returns:
        Tuple of (frames list, metadata dict)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if end_frame is None:
        end_frame = total_frames
    if max_frames is not None:
        end_frame = min(end_frame, start_frame + max_frames)
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
    cap.release()
    
    metadata = {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'loaded_frames': len(frames)
    }
    
    return frames, metadata


def save_video_frames(frames: List[np.ndarray],
                      output_path: str,
                      fps: float = 30.0,
                      codec: str = 'mp4v') -> str:
    """
    Save frames to a video file.
    
    Args:
        frames: List of BGR frames
        output_path: Output video path
        fps: Frames per second
        codec: Video codec (mp4v, avc1, etc.)
        
    Returns:
        Path to saved video
    """
    if not frames:
        raise ValueError("No frames to save")
        
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        writer.write(frame)
        
    writer.release()
    return output_path


def extract_audio(video_path: str, output_path: str = None) -> Optional[str]:
    """
    Extract audio from video using ffmpeg.
    
    Args:
        video_path: Path to video file
        output_path: Output audio path (default: same name with .wav)
        
    Returns:
        Path to extracted audio or None if no audio
    """
    if output_path is None:
        output_path = str(Path(video_path).with_suffix('.wav'))
        
    try:
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '44100', '-ac', '2',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError:
        return None


def merge_audio_video(video_path: str, 
                      audio_path: str,
                      output_path: str) -> str:
    """
    Merge audio and video using ffmpeg.
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        output_path: Output path for merged file
        
    Returns:
        Path to merged video
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info

