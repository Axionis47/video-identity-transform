"""
Image utility functions.
"""

import numpy as np
from typing import Tuple, Optional
import cv2


def resize_with_aspect_ratio(image: np.ndarray,
                             target_size: Tuple[int, int],
                             interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def pad_to_size(image: np.ndarray,
                target_size: Tuple[int, int],
                pad_value: int = 0) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad image to target size.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        pad_value: Padding value
        
    Returns:
        Tuple of (padded image, padding (top, bottom, left, right))
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    
    if len(image.shape) == 3:
        padded = np.full((target_h, target_w, image.shape[2]), pad_value, dtype=image.dtype)
    else:
        padded = np.full((target_h, target_w), pad_value, dtype=image.dtype)
        
    padded[pad_top:pad_top+h, pad_left:pad_left+w] = image
    
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def crop_and_resize(image: np.ndarray,
                    bbox: np.ndarray,
                    output_size: Tuple[int, int],
                    padding: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop region from image and resize.
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        output_size: Output (width, height)
        padding: Padding ratio around bbox
        
    Returns:
        Tuple of (cropped image, adjusted bbox)
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Add padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    # Crop
    cropped = image[y1:y2, x1:x2]
    
    # Resize
    resized = cv2.resize(cropped, output_size)
    
    # Adjusted bbox (relative to crop)
    adj_bbox = np.array([pad_x, pad_y, box_w + pad_x, box_h + pad_y])
    
    return resized, adj_bbox


def blend_images(foreground: np.ndarray,
                 background: np.ndarray,
                 mask: np.ndarray) -> np.ndarray:
    """
    Blend foreground onto background using mask.
    
    Args:
        foreground: Foreground image
        background: Background image
        mask: Blending mask (0-1 or 0-255)
        
    Returns:
        Blended image
    """
    if mask.max() > 1:
        mask = mask.astype(float) / 255.0
        
    if len(mask.shape) == 2:
        mask = np.stack([mask] * 3, axis=-1)
        
    blended = foreground * mask + background * (1 - mask)
    return blended.astype(np.uint8)


def apply_color_transfer(source: np.ndarray,
                         target: np.ndarray) -> np.ndarray:
    """
    Transfer color statistics from target to source.
    
    Args:
        source: Source image to modify
        target: Target image for color reference
        
    Returns:
        Color-transferred image
    """
    # Convert to LAB
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(float)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(float)
    
    # Calculate statistics
    for i in range(3):
        src_mean, src_std = source_lab[:, :, i].mean(), source_lab[:, :, i].std()
        tgt_mean, tgt_std = target_lab[:, :, i].mean(), target_lab[:, :, i].std()
        
        # Transfer
        source_lab[:, :, i] = (source_lab[:, :, i] - src_mean) * (tgt_std / (src_std + 1e-6)) + tgt_mean
        
    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)

