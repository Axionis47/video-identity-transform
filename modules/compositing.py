"""
Scene Compositing Module

Blends transformed persons back into the original scene with:
- Color and lighting matching
- Shadow generation
- Seamless edge blending
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import cv2


@dataclass
class CompositingResult:
    """Result of compositing operation."""
    frame_idx: int
    composited_frame: np.ndarray  # Final BGR frame
    debug_layers: Optional[Dict[str, np.ndarray]] = None


class SceneCompositor:
    """
    Composites transformed persons into the original scene.
    Handles color matching, lighting, shadows, and seamless blending.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.color_matching = config.get('color_matching', True)
        self.lighting_adjustment = config.get('lighting_adjustment', True)
        self.generate_shadows = config.get('generate_shadows', True)
        self.shadow_strength = config.get('shadow_strength', 0.6)
        self.blend_mode = config.get('blend_mode', 'poisson')
        self.feather_edges = config.get('feather_edges', 5)
        
    def composite_frame(self,
                       background: np.ndarray,
                       transformed_persons: List[Dict],
                       original_masks: List[np.ndarray]) -> CompositingResult:
        """
        Composite all transformed persons into the background.
        
        Args:
            background: Original frame with persons removed/inpainted
            transformed_persons: List of {rgba, bbox, track_id}
            original_masks: Original person masks for reference
            
        Returns:
            CompositingResult with final frame
        """
        result = background.copy()
        debug_layers = {} if self.config.get('save_intermediate_frames', False) else None
        
        for person_data in transformed_persons:
            rgba = person_data['rgba']
            bbox = person_data['bbox']
            track_id = person_data['track_id']
            
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Extract RGB and alpha
            person_rgb = rgba[:, :, :3]
            person_alpha = rgba[:, :, 3].astype(float) / 255.0
            
            # Color matching
            if self.color_matching:
                person_rgb = self._match_colors(
                    person_rgb, 
                    background[y1:y2, x1:x2],
                    person_alpha
                )
            
            # Lighting adjustment
            if self.lighting_adjustment:
                person_rgb = self._adjust_lighting(
                    person_rgb,
                    background[y1:y2, x1:x2],
                    person_alpha
                )
            
            # Generate shadows
            if self.generate_shadows:
                result = self._add_shadow(
                    result, person_alpha, bbox
                )
            
            # Feather edges
            if self.feather_edges > 0:
                person_alpha = self._feather_mask(person_alpha)
            
            # Blend into scene
            if self.blend_mode == 'poisson':
                result = self._poisson_blend(
                    result, person_rgb, person_alpha, bbox
                )
            elif self.blend_mode == 'multi_band':
                result = self._multiband_blend(
                    result, person_rgb, person_alpha, bbox
                )
            else:
                result = self._alpha_blend(
                    result, person_rgb, person_alpha, bbox
                )
                
            if debug_layers is not None:
                debug_layers[f'person_{track_id}'] = person_rgb
                
        return CompositingResult(
            frame_idx=0,
            composited_frame=result,
            debug_layers=debug_layers
        )
    
    def _match_colors(self, source: np.ndarray, 
                     target: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
        """Match source colors to target using histogram matching."""
        result = source.copy()
        
        # Convert to LAB color space
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(float)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(float)
        
        # Calculate statistics for masked regions
        mask_bool = mask > 0.5
        if mask_bool.sum() < 100:
            return source
            
        for i in range(3):
            src_mean = source_lab[:, :, i][mask_bool].mean()
            src_std = source_lab[:, :, i][mask_bool].std() + 1e-6
            tgt_mean = target_lab[:, :, i].mean()
            tgt_std = target_lab[:, :, i].std() + 1e-6
            
            # Transfer statistics
            source_lab[:, :, i] = (source_lab[:, :, i] - src_mean) * (tgt_std / src_std) + tgt_mean
            
        source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _adjust_lighting(self, source: np.ndarray,
                        target: np.ndarray,
                        mask: np.ndarray) -> np.ndarray:
        """Adjust lighting to match scene."""
        # Simple luminance matching
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(float)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(float)
        
        mask_bool = mask > 0.5
        if mask_bool.sum() < 100:
            return source
            
        src_lum = source_gray[mask_bool].mean()
        tgt_lum = target_gray.mean()
        
        ratio = tgt_lum / (src_lum + 1e-6)
        ratio = np.clip(ratio, 0.5, 2.0)
        
        result = (source.astype(float) * ratio).clip(0, 255).astype(np.uint8)
        return result

    def _add_shadow(self, frame: np.ndarray,
                   mask: np.ndarray,
                   bbox: np.ndarray) -> np.ndarray:
        """Add shadow beneath the person."""
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = mask.shape

        # Create shadow mask (shifted down and blurred)
        shadow_offset = int(h * 0.05)
        shadow_mask = np.zeros_like(frame[:, :, 0], dtype=float)

        # Place shadow
        sy1 = min(y1 + shadow_offset, frame.shape[0] - h)
        sy2 = min(sy1 + h, frame.shape[0])
        sx1, sx2 = x1, min(x2, frame.shape[1])

        if sy2 > sy1 and sx2 > sx1:
            mask_h = sy2 - sy1
            mask_w = sx2 - sx1
            shadow_mask[sy1:sy2, sx1:sx2] = cv2.resize(mask, (mask_w, mask_h))

        # Blur shadow
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)

        # Apply shadow
        shadow_mask = shadow_mask * self.shadow_strength
        for c in range(3):
            frame[:, :, c] = (frame[:, :, c] * (1 - shadow_mask * 0.5)).astype(np.uint8)

        return frame

    def _feather_mask(self, mask: np.ndarray) -> np.ndarray:
        """Feather mask edges for smoother blending."""
        kernel_size = self.feather_edges * 2 + 1
        return cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    def _alpha_blend(self, background: np.ndarray,
                    foreground: np.ndarray,
                    alpha: np.ndarray,
                    bbox: np.ndarray) -> np.ndarray:
        """Simple alpha blending."""
        x1, y1, x2, y2 = bbox.astype(int)

        # Ensure dimensions match
        fg_h, fg_w = foreground.shape[:2]
        bg_region = background[y1:y1+fg_h, x1:x1+fg_w]

        if bg_region.shape[:2] != foreground.shape[:2]:
            foreground = cv2.resize(foreground, (bg_region.shape[1], bg_region.shape[0]))
            alpha = cv2.resize(alpha, (bg_region.shape[1], bg_region.shape[0]))

        alpha_3d = np.stack([alpha] * 3, axis=-1)
        blended = (foreground * alpha_3d + bg_region * (1 - alpha_3d)).astype(np.uint8)

        result = background.copy()
        result[y1:y1+fg_h, x1:x1+fg_w] = blended

        return result

    def _poisson_blend(self, background: np.ndarray,
                      foreground: np.ndarray,
                      alpha: np.ndarray,
                      bbox: np.ndarray) -> np.ndarray:
        """Poisson blending for seamless compositing."""
        x1, y1, x2, y2 = bbox.astype(int)

        # Create mask for seamlessClone
        mask = (alpha * 255).astype(np.uint8)

        # Ensure dimensions match
        fg_h, fg_w = foreground.shape[:2]

        # Center point for seamlessClone
        center = (x1 + fg_w // 2, y1 + fg_h // 2)

        # Ensure foreground fits in background
        if (center[0] + fg_w // 2 > background.shape[1] or
            center[1] + fg_h // 2 > background.shape[0] or
            center[0] - fg_w // 2 < 0 or
            center[1] - fg_h // 2 < 0):
            # Fall back to alpha blend
            return self._alpha_blend(background, foreground, alpha, bbox)

        try:
            result = cv2.seamlessClone(
                foreground, background, mask, center, cv2.NORMAL_CLONE
            )
        except cv2.error:
            result = self._alpha_blend(background, foreground, alpha, bbox)

        return result

    def _multiband_blend(self, background: np.ndarray,
                        foreground: np.ndarray,
                        alpha: np.ndarray,
                        bbox: np.ndarray) -> np.ndarray:
        """Multi-band blending using Laplacian pyramids."""
        x1, y1, x2, y2 = bbox.astype(int)
        fg_h, fg_w = foreground.shape[:2]

        bg_region = background[y1:y1+fg_h, x1:x1+fg_w]

        if bg_region.shape[:2] != foreground.shape[:2]:
            return self._alpha_blend(background, foreground, alpha, bbox)

        # Build Laplacian pyramids
        levels = 4
        fg_pyr = self._build_laplacian_pyramid(foreground, levels)
        bg_pyr = self._build_laplacian_pyramid(bg_region, levels)

        # Build Gaussian pyramid for mask
        mask_pyr = self._build_gaussian_pyramid(alpha, levels)

        # Blend pyramids
        blended_pyr = []
        for fg_level, bg_level, mask_level in zip(fg_pyr, bg_pyr, mask_pyr):
            mask_3d = np.stack([mask_level] * 3, axis=-1)
            blended = fg_level * mask_3d + bg_level * (1 - mask_3d)
            blended_pyr.append(blended)

        # Reconstruct
        blended = self._reconstruct_from_pyramid(blended_pyr)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        result = background.copy()
        result[y1:y1+fg_h, x1:x1+fg_w] = blended

        return result

    def _build_gaussian_pyramid(self, img: np.ndarray, levels: int) -> List[np.ndarray]:
        """Build Gaussian pyramid."""
        pyramid = [img.astype(float)]
        for _ in range(levels - 1):
            img = cv2.pyrDown(pyramid[-1])
            pyramid.append(img)
        return pyramid

    def _build_laplacian_pyramid(self, img: np.ndarray, levels: int) -> List[np.ndarray]:
        """Build Laplacian pyramid."""
        gaussian = self._build_gaussian_pyramid(img.astype(float), levels)
        laplacian = []
        for i in range(levels - 1):
            size = (gaussian[i].shape[1], gaussian[i].shape[0])
            expanded = cv2.pyrUp(gaussian[i + 1], dstsize=size)
            laplacian.append(gaussian[i] - expanded)
        laplacian.append(gaussian[-1])
        return laplacian

    def _reconstruct_from_pyramid(self, pyramid: List[np.ndarray]) -> np.ndarray:
        """Reconstruct image from Laplacian pyramid."""
        img = pyramid[-1]
        for i in range(len(pyramid) - 2, -1, -1):
            size = (pyramid[i].shape[1], pyramid[i].shape[0])
            img = cv2.pyrUp(img, dstsize=size) + pyramid[i]
        return img

