"""
Utility functions for image processing, sample image generation, and visualizations.
"""

import os
from typing import Optional
import numpy as np
import cv2


def generate_synthetic_image(width: int = 512, height: int = 512, is_color: bool = True) -> np.ndarray:
    """
    Generates a rich synthetic benchmark image with distinct geometric regions,
    smooth gradients, and textured areas to effectively evaluate segmentation algorithms.
    """
    if is_color:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Background: subtle dark blue to purple gradient
        for y in range(height):
            v1 = int(30 + 50 * (y / height))
            v2 = int(20 + 30 * (y / height))
            img[y, :] = [v1, v2, 60]

        # Region 1: Large bright circular object (yellowish-orange)
        cv2.circle(img, (int(width * 0.3), int(height * 0.35)), int(min(width, height) * 0.22), (0, 180, 240), -1)

        # Region 2: Inner concentric disk (bright cyan)
        cv2.circle(img, (int(width * 0.3), int(height * 0.35)), int(min(width, height) * 0.10), (220, 200, 40), -1)

        # Region 3: Rotated rectangle (emerald green)
        rect_pts = np.array([
            [int(width * 0.55), int(height * 0.15)],
            [int(width * 0.85), int(height * 0.25)],
            [int(width * 0.75), int(height * 0.55)],
            [int(width * 0.45), int(height * 0.45)]
        ], np.int32)
        cv2.fillPoly(img, [rect_pts], (60, 200, 80))

        # Region 4: Triangular region (magenta / coral)
        tri_pts = np.array([
            [int(width * 0.20), int(height * 0.85)],
            [int(width * 0.50), int(height * 0.65)],
            [int(width * 0.45), int(height * 0.90)]
        ], np.int32)
        cv2.fillPoly(img, [tri_pts], (180, 80, 220))

        # Region 5: Gradient horizontal strip at bottom
        for x in range(int(width * 0.55), int(width * 0.92)):
            val = int(255 * ((x - width * 0.55) / (width * 0.37)))
            img[int(height * 0.65):int(height * 0.88), x] = [val, 120, 255 - val]

        # Add mild Gaussian noise to test homogeneity threshold resilience
        noise = np.random.normal(0, 3, img.shape).astype(np.float32)
        noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy_img
    else:
        # Grayscale version
        img = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            img[y, :] = int(40 + 40 * (y / height))

        cv2.circle(img, (int(width * 0.3), int(height * 0.35)), int(min(width, height) * 0.22), 200, -1)
        cv2.circle(img, (int(width * 0.3), int(height * 0.35)), int(min(width, height) * 0.10), 245, -1)

        rect_pts = np.array([
            [int(width * 0.55), int(height * 0.15)],
            [int(width * 0.85), int(height * 0.25)],
            [int(width * 0.75), int(height * 0.55)],
            [int(width * 0.45), int(height * 0.45)]
        ], np.int32)
        cv2.fillPoly(img, [rect_pts], 160)

        tri_pts = np.array([
            [int(width * 0.20), int(height * 0.85)],
            [int(width * 0.50), int(height * 0.65)],
            [int(width * 0.45), int(height * 0.90)]
        ], np.int32)
        cv2.fillPoly(img, [tri_pts], 110)

        for x in range(int(width * 0.55), int(width * 0.92)):
            val = int(255 * ((x - width * 0.55) / (width * 0.37)))
            img[int(height * 0.65):int(height * 0.88), x] = val

        noise = np.random.normal(0, 3, img.shape).astype(np.float32)
        noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy_img


def get_or_create_sample_image(
    image_path: Optional[str] = None,
    default_asset_dir: str = "assets",
    default_filename: str = "sample.png"
) -> np.ndarray:
    """
    Loads an image from the specified path.
    If no path is provided or the file is missing, generates a high-quality synthetic test image,
    saves it to assets/sample.png, and returns it.
    """
    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            return img
        print(f"[Warning] Failed to read image from '{image_path}'. Generating synthetic fallback.")

    os.makedirs(default_asset_dir, exist_ok=True)
    fallback_path = os.path.join(default_asset_dir, default_filename)

    if not os.path.exists(fallback_path):
        print(f"[Info] Generating default benchmark image at '{fallback_path}'...")
        synthetic = generate_synthetic_image(512, 512, is_color=True)
        cv2.imwrite(fallback_path, synthetic)
        return synthetic

    loaded = cv2.imread(fallback_path)
    if loaded is not None:
        return loaded

    # If all else fails, generate in memory
    return generate_synthetic_image(512, 512, is_color=True)
