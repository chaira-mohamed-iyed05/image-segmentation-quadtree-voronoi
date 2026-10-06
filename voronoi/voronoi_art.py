"""
Voronoi Diagram Image Segmentation and Mosaic Art.

This module partitions an image into Voronoi cells using SciPy spatial tools
(scipy.spatial.Voronoi and scipy.spatial.cKDTree). It supports adaptive edge-aware
stippling sampling, uniform random sampling, and grid-jittered sampling.
"""

from typing import Tuple, Dict, Optional, Literal
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, cKDTree


class VoronoiSegmenter:
    """
    Image segmentation and artistic mosaic rendering using Voronoi tessellations.
    """
    def __init__(
        self,
        n_points: int = 1500,
        sampling_method: Literal["adaptive", "random", "grid"] = "adaptive",
        edge_weight: float = 0.8
    ):
        """
        Args:
            n_points: Number of Voronoi seed points to generate.
            sampling_method: 'adaptive' (gradient-weighted), 'random', or 'grid'.
            edge_weight: For 'adaptive', weighting factor between gradient map and uniform base (0.0 to 1.0).
        """
        self.n_points = n_points
        self.sampling_method = sampling_method
        self.edge_weight = np.clip(edge_weight, 0.0, 1.0)
        self.points: Optional[np.ndarray] = None
        self.voronoi: Optional[Voronoi] = None

    def _sample_points(self, image: np.ndarray) -> np.ndarray:
        """
        Generates seed points (x, y) based on the chosen sampling strategy.
        """
        h, w = image.shape[:2]

        if self.sampling_method == "random":
            xs = np.random.uniform(0, w - 1, size=self.n_points)
            ys = np.random.uniform(0, h - 1, size=self.n_points)
            points = np.column_stack([xs, ys])

        elif self.sampling_method == "grid":
            # Grid with small random perturbation (jitter)
            aspect_ratio = w / h
            ny = int(np.sqrt(self.n_points / aspect_ratio))
            nx = int(self.n_points / ny)
            gx, gy = np.meshgrid(np.linspace(0, w - 1, nx), np.linspace(0, h - 1, ny))
            xs = gx.ravel() + np.random.uniform(-w / (2 * nx), w / (2 * nx), size=nx * ny)
            ys = gy.ravel() + np.random.uniform(-h / (2 * ny), h / (2 * ny), size=nx * ny)
            points = np.column_stack([np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1)])

        elif self.sampling_method == "adaptive":
            # Adaptive / edge-weighted importance sampling (stippling)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Compute gradient magnitude using Sobel operators
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            gradient_mag = cv2.magnitude(grad_x, grad_y)

            # Normalize gradient to [0, 1]
            g_min, g_max = gradient_mag.min(), gradient_mag.max()
            if g_max > g_min:
                gradient_norm = (gradient_mag - g_min) / (g_max - g_min)
            else:
                gradient_norm = np.zeros_like(gradient_mag)

            # Combine edge importance with uniform base probability to avoid empty regions
            pdf = (self.edge_weight * gradient_norm) + ((1.0 - self.edge_weight) * np.ones_like(gradient_norm))
            pdf /= pdf.sum()

            # Draw samples according to 2D probability distribution
            flat_indices = np.random.choice(h * w, size=self.n_points, replace=False, p=pdf.ravel())
            ys = (flat_indices // w).astype(np.float64)
            xs = (flat_indices % w).astype(np.float64)

            # Add subpixel jitter
            xs += np.random.uniform(-0.5, 0.5, size=self.n_points)
            ys += np.random.uniform(-0.5, 0.5, size=self.n_points)
            points = np.column_stack([np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1)])
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        # Add image 4 corner points to ensure tidy border tessellation
        corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64)
        points = np.vstack([points, corners])
        return points

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Partitions the image into Voronoi cells and computes mean color per cell.

        Returns:
            mosaic_img: Segmented image where each Voronoi cell is colored with its mean color.
            mosaic_with_edges: Segmented image with cell boundary ridges highlighted.
            stats: Dictionary of quantitative metrics.
        """
        h, w = image.shape[:2]
        is_color = (len(image.shape) == 3 and image.shape[2] == 3)

        self.points = self._sample_points(image)
        self.voronoi = Voronoi(self.points)

        # Build KDTree on the seed coordinates for fast vectorized pixel-to-cell assignment
        kdtree = cKDTree(self.points)

        # Generate coordinate grid for all pixels in image
        grid_y, grid_x = np.indices((h, w))
        pixel_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        # Query nearest seed for each pixel in O(N log K)
        _, cell_indices = kdtree.query(pixel_coords, k=1)
        cell_map = cell_indices.reshape(h, w)

        # Calculate mean color/intensity for each Voronoi cell
        n_cells = len(self.points)
        mosaic_img = np.zeros_like(image)

        if is_color:
            flat_img = image.reshape(-1, 3)
            # Accumulate channel sums and pixel counts per cell
            counts = np.bincount(cell_indices, minlength=n_cells)
            for c in range(3):
                channel_sums = np.bincount(cell_indices, weights=flat_img[:, c], minlength=n_cells)
                # Avoid division by zero for unpopulated cells
                valid_mask = counts > 0
                means = np.zeros(n_cells, dtype=np.float64)
                means[valid_mask] = channel_sums[valid_mask] / counts[valid_mask]
                mosaic_img[:, :, c] = means[cell_map].astype(np.uint8)
        else:
            flat_img = image.ravel()
            counts = np.bincount(cell_indices, minlength=n_cells)
            channel_sums = np.bincount(cell_indices, weights=flat_img, minlength=n_cells)
            valid_mask = counts > 0
            means = np.zeros(n_cells, dtype=np.float64)
            means[valid_mask] = channel_sums[valid_mask] / counts[valid_mask]
            mosaic_img = means[cell_map].astype(np.uint8)

        # Create cell boundary edge overlay using morphological gradient of cell labels
        kernel = np.ones((3, 3), np.uint8)
        # Shift differences to detect cell boundaries
        diff_h = cell_map[:, :-1] != cell_map[:, 1:]
        diff_v = cell_map[:-1, :] != cell_map[1:, :]
        edges_mask = np.zeros((h, w), dtype=bool)
        edges_mask[:, :-1] |= diff_h
        edges_mask[:-1, :] |= diff_v

        # Render mosaic with dark outlines
        mosaic_with_edges = mosaic_img.copy()
        if is_color:
            mosaic_with_edges[edges_mask] = [20, 20, 20]  # dark slate gray outline
        else:
            mosaic_with_edges[edges_mask] = 20

        # Calculate metrics (MSE, PSNR)
        if is_color:
            ref_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            res_gray = cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = image
            res_gray = mosaic_img

        mse = float(np.mean((ref_gray.astype(float) - res_gray.astype(float)) ** 2))
        psnr = float(10 * np.log10(255**2 / (mse + 1e-10)))

        stats = {
            "num_seed_points": len(self.points),
            "sampling_method": self.sampling_method,
            "avg_cell_area_px": float((h * w) / len(self.points)),
            "mse": mse,
            "psnr_db": psnr
        }

        return mosaic_img, mosaic_with_edges, stats


def generate_voronoi_segmentation(
    image: np.ndarray,
    n_points: int = 1500,
    sampling_method: Literal["adaptive", "random", "grid"] = "adaptive",
    edge_weight: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Executes Voronoi image segmentation pipeline.

    Returns:
        mosaic_img: Segmented Voronoi mosaic image.
        mosaic_with_edges: Voronoi mosaic with cell borders.
        seeds_img: Visual overlay of seed points on original image.
        stats: Dictionary of performance metrics.
    """
    segmenter = VoronoiSegmenter(n_points=n_points, sampling_method=sampling_method, edge_weight=edge_weight)
    mosaic_img, mosaic_with_edges, stats = segmenter.segment(image)

    # Visualize seed point locations on top of original image
    if len(image.shape) == 2:
        seeds_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        seeds_img = image.copy()

    for pt in segmenter.points:
        cv2.circle(seeds_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

    return mosaic_img, mosaic_with_edges, seeds_img, stats


def plot_voronoi_results(
    original: np.ndarray,
    seeds_img: np.ndarray,
    mosaic_with_edges: np.ndarray,
    mosaic_clean: np.ndarray,
    stats: Dict[str, float],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Displays a publication-quality 4-panel matplotlib figure for assignment report.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img

    cmap = "gray" if len(original.shape) == 2 else None

    # Panel 1: Original
    axes[0, 0].imshow(to_rgb(original), cmap=cmap)
    axes[0, 0].set_title("1. Original Input Image", fontsize=13, fontweight='bold')
    axes[0, 0].axis("off")

    # Panel 2: Seeds distribution
    axes[0, 1].imshow(to_rgb(seeds_img))
    axes[0, 1].set_title(
        f"2. Voronoi Seeds ({stats['num_seed_points']} pts | Method: {stats['sampling_method']})",
        fontsize=13, fontweight='bold'
    )
    axes[0, 1].axis("off")

    # Panel 3: Voronoi Mosaic with cell edges
    axes[1, 0].imshow(to_rgb(mosaic_with_edges), cmap=cmap)
    axes[1, 0].set_title("3. Voronoi Mosaic (with Ridges / Cell Borders)", fontsize=13, fontweight='bold')
    axes[1, 0].axis("off")

    # Panel 4: Clean Voronoi Segmented Image
    axes[1, 1].imshow(to_rgb(mosaic_clean), cmap=cmap)
    axes[1, 1].set_title(
        f"4. Segmented Partition (PSNR: {stats['psnr_db']:.2f} dB | Avg Area: {stats['avg_cell_area_px']:.1f} px)",
        fontsize=13, fontweight='bold'
    )
    axes[1, 1].axis("off")

    plt.suptitle("Voronoi Diagram Image Partitioning & Segmentation (SciPy)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Voronoi] Figure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    import os
    import sys

    # Allow importing sibling modules when run directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import get_or_create_sample_image

    parser = argparse.ArgumentParser(description="Voronoi Image Segmentation and Mosaic Art")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file (generates sample if omitted)")
    parser.add_argument("--points", type=int, default=1500, help="Number of Voronoi seed points")
    parser.add_argument("--sampling", type=str, choices=["adaptive", "random", "grid"], default="adaptive",
                        help="Sampling strategy: 'adaptive' (gradient-based), 'random', or 'grid'")
    parser.add_argument("--edge-weight", type=float, default=0.8, help="Importance weight for edges in adaptive sampling")
    parser.add_argument("--output", type=str, default="output/voronoi_result.png", help="Path to save output figure")
    parser.add_argument("--no-plot", action="store_true", help="Disable interactive matplotlib GUI display")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    img = get_or_create_sample_image(args.image)

    print(f"\n[Voronoi] Running Voronoi partitioning...")
    print(f"  - Input shape: {img.shape}")
    print(f"  - Parameters: points={args.points}, sampling={args.sampling}, edge_weight={args.edge_weight}")

    mosaic_clean, mosaic_edges, seeds_img, stats = generate_voronoi_segmentation(
        img,
        n_points=args.points,
        sampling_method=args.sampling,
        edge_weight=args.edge_weight
    )

    print(f"\n[Voronoi] Results Summary:")
    print(f"  - Seed Points:       {stats['num_seed_points']}")
    print(f"  - Sampling Strategy: {stats['sampling_method']}")
    print(f"  - Avg Cell Area:     {stats['avg_cell_area_px']:.1f} pixels")
    print(f"  - Reconstruction MSE: {stats['mse']:.2f}")
    print(f"  - Reconstruction PSNR: {stats['psnr_db']:.2f} dB")

    plot_voronoi_results(
        img, seeds_img, mosaic_edges, mosaic_clean, stats,
        save_path=args.output,
        show_plot=not args.no_plot
    )
