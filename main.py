"""
Image Segmentation Assignment Entry Point.

Runs Quadtree Split-and-Merge segmentation and/or Voronoi Diagram partitioning
on an input image (or an automatically generated benchmark image).
"""

import os
import sys
import argparse
from typing import Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import get_or_create_sample_image
from split_merge.quadtree import split_and_merge_segmentation, plot_split_merge_results
from voronoi.voronoi_art import generate_voronoi_segmentation, plot_voronoi_results


def plot_side_by_side_comparison(
    original: np.ndarray,
    quadtree_result: np.ndarray,
    voronoi_result: np.ndarray,
    qt_stats: dict,
    vor_stats: dict,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Generates a high-level comparison figure showing the original image next to
    the Quadtree Split-and-Merge segmentation and the Voronoi Partition.
    Ideal for direct inclusion in the university lab report.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img

    cmap = "gray" if len(original.shape) == 2 else None

    # Original
    axes[0].imshow(to_rgb(original), cmap=cmap)
    axes[0].set_title("1. Original Image", fontsize=13, fontweight='bold')
    axes[0].axis("off")

    # Quadtree Split-and-Merge
    axes[1].imshow(to_rgb(quadtree_result), cmap=cmap)
    axes[1].set_title(
        f"2. Quadtree Split-and-Merge\n({qt_stats['num_regions_merged']} Regions | PSNR: {qt_stats['psnr_db']:.2f} dB)",
        fontsize=12, fontweight='bold'
    )
    axes[1].axis("off")

    # Voronoi Partition
    axes[2].imshow(to_rgb(voronoi_result), cmap=cmap)
    axes[2].set_title(
        f"3. Voronoi Stippling Partition\n({vor_stats['num_seed_points']} Cells | PSNR: {vor_stats['psnr_db']:.2f} dB)",
        fontsize=12, fontweight='bold'
    )
    axes[2].axis("off")

    plt.suptitle("Comparative Evaluation: Quadtree Split-and-Merge vs Voronoi Partitioning",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Comparison] Figure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_split_merge(args, image: np.ndarray, output_dir: str):
    print("\n" + "="*60)
    print(" >>> STEP 1: QUADTREE SPLIT-AND-MERGE SEGMENTATION <<<")
    print("="*60)
    print(f"Parameters: Std Threshold = {args.std_thresh}, Min Block Size = {args.min_size}, Merge Threshold = {args.merge_thresh}")

    split_img, merged_img, grid_img, stats = split_and_merge_segmentation(
        image,
        std_threshold=args.std_thresh,
        min_size=args.min_size,
        merge_threshold=args.merge_thresh
    )

    print("\n--- Split-and-Merge Results ---")
    print(f"  - Leaf Blocks After Split:    {stats['num_leaves_split']}")
    print(f"  - Connected Regions (Merged): {stats['num_regions_merged']}")
    print(f"  - Region Reduction Factor:    {stats['reduction_ratio']:.2f}x")
    print(f"  - Reconstruction MSE:         {stats['mse']:.2f}")
    print(f"  - Reconstruction PSNR:        {stats['psnr_db']:.2f} dB")

    save_path = os.path.join(output_dir, "split_merge_result.png")
    plot_split_merge_results(
        image, grid_img, split_img, merged_img, stats,
        save_path=save_path,
        show_plot=not args.no_plot
    )
    return merged_img, stats


def run_voronoi(args, image: np.ndarray, output_dir: str):
    print("\n" + "="*60)
    print(" >>> STEP 2: VORONOI DIAGRAM PARTITIONING & MOSAIC <<<")
    print("="*60)
    print(f"Parameters: Seed Points = {args.points}, Sampling = '{args.sampling}', Edge Weight = {args.edge_weight}")

    mosaic_clean, mosaic_edges, seeds_img, stats = generate_voronoi_segmentation(
        image,
        n_points=args.points,
        sampling_method=args.sampling,
        edge_weight=args.edge_weight
    )

    print("\n--- Voronoi Partition Results ---")
    print(f"  - Total Seed Points:         {stats['num_seed_points']}")
    print(f"  - Sampling Strategy:         {stats['sampling_method']}")
    print(f"  - Average Cell Area:         {stats['avg_cell_area_px']:.1f} pixels")
    print(f"  - Reconstruction MSE:        {stats['mse']:.2f}")
    print(f"  - Reconstruction PSNR:       {stats['psnr_db']:.2f} dB")

    save_path = os.path.join(output_dir, "voronoi_result.png")
    plot_voronoi_results(
        image, seeds_img, mosaic_edges, mosaic_clean, stats,
        save_path=save_path,
        show_plot=not args.no_plot
    )
    return mosaic_clean, stats


def main():
    parser = argparse.ArgumentParser(
        description="University Assignment: Quadtree Split-Merge and Voronoi Image Segmentation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General configuration
    parser.add_argument("--mode", type=str, choices=["all", "split_merge", "voronoi"], default="all",
                        help="Select algorithm to execute: 'split_merge', 'voronoi', or 'all'")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image. If omitted or not found, a benchmark image is generated automatically.")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save output figures")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable interactive GUI windows (useful for batch execution or headless environments)")

    # Split-and-Merge arguments
    group_sm = parser.add_argument_group("Split-and-Merge Hyperparameters")
    group_sm.add_argument("--std-thresh", type=float, default=14.0,
                          help="Homogeneity standard deviation threshold (splitting criterion)")
    group_sm.add_argument("--min-size", type=int, default=4,
                          help="Minimum block width/height to stop splitting")
    group_sm.add_argument("--merge-thresh", type=float, default=12.0,
                          help="Maximum intensity difference to merge adjacent leaf blocks")

    # Voronoi arguments
    group_vor = parser.add_argument_group("Voronoi Partition Hyperparameters")
    group_vor.add_argument("--points", type=int, default=1500,
                           help="Number of Voronoi seed points")
    group_vor.add_argument("--sampling", type=str, choices=["adaptive", "random", "grid"], default="adaptive",
                           help="Sampling method: 'adaptive' (gradient-based), 'random', or 'grid'")
    group_vor.add_argument("--edge-weight", type=float, default=0.8,
                           help="Weight assigned to edge regions in adaptive sampling (0.0 to 1.0)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image = get_or_create_sample_image(args.image)

    print("\n" + "#"*65)
    print("  IMAGE SEGMENTATION & RECOGNITION TP PIPELINE")
    print("  University Assignment: Quadtree Split-Merge & Voronoi Partitioning")
    print("#"*65)
    print(f"Input image loaded successfully: Dimensions = {image.shape[1]}x{image.shape[0]} | Channels = {image.shape[2] if len(image.shape)==3 else 1}")

    merged_sm = None
    qt_stats = None
    mosaic_vor = None
    vor_stats = None

    if args.mode in ["all", "split_merge"]:
        merged_sm, qt_stats = run_split_merge(args, image, args.output_dir)

    if args.mode in ["all", "voronoi"]:
        mosaic_vor, vor_stats = run_voronoi(args, image, args.output_dir)

    # Comparative evaluation when running both algorithms
    if args.mode == "all" and merged_sm is not None and mosaic_vor is not None:
        comp_save_path = os.path.join(args.output_dir, "comparison_result.png")
        plot_side_by_side_comparison(
            image, merged_sm, mosaic_vor, qt_stats, vor_stats,
            save_path=comp_save_path,
            show_plot=not args.no_plot
        )

        print("\n" + "="*60)
        print(" >>> SUMMARY & COMPARISON <<<")
        print("="*60)
        print(f"{'Metric':<28} | {'Quadtree Split-Merge':<20} | {'Voronoi Partition':<20}")
        print("-" * 74)
        print(f"{'Regions / Cells Count':<28} | {qt_stats['num_regions_merged']:<20} | {vor_stats['num_seed_points']:<20}")
        print(f"{'Reconstruction MSE':<28} | {qt_stats['mse']:<20.2f} | {vor_stats['mse']:<20.2f}")
        print(f"{'Reconstruction PSNR (dB)':<28} | {qt_stats['psnr_db']:<20.2f} | {vor_stats['psnr_db']:<20.2f}")
        print("="*60)
        print(f"\nAll generated report figures are ready in: '{os.path.abspath(args.output_dir)}'")


if __name__ == "__main__":
    main()
