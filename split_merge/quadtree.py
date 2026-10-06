"""
Split-and-Merge Segmentation using Quadtree Decomposition.

This module implements the classic Horowitz-Pavlidis (1974) Split-and-Merge
image segmentation algorithm using a Quadtree structure and Disjoint Set Union (Union-Find).
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt


class DisjointSetUnion:
    """
    Disjoint Set Union (Union-Find) with path compression and rank optimization.
    Used for merging adjacent quadtree leaf regions.
    """
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return False

        if self.rank[root_i] < self.rank[root_j]:
            self.parent[root_i] = root_j
        elif self.rank[root_i] > self.rank[root_j]:
            self.parent[root_j] = root_i
        else:
            self.parent[root_j] = root_i
            self.rank[root_i] += 1
        return True


class QuadTreeNode:
    """
    Represents a node in the Quadtree.
    
    Attributes:
        x, y: Top-left coordinate of the region.
        w, h: Width and height of the region.
        mean_val: Mean pixel intensity (or color vector) in this region.
        std_val: Standard deviation of pixel intensities (homogeneity metric).
        is_leaf: True if this node is not subdivided.
        children: List of 4 sub-nodes [NW, NE, SW, SE] if not a leaf.
        node_id: Unique integer index assigned to leaf nodes.
    """
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.mean_val: float = 0.0
        self.mean_color: Optional[np.ndarray] = None
        self.std_val: float = 0.0
        self.is_leaf: bool = False
        self.children: List['QuadTreeNode'] = []
        self.node_id: int = -1

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class QuadTree:
    """
    Quadtree image decomposition and merge algorithm.
    """
    def __init__(self, std_threshold: float = 12.0, min_size: int = 4, merge_threshold: float = 10.0):
        """
        Args:
            std_threshold: Max standard deviation for a block to be considered homogeneous.
            min_size: Minimum block size (width/height) below which splitting stops.
            merge_threshold: Max difference between mean intensities of adjacent leaves to merge.
        """
        self.std_threshold = std_threshold
        self.min_size = min_size
        self.merge_threshold = merge_threshold
        self.root: Optional[QuadTreeNode] = None
        self.leaves: List[QuadTreeNode] = []

    def _split(self, node: QuadTreeNode, gray: np.ndarray, color: Optional[np.ndarray]) -> None:
        """
        Recursive splitting step based on homogeneity predicate.
        """
        x, y, w, h = node.x, node.y, node.w, node.h
        patch_gray = gray[y:y + h, x:x + w]

        node.mean_val = float(np.mean(patch_gray))
        node.std_val = float(np.std(patch_gray))
        if color is not None:
            node.mean_color = np.mean(color[y:y + h, x:x + w], axis=(0, 1))

        # Check homogeneity criterion and minimum dimensions
        is_homogeneous = node.std_val <= self.std_threshold
        can_split = (w > self.min_size) and (h > self.min_size)

        if is_homogeneous or not can_split:
            node.is_leaf = True
            node.node_id = len(self.leaves)
            self.leaves.append(node)
            return

        # Subdivide into 4 quadrants: NW, NE, SW, SE
        w1, w2 = w // 2, w - (w // 2)
        h1, h2 = h // 2, h - (h // 2)

        nw = QuadTreeNode(x, y, w1, h1)
        ne = QuadTreeNode(x + w1, y, w2, h1)
        sw = QuadTreeNode(x, y + h1, w1, h2)
        se = QuadTreeNode(x + w1, y + h1, w2, h2)

        node.children = [nw, ne, sw, se]
        for child in node.children:
            self._split(child, gray, color)

    def build_tree(self, image: np.ndarray) -> None:
        """
        Decomposes the input image using quadtree splitting.
        """
        self.leaves = []
        if len(image.shape) == 3:
            color = image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image[:, :, 0]
        else:
            color = None
            gray = image

        h, w = gray.shape[:2]
        self.root = QuadTreeNode(0, 0, w, h)
        self._split(self.root, gray, color)

    def _are_adjacent(self, a: QuadTreeNode, b: QuadTreeNode) -> bool:
        """
        Determines whether two rectangular leaf nodes share a common border.
        """
        # Horizontal adjacency
        touches_horizontally = (a.x + a.w == b.x) or (b.x + b.w == a.x)
        overlap_vertical = max(a.y, b.y) < min(a.y + a.h, b.y + b.h)
        if touches_horizontally and overlap_vertical:
            return True

        # Vertical adjacency
        touches_vertically = (a.y + a.h == b.y) or (b.y + b.h == a.y)
        overlap_horizontal = max(a.x, b.x) < min(a.x + a.w, b.x + b.w)
        if touches_vertically and overlap_horizontal:
            return True

        return False

    def merge_leaves(self) -> Tuple[DisjointSetUnion, Dict[int, List[QuadTreeNode]]]:
        """
        Merges adjacent leaf regions whose mean values differ by at most merge_threshold.
        Uses Disjoint Set Union (Union-Find) for connected components.
        """
        n_leaves = len(self.leaves)
        dsu = DisjointSetUnion(n_leaves)

        # To efficiently test adjacency, sort leaves or build a 2D map of leaf IDs
        # A pixel-lookup label map guarantees O(H*W) neighbor discovery without O(N^2) pairwise checks
        if not self.leaves:
            return dsu, {}

        # Build label map to accurately and quickly find all neighboring leaves
        max_h = max(leaf.y + leaf.h for leaf in self.leaves)
        max_w = max(leaf.x + leaf.w for leaf in self.leaves)
        leaf_grid = np.full((max_h, max_w), -1, dtype=np.int32)

        for leaf in self.leaves:
            leaf_grid[leaf.y:leaf.y + leaf.h, leaf.x:leaf.x + leaf.w] = leaf.node_id

        # Check adjacent pairs along internal grid borders
        # Horizontal neighbor transitions:
        diff_h = (leaf_grid[:, :-1] != leaf_grid[:, 1:])
        idx1_h = leaf_grid[:, :-1][diff_h]
        idx2_h = leaf_grid[:, 1:][diff_h]

        # Vertical neighbor transitions:
        diff_v = (leaf_grid[:-1, :] != leaf_grid[1:, :])
        idx1_v = leaf_grid[:-1, :][diff_v]
        idx2_v = leaf_grid[1:, :][diff_v]

        pairs1 = np.concatenate([idx1_h, idx1_v])
        pairs2 = np.concatenate([idx2_h, idx2_v])

        # Filter unique neighbor pairs
        neighbor_pairs = set()
        for u, v in zip(pairs1, pairs2):
            if u != v and u >= 0 and v >= 0:
                neighbor_pairs.add((min(u, v), max(u, v)))

        # Evaluate merge criterion for neighboring pairs
        for u, v in neighbor_pairs:
            leaf_u = self.leaves[u]
            leaf_v = self.leaves[v]
            if abs(leaf_u.mean_val - leaf_v.mean_val) <= self.merge_threshold:
                dsu.union(u, v)

        # Group leaves by connected component root
        groups: Dict[int, List[QuadTreeNode]] = {}
        for leaf in self.leaves:
            root = dsu.find(leaf.node_id)
            if root not in groups:
                groups[root] = []
            groups[root].append(leaf)

        return dsu, groups

    def reconstruct_split_image(self, shape: Tuple[int, ...], is_color: bool = False) -> np.ndarray:
        """
        Generates the segmented image after the Split step only (leaves filled with mean value).
        """
        if is_color:
            result = np.zeros(shape, dtype=np.uint8)
            for leaf in self.leaves:
                color = leaf.mean_color if leaf.mean_color is not None else np.array([leaf.mean_val]*3)
                result[leaf.y:leaf.y + leaf.h, leaf.x:leaf.x + leaf.w] = np.clip(color, 0, 255).astype(np.uint8)
        else:
            result = np.zeros(shape[:2], dtype=np.uint8)
            for leaf in self.leaves:
                result[leaf.y:leaf.y + leaf.h, leaf.x:leaf.x + leaf.w] = np.clip(leaf.mean_val, 0, 255).astype(np.uint8)
        return result

    def reconstruct_merged_image(self, shape: Tuple[int, ...], groups: Dict[int, List[QuadTreeNode]], is_color: bool = False) -> np.ndarray:
        """
        Generates the segmented image after both Split and Merge steps.
        Each merged region is assigned its area-weighted mean intensity/color.
        """
        if is_color:
            result = np.zeros(shape, dtype=np.uint8)
            for root, region_leaves in groups.items():
                total_area = sum(leaf.w * leaf.h for leaf in region_leaves)
                weighted_color = np.zeros(3, dtype=np.float64)
                for leaf in region_leaves:
                    area = leaf.w * leaf.h
                    col = leaf.mean_color if leaf.mean_color is not None else np.array([leaf.mean_val]*3)
                    weighted_color += col * area
                avg_color = np.clip(weighted_color / total_area, 0, 255).astype(np.uint8)

                for leaf in region_leaves:
                    result[leaf.y:leaf.y + leaf.h, leaf.x:leaf.x + leaf.w] = avg_color
        else:
            result = np.zeros(shape[:2], dtype=np.uint8)
            for root, region_leaves in groups.items():
                total_area = sum(leaf.w * leaf.h for leaf in region_leaves)
                weighted_mean = sum(leaf.mean_val * (leaf.w * leaf.h) for leaf in region_leaves)
                avg_mean = np.clip(weighted_mean / total_area, 0, 255).astype(np.uint8)

                for leaf in region_leaves:
                    result[leaf.y:leaf.y + leaf.h, leaf.x:leaf.x + leaf.w] = avg_mean

        return result

    def draw_quadtree_grid(self, base_image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        Draws the quadtree block bounding lines on the image.
        """
        if len(base_image.shape) == 2:
            vis = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = base_image.copy()

        for leaf in self.leaves:
            cv2.rectangle(vis, (leaf.x, leaf.y), (leaf.x + leaf.w, leaf.y + leaf.h), color, 1)

        return vis


def split_and_merge_segmentation(
    image: np.ndarray,
    std_threshold: float = 14.0,
    min_size: int = 4,
    merge_threshold: float = 12.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Executes complete Split-and-Merge segmentation pipeline.

    Returns:
        split_img: Image reconstructed from quadtree split blocks.
        merged_img: Final segmented image after merging adjacent regions.
        grid_img: Original image with quadtree decomposition grid overlay.
        stats: Dictionary containing quantitative metrics (leaves count, merged regions, etc.).
    """
    is_color = (len(image.shape) == 3 and image.shape[2] == 3)
    qt = QuadTree(std_threshold=std_threshold, min_size=min_size, merge_threshold=merge_threshold)
    qt.build_tree(image)

    split_img = qt.reconstruct_split_image(image.shape, is_color=is_color)
    dsu, groups = qt.merge_leaves()
    merged_img = qt.reconstruct_merged_image(image.shape, groups, is_color=is_color)
    grid_img = qt.draw_quadtree_grid(image)

    # Compute reconstruction metrics (PSNR / MSE)
    if is_color:
        ref_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        merged_gray = cv2.cvtColor(merged_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = image
        merged_gray = merged_img

    mse = float(np.mean((ref_gray.astype(float) - merged_gray.astype(float)) ** 2))
    psnr = float(10 * np.log10(255**2 / (mse + 1e-10)))

    stats = {
        "num_leaves_split": len(qt.leaves),
        "num_regions_merged": len(groups),
        "reduction_ratio": float(len(qt.leaves) / max(1, len(groups))),
        "mse": mse,
        "psnr_db": psnr
    }

    return split_img, merged_img, grid_img, stats


def plot_split_merge_results(
    original: np.ndarray,
    grid_img: np.ndarray,
    split_img: np.ndarray,
    merged_img: np.ndarray,
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

    # Panel 2: Quadtree Grid Overlay
    axes[0, 1].imshow(to_rgb(grid_img))
    axes[0, 1].set_title(f"2. Quadtree Split Grid ({stats['num_leaves_split']} leaf blocks)", fontsize=13, fontweight='bold')
    axes[0, 1].axis("off")

    # Panel 3: Split-Only Reconstruction
    axes[1, 0].imshow(to_rgb(split_img), cmap=cmap)
    axes[1, 0].set_title("3. Split-Only Reconstruction", fontsize=13, fontweight='bold')
    axes[1, 0].axis("off")

    # Panel 4: Final Merged Segmentation
    axes[1, 1].imshow(to_rgb(merged_img), cmap=cmap)
    axes[1, 1].set_title(
        f"4. Split & Merge Result ({stats['num_regions_merged']} regions | PSNR: {stats['psnr_db']:.2f} dB)",
        fontsize=13, fontweight='bold'
    )
    axes[1, 1].axis("off")

    plt.suptitle("Quadtree Split-and-Merge Image Segmentation (Horowitz-Pavlidis)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Split-Merge] Figure saved to {save_path}")

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

    parser = argparse.ArgumentParser(description="Quadtree Split-and-Merge Image Segmentation")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file (generates sample if omitted)")
    parser.add_argument("--std-thresh", type=float, default=14.0, help="Standard deviation threshold for quadtree splitting")
    parser.add_argument("--min-size", type=int, default=4, help="Minimum block size for quadtree splitting")
    parser.add_argument("--merge-thresh", type=float, default=12.0, help="Mean difference threshold for merging adjacent blocks")
    parser.add_argument("--output", type=str, default="output/split_merge_result.png", help="Path to save output figure")
    parser.add_argument("--no-plot", action="store_true", help="Disable interactive matplotlib GUI display")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    img = get_or_create_sample_image(args.image)

    print(f"\n[Split-Merge] Running segmentation...")
    print(f"  - Input shape: {img.shape}")
    print(f"  - Parameters: std_thresh={args.std_thresh}, min_size={args.min_size}, merge_thresh={args.merge_thresh}")

    split_img, merged_img, grid_img, stats = split_and_merge_segmentation(
        img,
        std_threshold=args.std_thresh,
        min_size=args.min_size,
        merge_threshold=args.merge_thresh
    )

    print(f"\n[Split-Merge] Results Summary:")
    print(f"  - Quadtree Leaf Blocks (Split): {stats['num_leaves_split']}")
    print(f"  - Final Merged Regions:         {stats['num_regions_merged']}")
    print(f"  - Region Reduction Factor:     {stats['reduction_ratio']:.2f}x")
    print(f"  - Reconstruction MSE:          {stats['mse']:.2f}")
    print(f"  - Reconstruction PSNR:         {stats['psnr_db']:.2f} dB")

    plot_split_merge_results(
        img, grid_img, split_img, merged_img, stats,
        save_path=args.output,
        show_plot=not args.no_plot
    )
