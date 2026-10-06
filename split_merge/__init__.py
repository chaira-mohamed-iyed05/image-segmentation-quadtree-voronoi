"""
Split-and-Merge Image Segmentation using Quadtree.
"""

def __getattr__(name):
    if name in ("QuadTree", "split_and_merge_segmentation", "plot_split_merge_results"):
        from . import quadtree
        return getattr(quadtree, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["QuadTree", "split_and_merge_segmentation"]
