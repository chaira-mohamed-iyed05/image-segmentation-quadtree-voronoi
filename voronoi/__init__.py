"""
Voronoi Diagram Partitioning and Stippling / Artistic Segmentation.
"""

def __getattr__(name):
    if name in ("VoronoiSegmenter", "generate_voronoi_segmentation", "plot_voronoi_results"):
        from . import voronoi_art
        return getattr(voronoi_art, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["VoronoiSegmenter", "generate_voronoi_segmentation"]
