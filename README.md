# Image Segmentation & Pattern Recognition Project
## Quadtree Split-and-Merge & Voronoi Diagram Partitioning

This project implements two core image segmentation paradigms in Python using **OpenCV**, **NumPy**, **SciPy**, and **Matplotlib**:
1. **Quadtree Decomposition & Region Merging (Horowitz-Pavlidis Algorithm)**
2. **Voronoi Diagram Partitioning & Adaptive Stippling Mosaic (SciPy Spatial)**

Designed for academic coursework in **Reconnaissance des Formes / Computer Vision (M2)**, with modular code, quantitative evaluation metrics (MSE, PSNR, region counts), and automated publication-quality visual plots ready for lab reports.

---

## 📁 Project Structure

```text
image-segmentation-quadtree-voronoi/
│
├── requirements.txt            # Project dependencies (numpy, opencv-python, scipy, matplotlib)
├── .gitignore                  # Ignores virtualenvs, caches, and generated images
├── README.md                   # Complete documentation and execution guide
├── utils.py                    # Benchmark image generator and image I/O helpers
├── main.py                     # Unified entry point for running both or either algorithm
│
├── split_merge/
│   ├── __init__.py
│   └── quadtree.py             # Quadtree decomposition, adjacency graph, and Union-Find merge
│
├── voronoi/
│   ├── __init__.py
│   └── voronoi_art.py          # Voronoi partitioning, adaptive stippling, and mosaic generator
│
├── assets/
│   └── sample.png              # Auto-generated synthetic test benchmark image
│
└── output/                     # Generated figures (screenshots for report)
    ├── split_merge_result.png
    ├── voronoi_result.png
    └── comparison_result.png
```

---

## 🧠 Theoretical Overview

### 1. Split-and-Merge with Quadtree
The **Split-and-Merge** algorithm is a region-based segmentation approach introduced by *Horowitz and Pavlidis (1974)*:
- **Split Phase (Quadtree)**:
  - Starts with the whole image as the root node.
  - A homogeneity predicate $P(R)$ evaluates region $R$. Here, $P(R) \iff \sigma(R) \le \theta_{\text{split}}$, where $\sigma$ is the standard deviation of pixel intensities within $R$.
  - If $P(R)$ is `False` and dimensions exceed `min_size`, the block is partitioned into 4 quadrants: North-West (NW), North-East (NE), South-West (SW), and South-East (SE).
  - Recursion terminates when $P(R)$ is `True` or the minimum block size is reached.
- **Merge Phase (Disjoint Set Union / Union-Find)**:
  - Splitting alone produces square block artifacts and fragments homogeneous regions across quadrant borders.
  - Two adjacent leaf blocks $R_i, R_j$ sharing a boundary edge are evaluated:
    $$\Delta\mu = |\mu(R_i) - \mu(R_j)| \le \theta_{\text{merge}}$$
  - Connected components are grouped using **Disjoint Set Union (DSU)** with path compression and rank heuristics.
  - Each merged region is rendered with its area-weighted mean color.

### 2. Voronoi Diagram Partitioning & Mosaic Stippling
A **Voronoi Tessellation** partitions 2D image space $\Omega$ based on a set of discrete seed points $S = \{p_1, p_2, \dots, p_n\}$:
$$V(p_i) = \{ q \in \Omega \mid \|q - p_i\| \le \|q - p_j\|, \; \forall j \neq i \}$$
- **Adaptive Sampling (Importance Stippling)**:
  - Rather than placing seeds purely at random, image gradient magnitude (Sobel operators: $\|\nabla I\|$) is computed and normalized into a 2D probability density function (PDF).
  - Seed points are sampled according to this distribution, concentrating fine, dense Voronoi cells along edges/contours and large cells in flat regions.
- **Vectorized Spatial Assignment**:
  - Uses `scipy.spatial.cKDTree` for $O(\log n)$ nearest-seed queries across all image pixels.
  - The mean RGB color/intensity is calculated for each Voronoi cell.
  - Cell boundary ridges are highlighted to create a clean stained-glass / mosaic effect.

---

## 🚀 Installation & Setup

1. Open your terminal in the project directory:
   ```powershell
   cd "c:\Users\Shift\Desktop\Univ\Studty\M2\Reconnaissance des Formes\TP\image-segmentation-quadtree-voronoi"
   ```

2. (Optional) Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 💻 Running the Code

### 1. Run Complete Pipeline (Both Algorithms + Comparison)
Runs both algorithms out of the box on the default benchmark image, displaying interactive figures and saving high-resolution figures into `output/`:
```powershell
python main.py
```

### 2. Run Only Quadtree Split-and-Merge
```powershell
python main.py --mode split_merge
```
Or directly run the module:
```powershell
python split_merge/quadtree.py
```

**Customizing Split-and-Merge parameters:**
- `--std-thresh <float>`: Splitting homogeneity threshold (default: `14.0`). Lower values produce finer splits.
- `--min-size <int>`: Minimum block dimension before stopping splits (default: `4`).
- `--merge-thresh <float>`: Mean difference threshold for merging neighbors (default: `12.0`). Higher values merge more aggressively.

Example:
```powershell
python main.py --mode split_merge --std-thresh 10.0 --min-size 4 --merge-thresh 15.0
```

### 3. Run Only Voronoi Partitioning
```powershell
python main.py --mode voronoi
```
Or directly run the module:
```powershell
python voronoi/voronoi_art.py
```

**Customizing Voronoi parameters:**
- `--points <int>`: Total number of seed points (default: `1500`).
- `--sampling {adaptive, random, grid}`: Sampling method (default: `adaptive`).
- `--edge-weight <float>`: Edge importance weight for adaptive sampling between `0.0` and `1.0` (default: `0.8`).

Example:
```powershell
python main.py --mode voronoi --points 2500 --sampling adaptive --edge-weight 0.9
```

### 4. Running on Your Own Image
Simply pass the path to any `.png`, `.jpg`, or `.bmp` file via `--image`:
```powershell
python main.py --image "path/to/your/photo.jpg"
```
*(If no image is provided, a synthetic multi-shape gradient benchmark image is automatically created and stored in `assets/sample.png`.)*

### 5. Headless Mode (No GUI Popups)
To run without displaying interactive matplotlib popup windows (e.g. for batch processing or remote scripts), add `--no-plot`:
```powershell
python main.py --no-plot
```

---

## 📊 Output Visualizations & Metrics for Lab Report

All figures are automatically exported at **300 DPI** to the `output/` directory:

| File | Content | Report Usage |
| :--- | :--- | :--- |
| `output/split_merge_result.png` | 4-panel comparison: (1) Original, (2) Quadtree Grid Overlay, (3) Split-only Reconstruction, (4) Merged Segmentation | Illustrates quadtree recursive splitting and the effect of neighbor merging. |
| `output/voronoi_result.png` | 4-panel comparison: (1) Original, (2) Seed Point Stippling Distribution, (3) Voronoi Mosaic with Ridges, (4) Clean Segmented Image | Illustrates spatial tessellation, adaptive point density along gradients, and cell reconstruction. |
| `output/comparison_result.png` | 3-panel comparative overview: Original vs Quadtree vs Voronoi | Direct visual comparison of regional hierarchical vs geometric dual segmentation models. |

### Quantitative Metrics Computed:
- **Number of Regions / Cells**: Demonstrates data compression and abstraction level.
- **Region Reduction Factor**: Ratio of initial quadtree leaf blocks to final merged connected components.
- **Mean Squared Error (MSE)**: $MSE = \frac{1}{HW}\sum_{x,y} (I(x,y) - \hat{I}(x,y))^2$
- **Peak Signal-to-Noise Ratio (PSNR)**: $PSNR = 10 \cdot \log_{10}\left(\frac{255^2}{MSE}\right)$ (dB)
