"""
Expected Possession Value (EPV) Module

EPV represents the probability that a possession will result in a goal,
given the current ball position. This module provides:
- Loading of pre-computed EPV grids
- Interpolation to arbitrary positions
- Grid generation utilities
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline


# Default field dimensions
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


def get_default_epv_path() -> Path:
    """Get path to the default EPV grid file."""
    return Path(__file__).parents[3] / "data" / "epv_grid.csv"


def load_epv_grid(
    filepath: Path | str | None = None,
    field_length: float = FIELD_LENGTH,
    field_width: float = FIELD_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load EPV grid from CSV file.

    The EPV grid represents goal probability from each position on the pitch.
    Values range from ~0.01 (defensive areas) to ~0.6 (in front of goal).

    Args:
        filepath: Path to EPV CSV (uses default if None)
        field_length: Pitch length in meters
        field_width: Pitch width in meters

    Returns:
        Tuple of:
        - epv_grid: (n_rows, n_cols) array of EPV values
        - x_coords: (n_cols,) array of x coordinates
        - y_coords: (n_rows,) array of y coordinates
    """
    if filepath is None:
        filepath = get_default_epv_path()

    filepath = Path(filepath)

    if not filepath.exists():
        # Generate default EPV grid if file doesn't exist
        print(f"EPV grid not found at {filepath}, generating default...")
        epv_grid, x_coords, y_coords = generate_default_epv_grid(
            field_length, field_width
        )
        save_epv_grid(epv_grid, x_coords, y_coords, filepath)
        return epv_grid, x_coords, y_coords

    # Load from file
    epv_grid = np.loadtxt(filepath, delimiter=",")

    # Generate coordinate arrays matching the grid
    n_rows, n_cols = epv_grid.shape
    x_coords = np.linspace(-field_length / 2, field_length / 2, n_cols)
    y_coords = np.linspace(-field_width / 2, field_width / 2, n_rows)

    return epv_grid, x_coords, y_coords


def save_epv_grid(
    epv_grid: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    filepath: Path | str,
) -> None:
    """Save EPV grid to CSV file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(filepath, epv_grid, delimiter=",", fmt="%.6f")


def generate_default_epv_grid(
    field_length: float = FIELD_LENGTH,
    field_width: float = FIELD_WIDTH,
    n_cols: int = 50,
    n_rows: int = 32,
    beta: float = 0.48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate scoring probability grid using Spearman's model (Equation 7).

    S(r|β) = Sf(|r - rg|)^β

    Where Sf(d) is the empirical scoring probability vs distance (Figure 4),
    approximated as exponential decay fitted to the curve:
      - At 0m: ~0.9
      - At 10m: ~0.1
      - At 20m: ~0.02
      - At 40m+: ~0

    The β parameter (default 0.48 from Table 1) adjusts for shot selection bias.
    β < 1 increases scoring probability (players only shoot when confident).

    Args:
        field_length: Pitch length in meters
        field_width: Pitch width in meters
        n_cols: Grid columns
        n_rows: Grid rows
        beta: Shot selection bias parameter (Table 1 MAP = 0.48)

    Returns:
        Tuple of (scoring_grid, x_coords, y_coords)
    """
    x_coords = np.linspace(-field_length / 2, field_length / 2, n_cols)
    y_coords = np.linspace(-field_width / 2, field_width / 2, n_rows)

    xx, yy = np.meshgrid(x_coords, y_coords)

    # Goal position (center of goal on right side for attacking team)
    goal_x = field_length / 2
    goal_y = 0

    # Distance to goal center
    dist_to_goal = np.sqrt((xx - goal_x) ** 2 + (yy - goal_y) ** 2)

    # Empirical scoring probability Sf(d) from Figure 4
    # Fitted as exponential decay: Sf(d) ≈ 0.9 * exp(-d/4.5)
    # This gives: Sf(0)≈0.9, Sf(10)≈0.1, Sf(20)≈0.01
    sf = 0.9 * np.exp(-dist_to_goal / 4.5)

    # Apply beta parameter (Equation 7): S(r|β) = Sf(d)^β
    # With β=0.48, this raises the curve (0.1^0.48 ≈ 0.32)
    scoring_prob = np.power(sf, beta)

    # Ensure minimum value for numerical stability
    scoring_prob = np.maximum(scoring_prob, 1e-6)

    return scoring_prob, x_coords, y_coords


class EPVInterpolator:
    """
    Fast EPV interpolation using bivariate splines.

    Provides efficient lookup of EPV values at arbitrary positions.
    """

    def __init__(
        self,
        epv_grid: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
    ):
        """
        Initialize the interpolator.

        Args:
            epv_grid: (n_rows, n_cols) array of EPV values
            x_coords: (n_cols,) array of x coordinates
            y_coords: (n_rows,) array of y coordinates
        """
        self.epv_grid = epv_grid
        self.x_coords = x_coords
        self.y_coords = y_coords

        # Create spline interpolator
        # Note: RectBivariateSpline expects (y, x) ordering
        self._spline = RectBivariateSpline(
            y_coords, x_coords, epv_grid, kx=3, ky=3
        )

        # Store bounds for clipping
        self.x_min, self.x_max = x_coords[0], x_coords[-1]
        self.y_min, self.y_max = y_coords[0], y_coords[-1]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get EPV values at specified positions.

        Args:
            x: x coordinates (any shape)
            y: y coordinates (same shape as x)

        Returns:
            EPV values (same shape as x)
        """
        # Clip to field bounds
        x_clipped = np.clip(x, self.x_min, self.x_max)
        y_clipped = np.clip(y, self.y_min, self.y_max)

        # Evaluate spline
        return self._spline(y_clipped, x_clipped, grid=False)

    def get_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        Get EPV values at grid positions.

        Args:
            grid: (n_grid_y, n_grid_x, 2) array of (x, y) coordinates

        Returns:
            (n_grid_y, n_grid_x) array of EPV values
        """
        x = grid[..., 0]
        y = grid[..., 1]
        return self(x, y)


def get_epv_at_location(
    position: np.ndarray,
    epv_grid: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    attacking_direction: int = 1,
) -> float:
    """
    Get EPV value at a specific position.

    Args:
        position: (2,) array [x, y]
        epv_grid: EPV grid array
        x_coords: x coordinate array
        y_coords: y coordinate array
        attacking_direction: +1 if attacking right, -1 if attacking left

    Returns:
        EPV value at the position
    """
    x, y = position

    # Flip if attacking left
    if attacking_direction == -1:
        x = -x

    # Find nearest grid indices
    ix = np.searchsorted(x_coords, x)
    iy = np.searchsorted(y_coords, y)

    # Clip to valid range
    ix = np.clip(ix, 0, len(x_coords) - 1)
    iy = np.clip(iy, 0, len(y_coords) - 1)

    return epv_grid[iy, ix]


def create_epv_for_grid(
    grid: np.ndarray,
    epv_grid: np.ndarray | None = None,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    attacking_direction: int = 1,
) -> np.ndarray:
    """
    Create EPV array matching a pitch control grid.

    Args:
        grid: (n_grid_y, n_grid_x, 2) pitch control grid
        epv_grid: Pre-loaded EPV grid (loads default if None)
        x_coords: EPV x coordinates
        y_coords: EPV y coordinates
        attacking_direction: +1 if attacking right, -1 if attacking left

    Returns:
        (n_grid_y, n_grid_x) EPV array aligned with the input grid
    """
    if epv_grid is None:
        epv_grid, x_coords, y_coords = load_epv_grid()

    # Flip EPV grid if attacking team goes left (Laurie's convention)
    if attacking_direction == -1:
        epv_grid = np.fliplr(epv_grid)

    interpolator = EPVInterpolator(epv_grid, x_coords, y_coords)
    return interpolator.get_grid(grid)
