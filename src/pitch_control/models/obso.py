"""
Off-Ball Scoring Opportunity (OBSO) Module

From Spearman (2018) "Beyond Expected Goals":

OBSO = P(Score | control, transition) = Σ P(S|C,T) × P(C|T) × P(T)

Per-location OBSO surface: Pitch Control × Scoring Probability
Integrated OBSO: weighted by Transition probability (in precompute.py)

Parameters from Spearman Table 1:
- λ = 3.99 Hz (control rate)
- κ = 1.72 (defensive advantage)
- β = 0.48 (scoring probability adjustment)
- σ = 23.9m (transition distance)
- α = 1.04 (transition control weighting)

Player-level OBSO attribution shows who is creating valuable space.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pitch_control.models.epv import create_epv_for_grid, load_epv_grid
from pitch_control.models.pitch_control import (
    PitchControlParams,
    compute_pitch_control,
    create_pitch_grid,
)


def compute_obso(
    pitch_control: np.ndarray,
    epv: np.ndarray,
) -> np.ndarray:
    """
    Compute Off-Ball Scoring Opportunity surface.

    OBSO = Pitch Control × EPV

    Args:
        pitch_control: (n_grid_y, n_grid_x) attacking team's control probability
        epv: (n_grid_y, n_grid_x) expected possession value

    Returns:
        (n_grid_y, n_grid_x) OBSO surface
    """
    return pitch_control * epv


def compute_total_obso(obso_surface: np.ndarray, grid: np.ndarray) -> float:
    """
    Compute total OBSO by integrating over the pitch.

    This gives a single number representing the team's total
    off-ball scoring opportunity at this moment.

    Args:
        obso_surface: (n_grid_y, n_grid_x) OBSO values
        grid: (n_grid_y, n_grid_x, 2) grid coordinates

    Returns:
        Total integrated OBSO value
    """
    # Compute grid cell area
    dx = grid[0, 1, 0] - grid[0, 0, 0]
    dy = grid[1, 0, 1] - grid[0, 0, 1]
    cell_area = abs(dx * dy)

    # Integrate
    return np.sum(obso_surface) * cell_area


def compute_player_obso(
    player_control: np.ndarray,
    epv: np.ndarray,
) -> np.ndarray:
    """
    Compute per-player OBSO contribution.

    For each player, integrates their individual pitch control × EPV
    to show who is creating/occupying dangerous space.

    Args:
        player_control: (n_players, n_grid_y, n_grid_x) per-player control
        epv: (n_grid_y, n_grid_x) expected possession value

    Returns:
        (n_players,) total OBSO contribution per player
    """
    # Multiply each player's control by EPV and sum over grid
    # player_control: (n_players, ny, nx)
    # epv: (ny, nx) -> broadcast to (1, ny, nx)
    player_obso_surfaces = player_control * epv[np.newaxis, :, :]

    # Sum over spatial dimensions
    player_obso = player_obso_surfaces.sum(axis=(1, 2))

    return player_obso


def compute_player_obso_surfaces(
    player_control: np.ndarray,
    epv: np.ndarray,
) -> np.ndarray:
    """
    Compute per-player OBSO surface (not just the total).

    Useful for visualization of where each player is creating danger.

    Args:
        player_control: (n_players, n_grid_y, n_grid_x) per-player control
        epv: (n_grid_y, n_grid_x) expected possession value

    Returns:
        (n_players, n_grid_y, n_grid_x) per-player OBSO surfaces
    """
    return player_control * epv[np.newaxis, :, :]


def compute_obso_for_frame(
    attacking_positions: np.ndarray,
    attacking_velocities: np.ndarray,
    defending_positions: np.ndarray,
    defending_velocities: np.ndarray,
    ball_position: np.ndarray,
    params: PitchControlParams | None = None,
    epv_grid: np.ndarray | None = None,
    epv_x: np.ndarray | None = None,
    epv_y: np.ndarray | None = None,
    grid: np.ndarray | None = None,
    attacking_direction: int = 1,
    ball_model: str = "simple",
    ball_flight_model=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute full OBSO analysis for a single frame.

    Combines pitch control computation with EPV fusion.

    Args:
        attacking_positions: (n_att, 2) attacking player positions
        attacking_velocities: (n_att, 2) attacking player velocities
        defending_positions: (n_def, 2) defending player positions
        defending_velocities: (n_def, 2) defending player velocities
        ball_position: (2,) ball position
        params: Model parameters
        epv_grid: Pre-loaded EPV grid (loads default if None)
        epv_x, epv_y: EPV coordinate arrays
        grid: Pre-computed pitch grid (creates one if None)
        attacking_direction: +1 if attacking right (home), -1 if attacking left (away)
        ball_model: "simple", "parabolic" (default), or "trajectory"
        ball_flight_model: Pre-loaded BallFlightModel (for trajectory model)

    Returns:
        Tuple of:
        - obso_surface: (n_grid_y, n_grid_x) OBSO values
        - pitch_control: (n_grid_y, n_grid_x) raw pitch control
        - player_obso: (n_att + n_def,) per-player OBSO totals
        - grid: (n_grid_y, n_grid_x, 2) grid coordinates
    """
    # Load EPV if not provided
    if epv_grid is None:
        epv_grid, epv_x, epv_y = load_epv_grid()

    # Compute pitch control
    pitch_control, grid, player_control = compute_pitch_control(
        attacking_positions,
        attacking_velocities,
        defending_positions,
        defending_velocities,
        ball_position,
        params,
        grid=grid,
        attack_direction=attacking_direction,
        ball_model=ball_model,
        ball_flight_model=ball_flight_model,
    )

    # Create EPV array matching our grid (flip if attacking left)
    epv_for_grid = create_epv_for_grid(
        grid, epv_grid, epv_x, epv_y, attacking_direction=attacking_direction
    )

    # Compute OBSO
    obso_surface = compute_obso(pitch_control, epv_for_grid)

    # Compute per-player OBSO
    player_obso = compute_player_obso(player_control, epv_for_grid)

    return obso_surface, pitch_control, player_obso, grid


class OBSOAnalyzer:
    """
    High-level interface for OBSO analysis across a match.

    Caches EPV grid and pitch grid for efficient frame-by-frame processing.
    """

    def __init__(
        self,
        params: PitchControlParams | None = None,
        epv_filepath: str | None = None,
        ball_model: str = "simple",
    ):
        """
        Initialize the analyzer.

        Args:
            params: Model parameters (uses defaults if None)
            epv_filepath: Path to EPV grid (uses default if None)
            ball_model: Ball flight model - "simple" (constant 15 m/s),
                "parabolic" (projectile motion, default), or "trajectory"
                (full drag model with lookup table).
        """
        from pitch_control.models.pitch_control import default_model_params

        self.params = params or default_model_params()
        self.ball_model = ball_model

        # Load EPV
        self.epv_grid, self.epv_x, self.epv_y = load_epv_grid(epv_filepath)

        # Pre-compute pitch grid
        self.grid = create_pitch_grid(self.params)

        # Pre-compute EPV for both attack directions (home=+1, away=-1)
        # Home attacks left→right (positive x), so default EPV is correct
        self.epv_for_home = create_epv_for_grid(
            self.grid, self.epv_grid, self.epv_x, self.epv_y, attacking_direction=1
        )
        # Away attacks right→left (negative x), so flip EPV
        self.epv_for_away = create_epv_for_grid(
            self.grid, self.epv_grid, self.epv_x, self.epv_y, attacking_direction=-1
        )

        # Pre-load ball flight model if using trajectory (drag) model
        self.ball_flight_model = None
        if ball_model == "trajectory":
            from pitch_control.models.ball_trajectory import get_ball_flight_model
            self.ball_flight_model = get_ball_flight_model()

    def analyze_frame(
        self,
        frame_data: dict,
        attacking_team: str = "home",
    ) -> dict:
        """
        Run full OBSO analysis on a frame.

        Args:
            frame_data: Output from metrica.get_frame_data()
            attacking_team: "home" or "away"

        Returns:
            Dict containing:
            - obso: OBSO surface
            - pitch_control: Raw pitch control surface
            - player_obso: Dict mapping player IDs to OBSO totals
            - total_obso: Integrated OBSO value
            - grid: Grid coordinates
        """
        if attacking_team == "home":
            att_data = frame_data["home"]
            def_data = frame_data["away"]
            attack_direction = 1  # Home attacks right (positive x)
        else:
            att_data = frame_data["away"]
            def_data = frame_data["home"]
            attack_direction = -1  # Away attacks left (negative x)

        # Compute pitch control
        pitch_control, _, player_control = compute_pitch_control(
            att_data["positions"],
            att_data["velocities"],
            def_data["positions"],
            def_data["velocities"],
            frame_data["ball"],
            self.params,
            grid=self.grid,
            attack_direction=attack_direction,
            ball_model=self.ball_model,
            ball_flight_model=self.ball_flight_model,
        )

        # Select EPV based on attacking team direction
        epv_for_grid = self.epv_for_home if attacking_team == "home" else self.epv_for_away

        # Compute OBSO
        obso_surface = compute_obso(pitch_control, epv_for_grid)
        total_obso = compute_total_obso(obso_surface, self.grid)

        # Per-player OBSO
        player_obso_values = compute_player_obso(player_control, epv_for_grid)

        # Build player dict
        n_att = len(att_data["jerseys"])
        player_obso_dict = {}

        for i, jersey in enumerate(att_data["jerseys"]):
            key = f"{attacking_team}_{jersey}"
            player_obso_dict[key] = player_obso_values[i]

        defending_team = "away" if attacking_team == "home" else "home"
        for i, jersey in enumerate(def_data["jerseys"]):
            key = f"{defending_team}_{jersey}"
            player_obso_dict[key] = player_obso_values[n_att + i]

        return {
            "obso": obso_surface,
            "pitch_control": pitch_control,
            "player_obso": player_obso_dict,
            "total_obso": total_obso,
            "grid": self.grid,
            "epv": epv_for_grid,
            "frame": frame_data["frame"],
            "time": frame_data["time"],
        }

    def analyze_frames(
        self,
        tracking_home,
        tracking_away,
        frames: list[int],
        attacking_team: str = "home",
        show_progress: bool = True,
    ) -> list[dict]:
        """
        Analyze multiple frames.

        Args:
            tracking_home: Home team tracking DataFrame
            tracking_away: Away team tracking DataFrame
            frames: List of frame numbers to analyze
            attacking_team: "home" or "away"
            show_progress: Whether to show progress bar

        Returns:
            List of analysis dicts, one per frame
        """
        from pitch_control.io.metrica import get_frame_data

        results = []

        iterator = frames
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(frames, desc="Analyzing frames")
            except ImportError:
                pass

        for frame in iterator:
            frame_data = get_frame_data(tracking_home, tracking_away, frame)
            result = self.analyze_frame(frame_data, attacking_team)
            results.append(result)

        return results
