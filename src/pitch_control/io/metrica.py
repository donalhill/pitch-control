"""
Metrica Sports tracking data loader with auto-download support.

Handles:
- Auto-downloading sample matches from GitHub
- Coordinate transformation (normalized 0-1 → meters with center origin)
- Playing direction normalization (home team attacks left→right)
- Velocity calculation with Savitzky-Golay smoothing
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import requests
from scipy.signal import savgol_filter

# Metrica sample data URLs
METRICA_BASE_URL = (
    "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
)

# Standard pitch dimensions (meters)
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


class MatchData(NamedTuple):
    """Container for a complete match dataset."""

    tracking_home: pd.DataFrame
    tracking_away: pd.DataFrame
    events: pd.DataFrame
    metadata: dict


def get_data_dir() -> Path:
    """Get the data directory, creating it if needed."""
    data_dir = Path(__file__).parents[4] / "data" / "metrica"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_sample_match(game_id: int = 1, force: bool = False) -> Path:
    """
    Download a Metrica sample match if not already cached.

    Args:
        game_id: Sample game number (1, 2, or 3)
        force: Re-download even if files exist

    Returns:
        Path to the game directory
    """
    if game_id not in (1, 2, 3):
        raise ValueError(f"game_id must be 1, 2, or 3, got {game_id}")

    data_dir = get_data_dir()
    game_dir = data_dir / f"Sample_Game_{game_id}"
    game_dir.mkdir(exist_ok=True)

    files = {
        f"Sample_Game_{game_id}_RawTrackingData_Home_Team.csv": "tracking_home",
        f"Sample_Game_{game_id}_RawTrackingData_Away_Team.csv": "tracking_away",
        f"Sample_Game_{game_id}_RawEventsData.csv": "events",
    }

    for filename, desc in files.items():
        filepath = game_dir / filename
        if filepath.exists() and not force:
            continue

        url = f"{METRICA_BASE_URL}/Sample_Game_{game_id}/{filename}"
        print(f"Downloading {desc} for game {game_id}...")

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        filepath.write_bytes(response.content)
        print(f"  Saved to {filepath}")

    return game_dir


def load_tracking_data(
    filepath: Path | str,
    team_name: str,
) -> pd.DataFrame:
    """
    Load tracking data for a single team.

    The Metrica format has:
    - Row 1: Team name repeated
    - Row 2: Player jersey numbers
    - Row 3: Column type (x/y)
    - Data starts at row 4

    Args:
        filepath: Path to the tracking CSV
        team_name: 'Home' or 'Away'

    Returns:
        DataFrame with columns like 'Home_11_x', 'Home_11_y', 'ball_x', 'ball_y'
    """
    # Read the jersey numbers from row 2
    with open(filepath) as f:
        # Skip team name row
        f.readline()
        # Read jersey numbers
        jerseys_line = f.readline().strip()

    jerseys = jerseys_line.split(",")[3:-2]  # Skip Period, Frame, Time and ball cols

    # Build column names
    columns = ["Period", "Frame", "Time"]
    for i in range(0, len(jerseys), 2):
        jersey = jerseys[i]
        if jersey:
            columns.append(f"{team_name}_{jersey}_x")
            columns.append(f"{team_name}_{jersey}_y")
    columns.extend(["ball_x", "ball_y"])

    # Read the actual data
    df = pd.read_csv(filepath, skiprows=3, names=columns, index_col="Frame")

    return df


def load_event_data(filepath: Path | str) -> pd.DataFrame:
    """Load event data from Metrica CSV."""
    events = pd.read_csv(filepath)

    # Standardize column names
    events.columns = [c.strip() for c in events.columns]

    return events


def to_metric_coordinates(
    data: pd.DataFrame,
    field_length: float = FIELD_LENGTH,
    field_width: float = FIELD_WIDTH,
) -> pd.DataFrame:
    """
    Convert Metrica's normalized (0-1) coordinates to meters with origin at center.

    Metrica uses:
    - x: 0 = left goal line, 1 = right goal line
    - y: 0 = top touchline, 1 = bottom touchline

    We convert to:
    - x: -field_length/2 to +field_length/2 (center = 0)
    - y: -field_width/2 to +field_width/2 (center = 0)
    """
    data = data.copy()

    # Find all x and y columns
    x_cols = [c for c in data.columns if c.endswith("_x")]
    y_cols = [c for c in data.columns if c.endswith("_y")]

    # Transform x: (0,1) → (-L/2, +L/2)
    data[x_cols] = (data[x_cols] - 0.5) * field_length

    # Transform y: (0,1) → (+W/2, -W/2) - note the flip for standard orientation
    data[y_cols] = -1 * (data[y_cols] - 0.5) * field_width

    return data


def find_goalkeeper(tracking: pd.DataFrame, team_name: str) -> str:
    """
    Find the goalkeeper by identifying the player closest to the goal line at kickoff.

    Returns the jersey number as a string.
    """
    # Get first frame of first half
    first_frame = tracking[tracking["Period"] == 1].iloc[0]

    # Get all player x positions for this team
    x_cols = [c for c in tracking.columns if c.startswith(f"{team_name}_") and c.endswith("_x")]

    # Find player with maximum absolute x (closest to either goal line)
    max_x = 0
    gk_jersey = None

    for col in x_cols:
        x_val = first_frame[col]
        if pd.notna(x_val) and abs(x_val) > max_x:
            max_x = abs(x_val)
            # Extract jersey number from column name like "Home_11_x"
            gk_jersey = col.split("_")[1]

    return gk_jersey


def find_playing_direction(tracking: pd.DataFrame, team_name: str) -> int:
    """
    Determine the team's playing direction in the first half.

    Returns:
        +1 if attacking left-to-right (positive x direction)
        -1 if attacking right-to-left (negative x direction)
    """
    gk_jersey = find_goalkeeper(tracking, team_name)
    gk_x_col = f"{team_name}_{gk_jersey}_x"

    # Get GK position at kickoff
    first_frame = tracking[tracking["Period"] == 1].iloc[0]
    gk_x = first_frame[gk_x_col]

    # If GK is at negative x (left side), team attacks right (+1)
    # If GK is at positive x (right side), team attacks left (-1)
    return 1 if gk_x < 0 else -1


def to_single_playing_direction(
    tracking_home: pd.DataFrame,
    tracking_away: pd.DataFrame,
    events: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Normalize so home team always attacks left-to-right (positive x).

    This flips coordinates as needed based on actual playing direction.
    """
    # Find home team's playing direction
    home_direction = find_playing_direction(tracking_home, "Home")

    tracking_home = tracking_home.copy()
    tracking_away = tracking_away.copy()

    # Get coordinate columns
    coord_cols_home = [c for c in tracking_home.columns if c.endswith(("_x", "_y"))]
    coord_cols_away = [c for c in tracking_away.columns if c.endswith(("_x", "_y"))]

    if home_direction == -1:
        # Home attacks right-to-left in first half, need to flip first half
        first_half_home = tracking_home["Period"] == 1
        first_half_away = tracking_away["Period"] == 1

        tracking_home.loc[first_half_home, coord_cols_home] *= -1
        tracking_away.loc[first_half_away, coord_cols_away] *= -1
    else:
        # Home attacks left-to-right in first half, need to flip second half
        second_half_home = tracking_home["Period"] == 2
        second_half_away = tracking_away["Period"] == 2

        tracking_home.loc[second_half_home, coord_cols_home] *= -1
        tracking_away.loc[second_half_away, coord_cols_away] *= -1

    # Also flip events if provided
    if events is not None:
        events = events.copy()
        coord_event_cols = ["Start X", "Start Y", "End X", "End Y"]
        existing_cols = [c for c in coord_event_cols if c in events.columns]

        if home_direction == -1:
            first_half_events = events["Period"] == 1
            events.loc[first_half_events, existing_cols] *= -1
        else:
            second_half_events = events["Period"] == 2
            events.loc[second_half_events, existing_cols] *= -1

    return tracking_home, tracking_away, events


def calculate_velocities(
    tracking: pd.DataFrame,
    team_name: str,
    smoothing: bool = True,
    window: int = 7,
    polyorder: int = 1,
    max_speed: float = 12.0,
) -> pd.DataFrame:
    """
    Calculate player velocities from position data.

    Uses Savitzky-Golay filter for smoothing and derivative calculation.
    Processes first and second halves separately to avoid discontinuities.

    Args:
        tracking: Tracking DataFrame with position columns
        team_name: 'Home' or 'Away'
        smoothing: Whether to apply Savitzky-Golay smoothing
        window: Window size for smoothing (must be odd)
        polyorder: Polynomial order for smoothing
        max_speed: Maximum realistic speed (m/s) for outlier removal

    Returns:
        DataFrame with added velocity columns (e.g., 'Home_11_vx', 'Home_11_vy')
    """
    tracking = tracking.copy()

    # Get time step (assuming 25 Hz = 0.04s)
    dt = tracking["Time"].diff().median()
    if pd.isna(dt):
        dt = 0.04  # Default to 25 Hz

    # Find all players for this team
    x_cols = [c for c in tracking.columns if c.startswith(f"{team_name}_") and c.endswith("_x")]

    # Process each half separately to avoid discontinuity at halftime
    for x_col in x_cols:
        y_col = x_col.replace("_x", "_y")
        vx_col = x_col.replace("_x", "_vx")
        vy_col = x_col.replace("_x", "_vy")

        if y_col not in tracking.columns:
            continue

        # Initialize velocity columns
        tracking[vx_col] = np.nan
        tracking[vy_col] = np.nan

        # Process each half separately
        for period in tracking["Period"].unique():
            if pd.isna(period):
                continue

            period_mask = tracking["Period"] == period
            period_indices = tracking.index[period_mask]

            if len(period_indices) < window:
                continue

            x = tracking.loc[period_indices, x_col].values
            y = tracking.loc[period_indices, y_col].values

            if smoothing and len(x) > window:
                # Savitzky-Golay filter for smoothed derivative
                # deriv=1 gives first derivative
                vx = savgol_filter(x, window, polyorder, deriv=1, delta=dt)
                vy = savgol_filter(y, window, polyorder, deriv=1, delta=dt)
            else:
                # Simple finite difference
                vx = np.gradient(x, dt)
                vy = np.gradient(y, dt)

            # Handle NaN propagation
            vx[np.isnan(x)] = np.nan
            vy[np.isnan(y)] = np.nan

            # Clip unrealistic speeds within this period
            speed = np.sqrt(vx**2 + vy**2)
            too_fast = speed > max_speed
            vx[too_fast] = np.nan
            vy[too_fast] = np.nan

            tracking.loc[period_indices, vx_col] = vx
            tracking.loc[period_indices, vy_col] = vy

    return tracking


def merge_tracking_data(
    tracking_home: pd.DataFrame,
    tracking_away: pd.DataFrame,
) -> pd.DataFrame:
    """Merge home and away tracking into a single DataFrame."""
    # Remove duplicate ball columns from home
    home_cols = [c for c in tracking_home.columns if not c.startswith("ball_")]
    away_cols = [c for c in tracking_away.columns if c.startswith("Away_")]

    merged = tracking_home[home_cols].copy()

    for col in away_cols:
        merged[col] = tracking_away[col]

    # Add ball from home (or away, should be same)
    merged["ball_x"] = tracking_home["ball_x"]
    merged["ball_y"] = tracking_home["ball_y"]

    return merged


def load_match(
    game_id: int = 1,
    data_dir: Path | str | None = None,
    auto_download: bool = True,
) -> MatchData:
    """
    Load a complete Metrica sample match with all preprocessing.

    This is the main entry point for loading data.

    Args:
        game_id: Sample game number (1, 2, or 3)
        data_dir: Custom data directory (default: project data/metrica/)
        auto_download: Whether to download if data not found

    Returns:
        MatchData with tracking_home, tracking_away, events, and metadata
    """
    if data_dir is None:
        data_dir = get_data_dir()
    else:
        data_dir = Path(data_dir)

    game_dir = data_dir / f"Sample_Game_{game_id}"

    # Check if data exists, download if needed
    tracking_home_path = game_dir / f"Sample_Game_{game_id}_RawTrackingData_Home_Team.csv"

    if not tracking_home_path.exists():
        if auto_download:
            download_sample_match(game_id)
        else:
            raise FileNotFoundError(
                f"Game {game_id} not found at {game_dir}. "
                "Set auto_download=True or download manually."
            )

    # Load raw data
    print(f"Loading game {game_id}...")

    tracking_home = load_tracking_data(
        game_dir / f"Sample_Game_{game_id}_RawTrackingData_Home_Team.csv",
        "Home",
    )
    tracking_away = load_tracking_data(
        game_dir / f"Sample_Game_{game_id}_RawTrackingData_Away_Team.csv",
        "Away",
    )
    events = load_event_data(game_dir / f"Sample_Game_{game_id}_RawEventsData.csv")

    # Convert to metric coordinates
    tracking_home = to_metric_coordinates(tracking_home)
    tracking_away = to_metric_coordinates(tracking_away)

    # Normalize playing direction (home always attacks left→right)
    tracking_home, tracking_away, events = to_single_playing_direction(
        tracking_home, tracking_away, events
    )

    # Calculate velocities
    tracking_home = calculate_velocities(tracking_home, "Home")
    tracking_away = calculate_velocities(tracking_away, "Away")

    # Find goalkeepers
    home_gk = find_goalkeeper(tracking_home, "Home")
    away_gk = find_goalkeeper(tracking_away, "Away")

    metadata = {
        "game_id": game_id,
        "home_gk": home_gk,
        "away_gk": away_gk,
        "field_length": FIELD_LENGTH,
        "field_width": FIELD_WIDTH,
        "fps": 25,
    }

    print(f"  Loaded {len(tracking_home)} frames")
    print(f"  Home GK: #{home_gk}, Away GK: #{away_gk}")

    return MatchData(
        tracking_home=tracking_home,
        tracking_away=tracking_away,
        events=events,
        metadata=metadata,
    )


def get_frame_data(
    tracking_home: pd.DataFrame,
    tracking_away: pd.DataFrame,
    frame: int,
) -> dict:
    """
    Extract player positions and velocities for a single frame.

    Returns:
        Dict with 'home' and 'away' sub-dicts, each containing:
        - positions: (n_players, 2) array
        - velocities: (n_players, 2) array
        - jerseys: list of jersey numbers
        - ball: (2,) array
    """
    home_frame = tracking_home.loc[frame]
    away_frame = tracking_away.loc[frame]

    def extract_team(frame_data: pd.Series, team: str) -> dict:
        x_cols = [c for c in frame_data.index if c.startswith(f"{team}_") and c.endswith("_x")]

        positions = []
        velocities = []
        jerseys = []

        for x_col in x_cols:
            jersey = x_col.split("_")[1]
            y_col = f"{team}_{jersey}_y"
            vx_col = f"{team}_{jersey}_vx"
            vy_col = f"{team}_{jersey}_vy"

            x, y = frame_data[x_col], frame_data[y_col]

            if pd.notna(x) and pd.notna(y):
                positions.append([x, y])

                vx = frame_data.get(vx_col, 0) or 0
                vy = frame_data.get(vy_col, 0) or 0
                velocities.append([vx if pd.notna(vx) else 0, vy if pd.notna(vy) else 0])

                jerseys.append(jersey)

        return {
            "positions": np.array(positions) if positions else np.zeros((0, 2)),
            "velocities": np.array(velocities) if velocities else np.zeros((0, 2)),
            "jerseys": jerseys,
        }

    ball_x = home_frame.get("ball_x", np.nan)
    ball_y = home_frame.get("ball_y", np.nan)
    ball = np.array([ball_x, ball_y])

    return {
        "home": extract_team(home_frame, "Home"),
        "away": extract_team(away_frame, "Away"),
        "ball": ball,
        "frame": frame,
        "time": home_frame.get("Time", 0),
        "period": home_frame.get("Period", 1),
    }
