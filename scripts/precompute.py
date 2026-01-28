#!/usr/bin/env python3
"""
Pre-compute pitch control and OBSO for event frames in a match.

Following Spearman's approach: analyze frames associated with events
(passes, shots) rather than every tracking frame.

Usage:
    python scripts/precompute.py --game-id 1
    python scripts/precompute.py --game-id 1 --event-types Pass Shot
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pitch_control.io.metrica import load_match, get_frame_data
from pitch_control.models.pitch_control import (
    compute_pitch_control,
    create_pitch_grid,
    default_model_params,
)
from pitch_control.models.obso import OBSOAnalyzer


def get_event_frames(events: pd.DataFrame, event_types: list[str] | None = None) -> pd.DataFrame:
    """
    Extract events with valid frame references.

    Args:
        events: Events DataFrame from Metrica
        event_types: List of event types to include (None = all)

    Returns:
        Filtered events DataFrame
    """
    # Filter by event type if specified
    if event_types:
        mask = events['Type'].isin(event_types)
        filtered = events[mask].copy()
    else:
        filtered = events.copy()

    # Ensure we have Start Frame column
    if 'Start Frame' not in filtered.columns:
        raise ValueError("Events missing 'Start Frame' column")

    # Remove events with invalid frames
    filtered = filtered[filtered['Start Frame'].notna()]
    filtered['Start Frame'] = filtered['Start Frame'].astype(int)

    return filtered


def precompute_events(
    game_id: int = 1,
    event_types: list[str] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Pre-compute pitch control and OBSO for event frames.

    Args:
        game_id: Metrica sample game ID (1, 2, or 3)
        event_types: Event types to process (default: Pass, Shot)
        output_dir: Output directory (default: data/precomputed/)

    Returns:
        Path to the saved .npz file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "precomputed"
    output_dir.mkdir(parents=True, exist_ok=True)

    if event_types is None:
        # All on-ball events (Spearman analyzes all moments of ball interaction)
        event_types = ['PASS', 'SHOT', 'RECOVERY', 'BALL LOST', 'CHALLENGE',
                       'SET PIECE', 'FAULT RECEIVED']

    # Load match
    print(f"Loading game {game_id}...")
    match = load_match(game_id=game_id, auto_download=True)

    # Get relevant events
    events = get_event_frames(match.events, event_types)
    n_events = len(events)
    print(f"Found {n_events} events of types: {event_types}")

    if n_events == 0:
        # Try lowercase
        event_types_lower = [t.lower().capitalize() for t in event_types]
        events = get_event_frames(match.events, event_types_lower)
        n_events = len(events)
        print(f"Retry with {event_types_lower}: found {n_events} events")

    # Initialize analyzer (caches EPV grid)
    analyzer = OBSOAnalyzer()
    grid = analyzer.grid
    n_grid_y, n_grid_x = grid.shape[:2]

    # Pre-allocate arrays
    pitch_control_home = np.zeros((n_events, n_grid_y, n_grid_x), dtype=np.float32)
    pitch_control_away = np.zeros((n_events, n_grid_y, n_grid_x), dtype=np.float32)
    obso_home = np.zeros((n_events, n_grid_y, n_grid_x), dtype=np.float32)
    obso_away = np.zeros((n_events, n_grid_y, n_grid_x), dtype=np.float32)

    # Store event metadata
    event_indices = events.index.to_numpy()
    frame_indices = events['Start Frame'].to_numpy().astype(np.int32)
    event_types_arr = events['Type'].to_numpy().astype(str)
    event_teams = events['Team'].to_numpy().astype(str)

    # Store timing info
    times = np.zeros(n_events, dtype=np.float32)
    periods = np.zeros(n_events, dtype=np.int8)

    # Store positions and velocities
    ball_positions = np.zeros((n_events, 2), dtype=np.float32)
    home_positions_list = []
    away_positions_list = []
    home_velocities_list = []
    away_velocities_list = []
    home_jerseys_list = []
    away_jerseys_list = []

    # Process events
    start_time = time.time()

    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(events.iterrows()), total=n_events, desc="Computing")
    except ImportError:
        iterator = enumerate(events.iterrows())
        print("(Install tqdm for progress bar)")

    for i, (idx, event) in iterator:
        frame = int(event['Start Frame'])

        # Get frame data
        try:
            frame_data = get_frame_data(
                match.tracking_home,
                match.tracking_away,
                frame
            )
        except KeyError:
            # Frame not in tracking data
            continue

        # Compute for both perspectives
        result_home = analyzer.analyze_frame(frame_data, attacking_team='home')
        result_away = analyzer.analyze_frame(frame_data, attacking_team='away')

        # Store results
        pitch_control_home[i] = result_home['pitch_control'].astype(np.float32)
        pitch_control_away[i] = result_away['pitch_control'].astype(np.float32)
        obso_home[i] = result_home['obso'].astype(np.float32)
        obso_away[i] = result_away['obso'].astype(np.float32)

        times[i] = frame_data['time']
        periods[i] = frame_data['period']
        ball_positions[i] = frame_data['ball']

        # Store player positions and velocities
        home_positions_list.append(frame_data['home']['positions'].astype(np.float32))
        away_positions_list.append(frame_data['away']['positions'].astype(np.float32))
        home_velocities_list.append(frame_data['home']['velocities'].astype(np.float32))
        away_velocities_list.append(frame_data['away']['velocities'].astype(np.float32))
        home_jerseys_list.append(frame_data['home']['jerseys'])
        away_jerseys_list.append(frame_data['away']['jerseys'])

    elapsed = time.time() - start_time
    eps = n_events / elapsed
    print(f"Computed {n_events} events in {elapsed:.1f}s ({eps:.1f} events/sec)")

    # Get consistent jersey lists
    home_jerseys = home_jerseys_list[0] if home_jerseys_list else []
    away_jerseys = away_jerseys_list[0] if away_jerseys_list else []

    # Convert positions to fixed-size arrays
    max_home = max(len(p) for p in home_positions_list) if home_positions_list else 11
    max_away = max(len(p) for p in away_positions_list) if away_positions_list else 11

    home_positions = np.full((n_events, max_home, 2), np.nan, dtype=np.float32)
    away_positions = np.full((n_events, max_away, 2), np.nan, dtype=np.float32)
    home_velocities = np.full((n_events, max_home, 2), np.nan, dtype=np.float32)
    away_velocities = np.full((n_events, max_away, 2), np.nan, dtype=np.float32)

    for i, (hp, ap, hv, av) in enumerate(zip(
        home_positions_list, away_positions_list,
        home_velocities_list, away_velocities_list
    )):
        home_positions[i, :len(hp)] = hp
        away_positions[i, :len(ap)] = ap
        home_velocities[i, :len(hv)] = hv
        away_velocities[i, :len(av)] = av

    # Also store event start/end positions from event data
    event_start_x = events['Start X'].to_numpy().astype(np.float32) if 'Start X' in events.columns else np.zeros(n_events)
    event_start_y = events['Start Y'].to_numpy().astype(np.float32) if 'Start Y' in events.columns else np.zeros(n_events)
    event_end_x = events['End X'].to_numpy().astype(np.float32) if 'End X' in events.columns else np.zeros(n_events)
    event_end_y = events['End Y'].to_numpy().astype(np.float32) if 'End Y' in events.columns else np.zeros(n_events)

    # Per-event OBSO totals using Spearman's model:
    #
    # OBSO = ∫ T(r) × PPCF(r) × S(r) dr
    #
    # Where T(r) is the transition probability (Equation 6):
    #   T(r | σ, α) = N(r, r_b, σ) · [Σ_k PPCF_k(r)]^α  (normalized to 1)
    #
    # This is a field integral, not a sum at player positions.
    # T(r) tells us where the ball is likely to go next.
    # We integrate the OBSO surface weighted by this transition probability.

    # Spearman Table 1 MAP parameters:
    SIGMA = 23.9  # Mean distance between on-ball events (meters)
    ALPHA = 1.04  # Preference for maintaining possession (PPCF weighting)

    obso_home_totals = np.zeros(n_events, dtype=np.float32)
    obso_away_totals = np.zeros(n_events, dtype=np.float32)

    for i, (idx, event) in enumerate(events.iterrows()):
        event_team = str(event['Team']).lower()
        ball_pos = ball_positions[i]

        # Skip events with invalid ball position
        if np.isnan(ball_pos).any():
            continue

        # Compute Gaussian weights for all grid points (distance from ball)
        dist_sq = (grid[..., 0] - ball_pos[0])**2 + (grid[..., 1] - ball_pos[1])**2
        gaussian = np.exp(-dist_sq / (2 * SIGMA**2))

        # Home OBSO: only count during home possession
        if 'home' in event_team:
            # Transition = Gaussian × PC^alpha, normalized to sum to 1
            transition = gaussian * (pitch_control_home[i] ** ALPHA)
            transition_sum = transition.sum()
            if transition_sum > 0:
                transition = transition / transition_sum

            # Field integral: OBSO = ∫ T(r) × PC(r) × S(r) dr
            # obso_home already contains PC × S (scoring probability)
            obso_home_totals[i] = (transition * obso_home[i]).sum()

        # Away OBSO: only count during away possession
        if 'away' in event_team:
            transition = gaussian * (pitch_control_away[i] ** ALPHA)
            transition_sum = transition.sum()
            if transition_sum > 0:
                transition = transition / transition_sum

            obso_away_totals[i] = (transition * obso_away[i]).sum()

    # Time-integrated OBSO surfaces (mean over all events for spatial pattern)
    obso_home_time_integrated = obso_home.mean(axis=0).astype(np.float32)
    obso_away_time_integrated = obso_away.mean(axis=0).astype(np.float32)

    # Extract goal events from full event data
    all_events = match.events
    goal_events = all_events[all_events['Type'].str.upper() == 'SHOT']
    # Check for subtype indicating goal (Metrica uses various conventions)
    if 'Subtype' in goal_events.columns:
        goal_events = goal_events[goal_events['Subtype'].str.upper().str.contains('GOAL', na=False)]

    goal_times_home = []
    goal_times_away = []
    for _, goal in goal_events.iterrows():
        goal_time = goal.get('Start Time [s]', 0)
        team = str(goal.get('Team', '')).lower()
        if 'home' in team:
            goal_times_home.append(float(goal_time))
        else:
            goal_times_away.append(float(goal_time))

    print(f"Found {len(goal_times_home)} home goals, {len(goal_times_away)} away goals")

    # Save to compressed file
    types_str = '_'.join(t.lower() for t in event_types)
    output_path = output_dir / f"game_{game_id}_events_{types_str}.npz"

    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        # Computed surfaces (both perspectives)
        pitch_control_home=pitch_control_home,
        pitch_control_away=pitch_control_away,
        obso_home=obso_home,
        obso_away=obso_away,
        # Grid
        grid=grid.astype(np.float32),
        epv_home=analyzer.epv_for_home.astype(np.float32),
        epv_away=analyzer.epv_for_away.astype(np.float32),
        # Event metadata
        event_indices=event_indices,
        frame_indices=frame_indices,
        event_types=event_types_arr,
        event_teams=event_teams,
        # Event positions
        event_start_x=event_start_x,
        event_start_y=event_start_y,
        event_end_x=event_end_x,
        event_end_y=event_end_y,
        # Timing
        times=times,
        periods=periods,
        # Ball and player positions/velocities
        ball_positions=ball_positions,
        home_positions=home_positions,
        away_positions=away_positions,
        home_velocities=home_velocities,
        away_velocities=away_velocities,
        # Jerseys
        home_jerseys=np.array(home_jerseys, dtype=str),
        away_jerseys=np.array(away_jerseys, dtype=str),
        # Match metadata
        home_gk=match.metadata['home_gk'],
        away_gk=match.metadata['away_gk'],
        game_id=game_id,
        # Per-event OBSO totals (for time series)
        obso_home_totals=obso_home_totals,
        obso_away_totals=obso_away_totals,
        # Time-integrated OBSO surfaces (for spatial heatmap)
        obso_home_time_integrated=obso_home_time_integrated,
        obso_away_time_integrated=obso_away_time_integrated,
        # Goal times
        goal_times_home=np.array(goal_times_home, dtype=np.float32),
        goal_times_away=np.array(goal_times_away, dtype=np.float32),
    )

    # Report file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {file_size_mb:.1f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Pre-compute pitch control for events")
    parser.add_argument("--game-id", type=int, default=1, choices=[1, 2, 3],
                        help="Metrica sample game ID")
    parser.add_argument("--event-types", nargs='+', default=None,
                        help="Event types to process (default: all on-ball events)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    output_path = precompute_events(
        game_id=args.game_id,
        event_types=args.event_types,
        output_dir=output_dir,
    )

    print(f"\nDone! Load in Dash app or with:")
    print(f"  data = np.load('{output_path}')")


if __name__ == "__main__":
    main()
