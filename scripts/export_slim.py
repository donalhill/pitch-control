#!/usr/bin/env python3
"""
Export slim precomputed data for web deployment.

Only includes:
- Time series data (all events' OBSO totals)
- Integrated OBSO heatmaps
- Top 10 OBSO events per team with pre-rendered images
"""

import argparse
import io
import sys
from pathlib import Path

# Set matplotlib backend before importing
import matplotlib
matplotlib.use('Agg')

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pitch_control.viz.pitch import plot_obso_decomposition, fig_to_base64

def export_slim(game_id: int, n_top: int = 10) -> Path:
    """
    Export slim version of precomputed data.

    Args:
        game_id: Game ID (1 or 2)
        n_top: Number of top OBSO events per team

    Returns:
        Path to exported file
    """
    # Load full precomputed data
    data_dir = Path(__file__).parent.parent / "data" / "precomputed"
    input_path = data_dir / f"game_{game_id}_events_pass_shot.npz"

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    print(f"Loading {input_path}...")
    data = np.load(input_path)

    # === TIME SERIES DATA (all events) ===
    times = data['times']
    obso_home_totals = data['obso_home_totals']
    obso_away_totals = data['obso_away_totals']
    goal_times_home = data['goal_times_home']
    goal_times_away = data['goal_times_away']
    periods = data['periods']

    # === INTEGRATED HEATMAPS ===
    obso_home_time_integrated = data['obso_home_time_integrated']
    obso_away_time_integrated = data['obso_away_time_integrated']
    grid = data['grid']

    # === SHOT/GOAL POSITIONS (for integrated map markers) ===
    event_types = data['event_types']
    event_teams = data['event_teams']
    event_start_x = data['event_start_x']
    event_start_y = data['event_start_y']
    ball_positions = data['ball_positions']  # Already in meters

    # Find shots - use ball_positions which are in meter coordinates
    shot_mask = np.char.upper(event_types.astype(str)) == 'SHOT'
    shot_ball_pos = ball_positions[shot_mask]
    shot_positions_x = shot_ball_pos[:, 0]
    shot_positions_y = shot_ball_pos[:, 1]
    shot_teams = event_teams[shot_mask]
    shot_times = times[shot_mask]

    # === TOP N EVENTS PER TEAM ===
    # Find top home events by OBSO
    home_mask = np.char.lower(event_teams.astype(str)) == 'home'
    home_indices = np.where(home_mask)[0]
    home_obso_values = obso_home_totals[home_mask]
    top_home_local = np.argsort(home_obso_values)[-n_top:][::-1]
    top_home_indices = home_indices[top_home_local]

    # Find top away events by OBSO
    away_mask = np.char.lower(event_teams.astype(str)) == 'away'
    away_indices = np.where(away_mask)[0]
    away_obso_values = obso_away_totals[away_mask]
    top_away_local = np.argsort(away_obso_values)[-n_top:][::-1]
    top_away_indices = away_indices[top_away_local]

    # Combine top indices
    top_indices = np.concatenate([top_home_indices, top_away_indices])
    n_top_events = len(top_indices)

    print(f"Top {n_top} home events: indices {top_home_indices}")
    print(f"Top {n_top} away events: indices {top_away_indices}")

    # Extract data for top events only
    top_event_indices = data['event_indices'][top_indices]
    top_frame_indices = data['frame_indices'][top_indices]
    top_event_types = event_types[top_indices]
    top_event_teams = event_teams[top_indices]
    top_event_start_x = event_start_x[top_indices]
    top_event_start_y = event_start_y[top_indices]
    top_times = times[top_indices]
    top_periods = periods[top_indices]
    top_ball_positions = data['ball_positions'][top_indices]

    # Player positions and velocities for top events
    top_home_positions = data['home_positions'][top_indices]
    top_away_positions = data['away_positions'][top_indices]
    top_home_velocities = data['home_velocities'][top_indices]
    top_away_velocities = data['away_velocities'][top_indices]

    # Pitch control and OBSO grids for top events
    # Store only the attacking team's perspective for each event
    top_pitch_control = np.zeros((n_top_events, grid.shape[0], grid.shape[1]), dtype=np.float32)
    top_obso = np.zeros((n_top_events, grid.shape[0], grid.shape[1]), dtype=np.float32)
    top_obso_totals = np.zeros(n_top_events, dtype=np.float32)

    for i, idx in enumerate(top_indices):
        team = str(event_teams[idx]).lower()
        if 'home' in team:
            top_pitch_control[i] = data['pitch_control_home'][idx]
            top_obso[i] = data['obso_home'][idx]
            top_obso_totals[i] = obso_home_totals[idx]
        else:
            top_pitch_control[i] = data['pitch_control_away'][idx]
            top_obso[i] = data['obso_away'][idx]
            top_obso_totals[i] = obso_away_totals[idx]

    # EPV grids (shared across all events)
    epv_home = data['epv_home']
    epv_away = data['epv_away']

    # Jerseys and GK info
    home_jerseys = data['home_jerseys']
    away_jerseys = data['away_jerseys']
    home_gk = data['home_gk']
    away_gk = data['away_gk']

    # === PRE-RENDER IMAGES FOR TOP EVENTS ===
    print("Pre-rendering decomposition images...")

    # Compute max velocity for consistent arrow scaling
    all_home_vel = top_home_velocities
    all_away_vel = top_away_velocities
    home_speeds = np.sqrt(all_home_vel[..., 0]**2 + all_home_vel[..., 1]**2)
    away_speeds = np.sqrt(all_away_vel[..., 0]**2 + all_away_vel[..., 1]**2)
    max_velocity = max(np.nanmax(home_speeds), np.nanmax(away_speeds), 1.0)

    # Spearman parameters for transition computation
    SIGMA = 23.9
    ALPHA = 1.04

    # Store images as base64 strings
    top_event_images = []

    for i in range(n_top_events):
        # Get data for this event
        pitch_control = top_pitch_control[i]
        obso = top_obso[i]
        obso_total = top_obso_totals[i]
        ball_pos = top_ball_positions[i]

        # Determine attacking team
        event_team = str(top_event_teams[i])
        attacking_team = "home" if "home" in event_team.lower() else "away"
        scoring = epv_home if attacking_team == 'home' else epv_away

        # Compute transition probability
        if not np.isnan(ball_pos).any():
            dist_sq = (grid[..., 0] - ball_pos[0])**2 + (grid[..., 1] - ball_pos[1])**2
            gaussian = np.exp(-dist_sq / (2 * SIGMA**2))
            transition = gaussian * (pitch_control ** ALPHA)
            transition_sum = transition.sum()
            if transition_sum > 0:
                transition = transition / transition_sum
        else:
            transition = np.zeros_like(pitch_control)

        # Get player positions and velocities
        home_pos = top_home_positions[i]
        away_pos = top_away_positions[i]
        valid_home = ~np.isnan(home_pos[:, 0])
        valid_away = ~np.isnan(away_pos[:, 0])
        home_pos = home_pos[valid_home]
        away_pos = away_pos[valid_away]

        home_vel = top_home_velocities[i][valid_home]
        away_vel = top_away_velocities[i][valid_away]

        home_jerseys_list = list(home_jerseys[:len(home_pos)])
        away_jerseys_list = list(away_jerseys[:len(away_pos)])

        # Format event info
        event_type = str(top_event_types[i])
        time_val = top_times[i]
        period = int(top_periods[i])

        # Format time
        mins = int(time_val // 60) + 1
        if mins % 10 == 1 and mins != 11:
            suffix = "st"
        elif mins % 10 == 2 and mins != 12:
            suffix = "nd"
        elif mins % 10 == 3 and mins != 13:
            suffix = "rd"
        else:
            suffix = "th"
        time_str = f"{mins}{suffix} minute"
        period_str = "First Half" if period == 1 else "Second Half"

        event_info = f"{event_type} by {event_team} | {period_str}, {time_str}"

        # Generate figure
        fig = plot_obso_decomposition(
            scoring=scoring,
            pitch_control=pitch_control,
            transition=transition,
            obso=obso,
            grid=grid,
            home_positions=home_pos,
            away_positions=away_pos,
            ball_position=ball_pos,
            home_velocities=home_vel,
            away_velocities=away_vel,
            home_jerseys=home_jerseys_list,
            away_jerseys=away_jerseys_list,
            obso_total=float(obso_total),
            event_info=event_info,
            attacking_team=attacking_team,
            max_velocity=max_velocity,
            figsize=(14, 10),
        )

        # Convert to base64
        img_base64 = fig_to_base64(fig, dpi=120)
        top_event_images.append(img_base64)

        print(f"  Rendered event {i+1}/{n_top_events}")

    # Convert to numpy array of strings
    top_event_images = np.array(top_event_images, dtype=object)

    # === SAVE SLIM FILE ===
    output_path = data_dir / f"game_{game_id}_slim.npz"

    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        # Time series (all events)
        times=times.astype(np.float32),
        periods=periods.astype(np.int8),
        obso_home_totals=obso_home_totals.astype(np.float32),
        obso_away_totals=obso_away_totals.astype(np.float32),
        goal_times_home=goal_times_home.astype(np.float32),
        goal_times_away=goal_times_away.astype(np.float32),

        # Integrated heatmaps
        obso_home_time_integrated=obso_home_time_integrated.astype(np.float32),
        obso_away_time_integrated=obso_away_time_integrated.astype(np.float32),
        grid=grid.astype(np.float32),

        # Shot positions (for integrated map markers)
        shot_positions_x=shot_positions_x.astype(np.float32),
        shot_positions_y=shot_positions_y.astype(np.float32),
        shot_teams=shot_teams,
        shot_times=shot_times.astype(np.float32),

        # EPV grids (shared)
        epv_home=epv_home.astype(np.float32),
        epv_away=epv_away.astype(np.float32),

        # Top events metadata
        top_indices=top_indices.astype(np.int32),
        top_event_indices=top_event_indices,
        top_frame_indices=top_frame_indices.astype(np.int32),
        top_event_types=top_event_types,
        top_event_teams=top_event_teams,
        top_event_start_x=top_event_start_x.astype(np.float32),
        top_event_start_y=top_event_start_y.astype(np.float32),
        top_times=top_times.astype(np.float32),
        top_periods=top_periods.astype(np.int8),
        top_ball_positions=top_ball_positions.astype(np.float32),
        top_obso_totals=top_obso_totals.astype(np.float32),

        # Top events player data
        top_home_positions=top_home_positions.astype(np.float32),
        top_away_positions=top_away_positions.astype(np.float32),
        top_home_velocities=top_home_velocities.astype(np.float32),
        top_away_velocities=top_away_velocities.astype(np.float32),

        # Top events grids (attacking team only)
        top_pitch_control=top_pitch_control.astype(np.float32),
        top_obso=top_obso.astype(np.float32),

        # Jerseys and metadata
        home_jerseys=home_jerseys,
        away_jerseys=away_jerseys,
        home_gk=home_gk,
        away_gk=away_gk,
        game_id=np.int32(game_id),
        n_top=np.int32(n_top),

        # Pre-rendered images (base64 PNG strings)
        top_event_images=top_event_images,
    )

    # Report sizes
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    input_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"Original: {input_size_mb:.1f} MB")
    print(f"Slim: {file_size_mb:.1f} MB")
    print(f"Reduction: {(1 - file_size_mb/input_size_mb)*100:.0f}%")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export slim precomputed data")
    parser.add_argument("--game-id", type=int, default=1, choices=[1, 2])
    parser.add_argument("--n-top", type=int, default=10, help="Top N events per team")
    args = parser.parse_args()

    export_slim(args.game_id, args.n_top)


if __name__ == "__main__":
    main()
