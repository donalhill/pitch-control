#!/usr/bin/env python3
"""
Benchmark single-frame computation times for interactive feasibility.

Tests whether pitch control and OBSO can be computed fast enough
for real-time player dragging interactions.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pitch_control.io.metrica import load_match, get_frame_data
from pitch_control.models.pitch_control import (
    compute_pitch_control,
    create_pitch_grid,
    default_model_params,
)
from pitch_control.models.obso import compute_obso
from pitch_control.models.epv import load_epv_grid, create_epv_for_grid


def benchmark():
    print("Loading game data...")
    match = load_match(game_id=1, auto_download=True)

    # Get a sample frame (first pass event)
    pass_events = match.events[match.events['Type'].str.upper() == 'PASS']
    frame = int(pass_events.iloc[0]['Start Frame'])

    frame_data = get_frame_data(match.tracking_home, match.tracking_away, frame)
    print(f"Using frame {frame}\n")

    # Setup
    params = default_model_params()
    grid = create_pitch_grid(params)
    epv_grid, epv_x, epv_y = load_epv_grid()
    epv = create_epv_for_grid(grid, epv_grid, epv_x, epv_y, attacking_direction=1)

    att_data = frame_data["home"]
    def_data = frame_data["away"]
    ball_pos = frame_data["ball"]

    # Spearman transition parameters
    SIGMA = 23.9
    ALPHA = 1.04

    # Warm-up run (JIT compilation)
    print("Warm-up run (JIT compilation)...")
    _ = compute_pitch_control(
        att_data["positions"], att_data["velocities"],
        def_data["positions"], def_data["velocities"],
        ball_pos, params, grid=grid
    )

    # Benchmark pitch control (the heavy computation)
    n_runs = 20
    print(f"\nBenchmarking {n_runs} runs...\n")

    pc_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        pitch_control, _, _ = compute_pitch_control(
            att_data["positions"], att_data["velocities"],
            def_data["positions"], def_data["velocities"],
            ball_pos, params, grid=grid
        )
        pc_times.append(time.perf_counter() - t0)

    # Benchmark transition probability
    trans_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        dist_sq = (grid[..., 0] - ball_pos[0])**2 + (grid[..., 1] - ball_pos[1])**2
        gaussian = np.exp(-dist_sq / (2 * SIGMA**2))
        transition = gaussian * (pitch_control ** ALPHA)
        transition = transition / transition.sum()
        trans_times.append(time.perf_counter() - t0)

    # Benchmark OBSO (just multiplication)
    obso_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        obso = compute_obso(pitch_control, epv)
        obso_times.append(time.perf_counter() - t0)

    # Results
    print("=" * 50)
    print("SINGLE FRAME COMPUTATION TIMES")
    print("=" * 50)
    print(f"Grid size: {grid.shape[0]} x {grid.shape[1]} = {grid.shape[0] * grid.shape[1]:,} cells")
    print()
    print(f"Pitch Control C(r):     {np.mean(pc_times)*1000:6.1f} ms  (std: {np.std(pc_times)*1000:.1f} ms)")
    print(f"Transition T(r):        {np.mean(trans_times)*1000:6.2f} ms  (std: {np.std(trans_times)*1000:.2f} ms)")
    print(f"OBSO (C × S):           {np.mean(obso_times)*1000:6.2f} ms  (std: {np.std(obso_times)*1000:.2f} ms)")
    print("-" * 50)
    total_ms = (np.mean(pc_times) + np.mean(trans_times) + np.mean(obso_times)) * 1000
    print(f"TOTAL:                  {total_ms:6.1f} ms")
    print()
    print(f"Max interactive FPS:    {1000/total_ms:.1f} fps")
    print()

    if total_ms < 100:
        print("✓ Fast enough for smooth interaction (<100ms)")
    elif total_ms < 300:
        print("~ Usable with slight lag (100-300ms)")
    else:
        print("✗ Too slow for interactive use (>300ms)")


if __name__ == "__main__":
    benchmark()
