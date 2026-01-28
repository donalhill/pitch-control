#!/usr/bin/env python3
"""
Test script to verify the pitch control setup.

Run this to:
1. Download sample Metrica data
2. Generate default EPV grid
3. Compute pitch control for a single frame
4. Generate a test visualization
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    print("=" * 60)
    print("Pitch Control Setup Test")
    print("=" * 60)

    # 1. Test data loading
    print("\n1. Testing Metrica data loading...")
    from pitch_control.io.metrica import load_match, get_frame_data

    match = load_match(game_id=1, auto_download=True)
    print(f"   ✓ Loaded {len(match.tracking_home)} frames")
    print(f"   ✓ Home GK: #{match.metadata['home_gk']}")
    print(f"   ✓ Away GK: #{match.metadata['away_gk']}")

    # 2. Test EPV grid
    print("\n2. Testing EPV grid...")
    from pitch_control.models.epv import load_epv_grid

    epv_grid, x, y = load_epv_grid()
    print(f"   ✓ EPV grid shape: {epv_grid.shape}")
    print(f"   ✓ EPV range: [{epv_grid.min():.3f}, {epv_grid.max():.3f}]")

    # 3. Test pitch control computation
    print("\n3. Testing pitch control computation...")
    from pitch_control.models.pitch_control import compute_pitch_control, default_model_params

    # Get a frame from the middle of the match
    frames = match.tracking_home.index.tolist()
    test_frame = frames[len(frames) // 2]

    frame_data = get_frame_data(
        match.tracking_home,
        match.tracking_away,
        test_frame
    )

    import time
    start = time.time()

    pc, grid, player_pc = compute_pitch_control(
        frame_data['home']['positions'],
        frame_data['home']['velocities'],
        frame_data['away']['positions'],
        frame_data['away']['velocities'],
        frame_data['ball'],
        default_model_params(),
    )

    elapsed = time.time() - start
    print(f"   ✓ Pitch control computed in {elapsed:.3f}s")
    print(f"   ✓ Grid shape: {grid.shape}")
    print(f"   ✓ Mean home control: {pc.mean()*100:.1f}%")

    # 4. Test OBSO
    print("\n4. Testing OBSO computation...")
    from pitch_control.models.obso import OBSOAnalyzer

    analyzer = OBSOAnalyzer()
    result = analyzer.analyze_frame(frame_data, attacking_team='home')

    print(f"   ✓ Total OBSO: {result['total_obso']:.4f}")
    print(f"   ✓ Top player OBSO contributions:")
    sorted_players = sorted(result['player_obso'].items(), key=lambda x: x[1], reverse=True)[:3]
    for player, obso in sorted_players:
        print(f"      - {player}: {obso:.4f}")

    # 5. Test visualization
    print("\n5. Testing visualization...")
    from pitch_control.viz.pitch import plot_pitch_control, fig_to_base64

    fig, ax = plot_pitch_control(
        result['pitch_control'],
        result['grid'],
        home_positions=frame_data['home']['positions'],
        away_positions=frame_data['away']['positions'],
        ball_position=frame_data['ball'],
        title=f"Test - Frame {test_frame}",
    )

    # Save test image
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_pitch_control.png"

    import matplotlib.pyplot as plt
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved test image to {output_path}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nTo run the Dash app:")
    print("  cd /Users/donalhill/Documents/Code/pitch_control")
    print("  pip install -e .")
    print("  python app/main.py")
    print("")


if __name__ == "__main__":
    main()
