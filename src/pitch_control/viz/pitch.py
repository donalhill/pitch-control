"""
Pitch visualization using mplsoccer.

Provides matplotlib-based pitch plots with pitch control overlays.
Returns base64-encoded images for embedding in Dash.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Try to import mplsoccer, provide fallback
try:
    from mplsoccer import Pitch, VerticalPitch
    HAS_MPLSOCCER = True
except ImportError:
    HAS_MPLSOCCER = False
    Pitch = None
    VerticalPitch = None


# Color schemes
COLORS = {
    'bg_primary': '#F8FAFC',
    'bg_card': '#FFFFFF',
    'pitch': '#FFFFFF',  # White background
    'pitch_lines': '#000000',  # Black lines
    'home': '#E63946',  # Red
    'away': '#457B9D',  # Blue
    'ball': '#000000',  # Black
    'text': '#1E293B',
}

# Custom colormaps for pitch control
def create_pitch_control_cmap():
    """Create red-white-blue colormap for pitch control."""
    colors = [COLORS['away'], '#FFFFFF', COLORS['home']]
    return LinearSegmentedColormap.from_list('pitch_control', colors, N=256)


def create_obso_cmap():
    """Create colormap for OBSO (danger/heat)."""
    colors = ['#000033', '#1a1a66', '#333399', '#4d4dcc', '#6666ff',
              '#ffff00', '#ffcc00', '#ff9900', '#ff6600', '#ff0000']
    return LinearSegmentedColormap.from_list('obso', colors, N=256)


PITCH_CONTROL_CMAP = create_pitch_control_cmap()
OBSO_CMAP = create_obso_cmap()


def fig_to_base64(fig: plt.Figure, dpi: int = 100, bg_color: str = None) -> str:
    """Convert matplotlib figure to base64 string for Dash embedding."""
    buffer = BytesIO()

    facecolor = bg_color or COLORS['bg_card']
    fig.savefig(
        buffer,
        format='png',
        dpi=dpi,
        bbox_inches='tight',
        facecolor=facecolor,
        edgecolor='none',
        pad_inches=0.1,
    )
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return f'data:image/png;base64,{img_base64}'


def plot_pitch(
    ax: Optional[plt.Axes] = None,
    pitch_color: str = None,
    line_color: str = None,
    figsize: tuple = (10.5, 6.8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Draw a football pitch.

    Uses mplsoccer if available, otherwise falls back to manual drawing.
    """
    pitch_color = pitch_color or COLORS['pitch']
    line_color = line_color or COLORS['pitch_lines']

    if HAS_MPLSOCCER:
        pitch = Pitch(
            pitch_type='custom',
            pitch_length=105,
            pitch_width=68,
            pitch_color=pitch_color,
            line_color=line_color,
            linewidth=1.5,
            goal_type='box',
        )
        if ax is None:
            fig, ax = pitch.draw(figsize=figsize)
        else:
            fig = ax.get_figure()
            pitch.draw(ax=ax)
        return fig, ax

    # Fallback: manual pitch drawing
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.set_facecolor(pitch_color)

    # Field dimensions (centered at origin)
    length, width = 105, 68

    # Outer boundary
    ax.plot([-length/2, length/2], [-width/2, -width/2], color=line_color, lw=1.5)
    ax.plot([-length/2, length/2], [width/2, width/2], color=line_color, lw=1.5)
    ax.plot([-length/2, -length/2], [-width/2, width/2], color=line_color, lw=1.5)
    ax.plot([length/2, length/2], [-width/2, width/2], color=line_color, lw=1.5)

    # Center line and circle
    ax.plot([0, 0], [-width/2, width/2], color=line_color, lw=1.5)
    circle = plt.Circle((0, 0), 9.15, fill=False, color=line_color, lw=1.5)
    ax.add_patch(circle)

    # Penalty areas (16.5m from goal, 40.3m wide)
    for sign in [-1, 1]:
        # Penalty box
        ax.plot([sign*length/2, sign*(length/2 - 16.5)], [-20.15, -20.15], color=line_color, lw=1.5)
        ax.plot([sign*length/2, sign*(length/2 - 16.5)], [20.15, 20.15], color=line_color, lw=1.5)
        ax.plot([sign*(length/2 - 16.5), sign*(length/2 - 16.5)], [-20.15, 20.15], color=line_color, lw=1.5)

        # Goal box (5.5m from goal, 18.3m wide)
        ax.plot([sign*length/2, sign*(length/2 - 5.5)], [-9.15, -9.15], color=line_color, lw=1.5)
        ax.plot([sign*length/2, sign*(length/2 - 5.5)], [9.15, 9.15], color=line_color, lw=1.5)
        ax.plot([sign*(length/2 - 5.5), sign*(length/2 - 5.5)], [-9.15, 9.15], color=line_color, lw=1.5)

        # Penalty spot
        ax.scatter([sign*(length/2 - 11)], [0], color=line_color, s=20, zorder=5)

    ax.set_xlim(-length/2 - 2, length/2 + 2)
    ax.set_ylim(-width/2 - 2, width/2 + 2)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax


def _transform_coords(positions: np.ndarray) -> np.ndarray:
    """Transform from centered coords (-52.5 to 52.5) to mplsoccer coords (0 to 105)."""
    if positions is None or len(positions) == 0:
        return positions
    transformed = positions.copy()
    if transformed.ndim == 1:
        transformed[0] += 52.5
        transformed[1] += 34.0
    else:
        transformed[:, 0] += 52.5
        transformed[:, 1] += 34.0
    return transformed


def plot_frame(
    home_positions: np.ndarray,
    away_positions: np.ndarray,
    ball_position: np.ndarray,
    home_velocities: Optional[np.ndarray] = None,
    away_velocities: Optional[np.ndarray] = None,
    home_jerseys: Optional[list] = None,
    away_jerseys: Optional[list] = None,
    ax: Optional[plt.Axes] = None,
    show_velocities: bool = True,
    figsize: tuple = (10.5, 6.8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot player positions for a single frame.
    """
    fig, ax = plot_pitch(ax=ax, figsize=figsize)

    # Transform coordinates from centered to mplsoccer system
    home_positions = _transform_coords(home_positions)
    away_positions = _transform_coords(away_positions)
    ball_position = _transform_coords(ball_position)

    # Plot home team
    if len(home_positions) > 0:
        ax.scatter(
            home_positions[:, 0],
            home_positions[:, 1],
            c=COLORS['home'],
            s=200,
            edgecolors='white',
            linewidths=2,
            zorder=10,
            label='Home',
        )

        # Velocity arrows
        if show_velocities and home_velocities is not None:
            for pos, vel in zip(home_positions, home_velocities):
                if not np.isnan(vel).any():
                    ax.arrow(
                        pos[0], pos[1],
                        vel[0] * 0.5, vel[1] * 0.5,
                        head_width=1, head_length=0.5,
                        fc=COLORS['home'], ec=COLORS['home'],
                        alpha=0.7, zorder=9,
                    )

        # Jersey numbers
        if home_jerseys:
            for pos, jersey in zip(home_positions, home_jerseys):
                ax.annotate(
                    jersey,
                    (pos[0], pos[1]),
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white', zorder=11,
                )

    # Plot away team
    if len(away_positions) > 0:
        ax.scatter(
            away_positions[:, 0],
            away_positions[:, 1],
            c=COLORS['away'],
            s=200,
            edgecolors='white',
            linewidths=2,
            zorder=10,
            label='Away',
        )

        if show_velocities and away_velocities is not None:
            for pos, vel in zip(away_positions, away_velocities):
                if not np.isnan(vel).any():
                    ax.arrow(
                        pos[0], pos[1],
                        vel[0] * 0.5, vel[1] * 0.5,
                        head_width=1, head_length=0.5,
                        fc=COLORS['away'], ec=COLORS['away'],
                        alpha=0.7, zorder=9,
                    )

        if away_jerseys:
            for pos, jersey in zip(away_positions, away_jerseys):
                ax.annotate(
                    jersey,
                    (pos[0], pos[1]),
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white', zorder=11,
                )

    # Plot ball
    if not np.isnan(ball_position).any():
        ax.scatter(
            [ball_position[0]],
            [ball_position[1]],
            c=COLORS['ball'],
            s=100,
            edgecolors='black',
            linewidths=1,
            zorder=12,
            marker='o',
        )

    return fig, ax


def plot_pitch_control(
    pitch_control: np.ndarray,
    grid: np.ndarray,
    home_positions: Optional[np.ndarray] = None,
    away_positions: Optional[np.ndarray] = None,
    ball_position: Optional[np.ndarray] = None,
    home_jerseys: Optional[list] = None,
    away_jerseys: Optional[list] = None,
    alpha: float = 0.7,
    cmap=None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    show_colorbar: bool = True,
    title: str = None,
    figsize: tuple = (12, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot pitch control surface with player overlay.

    Args:
        pitch_control: (n_grid_y, n_grid_x) control probability for attacking team
        grid: (n_grid_y, n_grid_x, 2) grid coordinates
        home_positions: Home player positions
        away_positions: Away player positions
        ball_position: Ball position
        alpha: Transparency of heatmap
        cmap: Colormap (defaults to red-white-blue)
        vmin, vmax: Colormap range
        show_colorbar: Whether to show colorbar
        title: Plot title
    """
    if cmap is None:
        cmap = PITCH_CONTROL_CMAP

    fig, ax = plot_pitch(figsize=figsize)

    # Transform extent from centered coords (-52.5 to 52.5) to mplsoccer coords (0 to 105)
    # mplsoccer 'custom' pitch uses 0→pitch_length, 0→pitch_width
    extent = [0, 105, 0, 68]

    # Plot heatmap
    im = ax.imshow(
        pitch_control,
        extent=extent,
        origin='lower',
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
        zorder=2,
    )

    # Colorbar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Pitch Control (Home)', fontsize=10)

    # Transform coordinates from centered to mplsoccer system
    home_positions = _transform_coords(home_positions) if home_positions is not None else None
    away_positions = _transform_coords(away_positions) if away_positions is not None else None
    ball_position = _transform_coords(ball_position) if ball_position is not None else None

    # Overlay players
    if home_positions is not None and len(home_positions) > 0:
        ax.scatter(
            home_positions[:, 0],
            home_positions[:, 1],
            c=COLORS['home'],
            s=150,
            edgecolors='white',
            linewidths=2,
            zorder=10,
        )
        if home_jerseys:
            for pos, jersey in zip(home_positions, home_jerseys):
                ax.annotate(
                    jersey, (pos[0], pos[1]),
                    ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    color='white', zorder=11,
                )

    if away_positions is not None and len(away_positions) > 0:
        ax.scatter(
            away_positions[:, 0],
            away_positions[:, 1],
            c=COLORS['away'],
            s=150,
            edgecolors='white',
            linewidths=2,
            zorder=10,
        )
        if away_jerseys:
            for pos, jersey in zip(away_positions, away_jerseys):
                ax.annotate(
                    jersey, (pos[0], pos[1]),
                    ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    color='white', zorder=11,
                )

    if ball_position is not None and not np.isnan(ball_position).any():
        ax.scatter(
            [ball_position[0]], [ball_position[1]],
            c=COLORS['ball'], s=80,
            edgecolors='black', linewidths=1,
            zorder=12,
        )

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    fig.tight_layout()
    return fig, ax


def plot_obso(
    obso: np.ndarray,
    grid: np.ndarray,
    home_positions: Optional[np.ndarray] = None,
    away_positions: Optional[np.ndarray] = None,
    ball_position: Optional[np.ndarray] = None,
    home_jerseys: Optional[list] = None,
    away_jerseys: Optional[list] = None,
    alpha: float = 0.8,
    show_colorbar: bool = True,
    title: str = None,
    figsize: tuple = (12, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot OBSO (Off-Ball Scoring Opportunity) surface.

    Uses a heat-style colormap to show dangerous areas.
    """
    return plot_pitch_control(
        obso,
        grid,
        home_positions=home_positions,
        away_positions=away_positions,
        ball_position=ball_position,
        home_jerseys=home_jerseys,
        away_jerseys=away_jerseys,
        alpha=alpha,
        cmap=OBSO_CMAP,
        vmin=0,
        vmax=obso.max() if obso.max() > 0 else 0.1,
        show_colorbar=show_colorbar,
        title=title or "Off-Ball Scoring Opportunity",
        figsize=figsize,
    )


def plot_player_obso_contribution(
    player_obso: dict,
    attacking_team: str = "home",
    top_n: int = 10,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Bar chart of player OBSO contributions.
    """
    # Filter to attacking team and sort
    team_prefix = "home_" if attacking_team == "home" else "away_"
    team_players = {k: v for k, v in player_obso.items() if k.startswith(team_prefix)}

    sorted_players = sorted(team_players.items(), key=lambda x: x[1], reverse=True)[:top_n]

    if not sorted_players:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return fig

    names = [p[0].replace(team_prefix, "#") for p in sorted_players]
    values = [p[1] for p in sorted_players]

    fig, ax = plt.subplots(figsize=figsize)

    color = COLORS['home'] if attacking_team == "home" else COLORS['away']
    bars = ax.barh(names[::-1], values[::-1], color=color, edgecolor='white')

    ax.set_xlabel("OBSO Contribution", fontsize=11)
    ax.set_title(f"Player Dangerous Space Creation ({attacking_team.title()})", fontsize=12, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


def plot_cumulative_obso_timeline(
    times: np.ndarray,
    obso_home: np.ndarray,
    obso_away: np.ndarray,
    goal_times_home: list[float] = None,
    goal_times_away: list[float] = None,
    figsize: tuple = (12, 7),
) -> plt.Figure:
    """
    Plot cumulative OBSO over time for both teams with momentum subplot.

    Like Spearman's "Integrated Scoring Opportunity" plot with residuals.

    Args:
        times: Event times in seconds
        obso_home: OBSO values for home team at each event
        obso_away: OBSO values for away team at each event
        goal_times_home: Times of home team goals (for markers)
        goal_times_away: Times of away team goals (for markers)
    """
    # Create figure with two subplots - cumulative on top, momentum below
    fig, (ax_cum, ax_mom) = plt.subplots(
        2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True
    )

    # Convert times to minutes
    times_min = times / 60.0

    # Compute cumulative OBSO
    cum_home = np.cumsum(obso_home)
    cum_away = np.cumsum(obso_away)

    # === TOP SUBPLOT: Cumulative OBSO ===
    ax_cum.plot(times_min, cum_home, color=COLORS['home'], linewidth=2, label='Home')
    ax_cum.plot(times_min, cum_away, color=COLORS['away'], linewidth=2, label='Away')

    # Add goal markers (open circles)
    if goal_times_home:
        for gt in goal_times_home:
            gt_min = gt / 60.0
            idx = np.searchsorted(times_min, gt_min)
            if idx > 0 and idx <= len(cum_home):
                y_val = cum_home[min(idx, len(cum_home)-1)]
                ax_cum.scatter([gt_min], [y_val], s=100, facecolors='none',
                              edgecolors=COLORS['home'], linewidths=2, zorder=5,
                              label='Home Goal' if gt == goal_times_home[0] else None)

    if goal_times_away:
        for gt in goal_times_away:
            gt_min = gt / 60.0
            idx = np.searchsorted(times_min, gt_min)
            if idx > 0 and idx <= len(cum_away):
                y_val = cum_away[min(idx, len(cum_away)-1)]
                ax_cum.scatter([gt_min], [y_val], s=100, facecolors='none',
                              edgecolors=COLORS['away'], linewidths=2, zorder=5,
                              label='Away Goal' if gt == goal_times_away[0] else None)

    ax_cum.set_ylabel('Cumulative OBSO', fontsize=10)
    ax_cum.legend(loc='upper left', frameon=True, fontsize=9)
    ax_cum.spines['top'].set_visible(False)
    ax_cum.spines['right'].set_visible(False)
    ax_cum.set_ylim(0, None)

    # === BOTTOM SUBPLOT: Momentum (binned OBSO) ===
    # Bin OBSO values into 1-minute intervals
    bin_width = 1.0  # minutes
    max_time = max(times_min) if len(times_min) > 0 else 100
    bins = np.arange(0, max_time + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2

    # Sum OBSO within each bin
    home_binned, _ = np.histogram(times_min, bins=bins, weights=obso_home)
    away_binned, _ = np.histogram(times_min, bins=bins, weights=obso_away)

    # Plot home bars (positive) and away bars (negative)
    ax_mom.bar(bin_centers, home_binned, width=bin_width * 0.9, color=COLORS['home'],
               alpha=0.8, align='center')
    ax_mom.bar(bin_centers, -away_binned, width=bin_width * 0.9, color=COLORS['away'],
               alpha=0.8, align='center')

    # Zero line
    ax_mom.axhline(y=0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)

    ax_mom.set_xlabel('Time (minutes)', fontsize=10)
    ax_mom.set_ylabel('Away(-) / Home(+)', fontsize=9)
    ax_mom.spines['top'].set_visible(False)
    ax_mom.spines['right'].set_visible(False)

    # Set symmetric y limits for momentum plot
    max_binned = max(home_binned.max(), away_binned.max()) if len(home_binned) > 0 else 0.1
    ax_mom.set_ylim(-max_binned * 1.1, max_binned * 1.1)

    # X-axis settings (shared)
    ax_mom.set_xlim(0, max_time)
    tick_interval = 5
    ax_mom.set_xticks(np.arange(0, max_time + 1, tick_interval))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)  # Add vertical space between subplots
    return fig


def plot_time_integrated_obso(
    obso_home_sum: np.ndarray,
    obso_away_sum: np.ndarray,
    grid: np.ndarray,
    home_goals: int = 0,
    away_goals: int = 0,
    home_total_obso: float = 0,
    away_total_obso: float = 0,
    home_shot_positions: Optional[np.ndarray] = None,
    away_shot_positions: Optional[np.ndarray] = None,
    home_goal_positions: Optional[np.ndarray] = None,
    away_goal_positions: Optional[np.ndarray] = None,
    figsize: tuple = (12, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot time-integrated OBSO heatmap showing spatial distribution.

    Shows where each team created dangerous space over the entire match.
    Both teams shown across full pitch - home (red) vs away (blue).
    Shots marked with x, goals marked with o.

    Args:
        obso_home_sum: Sum of home OBSO surfaces over all events
        obso_away_sum: Sum of away OBSO surfaces over all events
        grid: Grid coordinates
        home_goals: Home team goals scored
        away_goals: Away team goals scored
        home_total_obso: Total integrated OBSO for home
        away_total_obso: Total integrated OBSO for away
        home_shot_positions: (n, 2) array of home shot positions (centered coords)
        away_shot_positions: (n, 2) array of away shot positions (centered coords)
        home_goal_positions: (n, 2) array of home goal positions (centered coords)
        away_goal_positions: (n, 2) array of away goal positions (centered coords)
    """
    # Dark pitch background to match decomposition plots
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#000004')
    ax.set_facecolor('#000004')

    # Draw pitch with white lines on dark background
    if HAS_MPLSOCCER:
        pitch = Pitch(
            pitch_type='custom',
            pitch_length=105,
            pitch_width=68,
            pitch_color='#000004',
            line_color='white',
            linewidth=1.5,
            goal_type='box',
        )
        pitch.draw(ax=ax)
    else:
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.set_aspect('equal')

    # Hide axes spines (but keep pitch markings)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Transform extent from centered coords (-52.5 to 52.5) to mplsoccer coords (0 to 105)
    extent = [0, 105, 0, 68]

    # Normalize both surfaces to sum to their respective totals
    home_norm = obso_home_sum / (obso_home_sum.sum() + 1e-10) * home_total_obso
    away_norm = obso_away_sum / (obso_away_sum.sum() + 1e-10) * away_total_obso

    # Compute sigma separately for each team's distribution
    sigma_home = np.std(home_norm)
    sigma_away = np.std(away_norm)

    # Combined: home - away (positive = home, negative = away)
    combined = home_norm - away_norm

    # Create coordinate grids for contourf (in mplsoccer coords 0-105, 0-68)
    x = np.linspace(0, 105, combined.shape[1])
    y = np.linspace(0, 68, combined.shape[0])
    X, Y = np.meshgrid(x, y)

    # Sigma-based contour levels using each team's own sigma
    # Positive levels (home): 2σ, 3σ of home distribution
    # Negative levels (away): -2σ, -3σ of away distribution
    home_levels = np.array([2, 3]) * sigma_home
    away_levels = -np.array([2, 3]) * sigma_away

    # Combine and sort levels, add outer bounds
    max_val = max(abs(combined.min()), abs(combined.max()), 4 * max(sigma_home, sigma_away))
    sigma_levels = np.sort(np.concatenate([away_levels, home_levels]))
    levels = np.sort(np.concatenate([[-max_val], sigma_levels, [max_val]]))

    # Colors matching COLORS['home'] and COLORS['away'] for consistency
    # Bands: [-max,-3σ], [-3σ,-2σ], [-2σ,+2σ], [+2σ,+3σ], [+3σ,max]
    band_colors = [
        COLORS['away'],  # bright blue (away 3σ+)
        '#2d4a5e',       # muted blue (away 2-3σ)
        '#000004',       # dark background (central region)
        '#8b2d35',       # muted red (home 2-3σ)
        COLORS['home'],  # bright red (home 3σ+)
    ]

    im = ax.contourf(
        X, Y, combined,
        levels=levels,
        colors=band_colors,
        alpha=0.85,
        extend='neither',
        zorder=2,
    )

    # Add contour lines at sigma boundaries for clarity
    ax.contour(
        X, Y, combined,
        levels=sigma_levels,
        colors='white',
        linewidths=0.5,
        alpha=0.4,
        zorder=3,
    )

    # Re-draw pitch boundary lines on top to ensure uniform appearance
    # (mplsoccer lines can get obscured by contourf at edges)
    ax.plot([0, 105], [0, 0], color='white', linewidth=1.5, zorder=100)  # bottom
    ax.plot([0, 105], [68, 68], color='white', linewidth=1.5, zorder=100)  # top
    ax.plot([0, 0], [0, 68], color='white', linewidth=1.5, zorder=100)  # left
    ax.plot([105, 105], [0, 68], color='white', linewidth=1.5, zorder=100)  # right

    # Plot shots (x) and goals (o) - white for dark background
    if home_shot_positions is not None and len(home_shot_positions) > 0:
        pos = _transform_coords(home_shot_positions)
        ax.scatter(pos[:, 0], pos[:, 1], marker='x', s=40, c='white',
                   linewidths=1.5, zorder=100)

    if away_shot_positions is not None and len(away_shot_positions) > 0:
        pos = _transform_coords(away_shot_positions)
        ax.scatter(pos[:, 0], pos[:, 1], marker='x', s=40, c='white',
                   linewidths=1.5, zorder=100)

    if home_goal_positions is not None and len(home_goal_positions) > 0:
        pos = _transform_coords(home_goal_positions)
        ax.scatter(pos[:, 0], pos[:, 1], marker='o', s=80, c='white',
                   edgecolors='#000004', linewidths=1.5, zorder=101)

    if away_goal_positions is not None and len(away_goal_positions) > 0:
        pos = _transform_coords(away_goal_positions)
        ax.scatter(pos[:, 0], pos[:, 1], marker='o', s=80, c='white',
                   edgecolors='#000004', linewidths=1.5, zorder=101)

    # Add legend for shot/goal markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color='white', linestyle='None',
               markersize=8, markeredgewidth=2, label='Shot'),
        Line2D([0], [0], marker='o', color='white', linestyle='None',
               markersize=10, markeredgewidth=2, markerfacecolor='white',
               markeredgecolor='#000004', label='Goal'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
              facecolor='#2a2a4e', edgecolor='white', labelcolor='white',
              framealpha=0.9, fontsize=9)

    # Title with score and OBSO (as decimal values) - white for dark background
    title = f"Away vs Home\n{away_goals} ({away_total_obso:.2f}) - {home_goals} ({home_total_obso:.2f})"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10, color='white')

    fig.tight_layout()
    return fig, ax


def plot_obso_decomposition(
    scoring: np.ndarray,
    pitch_control: np.ndarray,
    transition: np.ndarray,
    obso: np.ndarray,
    grid: np.ndarray,
    home_positions: Optional[np.ndarray] = None,
    away_positions: Optional[np.ndarray] = None,
    ball_position: Optional[np.ndarray] = None,
    home_velocities: Optional[np.ndarray] = None,
    away_velocities: Optional[np.ndarray] = None,
    home_jerseys: Optional[list] = None,
    away_jerseys: Optional[list] = None,
    obso_total: float = 0.0,
    event_info: str = "",
    attacking_team: str = "home",
    max_velocity: float = 10.0,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Plot OBSO decomposition in 2x2 grid like Spearman Figure 5.

    a) Transition probability T(r)
    b) Control probability C(r)
    c) Scoring probability S(r)
    d) OBSO = T × C × S

    Args:
        scoring: Scoring probability grid S(r)
        pitch_control: Pitch control grid
        transition: Transition probability grid T(r)
        obso: OBSO grid (product of above)
        grid: Coordinate grid
        home_positions, away_positions: Player positions
        ball_position: Ball position
        home_jerseys, away_jerseys: Jersey numbers
        obso_total: Integrated OBSO value for title
        event_info: Event description for title
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Transform extent for mplsoccer coordinates
    extent = [0, 105, 0, 68]

    # Transform positions
    home_pos = _transform_coords(home_positions) if home_positions is not None else None
    away_pos = _transform_coords(away_positions) if away_positions is not None else None
    ball_pos = _transform_coords(ball_position) if ball_position is not None else None

    # Perceptually uniform magma colormap - dark = low, bright = high
    # Same for both home and away (no team colors on heatmaps)
    cmap = plt.cm.magma
    pitch_bg_color = '#000004'  # Dark background to match colormap minimum

    # Apply power transform to transition to boost mid-range visibility
    # (transition has concentrated peak near ball, rest is near zero)
    transition_display = np.power(transition / (transition.max() + 1e-10), 0.4)

    # All panels self-normalized (0 to their own max) for consistent visual intensity
    # Order: Transition → Control → Score → OBSO (matches logical sequence)
    panels = [
        (axes[0, 0], transition_display, "a) Transition probability T(r)", 0, 1),
        (axes[0, 1], pitch_control, "b) Control probability C(r)", 0, max(pitch_control.max(), 0.001)),
        (axes[1, 0], scoring, "c) Scoring probability S(r)", 0, max(scoring.max(), 0.001)),
        (axes[1, 1], obso, f"d) OBSO (integrated: {obso_total * 100:.2f}%)", 0, max(obso.max(), 0.001)),
    ]

    for ax, data, title, vmin, vmax in panels:
        # Use mplsoccer Pitch for consistent rendering (same as plot_time_integrated_obso)
        if HAS_MPLSOCCER:
            pitch = Pitch(
                pitch_type='custom',
                pitch_length=105,
                pitch_width=68,
                pitch_color=pitch_bg_color,
                line_color='white',
                linewidth=1.5,
                goal_type='box',
            )
            pitch.draw(ax=ax)
        else:
            # Fallback manual drawing
            ax.set_facecolor(pitch_bg_color)
            ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], 'w-', lw=1.5)
            ax.plot([52.5, 52.5], [0, 68], 'w-', lw=1)
            circle = plt.Circle((52.5, 34), 9.15, fill=False, color='w', lw=1)
            ax.add_patch(circle)
            ax.set_xlim(-4, 109)
            ax.set_ylim(-4, 72)
            ax.set_aspect('equal')
            ax.axis('off')

        # Heatmap
        ax.imshow(
            data, extent=extent, origin='lower', cmap=cmap,
            alpha=0.8, vmin=vmin, vmax=vmax, aspect='auto', zorder=2
        )

        # Find player with ball (closest to ball position)
        ball_carrier_home_idx = None
        ball_carrier_away_idx = None
        if ball_pos is not None and not np.isnan(ball_pos).any():
            min_dist = float('inf')
            if home_pos is not None and len(home_pos) > 0:
                for i, pos in enumerate(home_pos):
                    dist = np.sqrt((pos[0] - ball_pos[0])**2 + (pos[1] - ball_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        ball_carrier_home_idx = i
                        ball_carrier_away_idx = None
            if away_pos is not None and len(away_pos) > 0:
                for i, pos in enumerate(away_pos):
                    dist = np.sqrt((pos[0] - ball_pos[0])**2 + (pos[1] - ball_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        ball_carrier_away_idx = i
                        ball_carrier_home_idx = None

        # Players
        BALL_CARRIER_COLOR = '#228B22'  # Forest green for ball carrier highlight
        CIRCLE_RADIUS = 1.5  # Approximate player circle radius in meters
        MAX_ARROW_LENGTH = 4 * CIRCLE_RADIUS  # Max velocity = 4 circle radii

        if home_pos is not None and len(home_pos) > 0:
            for i, pos in enumerate(home_pos):
                if i == ball_carrier_home_idx:
                    ax.scatter([pos[0]], [pos[1]], c=COLORS['home'],
                               s=180, edgecolors=BALL_CARRIER_COLOR, linewidths=3, zorder=10)
                else:
                    ax.scatter([pos[0]], [pos[1]], c=COLORS['home'],
                               s=120, edgecolors='white', linewidths=1.5, zorder=10)

                # Draw velocity arrow (Spearman style: thin black, starts at circle edge)
                if home_velocities is not None and i < len(home_velocities):
                    vel = home_velocities[i]
                    if not np.isnan(vel).any():
                        speed = np.sqrt(vel[0]**2 + vel[1]**2)
                        if speed > 0.5:  # Only draw if moving
                            # Arrow length proportional to speed
                            arrow_len = (speed / max_velocity) * MAX_ARROW_LENGTH
                            # Unit direction
                            dx, dy = vel[0] / speed, vel[1] / speed
                            # Start at circle edge
                            start_x = pos[0] + dx * CIRCLE_RADIUS
                            start_y = pos[1] + dy * CIRCLE_RADIUS
                            ax.annotate('', xy=(start_x + dx * arrow_len, start_y + dy * arrow_len),
                                        xytext=(start_x, start_y),
                                        arrowprops=dict(arrowstyle='->', color='white',
                                                        lw=0.8, mutation_scale=8),
                                        zorder=9)

            if home_jerseys:
                for pos, jersey in zip(home_pos, home_jerseys[:len(home_pos)]):
                    ax.annotate(jersey, (pos[0], pos[1]), ha='center', va='center',
                                fontsize=7, fontweight='bold', color='white', zorder=11)

        if away_pos is not None and len(away_pos) > 0:
            for i, pos in enumerate(away_pos):
                if i == ball_carrier_away_idx:
                    ax.scatter([pos[0]], [pos[1]], c=COLORS['away'],
                               s=180, edgecolors=BALL_CARRIER_COLOR, linewidths=3, zorder=10)
                else:
                    ax.scatter([pos[0]], [pos[1]], c=COLORS['away'],
                               s=120, edgecolors='white', linewidths=1.5, zorder=10)

                # Draw velocity arrow
                if away_velocities is not None and i < len(away_velocities):
                    vel = away_velocities[i]
                    if not np.isnan(vel).any():
                        speed = np.sqrt(vel[0]**2 + vel[1]**2)
                        if speed > 0.5:
                            arrow_len = (speed / max_velocity) * MAX_ARROW_LENGTH
                            dx, dy = vel[0] / speed, vel[1] / speed
                            start_x = pos[0] + dx * CIRCLE_RADIUS
                            start_y = pos[1] + dy * CIRCLE_RADIUS
                            ax.annotate('', xy=(start_x + dx * arrow_len, start_y + dy * arrow_len),
                                        xytext=(start_x, start_y),
                                        arrowprops=dict(arrowstyle='->', color='white',
                                                        lw=0.8, mutation_scale=8),
                                        zorder=9)

            if away_jerseys:
                for pos, jersey in zip(away_pos, away_jerseys[:len(away_pos)]):
                    ax.annotate(jersey, (pos[0], pos[1]), ha='center', va='center',
                                fontsize=7, fontweight='bold', color='white', zorder=11)

        # Panel title (top-left box like Spearman)
        ax.text(0.02, 0.98, title, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

    # Main title (white text for dark background)
    if event_info:
        fig.suptitle(event_info, fontsize=12, fontweight='bold', y=0.98, color='white')

    # Dark figure background to match pitch
    fig.patch.set_facecolor('#000004')

    # Use subplots_adjust instead of tight_layout to prevent goal clipping
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05, hspace=0.1)
    return fig
