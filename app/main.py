"""
Pitch Control Dash Application

Interactive visualization of pitch control and OBSO for match events.
Loads pre-computed data for fast browsing.
"""

from __future__ import annotations

# Set matplotlib backend BEFORE any imports
import matplotlib
matplotlib.use('Agg')

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dash_bootstrap_components as dbc
import numpy as np
from dash import Dash, Input, Output, State, callback, dcc, html, ALL, ctx

from pitch_control.viz.pitch import (
    COLORS,
    fig_to_base64,
    plot_obso,
    plot_pitch_control,
    plot_cumulative_obso_timeline,
    plot_time_integrated_obso,
    plot_obso_decomposition,
)

# =============================================================================
# COLOR PALETTE & STYLING
# =============================================================================

APP_COLORS = {
    'bg_primary': '#F8FAFC',
    'bg_card': '#FFFFFF',
    'border': '#E2E8F0',
    'accent_primary': '#1E3A5F',
    'accent_secondary': '#2563EB',
    'text_primary': '#1E293B',
    'text_secondary': '#64748B',
    'home': '#E63946',
    'away': '#457B9D',
}

CUSTOM_CSS = """
/* CSS Custom Properties */
:root {
    --bg-primary: #F8FAFC;
    --bg-card: #FFFFFF;
    --border-color: #E2E8F0;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1);
    --shadow-hover: 0 10px 25px rgba(30,58,95,0.15);
    --accent-primary: #1E3A5F;
    --accent-secondary: #2563EB;
    --accent-light: #3B82F6;
    --text-primary: #1E293B;
    --text-secondary: #64748B;
    --radius-lg: 0.75rem;
    --home-color: #E63946;
    --away-color: #457B9D;
}

/* Base styles */
body {
    background-color: var(--bg-primary) !important;
    font-family: Inter, system-ui, -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
}

/* Hero section */
.hero-section {
    background: linear-gradient(135deg, rgba(30,58,95,0.08) 0%, rgba(37,99,235,0.08) 100%);
    border-bottom: 3px solid var(--accent-primary);
    padding: 2rem 0;
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
}

.hero-badge {
    display: inline-block;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 9999px;
    padding: 0.25rem 0.75rem;
    margin: 0.25rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
}

/* Card styling */
.chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    overflow: hidden;
    animation: fadeIn 0.3s ease-out;
}

.chart-card:hover {
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
}

.chart-card .card-header {
    background: var(--bg-card);
    border-bottom: 2px solid var(--accent-primary);
    padding: 1rem 1.25rem;
}

.chart-card .card-header h4 {
    color: var(--text-primary);
    font-weight: 600;
    font-size: 1.1rem;
    margin: 0;
}

.card-header-custom {
    background: var(--bg-card);
    border-bottom: 2px solid var(--accent-primary);
    padding: 0.75rem 1rem;
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-primary);
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.stat-card {
    background: var(--bg-primary);
    border-radius: 8px;
    padding: 0.75rem;
    text-align: center;
}

.stat-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--accent-secondary);
}

.stat-label {
    font-size: 0.7rem;
    color: var(--text-secondary);
    text-transform: uppercase;
}

.pitch-container img {
    max-width: 100%;
    height: auto;
}

.event-badge {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.5rem;
}

.event-badge.pass { background: #dbeafe; color: #1d4ed8; }
.event-badge.shot { background: #fef3c7; color: #d97706; }
.event-badge.home { background: #fee2e2; color: #dc2626; }
.event-badge.away { background: #e0e7ff; color: #4338ca; }

.control-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 0.25rem;
}

.event-list-item {
    padding: 0.6rem 0.75rem;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.15s ease;
    background: var(--bg-card);
}

.event-list-item:hover {
    background: #f1f5f9;
    border-color: var(--accent-secondary);
}

.event-list-item.selected {
    background: #eff6ff;
    border-color: var(--accent-secondary);
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
}

.event-list-item .event-rank {
    font-weight: 700;
    color: var(--accent-secondary);
    margin-right: 0.5rem;
}

.event-list-item .event-obso {
    font-weight: 600;
    color: var(--text-primary);
}

.event-list-item .event-time {
    font-size: 0.8rem;
    color: var(--text-secondary);
}
"""

# =============================================================================
# DATA LOADING
# =============================================================================

def get_precomputed_path(game_id: int = 1) -> Path:
    """Get path to pre-computed slim data file."""
    data_dir = Path(__file__).parent.parent / "data" / "precomputed"
    return data_dir / f"game_{game_id}_slim.npz"


def load_precomputed_data(game_id: int = 1) -> dict | None:
    """Load pre-computed slim pitch control data."""
    path = get_precomputed_path(game_id)
    if not path.exists():
        print(f"Pre-computed data not found at {path}")
        print("Run: python scripts/export_slim.py --game-id", game_id)
        return None

    print(f"Loading pre-computed data from {path}...")
    data = dict(np.load(path, allow_pickle=True))
    n_top = int(data.get('n_top', 10))
    print(f"  Loaded top {n_top} events per team (slim format)")
    return data


# Global state
DATA = None
CURRENT_GAME = None


def ensure_data_loaded(game_id: int = 1) -> bool:
    """Ensure data is loaded for the requested game."""
    global DATA, CURRENT_GAME

    if CURRENT_GAME == game_id and DATA is not None:
        return True

    DATA = load_precomputed_data(game_id)
    if DATA is not None:
        CURRENT_GAME = game_id
        return True
    return False


# Load default game on startup
ensure_data_loaded(1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_time(seconds: float) -> str:
    """Format seconds to minute with ordinal suffix (e.g., '45th minute')."""
    mins = int(seconds // 60) + 1  # Football minutes are 1-indexed
    if mins % 10 == 1 and mins != 11:
        suffix = "st"
    elif mins % 10 == 2 and mins != 12:
        suffix = "nd"
    elif mins % 10 == 3 and mins != 13:
        suffix = "rd"
    else:
        suffix = "th"
    return f"{mins}{suffix} minute"


def format_period(period: int) -> str:
    """Format period number to readable text."""
    if period == 1:
        return "First Half"
    elif period == 2:
        return "Second Half"
    else:
        return f"Period {period}"


# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True,
)

app.title = "Pitch Control Analysis"

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{CUSTOM_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

server = app.server


# =============================================================================
# LAYOUT
# =============================================================================

def create_hero_section():
    """Create the hero header."""
    return html.Div([
        dbc.Container([
            html.H1("Pitch Control Analysis", className="hero-title text-center"),
            html.Div([
                html.Span("Metrica Sports Open Data", className="hero-badge"),
                html.Span("Spearman (2018) OBSO Model", className="hero-badge"),
            ], className="text-center hero-subtitle"),
        ], fluid=True),
    ], className="hero-section")


def create_match_summary_row():
    """Row with cumulative OBSO timeline and time-integrated heatmap."""
    return dbc.Row([
        # Cumulative OBSO time series (left, wider)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Integrated Scoring Opportunity Over Time", className="card-header-custom"),
                dbc.CardBody([
                    dcc.Loading(
                        html.Img(id='timeline-plot', style={'width': '100%', 'height': 'auto'}),
                        type="circle",
                    ),
                ], style={'padding': '0.5rem'}),
            ], className="chart-card"),
        ], lg=7),

        # Time-integrated OBSO heatmap (right)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Spatial Distribution of Dangerous Space", className="card-header-custom"),
                dbc.CardBody([
                    dcc.Loading(
                        html.Img(id='integrated-heatmap', style={'width': '100%', 'height': 'auto'}),
                        type="circle",
                    ),
                ], style={'padding': '0.5rem'}),
            ], className="chart-card"),
        ], lg=5),
    ], className="mb-3")


def create_event_browser_row():
    """Row with event browser controls and 2x2 OBSO decomposition view."""
    return dbc.Row([
        # Controls (left)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top 10 OBSO Events", className="card-header-custom"),
                dbc.CardBody([
                    # Team selector
                    html.Label("Team", className="control-label"),
                    dcc.Dropdown(
                        id='team-dropdown',
                        options=[
                            {'label': 'Home', 'value': 'home'},
                            {'label': 'Away', 'value': 'away'},
                        ],
                        value='home',
                        clearable=False,
                        style={'marginBottom': '1rem'}
                    ),

                    # Clickable event list
                    html.Div(id='event-list', style={
                        'maxHeight': '400px',
                        'overflowY': 'auto',
                    }),
                ]),
            ], className="chart-card"),

            # Hidden stores
            dcc.Store(id='selected-event-idx', data=None),
        ], lg=3),

        # OBSO Decomposition view (right) - Spearman Figure 5 style
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Span("OBSO Decomposition"),
                    html.Span(id='frame-info', className="float-end text-secondary",
                              style={'fontSize': '0.8rem'}),
                ], className="card-header-custom"),
                dbc.CardBody([
                    dcc.Loading(
                        html.Div([
                            html.Img(id='pitch-image', style={'maxWidth': '100%', 'height': 'auto'}),
                        ], className="pitch-container text-center"),
                        type="circle",
                    ),
                ]),
            ], className="chart-card"),
        ], lg=9),
    ])


app.layout = html.Div([
    create_hero_section(),

    dbc.Container([
        # Game selector
        dbc.Row([
            dbc.Col([
                html.Label("Game", className="control-label"),
                dcc.Dropdown(
                    id='game-dropdown',
                    options=[
                        {'label': 'Sample Game 1', 'value': 1},
                        {'label': 'Sample Game 2', 'value': 2},
                    ],
                    value=1,
                    clearable=False,
                    style={'width': '200px'}
                ),
            ], width='auto'),
        ], className="mb-3"),

        # OBSO Explainer (per-event decomposition)
        dbc.Card([
            dbc.CardHeader("Per-Event OBSO Decomposition", className="card-header-custom"),
            dbc.CardBody([
                html.P([
                    "The Off-Ball Scoring Opportunity (OBSO) model from ",
                    html.A("Spearman (2018)", href="https://www.researchgate.net/publication/327139841_Beyond_Expected_Goals", target="_blank"),
                    " answers: ",
                    html.Em("\"How dangerous is this attacking position, considering where the ball could go next?\""),
                    " Think of it as three steps:"
                ], style={'marginBottom': '0.75rem'}),
                dbc.Row([
                    dbc.Col([
                        html.Strong("1. Pass - Transition T(r)", style={'color': '#2563EB'}),
                        html.P("Where could the ball be played next? More likely to nearby areas where teammates can receive.",
                               className="text-secondary mb-0", style={'fontSize': '0.85rem'}),
                    ], md=4),
                    dbc.Col([
                        html.Strong("2. Receive - Control C(r)", style={'color': '#2563EB'}),
                        html.P("If the ball arrives, can we win it? Based on which team's players can reach each location first.",
                               className="text-secondary mb-0", style={'fontSize': '0.85rem'}),
                    ], md=4),
                    dbc.Col([
                        html.Strong("3. Shoot - Scoring S(r)", style={'color': '#2563EB'}),
                        html.P("If we control it there, how likely is a goal? Higher near goal, lower further out.",
                               className="text-secondary mb-0", style={'fontSize': '0.85rem'}),
                    ], md=4),
                ], className="mb-2"),
                html.P([
                    html.Strong("OBSO"),
                    " - For every location on the pitch, we ask: (1) how likely is the ball to go there, (2) can we win it, and (3) can we score? Multiplying these probabilities together gives the Off-Ball Scoring Opportunity. Locations with higher values have the most potential attacking threat."
                ], className="mb-0", style={'backgroundColor': '#f1f5f9', 'padding': '0.5rem', 'borderRadius': '4px'}),
            ]),
        ], className="chart-card mb-3"),

        # Visual legend
        dbc.Card([
            dbc.CardHeader("Reading the Pitch", className="card-header-custom"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Strong("Players", style={'color': '#2563EB'}),
                        html.P([
                            html.Span("Red circles", style={'color': '#E63946', 'fontWeight': '600'}),
                            " = Home team. ",
                            html.Span("Blue circles", style={'color': '#457B9D', 'fontWeight': '600'}),
                            " = Away team. ",
                            html.Span("Green circle", style={'color': '#27ae60', 'fontWeight': '600'}),
                            " = Player on the ball."
                        ], className="text-secondary mb-0", style={'fontSize': '0.85rem'}),
                    ], md=4),
                    dbc.Col([
                        html.Strong("Arrows", style={'color': '#2563EB'}),
                        html.P("Black arrows show player velocity - direction of movement and speed. Longer arrows mean faster movement.",
                               className="text-secondary mb-0", style={'fontSize': '0.85rem'}),
                    ], md=4),
                    dbc.Col([
                        html.Strong("Heatmaps", style={'color': '#2563EB'}),
                        html.P("Intensity shows probability. Darker/brighter areas have higher values. Red heatmaps = home attacking, blue = away attacking.",
                               className="text-secondary mb-0", style={'fontSize': '0.85rem'}),
                    ], md=4),
                ]),
            ]),
        ], className="chart-card mb-3"),

        # Event browser row
        create_event_browser_row(),

        # Integrated OBSO Explainer
        dbc.Card([
            dbc.CardHeader("Match-Level OBSO", className="card-header-custom"),
            dbc.CardBody([
                html.P([
                    "While the decomposition above shows OBSO for a single moment, we can also aggregate across the match to see the bigger picture:"
                ], style={'marginBottom': '0.75rem'}),
                dbc.Row([
                    dbc.Col([
                        html.Strong("Timeline", style={'color': '#2563EB'}),
                        html.P("Cumulative OBSO over time. Shows how attacking threat builds through the match, with goal moments marked.",
                               className="text-secondary mb-0", style={'fontSize': '0.85rem'}),
                    ], md=6),
                    dbc.Col([
                        html.Strong("Heatmap", style={'color': '#2563EB'}),
                        html.P("Time-integrated OBSO surface. Shows which areas of the pitch generated the most attacking threat across all events.",
                               className="text-secondary mb-0", style={'fontSize': '0.85rem'}),
                    ], md=6),
                ]),
            ]),
        ], className="chart-card mb-3"),

        # Match summary row (timeline + heatmap)
        create_match_summary_row(),

    ], fluid=True),
])


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output('timeline-plot', 'src'),
    Output('integrated-heatmap', 'src'),
    Input('game-dropdown', 'value'),
)
def update_match_summary(game_id):
    """Update match summary visualizations when game changes."""
    global DATA, CURRENT_GAME

    if game_id != CURRENT_GAME:
        if not ensure_data_loaded(game_id):
            return "", ""

    if DATA is None:
        return "", ""

    # Cumulative OBSO timeline
    times = DATA['times']
    obso_home_totals = DATA['obso_home_totals']
    obso_away_totals = DATA['obso_away_totals']
    goal_times_home = list(DATA['goal_times_home'])
    goal_times_away = list(DATA['goal_times_away'])

    fig_timeline = plot_cumulative_obso_timeline(
        times, obso_home_totals, obso_away_totals,
        goal_times_home=goal_times_home,
        goal_times_away=goal_times_away,
        figsize=(10, 3.5),
    )
    timeline_img = fig_to_base64(fig_timeline, dpi=100)

    # Time-integrated OBSO heatmap
    obso_home_integrated = DATA['obso_home_time_integrated']
    obso_away_integrated = DATA['obso_away_time_integrated']
    grid = DATA['grid']

    home_total = float(obso_home_totals.sum())
    away_total = float(obso_away_totals.sum())

    # Extract shot and goal positions from slim format (already in meters)
    shot_x = DATA['shot_positions_x']
    shot_y = DATA['shot_positions_y']
    shot_teams = DATA['shot_teams']
    shot_times = DATA['shot_times']

    home_shot_pos = []
    away_shot_pos = []
    home_goal_pos = []
    away_goal_pos = []

    for i in range(len(shot_x)):
        x, y = float(shot_x[i]), float(shot_y[i])
        if np.isnan(x) or np.isnan(y):
            continue

        is_home = 'home' in str(shot_teams[i]).lower()
        shot_time = shot_times[i]

        # Check if this shot is a goal by matching time
        is_goal = False
        goal_times = goal_times_home if is_home else goal_times_away
        for gt in goal_times:
            if abs(shot_time - gt) < 0.5:  # Within 0.5 seconds
                is_goal = True
                break

        if is_goal:
            if is_home:
                home_goal_pos.append([x, y])
            else:
                away_goal_pos.append([x, y])
        else:
            if is_home:
                home_shot_pos.append([x, y])
            else:
                away_shot_pos.append([x, y])

    fig_heatmap, _ = plot_time_integrated_obso(
        obso_home_integrated, obso_away_integrated, grid,
        home_goals=len(goal_times_home),
        away_goals=len(goal_times_away),
        home_total_obso=home_total,
        away_total_obso=away_total,
        home_shot_positions=np.array(home_shot_pos) if home_shot_pos else None,
        away_shot_positions=np.array(away_shot_pos) if away_shot_pos else None,
        home_goal_positions=np.array(home_goal_pos) if home_goal_pos else None,
        away_goal_positions=np.array(away_goal_pos) if away_goal_pos else None,
        figsize=(8, 5),
    )
    heatmap_img = fig_to_base64(fig_heatmap, dpi=100)

    return timeline_img, heatmap_img


@callback(
    Output('event-list', 'children'),
    Input('game-dropdown', 'value'),
    Input('team-dropdown', 'value'),
)
def update_event_list(game_id, team):
    """Generate clickable list of top 10 OBSO events for selected team."""
    global DATA, CURRENT_GAME

    if game_id != CURRENT_GAME:
        if not ensure_data_loaded(game_id):
            return html.P("No data", className="text-secondary")

    if DATA is None:
        return html.P("No data", className="text-secondary")

    n_top = int(DATA.get('n_top', 10))

    # In slim format, top events are already sorted by OBSO
    # First n_top are home team, next n_top are away team
    if team == 'home':
        start_idx = 0
    else:
        start_idx = n_top

    items = []
    for rank in range(1, n_top + 1):
        slim_idx = start_idx + rank - 1
        obso_val = DATA['top_obso_totals'][slim_idx]
        time = DATA['top_times'][slim_idx]
        period = int(DATA['top_periods'][slim_idx])
        event_type = str(DATA['top_event_types'][slim_idx])

        item = html.Div([
            html.Span(f"#{rank}", className="event-rank"),
            html.Span(f"OBSO: {obso_val * 100:.2f}%", className="event-obso"),
            html.Span(f" ({event_type})", className="event-type-label", style={'fontSize': '0.75rem', 'color': '#64748B'}),
            html.Br(),
            html.Span(f"{format_period(period)}, {format_time(time)}", className="event-time"),
        ], className="event-list-item", id={'type': 'event-item', 'index': slim_idx})
        items.append(item)

    return items


@callback(
    Output('selected-event-idx', 'data'),
    Input({'type': 'event-item', 'index': ALL}, 'n_clicks'),
    State({'type': 'event-item', 'index': ALL}, 'id'),
    prevent_initial_call=True,
)
def select_event(n_clicks_list, ids):
    """Handle click on event list item."""
    if not ctx.triggered_id:
        return None
    return ctx.triggered_id['index']


@callback(
    Output('pitch-image', 'src'),
    Output('frame-info', 'children'),
    Input('selected-event-idx', 'data'),
    Input('team-dropdown', 'value'),
    prevent_initial_call=False,
)
def update_event_visualization(slim_idx, team_filter):
    """Serve pre-rendered OBSO decomposition image."""
    if DATA is None or slim_idx is None:
        return "", ""

    # Serve pre-rendered image (instant!)
    pitch_img = str(DATA['top_event_images'][slim_idx])

    frame = int(DATA['top_frame_indices'][slim_idx])

    return pitch_img, f"Frame {frame}"


# =============================================================================
# MAIN
# =============================================================================

def main():
    port = int(os.environ.get('PORT', 8050))
    host = '0.0.0.0' if os.environ.get('RENDER') else '127.0.0.1'
    debug = not os.environ.get('RENDER')

    print(f"\nStarting Pitch Control app at http://{host}:{port}\n")
    app.run(debug=debug, port=port, host=host)


if __name__ == '__main__':
    main()
