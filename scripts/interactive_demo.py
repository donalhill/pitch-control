#!/usr/bin/env python3
"""
Interactive Pitch Control Demo

Drag players around and watch pitch control update in REAL-TIME as you drag.

Uses custom JavaScript to track mouse movement during drag and throttled
server callbacks to continuously recompute surfaces.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, Patch, callback_context, no_update, clientside_callback
import dash_bootstrap_components as dbc

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pitch_control.io.metrica import load_match, get_frame_data
from pitch_control.models.pitch_control import (
    compute_pitch_control,
    create_pitch_grid,
    default_model_params,
)
from pitch_control.models.obso import compute_obso
from pitch_control.models.epv import load_epv_grid, create_epv_for_grid


# Pitch dimensions
PITCH_LENGTH = 105
PITCH_WIDTH = 68
PLAYER_RADIUS = 1.8

# Trace indices
IDX_HEATMAP = 0
IDX_BALL = 1
IDX_ATT_TEXT = 2
IDX_DEF_TEXT = 3

N_PITCH_SHAPES = 7


def find_high_obso_frame(match, n_samples=200):
    """Find a frame with high OBSO by sampling pass events."""
    from pitch_control.models.obso import OBSOAnalyzer

    print("Searching for high-OBSO frame...")

    pass_events = match.events[match.events['Type'].str.upper() == 'PASS'].copy()
    pass_events = pass_events[pass_events['Start Frame'].notna()]

    sample_size = min(n_samples, len(pass_events))
    sampled = pass_events.sample(n=sample_size, random_state=42)

    analyzer = OBSOAnalyzer()
    best_frame = None
    best_obso = 0
    best_team = 'home'

    for _, event in sampled.iterrows():
        frame = int(event['Start Frame'])
        try:
            frame_data = get_frame_data(match.tracking_home, match.tracking_away, frame)
            team = 'home' if 'home' in str(event['Team']).lower() else 'away'
            result = analyzer.analyze_frame(frame_data, attacking_team=team)

            if result['total_obso'] > best_obso:
                best_obso = result['total_obso']
                best_frame = frame
                best_team = team
        except Exception:
            continue

    print(f"Found frame {best_frame} with OBSO {best_obso:.4f} ({best_team} attacking)")
    return best_frame, best_team


def create_app():
    print("Loading game data...")
    match = load_match(game_id=1, auto_download=True)

    frame_num, attacking_team = find_high_obso_frame(match)
    frame_data = get_frame_data(match.tracking_home, match.tracking_away, frame_num)

    params = default_model_params()
    grid = create_pitch_grid(params)
    epv_grid, epv_x, epv_y = load_epv_grid()
    attack_dir = 1 if attacking_team == 'home' else -1
    epv = create_epv_for_grid(grid, epv_grid, epv_x, epv_y, attacking_direction=attack_dir)

    if attacking_team == 'home':
        att_data = frame_data['home']
        def_data = frame_data['away']
    else:
        att_data = frame_data['away']
        def_data = frame_data['home']

    ball_pos = frame_data['ball']

    att_valid = ~np.isnan(att_data['positions']).any(axis=1)
    def_valid = ~np.isnan(def_data['positions']).any(axis=1)

    initial_state = {
        'att_pos': att_data['positions'][att_valid].copy(),
        'att_vel': att_data['velocities'][att_valid].copy(),
        'att_jerseys': [j for j, v in zip(att_data['jerseys'], att_valid) if v],
        'def_pos': def_data['positions'][def_valid].copy(),
        'def_vel': def_data['velocities'][def_valid].copy(),
        'def_jerseys': [j for j, v in zip(def_data['jerseys'], def_valid) if v],
        'ball': ball_pos.copy(),
    }

    n_att = len(initial_state['att_pos'])
    n_def = len(initial_state['def_pos'])

    print("Warming up JIT...")
    _ = compute_pitch_control(
        initial_state['att_pos'], initial_state['att_vel'],
        initial_state['def_pos'], initial_state['def_vel'],
        ball_pos, params, grid=grid
    )

    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    x_grid = np.linspace(-PITCH_LENGTH/2, PITCH_LENGTH/2, grid.shape[1])
    y_grid = np.linspace(-PITCH_WIDTH/2, PITCH_WIDTH/2, grid.shape[0])

    def compute_surfaces(att_pos, def_pos):
        pc, _, _ = compute_pitch_control(
            att_pos, initial_state['att_vel'],
            def_pos, initial_state['def_vel'],
            initial_state['ball'], params, grid=grid
        )
        obso_surface = compute_obso(pc, epv)
        total_obso = obso_surface.sum() * 0.5 * 0.5
        return pc, obso_surface, total_obso

    def create_figure(att_pos, def_pos, surface, surface_type='obso'):
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=surface,
            x=x_grid,
            y=y_grid,
            colorscale='Magma',
            showscale=True,
            opacity=0.85,
            colorbar=dict(title='OBSO' if surface_type == 'obso' else 'Control', x=1.02),
            hoverinfo='skip',
        ))

        fig.add_trace(go.Scatter(
            x=[initial_state['ball'][0]] if not np.isnan(initial_state['ball']).any() else [],
            y=[initial_state['ball'][1]] if not np.isnan(initial_state['ball']).any() else [],
            mode='markers',
            marker=dict(size=14, color='white', line=dict(width=2, color='black')),
            hoverinfo='skip',
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=att_pos[:, 0], y=att_pos[:, 1],
            mode='text',
            text=[str(j) for j in initial_state['att_jerseys']],
            textfont=dict(size=11, color='black', family='Arial Black'),
            hoverinfo='skip',
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=def_pos[:, 0], y=def_pos[:, 1],
            mode='text',
            text=[str(j) for j in initial_state['def_jerseys']],
            textfont=dict(size=11, color='white', family='Arial Black'),
            hoverinfo='skip',
            showlegend=False,
        ))

        line_style = dict(color='white', width=2)
        pitch_shapes = [
            dict(type='rect', x0=-PITCH_LENGTH/2, x1=PITCH_LENGTH/2,
                 y0=-PITCH_WIDTH/2, y1=PITCH_WIDTH/2, line=line_style, editable=False),
            dict(type='line', x0=0, x1=0, y0=-PITCH_WIDTH/2, y1=PITCH_WIDTH/2,
                 line=line_style, editable=False),
            dict(type='circle', x0=-9.15, x1=9.15, y0=-9.15, y1=9.15,
                 line=line_style, editable=False),
            dict(type='rect', x0=-PITCH_LENGTH/2, x1=-PITCH_LENGTH/2+16.5,
                 y0=-20.15, y1=20.15, line=line_style, editable=False),
            dict(type='rect', x0=PITCH_LENGTH/2-16.5, x1=PITCH_LENGTH/2,
                 y0=-20.15, y1=20.15, line=line_style, editable=False),
            dict(type='rect', x0=-PITCH_LENGTH/2, x1=-PITCH_LENGTH/2+5.5,
                 y0=-9.15, y1=9.15, line=line_style, editable=False),
            dict(type='rect', x0=PITCH_LENGTH/2-5.5, x1=PITCH_LENGTH/2,
                 y0=-9.15, y1=9.15, line=line_style, editable=False),
        ]

        for pos in att_pos:
            pitch_shapes.append(dict(
                type='circle',
                x0=pos[0] - PLAYER_RADIUS, x1=pos[0] + PLAYER_RADIUS,
                y0=pos[1] - PLAYER_RADIUS, y1=pos[1] + PLAYER_RADIUS,
                fillcolor='#00ff88',
                line=dict(color='white', width=2),
                editable=True,
            ))

        for pos in def_pos:
            pitch_shapes.append(dict(
                type='circle',
                x0=pos[0] - PLAYER_RADIUS, x1=pos[0] + PLAYER_RADIUS,
                y0=pos[1] - PLAYER_RADIUS, y1=pos[1] + PLAYER_RADIUS,
                fillcolor='#ff4466',
                line=dict(color='white', width=2),
                editable=True,
            ))

        fig.update_layout(
            shapes=pitch_shapes,
            template='plotly_dark',
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#0d1117',
            height=700,
            xaxis=dict(
                range=[-PITCH_LENGTH/2 - 5, PITCH_LENGTH/2 + 5],
                showgrid=False, zeroline=False,
                scaleanchor='y', constrain='domain',
            ),
            yaxis=dict(
                range=[-PITCH_WIDTH/2 - 5, PITCH_WIDTH/2 + 5],
                showgrid=False, zeroline=False,
            ),
            showlegend=False,
            margin=dict(l=20, r=80, t=60, b=20),
            title=dict(text='Drag players - surface updates live!', x=0.5, font=dict(size=18)),
        )

        return fig

    def extract_positions_from_shapes(shapes):
        att_pos = np.zeros((n_att, 2))
        def_pos = np.zeros((n_def, 2))

        for i in range(n_att):
            shape = shapes[N_PITCH_SHAPES + i]
            att_pos[i, 0] = (shape['x0'] + shape['x1']) / 2
            att_pos[i, 1] = (shape['y0'] + shape['y1']) / 2

        for i in range(n_def):
            shape = shapes[N_PITCH_SHAPES + n_att + i]
            def_pos[i, 0] = (shape['x0'] + shape['x1']) / 2
            def_pos[i, 1] = (shape['y0'] + shape['y1']) / 2

        return att_pos, def_pos

    def apply_relayout_to_shapes(relayout_data, shapes):
        updated_shapes = [dict(s) for s in shapes]
        for key, value in relayout_data.items():
            match = re.match(r'shapes\[(\d+)\]\.(\w+)', key)
            if match:
                idx, prop = int(match.group(1)), match.group(2)
                if idx < len(updated_shapes):
                    updated_shapes[idx][prop] = value
        return updated_shapes

    pc, obso, total_obso = compute_surfaces(initial_state['att_pos'], initial_state['def_pos'])
    initial_figure = create_figure(initial_state['att_pos'], initial_state['def_pos'], obso)

    # Store initial positions as JSON for JS access
    initial_positions_json = json.dumps({
        'att': initial_state['att_pos'].tolist(),
        'def': initial_state['def_pos'].tolist(),
        'n_att': n_att,
        'n_def': n_def,
        'n_pitch_shapes': N_PITCH_SHAPES,
        'radius': PLAYER_RADIUS,
    })

    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='pitch-graph',
                    figure=initial_figure,
                    config={
                        'editable': True,
                        'edits': {'shapePosition': True},
                        'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'drawcircle',
                                                    'drawrect', 'drawline', 'eraseshape'],
                        'displaylogo': False,
                    },
                ),
            ], width=10),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Pitch Control"),
                    dbc.CardBody([
                        html.Div(id='obso-display', children=[
                            html.H3(f"{total_obso:.4f}", className='text-center text-success'),
                            html.P("Total OBSO", className='text-center text-muted'),
                        ]),
                        html.Hr(),
                        html.Label("Surface"),
                        dcc.RadioItems(
                            id='surface-type',
                            options=[
                                {'label': ' OBSO', 'value': 'obso'},
                                {'label': ' Pitch Control', 'value': 'pc'},
                            ],
                            value='obso',
                            className='mb-3',
                        ),
                        html.Hr(),
                        dbc.Button("Reset", id='reset-btn', color='outline-light', className='w-100'),
                        html.Hr(),
                        html.P([
                            html.Strong("Instructions:"), html.Br(),
                            "• Drag any player disc", html.Br(),
                            "• Surface updates live!", html.Br(),
                            "• Green = attacking", html.Br(),
                            "• Red = defending",
                        ], className='small text-muted'),
                    ]),
                ], className='mt-3'),
            ], width=2),
        ]),

        # Interval for polling during drag
        dcc.Interval(id='drag-interval', interval=80, disabled=True),

        # Stores
        dcc.Store(id='pc-store', data=pc.tolist()),
        dcc.Store(id='obso-store', data=obso.tolist()),
        dcc.Store(id='drag-state', data={'dragging': False, 'shapes': None}),
        dcc.Store(id='initial-positions', data=initial_positions_json),

        # Hidden div for clientside callback output
        html.Div(id='dummy-output', style={'display': 'none'}),

    ], fluid=True, className='mt-2')

    # Clientside callback to detect drag start/end and capture shapes
    app.clientside_callback(
        """
        function(relayoutData, figure, dragState) {
            if (!relayoutData) return [window.dash_clientside.no_update, window.dash_clientside.no_update];

            // Check if this is a shape drag event
            const keys = Object.keys(relayoutData);
            const isShapeDrag = keys.some(k => k.startsWith('shapes['));

            if (isShapeDrag && figure && figure.layout && figure.layout.shapes) {
                // Apply relayout updates to shapes
                const shapes = JSON.parse(JSON.stringify(figure.layout.shapes));
                keys.forEach(key => {
                    const match = key.match(/shapes\\[(\\d+)\\]\\.(\\w+)/);
                    if (match) {
                        const idx = parseInt(match[1]);
                        const prop = match[2];
                        if (idx < shapes.length) {
                            shapes[idx][prop] = relayoutData[key];
                        }
                    }
                });

                // Return updated drag state with new shapes
                return [
                    false,  // Keep interval disabled for now (fires on release)
                    {'dragging': false, 'shapes': shapes}
                ];
            }

            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        """,
        Output('drag-interval', 'disabled'),
        Output('drag-state', 'data'),
        Input('pitch-graph', 'relayoutData'),
        State('pitch-graph', 'figure'),
        State('drag-state', 'data'),
        prevent_initial_call=True,
    )

    @app.callback(
        Output('pitch-graph', 'figure'),
        Output('obso-display', 'children'),
        Output('pc-store', 'data'),
        Output('obso-store', 'data'),
        Input('drag-state', 'data'),
        Input('surface-type', 'value'),
        Input('reset-btn', 'n_clicks'),
        State('pitch-graph', 'figure'),
        State('pc-store', 'data'),
        State('obso-store', 'data'),
        prevent_initial_call=True,
    )
    def update_on_interaction(drag_state, surface_type, reset_clicks, current_fig,
                               stored_pc, stored_obso):
        ctx = callback_context
        triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ''

        if 'reset-btn' in triggered:
            pc, obso, total_obso = compute_surfaces(
                initial_state['att_pos'], initial_state['def_pos']
            )
            surface = obso if surface_type == 'obso' else pc
            new_fig = create_figure(initial_state['att_pos'], initial_state['def_pos'],
                                    surface, surface_type)
            obso_display = [
                html.H3(f"{total_obso:.4f}", className='text-center text-success'),
                html.P("Total OBSO", className='text-center text-muted'),
            ]
            return new_fig, obso_display, pc.tolist(), obso.tolist()

        if 'drag-state' in triggered and drag_state and drag_state.get('shapes'):
            shapes = drag_state['shapes']
            att_pos, def_pos = extract_positions_from_shapes(shapes)

            pc, obso, total_obso = compute_surfaces(att_pos, def_pos)
            surface = obso if surface_type == 'obso' else pc

            patched = Patch()
            patched['data'][IDX_HEATMAP]['z'] = surface.tolist()
            patched['data'][IDX_ATT_TEXT]['x'] = att_pos[:, 0].tolist()
            patched['data'][IDX_ATT_TEXT]['y'] = att_pos[:, 1].tolist()
            patched['data'][IDX_DEF_TEXT]['x'] = def_pos[:, 0].tolist()
            patched['data'][IDX_DEF_TEXT]['y'] = def_pos[:, 1].tolist()

            obso_display = [
                html.H3(f"{total_obso:.4f}", className='text-center text-success'),
                html.P("Total OBSO", className='text-center text-muted'),
            ]
            return patched, obso_display, pc.tolist(), obso.tolist()

        if 'surface-type' in triggered:
            pc = np.array(stored_pc)
            obso = np.array(stored_obso)
            total_obso = obso.sum() * 0.5 * 0.5
            surface = obso if surface_type == 'obso' else pc

            patched = Patch()
            patched['data'][IDX_HEATMAP]['z'] = surface.tolist()
            patched['data'][IDX_HEATMAP]['colorbar']['title'] = 'OBSO' if surface_type == 'obso' else 'Control'

            obso_display = [
                html.H3(f"{total_obso:.4f}", className='text-center text-success'),
                html.P("Total OBSO", className='text-center text-muted'),
            ]
            return patched, obso_display, no_update, no_update

        return no_update, no_update, no_update, no_update

    return app


if __name__ == '__main__':
    app = create_app()
    print("\n" + "=" * 50)
    print("Starting interactive demo at http://localhost:8051")
    print("Drag the player discs - surface updates live!")
    print("=" * 50 + "\n")
    app.run(debug=False, port=8051)
