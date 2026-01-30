#!/usr/bin/env python3
"""
Interactive Pitch Control Demo - Canvas Edition

51st minute away team shot (frame 75262) - "should have passed" moment.
Uses exact Spearman OBSO calculation matching the deployed app.

Flask + HTML5 Canvas for smooth drag with live surface updates.
"""

import sys
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
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

app = Flask(__name__)

# Spearman Table 1 MAP parameters for transition probability
SIGMA = 23.9  # Mean distance between on-ball events (meters)
ALPHA = 1.04  # Preference for maintaining possession

# Global state
STATE = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Pitch Control - 51st Minute Shot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #1a1a2e;
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            display: flex;
            gap: 20px;
            align-items: flex-start;
        }
        canvas {
            border-radius: 8px;
            cursor: grab;
        }
        canvas:active { cursor: grabbing; }
        .sidebar {
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            width: 220px;
        }
        .sidebar h2 { margin-bottom: 10px; font-size: 16px; }
        .context {
            background: #1a1a2e;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 15px;
            font-size: 13px;
            line-height: 1.5;
        }
        .context .time { color: #ffd700; font-weight: bold; }
        .context .event { color: #ff6b6b; }
        .obso-value {
            font-size: 36px;
            color: #00ff88;
            text-align: center;
            margin: 10px 0;
        }
        .obso-label {
            text-align: center;
            color: #888;
            font-size: 12px;
        }
        hr { border: none; border-top: 1px solid #333; margin: 15px 0; }
        .radio-group label {
            display: block;
            padding: 8px;
            cursor: pointer;
            border-radius: 4px;
            font-size: 14px;
        }
        .radio-group label:hover { background: #1a1a2e; }
        .radio-group input { margin-right: 8px; }
        button {
            width: 100%;
            padding: 10px;
            background: transparent;
            border: 1px solid #fff;
            color: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-bottom: 8px;
        }
        button:hover { background: #333; }
        .instructions {
            margin-top: 15px;
            font-size: 11px;
            color: #666;
            line-height: 1.6;
        }
        .fps { font-size: 11px; color: #444; text-align: center; margin-top: 10px; }
        .team-legend {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin: 10px 0;
            font-size: 12px;
        }
        .team-legend span {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .dot.att { background: #457B9D; }
        .dot.def { background: #E63946; }
    </style>
</head>
<body>
    <div class="container">
        <canvas id="pitch" width="900" height="600"></canvas>
        <div class="sidebar">
            <h2>OBSO Analysis</h2>
            <div class="context">
                <span class="time">51st minute</span><br>
                <span class="event">Away shot - saved</span><br>
                Should they have passed?
            </div>

            <div class="obso-value" id="obso-value">2.75%</div>
            <div class="obso-label">Integrated OBSO (Spearman)</div>

            <div class="team-legend">
                <span><div class="dot att"></div> Away (attacking)</span>
                <span><div class="dot def"></div> Home (defending)</span>
            </div>

            <hr>
            <div class="radio-group">
                <label><input type="radio" name="surface" value="obso" checked> OBSO</label>
                <label><input type="radio" name="surface" value="transition"> Transition T(r)</label>
                <label><input type="radio" name="surface" value="pc"> Pitch Control</label>
                <label><input type="radio" name="surface" value="scoring"> Scoring S(r)</label>
            </div>
            <hr>
            <button id="reset-btn">Reset Positions</button>
            <div class="instructions">
                <strong>Drag players</strong> to explore<br>
                alternative passing options.<br><br>
                Surface updates in real-time<br>
                using Spearman's model.
            </div>
            <div class="fps" id="fps">-- fps</div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('pitch');
        const ctx = canvas.getContext('2d');

        const PITCH_LENGTH = 105;
        const PITCH_WIDTH = 68;
        const PLAYER_RADIUS = 1.8;

        const MARGIN = 40;
        const scaleX = (canvas.width - 2 * MARGIN) / PITCH_LENGTH;
        const scaleY = (canvas.height - 2 * MARGIN) / PITCH_WIDTH;
        const scale = Math.min(scaleX, scaleY);
        const offsetX = (canvas.width - PITCH_LENGTH * scale) / 2;
        const offsetY = (canvas.height - PITCH_WIDTH * scale) / 2;

        function toCanvas(x, y) {
            return [
                offsetX + (x + PITCH_LENGTH / 2) * scale,
                offsetY + (PITCH_WIDTH / 2 - y) * scale
            ];
        }

        function toPitch(cx, cy) {
            return [
                (cx - offsetX) / scale - PITCH_LENGTH / 2,
                PITCH_WIDTH / 2 - (cy - offsetY) / scale
            ];
        }

        // State from server
        let attPos = {{ att_pos | tojson }};
        let defPos = {{ def_pos | tojson }};
        const attJerseys = {{ att_jerseys | tojson }};
        const defJerseys = {{ def_jerseys | tojson }};
        const ballPos = {{ ball_pos | tojson }};
        const initialAttPos = JSON.parse(JSON.stringify(attPos));
        const initialDefPos = JSON.parse(JSON.stringify(defPos));

        let surface = {{ initial_surface | tojson }};
        let surfaceType = 'obso';
        let integratedObso = {{ initial_obso }};

        let dragging = null;
        let computing = false;
        let lastComputeTime = 0;
        let frameCount = 0;
        let lastFpsTime = Date.now();

        // Magma colormap
        function magma(t) {
            t = Math.max(0, Math.min(1, t));
            if (t < 0.25) {
                const s = t / 0.25;
                return [Math.floor(s * 60), 0, Math.floor(s * 80)];
            } else if (t < 0.5) {
                const s = (t - 0.25) / 0.25;
                return [60 + Math.floor(s * 100), Math.floor(s * 20), 80 + Math.floor(s * 40)];
            } else if (t < 0.75) {
                const s = (t - 0.5) / 0.25;
                return [160 + Math.floor(s * 60), 20 + Math.floor(s * 80), 120 - Math.floor(s * 60)];
            } else {
                const s = (t - 0.75) / 0.25;
                return [220 + Math.floor(s * 35), 100 + Math.floor(s * 120), 60 + Math.floor(s * 80)];
            }
        }

        function drawPitch() {
            ctx.fillStyle = '#0d1117';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw surface
            const rows = surface.length;
            const cols = surface[0].length;
            const cellW = PITCH_LENGTH / cols * scale;
            const cellH = PITCH_WIDTH / rows * scale;

            let maxVal = 0;
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    if (surface[r][c] > maxVal) maxVal = surface[r][c];
                }
            }
            if (maxVal === 0) maxVal = 1;

            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const val = surface[r][c] / maxVal;
                    const [cr, cg, cb] = magma(val);
                    ctx.fillStyle = `rgba(${cr},${cg},${cb},0.85)`;
                    const px = -PITCH_LENGTH/2 + c * PITCH_LENGTH / cols;
                    // Row 0 is at y=-34 (bottom), last row at y=34 (top)
                    const py = -PITCH_WIDTH/2 + r * PITCH_WIDTH / rows;
                    const [cx, cy] = toCanvas(px, py);
                    ctx.fillRect(cx, cy, cellW + 1, cellH + 1);
                }
            }

            // Pitch markings
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;

            const [ox1, oy1] = toCanvas(-PITCH_LENGTH/2, PITCH_WIDTH/2);
            const [ox2, oy2] = toCanvas(PITCH_LENGTH/2, -PITCH_WIDTH/2);
            ctx.strokeRect(ox1, oy1, ox2 - ox1, oy2 - oy1);

            const [cx1, cy1] = toCanvas(0, PITCH_WIDTH/2);
            const [cx2, cy2] = toCanvas(0, -PITCH_WIDTH/2);
            ctx.beginPath();
            ctx.moveTo(cx1, cy1);
            ctx.lineTo(cx2, cy2);
            ctx.stroke();

            const [ccx, ccy] = toCanvas(0, 0);
            ctx.beginPath();
            ctx.arc(ccx, ccy, 9.15 * scale, 0, Math.PI * 2);
            ctx.stroke();

            const [pa1x, pa1y] = toCanvas(-PITCH_LENGTH/2, 20.15);
            const [pa2x, pa2y] = toCanvas(-PITCH_LENGTH/2 + 16.5, -20.15);
            ctx.strokeRect(pa1x, pa1y, pa2x - pa1x, pa2y - pa1y);

            const [pa3x, pa3y] = toCanvas(PITCH_LENGTH/2 - 16.5, 20.15);
            const [pa4x, pa4y] = toCanvas(PITCH_LENGTH/2, -20.15);
            ctx.strokeRect(pa3x, pa3y, pa4x - pa3x, pa4y - pa3y);

            const [sb1x, sb1y] = toCanvas(-PITCH_LENGTH/2, 9.15);
            const [sb2x, sb2y] = toCanvas(-PITCH_LENGTH/2 + 5.5, -9.15);
            ctx.strokeRect(sb1x, sb1y, sb2x - sb1x, sb2y - sb1y);

            const [sb3x, sb3y] = toCanvas(PITCH_LENGTH/2 - 5.5, 9.15);
            const [sb4x, sb4y] = toCanvas(PITCH_LENGTH/2, -9.15);
            ctx.strokeRect(sb3x, sb3y, sb4x - sb3x, sb4y - sb3y);

            // Ball
            const [bx, by] = toCanvas(ballPos[0], ballPos[1]);
            ctx.beginPath();
            ctx.arc(bx, by, 6, 0, Math.PI * 2);
            ctx.fillStyle = 'white';
            ctx.fill();
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Players
            function drawPlayer(pos, jersey, fillColor, textColor, highlight = false) {
                const [cx, cy] = toCanvas(pos[0], pos[1]);
                const r = PLAYER_RADIUS * scale;

                // Green highlight for ball carrier
                if (highlight) {
                    ctx.beginPath();
                    ctx.arc(cx, cy, r + 4, 0, Math.PI * 2);
                    ctx.strokeStyle = '#00ff88';
                    ctx.lineWidth = 3;
                    ctx.stroke();
                }

                ctx.beginPath();
                ctx.arc(cx, cy, r, 0, Math.PI * 2);
                ctx.fillStyle = fillColor;
                ctx.fill();
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.fillStyle = textColor;
                ctx.font = 'bold 11px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(jersey.toString(), cx, cy);
            }

            // Find ball carrier (closest attacking player to ball)
            let ballCarrierIdx = 0;
            let minDist = Infinity;
            attPos.forEach((pos, i) => {
                const dx = pos[0] - ballPos[0];
                const dy = pos[1] - ballPos[1];
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < minDist) {
                    minDist = dist;
                    ballCarrierIdx = i;
                }
            });

            // Away = attacking (blue), Home = defending (red)
            attPos.forEach((pos, i) => drawPlayer(pos, attJerseys[i], '#457B9D', 'white', i === ballCarrierIdx));
            defPos.forEach((pos, i) => drawPlayer(pos, defJerseys[i], '#E63946', 'white'));

            // Update display
            document.getElementById('obso-value').textContent = (integratedObso * 100).toFixed(2) + '%';

            frameCount++;
            const now = Date.now();
            if (now - lastFpsTime > 1000) {
                document.getElementById('fps').textContent = frameCount + ' fps';
                frameCount = 0;
                lastFpsTime = now;
            }
        }

        function findPlayer(cx, cy) {
            const [px, py] = toPitch(cx, cy);

            for (let i = 0; i < attPos.length; i++) {
                const dx = attPos[i][0] - px;
                const dy = attPos[i][1] - py;
                if (Math.sqrt(dx*dx + dy*dy) < PLAYER_RADIUS * 1.5) {
                    return {type: 'att', index: i};
                }
            }

            for (let i = 0; i < defPos.length; i++) {
                const dx = defPos[i][0] - px;
                const dy = defPos[i][1] - py;
                if (Math.sqrt(dx*dx + dy*dy) < PLAYER_RADIUS * 1.5) {
                    return {type: 'def', index: i};
                }
            }

            return null;
        }

        async function computeSurface() {
            const now = Date.now();
            if (now - lastComputeTime < 60) return;
            if (computing) return;

            computing = true;
            lastComputeTime = now;

            try {
                const response = await fetch('/compute', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        att_pos: attPos,
                        def_pos: defPos,
                        surface_type: surfaceType
                    })
                });

                if (!response.ok) {
                    console.error('Server error:', response.status);
                    computing = false;
                    return;
                }

                const data = await response.json();
                if (data.surface && data.surface.length > 0) {
                    surface = data.surface;
                    integratedObso = data.integrated_obso;
                    drawPitch();
                }
            } catch (e) {
                console.error('Compute error:', e);
            }
            computing = false;
        }

        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            dragging = findPlayer(e.clientX - rect.left, e.clientY - rect.top);
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!dragging) return;

            const rect = canvas.getBoundingClientRect();
            const [px, py] = toPitch(e.clientX - rect.left, e.clientY - rect.top);

            if (dragging.type === 'att') {
                attPos[dragging.index] = [px, py];
            } else {
                defPos[dragging.index] = [px, py];
            }

            drawPitch();
            computeSurface();
        });

        canvas.addEventListener('mouseup', () => {
            if (dragging) computeSurface();
            dragging = null;
        });

        canvas.addEventListener('mouseleave', () => { dragging = null; });

        document.querySelectorAll('input[name="surface"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                surfaceType = e.target.value;
                computeSurface();
            });
        });

        document.getElementById('reset-btn').addEventListener('click', () => {
            attPos = JSON.parse(JSON.stringify(initialAttPos));
            defPos = JSON.parse(JSON.stringify(initialDefPos));
            computeSurface();
        });

        drawPitch();
    </script>
</body>
</html>
"""


def initialize():
    print("Loading game data...")
    match = load_match(game_id=1, auto_download=True)

    # 51st minute away shot - frame 75262
    frame_num = 75262
    print(f"Using frame {frame_num} (51st minute away shot)")

    frame_data = get_frame_data(match.tracking_home, match.tracking_away, frame_num)

    params = default_model_params()
    grid = create_pitch_grid(params)
    epv_grid, epv_x, epv_y = load_epv_grid()

    # Away team attacking (direction -1)
    epv = create_epv_for_grid(grid, epv_grid, epv_x, epv_y, attacking_direction=-1)

    # Away = attacking, Home = defending
    att_data = frame_data['away']
    def_data = frame_data['home']
    ball_pos = frame_data['ball']

    att_valid = ~np.isnan(att_data['positions']).any(axis=1)
    def_valid = ~np.isnan(def_data['positions']).any(axis=1)

    STATE['params'] = params
    STATE['grid'] = grid
    STATE['epv'] = epv
    STATE['att_vel'] = att_data['velocities'][att_valid].copy()
    STATE['def_vel'] = def_data['velocities'][def_valid].copy()
    STATE['ball'] = ball_pos.copy()
    STATE['att_pos'] = att_data['positions'][att_valid].tolist()
    STATE['def_pos'] = def_data['positions'][def_valid].tolist()
    STATE['att_jerseys'] = [int(j) for j, v in zip(att_data['jerseys'], att_valid) if v]
    STATE['def_jerseys'] = [int(j) for j, v in zip(def_data['jerseys'], def_valid) if v]

    # Away attacking left, so attack_direction = -1
    STATE['attack_direction'] = -1

    # Find defending (home) goalkeeper index - jersey #11
    home_gk_jersey = match.metadata['home_gk']
    def_jerseys_list = [j for j, v in zip(def_data['jerseys'], def_valid) if v]
    STATE['defending_gk_idx'] = def_jerseys_list.index(home_gk_jersey) if home_gk_jersey in def_jerseys_list else None

    print(f"Attack direction: {STATE['attack_direction']} (away attacking left)")
    print(f"Defending GK index: {STATE['defending_gk_idx']} (jersey #{home_gk_jersey})")

    # Warm up JIT
    print("Warming up JIT...")
    pc, obso, transition, integrated = compute_all(STATE['att_pos'], STATE['def_pos'])

    STATE['initial_surface'] = obso.tolist()
    STATE['initial_obso'] = integrated
    print(f"Initial integrated OBSO: {integrated*100:.2f}%")


def compute_all(att_pos, def_pos):
    """Compute pitch control, OBSO, transition, and integrated OBSO."""
    pc, _, _ = compute_pitch_control(
        np.array(att_pos), STATE['att_vel'],
        np.array(def_pos), STATE['def_vel'],
        STATE['ball'], STATE['params'], grid=STATE['grid'],
        defending_gk_idx=STATE['defending_gk_idx'],
        attack_direction=STATE['attack_direction'],
    )

    obso = compute_obso(pc, STATE['epv'])

    # Transition probability (Spearman Eq. 6)
    grid = STATE['grid']
    ball = STATE['ball']
    dist_sq = (grid[..., 0] - ball[0])**2 + (grid[..., 1] - ball[1])**2
    gaussian = np.exp(-dist_sq / (2 * SIGMA**2))
    transition = gaussian * (pc ** ALPHA)
    transition_sum = transition.sum()
    if transition_sum > 0:
        transition = transition / transition_sum

    # Integrated OBSO = ∫ T(r) × OBSO(r) dr
    integrated = float((transition * obso).sum())

    return pc, obso, transition, integrated


@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        att_pos=STATE['att_pos'],
        def_pos=STATE['def_pos'],
        att_jerseys=STATE['att_jerseys'],
        def_jerseys=STATE['def_jerseys'],
        ball_pos=STATE['ball'].tolist(),
        initial_surface=STATE['initial_surface'],
        initial_obso=STATE['initial_obso'],
    )


@app.route('/compute', methods=['POST'])
def compute():
    data = request.json
    att_pos = data['att_pos']
    def_pos = data['def_pos']
    surface_type = data.get('surface_type', 'obso')

    pc, obso, transition, integrated = compute_all(att_pos, def_pos)

    # Select which surface to return
    if surface_type == 'obso':
        surf = obso
    elif surface_type == 'transition':
        surf = transition
    elif surface_type == 'pc':
        surf = pc
    elif surface_type == 'scoring':
        surf = STATE['epv']
    else:
        surf = obso

    return jsonify({
        'surface': surf.tolist(),
        'integrated_obso': integrated,
    })


if __name__ == '__main__':
    initialize()
    print("\n" + "=" * 50)
    print("Starting at http://localhost:8052")
    print("51st minute away shot - drag players to explore!")
    print("=" * 50 + "\n")
    app.run(debug=False, port=8052, threaded=True)
