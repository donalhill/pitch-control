"""
Vectorized Pitch Control Model

Implementation of Spearman's Potential Pitch Control Field (PPCF) using
NumPy broadcasting and Numba JIT compilation for performance.

Parameters from Spearman (2018) Table 1 MAP estimates:
- λ = 3.99 Hz (control rate)
- κ = 1.72 (defensive advantage: defenders 72% faster to control)
- s = 0.54s (reaction time)

Features:
- Ball time-of-flight handling
- Velocity-aware time-to-intercept
- Offside checking
- Goalkeeper advantage (lambda_gk = lambda_def * κ * 3)

References:
- Spearman (2018) "Beyond Expected Goals" - MIT Sloan Sports Analytics
- Spearman et al. (2017) "Physics-Based Modeling of Pass Probabilities"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numba
import numpy as np


@dataclass
class PitchControlParams:
    """
    Parameters for the pitch control model.

    Defaults from Spearman (2018) Table 1 MAP estimates.
    """

    # Player motion parameters
    max_speed: float = 5.0  # Maximum sustainable speed (m/s)
    reaction_time: float = 0.54  # Time before player starts moving (s) - Table 1: s=0.54

    # Control rate parameters (Poisson process)
    # From Spearman Table 1: λ = 3.99 Hz, κ = 1.72
    lambda_att: float = 3.99  # Attacking player control rate (Hz) - Table 1: λ=3.99
    lambda_def: float = 3.99  # Base defending player control rate (Hz)
    kappa_def: float = 1.72  # Defensive advantage multiplier - Table 1: κ=1.72
    lambda_gk_factor: float = 3.0  # GK multiplier: lambda_gk = lambda_def * kappa_def * factor

    # Uncertainty parameter for time-to-intercept (same as reaction time per Spearman)
    tti_sigma: float = 0.54  # Table 1: s=0.54 used in logistic CDF f_j(t,r,T|s)

    # Ball parameters
    ball_speed: float = 15.0  # Average ball speed for passes (m/s)

    # Integration parameters
    time_step: float = 0.04  # Integration time step (s) - matches 25Hz tracking
    max_int_time: float = 10.0  # Maximum integration time after ball arrives (s)
    convergence_tol: float = 0.01  # Stop when (1 - total_prob) < tol

    # Grid parameters (0.5m resolution for smooth contours)
    n_grid_x: int = 210  # Number of grid cells in x direction (0.5m resolution)
    n_grid_y: int = 136  # Number of grid cells in y direction (0.5m resolution)

    # Field dimensions
    field_length: float = 105.0  # Pitch length (m)
    field_width: float = 68.0  # Pitch width (m)

    @property
    def lambda_def_effective(self) -> float:
        """Effective defending control rate = lambda_def * kappa_def."""
        return self.lambda_def * self.kappa_def

    @property
    def lambda_gk(self) -> float:
        """Goalkeeper control rate = lambda_def * kappa_def * gk_factor."""
        return self.lambda_def_effective * self.lambda_gk_factor

    def time_to_control(self, is_attacking: bool) -> float:
        """
        Time for a player to gain control once at ball location.

        Based on Laurie's formula: log(10) * sigma / lambda
        """
        lam = self.lambda_att if is_attacking else self.lambda_def_effective
        return np.log(10) * self.tti_sigma / lam


def default_model_params() -> PitchControlParams:
    """Return default model parameters from Spearman (2018) Table 1."""
    return PitchControlParams()


def filter_nan_players(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out players with NaN positions and fix NaN velocities.

    Args:
        positions: (n_players, 2) player positions
        velocities: (n_players, 2) player velocities

    Returns:
        Tuple of:
        - filtered_positions: positions without NaN
        - filtered_velocities: velocities with NaN replaced by zeros
        - valid_mask: boolean mask of which players were kept
    """
    # Find players with valid positions (no NaN in either x or y)
    valid_mask = ~np.isnan(positions).any(axis=1)

    filtered_positions = positions[valid_mask]
    filtered_velocities = velocities[valid_mask].copy()

    # Replace NaN velocities with zeros (stationary assumption)
    nan_vel_mask = np.isnan(filtered_velocities)
    filtered_velocities[nan_vel_mask] = 0.0

    return filtered_positions, filtered_velocities, valid_mask


def create_pitch_grid(params: PitchControlParams) -> np.ndarray:
    """
    Create a grid of target positions covering the pitch.

    Returns:
        Array of shape (n_grid_y, n_grid_x, 2) containing (x, y) coordinates
    """
    x = np.linspace(
        -params.field_length / 2,
        params.field_length / 2,
        params.n_grid_x,
    )
    y = np.linspace(
        -params.field_width / 2,
        params.field_width / 2,
        params.n_grid_y,
    )

    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1)


def compute_ball_travel_time(
    ball_position: np.ndarray,  # (2,)
    target_positions: np.ndarray,  # (n_grid_y, n_grid_x, 2)
    ball_speed: float,
) -> np.ndarray:
    """
    Compute time for ball to travel from current position to each target.

    Simple model: constant ball speed (no drag).

    Returns:
        Array of shape (n_grid_y, n_grid_x) with travel times in seconds
    """
    if np.isnan(ball_position).any():
        # No ball position - assume instantaneous
        return np.zeros(target_positions.shape[:2])

    # Distance from ball to each grid point
    diff = target_positions - ball_position
    distances = np.sqrt(np.sum(diff**2, axis=-1))

    return distances / ball_speed


def compute_ball_travel_time_with_trajectory(
    ball_position: np.ndarray,  # (2,)
    target_positions: np.ndarray,  # (n_grid_y, n_grid_x, 2)
    attacker_arrival_times: np.ndarray,  # (n_grid_y, n_grid_x) - min attacker time
    ball_flight_model=None,
) -> np.ndarray:
    """
    Compute ball travel time using realistic trajectory model.

    Following Spearman's approach: select the ball trajectory where flight time
    most closely matches the nearest attacker's arrival time.

    Args:
        ball_position: Current ball position (2,)
        target_positions: Grid of target positions (n_grid_y, n_grid_x, 2)
        attacker_arrival_times: Fastest attacker arrival at each grid point
        ball_flight_model: BallFlightModel instance (lazy loaded if None)

    Returns:
        Array of shape (n_grid_y, n_grid_x) with travel times in seconds
    """
    if np.isnan(ball_position).any():
        return np.zeros(target_positions.shape[:2])

    # Lazy load the ball flight model
    if ball_flight_model is None:
        from pitch_control.models.ball_trajectory import get_ball_flight_model
        ball_flight_model = get_ball_flight_model()

    # Distance from ball to each grid point
    diff = target_positions - ball_position
    distances = np.sqrt(np.sum(diff**2, axis=-1))

    # Get flight time range for each distance and match to attacker arrival
    n_grid_y, n_grid_x = distances.shape
    ball_times = np.zeros_like(distances)

    for iy in range(n_grid_y):
        for ix in range(n_grid_x):
            dist = distances[iy, ix]
            att_time = attacker_arrival_times[iy, ix]
            ball_times[iy, ix] = ball_flight_model.get_matched_flight_time(dist, att_time)

    return ball_times


def compute_time_to_intercept(
    player_positions: np.ndarray,  # (n_players, 2)
    player_velocities: np.ndarray,  # (n_players, 2)
    target_positions: np.ndarray,  # (n_grid_y, n_grid_x, 2)
    reaction_time: float,
    max_speed: float,
) -> np.ndarray:
    """
    Compute time for each player to reach each target position.

    Uses Laurie's model:
    1. During reaction_time, player continues with current velocity
    2. After reaction_time, player moves at max_speed toward target

    Returns:
        Array of shape (n_players, n_grid_y, n_grid_x)
    """
    n_players = player_positions.shape[0]

    # Reshape for broadcasting:
    # positions: (n_players, 1, 1, 2)
    # velocities: (n_players, 1, 1, 2)
    # targets: (1, n_grid_y, n_grid_x, 2)
    pos = player_positions[:, np.newaxis, np.newaxis, :]
    vel = player_velocities[:, np.newaxis, np.newaxis, :]
    targets = target_positions[np.newaxis, :, :, :]

    # Position after reaction time (player continues with current velocity)
    pos_after_reaction = pos + vel * reaction_time

    # Distance to target from post-reaction position
    diff = targets - pos_after_reaction
    dist_to_target = np.sqrt(np.sum(diff**2, axis=-1))

    # Time to cover remaining distance at max speed
    time_to_target = dist_to_target / max_speed

    # Total time = reaction time + travel time
    return reaction_time + time_to_target


def check_offside(
    attacking_positions: np.ndarray,  # (n_att, 2)
    defending_positions: np.ndarray,  # (n_def, 2)
    ball_position: np.ndarray,  # (2,)
    defending_gk_idx: int | None = None,
    attack_direction: int = 1,  # +1 = attacking right, -1 = attacking left
) -> np.ndarray:
    """
    Check which attacking players are in offside position.

    Args:
        attacking_positions: Positions of attacking players
        defending_positions: Positions of defending players
        ball_position: Current ball position
        defending_gk_idx: Index of defending goalkeeper
        attack_direction: +1 if attacking toward positive x, -1 if negative x

    Returns:
        Boolean array (n_att,) - True if player is onside (can participate)
    """
    n_att = attacking_positions.shape[0]

    if np.isnan(ball_position).any():
        return np.ones(n_att, dtype=bool)

    # Get x-coordinates (adjusted for attack direction)
    att_x = attacking_positions[:, 0] * attack_direction
    def_x = defending_positions[:, 0] * attack_direction
    ball_x = ball_position[0] * attack_direction

    # Find second-last defender (excluding GK who is typically last)
    if defending_gk_idx is not None and len(def_x) > 1:
        def_x_no_gk = np.delete(def_x, defending_gk_idx)
        if len(def_x_no_gk) > 0:
            second_last_def_x = np.sort(def_x_no_gk)[-1]  # Furthest forward non-GK
        else:
            second_last_def_x = np.sort(def_x)[-2] if len(def_x) > 1 else def_x[0]
    else:
        # Sort defenders and take second-last
        sorted_def_x = np.sort(def_x)
        second_last_def_x = sorted_def_x[-2] if len(sorted_def_x) > 1 else sorted_def_x[0]

    # Offside line is max of: second-last defender, ball position, halfway line
    # (with small tolerance for tracking noise)
    offside_line = max(second_last_def_x, ball_x, 0) - 0.2

    # Player is onside if they're behind the offside line
    is_onside = att_x <= offside_line

    return is_onside


@numba.njit(cache=True)
def _logistic_cdf(T: float, tau: float, sigma: float) -> float:
    """
    Logistic CDF for probability player can intercept by time T.

    Uses the form from Spearman: 1 / (1 + exp(-π/√3 × (T - tau) / sigma))
    """
    scale = np.pi / np.sqrt(3.0) / sigma
    return 1.0 / (1.0 + np.exp(-scale * (T - tau)))


@numba.njit(cache=True)
def _get_time_to_control_advantage(
    time_to_intercept: np.ndarray,  # (n_players,) for a single grid point
    is_attacking: np.ndarray,
    is_onside: np.ndarray,
    time_to_control_att: float,
    time_to_control_def: float,
) -> Tuple[float, float]:
    """
    Check if one team has a clear time advantage at this grid point.

    Returns (min_att_time, min_def_time) - the fastest player from each team.
    """
    min_att = 1e10
    min_def = 1e10

    for p in range(len(time_to_intercept)):
        tti = time_to_intercept[p]
        if is_attacking[p]:
            if is_onside[p]:
                total_time = tti + time_to_control_att
                if total_time < min_att:
                    min_att = total_time
        else:
            total_time = tti + time_to_control_def
            if total_time < min_def:
                min_def = total_time

    return min_att, min_def


@numba.njit(parallel=True, cache=True)
def _integrate_pitch_control(
    time_to_intercept: np.ndarray,  # (n_players, n_grid_y, n_grid_x)
    ball_travel_time: np.ndarray,  # (n_grid_y, n_grid_x)
    is_attacking: np.ndarray,  # (n_players,) bool
    is_onside: np.ndarray,  # (n_players,) bool - only matters for attackers
    is_goalkeeper: np.ndarray,  # (n_players,) bool
    lambda_att: float,
    lambda_def: float,
    lambda_gk: float,
    sigma: float,
    dt: float,
    max_int_time: float,
    convergence_tol: float,
    time_to_control_att: float,
    time_to_control_def: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated pitch control integration.

    For each grid point, integrate the differential equation:
    dPPCF_i/dT = (1 - Σ_k PPCF_k) × f_i(T) × λ_i

    Integration starts at ball_travel_time (when ball arrives at target).

    Returns:
        Tuple of:
        - team_control: (n_grid_y, n_grid_x) - attacking team's total control
        - player_control: (n_players, n_grid_y, n_grid_x) - per-player control
    """
    n_players, n_grid_y, n_grid_x = time_to_intercept.shape

    # Output arrays
    team_control = np.zeros((n_grid_y, n_grid_x))
    player_control = np.zeros((n_players, n_grid_y, n_grid_x))

    # Parallel over grid rows
    for iy in numba.prange(n_grid_y):
        for ix in range(n_grid_x):
            # Ball travel time to this grid point
            t_ball = ball_travel_time[iy, ix]

            # Per-player control probability at this grid point
            ppcf = np.zeros(n_players)
            total_ppcf = 0.0

            # Start integration when ball arrives (per Spearman - ball time-of-flight is the gate)
            t_start = max(0.0, t_ball - dt)
            t_end = t_ball + max_int_time

            # Time integration
            T = t_start
            while T < t_end:
                # Check for convergence
                if (1.0 - total_ppcf) < convergence_tol:
                    break

                remaining = 1.0 - total_ppcf

                for p in range(n_players):
                    # Skip offside attacking players
                    if is_attacking[p] and not is_onside[p]:
                        continue

                    # Get player's time to intercept
                    tau = time_to_intercept[p, iy, ix]

                    # Probability player can intercept by time T
                    f = _logistic_cdf(T, tau, sigma)

                    # Get control rate for this player
                    if is_goalkeeper[p]:
                        lam = lambda_gk
                    elif is_attacking[p]:
                        lam = lambda_att
                    else:
                        lam = lambda_def

                    # dPPCF_p/dT = (1 - Σ PPCF) × f_p × λ_p
                    d_ppcf = remaining * f * lam * dt
                    ppcf[p] += d_ppcf
                    total_ppcf += d_ppcf

                T += dt

            # Store results
            att_control = 0.0
            for p in range(n_players):
                player_control[p, iy, ix] = ppcf[p]
                if is_attacking[p] and is_onside[p]:
                    att_control += ppcf[p]

            team_control[iy, ix] = att_control

    return team_control, player_control


def compute_pitch_control(
    attacking_positions: np.ndarray,  # (n_att, 2)
    attacking_velocities: np.ndarray,  # (n_att, 2)
    defending_positions: np.ndarray,  # (n_def, 2)
    defending_velocities: np.ndarray,  # (n_def, 2)
    ball_position: np.ndarray,  # (2,)
    params: PitchControlParams | None = None,
    attacking_gk_idx: int | None = None,
    defending_gk_idx: int | None = None,
    grid: np.ndarray | None = None,
    check_offsides: bool = True,
    attack_direction: int = 1,
    use_ball_trajectory: bool = False,
    ball_flight_model=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pitch control surface for the attacking team.

    Args:
        attacking_positions: Positions of attacking players (n_att, 2)
        attacking_velocities: Velocities of attacking players (n_att, 2)
        defending_positions: Positions of defending players (n_def, 2)
        defending_velocities: Velocities of defending players (n_def, 2)
        ball_position: Current ball position (2,)
        params: Model parameters (uses defaults if None)
        attacking_gk_idx: Index of attacking team's goalkeeper (or None)
        defending_gk_idx: Index of defending team's goalkeeper (or None)
        grid: Pre-computed grid (or None to create one)
        check_offsides: Whether to exclude offside players
        attack_direction: +1 if attacking right, -1 if attacking left
        use_ball_trajectory: If True, use realistic ball trajectory model with
            aerodynamic drag (Asai & Seo 2013) and match flight time to attacker
            arrival (Spearman 2018). If False, use simple constant speed model.
        ball_flight_model: Pre-loaded BallFlightModel (lazy loads if None)

    Returns:
        Tuple of:
        - pitch_control: (n_grid_y, n_grid_x) - attacking team control probability
        - grid: (n_grid_y, n_grid_x, 2) - grid coordinates used
        - player_control: (n_players, n_grid_y, n_grid_x) - per-player control
    """
    if params is None:
        params = default_model_params()

    if grid is None:
        grid = create_pitch_grid(params)

    # Filter out players with NaN positions, fix NaN velocities
    att_pos, att_vel, att_valid = filter_nan_players(attacking_positions, attacking_velocities)
    def_pos, def_vel, def_valid = filter_nan_players(defending_positions, defending_velocities)

    n_att = att_pos.shape[0]
    n_def = def_pos.shape[0]
    n_players = n_att + n_def

    # Handle edge case: no valid players
    if n_players == 0:
        empty_control = np.zeros((grid.shape[0], grid.shape[1]))
        empty_player = np.zeros((0, grid.shape[0], grid.shape[1]))
        return empty_control, grid, empty_player

    # Combine all valid players
    all_positions = np.vstack([att_pos, def_pos])
    all_velocities = np.vstack([att_vel, def_vel])

    # Create indicator arrays
    is_attacking = np.zeros(n_players, dtype=np.bool_)
    is_attacking[:n_att] = True

    # Map GK indices from original to filtered arrays
    is_goalkeeper = np.zeros(n_players, dtype=np.bool_)
    if attacking_gk_idx is not None and att_valid[attacking_gk_idx]:
        # Find new index after filtering
        new_att_gk_idx = np.sum(att_valid[:attacking_gk_idx])
        is_goalkeeper[new_att_gk_idx] = True

    filtered_def_gk_idx = None
    if defending_gk_idx is not None and def_valid[defending_gk_idx]:
        # Find new index after filtering
        filtered_def_gk_idx = np.sum(def_valid[:defending_gk_idx])
        is_goalkeeper[n_att + filtered_def_gk_idx] = True

    # Check offsides (use filtered positions)
    if check_offsides:
        att_onside = check_offside(
            att_pos,
            def_pos,
            ball_position,
            filtered_def_gk_idx,
            attack_direction,
        )
    else:
        att_onside = np.ones(n_att, dtype=bool)

    # All defenders are "onside" by definition
    is_onside = np.concatenate([att_onside, np.ones(n_def, dtype=bool)])

    # Compute time to intercept for all players at all grid points
    time_to_intercept = compute_time_to_intercept(
        all_positions,
        all_velocities,
        grid,
        params.reaction_time,
        params.max_speed,
    )

    # Compute ball travel time to each grid point
    if use_ball_trajectory:
        # Find minimum attacker arrival time at each grid point (for matching)
        att_tti = time_to_intercept[:n_att][att_onside]  # Only onside attackers
        if len(att_tti) > 0:
            min_att_arrival = np.min(att_tti, axis=0)
        else:
            min_att_arrival = np.full(grid.shape[:2], np.inf)

        ball_travel_time = compute_ball_travel_time_with_trajectory(
            ball_position, grid, min_att_arrival, ball_flight_model
        )
    else:
        ball_travel_time = compute_ball_travel_time(ball_position, grid, params.ball_speed)

    # Run the integration
    team_control, filtered_player_control = _integrate_pitch_control(
        time_to_intercept,
        ball_travel_time,
        is_attacking,
        is_onside,
        is_goalkeeper,
        params.lambda_att,
        params.lambda_def_effective,
        params.lambda_gk,
        params.tti_sigma,
        params.time_step,
        params.max_int_time,
        params.convergence_tol,
        params.time_to_control(is_attacking=True),
        params.time_to_control(is_attacking=False),
    )

    # Reconstruct full player_control array with zeros for filtered players
    n_att_orig = attacking_positions.shape[0]
    n_def_orig = defending_positions.shape[0]
    n_players_orig = n_att_orig + n_def_orig

    player_control = np.zeros((n_players_orig, grid.shape[0], grid.shape[1]))

    # Map filtered results back to original indices
    att_indices = np.where(att_valid)[0]
    def_indices = np.where(def_valid)[0]

    for i, orig_idx in enumerate(att_indices):
        player_control[orig_idx] = filtered_player_control[i]

    for i, orig_idx in enumerate(def_indices):
        player_control[n_att_orig + orig_idx] = filtered_player_control[n_att + i]

    return team_control, grid, player_control


def compute_player_pitch_control(
    attacking_positions: np.ndarray,
    attacking_velocities: np.ndarray,
    defending_positions: np.ndarray,
    defending_velocities: np.ndarray,
    ball_position: np.ndarray,
    params: PitchControlParams | None = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-player pitch control contribution.

    Returns:
        Tuple of:
        - attacking_player_control: (n_att, n_grid_y, n_grid_x)
        - defending_player_control: (n_def, n_grid_y, n_grid_x)
        - grid: (n_grid_y, n_grid_x, 2)
    """
    team_control, grid, player_control = compute_pitch_control(
        attacking_positions,
        attacking_velocities,
        defending_positions,
        defending_velocities,
        ball_position,
        params,
        **kwargs,
    )

    n_att = attacking_positions.shape[0]
    return player_control[:n_att], player_control[n_att:], grid


def compute_pitch_control_at_frame(
    frame_data: dict,
    attacking_team: str = "home",
    params: PitchControlParams | None = None,
    check_offsides: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Convenience function to compute pitch control from frame data dict.

    Args:
        frame_data: Output from metrica.get_frame_data()
        attacking_team: "home" or "away"
        params: Model parameters

    Returns:
        Tuple of:
        - pitch_control: (n_grid_y, n_grid_x)
        - grid: (n_grid_y, n_grid_x, 2)
        - player_control_dict: Dict mapping jersey -> control surface
    """
    if attacking_team == "home":
        att_data = frame_data["home"]
        def_data = frame_data["away"]
        attack_direction = 1  # Home attacks right (positive x)
    else:
        att_data = frame_data["away"]
        def_data = frame_data["home"]
        attack_direction = -1  # Away attacks left (negative x)

    team_control, grid, player_control = compute_pitch_control(
        att_data["positions"],
        att_data["velocities"],
        def_data["positions"],
        def_data["velocities"],
        frame_data["ball"],
        params,
        check_offsides=check_offsides,
        attack_direction=attack_direction,
    )

    # Build player control dict
    n_att = len(att_data["jerseys"])
    player_control_dict = {}

    for i, jersey in enumerate(att_data["jerseys"]):
        player_control_dict[f"{attacking_team}_{jersey}"] = player_control[i]

    for i, jersey in enumerate(def_data["jerseys"]):
        defending_team = "away" if attacking_team == "home" else "home"
        player_control_dict[f"{defending_team}_{jersey}"] = player_control[n_att + i]

    return team_control, grid, player_control_dict
