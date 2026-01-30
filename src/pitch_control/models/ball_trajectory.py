"""
Ball trajectory simulation with aerodynamic drag.

Based on Asai & Seo (2013) "Aerodynamic drag of modern soccer balls"
SpringerPlus 2:171

Implements realistic ball flight physics to compute time-of-flight
for different pass distances, enabling accurate pitch control calculations.
"""

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Ball physical parameters (FIFA regulation)
BALL_MASS = 0.43  # kg
BALL_DIAMETER = 0.22  # m
BALL_RADIUS = BALL_DIAMETER / 2
BALL_AREA = np.pi * BALL_RADIUS**2  # m^2

# Air parameters (sea level, 20°C)
AIR_DENSITY = 1.2  # kg/m^3
AIR_VISCOSITY = 1.8e-5  # kg/(m·s)

# Gravity
G = 9.81  # m/s^2


def drag_coefficient(speed: float) -> float:
    """
    Compute drag coefficient Cd as function of ball speed.

    Based on wind tunnel data from Asai & Seo (2013) Figure 4.
    Uses a smooth sigmoid transition between subcritical and supercritical regimes.

    Args:
        speed: Ball speed in m/s

    Returns:
        Drag coefficient Cd (dimensionless)
    """
    # Reynolds number
    Re = AIR_DENSITY * speed * BALL_DIAMETER / AIR_VISCOSITY

    # Drag coefficient regimes (based on Tango 12 / typical modern ball)
    # Subcritical: Cd ~ 0.47
    # Supercritical: Cd ~ 0.15-0.18
    # Critical Re ~ 2.4 × 10^5 (corresponds to ~16 m/s)

    Cd_subcritical = 0.47
    Cd_supercritical = 0.15
    Re_critical = 2.4e5
    Re_width = 0.5e5  # Width of transition region

    # Smooth sigmoid transition
    x = (Re - Re_critical) / Re_width
    sigmoid = 1 / (1 + np.exp(-x))

    Cd = Cd_subcritical + (Cd_supercritical - Cd_subcritical) * sigmoid

    return Cd


@njit
def drag_coefficient_fast(speed: float) -> float:
    """Numba-compiled version of drag coefficient calculation."""
    Re = 1.2 * speed * 0.22 / 1.8e-5

    Cd_subcritical = 0.47
    Cd_supercritical = 0.15
    Re_critical = 2.4e5
    Re_width = 0.5e5

    x = (Re - Re_critical) / Re_width
    sigmoid = 1.0 / (1.0 + np.exp(-x))

    return Cd_subcritical + (Cd_supercritical - Cd_subcritical) * sigmoid


def trajectory_derivatives(t, state):
    """
    Compute derivatives for 2D trajectory with drag.

    State: [x, y, vx, vy]
    """
    x, y, vx, vy = state

    # Speed magnitude
    v = np.sqrt(vx**2 + vy**2)

    if v < 0.01:  # Avoid division by zero
        return [vx, vy, 0, -G]

    # Drag coefficient
    Cd = drag_coefficient(v)

    # Drag force magnitude: F = 0.5 * rho * Cd * A * v^2
    # Drag acceleration: a = F/m = 0.5 * rho * Cd * A * v^2 / m
    drag_accel = 0.5 * AIR_DENSITY * Cd * BALL_AREA * v**2 / BALL_MASS

    # Drag acts opposite to velocity
    ax = -drag_accel * vx / v
    ay = -drag_accel * vy / v - G

    return [vx, vy, ax, ay]


def simulate_trajectory(
    initial_speed: float,
    launch_angle: float,
    max_time: float = 10.0,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate ball trajectory with drag.

    Args:
        initial_speed: Launch speed in m/s
        launch_angle: Launch angle in degrees (0 = horizontal, 90 = vertical)
        max_time: Maximum simulation time in seconds
        dt: Time step for output

    Returns:
        t, x, y, speed arrays
    """
    # Initial conditions
    angle_rad = np.radians(launch_angle)
    vx0 = initial_speed * np.cos(angle_rad)
    vy0 = initial_speed * np.sin(angle_rad)
    state0 = [0, 0, vx0, vy0]

    # Solve ODE
    t_eval = np.arange(0, max_time, dt)

    # Event: ball hits ground (y = 0 after launch)
    def ground_event(t, state):
        return state[1] if t > 0.01 else 1.0
    ground_event.terminal = True
    ground_event.direction = -1

    sol = solve_ivp(
        trajectory_derivatives,
        [0, max_time],
        state0,
        t_eval=t_eval,
        events=ground_event,
        method='RK45',
        max_step=0.05,
    )

    x = sol.y[0]
    y = sol.y[1]
    vx = sol.y[2]
    vy = sol.y[3]
    speed = np.sqrt(vx**2 + vy**2)

    return sol.t, x, y, speed


def flight_time_to_distance(
    target_distance: float,
    initial_speed: float,
    launch_angle: float,
) -> float | None:
    """
    Compute flight time to reach a target horizontal distance.

    Args:
        target_distance: Horizontal distance in meters
        initial_speed: Launch speed in m/s
        launch_angle: Launch angle in degrees

    Returns:
        Flight time in seconds, or None if target not reachable
    """
    t, x, y, _ = simulate_trajectory(initial_speed, launch_angle)

    # Check if we reach the target distance before landing
    if x[-1] < target_distance:
        return None

    # Find where x crosses target_distance
    idx = np.searchsorted(x, target_distance)
    if idx == 0:
        return 0.0

    # Linear interpolation
    x0, x1 = x[idx-1], x[idx]
    t0, t1 = t[idx-1], t[idx]

    frac = (target_distance - x0) / (x1 - x0)
    flight_time = t0 + frac * (t1 - t0)

    # Check ball is still above ground at this point
    y0, y1 = y[idx-1], y[idx]
    height = y0 + frac * (y1 - y0)
    if height < 0:
        return None

    return flight_time


def compute_flight_time_range(
    target_distance: float,
    speed_range: tuple[float, float] = (5, 35),
    angle_range: tuple[float, float] = (5, 70),
    n_speeds: int = 20,
    n_angles: int = 30,
) -> tuple[float, float] | None:
    """
    Compute min and max flight times to reach a target distance.

    Explores different launch angles and speeds to find the fastest
    and slowest valid trajectories.

    Args:
        target_distance: Horizontal distance in meters
        speed_range: (min_speed, max_speed) in m/s
        angle_range: (min_angle, max_angle) in degrees
        n_speeds: Number of speeds to test
        n_angles: Number of angles to test

    Returns:
        (min_time, max_time) tuple, or None if unreachable
    """
    speeds = np.linspace(speed_range[0], speed_range[1], n_speeds)
    angles = np.linspace(angle_range[0], angle_range[1], n_angles)

    valid_times = []

    for speed in speeds:
        for angle in angles:
            t = flight_time_to_distance(target_distance, speed, angle)
            if t is not None and t > 0:
                valid_times.append(t)

    if not valid_times:
        return None

    return min(valid_times), max(valid_times)


def build_flight_time_lookup(
    max_distance: float = 80,
    resolution: float = 1.0,
    speed_range: tuple[float, float] = (5, 35),
    angle_range: tuple[float, float] = (5, 70),
) -> dict:
    """
    Build a lookup table of flight time ranges for different distances.

    Args:
        max_distance: Maximum distance to compute (meters)
        resolution: Distance step (meters)
        speed_range: (min_speed, max_speed) for trajectory search
        angle_range: (min_angle, max_angle) for trajectory search

    Returns:
        Dict with 'distances', 'min_times', 'max_times' arrays
    """
    distances = np.arange(0, max_distance + resolution, resolution)
    min_times = np.zeros_like(distances)
    max_times = np.zeros_like(distances)

    for i, d in enumerate(distances):
        if d == 0:
            min_times[i] = 0
            max_times[i] = 0
            continue

        result = compute_flight_time_range(d, speed_range, angle_range)
        if result is not None:
            min_times[i], max_times[i] = result
        else:
            # Unreachable - use extrapolation from last valid
            if i > 0:
                min_times[i] = min_times[i-1] * 1.2
                max_times[i] = max_times[i-1] * 1.2

    return {
        'distances': distances,
        'min_times': min_times,
        'max_times': max_times,
    }


class BallFlightModel:
    """
    Cached ball flight model for efficient lookup.

    Pre-computes flight time ranges and provides fast interpolation.
    Saves/loads from disk cache to avoid rebuilding.
    """

    def __init__(
        self,
        max_distance: float = 80,
        resolution: float = 0.5,
        cache_path: str | None = None,
    ):
        """
        Initialize the ball flight model.

        Args:
            max_distance: Maximum distance to pre-compute (meters)
            resolution: Distance resolution (meters)
            cache_path: Path to cache file (auto-generated if None)
        """
        from pathlib import Path

        # Default cache location
        if cache_path is None:
            cache_dir = Path(__file__).parent.parent.parent.parent / "data"
            cache_dir.mkdir(exist_ok=True)
            cache_path = cache_dir / "ball_flight_lookup.npz"
        else:
            cache_path = Path(cache_path)

        # Try to load from cache
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                if (data['max_distance'] == max_distance and
                    data['resolution'] == resolution):
                    self.distances = data['distances']
                    self.min_times = data['min_times']
                    self.max_times = data['max_times']
                    print(f"Loaded ball flight model from cache ({len(self.distances)} points)")
                    return
            except Exception:
                pass  # Cache invalid, rebuild

        # Build lookup table
        print("Building ball flight lookup table (this takes ~30s first time)...")
        lookup = build_flight_time_lookup(max_distance, resolution)

        self.distances = lookup['distances']
        self.min_times = lookup['min_times']
        self.max_times = lookup['max_times']
        print(f"  Computed {len(self.distances)} distance points")

        # Save to cache
        try:
            np.savez(
                cache_path,
                distances=self.distances,
                min_times=self.min_times,
                max_times=self.max_times,
                max_distance=max_distance,
                resolution=resolution,
            )
            print(f"  Saved to cache: {cache_path}")
        except Exception as e:
            print(f"  Warning: Could not save cache: {e}")

    def get_flight_time_range(self, distance: float) -> tuple[float, float]:
        """
        Get min/max flight time for a given distance.

        Args:
            distance: Horizontal distance in meters

        Returns:
            (min_time, max_time) tuple
        """
        # Interpolate
        min_t = np.interp(distance, self.distances, self.min_times)
        max_t = np.interp(distance, self.distances, self.max_times)
        return min_t, max_t

    def get_matched_flight_time(
        self,
        distance: float,
        attacker_arrival_time: float,
    ) -> float:
        """
        Get ball flight time matched to attacker arrival.

        Following Spearman's approach: select the trajectory where
        ball arrival closely matches the nearest attacker's arrival.
        This "advantages successful passing."

        Args:
            distance: Distance to target in meters
            attacker_arrival_time: Time for attacker to reach target

        Returns:
            Ball flight time (clamped to valid range)
        """
        min_t, max_t = self.get_flight_time_range(distance)

        # Clamp to valid range
        return np.clip(attacker_arrival_time, min_t, max_t)

    def get_simple_flight_time(self, distance: float) -> float:
        """
        Get a simple flight time estimate (average of min/max).

        Useful for quick calculations without attacker timing.
        """
        min_t, max_t = self.get_flight_time_range(distance)
        return (min_t + max_t) / 2


# Module-level cached instance
_cached_model = None


def get_ball_flight_model() -> BallFlightModel:
    """Get or create the cached ball flight model."""
    global _cached_model
    if _cached_model is None:
        _cached_model = BallFlightModel()
    return _cached_model


def ball_flight_time(
    distance: float,
    attacker_arrival: float | None = None,
) -> float:
    """
    Convenience function to get ball flight time.

    Args:
        distance: Distance in meters
        attacker_arrival: Optional attacker arrival time to match

    Returns:
        Ball flight time in seconds
    """
    model = get_ball_flight_model()

    if attacker_arrival is not None:
        return model.get_matched_flight_time(distance, attacker_arrival)
    else:
        return model.get_simple_flight_time(distance)


# =============================================================================
# Parabolic (no drag) ball flight model
# =============================================================================

G = 9.81  # Gravity (m/s²)


def parabolic_flight_time_range(
    distance: float,
    v_max: float = 35.0,
    theta_min: float = 5.0,
    theta_max: float = 70.0,
) -> tuple[float, float] | None:
    """
    Compute min/max flight time for parabolic trajectory (no drag).

    Physics:
        Range: D = v₀² sin(2θ) / g
        Flight time: T = √(2D tan(θ) / g)

    Args:
        distance: Horizontal distance in meters
        v_max: Maximum ball speed in m/s
        theta_min: Minimum launch angle in degrees
        theta_max: Maximum launch angle in degrees

    Returns:
        (min_time, max_time) tuple, or None if unreachable
    """
    if distance <= 0:
        return 0.0, 0.0

    # Convert angles to radians
    theta_min_rad = np.radians(theta_min)
    theta_max_rad = np.radians(theta_max)

    # Minimum speed required to reach distance (at 45°)
    v_min_required = np.sqrt(distance * G)

    if v_min_required > v_max:
        return None  # Unreachable

    # Valid angle range: sin(2θ) ≥ D*g/v_max²
    sin_2theta_min = min(1.0, distance * G / v_max**2)

    # Angles where we can reach with v_max
    # sin(2θ) = sin_2theta_min has two solutions: θ_low and 90° - θ_low
    theta_low = 0.5 * np.arcsin(sin_2theta_min)
    theta_high = np.pi/2 - theta_low

    # Clamp to allowed angle range
    theta_low = max(theta_low, theta_min_rad)
    theta_high = min(theta_high, theta_max_rad)

    if theta_low > theta_high:
        # No valid angles in our range
        return None

    # Flight time: T = √(2D tan(θ) / g)
    # Minimum time at lowest angle (driven pass)
    # Maximum time at highest angle (lobbed pass)
    t_min = np.sqrt(2 * distance * np.tan(theta_low) / G)
    t_max = np.sqrt(2 * distance * np.tan(theta_high) / G)

    return t_min, t_max


def parabolic_matched_flight_time(
    distance: float,
    attacker_arrival: float,
    v_max: float = 35.0,
) -> float:
    """
    Get ball flight time matched to attacker arrival (parabolic model).

    Following Spearman: select trajectory where ball arrives when attacker arrives.

    Args:
        distance: Distance to target in meters
        attacker_arrival: Time for attacker to reach target
        v_max: Maximum ball speed

    Returns:
        Ball flight time in seconds
    """
    result = parabolic_flight_time_range(distance, v_max)

    if result is None:
        # Unreachable - return large time
        return attacker_arrival * 2

    t_min, t_max = result

    # Clamp to valid range (match to attacker arrival)
    return np.clip(attacker_arrival, t_min, t_max)


def compute_ball_travel_time_parabolic(
    ball_position: np.ndarray,
    target_positions: np.ndarray,
    attacker_arrival_times: np.ndarray,
    v_max: float = 35.0,
) -> np.ndarray:
    """
    Compute ball travel time using inverse kinematics approach.

    For each grid point, solves: what trajectory makes the ball arrive
    when the attacker does? If achievable, ball_time = attacker_time.
    If not (requires v > v_max), uses closest achievable time.

    Physics (parabolic trajectory, no drag):
        Given distance D and desired flight time T:
        - Required angle: θ = arctan(gT²/2D)
        - Required speed: v₀ = √[(gT/2)² + (D/T)²]

        If v₀ ≤ v_max: trajectory is achievable
        If v₀ > v_max: use closest achievable time with v_max

    Args:
        ball_position: Current ball position (2,)
        target_positions: Grid of target positions (n_grid_y, n_grid_x, 2)
        attacker_arrival_times: Fastest attacker arrival at each point
        v_max: Maximum ball speed (m/s)

    Returns:
        Ball travel times (n_grid_y, n_grid_x)
    """
    if np.isnan(ball_position).any():
        return np.zeros(target_positions.shape[:2])

    # Distances from ball to each grid point
    diff = target_positions - ball_position
    D = np.sqrt(np.sum(diff**2, axis=-1))
    T = attacker_arrival_times

    # Avoid division by zero
    D_safe = np.maximum(D, 0.01)
    T_safe = np.maximum(T, 0.01)

    # Required velocity to match attacker arrival: v₀ = √[(gT/2)² + (D/T)²]
    v_required = np.sqrt((G * T_safe / 2)**2 + (D_safe / T_safe)**2)

    # Check if achievable
    achievable = v_required <= v_max

    # For achievable trajectories, ball arrives when attacker does
    ball_times = T.copy()

    # For non-achievable, compute closest time using v_max
    # The achievable time range with v_max:
    # - Min time: lowest angle that reaches D with v_max (driven pass)
    # - Max time: highest angle that reaches D with v_max (lob)

    # First check if distance is reachable at all (v_min = √(Dg) at 45°)
    v_min_for_distance = np.sqrt(D_safe * G)
    distance_reachable = v_min_for_distance <= v_max

    # sin(2θ) = Dg/v_max² gives valid angle range
    sin_2theta = np.clip(D_safe * G / v_max**2, 0, 1)
    theta_low = 0.5 * np.arcsin(sin_2theta)  # Low angle (driven)
    theta_high = np.pi/2 - theta_low          # High angle (lob)

    # Clamp to practical angles
    theta_low = np.maximum(theta_low, np.radians(5))
    theta_high = np.minimum(theta_high, np.radians(75))

    # Flight times at boundary angles: T = √(2D tan(θ) / g)
    t_min = np.sqrt(2 * D_safe * np.tan(theta_low) / G)
    t_max = np.sqrt(2 * D_safe * np.tan(theta_high) / G)

    # For non-achievable: use closest boundary (min if T too short, max if T too long)
    too_fast = T < t_min  # Attacker arrives before fastest possible ball
    too_slow = T > t_max  # Attacker arrives after slowest possible ball

    ball_times = np.where(~achievable & too_fast, t_min, ball_times)
    ball_times = np.where(~achievable & too_slow, t_max, ball_times)

    # For unreachable distances, use large time (defenders will dominate)
    ball_times = np.where(~distance_reachable, T * 2, ball_times)

    # Zero distance = zero time
    ball_times = np.where(D < 0.01, 0, ball_times)

    return ball_times
