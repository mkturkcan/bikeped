"""
Shared decision pipeline for the bikeped collision-warning system.

This module is imported by both mqtt_bridge.py (deployed system) and
decision_testbench.py (offline analysis). Any change here applies to both.

Designed for the Jetson AGX Orin: no heavy imports, no allocations in the
hot path, numpy used only for sqrt.
"""

from collections import deque
from math import sqrt


def stopping_distance(speed_ms, reaction_s=0.84, decel_ms2=1.96):
    """Cyclist stopping distance (m).

    Default values: 85th-percentile field measurements from Martin &
    Bigazzi (2025) for conventional bicycles responding to unexpected
    hazards (n=302).

    Args:
        speed_ms: cyclist ground speed in m/s.
        reaction_s: cyclist perception-reaction time (s).
        decel_ms2: comfortable braking deceleration (m/s^2).
    """
    return speed_ms * reaction_s + speed_ms * speed_ms / (2.0 * decel_ms2)


class DecisionPipeline:
    """Three-stage decision pipeline: pedestrian presence, cyclist memory,
    closing proximity with speed-adaptive bounds.

    This class is stateful: it maintains the cyclist memory buffer and
    per-object track histories across frames. Call reset() between scenarios
    in offline simulation.
    """

    # Supported decision rules for Stage 3:
    #   'pairwise'      - (default) compare dist(bike_t, ped_t) vs dist(bike_{t-k}, ped_{t-k})
    #   'distance_only' - alert if any bike-ped pair within d_max
    #   'naive_closing'  - compare dist(bike_{t-k}, ped_t) vs dist(bike_t, ped_t)
    #                      (uses bike's past position against ped's CURRENT position)
    #   'ttc'           - alert if time-to-collision < ttc_threshold_s
    DECISION_RULES = ('pairwise', 'distance_only', 'naive_closing', 'ttc')

    def __init__(self, bike_memory_frames=77, prox_min=0.5, prox_max=20.7,
                 prox_speed_min=0.098, lookback_frames=1, max_range=25.0,
                 speed_adaptive=True, cyclist_reaction_s=0.84,
                 cyclist_decel_ms2=1.96, fps=30, decision_rule='pairwise',
                 ttc_threshold_s=3.0):
        self.bike_memory_frames = bike_memory_frames
        self.prox_min = prox_min
        self.prox_max = prox_max
        self.prox_speed_min = prox_speed_min
        self.lookback_frames = lookback_frames
        self.max_range = max_range
        self.speed_adaptive = speed_adaptive
        self.cyclist_reaction_s = cyclist_reaction_s
        self.cyclist_decel_ms2 = cyclist_decel_ms2
        self.fps = fps
        self.decision_rule = decision_rule
        self.ttc_threshold_s = ttc_threshold_s

        self.bike_memory = deque()
        self.track_positions = {}   # track_id -> deque of (x, y, t)

    def reset(self):
        self.bike_memory.clear()
        self.track_positions.clear()

    def update_track(self, track_id, x, y, t):
        """Record a position for a tracked object. Call once per detection."""
        if track_id not in self.track_positions:
            self.track_positions[track_id] = deque(maxlen=60)
        self.track_positions[track_id].append((x, y, t))

    def _estimate_speed_ms(self, track_id, window=4):
        """Estimate ground speed in m/s from recent track history."""
        hist = self.track_positions.get(track_id)
        if hist is None or len(hist) < 2:
            return 0.0
        n = min(window, len(hist) - 1)
        p0 = hist[-1 - n]
        p1 = hist[-1]
        dt = p1[2] - p0[2]
        if dt < 1e-6:
            return 0.0
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        return sqrt(dx * dx + dy * dy) / dt

    def evaluate(self, persons, bikes, bike_in_scene, sim_time):
        """Run the three-stage decision pipeline.

        Args:
            persons: list of dicts with keys: track_id, x, y.
            bikes:   list of dicts with keys: track_id, x, y.
            bike_in_scene: bool, whether any cyclist class is in the current frame.
            sim_time: current time in seconds (for memory expiry).

        Returns:
            state: one of 'idle', 'safe', 'warning', 'alert'.
            alert_pair: tuple (bike_track_id, ped_track_id) or None.
        """
        # Update bike memory
        if bike_in_scene:
            self.bike_memory.append(sim_time)
        mem_sec = self.bike_memory_frames / self.fps
        while self.bike_memory and (sim_time - self.bike_memory[0]) > mem_sec:
            self.bike_memory.popleft()

        # Stage 1: pedestrian presence
        if not persons:
            return 'idle', None

        # Stage 2: cyclist memory
        if self.bike_memory_frames > 0:
            if not self.bike_memory:
                return 'safe', None
        else:
            if not bikes:
                return 'safe', None

        # Stage 3: proximity check (rule-dependent)
        for bike in bikes:
            bx, by = bike['x'], bike['y']
            bid = bike['track_id']
            bike_hist = self.track_positions.get(bid)

            bike_range = sqrt(bx * bx + by * by)
            if bike_range > self.max_range:
                continue

            # Speed-adaptive upper bound
            d_max = self.prox_max
            if self.speed_adaptive:
                v_bike = self._estimate_speed_ms(bid)
                sd = stopping_distance(v_bike, self.cyclist_reaction_s,
                                       self.cyclist_decel_ms2)
                if sd > d_max:
                    d_max = sd

            for ped in persons:
                px, py = ped['x'], ped['y']
                pid = ped['track_id']
                ped_hist = self.track_positions.get(pid)

                ped_range = sqrt(px * px + py * py)
                if ped_range > self.max_range:
                    continue

                dx = bx - px
                dy = by - py
                d_cur = sqrt(dx * dx + dy * dy)
                if d_cur < self.prox_min or d_cur > d_max:
                    continue

                # ── distance_only: alert on proximity alone ──
                if self.decision_rule == 'distance_only':
                    return 'alert', (bid, pid)

                # Rules below require track history
                if (bike_hist is None or len(bike_hist) < 2
                        or ped_hist is None or len(ped_hist) < 2):
                    continue

                lb = min(self.lookback_frames,
                         len(bike_hist) - 1, len(ped_hist) - 1)
                prev_bike = bike_hist[-1 - lb]

                bike_disp_x = bx - prev_bike[0]
                bike_disp_y = by - prev_bike[1]
                bike_disp = sqrt(bike_disp_x * bike_disp_x
                                 + bike_disp_y * bike_disp_y)

                # ── ttc: alert if time-to-collision < threshold ──
                if self.decision_rule == 'ttc':
                    if bike_disp < 1e-6:
                        continue
                    dt_hist = sim_time - prev_bike[2]
                    if dt_hist < 1e-6:
                        continue
                    v_closing = -(d_cur - sqrt(
                        (prev_bike[0] - ped_hist[-1 - lb][0])**2 +
                        (prev_bike[1] - ped_hist[-1 - lb][1])**2
                    )) / dt_hist
                    if v_closing > 0.1:
                        ttc_est = d_cur / v_closing
                        if ttc_est < self.ttc_threshold_s:
                            return 'alert', (bid, pid)
                    continue

                # ── naive_closing: bike's past vs ped's CURRENT position ──
                if self.decision_rule == 'naive_closing':
                    d_naive = sqrt((prev_bike[0] - px)**2 +
                                   (prev_bike[1] - py)**2)
                    if d_cur < d_naive and bike_disp > self.prox_speed_min:
                        return 'alert', (bid, pid)
                    continue

                # ── pairwise (default): both agents' historical positions ──
                prev_ped = ped_hist[-1 - lb]
                pbx, pby = prev_bike[0], prev_bike[1]
                ppx, ppy = prev_ped[0], prev_ped[1]
                d_prev = sqrt((pbx - ppx)**2 + (pby - ppy)**2)

                if d_cur < d_prev and bike_disp > self.prox_speed_min:
                    return 'alert', (bid, pid)

        return 'warning', None
