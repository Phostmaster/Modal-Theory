import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Incident:
    segment: int
    start_step: int
    end_step: int
    capacity_factor: float


@dataclass
class TrafficParams:
    n_segments: int = 40
    dx_km: float = 0.5
    dt_hr: float = 5.0 / 3600.0          # 5 seconds
    steps: int = 720                     # 1 hour

    free_speed_kph: float = 100.0
    min_speed_limit_kph: float = 55.0
    jam_density: float = 140.0           # veh/km/lane
    critical_density: float = 32.0       # veh/km/lane
    wave_speed_kph: float = 20.0         # backward wave speed
    base_capacity_vph: float = 2200.0    # veh/hour/lane

    demand_main_vph: float = 1750.0
    demand_ramp_vph: float = 420.0
    ramp_segment: int = 10

    control_zone_upstream: int = 8
    control_zone_downstream: int = 4
    ramp_meter_min: float = 0.55
    shoulder_capacity_boost: float = 0.35
    post_incident_flush_boost: float = 0.15

    p_threshold: float = 0.52
    persistence_recovery_band: float = 0.05   # 5%
    recovery_hold_steps: int = 12             # 1 minute hold
    cascade_speed_drop_kph: float = 20.0
    cascade_quiet_steps: int = 12             # separate events if quiet 1 minute
    cascade_window_steps: int = 60            # 5 minutes

    density_noise: float = 0.01

    depth_weight: float = 1.0
    stiffness_weight: float = 1.0
    viscosity_weight: float = 1.0

    incidents: List[Incident] = field(default_factory=list)


class GEMTrafficSim:
    def __init__(self, params: TrafficParams, use_gem_control: bool):
        self.p = params
        self.use_gem_control = use_gem_control

        self.density = np.full(self.p.n_segments, 18.0, dtype=float)
        self.memory = np.zeros(self.p.n_segments, dtype=float)

        self.history_density = []
        self.history_speed = []
        self.history_flow = []
        self.history_persistence = []
        self.history_depth = []
        self.history_stiffness = []
        self.history_viscosity = []
        self.history_speed_limits = []
        self.history_ramp_meter = []
        self.history_congestion = []
        self.history_downstream_outflow = []

    def active_incidents(self, step: int) -> List[Incident]:
        return [
            inc for inc in self.p.incidents
            if inc.start_step <= step <= inc.end_step
        ]

    def incident_capacity_profile(self, step: int) -> np.ndarray:
        cap = np.full(self.p.n_segments, self.p.base_capacity_vph, dtype=float)
        for inc in self.active_incidents(step):
            cap[inc.segment] *= inc.capacity_factor
        return cap

    def effective_capacity_with_control(
        self,
        step: int,
        base_capacity: np.ndarray,
        persistence: np.ndarray,
        depth: np.ndarray,
        viscosity: np.ndarray,
    ) -> np.ndarray:
        cap = base_capacity.copy()

        if not self.use_gem_control:
            return cap

        # Active incident support and post-incident flush
        active = self.active_incidents(step)

        segments_to_boost = []
        for inc in active:
            for j in range(inc.segment, min(self.p.n_segments, inc.segment + self.p.control_zone_downstream + 1)):
                segments_to_boost.append(j)

        # Short post-incident recovery assist
        for inc in self.p.incidents:
            if inc.end_step < step <= inc.end_step + 72:  # 6 minutes after incident
                for j in range(inc.segment, min(self.p.n_segments, inc.segment + self.p.control_zone_downstream + 1)):
                    segments_to_boost.append(j)

        segments_to_boost = sorted(set(segments_to_boost))

        for j in segments_to_boost:
            sev = max(0.0, self.p.p_threshold - persistence[j])
            extra = 0.0

            # Active incident: strong anti-viscosity push
            if any(j >= inc.segment and j <= min(self.p.n_segments - 1, inc.segment + self.p.control_zone_downstream)
                   for inc in active):
                extra += self.p.shoulder_capacity_boost * (0.5 + sev)

            # Post-incident: gentle flush
            if not active:
                extra += self.p.post_incident_flush_boost * (0.5 + sev)

            cap[j] *= (1.0 + extra)

        return cap

    def control_speed_limits(
        self,
        step: int,
        depth: np.ndarray,
        stiffness: np.ndarray,
        viscosity: np.ndarray,
        persistence: np.ndarray,
    ) -> np.ndarray:
        limits = np.full(self.p.n_segments, self.p.free_speed_kph, dtype=float)

        if not self.use_gem_control:
            return limits

        for inc in self.active_incidents(step):
            up0 = max(0, inc.segment - self.p.control_zone_upstream)
            up1 = inc.segment

            for j in range(up0, up1):
                # Only harmonise when stiffness is low, but do not choke a critically shallow system
                if depth[j] < 0.15:
                    limits[j] = self.p.free_speed_kph
                    continue

                low_stiffness = max(0.0, 0.65 - stiffness[j]) / 0.65
                high_viscosity = max(0.0, viscosity[j] - 0.35) / 0.65
                severity = 0.7 * low_stiffness + 0.3 * high_viscosity

                reduction = min(0.25, 0.8 * severity)
                limits[j] = max(self.p.min_speed_limit_kph, self.p.free_speed_kph * (1.0 - reduction))

        return limits

    def control_ramp_meter(
        self,
        depth: np.ndarray,
        stiffness: np.ndarray,
        viscosity: np.ndarray,
        persistence: np.ndarray,
    ) -> float:
        if not self.use_gem_control:
            return 1.0

        idx = self.p.ramp_segment
        sl = slice(max(0, idx - 1), min(self.p.n_segments, idx + 2))

        local_depth = np.mean(depth[sl])
        local_stiffness = np.mean(stiffness[sl])
        local_viscosity = np.mean(viscosity[sl])
        local_p = np.mean(persistence[sl])

        sev_depth = max(0.0, 0.35 - local_depth) / 0.35
        sev_visc = max(0.0, local_viscosity - 0.35) / 0.65
        sev_p = max(0.0, self.p.p_threshold - local_p)

        severity = 0.5 * sev_depth + 0.3 * sev_visc + 0.2 * sev_p

        rate = 1.0 - min(0.4, 1.0 * severity)
        return max(self.p.ramp_meter_min, rate)

    def fundamental_speed(self, density: np.ndarray, speed_limits: np.ndarray) -> np.ndarray:
        raw_speed = self.p.free_speed_kph * np.maximum(0.05, 1.0 - density / self.p.jam_density)
        return np.minimum(raw_speed, speed_limits)

    def sending(self, density: np.ndarray, speed: np.ndarray, capacity: np.ndarray) -> np.ndarray:
        return np.minimum(density * speed, capacity)

    def receiving(self, density: np.ndarray, capacity: np.ndarray) -> np.ndarray:
        return np.minimum(capacity, self.p.wave_speed_kph * np.maximum(0.0, self.p.jam_density - density))

    def compute_axes(self, density: np.ndarray, speed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Depth: spare density before criticality
        depth = np.clip((self.p.critical_density - density) / self.p.critical_density, 0.0, 1.2)

        # Stiffness: penalise sharp local gradients in speed and density
        speed_grad = np.zeros_like(speed)
        dens_grad = np.zeros_like(density)

        speed_grad[1:-1] = 0.5 * np.abs(speed[2:] - speed[:-2]) / max(self.p.free_speed_kph, 1e-9)
        speed_grad[0] = np.abs(speed[1] - speed[0]) / max(self.p.free_speed_kph, 1e-9)
        speed_grad[-1] = np.abs(speed[-1] - speed[-2]) / max(self.p.free_speed_kph, 1e-9)

        dens_grad[1:-1] = 0.5 * np.abs(density[2:] - density[:-2]) / max(self.p.critical_density, 1e-9)
        dens_grad[0] = np.abs(density[1] - density[0]) / max(self.p.critical_density, 1e-9)
        dens_grad[-1] = np.abs(density[-1] - density[-2]) / max(self.p.critical_density, 1e-9)

        shock = 0.65 * speed_grad + 0.35 * dens_grad
        stiffness = 1.0 / (1.0 + 6.0 * shock)
        stiffness = np.clip(stiffness, 0.0, 1.0)

        # Viscosity: congestion memory
        congested = density > self.p.critical_density
        self.memory = 0.985 * self.memory + 0.07 * congested.astype(float)
        self.memory *= np.where(congested, 1.0, 0.993)
        viscosity = np.clip(self.memory, 0.0, 1.0)

        persistence = (
            np.power(np.maximum(depth, 1e-6), self.p.depth_weight)
            * np.power(np.maximum(stiffness, 1e-6), self.p.stiffness_weight)
            * np.power(np.maximum(1.0 - viscosity, 1e-6), self.p.viscosity_weight)
        )
        persistence = np.clip(persistence, 0.0, 1.0)

        return depth, stiffness, viscosity, persistence

    def step(self, step_idx: int):
        density = self.density.copy()

        base_capacity = self.incident_capacity_profile(step_idx)

        # Provisional axes from uncontrolled free-speed state
        provisional_limits = np.full(self.p.n_segments, self.p.free_speed_kph)
        provisional_speed = self.fundamental_speed(density, provisional_limits)
        depth, stiffness, viscosity, persistence = self.compute_axes(density, provisional_speed)

        speed_limits = self.control_speed_limits(step_idx, depth, stiffness, viscosity, persistence)
        ramp_meter = self.control_ramp_meter(depth, stiffness, viscosity, persistence)
        capacity = self.effective_capacity_with_control(step_idx, base_capacity, persistence, depth, viscosity)

        speed = self.fundamental_speed(density, speed_limits)
        send = self.sending(density, speed, capacity)
        recv = self.receiving(density, capacity)

        next_density = density.copy()

        # Upstream boundary
        inflow0 = min(self.p.demand_main_vph, recv[0])
        outflow0 = min(send[0], recv[1]) if self.p.n_segments > 1 else send[0]
        next_density[0] += (self.p.dt_hr / self.p.dx_km) * (inflow0 - outflow0)

        flows_between = np.zeros(self.p.n_segments - 1, dtype=float)

        for i in range(self.p.n_segments - 1):
            flows_between[i] = min(send[i], recv[i + 1])

        # Interior cells
        for i in range(1, self.p.n_segments - 1):
            inflow = flows_between[i - 1]

            if i == self.p.ramp_segment:
                spare_receive = max(0.0, recv[i] - inflow)
                ramp_inflow = min(self.p.demand_ramp_vph * ramp_meter, spare_receive)
                inflow += ramp_inflow

            outflow = flows_between[i]
            next_density[i] += (self.p.dt_hr / self.p.dx_km) * (inflow - outflow)

        # Last cell to sink
        inflow_last = flows_between[-1] if self.p.n_segments > 1 else inflow0
        outflow_last = send[-1]
        next_density[-1] += (self.p.dt_hr / self.p.dx_km) * (inflow_last - outflow_last)

        # Tiny perturbation
        next_density += np.random.normal(0.0, self.p.density_noise, size=self.p.n_segments)

        self.density = np.clip(next_density, 0.0, self.p.jam_density)

        total_congestion = np.sum(np.maximum(self.density - self.p.critical_density, 0.0))

        self.history_density.append(self.density.copy())
        self.history_speed.append(speed.copy())
        self.history_flow.append(send.copy())
        self.history_persistence.append(persistence.copy())
        self.history_depth.append(depth.copy())
        self.history_stiffness.append(stiffness.copy())
        self.history_viscosity.append(viscosity.copy())
        self.history_speed_limits.append(speed_limits.copy())
        self.history_ramp_meter.append(ramp_meter)
        self.history_congestion.append(total_congestion)
        self.history_downstream_outflow.append(outflow_last)

    def run(self) -> Dict[str, np.ndarray]:
        for step_idx in range(self.p.steps):
            self.step(step_idx)

        return {
            "density": np.array(self.history_density),
            "speed": np.array(self.history_speed),
            "flow": np.array(self.history_flow),
            "persistence": np.array(self.history_persistence),
            "depth": np.array(self.history_depth),
            "stiffness": np.array(self.history_stiffness),
            "viscosity": np.array(self.history_viscosity),
            "speed_limits": np.array(self.history_speed_limits),
            "ramp_meter": np.array(self.history_ramp_meter),
            "congestion": np.array(self.history_congestion),
            "downstream_outflow": np.array(self.history_downstream_outflow),
        }


def first_sustained_recovery_time(
    signal: np.ndarray,
    baseline: float,
    start_step: int,
    dt_minutes: float,
    tol_frac: float,
    hold_steps: int,
) -> float:
    lower = baseline * (1.0 - tol_frac)
    upper = baseline * (1.0 + tol_frac)

    for i in range(start_step, len(signal) - hold_steps + 1):
        window = signal[i:i + hold_steps]
        if np.all((window >= lower) & (window <= upper)):
            return (i - start_step) * dt_minutes
    return float("nan")


def count_cascade_events(
    speed_hist: np.ndarray,
    params: TrafficParams,
    main_incident: Incident,
) -> int:
    pre_slice = slice(max(0, main_incident.start_step - 60), main_incident.start_step)
    baseline_speed_per_segment = np.mean(speed_hist[pre_slice], axis=0)

    t0 = main_incident.start_step
    t1 = min(speed_hist.shape[0], main_incident.start_step + params.cascade_window_steps)

    exclude_lo = max(0, main_incident.segment - 1)
    exclude_hi = min(params.n_segments - 1, main_incident.segment + 1)

    cascade_steps = []
    for t in range(t0, t1):
        speed_drop = baseline_speed_per_segment - speed_hist[t]
        affected = np.where(speed_drop > params.cascade_speed_drop_kph)[0]
        affected = [a for a in affected if not (exclude_lo <= a <= exclude_hi)]
        cascade_steps.append(len(affected) > 0)

    count = 0
    in_event = False
    quiet = 0

    for active in cascade_steps:
        if active and not in_event:
            count += 1
            in_event = True
            quiet = 0
        elif active and in_event:
            quiet = 0
        elif not active and in_event:
            quiet += 1
            if quiet >= params.cascade_quiet_steps:
                in_event = False

    return count


def compute_metrics(results: Dict[str, np.ndarray], params: TrafficParams, scenario_name: str) -> Dict[str, float]:
    t_min = np.arange(params.steps) * params.dt_hr * 60.0
    avg_speed = np.mean(results["speed"], axis=1)
    avg_p = np.mean(results["persistence"], axis=1)
    avg_density = np.mean(results["density"], axis=1)

    main_incident = min(params.incidents, key=lambda inc: inc.start_step)

    pre_slice = slice(max(0, main_incident.start_step - 60), main_incident.start_step)
    incident_slice = slice(main_incident.start_step, main_incident.end_step + 1)
    post_slice = slice(main_incident.end_step + 1, min(params.steps, main_incident.end_step + 121))

    baseline_flow = np.mean(results["downstream_outflow"][pre_slice])

    recovery_from_start = first_sustained_recovery_time(
        results["downstream_outflow"],
        baseline_flow,
        main_incident.start_step,
        params.dt_hr * 60.0,
        params.persistence_recovery_band,
        params.recovery_hold_steps,
    )

    recovery_from_end = first_sustained_recovery_time(
        results["downstream_outflow"],
        baseline_flow,
        min(main_incident.end_step + 1, params.steps - 1),
        params.dt_hr * 60.0,
        params.persistence_recovery_band,
        params.recovery_hold_steps,
    )

    # Throughput = total vehicles leaving downstream boundary
    total_throughput = np.sum(results["downstream_outflow"]) * params.dt_hr

    # Energy / stop-go burden
    dv = np.diff(results["speed"], axis=0)
    stop_go_burden = np.sum(np.abs(dv))

    # Persistence debt: time spent below threshold
    persistence_debt = np.sum(np.maximum(0.0, params.p_threshold - avg_p)) * params.dt_hr

    # Collapse footprint
    segments_over_critical = np.sum(results["density"] > params.critical_density, axis=1)
    max_jam_extent = np.max(segments_over_critical)
    mean_jam_extent = np.mean(segments_over_critical)

    # Worst low-speed phase
    sustained_low_speed = np.sum(avg_speed < 35.0) * params.dt_hr * 60.0

    metrics = {
        "avg_speed_kph": float(np.mean(avg_speed)),
        "avg_persistence": float(np.mean(avg_p)),
        "mean_congestion_excess": float(np.mean(results["congestion"])),
        "peak_congestion_excess": float(np.max(results["congestion"])),
        "recovery_from_incident_start_min": float(recovery_from_start),
        "recovery_from_incident_end_min": float(recovery_from_end),
        "speed_std_normal": float(np.std(avg_speed[pre_slice])),
        "speed_std_incident": float(np.std(avg_speed[incident_slice])),
        "speed_std_post": float(np.std(avg_speed[post_slice])),
        "cascade_failures": int(count_cascade_events(results["speed"], params, main_incident)),
        "total_throughput_vehicles": float(total_throughput),
        "stop_go_burden": float(stop_go_burden),
        "persistence_debt": float(persistence_debt),
        "max_jam_extent_segments": float(max_jam_extent),
        "mean_jam_extent_segments": float(mean_jam_extent),
        "sustained_low_speed_minutes": float(sustained_low_speed),
    }

    print(f"\n{scenario_name}")
    print("-" * len(scenario_name))
    print(f"Average speed:                    {metrics['avg_speed_kph']:.2f} km/h")
    print(f"Average persistence:              {metrics['avg_persistence']:.3f}")
    print(f"Mean congestion excess:           {metrics['mean_congestion_excess']:.2f}")
    print(f"Peak congestion excess:           {metrics['peak_congestion_excess']:.2f}")
    print(f"Recovery from incident start:     {metrics['recovery_from_incident_start_min']:.2f} min")
    print(f"Recovery from incident end:       {metrics['recovery_from_incident_end_min']:.2f} min")
    print(f"Speed std (normal / incident / post): "
          f"{metrics['speed_std_normal']:.2f} / "
          f"{metrics['speed_std_incident']:.2f} / "
          f"{metrics['speed_std_post']:.2f}")
    print(f"Cascade failures:                 {metrics['cascade_failures']}")
    print(f"Total throughput:                 {metrics['total_throughput_vehicles']:.2f} vehicles")
    print(f"Stop-go burden:                   {metrics['stop_go_burden']:.2f}")
    print(f"Persistence debt:                 {metrics['persistence_debt']:.4f} hr")
    print(f"Max jam extent:                   {metrics['max_jam_extent_segments']:.0f} segments")
    print(f"Mean jam extent:                  {metrics['mean_jam_extent_segments']:.2f} segments")
    print(f"Sustained low-speed time:         {metrics['sustained_low_speed_minutes']:.2f} min")

    return metrics


def plot_comparison(
    base: Dict[str, np.ndarray],
    gem: Dict[str, np.ndarray],
    params: TrafficParams,
    title_prefix: str,
):
    t_min = np.arange(params.steps) * params.dt_hr * 60.0
    incident = min(params.incidents, key=lambda inc: inc.start_step)

    base_avg_speed = np.mean(base["speed"], axis=1)
    gem_avg_speed = np.mean(gem["speed"], axis=1)

    base_avg_p = np.mean(base["persistence"], axis=1)
    gem_avg_p = np.mean(gem["persistence"], axis=1)

    fig = plt.figure(figsize=(14, 12))

    ax1 = plt.subplot(3, 2, 1)
    im1 = ax1.imshow(
        base["density"].T,
        origin="lower",
        aspect="auto",
        extent=[0, t_min[-1], 0, params.n_segments - 1],
    )
    ax1.set_title(f"{title_prefix} baseline density")
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Segment")
    plt.colorbar(im1, ax=ax1, label="Density")

    ax2 = plt.subplot(3, 2, 2)
    im2 = ax2.imshow(
        gem["density"].T,
        origin="lower",
        aspect="auto",
        extent=[0, t_min[-1], 0, params.n_segments - 1],
    )
    ax2.set_title(f"{title_prefix} GEM density")
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Segment")
    plt.colorbar(im2, ax=ax2, label="Density")

    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(t_min, base_avg_speed, label="Baseline")
    ax3.plot(t_min, gem_avg_speed, label="GEM")
    ax3.axvspan(
        incident.start_step * params.dt_hr * 60.0,
        incident.end_step * params.dt_hr * 60.0,
        alpha=0.15,
    )
    ax3.set_title("Average corridor speed")
    ax3.set_xlabel("Time (minutes)")
    ax3.set_ylabel("km/h")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(t_min, base_avg_p, label="Baseline")
    ax4.plot(t_min, gem_avg_p, label="GEM")
    ax4.axhline(params.p_threshold, linestyle="--", label="P threshold")
    ax4.axvspan(
        incident.start_step * params.dt_hr * 60.0,
        incident.end_step * params.dt_hr * 60.0,
        alpha=0.15,
    )
    ax4.set_title("Average persistence")
    ax4.set_xlabel("Time (minutes)")
    ax4.set_ylabel("P")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(t_min, base["congestion"], label="Baseline")
    ax5.plot(t_min, gem["congestion"], label="GEM")
    ax5.axvspan(
        incident.start_step * params.dt_hr * 60.0,
        incident.end_step * params.dt_hr * 60.0,
        alpha=0.15,
    )
    ax5.set_title("Total congestion excess")
    ax5.set_xlabel("Time (minutes)")
    ax5.set_ylabel("Excess density above critical")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(t_min, gem["ramp_meter"], label="Ramp meter fraction")
    ax6.plot(
        t_min,
        np.mean(gem["speed_limits"], axis=1) / params.free_speed_kph,
        label="Mean speed-limit fraction",
    )
    ax6.axvspan(
        incident.start_step * params.dt_hr * 60.0,
        incident.end_step * params.dt_hr * 60.0,
        alpha=0.15,
    )
    ax6.set_title("GEM control actions")
    ax6.set_xlabel("Time (minutes)")
    ax6.set_ylabel("Control level")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()
    plt.show()


def compare_metrics(title: str, base_metrics: Dict[str, float], gem_metrics: Dict[str, float]):
    print(f"\n{title}")
    print("=" * len(title))
    metric_order = [
        "avg_speed_kph",
        "avg_persistence",
        "mean_congestion_excess",
        "peak_congestion_excess",
        "recovery_from_incident_start_min",
        "recovery_from_incident_end_min",
        "speed_std_normal",
        "speed_std_incident",
        "speed_std_post",
        "cascade_failures",
        "total_throughput_vehicles",
        "stop_go_burden",
        "persistence_debt",
        "max_jam_extent_segments",
        "mean_jam_extent_segments",
        "sustained_low_speed_minutes",
    ]

    for key in metric_order:
        b = base_metrics[key]
        g = gem_metrics[key]
        print(f"{key:32s}  baseline={b:10.3f}   GEM={g:10.3f}")


def run_scenario(params: TrafficParams, title: str, do_plot: bool = True):
    np.random.seed(42)
    base_sim = GEMTrafficSim(params, use_gem_control=False)
    base_results = base_sim.run()
    base_metrics = compute_metrics(base_results, params, f"{title} — Baseline")

    np.random.seed(42)
    gem_sim = GEMTrafficSim(params, use_gem_control=True)
    gem_results = gem_sim.run()
    gem_metrics = compute_metrics(gem_results, params, f"{title} — GEM-controlled")

    compare_metrics(f"{title} Metrics Comparison", base_metrics, gem_metrics)

    if do_plot:
        plot_comparison(base_results, gem_results, params, title)

    return base_results, gem_results, base_metrics, gem_metrics


if __name__ == "__main__":
    # Scenario 1: single incident
    single_params = TrafficParams(
        incidents=[
            Incident(segment=22, start_step=180, end_step=420, capacity_factor=0.38),
        ]
    )

    run_scenario(single_params, "Single Incident Scenario", do_plot=True)

    # Scenario 2: worst-case resilience — multiple incidents
    multi_params = TrafficParams(
        incidents=[
            Incident(segment=22, start_step=140, end_step=260, capacity_factor=0.42),
            Incident(segment=14, start_step=300, end_step=360, capacity_factor=0.55),
            Incident(segment=29, start_step=430, end_step=520, capacity_factor=0.50),
        ]
    )

    run_scenario(multi_params, "Multi-Incident Resilience Scenario", do_plot=True)