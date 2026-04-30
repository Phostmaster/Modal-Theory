import math
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class GEMParams:
    depth: float = 0.8          # D: basin / damage budget
    stiffness: float = 0.7      # kappa: local restoring curvature
    viscosity: float = 0.6      # gamma: relaxation lag
    cycles: int = 2000
    cycle_period: int = 50      # steps per load cycle
    load_amplitude: float = 0.8
    damage_threshold: float = 0.6
    damage_rate: float = 0.002
    nonlinear_alpha: float = 1.2   # quartic basin strength
    residual_softening: float = 0.8
    residual_viscosity_gain: float = 0.5
    dt: float = 0.05
    output_dir: str = "gem_outputs"
    run_name: str = "single_run"


class GEMPolymerSim:
    """
    Toy GEM polymer fatigue simulator.

    State variables
    ---------------
    x : instantaneous deformation
    residual : accumulated permanent set / damage
    D_eff : remaining effective depth
    kappa_eff : damage-softened stiffness
    gamma_eff : damage-increased viscosity

    Core idea
    ---------
    - nonlinear basin around x=0
    - cyclic loading drives the system
    - damage accumulates above threshold
    - damage feeds back into stiffness and viscosity
    - failure happens when residual exceeds depth budget
    """

    def __init__(self, params: GEMParams):
        self.p = params
        self.x = 0.0
        self.residual = 0.0
        self.failed = False
        self.failure_cycle = None

    def load(self, step: int) -> float:
        phase = 2.0 * math.pi * step / self.p.cycle_period
        return self.p.load_amplitude * math.sin(phase)

    def effective_depth(self) -> float:
        return max(0.0, self.p.depth - self.residual)

    def effective_stiffness(self) -> float:
        # damage lowers stiffness
        frac = max(0.0, 1.0 - self.p.residual_softening * (self.residual / max(self.p.depth, 1e-9)))
        return max(1e-6, self.p.stiffness * frac)

    def effective_viscosity(self) -> float:
        # damage increases viscosity / slows recovery
        frac = 1.0 + self.p.residual_viscosity_gain * (self.residual / max(self.p.depth, 1e-9))
        return max(1e-6, self.p.viscosity * frac)

    def restoring_force(self, x: float, kappa_eff: float) -> float:
        """
        Nonlinear basin force with a real escape barrier.
        - Inside safe basin (|x| < self.p.damage_threshold): strong restoring force.
        - Near/ beyond basin edge: force weakens sharply, allowing escape (brittle) or slow creep.
        """
        # Core harmonic + quartic restoring force
        force = kappa_eff * x + self.p.nonlinear_alpha * (x ** 3)
        
        # Basin edge softening - depth acts as a real barrier
        dist_from_center = abs(x)
        if dist_from_center > self.p.damage_threshold:
            # Sharp drop in restoring force near/ beyond the barrier
            overshoot = dist_from_center - self.p.damage_threshold
            softening = max(0.0, 1.0 - overshoot / (0.3 * self.p.depth))  # tune 0.3 if needed
            force *= softening
        
        return force

    def step(self, step_idx: int) -> Dict[str, float]:
        if self.failed:
            return {}
        load = self.load(step_idx)
        D_eff = self.effective_depth()
        kappa_eff = self.effective_stiffness()
        gamma_eff = self.effective_viscosity()

        # overdamped gradient-flow update
        force = load - self.restoring_force(self.x, kappa_eff)
        dx = (force / gamma_eff) * self.p.dt
        self.x += dx

        # Damage only when deformation exceeds threshold — gentler
        excess = max(0.0, abs(self.x) - self.p.damage_threshold)
        if excess > 0.0:
            self.residual += self.p.damage_rate * excess * self.p.dt * (0.25 + 0.6 * self.p.viscosity)

        # <<< STRONGER incomplete recovery scaled with viscosity >>>
        # High viscosity now leaves significantly more lingering distortion each cycle
        incomplete_recovery = 0.035 * self.p.viscosity * abs(self.x) * self.p.dt
        self.residual += incomplete_recovery

        # update effective parameters after damage
        D_eff = self.effective_depth()
        kappa_eff = self.effective_stiffness()
        gamma_eff = self.effective_viscosity()

        # persistence score
        P = (
            (D_eff / max(self.p.depth, 1e-9))
            * (kappa_eff / max(self.p.stiffness, 1e-9))
            * math.exp(-(self.x ** 2))
            / (gamma_eff / max(self.p.viscosity, 1e-9))
        )

        # failure criterion
        if self.residual >= self.p.depth:
            self.failed = True
            self.failure_cycle = step_idx

        return {
            "step": step_idx,
            "cycle": step_idx / self.p.cycle_period,
            "load": load,
            "deformation": self.x,
            "residual": self.residual,
            "effective_depth": D_eff,
            "effective_stiffness": kappa_eff,
            "effective_viscosity": gamma_eff,
            "persistence": P,
            "failed": int(self.failed),
        }

    def run(self) -> Tuple[List[Dict[str, float]], str]:
        results: List[Dict[str, float]] = []
        total_steps = self.p.cycles * self.p.cycle_period

        for step_idx in range(total_steps):
            row = self.step(step_idx)
            if row:
                results.append(row)
            if self.failed:
                break

        if self.failed:
            outcome = f"Collapse at cycle {self.failure_cycle / self.p.cycle_period:.2f}"
        else:
            outcome = f"Survived {self.p.cycles} cycles"

        return results, outcome


def classify_run(results: List[Dict[str, float]], params: GEMParams) -> str:
    if not results:
        return "invalid"

    arr = results
    residual_final = arr[-1]["residual"]
    deformation_final = arr[-1]["deformation"]
    persistence_final = arr[-1]["persistence"]

    if arr[-1]["failed"] == 1:
        # brittle if collapse occurs with low accumulated residual history length
        frac = len(arr) / max(params.cycles * params.cycle_period, 1)
        if frac < 0.25:
            return "brittle collapse"
        return "collapse"

    if residual_final > 0.35 * params.depth and abs(deformation_final) > 0.2:
        return "trapped metastable"

    if residual_final > 0.15 * params.depth:
        return "creeping distortion"

    if persistence_final > 0.5 and residual_final < 0.05 * params.depth:
        return "clean recovery"

    return "stable with drift"


def save_csv(results: List[Dict[str, float]], out_csv: Path) -> None:
    if not results:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def plot_single_run(results: List[Dict[str, float]], params: GEMParams, out_png: Path, title: str) -> None:
    if not results:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    steps = [r["step"] for r in results]
    cycles = [r["cycle"] for r in results]
    deformation = [r["deformation"] for r in results]
    residual = [r["residual"] for r in results]
    persistence = [r["persistence"] for r in results]
    depth_eff = [r["effective_depth"] for r in results]
    load = [r["load"] for r in results]

    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(cycles, load, label="Load")
    plt.plot(cycles, deformation, label="Deformation")
    plt.ylabel("Load / Deformation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(cycles, residual, label="Residual damage")
    plt.plot(cycles, depth_eff, label="Effective depth")
    plt.ylabel("Damage / Depth")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(cycles, persistence, label="Persistence")
    plt.axhline(0.5, linestyle="--", label="Reference threshold")
    plt.ylabel("Persistence")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(deformation, load)
    plt.xlabel("Deformation")
    plt.ylabel("Load")
    plt.title("Hysteresis-like loop trace")
    plt.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_multiple_runs(run_bundle, out_png: Path, title: str) -> None:
    """
    run_bundle: list of tuples
        [
            (label, results, params, outcome, classification),
            ...
        ]
    """
    if not run_bundle:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    for label, results, _, _, _ in run_bundle:
        cycles = [r["cycle"] for r in results]
        deformation = [r["deformation"] for r in results]
        plt.plot(cycles, deformation, label=label)
    plt.ylabel("Deformation")
    plt.title("Deformation comparison")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    plt.subplot(3, 1, 2)
    for label, results, _, _, _ in run_bundle:
        cycles = [r["cycle"] for r in results]
        residual = [r["residual"] for r in results]
        plt.plot(cycles, residual, label=label)
    plt.ylabel("Residual damage")
    plt.title("Residual comparison")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    for label, results, _, _, _ in run_bundle:
        cycles = [r["cycle"] for r in results]
        persistence = [r["persistence"] for r in results]
        plt.plot(cycles, persistence, label=label)
    plt.xlabel("Cycle")
    plt.ylabel("Persistence")
    plt.title("Persistence comparison")
    plt.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def write_summary(results: List[Dict[str, float]], params: GEMParams, outcome: str, out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    classification = classify_run(results, params)

    if results:
        last = results[-1]
        lines = [
            f"Run name: {params.run_name}",
            f"Outcome: {outcome}",
            f"Classification: {classification}",
            f"Final cycle: {last['cycle']:.2f}",
            f"Final deformation: {last['deformation']:.6f}",
            f"Final residual: {last['residual']:.6f}",
            f"Final effective depth: {last['effective_depth']:.6f}",
            f"Final effective stiffness: {last['effective_stiffness']:.6f}",
            f"Final effective viscosity: {last['effective_viscosity']:.6f}",
            f"Final persistence: {last['persistence']:.6f}",
            "",
            "Parameters:",
        ]
        for k, v in asdict(params).items():
            lines.append(f"  {k}: {v}")
    else:
        lines = [f"Run name: {params.run_name}", "No results."]

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def run_single_example() -> None:
    params = GEMParams(
        depth=0.75,
        stiffness=0.4,
        viscosity=0.85,
        cycles=5000,
        run_name="single_example",
    )

    sim = GEMPolymerSim(params)
    results, outcome = sim.run()

    out_dir = Path(params.output_dir) / params.run_name
    save_csv(results, out_dir / "run_data.csv")
    plot_single_run(results, params, out_dir / "run_plot.png", f"GEM Polymer Sim — {outcome}")
    write_summary(results, params, outcome, out_dir / "summary.txt")

    print(outcome)
    print(f"Classification: {classify_run(results, params)}")
    print(f"Saved to: {out_dir.resolve()}")

def run_quick_comparison() -> None:
    """
    Run a few hand-picked regimes so you can see GEM archetypes side by side.
    """
    presets = [
        ("brittle_like", GEMParams(depth=0.45, stiffness=0.95, viscosity=0.35, cycles=2000, run_name="brittle_like")),
        ("creep_like", GEMParams(depth=0.75, stiffness=0.25, viscosity=1.10, cycles=2000, run_name="creep_like")),
        ("metastable_like", GEMParams(depth=1.00, stiffness=0.65, viscosity=1.20, cycles=2000, run_name="metastable_like")),
        ("balanced", GEMParams(depth=0.85, stiffness=0.70, viscosity=0.55, cycles=2000, run_name="balanced")),
    ]

    out_dir = Path("gem_outputs") / "quick_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = []
    summary_rows = []

    for label, params in presets:
        sim = GEMPolymerSim(params)
        results, outcome = sim.run()
        classification = classify_run(results, params)
        bundle.append((label, results, params, outcome, classification))

        last = results[-1] if results else {}
        summary_rows.append({
            "label": label,
            "depth": params.depth,
            "stiffness": params.stiffness,
            "viscosity": params.viscosity,
            "outcome": outcome,
            "classification": classification,
            "final_cycle": last.get("cycle", np.nan),
            "final_residual": last.get("residual", np.nan),
            "final_persistence": last.get("persistence", np.nan),
        })

    # save summary csv
    out_csv = out_dir / "quick_comparison_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_multiple_runs(
        bundle,
        out_dir / "quick_comparison_plot.png",
        "GEM Quick Comparison: brittle vs creep vs metastable vs balanced",
    )

    print(f"Quick comparison saved to: {out_dir.resolve()}")


def axis_scan(
    depth_values: List[float],
    stiffness_values: List[float],
    viscosity_values: List[float],
    base_params: GEMParams,
) -> None:
    out_dir = Path(base_params.output_dir) / "axis_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_rows = []

    for D in depth_values:
        for K in stiffness_values:
            for G in viscosity_values:
                params = GEMParams(**asdict(base_params))
                params.depth = D
                params.stiffness = K
                params.viscosity = G
                params.run_name = f"D{D:.2f}_K{K:.2f}_G{G:.2f}"

                sim = GEMPolymerSim(params)
                results, outcome = sim.run()
                classification = classify_run(results, params)

                last = results[-1] if results else {}
                scan_rows.append({
                    "depth": D,
                    "stiffness": K,
                    "viscosity": G,
                    "outcome": outcome,
                    "classification": classification,
                    "final_cycle": last.get("cycle", np.nan),
                    "final_residual": last.get("residual", np.nan),
                    "final_persistence": last.get("persistence", np.nan),
                    "failed": last.get("failed", np.nan),
                })

    # save scan table
    scan_csv = out_dir / "axis_scan_summary.csv"
    with scan_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scan_rows)
    # grouped summary by classification
    class_counts = {}
    for row in scan_rows:
        cls = row["classification"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    grouped_txt = out_dir / "axis_scan_grouped_summary.txt"
    lines = ["Axis Scan Grouped Summary", "=========================", ""]
    total = len(scan_rows)

    for cls, count in sorted(class_counts.items(), key=lambda x: x[0]):
        frac = count / total if total else 0.0
        lines.append(f"{cls}: {count} runs ({frac:.1%})")

    lines.append("")
    lines.append("Per-viscosity breakdown:")

    for G in viscosity_values:
        lines.append(f"\nviscosity = {G:.2f}")
        subset = [r for r in scan_rows if abs(r['viscosity'] - G) < 1e-12]
        local_counts = {}
        for row in subset:
            cls = row["classification"]
            local_counts[cls] = local_counts.get(cls, 0) + 1
        for cls, count in sorted(local_counts.items(), key=lambda x: x[0]):
            frac = count / len(subset) if subset else 0.0
            lines.append(f"  {cls}: {count} runs ({frac:.1%})")

    grouped_txt.write_text("\n".join(lines), encoding="utf-8")

    # sort by final persistence descending
    ranked = sorted(
        scan_rows,
        key=lambda r: (
            -float(r["final_persistence"]) if not np.isnan(r["final_persistence"]) else float("inf"),
            float(r["final_residual"]) if not np.isnan(r["final_residual"]) else float("inf"),
        )
    )

    top_rows = ranked[:10]
    bottom_rows = ranked[-10:] if len(ranked) >= 10 else ranked

    ranked_txt = out_dir / "axis_scan_ranked_summary.txt"
    lines = ["Top 10 runs by final persistence", "==============================", ""]
    for row in top_rows:
        lines.append(
            f"D={row['depth']:.2f}, K={row['stiffness']:.2f}, G={row['viscosity']:.2f} | "
            f"{row['classification']} | persistence={row['final_persistence']:.4f} | residual={row['final_residual']:.4f}"
        )

    lines.append("")
    lines.append("Bottom 10 runs by final persistence")
    lines.append("===================================")
    lines.append("")
    for row in bottom_rows:
        lines.append(
            f"D={row['depth']:.2f}, K={row['stiffness']:.2f}, G={row['viscosity']:.2f} | "
            f"{row['classification']} | persistence={row['final_persistence']:.4f} | residual={row['final_residual']:.4f}"
        )

    ranked_txt.write_text("\n".join(lines), encoding="utf-8")


    # simple heatmap slices by viscosity
    for G in viscosity_values:
        subset = [r for r in scan_rows if abs(r["viscosity"] - G) < 1e-12]
        if not subset:
            continue

        Z = np.zeros((len(depth_values), len(stiffness_values)))
        for i, D in enumerate(depth_values):
            for j, K in enumerate(stiffness_values):
                match = next(r for r in subset if r["depth"] == D and r["stiffness"] == K)
                # score for heatmap
                if match["classification"] == "clean recovery":
                    val = 4
                elif match["classification"] == "stable with drift":
                    val = 3
                elif match["classification"] == "trapped metastable":
                    val = 2
                elif match["classification"] == "creeping distortion":
                    val = 1
                else:
                    val = 0
                Z[i, j] = val

        plt.figure(figsize=(8, 6))
        plt.imshow(Z, origin="lower", aspect="auto")
        plt.xticks(range(len(stiffness_values)), [f"{v:.2f}" for v in stiffness_values])
        plt.yticks(range(len(depth_values)), [f"{v:.2f}" for v in depth_values])
        plt.xlabel("Stiffness")
        plt.ylabel("Depth")
        plt.title(f"Outcome map at viscosity={G:.2f}")
        plt.colorbar(label="0 collapse ... 4 clean recovery")
        plt.tight_layout()
        plt.savefig(out_dir / f"heatmap_viscosity_{G:.2f}.png", dpi=160)
        plt.close()

    print(f"Axis scan complete. Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    # 1) single detailed run
    run_single_example()

    # 2) quick multi-run comparison
    run_quick_comparison()

    # 3) automatic axis scan
    base = GEMParams(
        cycles=1500,
        cycle_period=50,
        load_amplitude=0.8,
        damage_threshold=0.6,
        damage_rate=0.002,
        nonlinear_alpha=1.2,
        dt=0.05,
        output_dir="gem_outputs",
        run_name="scan_base",
    )

    axis_scan(
        depth_values=[0.4, 0.6, 0.8, 1.0],
        stiffness_values=[0.2, 0.4, 0.6, 0.8, 1.0],
        viscosity_values=[0.4, 0.8, 1.2],
        base_params=base,
    )

