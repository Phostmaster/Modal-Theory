import math
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class GEMParams:
    depth: float = 0.8
    stiffness: float = 0.7
    viscosity: float = 0.6

    cycles: int = 1500
    cycle_period: int = 50
    load_amplitude: float = 0.68
    dt: float = 0.05

    damage_threshold: float = 0.38
    damage_rate: float = 0.003
    nonlinear_alpha: float = 1.0

    residual_softening: float = 0.55
    residual_viscosity_gain: float = 0.45

    memory_gain: float = 0.060
    memory_decay_scale: float = 1.60

    edge_width_factor: float = 0.22
    edge_softening_strength: float = 0.35

    brittle_gain: float = 0.18
    brittle_depth_floor: float = 0.55

    output_dir: str = "gem_outputs"
    run_name: str = "single_run"


class GEMPolymerSim:
    """
    3-variable GEM toy fatigue model.

    State variables
    ---------------
    x       : fast deformation
    memory  : recoverable lingering distortion
    damage  : irreversible accumulated damage

    Interpretation
    --------------
    - depth controls remaining damage budget / basin support
    - stiffness controls immediate restoring force
    - viscosity controls recovery lag and memory retention
    """

    def __init__(self, params: GEMParams):
        self.p = params
        self.x = 0.0
        self.memory = 0.0
        self.damage = 0.0
        self.failed = False
        self.failure_cycle = None

    def load(self, step: int) -> float:
        phase = 2.0 * math.pi * step / self.p.cycle_period
        return self.p.load_amplitude * math.sin(phase)

    def effective_depth(self) -> float:
        return max(1e-9, self.p.depth - self.damage)

    def effective_stiffness(self) -> float:
        frac = max(
            0.10,
            1.0 - self.p.residual_softening * (self.damage / max(self.p.depth, 1e-9))
        )
        return max(1e-6, self.p.stiffness * frac)

    def effective_viscosity(self) -> float:
        frac = 1.0 + self.p.residual_viscosity_gain * (self.damage / max(self.p.depth, 1e-9))
        return max(1e-6, self.p.viscosity * frac)

    def restoring_force(self, total_x: float, kappa_eff: float) -> float:
        """
        Nonlinear restoring force with gentle basin-edge weakening.
        """
        force = kappa_eff * total_x + self.p.nonlinear_alpha * (total_x ** 3)

        edge = self.p.damage_threshold
        dist = abs(total_x)

        if dist > edge:
            overshoot = dist - edge
            width = max(1e-9, self.p.edge_width_factor * self.effective_depth())
            softening = 1.0 - self.p.edge_softening_strength * min(1.0, overshoot / width)
            softening = max(0.35, softening)
            force *= softening

        return force

    def brittle_damage_multiplier(self, total_x: float, kappa_eff: float, D_eff: float) -> float:
        """
        Selective brittle spike:
        - only matters beyond the damage threshold
        - stronger when stiffness is high
        - stronger when remaining depth is low
        """
        excess = max(0.0, abs(total_x) - self.p.damage_threshold)
        if excess <= 0.0:
            return 1.0

        stiffness_ratio = kappa_eff / max(self.p.stiffness, 1e-9)
        depth_ratio = D_eff / max(self.p.depth, 1e-9)

        if depth_ratio >= self.p.brittle_depth_floor:
            return 1.0

        shallow_factor = (self.p.brittle_depth_floor - depth_ratio) / max(self.p.brittle_depth_floor, 1e-9)
        stiffness_factor = max(0.0, stiffness_ratio - 0.90)
        edge_factor = excess / max(self.p.damage_threshold, 1e-9)

        return 1.0 + self.p.brittle_gain * stiffness_factor * shallow_factor * (1.0 + edge_factor)

    def step(self, step_idx: int) -> Dict[str, float]:
        if self.failed:
            return {}

        load = self.load(step_idx)

        D_eff = self.effective_depth()
        kappa_eff = self.effective_stiffness()
        gamma_eff = self.effective_viscosity()

        total_x_before = self.x + self.memory

        # Fast deformation channel
        force = load - self.restoring_force(total_x_before, kappa_eff)
        dx = (force / gamma_eff) * self.p.dt
        self.x += dx

        # Visible deformation after fast update
        total_x = self.x + self.memory

        # Recoverable memory channel
        # Higher viscosity should retain more distortion, but not explode
        memory_inflow = self.p.memory_gain * self.p.viscosity * abs(self.x) * self.p.dt
        memory_decay = (self.memory / max(self.p.memory_decay_scale * gamma_eff, 1e-9)) * self.p.dt
        self.memory += memory_inflow - memory_decay
        self.memory = max(0.0, self.memory)

        # Recompute visible deformation after memory update
        total_x = self.x + self.memory

        # Irreversible damage only near / beyond basin edge
        excess = max(0.0, abs(total_x) - self.p.damage_threshold)
        if excess > 0.0:
            edge_factor = 1.0 + 0.5 * excess / max(D_eff, 1e-9)
            brittle_factor = self.brittle_damage_multiplier(total_x, kappa_eff, D_eff)
            # Lower baseline damage than your last version, so low-viscosity is not universally fatal
            self.damage += self.p.damage_rate * excess * edge_factor * brittle_factor * self.p.dt

        # Update effective parameters after damage
        D_eff = self.effective_depth()
        kappa_eff = self.effective_stiffness()
        gamma_eff = self.effective_viscosity()

        # Persistence score
        P = (
            (D_eff / max(self.p.depth, 1e-9))
            * (kappa_eff / max(self.p.stiffness, 1e-9))
            * math.exp(-(total_x ** 2))
            / (gamma_eff / max(self.p.viscosity, 1e-9))
        )

        # Failure criterion
        if self.damage >= self.p.depth:
            self.failed = True
            self.failure_cycle = step_idx

        return {
            "step": step_idx,
            "cycle": step_idx / self.p.cycle_period,
            "load": load,
            "deformation": self.x,
            "memory": self.memory,
            "total_deformation": total_x,
            "damage": self.damage,
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

    last = results[-1]
    damage_final = last.get("damage", 0.0)
    memory_final = last.get("memory", 0.0)
    total_def_final = last.get("total_deformation", 0.0)
    persistence_final = last.get("persistence", 0.0)
    failed = last.get("failed", 0)
    stiffness_final = last.get("effective_stiffness", params.stiffness)
    depth_final = last.get("effective_depth", params.depth)

    if failed == 1:
        frac = len(results) / max(params.cycles * params.cycle_period, 1)
        depth_ratio = depth_final / max(params.depth, 1e-9)
        if frac < 0.35 and stiffness_final > 0.70 * params.stiffness and depth_ratio < 0.40:
            return "brittle collapse"
        return "collapse"

    # Strong trapped regime: a lot of memory, little real damage
    if memory_final > 0.03 * params.depth and damage_final < 0.06 * params.depth:
        return "trapped metastable"

    if damage_final > 0.15 * params.depth:
        return "creeping distortion"

    if (
        persistence_final > 0.60
        and damage_final < 0.04 * params.depth
        and memory_final < 0.04 * params.depth
    ):
        return "clean recovery"

    if abs(total_def_final) > 0.08 or memory_final > 0.04 * params.depth:
        return "stable with drift"

    return "stable"


def save_csv(results: List[Dict[str, float]], out_csv: Path) -> None:
    if not results:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


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
            f"Final memory: {last['memory']:.6f}",
            f"Final total deformation: {last['total_deformation']:.6f}",
            f"Final damage: {last['damage']:.6f}",
            f"Final effective_depth: {last['effective_depth']:.6f}",
            f"Final effective_stiffness: {last['effective_stiffness']:.6f}",
            f"Final effective_viscosity: {last['effective_viscosity']:.6f}",
            f"Final persistence: {last['persistence']:.6f}",
            "",
            "Parameters:",
        ]
        for k, v in asdict(params).items():
            lines.append(f"  {k}: {v}")
    else:
        lines = [f"Run name: {params.run_name}", "No results."]

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def plot_single_run(results: List[Dict[str, float]], params: GEMParams, out_png: Path, title: str) -> None:
    if not results:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    cycles = [r["cycle"] for r in results]
    load = [r["load"] for r in results]
    memory = [r["memory"] for r in results]
    total_deformation = [r["total_deformation"] for r in results]
    damage = [r["damage"] for r in results]
    depth_eff = [r["effective_depth"] for r in results]
    persistence = [r["persistence"] for r in results]

    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(cycles, load, label="Load")
    plt.plot(cycles, total_deformation, label="Total deformation")
    plt.ylabel("Load / Deformation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(cycles, memory, label="Recoverable memory")
    plt.plot(cycles, damage, label="Irreversible damage")
    plt.plot(cycles, depth_eff, label="Effective depth")
    plt.ylabel("Memory / Damage / Depth")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(cycles, persistence, label="Persistence")
    plt.axhline(0.5, linestyle="--", label="Reference threshold")
    plt.ylabel("Persistence")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(total_deformation, load)
    plt.xlabel("Total deformation")
    plt.ylabel("Load")
    plt.title("Loop trace")
    plt.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_multiple_runs(run_bundle, out_png: Path, title: str) -> None:
    if not run_bundle:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    for label, results, _, _, _ in run_bundle:
        cycles = [r["cycle"] for r in results]
        total_deformation = [r["total_deformation"] for r in results]
        plt.plot(cycles, total_deformation, label=label)
    plt.ylabel("Total deformation")
    plt.title("Deformation comparison")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    plt.subplot(3, 1, 2)
    for label, results, _, _, _ in run_bundle:
        cycles = [r["cycle"] for r in results]
        damage = [r["damage"] for r in results]
        plt.plot(cycles, damage, label=label)
    plt.ylabel("Damage")
    plt.title("Irreversible damage comparison")
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


def run_single_example() -> None:
    params = GEMParams(
        depth=0.85,
        stiffness=0.70,
        viscosity=0.55,
        cycles=1000,
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
    presets = [
        (
            "brittle_like",
            GEMParams(
                depth=0.35,
                stiffness=0.98,
                viscosity=0.35,
                cycles=700,
                load_amplitude=0.92,
                damage_threshold=0.34,
                brittle_gain=0.26,
                run_name="brittle_like",
            ),
        ),
        (
            "creep_like",
            GEMParams(
                depth=0.75,
                stiffness=0.25,
                viscosity=0.65,
                cycles=1800,
                load_amplitude=0.78,
                run_name="creep_like",
            ),
        ),
        (
            "metastable_like",
            GEMParams(
                depth=1.00,
                stiffness=0.60,
                viscosity=1.25,
                cycles=1500,
                load_amplitude=0.72,
                memory_gain=0.075,
                memory_decay_scale=2.0,
                run_name="metastable_like",
            ),
        ),
        (
            "balanced",
            GEMParams(
                depth=0.85,
                stiffness=0.70,
                viscosity=0.55,
                cycles=1000,
                load_amplitude=0.62,
                run_name="balanced",
            ),
        ),
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
            "final_memory": last.get("memory", np.nan),
            "final_damage": last.get("damage", np.nan),
            "final_persistence": last.get("persistence", np.nan),
        })

    out_csv = out_dir / "quick_comparison_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_multiple_runs(bundle, out_dir / "quick_comparison_plot.png", "GEM Quick Comparison")
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
                    "final_memory": last.get("memory", np.nan),
                    "final_damage": last.get("damage", np.nan),
                    "final_total_deformation": last.get("total_deformation", np.nan),
                    "final_persistence": last.get("persistence", np.nan),
                    "failed": last.get("failed", np.nan),
                })

    scan_csv = out_dir / "axis_scan_summary.csv"
    with scan_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scan_rows)

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
        subset = [r for r in scan_rows if abs(r["viscosity"] - G) < 1e-12]
        local_counts = {}
        for row in subset:
            cls = row["classification"]
            local_counts[cls] = local_counts.get(cls, 0) + 1
        for cls, count in sorted(local_counts.items(), key=lambda x: x[0]):
            frac = count / len(subset) if subset else 0.0
            lines.append(f"  {cls}: {count} runs ({frac:.1%})")

    grouped_txt.write_text("\n".join(lines), encoding="utf-8")

    ranked = sorted(
        scan_rows,
        key=lambda r: (
            -float(r["final_persistence"]) if not np.isnan(r["final_persistence"]) else float("inf"),
            float(r["final_damage"]) if not np.isnan(r["final_damage"]) else float("inf"),
        )
    )

    top_rows = ranked[:10]
    bottom_rows = ranked[-10:] if len(ranked) >= 10 else ranked

    ranked_txt = out_dir / "axis_scan_ranked_summary.txt"
    lines = ["Top 10 runs by final persistence", "==============================", ""]
    for row in top_rows:
        lines.append(
            f"D={row['depth']:.2f}, K={row['stiffness']:.2f}, G={row['viscosity']:.2f} | "
            f"{row['classification']} | persistence={row['final_persistence']:.4f} | "
            f"memory={row['final_memory']:.4f} | damage={row['final_damage']:.4f}"
        )

    lines.append("")
    lines.append("Bottom 10 runs by final persistence")
    lines.append("===================================")
    lines.append("")
    for row in bottom_rows:
        lines.append(
            f"D={row['depth']:.2f}, K={row['stiffness']:.2f}, G={row['viscosity']:.2f} | "
            f"{row['classification']} | persistence={row['final_persistence']:.4f} | "
            f"memory={row['final_memory']:.4f} | damage={row['final_damage']:.4f}"
        )

    ranked_txt.write_text("\n".join(lines), encoding="utf-8")

    for G in viscosity_values:
        subset = [r for r in scan_rows if abs(r["viscosity"] - G) < 1e-12]
        if not subset:
            continue

        Z = np.zeros((len(depth_values), len(stiffness_values)))
        for i, D in enumerate(depth_values):
            for j, K in enumerate(stiffness_values):
                match = next(r for r in subset if r["depth"] == D and r["stiffness"] == K)
                cls = match["classification"]
                if cls == "clean recovery":
                    val = 5
                elif cls == "stable":
                    val = 4
                elif cls == "stable with drift":
                    val = 3
                elif cls == "trapped metastable":
                    val = 2
                elif cls == "creeping distortion":
                    val = 1
                elif cls == "brittle collapse":
                    val = 0
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
        plt.colorbar(label="0 brittle/collapse ... 5 clean recovery")
        plt.tight_layout()
        plt.savefig(out_dir / f"heatmap_viscosity_{G:.2f}.png", dpi=160)
        plt.close()

    print(f"Axis scan complete. Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    run_single_example()
    run_quick_comparison()

    base = GEMParams(
        cycles=1200,
        cycle_period=50,
        load_amplitude=0.75,
        damage_threshold=0.38,
        damage_rate=0.004,
        memory_gain=0.060,
        memory_decay_scale=1.60,
        brittle_gain=0.18,
        brittle_depth_floor=0.55,
        output_dir="gem_outputs",
        run_name="scan_base",
    )

    print("Axis scan parameters:")
    print("  damage_rate =", base.damage_rate)
    print("  load_amplitude =", base.load_amplitude)
    print("  damage_threshold =", base.damage_threshold)
    print("  memory_gain =", base.memory_gain)
    print("  memory_decay_scale =", base.memory_decay_scale)

    axis_scan(
        depth_values=[0.4, 0.6, 0.8, 1.0],
        stiffness_values=[0.2, 0.4, 0.6, 0.8, 1.0],
        viscosity_values=[0.4, 0.8, 1.2],
        base_params=base,
    )