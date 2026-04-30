import numpy as np
import matplotlib.pyplot as plt

class GEMGravityToyV2:
    """
    Toy model of GEM-style gravity:
    A test particle moves under a depth field D(x, y),
    experiencing an effective force F = -grad(D).
    This is an illustrative ansatz, not a full GR derivation.
    """

    def __init__(
        self,
        depth_strength: float = 5.0,
        dt: float = 0.005,
        steps: int = 8000,
        softening: float = 1e-4,
        damping: float = 0.995,
    ):
        self.depth_strength = depth_strength
        self.dt = dt
        self.steps = steps
        self.softening = softening
        self.damping = damping

        self.positions = []
        self.velocities = []
        self.energies = []
        self.times = []

    def reset(self) -> None:
        self.positions = []
        self.velocities = []
        self.energies = []
        self.times = []

    def depth_field(self, x: float, y: float) -> float:
        """Simple 1/r depth well (deeper near origin)."""
        r = np.sqrt(x*x + y*y + self.softening)
        return self.depth_strength / r

    def force_field(self, pos: np.ndarray) -> np.ndarray:
        """Analytic force F = -∇D for D = A / r."""
        r2 = pos[0]**2 + pos[1]**2 + self.softening
        r = np.sqrt(r2)
        return -self.depth_strength * pos / (r**3)

    def potential_energy(self, pos: np.ndarray) -> float:
        """Effective potential corresponding to attractive 1/r field."""
        r = np.sqrt(pos[0]**2 + pos[1]**2 + self.softening)
        return -self.depth_strength / r

    def total_energy(self, pos: np.ndarray, vel: np.ndarray) -> float:
        kinetic = 0.5 * np.dot(vel, vel)
        potential = self.potential_energy(pos)
        return kinetic + potential

    def run(self, start_pos=(5.0, 0.0), start_vel=(0.0, -1.25)) -> None:
        self.reset()
        pos = np.array(start_pos, dtype=float)
        vel = np.array(start_vel, dtype=float)

        for step in range(self.steps):
            t = step * self.dt
            self.positions.append(pos.copy())
            self.velocities.append(vel.copy())
            self.energies.append(self.total_energy(pos, vel))
            self.times.append(t)

            force = self.force_field(pos)

            # Semi-implicit Euler update
            vel += force * self.dt
            vel *= self.damping          # gentle damping for stability
            pos += vel * self.dt

        self.positions = np.array(self.positions)
        self.velocities = np.array(self.velocities)
        self.energies = np.array(self.energies)
        self.times = np.array(self.times)

    def classify_orbit(self) -> str:
        if len(self.positions) < 10:
            return "insufficient data"

        radii = np.sqrt(np.sum(self.positions**2, axis=1))
        r_min = np.min(radii)
        r_max = np.max(radii)
        e0 = self.energies[0]
        e1 = self.energies[-1]

        if np.isnan(r_min) or np.isnan(r_max):
            return "numerical instability"

        if r_min < 0.2:
            return "deep plunge / near-capture"

        if r_max > 20:
            return "escaping / unbound-like"

        if abs(e1 - e0) < 0.1 * max(1e-9, abs(e0)):
            return "approximately bound orbit"

        if self.damping < 1.0:
            return "damped inspiral / bound decay"

        return "bound but numerically drifting"

    def plot_trajectory(self) -> None:
        if len(self.positions) == 0:
            print("No trajectory data. Run the simulation first.")
            return

        plt.figure(figsize=(8, 8))
        plt.plot(self.positions[:, 0], self.positions[:, 1], "b-", lw=1.5, label="Test particle path")
        plt.scatter([0], [0], color="red", s=120, label="Central depth well")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("GEM Gravity Toy v2: Depth-Gradient Motion")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_energy(self) -> None:
        if len(self.energies) == 0:
            print("No energy data. Run the simulation first.")
            return

        plt.figure(figsize=(10, 4))
        plt.plot(self.times, self.energies, "k-", lw=1.2)
        plt.xlabel("Time")
        plt.ylabel("Total energy")
        plt.title("Energy History")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_radius(self) -> None:
        if len(self.positions) == 0:
            print("No position data. Run the simulation first.")
            return

        radii = np.sqrt(np.sum(self.positions**2, axis=1))
        plt.figure(figsize=(10, 4))
        plt.plot(self.times, radii, "g-", lw=1.2)
        plt.xlabel("Time")
        plt.ylabel("Radius r")
        plt.title("Radius vs Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def summary(self) -> None:
        if len(self.positions) == 0:
            print("No data. Run the simulation first.")
            return

        radii = np.sqrt(np.sum(self.positions**2, axis=1))
        print("GEM Gravity Toy v2 Summary")
        print("-------------------------")
        print(f"Depth strength : {self.depth_strength}")
        print(f"dt             : {self.dt}")
        print(f"steps          : {self.steps}")
        print(f"damping        : {self.damping}")
        print(f"Initial energy : {self.energies[0]:.6f}")
        print(f"Final energy   : {self.energies[-1]:.6f}")
        print(f"Min radius     : {np.min(radii):.6f}")
        print(f"Max radius     : {np.max(radii):.6f}")
        print(f"Classification : {self.classify_orbit()}")


if __name__ == "__main__":
    sim = GEMGravityToyV2(
        depth_strength=10.0,
        dt=0.01,
        steps=8000,
        softening=1e-3,
        damping=0.999,          # gentle damping for stable orbits
    )

    # Example: elliptical-like orbit
    sim.run(start_pos=(5.0, 0.0), start_vel=(0.0, -0.85))
    sim.summary()
    sim.plot_trajectory()
    sim.plot_energy()
    sim.plot_radius()