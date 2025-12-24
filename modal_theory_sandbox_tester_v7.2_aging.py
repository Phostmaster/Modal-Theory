import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# MT Physics Toy Model — SANDBOX TESTER (AMENDED & CONSOLIDATED)
# ------------------------------------------------------------
# 2-field lattice (phi1, phi2)
# Δθ = phi1 - phi2
#
# V_eff(Δθ) = -g_mode cos(Δθ) + K_quad * wrap(Δθ - Δ_target)^2
# Gradient stiffness: cell_lambda(x,y) couples to Laplacian term
#
# Integrator: Verlet (2nd order) with correct velocity seeding
# Laplacian/grad: periodic BC (np.roll)
#
# Optional: CoherenceShell for thrust proxy
# Diagnostics:
#   - Energies: total, kinetic, gradient (effective with cell_lambda), potential
#   - Stability: circular mean/std + lock error
#   - Thrust and mass proxy (dynamic mean(phi1*phi2))
#   - Optional "health" proxies: tumor_fraction, coherence_score, kappa, etc.
#
# Fixes vs old draft:
#   1) No out-of-scope globals inside run_full_simulation()
#   2) Single coherent Booth/healing block (no duplicates)
#   3) Verlet seeding uses initial velocities
#   4) Dynamic phi1*phi2 mean for thrust/mass (global or local)
#   5) Energy gradient term uses cell_lambda (effective energy)
# ============================================================


# -------------------------
# Helpers: angles / wrapping
# -------------------------
def wrap_angle(rad):
    """Wrap angle to (-pi, pi]."""
    return np.arctan2(np.sin(rad), np.cos(rad))


def angle_deg_0_360(rad):
    """Convert radians to degrees in [0, 360)."""
    return float(np.rad2deg(rad) % 360.0)


def circular_mean(angle_field):
    """Circular mean of angles (radians)."""
    s = float(np.mean(np.sin(angle_field)))
    c = float(np.mean(np.cos(angle_field)))
    return np.arctan2(s, c)


def circular_std(angle_field):
    """Circular standard deviation (radians). Approx sigma = sqrt(-2 ln R)."""
    s = float(np.mean(np.sin(angle_field)))
    c = float(np.mean(np.cos(angle_field)))
    R = float(np.sqrt(s * s + c * c))
    if R <= 1e-12:
        return np.pi / np.sqrt(3.0)
    return float(np.sqrt(max(0.0, -2.0 * np.log(R))))


# -------------------------
# Discrete operators (periodic)
# -------------------------
def laplacian_2d(field, dx):
    """2D Laplacian with periodic boundary conditions."""
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
        4.0 * field
    ) / (dx ** 2)


def grad_sq_periodic(field, dx):
    """|grad field|^2 with periodic BC using forward differences."""
    fx = (np.roll(field, -1, axis=1) - field) / dx
    fy = (np.roll(field, -1, axis=0) - field) / dx
    return fx * fx + fy * fy


# -------------------------
# CoherenceShell Class
# -------------------------
class CoherenceShell:
    def __init__(  # <-- double underscores before and after "init"
        self,
        x_center,
        y_center,
        radius,
        delta_theta_internal_deg,
        mass=1.0,
        grid_size=100,
        reflect_y=False,
        damp=0.9,
        v_threshold=0.05
    ):
        self.x = float(x_center)
        self.y = float(y_center)
        self.radius = float(radius)
        self.delta_theta_internal = float(np.deg2rad(delta_theta_internal_deg))
        self.mass = float(mass)
        self.vx = 0.0
        self.vy = 0.0
        self.grid_size = int(grid_size)
        self.reflect_y = bool(reflect_y)
        self.damp = float(damp)
        self.v_threshold = float(v_threshold)
    
    def _reflect_axis(self, pos, vel, min_pos, max_pos, force_component):
        if pos < min_pos:
            pos = min_pos
            if abs(vel * self.damp) < self.v_threshold:
                vel = 0.0 if force_component > 0 else -self.v_threshold
            else:
                vel = -vel * self.damp
        elif pos > max_pos:
            pos = max_pos
            if abs(vel * self.damp) < self.v_threshold:
                vel = 0.0 if force_component < 0 else self.v_threshold
            else:
                vel = -vel * self.damp
        return pos, vel

    def update(self, force_x, force_y, dt):
        ax = force_x / self.mass
        ay = force_y / self.mass
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        self.x, self.vx = self._reflect_axis(self.x, self.vx, 0.0, float(self.grid_size - 1), force_x)
        if self.reflect_y:
            self.y, self.vy = self._reflect_axis(self.y, self.vy, 0.0, float(self.grid_size - 1), force_y)
        else:
            self.y = max(0.0, min(self.y, float(self.grid_size - 1)))


# -------------------------
# Stability Guard (DT/DX)
# -------------------------
def stability_guard_dt(dx, dt, lambda_val, g_mode, K_quad,
                       safety_wave=0.45, safety_pot=0.20,
                       auto_adjust=True):
    """
    Practical stability guard for explicit Verlet-like dynamics.

    Wave/CFL-like:
      dt <= safety_wave * dx / (sqrt(2) * sqrt(lambda))

    Potential stiffness:
      k_eff ~ |g_mode| + 2|K_quad|  -> omega ~ sqrt(k_eff)
      dt <= safety_pot / omega
    """
    lam = float(max(lambda_val, 1e-12))
    dx = float(dx)
    dt = float(dt)

    c_eff = float(np.sqrt(lam))
    dt_max_wave = safety_wave * dx / (np.sqrt(2.0) * c_eff)

    k_eff = float(abs(g_mode) + 2.0 * abs(K_quad))
    omega = float(np.sqrt(max(k_eff, 1e-12)))
    dt_max_pot = safety_pot / omega

    dt_max = float(min(dt_max_wave, dt_max_pot))

    report = {
        "dt_in": dt,
        "dt_max": dt_max,
        "dt_max_wave": float(dt_max_wave),
        "dt_max_pot": float(dt_max_pot),
        "lambda": lam,
        "k_eff": k_eff,
        "omega": omega,
    }

    if dt <= dt_max:
        return dt, dt_max, report

    if not auto_adjust:
        raise ValueError(
            f"[StabilityGuard] dt={dt:.3e} exceeds dt_max={dt_max:.3e} "
            f"(wave {dt_max_wave:.3e}, pot {dt_max_pot:.3e})."
        )

    dt_new = dt_max
    report["dt_adjusted_to"] = dt_new
    return dt_new, dt_max, report


# -------------------------
# Energy diagnostics (effective)
# -------------------------
def compute_field_energy(phi1, phi2, phi1_prev, phi2_prev,
                         g_mode, cell_lambda, K_quad, target_delta_0_rad, dx, dt):
    """
    Toy energy (averaged per cell):
      E_kin  = 0.5 <phi1_dot^2 + phi2_dot^2>
      E_grad = 0.5 < cell_lambda * |grad(Δθ)|^2 >
      E_pot  = < -g cos(Δθ) + K * wrap(Δθ-Δ*)^2 >
    """
    dtheta = phi1 - phi2
    v1 = (phi1 - phi1_prev) / dt
    v2 = (phi2 - phi2_prev) / dt
    E_kin = 0.5 * float(np.mean(v1 * v1 + v2 * v2))

    grad2 = grad_sq_periodic(dtheta, dx)
    E_grad = 0.5 * float(np.mean(cell_lambda * grad2))

    wrapped = wrap_angle(dtheta - target_delta_0_rad)
    V_pot_local = (-float(g_mode) * np.cos(dtheta)) + (float(K_quad) * wrapped * wrapped)
    E_pot = float(np.mean(V_pot_local))

    E_total = E_kin + E_grad + E_pot
    return E_total, E_kin, E_grad, E_pot


# -------------------------
# Mass Proxy
# -------------------------
def calculate_mass_proxy(g_mode, phi_product_mean_abs):
    """
    m_f ≈ g_mode * |<φ1 φ2>| * 32.58 (toy proxy)
    """
    return float(g_mode) * float(phi_product_mean_abs) * 32.58


# -------------------------
# Sampling helper for local thrust coupling
# -------------------------
def mean_phi_product_local(phi1, phi2, x, y, radius, grid_size):
    """
    Compute mean(phi1*phi2) in a disk around (x,y) with periodic wrap.
    x,y,radius in grid index units.
    """
    cx = int(round(x)) % grid_size
    cy = int(round(y)) % grid_size
    r = int(max(1, round(radius)))

    # build local window with periodic wrap by indexing
    xs = (np.arange(cx - r, cx + r + 1) % grid_size).astype(int)
    ys = (np.arange(cy - r, cy + r + 1) % grid_size).astype(int)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    dx2 = (xx - cx)
    dy2 = (yy - cy)
    mask = (dx2 * dx2 + dy2 * dy2) <= (r * r)

    patch_prod = (phi1[yy, xx] * phi2[yy, xx])[mask]
    if patch_prod.size == 0:
        return float(np.mean(phi1 * phi2))
    return float(np.mean(patch_prod))


# -------------------------
# Simulation (AUTO-STOP + cleaned booth/tumor)
# -------------------------
def run_full_simulation(
    initial_phi1,
    initial_phi2,
    initial_phi1_vel,
    initial_phi2_vel,
    g_mode,
    initial_lambda_val,
    K_quad,
    target_delta_0_rad,
    dx,
    dt,
    iterations,
    shell_obj=None,
    thrust_mode="global",      # "global" or "local"
    # Tumor/Booth optional
    tumor_center=None,
    tumor_radius=None,
    booth_center=None,
    booth_radius=None,
    # Tumor dynamics
    tumor_kappa_threshold=5e-4,
    tumor_decay=0.995,
    lambda_floor_frac=0.05,    # floor as fraction of initial_lambda_val
    # Booth dynamics
    booth_every=500,
    booth_nudge_strength=0.15, # pulls Δθ toward target in booth region
    booth_heal_strength=0.25,  # heals lambda in booth region
    # Global recovery
    lambda_recover_rate=0.001, # per step pull toward initial_lambda_val
    # Logging/monitoring
    log_every=1000,
    monitor_every=200,
    store_series=True,
    # Auto-stop
    enable_autostop=True,
    err_tol_deg=0.25,
    std_tol_deg=0.50,
    stable_windows_required=12,
    # Safety abort
    abort_energy_abs=1e12
):
    phi1 = initial_phi1.copy()
    phi2 = initial_phi2.copy()

    # Correct Verlet seeding with initial velocities:
    # phi(t - dt) = phi(t) - v(t)*dt
    phi1_prev = phi1 - initial_phi1_vel * dt
    phi2_prev = phi2 - initial_phi2_vel * dt

    grid_size = phi1.shape[0]
    if phi1.shape != (grid_size, grid_size) or phi2.shape != (grid_size, grid_size):
        raise ValueError("Fields must be square 2D arrays with same shape.")

    # Dynamic lambda field (stiffness)
    cell_lambda = np.ones((grid_size, grid_size), dtype=np.float64) * float(initial_lambda_val)
    lambda_floor = float(initial_lambda_val) * float(lambda_floor_frac)

    # Tumor region mask
    in_tumor = np.full((grid_size, grid_size), False, dtype=bool)
    if tumor_center is not None and tumor_radius is not None:
        i_grid, j_grid = np.ogrid[:grid_size, :grid_size]
        dist = np.sqrt((i_grid - tumor_center[0])*2 + (j_grid - tumor_center[1])*2)
        in_tumor = dist < float(tumor_radius)
        # seed weaker stiffness in tumor region
        cell_lambda[in_tumor] = np.maximum(lambda_floor, cell_lambda[in_tumor] * 0.3)

    # Booth region mask
    in_booth = np.full((grid_size, grid_size), False, dtype=bool)
    if booth_center is not None and booth_radius is not None:
        i_grid, j_grid = np.ogrid[:grid_size, :grid_size]
        dist = np.sqrt((i_grid - booth_center[0])*2 + (j_grid - booth_center[1])*2)
        in_booth = dist < float(booth_radius)

    series = None
    if store_series:
        series = {
            "t": [],
            "E_total": [], "E_kin": [], "E_grad": [], "E_pot": [],
            "mean_dtheta_deg": [], "std_dtheta_deg": [], "lock_error_deg": [],
            "kappa": [], "coherence_score": [], "tumor_fraction": [],
            "phi_prod_mean": [], "thrust": [], "mass_proxy": [],
            "shell_x": [], "shell_y": [], "shell_vx": [], "shell_KE": [], "shell_energy_key": [],
            "stable_count": []
        }

    shell_trajectory = []
    if shell_obj is not None:
        shell_trajectory.append((shell_obj.x, shell_obj.y, shell_obj.vx))

    eps_key = 1e-12
    stable_count = 0
    stop_step = None

    # Precompute mean_c for baryo-style derivative if needed later (not used now)
    # prev_mean_c = circular_mean(phi1 - phi2)

    for step in range(iterations):
        dtheta = phi1 - phi2
        lap = laplacian_2d(dtheta, dx)

        wrapped_err = wrap_angle(dtheta - target_delta_0_rad)

        # Base force on Δθ from potential:
        # d/d(Δθ) [-g cos(Δθ)] = +g sin(Δθ), so force (negative gradient) is -g sin(Δθ)
        # quadratic term K*wrap^2 -> derivative 2K*wrap, so force is -2K*wrap
        force_delta = -(float(g_mode) * np.sin(dtheta) + 2.0 * float(K_quad) * wrapped_err)

        # --- Tumor dynamics: weaken lambda if global decoherence high ---
        kappa_now = float(np.mean(grad_sq_periodic(dtheta, dx)))
        if in_tumor.any() and (kappa_now > float(tumor_kappa_threshold)):
            cell_lambda[in_tumor] = np.maximum(lambda_floor, cell_lambda[in_tumor] * float(tumor_decay))

        # --- Booth dynamics: single coherent nudge + heal ---
        if in_booth.any() and (booth_every is not None) and (booth_every > 0) and (step % int(booth_every) == 0):
            # Nudge: pull Δθ toward target inside booth
            # Use wrapped error so the pull is shortest-angle
            local_pull = -float(booth_nudge_strength) * wrap_angle(dtheta - target_delta_0_rad)
            force_delta[in_booth] += local_pull[in_booth]

            # Heal lambda toward initial in booth region
            # Simple saturating heal: move upward by fraction of gap
            gap = float(initial_lambda_val) - cell_lambda[in_booth]
            cell_lambda[in_booth] = np.minimum(float(initial_lambda_val), cell_lambda[in_booth] + float(booth_heal_strength) * gap)

        # --- Global recovery (self-heal) ---
        # Pull lambda toward initial slightly each step
        cell_lambda += float(lambda_recover_rate) * (float(initial_lambda_val) - cell_lambda)
        cell_lambda = np.maximum(lambda_floor, np.minimum(float(initial_lambda_val), cell_lambda))

        # Field accelerations
        accel1 = force_delta - cell_lambda * lap
        accel2 = -force_delta + cell_lambda * lap

        # Verlet update
        phi1_new = 2.0 * phi1 - phi1_prev + (dt * dt) * accel1
        phi2_new = 2.0 * phi2 - phi2_prev + (dt * dt) * accel2

        if np.any(np.isnan(phi1_new)) or np.any(np.isnan(phi2_new)):
            print(f"[ABORT] NaN detected at step {step}.")
            stop_step = step
            break

        # advance
        phi1_prev, phi2_prev = phi1, phi2
        phi1, phi2 = phi1_new, phi2_new

        # Thrust / shell update (dynamic phi product)
        thrust = 0.0
        phi_prod_mean = float(np.mean(phi1 * phi2))
        phi_prod_mean_abs = float(abs(phi_prod_mean))
        mass_proxy_step = calculate_mass_proxy(g_mode, phi_prod_mean_abs)

        shell_ke = 0.0
        shell_energy_key = 0.0

        if shell_obj is not None:
            if thrust_mode == "local":
                phi_prod_used = mean_phi_product_local(phi1, phi2, shell_obj.x, shell_obj.y, shell_obj.radius, grid_size)
            else:
                phi_prod_used = phi_prod_mean

            thrust = float(g_mode) * float(phi_prod_used) * float(np.sin(shell_obj.delta_theta_internal))
            shell_obj.update(thrust, 0.0, dt)

            shell_ke = 0.5 * shell_obj.mass * (shell_obj.vx ** 2 + shell_obj.vy ** 2)
            shell_trajectory.append((shell_obj.x, shell_obj.y, shell_obj.vx))

        # Monitoring / diagnostics
        if monitor_every and (step % int(monitor_every) == 0):
            dtheta_now = phi1 - phi2
            mean_c = circular_mean(dtheta_now)
            std_c = circular_std(dtheta_now)
            lock_err = wrap_angle(mean_c - target_delta_0_rad)

            E_total, E_kin, E_grad, E_pot = compute_field_energy(
                phi1, phi2, phi1_prev, phi2_prev,
                g_mode, cell_lambda, K_quad, target_delta_0_rad, dx, dt
            )

            if abs(E_total) > float(abort_energy_abs):
                print(f"[ABORT] Energy blew up: |E_total|={abs(E_total):.3e} at step {step}.")
                stop_step = step
                break

            err_deg = float(np.rad2deg(lock_err))
            std_deg = float(np.rad2deg(std_c))

            coherence_score = float(np.mean(cell_lambda) / float(initial_lambda_val))
            tumor_fraction = float(np.mean(cell_lambda[in_tumor] < (0.5 * float(initial_lambda_val)))) if in_tumor.any() else 0.0

            if shell_obj is not None:
                shell_energy_key = shell_ke / (abs(E_total) + eps_key)

            # Auto-stop
            if enable_autostop:
                if (abs(err_deg) <= float(err_tol_deg)) and (std_deg <= float(std_tol_deg)):
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= int(stable_windows_required):
                    print(
                        f"[AUTO-STOP] Stable lock achieved for {stable_windows_required} consecutive monitors "
                        f"(err_tol={err_tol_deg}°, std_tol={std_tol_deg}°) at step {step}."
                    )
                    stop_step = step
                    # store final monitor point before breaking
                    if store_series:
                        t = step * dt
                        series["t"].append(t)
                        series["E_total"].append(E_total)
                        series["E_kin"].append(E_kin)
                        series["E_grad"].append(E_grad)
                        series["E_pot"].append(E_pot)
                        series["mean_dtheta_deg"].append(angle_deg_0_360(mean_c))
                        series["std_dtheta_deg"].append(std_deg)
                        series["lock_error_deg"].append(err_deg)
                        series["kappa"].append(kappa_now)
                        series["coherence_score"].append(coherence_score)
                        series["tumor_fraction"].append(tumor_fraction)
                        series["phi_prod_mean"].append(phi_prod_mean)
                        series["thrust"].append(thrust)
                        series["mass_proxy"].append(mass_proxy_step)
                        series["stable_count"].append(stable_count)
                        if shell_obj is not None:
                            series["shell_x"].append(shell_obj.x)
                            series["shell_y"].append(shell_obj.y)
                            series["shell_vx"].append(shell_obj.vx)
                            series["shell_KE"].append(shell_ke)
                            series["shell_energy_key"].append(shell_energy_key)
                        else:
                            series["shell_x"].append(np.nan)
                            series["shell_y"].append(np.nan)
                            series["shell_vx"].append(np.nan)
                            series["shell_KE"].append(0.0)
                            series["shell_energy_key"].append(0.0)
                    break

            # Store series
            if store_series:
                t = step * dt
                series["t"].append(t)
                series["E_total"].append(E_total)
                series["E_kin"].append(E_kin)
                series["E_grad"].append(E_grad)
                series["E_pot"].append(E_pot)
                series["mean_dtheta_deg"].append(angle_deg_0_360(mean_c))
                series["std_dtheta_deg"].append(std_deg)
                series["lock_error_deg"].append(err_deg)
                series["kappa"].append(kappa_now)
                series["coherence_score"].append(coherence_score)
                series["tumor_fraction"].append(tumor_fraction)
                series["phi_prod_mean"].append(phi_prod_mean)
                series["thrust"].append(thrust)
                series["mass_proxy"].append(mass_proxy_step)
                series["stable_count"].append(stable_count)

                if shell_obj is not None:
                    series["shell_x"].append(shell_obj.x)
                    series["shell_y"].append(shell_obj.y)
                    series["shell_vx"].append(shell_obj.vx)
                    series["shell_KE"].append(shell_ke)
                    series["shell_energy_key"].append(shell_energy_key)
                else:
                    series["shell_x"].append(np.nan)
                    series["shell_y"].append(np.nan)
                    series["shell_vx"].append(np.nan)
                    series["shell_KE"].append(0.0)
                    series["shell_energy_key"].append(0.0)

            if log_every and (step % int(log_every) == 0):
                print(
                    f"Step {step:6d} | t={step*dt:8.3f}s | "
                    f"<Δθ>={angle_deg_0_360(mean_c):7.2f}° | err={err_deg:+8.3f}° | std={std_deg:7.3f}° | "
                    f"kappa={kappa_now:.2e} | coh={coherence_score:.3f} | tumor_frac={tumor_fraction:.3f} | "
                    f"E={E_total:.6e} (kin {E_kin:.2e}, grad {E_grad:.2e}, pot {E_pot:.2e}) | "
                    f"phi12={phi_prod_mean:+.3e} | thrust={thrust:+.3e} | m_proxy={mass_proxy_step:.3e} | "
                    f"stable={stable_count:02d}/{stable_windows_required if enable_autostop else 0}"
                )

        # If autostop triggered, break out of loop cleanly
        if stop_step is not None and stop_step == step:
            break

    return phi1, phi2, shell_obj, shell_trajectory, series, stop_step


# -------------------------
# MAIN — SANDBOX TESTER
# -------------------------
if __name__ == "__main__":
    # Core physics parameters
    g_mode = 0.085
    initial_lambda_val = 0.0112
    K_quad = 5.0
    target_delta_0_rad = np.deg2rad(255.0)
    # Spatial/time discretization
    dx = 0.1
    dt = 0.001
    # Stability guard
    dt_actual, dt_max_report, guard_report = stability_guard_dt(
        dx=dx, dt=dt, lambda_val=initial_lambda_val, g_mode=g_mode, K_quad=K_quad,
        safety_wave=0.45, safety_pot=0.20, auto_adjust=True
    )
    if "dt_adjusted_to" in guard_report:
        print(
            "[StabilityGuard] dt was reduced for safety:\n"
            f" dt_in = {guard_report['dt_in']:.6e}\n"
            f" dt_max = {guard_report['dt_max']:.6e}\n"
            f" dt_waveMax = {guard_report['dt_max_wave']:.6e}\n"
            f" dt_potMax = {guard_report['dt_max_pot']:.6e}\n"
            f" dt_used = {guard_report['dt_adjusted_to']:.6e}\n"
        )
        dt = dt_actual
    else:
        print(
            "[StabilityGuard] dt ok:\n"
            f" dt_used = {dt:.6e}\n"
            f" dt_max = {dt_max_report:.6e}\n"
            f" dt_waveMax = {guard_report['dt_max_wave']:.6e}\n"
            f" dt_potMax = {guard_report['dt_max_pot']:.6e}\n"
        )
    amplitude_avg_fields = 25.0
    grid_size = 200
    steps = 15000
    # Tumor / Booth parameters (optional)
    tumor_center = (grid_size // 2, grid_size // 2)
    tumor_radius = 30
    booth_center = (grid_size // 4, grid_size // 4)
    booth_radius = 50
    # Initialize fields near target lock
    phi1_base = amplitude_avg_fields + target_delta_0_rad / 2.0
    phi2_base = amplitude_avg_fields - target_delta_0_rad / 2.0
    phi1_initial = np.ones((grid_size, grid_size), dtype=np.float64) * phi1_base
    phi2_initial = np.ones((grid_size, grid_size), dtype=np.float64) * phi2_base
    # Small perturbation seed
    phi1_initial[50, 50] += 0.1
    phi2_initial[50, 50] -= 0.1
    # Initial velocities
    phi1_vel_initial = np.zeros((grid_size, grid_size), dtype=np.float64)
    phi2_vel_initial = np.zeros((grid_size, grid_size), dtype=np.float64)
    # Shell proxy configuration
    shell_for_thrust = CoherenceShell(
        x_center=137.19,
        y_center=50.00,
        radius=5.0,
        delta_theta_internal_deg=90.0,
        mass=1.0,
        grid_size=grid_size,
        reflect_y=False
    )
    # Run simulation
    phi1_final, phi2_final, shell_final, trajectory, series, stop_step = run_full_simulation(
        initial_phi1=phi1_initial,
        initial_phi2=phi2_initial,
        initial_phi1_vel=phi1_vel_initial,
        initial_phi2_vel=phi2_vel_initial,
        g_mode=g_mode,
        initial_lambda_val=initial_lambda_val,
        K_quad=K_quad,
        target_delta_0_rad=target_delta_0_rad,
        dx=dx,
        dt=dt,
        iterations=steps,
        shell_obj=shell_for_thrust,
        thrust_mode="global",  # or "local" for shell-local sampling
        tumor_center=tumor_center,
        tumor_radius=tumor_radius,
        booth_center=booth_center,
        booth_radius=booth_radius,
        tumor_decay=0.995,
        booth_every=500,
        booth_nudge_strength=0.15,
        booth_heal_strength=0.25,
        lambda_recover_rate=0.001,
        log_every=1000,
        monitor_every=200,
        store_series=True,
        enable_autostop=True
    )
    print("--- Final Simulation Results ---")
    if stop_step is not None:
        print(f"AUTO-STOP or abort at step {stop_step}")
    else:
        print(f"Reached full iterations: {steps} steps.")
    # Print key finals
    final_dtheta = phi1_final - phi2_final
    mean_c = circular_mean(final_dtheta)
    std_c = circular_std(final_dtheta)
    lock_err = wrap_angle(mean_c - target_delta_0_rad)
    print(f"Final circular mean Δθ: {angle_deg_0_360(mean_c):.3f} degrees")
    print(f"Final circular std Δθ: {np.rad2deg(std_c):.6f} degrees")
    print(f"Final lock error (mean-target): {np.rad2deg(lock_err):+8.6f} degrees")
    phi_prod_mean = np.mean(phi1_final * phi2_final)
    print(f"Mean phi1*phi2: {phi_prod_mean:.6f}")
    thrust_final = g_mode * phi_prod_mean * np.sin(shell_final.delta_theta_internal) if shell_final else 0.0
    print(f"Simulated Thrust (proxy magnitude): {abs(thrust_final):.6f} toy units")
    mass_proxy = calculate_mass_proxy(g_mode, abs(phi_prod_mean))
    print(f"Final Mass Proxy: {mass_proxy:.6f} toy units")
    if shell_final:
        print(f"Shell final position: ({shell_final.x:.2f}, {shell_final.y:.2f})")
        print(f"Shell final velocity: ({shell_final.vx:.6f}, {shell_final.vy:.6f})")