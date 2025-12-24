import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# MT Physics Toy Model — SANDBOX TESTER (FULL)
# ------------------------------------------------------------
# 2-field lattice (phi1, phi2)
# Δθ = phi1 - phi2
# V_eff(Δθ) = -g_mode cos(Δθ) + K_quad * wrap(Δθ - Δ_target)^2
# + gradient stiffness term ~ (lambda_val/2) |∇Δθ|^2
#
# Integrator: Verlet (second order)
# Laplacian: periodic
# Optional: CoherenceShell thrust proxy
#
# Diagnostics:
#   - Energy: E_total, E_kin_fields, E_grad, E_pot
#   - Shell KE + "Shell Energy Key"
#   - Stability: circular mean/std + lock error
#
# Added:
#   - Stability Guard: dt vs dx vs lambda + potential stiffness
#   - Auto-stop: stop early when lock achieved and stable
#   - Comprehensive time series plotting
#   - Enhanced CoherenceShell for dynamic interaction
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
    """
    Circular standard deviation (radians).
    Approx: sigma = sqrt(-2 ln R), where R = |<exp(iθ)>|.
    """
    s = float(np.mean(np.sin(angle_field)))
    c = float(np.mean(np.cos(angle_field)))
    R = float(np.sqrt(s * s + c * c))
    if R <= 1e-12:  # Avoid log(0)
        return np.pi / np.sqrt(3.0)  # Approx. for uniform distribution
    return float(np.sqrt(max(0.0, -2.0 * np.log(R)))) # max(0.0, ...) for numerical safety


# -------------------------
# Discrete operators (periodic)
# -------------------------
def laplacian_2d(field, dx):
    """2D Laplacian using finite differences with periodic boundary conditions."""
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
    def __init__(  # Corrected from _init_
        self,
        x_center,
        y_center,
        radius,
        delta_theta_internal,
        mass=1.0,
        grid_size=100,
        reflect_y=False,  # Typically only X boundaries for thrust tests
        damp=0.9,
        v_threshold=0.1
    ):
        self.x = float(x_center)
        self.y = float(y_center)
        self.radius = float(radius)
        self.delta_theta_internal = float(np.deg2rad(delta_theta_internal))
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
            if abs(vel * self.damp) < self.v_threshold: # Check dampened velocity
                vel = 0.0 if force_component > 0 else -self.v_threshold/2.0 # Stop if force pushes away, else tiny push into wall
            else:
                vel = -vel * self.damp
        elif pos > max_pos:
            pos = max_pos
            if abs(vel * self.damp) < self.v_threshold:
                vel = 0.0 if force_component < 0 else self.v_threshold/2.0
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

        self.x, self.vx = self._reflect_axis(self.x, self.vx, 0.0, float(self.grid_size), force_x)
        if self.reflect_y:
            self.y, self.vy = self._reflect_axis(self.y, self.vy, 0.0, float(self.grid_size), force_y)
        else: # Clamp Y if not reflecting
            self.y = max(0.0, min(self.y, float(self.grid_size)))


# -------------------------
# Stability Guard (DT/DX)
# -------------------------
def stability_guard_dt(dx, dt, lambda_val, g_mode, K_quad,
                       safety_wave=0.45, safety_pot=0.20,
                       auto_adjust=True):
    """
    Practical stability guard for explicit Verlet-like dynamics.

    1) Wave/CFL-like guard from gradient term:
       Δθ_tt ~ lambda * ∇^2 Δθ  -> effective wave speed ~ sqrt(lambda)
       A safe rule in 2D: dt <= safety_wave * dx / (sqrt(2) * sqrt(lambda))

    2) Potential stiffness guard:
       Near a minimum, V ~ (1/2)k_eff(δθ)^2, so ω ~ sqrt(k_eff).
       Here k_eff ~ (g_mode * |cos(θ*)| + 2*K_quad) (order-of-magnitude bound).
       Safe: dt <= safety_pot / ω

    Returns: (dt_used, dt_max, report_dict)
    """
    lam = float(max(lambda_val, 1e-12)) # Ensure lambda is not zero
    dx = float(dx)
    dt = float(dt)

    # (A) CFL-like for wave propagation
    c_eff = float(np.sqrt(lam))
    dt_max_wave = safety_wave * dx / (np.sqrt(2.0) * c_eff)

    # (B) Potential stiffness (oscillations around minimum)
    # Use conservative bound: cos term magnitude <= 1
    # This k_eff is for the effective spring constant
    k_eff = float(abs(g_mode) + 2.0 * abs(K_quad))
    omega = float(np.sqrt(max(k_eff, 1e-12))) # Ensure omega is not zero
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
            f"(wave {dt_max_wave:.3e}, pot {dt_max_pot:.3e}). "
            f"Reduce dt or increase dx / reduce stiffness."
        )

    # Auto adjust
    dt_new = dt_max
    report["dt_adjusted_to"] = dt_new
    return dt_new, dt_max, report


# -------------------------
# Energy diagnostics for field
# -------------------------
def compute_field_energy(phi1, phi2, phi1_prev, phi2_prev,
                         g_mode, lambda_val, K_quad, target_delta_0_rad, dx, dt):
    """
    Energy density (toy) per lattice cell averaged over grid:
      E_kin = 0.5 <phi1_dot^2 + phi2_dot^2>
      E_grad = 0.5*lambda <|grad(Δθ)|^2>
      E_pot = < -g cos(Δθ) + K * wrap(Δθ-Δ*)^2 >
      E_total = E_kin + E_grad + E_pot
    """
    dtheta = phi1 - phi2
    v1 = (phi1 - phi1_prev) / dt
    v2 = (phi2 - phi2_prev) / dt
    E_kin = 0.5 * float(np.mean(v1 * v1 + v2 * v2))

    E_grad = 0.5 * float(lambda_val) * float(np.mean(grad_sq_periodic(dtheta, dx)))

    wrapped = wrap_angle(dtheta - target_delta_0_rad)
    V_pot_local = (-float(g_mode) * np.cos(dtheta)) + (float(K_quad) * wrapped * wrapped)
    E_pot = float(np.mean(V_pot_local))

    E_total = E_kin + E_grad + E_pot
    return E_total, E_kin, E_grad, E_pot


# -------------------------
# Main Simulation Loop
# -------------------------
def run_full_simulation(
    initial_phi1,
    initial_phi2,
    initial_phi1_vel, # These are passed but not used directly in Verlet init; kept for signature consistency
    initial_phi2_vel, # These are passed but not used directly in Verlet init; kept for signature consistency
    g_mode,
    lambda_val,
    K_quad,
    target_delta_0_rad,
    dx,
    dt,
    iterations,
    shell_obj=None,
    # logging/monitoring
    log_every=1000,
    monitor_every=200,
    store_series=True,
    # auto-stop settings
    enable_autostop=True,
    err_tol_deg=0.25,
    std_tol_deg=0.50,
    stable_windows_required=12,
    # safety abort
    abort_energy_abs=1e12
):
    global cell_lambda  # Allow modification of main's cell_lambda
    phi1 = initial_phi1.copy()
    phi2 = initial_phi2.copy()

    # Verlet seed: phi_prev = phi - v*dt (assuming initial_phi_vel is used to calculate first phi_prev)
    # However, for initial vel=0, phi_prev = phi is sufficient for the first step
    phi1_prev_verlet = phi1.copy() 
    phi2_prev_verlet = phi2.copy() 

    # History tracking for shell
    shell_trajectory = []
    if shell_obj is not None:
        # Initial point for trajectory with Vx
        shell_trajectory.append((shell_obj.x, shell_obj.y, shell_obj.vx)) 

    series = None
    if store_series:
        series = {
            "t": [],
            "E_total": [], "E_kin": [], "E_grad": [], "E_pot": [],
            "shell_KE": [], "shell_energy_key": [],
            "mean_dtheta_deg": [], "std_dtheta_deg": [], "lock_error_deg": [],
            "thrust": []
        }

    eps_key = 1e-12 # Small epsilon to prevent division by zero
    stable_count = 0
    stop_step = None

    # Calculate phi_product_mean_global once, assuming field magnitude stabilizes quickly
    # This is used for thrust calculation.
    phi_product_mean_global = float(np.mean(initial_phi1 * initial_phi2))

    for step in range(iterations):
    # Local Δθ per cell
        dtheta = phi1 - phi2  # Still global fields, but we'll modulate forces locally
        lap = laplacian_2d(dtheta, dx)

        # wrapped delta around target
        wrapped = wrap_angle(dtheta - target_delta_0_rad)

        # Force on Δθ from V(Δθ) = -g cosΔθ + K*wrapped^2 (dV/dΔθ = g sinΔθ + 2K*wrapped)
        # Accel for Phi1 gets +F_Delta_0, Accel for Phi2 gets -F_Delta_0
        force_delta = -(g_mode * np.sin(dtheta) + 2.0 * K_quad * wrapped)

        # Gradient stiffness -> -lambda * Laplacian(Δθ)
        accel1 = force_delta - lambda_val * lap
        accel2 = -force_delta + lambda_val * lap

        # Vectorized Health Coherence: Cancer proxy - weakened λ causes phase scatter
        mean_kappa = np.mean(grad_sq_periodic(dtheta, dx))
        if mean_kappa > 5e-4:
            cell_lambda[in_tumor] *= 0.99  # Progressive weakening in tumor zone

        # Mild natural healing (no booth)
        cell_lambda += 0.001 * (1.0 - cell_lambda)  # Slow recovery to full λ

        # Local force modulation (vectorized)
        accel1 *= cell_lambda
        accel2 *= cell_lambda

        # Health metrics
        tumor_fraction = np.mean(cell_lambda < 0.5)  # Fraction "cancerous"
        coherence_score = np.mean(cell_lambda)  # Overall body health
        series["tumor_fraction"] = series.get("tumor_fraction", []) + [tumor_fraction]
        series["coherence_score"] = series.get("coherence_score", []) + [coherence_score]

        
        # Verlet update
        phi1_new = 2.0 * phi1 - phi1_prev_verlet + (dt * dt) * accel1
        phi2_new = 2.0 * phi2 - phi2_prev_verlet + (dt * dt) * accel2

        if np.any(np.isnan(phi1_new)) or np.any(np.isnan(phi2_new)):
            print(f"[ABORT] NaN detected at step {step}.")
            stop_step = step
            break

        # Advance fields
        phi1_prev_verlet, phi2_prev_verlet = phi1, phi2 # Current becomes previous
        phi1, phi2 = phi1_new, phi2_new # New becomes current

        # Shell proxy update
        thrust = 0.0
        shell_ke = 0.0
        shell_energy_key = 0.0

        if shell_obj is not None:
            thrust = phi_product_mean_global * g_mode * np.sin(shell_obj.delta_theta_internal)
            shell_obj.update(thrust, 0.0, dt)

            shell_ke = 0.5 * shell_obj.mass * (shell_obj.vx ** 2 + shell_obj.vy ** 2)
            shell_trajectory.append((shell_obj.x, shell_obj.y, shell_obj.vx)) # Store Vx for velocity plot

        # Monitor block (for stability and energy tracking)
        if monitor_every and (step % monitor_every == 0):
            dtheta_now = phi1 - phi2
            mean_c = circular_mean(dtheta_now)
            std_c = circular_std(dtheta_now)
            lock_err = wrap_angle(mean_c - target_delta_0_rad)

            # Pass correct previous values for field velocities
            E_total, E_kin, E_grad, E_pot = compute_field_energy(
                phi1, phi2, phi1_prev_verlet, phi2_prev_verlet, 
                g_mode, lambda_val, K_quad, target_delta_0_rad, dx, dt
            )

            if abs(E_total) > abort_energy_abs:
                print(f"[ABORT] Energy blew up: |E_total|={abs(E_total):.3e} at step {step}.")
                stop_step = step
                break

            shell_energy_key = shell_ke / (abs(E_total) + eps_key)

            err_deg = float(np.rad2deg(lock_err))
            std_deg = float(np.rad2deg(std_c))

            # CP Asymmetry Proxy (direct MT echo)
            mean_deg = angle_deg_0_360(mean_c)
            mean_rad = np.deg2rad(mean_deg)
            cp_asym = np.cos(mean_rad)  # Core cos(<Δθ>) → -0.2588 at 255°

            # Kappa (decoherence)
            kappa = float(np.mean(grad_sq_periodic(dtheta_now, dx)))

            # Baryogenesis proxy recalc (for log/print)
            if step > 0:
                dtheta_dt = (mean_c - prev_mean_c) / (monitor_every * dt)
            else:
                dtheta_dt = 0.0
            chiral_bias = np.sin(mean_c - target_delta_0_rad)
            baryo_rate = dtheta_dt * chiral_bias * kappa
            if step == 0:
                cum_baryo = 0.0
            else:
                cum_baryo = series.get("cum_baryo", [0.0])[-1] + baryo_rate * (monitor_every * dt)
            prev_mean_c = mean_c  # Remember for next dΔθ/dt
            # Store series
            if store_series:
                t = step * dt
                series["t"].append(t)
                series["E_total"].append(E_total)
                series["E_kin"].append(E_kin)
                series["E_grad"].append(E_grad)
                series["E_pot"].append(E_pot)
                series["shell_KE"].append(shell_ke)
                series["shell_energy_key"].append(shell_energy_key)
                series["mean_dtheta_deg"].append(angle_deg_0_360(mean_c))
                series["std_dtheta_deg"].append(std_deg)
                series["lock_error_deg"].append(err_deg)
                series["thrust"].append(thrust)

            # Auto-stop logic
            if enable_autostop:
                if (abs(err_deg) <= err_tol_deg) and (std_deg <= std_tol_deg):
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= stable_windows_required:
                    print(
                        f"[AUTO-STOP] Stable lock achieved for {stable_windows_required} consecutive monitors "
                        f"(err_tol={err_tol_deg}°, std_tol={std_tol_deg}°) at step {step}."
                    )
                    stop_step = step
                    break

            # Print summary
            if log_every and (step % log_every == 0):
                print(
                    f"Step {step:6d} | t={step*dt:8.3f}s | "
                    f"<Δθ>={angle_deg_0_360(mean_c):7.2f}° | "
                    f"err={err_deg:+8.3f}° | std={std_deg:7.3f}° | "
                    f"CP={cp_asym:+.5f} | kappa={kappa:.2e} | "
                    f"baryo_rate={baryo_rate:.2e} | cumη={cum_baryo:.2e} | "
                    f"tumor_frac={tumor_fraction:.3f} | coh_score={coherence_score:.3f} | "  # <--- THIS LINE
                    f"E={E_total:.6e} (kin {E_kin:.2e}, grad {E_grad:.2e}, pot {E_pot:.2e}) | "
                    f"shellKE={shell_ke:.3e} | key={shell_energy_key:.3e} | thrust={thrust:.3e} | "
                    f"stable_count={stable_count:02d}/{stable_windows_required if enable_autostop else 0}"
                )
    
    # If auto-stop occurred, truncate series at stop_step
    if stop_step is not None and store_series:
        # Determine the number of stored monitor points
        num_monitor_points = (stop_step // monitor_every) + 1
        for key in series:
            series[key] = series[key][:num_monitor_points]


    return phi1, phi2, shell_obj, shell_trajectory, series, stop_step


# -------------------------
# MAIN — SANDBOX TESTER
# -------------------------
if __name__ == "__main__": # Corrected from _name_
    # Core physics parameters
    g_mode = 0.085
    lambda_val = 0.0112
    K_quad = 5.0
    target_delta_0_rad = np.deg2rad(255.0)

    # Spatial/time discretization
    dx = 0.1
    dt = 0.001

    # Stability guard (auto-adjust dt if needed)
    dt_actual, dt_max_report, guard_report = stability_guard_dt(
        dx=dx, dt=dt, lambda_val=lambda_val, g_mode=g_mode, K_quad=K_quad,
        safety_wave=0.45, safety_pot=0.20,
        auto_adjust=True
    )
    if "dt_adjusted_to" in guard_report:
        print(
            "[StabilityGuard] dt was reduced for safety:\n"
            f"  dt_in      = {guard_report['dt_in']:.6e}\n"
            f"  dt_max     = {guard_report['dt_max']:.6e}\n"
            f"  dt_waveMax = {guard_report['dt_max_wave']:.6e}\n"
            f"  dt_potMax  = {guard_report['dt_max_pot']:.6e}\n"
            f"  dt_used    = {guard_report['dt_adjusted_to']:.6e}\n"
        )
        dt = dt_actual # Use the adjusted dt
    else:
        print(
            "[StabilityGuard] dt ok:\n"
            f"  dt_used    = {dt:.6e}\n"
            f"  dt_max     = {dt_max_report:.6e}\n"
            f"  dt_waveMax = {guard_report['dt_max_wave']:.6e}\n"
            f"  dt_potMax  = {guard_report['dt_max_pot']:.6e}\n"
        )

    amplitude_avg_fields = 25.0
    grid_size = 200
    steps = 15000  

    # Health Coherence parameters
    tumor_center = (grid_size // 2, grid_size // 2)  # Center tumor
    tumor_radius = 30
    booth_center = (grid_size // 4, grid_size // 4)  # Booth position
    booth_radius = 50

    # Health Coherence init
    cell_lambda = np.ones((grid_size, grid_size)) * lambda_val  # Healthy
    i_grid, j_grid = np.ogrid[:grid_size, :grid_size]
    dist_tumor = np.sqrt((i_grid - tumor_center[0])**2 + (j_grid - tumor_center[1])**2)
    in_tumor = dist_tumor < tumor_radius
    cell_lambda[in_tumor] *= 0.3  # Tumor seed
    booth_center = (grid_size // 4, grid_size // 4)  # Example
    booth_radius = 50

    # Initialize fields near target lock
    phi1_base_init = amplitude_avg_fields + target_delta_0_rad / 2.0
    phi2_base_init = amplitude_avg_fields - target_delta_0_rad / 2.0

    phi1_initial = np.ones((grid_size, grid_size), dtype=np.float64) * phi1_base_init
    phi2_initial = np.ones((grid_size, grid_size), dtype=np.float64) * phi2_base_init

    # Small perturbation seed
    phi1_initial[50, 50] += 0.1
    phi2_initial[50, 50] -= 0.1

    phi1_vel_initial = np.zeros((grid_size, grid_size), dtype=np.float64)
    phi2_vel_initial = np.zeros((grid_size, grid_size), dtype=np.float64)

    # Shell proxy configuration
    shell_for_thrust = CoherenceShell(
        x_center=100.0,
        y_center=50.0,
        radius=5.0,
        delta_theta_internal=90.0, # Max thrust to the right
        mass=5.0,
        grid_size=grid_size,
        reflect_y=False # Only X-axis boundaries active for reflection
    )

    # Auto-stop configuration (tweak to taste)
    enable_autostop = True
    err_tol_deg = 0.25
    std_tol_deg = 0.50
    stable_windows_required = 12  # Number of consecutive monitors to trigger auto-stop

    # Run simulation
    phi1_final, phi2_final, shell_final, shell_trajectory, series, stop_step = run_full_simulation(
        phi1_initial, phi2_initial,
        phi1_vel_initial, phi2_vel_initial,
        g_mode, lambda_val, K_quad, target_delta_0_rad,
        dx, dt_actual, steps, # Use dt_actual from stability_guard_dt
        shell_obj=shell_for_thrust,
        log_every=1000,
        monitor_every=200,
        store_series=True,
        enable_autostop=enable_autostop,
        err_tol_deg=err_tol_deg,
        std_tol_deg=std_tol_deg,
        stable_windows_required=stable_windows_required,
        abort_energy_abs=1e12
    )

    # Final analysis
    dtheta_final = phi1_final - phi2_final
    dtheta_wrapped_0_2pi = np.mod(dtheta_final, 2.0 * np.pi)

    mean_final = circular_mean(dtheta_final)
    std_final = circular_std(dtheta_final)
    err_final = wrap_angle(mean_final - target_delta_0_rad)

    phi_product_mean = float(np.mean(phi1_final * phi2_final))
    f_thrust_signed = g_mode * phi_product_mean * np.sin(shell_final.delta_theta_internal)
    f_thrust_mag = abs(f_thrust_signed)

    print("\n--- Final Simulation Results ---")
    if stop_step is not None and stop_step < steps:
        print(f"Stopped at step: {stop_step} / {steps} (Auto-stop triggered).")
    else:
        print(f"Reached full iterations: {steps} steps.")

    print(f"Final circular mean Δθ: {angle_deg_0_360(mean_final):.3f} degrees")
    print(f"Final circular std  Δθ: {np.rad2deg(std_final):.6f} degrees")
    print(f"Final lock error (mean-target): {np.rad2deg(err_final):+.6f} degrees")
    print(f"Mean phi1*phi2: {phi_product_mean:.6f}")
    print(f"Simulated Thrust (proxy magnitude): {f_thrust_mag:.6f} toy units")

    # ZPE bookkeeping (toy)
    shell_radius_m = shell_final.radius * dx
    volume_spherical_m3 = (4.0 / 3.0) * np.pi * (shell_radius_m ** 3)
    rho_ZPE = 1e-7
    lock_eff = abs(np.sin(mean_final))
    power_W = rho_ZPE * volume_spherical_m3 * (lock_eff ** 2)
    power_MW_per_m3 = (power_W / volume_spherical_m3) * 1e6 if volume_spherical_m3 > 0 else 0.0

    print("\n--- ZPE Extraction Results (toy) ---")
    print(f"Volume of Coherence Shell: {volume_spherical_m3:.6f} m³")
    print(f"Lock efficiency: {lock_eff:.6f}")
    print(f"Total ZPE power: {power_W:.3e} W")
    print(f"ZPE power density: {power_MW_per_m3:.6f} MW/m³")

    # Visualisation: Δθ field + trajectory
    ny, nx = dtheta_wrapped_0_2pi.shape
    plt.figure(figsize=(10, 8))
    plt.imshow(
        dtheta_wrapped_0_2pi,
        cmap="viridis",
        origin="lower",
        extent=[0, nx * dx, 0, ny * dx]
    )
    plt.colorbar(label="Δθ (radians, wrapped to [0, 2π))")
    plt.title("Final Δθ Distribution + CoherenceShell Trajectory")

    circle = plt.Circle(
        (shell_final.x * dx, shell_final.y * dx),
        shell_final.radius * dx,
        color="red",
        fill=False,
        linewidth=1.5
    )
    plt.gca().add_patch(circle)

    traj_x = [p[0] * dx for p in shell_trajectory]
    traj_y = [p[1] * dx for p in shell_trajectory]
    plt.plot(traj_x, traj_y, "w--", linewidth=0.6, alpha=0.75, label="Shell Trajectory")
    plt.legend()
    plt.xlabel("X (m, toy)")
    plt.ylabel("Y (m, toy)")
    plt.show()

    # Visualisation: time series (energy + stability + shell key)
    if series is not None and len(series["t"]) > 2: # Ensure enough data for plotting
        t = np.array(series["t"], dtype=np.float64)

        plt.figure(figsize=(10, 6))
        plt.plot(t, series["E_total"], label="E_total")
        plt.plot(t, series["E_kin"], label="E_kin_fields")
        plt.plot(t, series["E_grad"], label="E_grad")
        plt.plot(t, series["E_pot"], label="E_pot")
        plt.plot(t, series["shell_KE"], label="shell_KE")
        plt.title("Energy Diagnostics (toy)")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (toy units, avg per cell)")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(t, series["mean_dtheta_deg"], label="mean Δθ (deg, 0..360)")
        plt.plot(t, series["lock_error_deg"], label="lock error (deg, wrapped)")
        plt.plot(t, series["std_dtheta_deg"], label="circular std (deg)")
        plt.title("Stability Monitor")
        plt.xlabel("Time (s)")
        plt.ylabel("Degrees")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(t, series["shell_energy_key"], label="shell_energy_key")
        plt.plot(t, series["thrust"], label="thrust (proxy)")
        plt.title("Shell Energy Key + Thrust")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.show()

    # Health Coherence Plot
    if "tumor_fraction" in series and "coherence_score" in series:
        plt.figure(figsize=(10, 6))
        plt.plot(t, series["tumor_fraction"], label="Tumor Fraction (decoherence)", color="red")
        plt.plot(t, series["coherence_score"], label="Coherence Score (body health)", color="green", linewidth=2)
        plt.axhline(0.5, color="gray", linestyle="--", label="Cancer threshold")
        plt.title("Health Coherence: Tumor Growth vs Booth Healing")
        plt.xlabel("Time (s)")
        plt.ylabel("Fraction / Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    print("\n--- Shell Dynamics ---")
    print(f"Shell initial position: ({shell_for_thrust.x:.2f}, {shell_for_thrust.y:.2f})")
    print(f"Shell final position:   ({shell_final.x:.2f}, {shell_final.y:.2f})")
    print(f"Shell final velocity:   ({shell_final.vx:.6f}, {shell_final.vy:.6f})")
    print(f"Final thrust direction: {'Left' if f_thrust_signed < 0 else 'Right' if f_thrust_signed > 0 else 'None'}")
