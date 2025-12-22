import numpy as np
import matplotlib.pyplot as plt
# --- 1. CoherenceShell Class ---
class CoherenceShell:
    def **init**(self, x_center, y_center, radius, delta_theta_internal, mass=1.0, grid_size=100):
        self.x = x_center
        self.y = y_center
        self.radius = radius
        self.delta_theta_internal = np.deg2rad(delta_theta_internal) # Internal phase for thrust generation
        self.mass = mass
        self.vx = 0.0
        self.vy = 0.0
        self.grid_size = grid_size # Number of grid cells (e.g., 100)
       
    def update(self, force_x, force_y, dt, other_shells=None): # Added other_shells for collision
        ax = force_x / self.mass
        ay = force_y / self.mass
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
       
        # --- Enhanced Boundary Handling with Reflection and Velocity Threshold ---
        if self.x < 0.0: # Hit left boundary (or went slightly past)
            self.x = 0.0 # Clamp position
            if abs(self.vx * 0.9) < 0.05: # If dampened velocity is very small
                self.vx = 0.0 if force_x > 0 else -0.05 # Stop if force pushes right, else a tiny push left
            else:
                self.vx = -self.vx * 0.9 # Reverse and dampen velocity
        elif self.x > self.grid_size: # Hit right boundary (or went slightly past)
            self.x = float(self.grid_size) # Clamp position
            if abs(self.vx * 0.9) < 0.05: # If dampened velocity is very small
                self.vx = 0.0 if force_x < 0 else 0.05 # Stop if force pushes left, else a tiny push right
            else:
                self.vx = -self.vx * 0.9 # Reverse and dampen velocity
       
        self.y = max(0.0, min(self.y, float(self.grid_size)))
        # --- Collision Detection and Response ---
        if other_shells:
            for other in other_shells:
                if other != self: # Don't collide with self
                    # Simple 1D collision check (overlap in X, assuming Y is same)
                    if abs(self.x - other.x) * dx < 2 * self.radius * dx: # Check for overlap in meters
                        # Basic elastic collision with dampening
                        relative_vx = self.vx - other.vx
                       
                        # Apply impulse to both shells (simplified for 1D, equal mass)
                        # New velocity for self after collision
                        self.vx = other.vx - 0.9 * relative_vx # 0.9 restitution coefficient
                       
                        # New velocity for other after collision
                        other.vx = self.vx + 0.9 * relative_vx # Note: this applies to 'other' in the loop
                       
                        # Prevent immediate re-collision by slightly separating (optional, but good for stability)
                        # if self.x < other.x:
                        # self.x -= 0.1 * dx # Push self slightly left
                        # other.x += 0.1 * dx # Push other slightly right
                        # else:
                        # self.x += 0.1 * dx
                        # other.x -= 0.1 * dx
# --- 2. 2D Laplacian Function ---
def laplacian_2d(field, dx):
    """Calculates the 2D Laplacian of a field using finite differences with periodic boundary conditions."""
    return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
            4 * field) / (dx ** 2)
# --- 3. Main Simulation Function (for single shell) ---
def run_full_simulation(
    initial_phi1, initial_phi2, initial_phi1_vel, initial_phi2_vel,
    g_mode, lambda_val, K_quad, target_delta_0_rad, dx, dt, iterations,
    shell_obj, track_history=True # Added track_history flag
):
    phi1, phi2 = initial_phi1.copy(), initial_phi2.copy()
    phi1_prev_verlet = phi1.copy()
    phi2_prev_verlet = phi2.copy()
    # History tracking
    shell_trajectory = [(shell_obj.x, shell_obj.y, shell_obj.vx)] if track_history else []
    ke_history = [] if track_history else None
    # These are global constants for the field, so we calculate them once to avoid repeated np.mean
    phi_product_mean_global = np.mean(initial_phi1 * initial_phi2) # Assumes fields stabilize quickly in magnitude
    for t_step in range(iterations):
        # --- Field Dynamics Update ---
        # This part ensures the field remains locked at 255 degrees
        delta_theta_current = phi1 - phi2
        delta_theta_laplacian = laplacian_2d(delta_theta_current, dx)
        wrapped_delta = np.arctan2(np.sin(delta_theta_current - target_delta_0_rad), np.cos(delta_theta_current - target_delta_0_rad))
        force_on_delta_0_from_potential = -(g_mode * np.sin(delta_theta_current) + 2 * K_quad * wrapped_delta)
        accel1_current = force_on_delta_0_from_potential - lambda_val * delta_theta_laplacian
        accel2_current = -force_on_delta_0_from_potential + lambda_val * delta_theta_laplacian
       
        phi1_new = 2 * phi1 - phi1_prev_verlet + dt * dt * accel1_current
        phi2_new = 2 * phi2 - phi2_prev_verlet + dt * dt * accel2_current
       
        if np.any(np.isnan(phi1_new)) or np.any(np.isnan(phi2_new)):
            print(f"NaN detected in field values at step {t_step}. Aborting simulation.")
            break
        phi1_prev_verlet = phi1.copy()
        phi2_prev_verlet = phi2.copy()
        phi1, phi2 = phi1_new.copy(), phi2_new.copy()
        # --- Shell Thrust and Update ---
        if shell_obj:
            # Thrust based on the globally stable phi_product_mean_global (calculated once)
            f_thrust = g_mode * phi_product_mean_global * np.sin(shell_obj.delta_theta_internal)
            force_x_shell = f_thrust
            force_y_shell = 0.0
           
            # For single shell run, other_shells is None
            shell_obj.update(force_x_shell, force_y_shell, dt, other_shells=None)
           
            if track_history:
                shell_trajectory.append((shell_obj.x, shell_obj.y, shell_obj.vx))
                if t_step % 1000 == 0: # Only record KE at logging steps
                    ke = 0.5 * shell_obj.mass * (shell_obj.vx**2 + shell_obj.vy**2)
                    ke_history.append(ke)
                    # print(f"Step {t_step} (Time {t_step*dt:.3f}s), Shell Vx: {shell_obj.vx:.2f}, X: {shell_obj.x:.2f}, KE: {ke:.2f}")
           
    return phi1, phi2, shell_obj, shell_trajectory, ke_history
# --- 4. Main Simulation Setup and Execution ---
if **name** == "**main**":
    # --- General Parameters ---
    g_mode = 0.085 # Coupling constant (4piG in natural units, ~0.085)
    lambda_val = 0.0112 # Coherence growth term
    K_quad = 5.0 # Quadratic stabilization strength (ensures 255 deg minimum)
    target_delta_0_rad = np.deg2rad(255.0) # Target phase difference (255°)
    dx = 0.1 # Spatial step size (m)
    dt = 0.001 # Time step (s)
    amplitude_avg_fields = 25.0 # Average magnitude for phi1, phi2 fields
   
    # --- Adjusted grid_size ---
    grid_size = 200 # Grid dimensions (e.g., 200x200 cells)
    steps = 15000 # Number of simulation steps for convergence
    # Initialize fields (once for all test runs)
    phi1_base_init = amplitude_avg_fields + target_delta_0_rad / 2
    phi2_base_init = amplitude_avg_fields - target_delta_0_rad / 2
   
    phi1_initial_base = np.ones((grid_size, grid_size)) * phi1_base_init
    phi2_initial_base = np.ones((grid_size, grid_size)) * phi2_base_init
   
    # Apply a small perturbation to one cell to initiate dynamics and test stability
    phi1_initial_base[50, 50] += 0.1
    phi2_initial_base[50, 50] -= 0.1
   
    phi1_vel_initial_base = np.zeros((grid_size, grid_size))
    phi2_vel_initial_base = np.zeros((grid_size, grid_size))
    # Calculate phi_product_mean_for_thrust_calc once for all tests, assuming stable fields
    # Perform a dummy run or just use initial state's phi_product for consistent thrust calculation
    dummy_phi1 = phi1_initial_base.copy()
    dummy_phi2 = phi2_initial_base.copy()
    phi_product_mean_for_thrust_calc = np.mean(dummy_phi1 * dummy_phi2) # Should be 625.0
    # --- Test 1: Vary delta_theta_internal ---
    print("\n--- Running Test 1: Varying Delta_theta_internal ---")
    theta_internals = [0.0, 45.0, 90.0, 180.0, 270.0]
    results_theta_test = {}
    for theta_int in theta_internals:
        # Re-initialize fields and shell for each run
        phi1_initial_current = phi1_initial_base.copy()
        phi2_initial_current = phi2_initial_base.copy()
        phi1_vel_initial_current = phi1_vel_initial_base.copy()
        phi2_vel_initial_current = phi2_vel_initial_base.copy()
        shell_current = CoherenceShell(x_center=100.0, y_center=50.0, radius=5.0,
                                       delta_theta_internal=theta_int, mass=5.0, grid_size=grid_size)
       
        phi1_f, phi2_f, shell_f, traj_f, ke_h_f = run_full_simulation(
            phi1_initial_current, phi2_initial_current, phi1_vel_initial_current, phi2_vel_initial_current,
            g_mode, lambda_val, K_quad, target_delta_0_rad, dx, dt, steps, shell_current, track_history=False # No history for individual runs in this loop
        )
       
        # Calculate thrust based on the stable phi_product and current theta_int
        f_thrust_final_calculated = g_mode * phi_product_mean_for_thrust_calc * np.sin(np.deg2rad(theta_int))
       
        results_theta_test[theta_int] = {
            'final_vx': shell_f.vx,
            'final_position': (shell_f.x, shell_f.y),
            'thrust_calculated': f_thrust_final_calculated
        }
        print(f"Theta_internal = {theta_int}°: Vx = {shell_f.vx:.2f}, Position = ({shell_f.x:.2f}, {shell_f.y:.2f}), Thrust (calc) = {f_thrust_final_calculated:.2f}")
    # --- Test 2: Multiple Shells (with collision and full field simulation for consistency) ---
    print("\n--- Running Test 2: Multiple Shells ---")
   
    # Re-initialize fields for the multi-shell run to ensure a clean state
    phi1_multi = phi1_initial_base.copy()
    phi2_multi = phi2_initial_base.copy()
    phi1_vel_multi = phi1_vel_initial_base.copy()
    phi2_vel_multi = phi2_vel_initial_base.copy()
    shells = [
        CoherenceShell(x_center=150.0, y_center=50.0, radius=5.0, delta_theta_internal=90.0, mass=5.0, grid_size=grid_size), # Moves right
        CoherenceShell(x_center=50.0, y_center=50.0, radius=5.0, delta_theta_internal=270.0, mass=5.0, grid_size=grid_size) # Moves left
    ]
   
    # History tracking for multiple shells
    trajectories = [[] for _ in shells]
    ke_histories = [[] for _ in shells]
    vx_histories = [[] for _ in shells]
    thrust_applied_histories = [[] for _ in shells] # To store the actual thrust applied
    # --- Main Multi-Shell Simulation Loop ---
    # Need to integrate phi1/phi2 fields within this loop too
    # phi1, phi2 = phi1_initial_base.copy(), phi2_initial_base.copy() # Fields for this run
    phi1_prev_verlet = phi1_initial_base.copy()
    phi2_prev_verlet = phi2_initial_base.copy()
    #phi_product_mean_global_multi = np.mean(phi1_initial_base * phi2_initial_base) # Stable mean product
   
    for t_step in range(steps):
        # --- Field Dynamics Update (Same as run_full_simulation) ---
        delta_theta_current = phi1_multi - phi2_multi
        delta_theta_laplacian = laplacian_2d(delta_theta_current, dx)
        wrapped_delta = np.arctan2(np.sin(delta_theta_current - target_delta_0_rad), np.cos(delta_theta_current - target_delta_0_rad))
        force_on_delta_0_from_potential = -(g_mode * np.sin(delta_theta_current) + 2 * K_quad * wrapped_delta)
        accel1_current = force_on_delta_0_from_potential - lambda_val * delta_theta_laplacian
        accel2_current = -force_on_delta_0_from_potential + lambda_val * delta_theta_laplacian
       
        phi1_new = 2 * phi1_multi - phi1_prev_verlet + dt * dt * accel1_current
        phi2_new = 2 * phi2_multi - phi2_prev_verlet + dt * dt * accel2_current
       
        if np.any(np.isnan(phi1_new)) or np.any(np.isnan(phi2_new)):
            print(f"NaN detected in field values at step {t_step}. Aborting multi-shell simulation.")
            break
        phi1_prev_verlet = phi1_multi.copy()
        phi2_prev_verlet = phi2_multi.copy()
        phi1_multi, phi2_multi = phi1_new.copy(), phi2_new.copy()
        # --- Shell Updates (Thrust and Collision) ---
        phi_product_mean_local_multi = np.mean(phi1_multi * phi2_multi) # Dynamic phi_product for thrust
       
        for i, shell in enumerate(shells):
            f_thrust = g_mode * phi_product_mean_local_multi * np.sin(shell.delta_theta_internal)
            force_x = f_thrust
            force_y = 0.0
            shell.update(force_x, force_y, dt, other_shells=shells) # Pass all shells for collision
           
            # Store history
            trajectories[i].append((shell.x, shell.y, shell.vx))
            if t_step % 1000 == 0:
                ke = 0.5 * shell.mass * (shell.vx**2 + shell.vy**2)
                ke_histories[i].append(ke)
                vx_histories[i].append(shell.vx)
                thrust_applied_histories[i].append(f_thrust)
                print(f"Step {t_step} (Time {t_step*dt:.3f}s), Shell {i} X: {shell.x:.2f}, Vx: {shell.vx:.2f}, KE: {ke:.2f}")
    # --- Multi-Shell Visualization ---
    plt.figure(figsize=(12, 10))
    ax1_multi = plt.subplot(3, 1, 1)
   
    # Plot final Delta_0 distribution (from multi-shell run)
    delta_theta_final_multi_unwrapped = phi1_multi - phi2_multi
    delta_theta_final_multi_wrapped = np.mod(delta_theta_final_multi_unwrapped, 2 * np.pi)
    plt.imshow(delta_theta_final_multi_wrapped, cmap='viridis', origin='lower', extent=[0, grid_size*dx, 0, grid_size*dx])
    plt.colorbar(label='Delta_0 (radians)')
    plt.title('Final Delta_0 Distribution with Multiple Coherence Shell Trajectories')
   
    # Plot shells final positions
    for i, shell_final in enumerate(shells):
        circle = plt.Circle((shell_final.x * dx, shell_final.y * dx), shell_final.radius * dx, color=['red', 'yellow'][i], fill=False)
        ax1_multi.add_patch(circle)
   
    # Plot trajectories
    colors_traj = ['r--', 'y--']
    labels_traj = ['Shell 1 (90°)', 'Shell 2 (270°)']
    for i, traj in enumerate(trajectories):
        traj_x = [p[0] * dx for p in traj]
        traj_y = [p[1] * dx for p in traj]
        plt.plot(traj_x, traj_y, colors_traj[i], label=labels_traj[i])
    ax1_multi.legend()
    ax1_multi.set_ylabel('Y (toy units, m)')
    # Multi-shell KE over time
    ax2_multi = plt.subplot(3, 1, 2)
    time_steps_multi_ke = [t * dt for t in range(0, steps, 1000)]
    for i in range(len(shells)):
        plt.plot(time_steps_multi_ke, ke_histories[i], ['r-', 'y-'][i], label=f'Shell {i} KE')
    plt.title('Shells Kinetic Energy Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Kinetic Energy (toy units)')
    plt.grid(True)
    ax2_multi.legend()
    # Multi-shell Vx over time with Thrust Overlay
    ax3_multi = plt.subplot(3, 1, 3)
    time_steps_multi_vx = [t * dt for t in range(0, steps, 1000)]
    for i in range(len(shells)):
        plt.plot(time_steps_multi_vx, vx_histories[i], ['r-', 'y-'][i], label=f'Shell {i} Vx')
       
        # Overlay actual thrust (acceleration) for each shell
        thrust_overlay_accel = [thrust_applied_histories[i][j] / shells[i].mass for j in range(len(time_steps_multi_vx))]
        plt.plot(time_steps_multi_vx, thrust_overlay_accel, ['r--', 'y--'][i], label=f'Shell {i} Thrust Accel', alpha=0.5)
    plt.title('Shells Velocity X and Thrust Acceleration Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity X / Accel (toy units/s / s^2)')
    plt.grid(True)
    ax3_multi.legend()
   
    plt.tight_layout()
    plt.savefig('multi_shell_simulation_plot.png', dpi=300, bbox_inches='tight')
    plt.show()