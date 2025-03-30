#%%
import matplotlib.pyplot as plt
import numpy as np # Using numpy for time array generation

# --- Corrected Motor Model Function ---
# Based on τ_m = (K_t/R)*V - (K_t*K_e/R)*ω
# and dω/dt = (1/J) * (τ_m - b*ω)
def motor_model_dynamics(voltage, omega, R, K_t, K_e, J, b):
    """
    Calculates the angular acceleration (dω/dt) of the DC motor.

    Args:
        voltage (float): Current input voltage (V).
        omega (float): Current angular velocity (rad/s).
        R (float): Armature Resistance (Ohms).
        K_t (float): Torque Constant (Nm/A).
        K_e (float): Back EMF Constant (V/(rad/s)).
        J (float): Rotor Inertia (kg.m^2).
        b (float): Viscous Friction Coefficient (Nm.s/rad).

    Returns:
        float: Angular acceleration (dω/dt) in rad/s^2.
    """
    # Avoid division by zero if R or J are zero
    if R == 0 or J == 0:
        return 0.0

    # Calculate current based on V, omega, K_e, R (can skip if not needed elsewhere)
    # current_I = (voltage - K_e * omega) / R
    # Calculate motor torque (τ_m)
    # tau_m = K_t * current_I
    # Combined torque calculation:
    tau_m = (K_t / R) * voltage - (K_t * K_e / R) * omega

    # Calculate angular acceleration (dω/dt)
    domega_dt = (1.0 / J) * (tau_m - b * omega)

    return domega_dt

# --- Simulation and Display Function ---
def simulate_and_display_ramp_voltage(
    V_max=12.0,          # Maximum voltage to ramp up to (V)
    t_final=10.0,        # Total simulation time (s)
    dt=0.01,             # Simulation time step (s)
    # --- Default Motor Parameters (Example Values) ---
    R=1.0,               # Armature Resistance (Ohms)
    K_t=0.1,             # Torque Constant (Nm/A)
    K_e=0.1,             # Back EMF Constant (V/(rad/s)) - Often numerically equal to K_t in SI
    J=0.01,              # Rotor Inertia (kg.m^2)
    b=0.001              # Viscous Friction Coefficient (Nm.s/rad)
    ):
    """
    Simulates the DC motor with a slowly increasing voltage and plots the results.

    Args:
        V_max (float): Maximum voltage to reach.
        t_final (float): Duration of the simulation.
        dt (float): Time step for numerical integration.
        R, K_t, K_e, J, b (float): Motor parameters.
    """
    # --- Simulation Setup ---
    n_steps = int(t_final / dt)
    time_points = np.linspace(0, t_final, n_steps + 1)

    # Initialize state variables and history lists
    omega = 0.0  # Initial angular velocity
    omega_history = [omega]
    voltage_history = []
    voltages = np.sin(np.linspace(0, 10*np.pi, n_steps + 1)) * V_max
    # --- Simulation Loop (Euler Integration) ---
    for i, t in enumerate(time_points):
        # Calculate current voltage based on linear ramp
        current_voltage = voltages[i]
        # Ensure voltage doesn't ramp down if t slightly exceeds t_final due to float precision
        if t >= t_final/2:
             current_voltage = V_max

        voltage_history.append(current_voltage)

        # Calculate acceleration using the model
        domega_dt = motor_model_dynamics(current_voltage, omega, R, K_t, K_e, J, b)

        # Update velocity for the *next* step (Euler method)
        # Note: Skip update for the very last time point as we already stored the final state
        if t < t_final:
             omega = omega + domega_dt * dt
             omega_history.append(omega)

    # --- Plotting Results ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Angular Velocity (Omega)
    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angular Velocity ω (rad/s)', color=color)
    ax1.plot(time_points, omega_history, color=color, label='Angular Velocity (ω)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    # Create a second y-axis for Voltage
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Input Voltage (V)', color=color)
    ax2.plot(time_points, voltage_history, color=color, linestyle='--', label='Input Voltage (V)')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add titles and legend
    plt.title('Motor Angular Velocity Response to Ramped Voltage')
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

# --- Run the Simulation ---
if __name__ == "__main__":
    print("Running DC motor simulation with ramped voltage...")
    # You can customize parameters here:
    simulate_and_display_ramp_voltage(
        V_max=12.0,
        t_final=15.0, # Increased time slightly
        dt=0.01,
        # --- Using the default motor parameters ---
        # R=1.0, K_t=0.1, K_e=0.1, J=0.01, b=0.001
    )
    print("Simulation complete.")