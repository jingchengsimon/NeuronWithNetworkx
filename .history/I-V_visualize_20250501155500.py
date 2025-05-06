import numpy as np
import matplotlib.pyplot as plt

# Define the mggate function
def mggate(v, mg=1.0):  # v in mV, mg in mM
    return 1 / (1 + np.exp(0.08 * -v) * (mg / 3.57))

# Voltage range (in mV)
v_values = np.linspace(-100, 50, 500)

# Different Mg²⁺ concentrations (in mM)
mg_concentrations = [0.0, 0.5, 1.0, 1.5, 2.0]

# Plot
plt.figure(figsize=(8, 6))
for mg in mg_concentrations:
    plt.plot(v_values, mggate(v_values, mg), label=f"[Mg²⁺] = {mg} mM")

plt.title("Mg²⁺ Block (mggate) as a Function of Membrane Potential")
plt.xlabel("Membrane Potential (mV)")
plt.ylabel("mggate (relief from Mg²⁺ block)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
