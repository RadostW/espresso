import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# =========================
# Paths
# =========================

HERE = Path(__file__).resolve().parent

OUTPUT_FILE = HERE / "figures" / "phi_limit.pdf"
STYLE_FILE = HERE / "styles" / "espresso.mplstyle"

# =========================
# Figure and plot settings
# =========================

FIGURE_WIDTH = 3.3
FIGURE_HEIGHT = 2.2
FONT_SIZE = 9

plt.style.use(STYLE_FILE)
plt.rcParams.update({"lines.marker": ""})  # no default markers

# =========================
# theoretical functions
# =========================


def limit_teo(p):
    return p * (4 - 6 * p + 4 * p**2 - p**3)


def full_teo(p, f):
    """
    Compute Q_hat given p and f.

    Note:
        Variable names are changed from the original LaTeX:
        - p corresponds to \hat{P}
        - f corresponds to \Phi
    """
    numerator = 6 * (f - 1) ** 3 * np.log(1 - p * f) - p * f * (
        (p * (2 * p - 9) + 18) * f**2 + 3 * (p - 6) * f + 6
    )

    denominator = f * ((15 - 11 * f) * f - 6) + 6 * (f - 1) ** 3 * np.log(1 - f)

    return numerator / denominator


# =========================
# Plot
# =========================

fig, ax = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
    constrained_layout=True,
)

p_vals = np.linspace(0, 1, 100)
plt.plot(p_vals, full_teo(p_vals, f=0.5), label=r"$\Phi = 0.5$")
plt.plot(p_vals, full_teo(p_vals, f=0.2), label=r"$\Phi = 0.2$")
plt.plot(p_vals, full_teo(p_vals, f=0.1), label=r"$\Phi = 0.1$")

plt.plot(
    p_vals,
    limit_teo(p_vals),
    label=r"$\Phi \to 0$",
    linestyle="-",
    dashes=[3, 0.3],
    color="k",
)


ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_xlabel(r"$\hat{P}$")
ax.set_ylabel(r"$\hat{Q}$")

ax.legend()

# =========================
# Save
# =========================

plt.savefig(OUTPUT_FILE)
if __name__ == "__main__":
    plt.show()

print(f"Saved {OUTPUT_FILE}")
