import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Paths
# =========================

HERE = Path(__file__).resolve().parent

OUTPUT_FILE = HERE / "figures" / "brew_restarting.pdf"
INPUT_FILE = HERE / "measurements_brew_restarting" / "brew_restarting.txt"
STYLE_FILE = HERE / "styles" / "espresso.mplstyle"

# =========================
# Figure settings
# =========================

FIGURE_WIDTH = 3.3
FIGURE_HEIGHT = 2.2

plt.style.use(STYLE_FILE)
plt.rcParams.update({"lines.marker": ""})

# =========================
# Load data
# =========================

df = pd.read_json(INPUT_FILE, lines=True)

df["time__s"] = df["t"] / 1e3
df["driving_pressure__bar"] = df["p2"] / 1e2
df["flow_rate__g_per_s"] = (
    1e3 * df["m"].rolling(5).mean().diff() / df["t"].diff()
)

# =========================
# Plot
# =========================

fig, ax_left = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
    constrained_layout=True,
)

# Right axis FIRST (lower layer)
ax_right = ax_left.twinx()
ax_right.plot(
    df["time__s"],
    df["driving_pressure__bar"],
    color="C1",
    label="Driving pressure",
)
ax_right.set_ylabel("Driving pressure [bar]", color="C1")
ax_right.tick_params(axis="y", labelcolor="C1")

# Left axis SECOND (top layer)
ax_left.plot(
    df["time__s"],
    df["flow_rate__g_per_s"],
    color="C0",
    label="Flow rate",
)
ax_left.set_ylabel("Flow rate [g/s]", color="C0")
ax_left.tick_params(axis="y", labelcolor="C0")
ax_left.set_xlabel("Time [s]")

# === CRITICAL PART ===
ax_right.set_zorder(1)
ax_left.set_zorder(2)
ax_left.patch.set_visible(False)
# =====================

plt.savefig(OUTPUT_FILE)

if __name__ == "__main__":
    plt.show()

plt.close(fig)
print(f"Saved {OUTPUT_FILE}")
