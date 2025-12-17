print("=== Calibrating brewer ===")
import format_measurements_brewer_calibration
import fit_model_brewer_calibration
import plot_multipanel_brewer_calibration

print("=== Calibrating static model ===")
import format_measurements_time_dependent
import plot_multipanel_time_dependent
import fit_model_static_flow_rate
import plot_singlepanel_theory_vs_pressure

print("=== Calibrating dynamic model ===")
import format_measurements_tds
import fit_model_tds
import fit_model_solids
import plot_multipanel_tds

print("=== Ploting mastersizer ===")
import plot_singlepanel_mastersizer

print("=== Ploting comparison ===")
import plot_singlepanel_phi_limit
import plot_multipanel_theory_vs_time


