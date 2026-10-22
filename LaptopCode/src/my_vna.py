import definitions, time, os, pickle
import skrf as rf
import matplotlib.pyplot as plt
import numpy as np
from skrf.vi.vna.nanovna import NanoVNAv2

_cal_standards = {}   # e.g. {"open": Network, "short": Network, "load": Network}
_calibration = None   # active skrf OnePort calibration, once run() succeeds

CAL_STANDARD_NAMES = ["open", "short", "load"]

def vna_connect(port=definitions.VNA_PORT):
    address = f"ASRL{port.replace('COM','')}::INSTR"
    return NanoVNAv2(address)

def vna_measure_raw(vna, f_start, f_stop, f_points=None, verbose=False):
    """Performs a raw measurement -- no calibration correction applied."""
    if f_points is None:
        f_points = definitions.F_POINTS
    if verbose:
        print(f"Measuring from {f_start/1e6:.2f}MHz to {f_stop/1e9:.2f}GHz across {f_points} points.")
    freq = rf.Frequency(start=f_start, stop=f_stop, npoints=f_points, unit="Hz")
    vna.frequency = freq
    time.sleep(0.15) # Allow time to set?
    s11, s21 = vna.get_s11_s21()
    return s11

def vna_measure(vna, f_start, f_stop, f_points=None, verbose=False):
    """Performs a measurement and applies the active calibration, if any."""
    if f_points is None:
        f_points = definitions.F_POINTS
    if not vna:
        print("No VNA connected.")
    s11 = vna_measure_raw(vna, f_start, f_stop, f_points, verbose=verbose)
    return cal_apply(s11)

def vna_save(s11, output_dir, output_name):
    os.makedirs(output_dir, exist_ok=True)
    s11.write_touchstone(filename=output_name, dir=output_dir)

def vna_load(file_dir, file_name: str, warn=True):
    try: 
        if not (file_name.endswith(".s1p")):
            file_name = f"{file_name}.s1p"
        filepath = f"{file_dir}/{file_name}"
        return rf.Network(filepath)
    except Exception as e:
        if warn:
            print(f"Specified s1p file not found: {e}")
        return None

def plot_s11_db(mut_name, s11_dir=definitions.VNA_MEASURE_FILE, plot_dir=definitions.VNA_PLOT_FILE, title="S11 Return"):
    """
    Plots S11 in dB against frequency for a scikit-rf Network object.
    
    Parameters:
        s11: skrf.Network object
        title: Title string for the plot
    """
    # Frequency in Hz converted to GHz
    try:
        s11_filename = f"{s11_dir}/{mut_name}.s1p"
        s11 = rf.Network(s11_filename)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    freq_ghz = s11.f / 1e9
    
    # Extract S11 in dB (s_db array shape is [f_points, n_ports, n_ports])
    s11_db = s11.s_db[:, 0, 0] if s11.s_db.ndim == 3 else s11.s_db.flatten()

    plt.figure(figsize=(8, 5))
    plt.scatter(freq_ghz, s11_db, label="$S_{11}$", color="b", linewidth=0.1)
    plt.title(title)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    #plt.ylim([-20, 2])
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{mut_name}.png", dpi=300, bbox_inches='tight')
    #plt.show()

def vna_scan(vna, output_name, f_start=None, f_stop=None, f_points=None, 
             output_dir=definitions.VNA_SAVE_FILE, verbose=False):
    if f_start is None:
        f_start = max(definitions.F_START, definitions.VNA_MIN_FREQ)
    if f_stop is None:
        f_stop = min(definitions.F_STOP, definitions.VNA_MAX_FREQ)
    if f_points is None:
        f_points = definitions.F_POINTS

    s11 = vna_measure(vna, f_start, f_stop, f_points, verbose=verbose)
    vna_save(s11, output_dir=output_dir, output_name=output_name)

# =====================================================================
# Calibration (1-port Open/Short/Load)
# =====================================================================

def cal_reset():
    """Clears any in-progress standard measurements and the active calibration."""
    global _cal_standards, _calibration
    _cal_standards = {}
    _calibration = None
    print("Calibration reset.")

def cal_measure_standard(vna, standard, f_start=None, f_stop=None, f_points=None):
    """Measures one calibration standard ('open'/'short'/'load') and stores it.
    Call once per standard, then cal_run() once all three are done."""
    if f_start is None:
        f_start = definitions.VNA_MIN_FREQ
    if f_stop is None:
        f_stop = definitions.VNA_MAX_FREQ
    if f_points is None:
        f_points = definitions.F_POINTS

    standard = standard.lower()
    if standard not in CAL_STANDARD_NAMES:
        print(f"Unknown standard '{standard}'. Must be one of {CAL_STANDARD_NAMES}.")
        return
    s11 = vna_measure_raw(vna, f_start, f_stop, f_points)
    _cal_standards[standard] = s11
    print(f"Measured calibration standard: {standard}")

def cal_status():
    have = list(_cal_standards.keys())
    missing = [s for s in CAL_STANDARD_NAMES if s not in have]
    print(f"Standards measured: {have if have else 'none'}")
    if missing:
        print(f"Still needed: {missing}")
    print(f"Active calibration: {'yes' if _calibration is not None else 'no'}")

def _ideal_standard(freq, s_value):
    """Builds an ideal one-port standard network (perfect open/short/load,
    no delay/loss) at the given frequency points."""
    s = np.full((len(freq), 1, 1), s_value, dtype=complex)
    return rf.Network(frequency=freq, s=s, z0=50)

def cal_run(save_path=None):
    """Builds and runs a OnePort (OSL) calibration from the three
    previously measured standards. Requires cal_measure_standard() to
    have been called for 'open', 'short', and 'load' first."""
    global _calibration
    missing = [s for s in CAL_STANDARD_NAMES if s not in _cal_standards]
    if missing:
        print(f"Cannot run calibration, missing standards: {missing}")
        return False

    meas_open, meas_short, meas_load = (
        _cal_standards["open"], _cal_standards["short"], _cal_standards["load"]
    )
    freq = meas_open.frequency

    cal = rf.calibration.OnePort(
        ideals=[_ideal_standard(freq, 1.0), _ideal_standard(freq, -1.0), _ideal_standard(freq, 0.0)],
        measured=[meas_open, meas_short, meas_load],
    )
    cal.run()
    _calibration = cal
    print("Calibration complete and active.")

    if save_path:
        cal_save(save_path)
    return True

def cal_save(path):
    if _calibration is None:
        print("No active calibration to save.")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(_calibration, f)
    print(f"Calibration saved to {path}")

def cal_load(path):
    global _calibration
    try:
        with open(path, "rb") as f:
            _calibration = pickle.load(f)
        print(f"Calibration loaded from {path}")
        return True
    except Exception as e:
        print(f"Failed to load calibration: {e}")
        return False

def cal_apply(s11):
    """Applies the active calibration to a raw S11 measurement, or
    returns it unchanged if no calibration is active."""
    if _calibration is None:
        return s11
    return _calibration.apply_cal(s11)