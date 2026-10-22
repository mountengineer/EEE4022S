import my_vna, definitions, os
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import PyOECP.Transform as Transform
import PyOECP.References as References

# =====================================================================
# PROBE GEOMETRY PARAMETERS 
# =====================================================================
R_I = 0.6      # Inner conductor radius in mm (e.g., RG-402 / 3.5mm probe)
R_O = 2.00      # Outer conductor inside radius in mm
EPSILON_C = 2.1  # Dielectric constant of insulator (2.1 for PTFE/Teflon)
M_TERMS = 50     # Number of summation terms


def get_ref_measurement(vna, ref_dir, ref_name, f_start=None, f_stop=None,
                                  f_points=None, prompt=''):
    """Loads a saved reference measurement if it exists, otherwise
    prompts the user and measures it fresh.

    Returns
    -------
    s11 : skrf.Network
    """
    if f_start is None:
        f_start = definitions.VNA_MIN_FREQ
    if f_stop is None:
        f_stop = definitions.VNA_MAX_FREQ
    if f_points is None:
        f_points = definitions.F_POINTS

    filepath = os.path.join(ref_dir, f"{ref_name}.s1p")
    if os.path.isfile(filepath):
        print(f"Found existing measurement for '{ref_name}', loading from file.")
        return rf.Network(filepath)

    if prompt == "File not found":
        print(prompt)
        return
    elif prompt:
        input(prompt)
    if vna == None:
        print("No VNA connected.")
        return
    s11 = my_vna.vna_measure(vna, f_start, f_stop, f_points, verbose=True)
    my_vna.vna_save(s11, ref_dir, ref_name)
    return s11

def get_new_ref(vna, ref_name):
    if ref_name not in definitions.REFERENCES:
        print(f"Error: Reference name must be in {definitions.REFERENCES}")
    s11 = my_vna.vna_measure(vna, definitions.F_START, definitions.F_STOP, definitions.F_POINTS, verbose=True)
    my_vna.vna_save(s11, definitions.OECP_REF_FILE, ref_name)

def test_material(mut_name, vna, overwrite = False, ref_dir=definitions.OECP_REF_FILE, 
                  f_start=None, f_stop=None, f_points=None):
    if f_start is None:
        f_start = definitions.VNA_MIN_FREQ
    if f_stop is None:
        f_stop = definitions.VNA_MAX_FREQ
    if f_points is None:
        f_points = definitions.F_POINTS
    
    if overwrite and not vna:
        print("No VNA given.")
        return
    s11_r1 = get_ref_measurement(vna, ref_dir, definitions.R1, f_start, f_stop, f_points, "File not found")
    s11_r2 = get_ref_measurement(vna, ref_dir, definitions.R2, f_start, f_stop, f_points, "File not found")
    s11_r3 = get_ref_measurement(vna, ref_dir, definitions.R3, f_start, f_stop, f_points, "File not found")
    s11_m = my_vna.vna_load(definitions.VNA_MEASURE_FILE, mut_name, False)
    if not s11_m or overwrite:
        s11_m = my_vna.vna_measure(vna, definitions.F_START, definitions.F_STOP, definitions.F_POINTS, verbose=True)
        my_vna.vna_save(s11_m, definitions.VNA_MEASURE_FILE, mut_name)
        print(f"Acquired new reading for {mut_name}")
    else:
        print(f"Using previous reading for {mut_name}")

    # 1. Frequency as a 1D vector (N,)
    freqs = np.asarray(s11_r1.f)
    epsilon = get_dielectric(freqs, s11_r1, s11_r2, s11_r3, s11_m)

    freq_ghz = freqs / 1e9
    # Filter data above 2 GHz 
    mask = (freq_ghz >= 1)
    freq_plot = freq_ghz[mask]

    # Extract real dielectric constant (ε') and loss factor (ε'')
    eps_real = np.real(epsilon[mask])
    eps_imag = np.abs(np.imag(epsilon[mask]))  # abs() handles potential sign convention differences

    # Create 2-panel plot for permittivity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    real_broken = eps_real > 1e3
    eps_real[real_broken] = -1
    imag_broken = eps_imag > 1e3
    eps_imag[imag_broken] = -1

    # Real Permittivity (ε')
    ax1.plot(freq_plot, eps_real, color='navy', lw=2, label="ε' (Real)")
    ax1.set_ylabel("Dielectric Constant (ε')")
    #ax1.set_ylim([30, 50])
    ax1.set_title("Material Under Test — Complex Permittivity Spectrum")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    # Imaginary Permittivity (ε'')
    ax2.plot(freq_plot, eps_imag, color='crimson', lw=2, label="ε'' (Loss Factor)")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylim([-1, 20])
    ax2.set_ylabel("Loss Factor (ε'')")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{definitions.OECP_SAVE_FILE}/{mut_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
        
def get_dielectric(freqs, r1, r2, r3, mut):
    n_freqs = freqs.shape[0]

    def get_s11(net, label):
        gamma = np.asarray(net.s[:, 0, 0]).flatten()
        assert gamma.shape[0] == n_freqs, (
            f"{label} has {gamma.shape[0]} points, expected {n_freqs}. "
            "Did every measurement use the same F_START/F_STOP/F_POINTS?"
        )
        return gamma

    gamma_r1 = get_s11(r1, "r1 (air)")
    gamma_r2 = get_s11(r2, "r2 (ethanol)")
    gamma_r3 = get_s11(r3, "r3 (DI water)")
    gamma_m = get_s11(mut, "MUT")

    for name, arr in [
        ("freqs", freqs), ("gamma_m", gamma_m),
        ("gamma_r1", gamma_r1), ("gamma_r2", gamma_r2), ("gamma_r3", gamma_r3)
    ]:
        assert arr.ndim == 1 and arr.shape[0] == n_freqs, (
            f"{name} has shape {arr.shape}, expected ({n_freqs},)"
        )

    komarov_model = Transform.Komarov(
        frequency=freqs,
        S11m=np.column_stack((freqs, np.real(gamma_m), np.imag(gamma_m))),
        S11r1=np.column_stack((freqs, np.real(gamma_r1), np.imag(gamma_r1))),
        S11r2=np.column_stack((freqs, np.real(gamma_r2), np.imag(gamma_r2))),
        S11r3=np.column_stack((freqs, np.real(gamma_r3), np.imag(gamma_r3))),
        m1=definitions.M1,
        m2=definitions.M2,
        m3=definitions.M3,
        Ri=R_I,
        Ro=R_O,
        epsilonC=EPSILON_C,
        M=M_TERMS
    )

    # PyOECP stores the complex permittivity output in an attribute
    eps_mut = getattr(komarov_model, 'eps', None)
    if eps_mut is None:
        eps_mut = getattr(komarov_model, 'epsilon', komarov_model)
    eps_mut = np.asarray(eps_mut).flatten()
    return eps_mut
