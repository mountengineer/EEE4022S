import definitions, skrf, glob, os, re, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.signal import wiener
from pathlib import Path
from scipy.signal import hilbert

# def load_s1p_sinogram(directory):
#     """Loads all .s1p files in a directory into a sinogram array

#     Assumes filenames sort into the correct probe-position order
#     (e.g. zero-padded indices), and that every file shares the same
#     frequency points.

#     Returns
#     -------
#     freqs : array_like
#         Frequency points (Hz), shared across all positions
#     sinogram : array_like
#         S11 data, shape (n_freqs, n_positions)
#     """
#     files = sorted(glob.glob(os.path.join(directory, '*.s1p')))
#     assert len(files) > 0, f"No .s1p files found in {directory}"

#     networks = [skrf.Network(f) for f in files]
#     freqs = networks[0].f

#     sinogram = np.stack([n.s[:, 0, 0] for n in networks], axis=1)

#     return freqs, sinogram

def load_s1p_sinogram_and_coords(directory):
    files = glob.glob(os.path.join(directory, '*.s1p'))
    assert len(files) > 0, f"No .s1p files found in {directory}"

    # Sort files numerically by row, then col, using regex extraction
    def sort_key(f):
        match = re.search(r'Scan_(\d+)_(\d+)', os.path.basename(f))
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (0, 0)
    
    files = sorted(files, key=sort_key)

    networks = [skrf.Network(f) for f in files]
    freqs = networks[0].f
    sinogram = np.stack([n.s[:, 0, 0] for n in networks], axis=1)

    # Extract coordinates directly from the sorted filenames
    x_idx, y_idx = [], []
    for f in files:
        match = re.search(r'Scan_(\d+)_(\d+)', os.path.basename(f))
        y_idx.append(int(match.group(1))) # Row
        x_idx.append(int(match.group(2))) # Col
        
    probe_coordinates = np.array([x_idx, y_idx])

    return freqs, sinogram, probe_coordinates

def calibrate_sinograms(tar_dir, ref_dir):
    """Calibrates a target sinogram against a reference sinogram

    Loads target (tumour/object-present) and reference (empty-scene)
    .s1p sweeps from disk and subtracts the reference from the target,
    position-by-position, to remove static clutter/antenna coupling.

    Parameters
    ----------
    tar_dir : str
        Directory containing the target scan's .s1p files
    ref_dir : str
        Directory containing the reference scan's .s1p files

    Returns
    -------
    freqs : array_like
        Frequency points (Hz)
    cal_data : array_like
        Calibrated frequency-domain sinogram, shape (n_freqs, n_positions)
    """
    print("Beginning...Making Calibrated Sinogram...")

    #tar_freqs, tar_fd_data = load_s1p_sinogram(tar_dir)
    #ref_freqs, ref_fd_data = load_s1p_sinogram(ref_dir)

    tar_freqs, tar_fd_data, probe_coordinates_tar= load_s1p_sinogram_and_coords(tar_dir)
    if ref_dir is None:
        return tar_freqs, tar_fd_data, probe_coordinates_tar
    ref_freqs, ref_fd_data, probe_coordinates_ref = load_s1p_sinogram_and_coords(ref_dir)
    #if probe_coordinates_ref != probe_coordinates_tar:
    #    print("Reference and target scans do not have equal shape.")

    assert tar_fd_data.shape == ref_fd_data.shape, \
        "Target and reference sinograms must have matching shape " \
        f"(got {tar_fd_data.shape} vs {ref_fd_data.shape})"
    assert np.allclose(tar_freqs, ref_freqs), \
        "Target and reference frequency points do not match"

    cal_data = tar_fd_data - ref_fd_data

    return tar_freqs, cal_data, probe_coordinates_tar

def load_scan_metadata(scan_name, fallback_step_size=None):
    """Loads scan metadata or falls back gracefully to defaults."""
    scan_dir = f"{definitions.VNA_SAVE_FILE}/{scan_name}"
    meta_path = os.path.join(scan_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    
    # Fallback default if metadata file is missing
    return {
        "step_size_mm": fallback_step_size or definitions.STEP_SIZE
    }

# def get_probe_distances(probe_coordinates, num_pixels_x, num_pixels_y, 
#                         sample_width=definitions.RASTER_WIDTH, sample_height=definitions.RASTER_HEIGHT, standoff_z_mm = 3):
#     x_img = np.linspace(0, sample_width / 1000, num_pixels_x) # convert to m
#     y_img = np.linspace(0, sample_height / 1000, num_pixels_y) # convert to m

#     X, Y = np.meshgrid(x_img, y_img)
#     probe_x = probe_coordinates[0]
#     probe_y = probe_coordinates[1]

#     xi = np.asarray(probe_x) * definitions.STEP_SIZE / 1000 # convert to m
#     yi = np.asarray(probe_y) * definitions.STEP_SIZE / 1000 # convert to m

#     xi = xi.reshape(-1, 1, 1)
#     yi = yi.reshape(-1, 1, 1)

#     standoff_z_m = standoff_z_mm / 1000
#     distances = np.sqrt((X - xi)**2 + (Y - yi)**2 + standoff_z_m**2)

#     return distances

def get_probe_distances_3d(probe_coordinates, num_pixels_x, num_pixels_y, num_pixels_z,
                           z_min_mm=0.0, z_max_mm=100.0, margin_mm = 0.0, step_size=definitions.STEP_SIZE,
                           sample_width=definitions.RASTER_WIDTH, sample_height=definitions.RASTER_HEIGHT):
    """Calculates 3D spatial distance matrices and spatial axis vectors in mm."""
    x_vec_mm = np.linspace(-margin_mm, sample_width+margin_mm, num_pixels_x)
    y_vec_mm = np.linspace(-margin_mm, sample_height+margin_mm, num_pixels_y)
    z_vec_mm = np.linspace(z_min_mm, z_max_mm, num_pixels_z)

    # Convert to meters for distance calculation
    X, Y, Z = np.meshgrid(x_vec_mm / 1000.0, y_vec_mm / 1000.0, z_vec_mm / 1000.0, indexing='xy')

    probe_x = np.asarray(probe_coordinates[0]) * definitions.STEP_SIZE / 1000.0 # meters
    probe_y = np.asarray(probe_coordinates[1]) * definitions.STEP_SIZE / 1000.0 # meters

    xi = probe_x.reshape(-1, 1, 1, 1)
    yi = probe_y.reshape(-1, 1, 1, 1)

    distances = np.sqrt((X[None, ...] - xi)**2 + (Y[None, ...] - yi)**2 + Z[None, ...]**2)

    return distances, x_vec_mm, y_vec_mm, z_vec_mm

def get_scan_times(ini_f, fin_f, n_freqs):
    """Returns the times-of-response obtained after using the IDFT

    Gets the vector of time-points for the time-domain representation
    of the radar signals, when the IDFT is used to convert the data
    from the frequency to the time-domain.

    Parameters
    ----------
    ini_f : float
        The initial frequency, in Hz, used in the scan
    fin_f : float
        The final frequency, in Hz, used in the scan
    n_freqs : int
        The number of frequency points used in the scan

    Returns
    -------
    scan_times : array_like
        Vector of the time-points used to represent the radar signal
        when the IDFT is used to convert the data from the
        frequency-to-time domain, in seconds
    """
    # Get the vector of the frequencies used in the scan
    freqs = np.linspace(ini_f, fin_f, n_freqs)

    # Get the incremental frequency step-size
    freq_step = freqs[1] - freqs[0]

    # Compute the corresponding time-domain step-size when the IDFT is
    # used to convert the measured data to the time domain
    time_step = 1 / (n_freqs * freq_step)

    # Get the vector of the time-points used to represent the
    # IDFT-obtained signal
    scan_times = np.linspace(0, n_freqs * time_step, n_freqs)

    return scan_times

def phase_compensate(td_data, ini_f, ini_t, fin_t, n_time_pts):
    """Applies phase compensation to TD signals obtained with the ICZT

    Parameters
    ----------
    td_data : array_like
        Time-domain signals obtained via the ICZT
    ini_f : float
        Initial frequency used in the scan, in Hz
    ini_t : float
        Initial time point of the td_data, in seconds
    fin_t : float
        Final time point of the td_data, in seconds
    n_time_pts : int
        Number of time-points used to create the td_data

    Returns
    -------
    compensated_td_data : array_like
        Time-domain signals after phase compensation
    """

    n_dim = len(np.shape(td_data))

    assert n_dim in [1, 2], "td_data must be 1D or 2D arr"

    # Create vector of the time points used to represent the td_data
    time_vec = np.linspace(ini_t, fin_t, n_time_pts)

    # Phase correction factor
    phase_fac = np.exp(1j * 2 * np.pi * ini_f * time_vec)

    if n_dim == 1:  # If td_data was 1D arr

        compensated_td_data = td_data * phase_fac  # Apply to measured data

    else:  # If td_data was 2D arr

        compensated_td_data = td_data * phase_fac[:, None]

    return compensated_td_data

def iczt(fd_data, ini_t, fin_t, n_time_pts, ini_f, fin_f, axis=0):
    """Computes the ICZT of a 1D or 2D array

    Computes the inverse chirp z-transform (ICZT) on an array in the
    frequency domain, evaluating the transform at the specified
    n_time_pts between ini_t and fin_t.

    Parameters
    ----------
    fd_data : array_like
        The frequency-domain arr to be transformed via the ICZT to the
        time-domain
    ini_t : float
        The starting time-of-response to be used for computing the ICZT,
        in seconds
    fin_t : float
        The stopping time-of-response to be used for computing the ICZT,
        in seconds
    n_time_pts : int
        The number of points in the time-domain at which the transform
        will be evaluated
    ini_f : float
        The initial frequency used in the scan, in Hz
    fin_f : float
        The final frequency used in the scan, in Hz
    axis : int
        The axis along which to compute the transform

    Returns
    -------
    iczt_data : array_like
        Array of the transformed data, after applying the ICZT to the
        input fd_data
    """

    # Number of dimensions of fd_dat
    n_dimensions = len(np.shape(fd_data))

    # Assert fd_data is 1D or 2D
    assert n_dimensions in [1, 2], 'Error: fd_data not 1D or 2D array'

    if n_dimensions == 2:

        iczt_data = _iczt_two_dimension(fd_data, ini_t, fin_t, n_time_pts,
                                        ini_f, fin_f, axis=axis)
    else:

        # Find the number of frequencies use
        n_freqs = np.size(fd_data)

        # Find the conversion factor to convert from time-of-response to
        # angle around the unit circle
        time_to_angle = (2 * np.pi) / np.max(get_scan_times(ini_f, fin_f,
                                                            n_freqs))

        # Find the parameters for computing the ICZT over the specified
        # time window
        theta_naught = ini_t * time_to_angle
        phi_naught = (fin_t - ini_t) * time_to_angle / (n_time_pts - 1)

        # Compute the exponential values only once
        exp_theta_naught = np.exp(1j * theta_naught)
        exp_phi_naught = np.exp(1j * phi_naught)

        # Make a dummy vector to facilitate vectorized computation
        dummy_vec = np.arange(n_freqs)

        time_pts = np.arange(n_time_pts)  # Get time-index vector

        # Find the z-value matrix, to facilitate vectorized computation
        z_vals = exp_theta_naught * exp_phi_naught ** time_pts
        zs_power = np.power(z_vals[None, :], dummy_vec[:, None])

        # Compute the 1D ICZT, converting this frequency domain data
        # to the time domain
        iczt_data = _iczt_one_dimension(fd_data[:], zs_power, n_freqs)

    # Apply phase compensation
    iczt_data = phase_compensate(iczt_data, ini_f=ini_f, ini_t=ini_t,
                                 fin_t=fin_t, n_time_pts=n_time_pts)

    return iczt_data

def _iczt_two_dimension(fd_data, ini_t, fin_t, n_time_pts, ini_f, fin_f,
                        axis=0):
    """Computes the ICZT of a 2D-array

    Computes the inverse chirp z-transform (ICZT) on a 2D array in the
    frequency domain, evaluating the transform at the specified
    n_time_pts between ini_t and fin_t.

    Parameters
    ----------
    fd_data : array_like
        The frequency-domain arr to be transformed via the ICZT to the
        time-domain
    ini_t : float
        The starting time-of-response to be used for computing the ICZT,
        in seconds
    fin_t : float
        The stopping time-of-response to be used for computing the ICZT,
        in seconds
    n_time_pts : int
        The number of points in the time-domain at which the transform
        will be evaluated
    ini_f : float
        The initial frequency used in the scan, in Hz
    fin_f : float
        The final frequency used in the scan, in Hz
    axis : int
        The axis along which to compute the transform

    Returns
    -------
    iczt_data : array_like
        Array of the transformed data, after applying the ICZT to the
        input fd_data
    """

    # Assert that this is only for 2D arrays
    assert axis in [0, 1], 'Axis error: axis must be in [0, 1]'

    # Find the number of frequencies use
    n_freqs = np.size(fd_data, axis=0)

    # Find the conversion factor to convert from time-of-response to
    # angle around the unit circle
    time_to_angle = (2 * np.pi) / np.max(get_scan_times(ini_f, fin_f, n_freqs))

    # Find the parameters for computing the ICZT over the specified time
    # window
    theta_naught = ini_t * time_to_angle
    phi_naught = (fin_t - ini_t) * time_to_angle / (n_time_pts - 1)

    # Compute the exponential values only once
    exp_theta_naught = np.exp(1j * theta_naught)
    exp_phi_naught = np.exp(1j * phi_naught)

    # Make a dummy vector to facilitate vectorized computation
    dummy_vec = np.arange(n_freqs)

    time_pts = np.arange(n_time_pts)  # Get time-index vector

    # Find the z-value matrix, to facilitate vectorized computation
    z_vals = exp_theta_naught * exp_phi_naught**time_pts
    zs_power = np.power(z_vals[None, :], dummy_vec[:, None])

    # If wanting to compute the transform along the 0th axis
    if axis == 0:

        # Init return arr
        iczt_data = np.zeros([n_time_pts, np.size(fd_data, axis=1)],
                             dtype=np.complex64)

        # For every point along the other dimension
        for ii in range(np.size(fd_data, axis=1)):

            # Compute the 1D ICZT, converting this frequency domain data
            # to the time domain
            iczt_data[:, ii] = _iczt_one_dimension(fd_data[:, ii], zs_power,
                                                   n_freqs)

    else:  # If wanting to compute along the 1st axis
        iczt_data = np.zeros([np.size(fd_data, axis=0), n_time_pts],
                             dtype=np.complex64)

        # For every point along the other dimension
        for ii in range(np.size(fd_data, axis=0)):

            # Compute the 1D ICZT, converting this frequency domain data
            # to the time domain
            iczt_data[ii, :] = _iczt_one_dimension(fd_data[ii, :], zs_power,
                                                   n_freqs)

    return iczt_data

def _iczt_one_dimension(fd_data, zs_power, n_freqs):
    """Computes the ICZT of a 1D array

    Computes the inverse chirp z-transform (ICZT) on a 1D vector in the
    frequency domain, evaluating the transform at the specified
    n_time_pts between ini_t and fin_t, defined by zs_power

    Parameters
    ----------
    fd_data : array_like
        The frequency-domain vector to be transformed via the ICZT to
        the time-domain
    zs_power : array_like
        2D z-matrix, created in _iczt_two_dimension to facilitate
        vectorized computation
    n_freqs : int
        The number of frequencies used

    Returns
    -------
    iczt_data : array_like
        1D arr of the transformed data, after applying the ICZT to the
        input fd_data
    """

    # Compute the ICZT
    iczt_data = np.sum(fd_data[:, None] * zs_power, axis=0) / n_freqs

    return iczt_data

# def back_proj(td_data, pix_angs, pix_dists, prop_speed, ini_t, fin_t,
#               use_dmas=False):
#     """Back-projects time-domain data onto the image-space

#     Performs back-projection using time-domain data. Uses either the
#     DAS or DMAS beamformer as its basis.

#     Parameters
#     ----------
#     td_data : array_lke
#         The time-domain data (sinogram) that will be back-projected into
#         the image space
#     pix_angs : array_like
#         The angles off of the central axis of the antenna of every
#         pixel, for each antenna position, in degrees
#     pix_dists : array_like
#         The distances between each pixel and the antenna, for each
#         antenna position, in meters
#     prop_speed : float
#         The estimated propagation speed of the microwave signal in m/s
#     ini_t : float
#         The initial time (s) of the td_data (i.e., the first element
#         in each radar signal corresponds to the ini_t of response)
#     fin_t : float
#         The final time (s) of the td_data (i.e., the last element in
#         each radar signal corresponds to the fin_t of response)
#     use_dmas : bool
#         If True, uses the delay-multiply-and-sum beamformer as the
#         back-projection operator. Otherwise, uses the delay-and-sum
#         beamformer

#     Returns
#     -------
#     back_projection : array_like
#         The back-projection, in the image-space, of the td_data
#     """

#     td_data = np.abs(td_data)  # Take the abs-value

#     # Find the number of points in the time-domain used to represent the
#     # signal, and the number of antenna positions
#     n_time_pts, n_ant_pos = td_data.shape

#     # Find the vector of the time-points used to represent the time
#     # domain signal, and the incremental step-size
#     scan_times = np.linspace(ini_t, fin_t, n_time_pts)
#     scan_time_step = np.abs(scan_times[1] - scan_times[0])

#     # Init arr for storing the back-projection from each antenna
#     # position
#     # back_projections = np.zeros_like(probe_coordinates)
#     back_projections = np.zeros_like(pix_angs)

#     # Init arr to return
#     # back_projection = np.zeros(np.shape(probe_coordinates)[1:])
#     back_projection = np.zeros(np.shape(pix_angs)[1:])

#     # pix_front_of_ant = np.abs(pix_dists) > 0.03

#     # Find the response-times for every pixel
#     pix_times = (2.0 * pix_dists) / prop_speed

#     #pix_times[np.abs(pix_dists) <= 0.038] = np.nan

#     # Find the upper and lower bounds on the time-of-response estimates
#     # for each pixel
#     upper_pix_times = scan_times + 0.5 * scan_time_step
#     lower_pix_times = scan_times - 0.5 * scan_time_step

#     # For every antenna position used in the scan
#     for ant_pos in range(n_ant_pos):

#         # Find the response times of each pixel, for this antenna
#         # position
#         pix_times_here = pix_times[ant_pos, :, :]

#         # For every time-point of response
#         for time_pt in range(np.size(td_data, axis=0)):

#             # Find the coordinates of the pixels that correspond to this
#             # time-of-response
#             resp_idxs = np.logical_and(pix_times_here
#                                        > lower_pix_times[time_pt],
#                                        pix_times_here
#                                        < upper_pix_times[time_pt])

#             # Get the value that will be back-projected to the pixels at
#             # the resp_idxs
#             back_proj_val = td_data[time_pt, ant_pos]

#             # Back-project the value to the image-space
#             # back_projections[ant_pos, :, :][pix_front_of_ant[ant_pos]] += back_proj_val
#             back_projections[ant_pos, :, :][resp_idxs] += back_proj_val

#     # If using the DMAS beamformer as the foundation of the
#     # back-projection method
#     if use_dmas:

#         # For every antenna position, perform pair-wise signal
#         # multiplication with every other antenna position
#         for ant_pos in range(n_ant_pos):
#             for second_ant_pos in range(ant_pos + 1, n_ant_pos):

#                 # Perform pair-wise signal multiplication before
#                 # summation
#                 back_projection += (back_projections[ant_pos, :, :]
#                                     * back_projections[second_ant_pos, :, :])

#     else:  # Use the DAS as the foundation of the back-projection method

#         # Sum the back-projections from each antenna position
#         back_projection = np.sum(back_projections, axis=0)

#     return back_projection

# def back_proj_3d(td_data, pix_dists, prop_speed, ini_t, fin_t, antenna_delay=0.0, use_dmas=True):
#     """Back-projects complex time-domain data into a 3D voxel volume while preserving phase coherence.

#     Parameters
#     ----------
#     td_data : ndarray
#         Complex time-domain signal from ICZT, shape (n_time_pts, n_ant_pos)
#     pix_dists : ndarray
#         3D distance matrix, shape (n_ant_pos, ny, nx, nz) in meters
#     prop_speed : float
#         Propagation velocity in m/s (3e8 for free space)
#     ini_t, fin_t : float
#         Time window start and end in seconds
#     antenna_delay : float, optional
#         Internal antenna delay in seconds (round-trip time offset)
#     use_dmas : bool
#         If True, applies 3D Delay-Multiply-and-Sum beamforming

#     Returns
#     -------
#     vol_3d : ndarray
#         Reconstructed 3D volume intensity matrix, shape (ny, nx, nz)
#     """
#     n_time_pts, n_ant_pos = td_data.shape
#     scan_times = np.linspace(ini_t, fin_t, n_time_pts)

#     # Compute expected round-trip travel time (2 * d / v + antenna internal delay)
#     pix_times = (2.0 * pix_dists) / prop_speed + antenna_delay

#     ny, nx, nz = pix_dists.shape[1:]
#     # Allocate complex array to preserve phase during back-projection
#     back_projections = np.zeros((n_ant_pos, ny, nx, nz), dtype=np.complex64)

#     # Interpolate complex time-domain values at exact pixel travel times per probe position
#     for ant_pos in range(n_ant_pos):
#         signal = td_data[:, ant_pos]
#         t_flattened = pix_times[ant_pos].ravel()

#         # Interpolate real and imaginary components separately to preserve phase
#         interp_real = np.interp(t_flattened, scan_times, np.real(signal), left=0.0, right=0.0)
#         interp_imag = np.interp(t_flattened, scan_times, np.imag(signal), left=0.0, right=0.0)
        
#         back_projections[ant_pos] = (interp_real + 1j * interp_imag).reshape(ny, nx, nz)

#     if use_dmas:
#         # Coherent Complex DMAS: 0.5 * [ |Sum S_i|^2 - Sum |S_i|^2 ]
#         sum_s = np.sum(back_projections, axis=0)
#         sum_s_mag_sq = np.abs(sum_s)**2
#         sum_s_sq_mag = np.sum(np.abs(back_projections)**2, axis=0)
        
#         vol_3d = 0.5 * (sum_s_mag_sq - sum_s_sq_mag)
#         vol_3d = np.maximum(vol_3d, 0.0)  # Suppress slight floating-point negative residuals

#         # Normalize by the number of multiplied antenna pairs
#         n_pairs = (n_ant_pos * (n_ant_pos - 1)) / 2.0
#         vol_3d = vol_3d / n_pairs
#     else:
#         # Standard Coherent Delay-and-Sum
#         vol_3d = np.abs(np.sum(back_projections, axis=0))

#     return vol_3d

def plot_sample_time_signals(td_data, ini_t, fin_t, probe_coordinates, filename,
                              n_signals=5):
    """Plots a handful of time-domain signals from across the scan grid.

    td_data : complex array, shape (n_time_pts, n_positions) -- iczt() output
    probe_coordinates : [x_idx, y_idx], same order as td_data's position axis
    """
    n_time_pts, n_positions = td_data.shape
    time_axis = np.linspace(ini_t, fin_t, n_time_pts) * 1e9  # ns

    # Evenly spaced positions across the scan, not just the first few
    pos_idxs = np.linspace(0, n_positions - 1, n_signals, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in pos_idxs:
        x, y = probe_coordinates[0][idx], probe_coordinates[1][idx]
        ax.plot(time_axis, np.abs(td_data[:, idx]), label=f"pos ({x},{y})")

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Magnitude")
    ax.set_title("Time-domain signals across scan grid")
    ax.legend()
    ax.grid(True, alpha=0.3)

    filepath = f"{definitions.PROCESS_SAVE_FILE}/td_signals/{filename}.png"
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

# def das(td_data, probe_coordinates, ini_t, fin_t, ant_rad, speed, m_size, ini_ant_ang=-130.0, 
#         sample_width=definitions.RASTER_WIDTH, sample_height=definitions.RASTER_HEIGHT, standoff_z_mm=3, use_dmas=True):
#     """Reconstructs an image using the delay-and-sum (DAS) beamformer

#     Uses the delay-and-sum beamformer to reconstruct an image.

#     Parameters
#     ----------
#     td_data : array_lke
#         The time-domain data (sinogram) that will be back-projected into
#         the image space
#     ini_t : float
#         The initial time (s) of the return_td (i.e., the first element
#         in each radar signal corresponds to the ini_t of response)
#     fin_t : float
#         The final time (s) of the return_td (i.e., the last element in
#         each radar signal corresponds to the fin_t of response)
#     ant_rad : float
#                 The radius of the antennas trajectory during the scan,
#         after accounting for phase-delay using apply_ant_t_delay() func
#     speed : float
#         The estimated propagation speed of the microwave signal in m/s
#     m_size : int
#         The number of pixels along each dimension used when forming the
#         reconstruction
#     ini_ant_ang : float
#         The angle off of the positive x-axis of the *initial* antenna
#         position during the scan

#     Returns
#     -------
#     img : array_like
#         The back-projection, in the image-space, of the return_td
#     """
#     distances = get_probe_distances(probe_coordinates, m_size, m_size, sample_width=sample_width, sample_height=sample_height, standoff_z_mm=standoff_z_mm)
#     # Back-project using the DAS beamformer
#     img = back_proj(td_data, distances, distances, speed, ini_t, fin_t, use_dmas=use_dmas)
#     # img = back_proj(td_data, pix_angs, distances, speed, ini_t, fin_t, use_dmas=True)

#     return img

# def das_3d(td_data, probe_coordinates, ini_t, fin_t, speed,
#            m_size_x=100, m_size_y=100, m_size_z=50,
#            z_min_mm=70.0, z_max_mm=120.0, margin_mm=0.0,
#            sample_width=definitions.RASTER_WIDTH, sample_height=definitions.RASTER_HEIGHT,
#            antenna_delay=0.0, use_dmas=True, weighting_mode='hybrid', cf_power=1.0, dist_power=1.0,):
#     """Wrapper function for 3D DAS / DMAS volumetric reconstruction."""
#     distances, x_vec_mm, y_vec_mm, z_vec_mm = get_probe_distances_3d(
#         probe_coordinates,
#         num_pixels_x=m_size_x,
#         num_pixels_y=m_size_y,
#         num_pixels_z=m_size_z,
#         z_min_mm=z_min_mm,
#         z_max_mm=z_max_mm,
#         sample_width=sample_width,
#         sample_height=sample_height,
#         margin_mm=margin_mm
#     )

#     vol_3d = back_proj_3d(
#         td_data, 
#         distances, 
#         speed, 
#         ini_t, 
#         fin_t, 
#         antenna_delay=antenna_delay, 
#         use_dmas=use_dmas,
#         weighting_mode=weighting_mode,
#         cf_power=cf_power,
#         dist_power=dist_power
#     )

#     return vol_3d, x_vec_mm, y_vec_mm, z_vec_mm

def svd_clutter_removal(td_data, num_components_to_remove=1):
    """Removes dominant early-time skin clutter using SVD/PCA.
    
    Parameters
    ----------
    td_data : ndarray
        Complex or real time-domain signals, shape (n_time_pts, n_ant_pos)
    num_components_to_remove : int
        Number of dominant singular vectors to zero out (usually 1 or 2)
    """
    # Compute Singular Value Decomposition
    U, S, Vh = np.linalg.svd(td_data, full_matrices=False)
    
    # Zero out the dominant skin component(s)
    S_filtered = S.copy()
    S_filtered[:num_components_to_remove] = 0.0
    
    # Reconstruct signal matrix without dominant clutter
    td_clean = np.dot(U, np.dot(np.diag(S_filtered), Vh))
    return td_clean

def auto_time_gate(td_data, scan_times, pulse_duration_ns=0.8e-9):
    """Dynamically applies a smooth time window that mutes early skin response 
    and late-time noise automatically.
    """
    n_time_pts, n_ant_pos = td_data.shape
    gated_data = np.zeros_like(td_data)
    
    for ant in range(n_ant_pos):
        signal = td_data[:, ant]
        envelope = np.abs(hilbert(np.real(signal)))
        
        # 1. Detect skin peak index
        skin_peak_idx = np.argmax(envelope)
        t_skin = scan_times[skin_peak_idx]
        
        # 2. Gate starts after skin reflection duration
        t_start_gate = t_skin + pulse_duration_ns
        
        # 3. Create a smooth Tukey or logistic ramp-on mask
        mask = 1.0 / (1.0 + np.exp(-15 * (scan_times - t_start_gate) / pulse_duration_ns))
        
        gated_data[:, ant] = signal * mask
        
    return gated_data

def compute_coherence_factor(back_projections, eps=1e-12):
    """Computes the spatial Coherence Factor (CF) map across all antenna positions.

    Parameters
    ----------
    back_projections : ndarray
        Complex signals per antenna, shape (n_ant_pos, ny, nx, nz)
    eps : float
        Small offset to avoid zero division

    Returns
    -------
    cf_map : ndarray
        Coherence factor map bounded in [0, 1], shape (ny, nx, nz)
    """
    n_ant_pos = back_projections.shape[0]
    
    # Coherent sum squared magnitude: | \sum S_i |^2
    coherent_energy = np.abs(np.sum(back_projections, axis=0)) ** 2
    
    # Incoherent sum of squared magnitudes: N * \sum |S_i|^2
    incoherent_energy = n_ant_pos * np.sum(np.abs(back_projections) ** 2, axis=0)
    
    # Calculate CF
    cf_map = coherent_energy / (incoherent_energy + eps)
    return np.clip(cf_map, 0.0, 1.0)

def back_proj_3d(td_data, pix_dists, prop_speed, ini_t, fin_t, antenna_delay=0.0,
                      weighting_mode='none', cf_power=1.0, dist_power=1.0, use_dmas=False):
    """Back-projects complex time-domain data into 3D volume using DAS/DMAS combined with 
    Distance Weighting, Coherence Factor (CF) Weighting, or Hybrid WDAS.

    Parameters
    ----------
    td_data : ndarray
        Complex time-domain signals from ICZT, shape (n_time_pts, n_ant_pos)
    pix_dists : ndarray
        Distance matrix, shape (n_ant_pos, ny, nx, nz) in meters
    prop_speed : float
        Propagation velocity in m/s
    ini_t, fin_t : float
        Time window boundaries in seconds
    antenna_delay : float
        Internal antenna delay in seconds
    weighting_mode : str
        'none'     -> Standard DAS / DMAS
        'distance' -> Distance path-loss weighted back-projection
        'cf'       -> Coherence Factor weighted reconstruction
        'hybrid'   -> Combined Distance Compensation and CF Weighting
    cf_power : float
        Exponent applied to the Coherence Factor map (default 1.0)
    dist_power : float
        Path-loss distance compensation factor exponent (default 1.0)
    use_dmas : bool
        If True, uses DMAS as base beamformer; otherwise uses standard DAS.

    Returns
    -------
    vol_3d : ndarray
        Reconstructed 3D volume, shape (ny, nx, nz)
    """
    n_time_pts, n_ant_pos = td_data.shape
    scan_times = np.linspace(ini_t, fin_t, n_time_pts)

    # Compute expected round-trip travel time
    pix_times = (2.0 * pix_dists) / prop_speed + antenna_delay
    ny, nx, nz = pix_dists.shape[1:]

    # 1. Compute Distance Weighting factor per probe element
    if weighting_mode in ['distance', 'dist', 'hybrid']:
        # Compensate for spherical spreading attenuation ~ d^alpha
        dist_weights = pix_dists ** dist_power
    elif weighting_mode in ['none', 'cf']:
        dist_weights = np.ones_like(pix_dists)
    else: 
        print(f"Warning: Given weighting mode {weighting_mode} not in [distance, cf, hybrid, none]")

    back_projections = np.zeros((n_ant_pos, ny, nx, nz), dtype=np.complex64)

    # 2. Time-delay interpolation and distance-weighted back-projection
    for ant_pos in range(n_ant_pos):
        signal = td_data[:, ant_pos]
        t_flattened = pix_times[ant_pos].ravel()

        interp_real = np.interp(t_flattened, scan_times, np.real(signal), left=0.0, right=0.0)
        interp_imag = np.interp(t_flattened, scan_times, np.imag(signal), left=0.0, right=0.0)
        
        interp_signal = (interp_real + 1j * interp_imag).reshape(ny, nx, nz)
        
        # Apply element-level distance weighting
        back_projections[ant_pos] = interp_signal * dist_weights[ant_pos]

    # 3. Base Beamforming Reconstruction (DAS or DMAS)
    if use_dmas:
        sum_s = np.sum(back_projections, axis=0)
        sum_s_mag_sq = np.abs(sum_s)**2
        sum_s_sq_mag = np.sum(np.abs(back_projections)**2, axis=0)
        
        vol_3d = 0.5 * (sum_s_mag_sq - sum_s_sq_mag)
        vol_3d = np.maximum(vol_3d, 0.0)
        n_pairs = (n_ant_pos * (n_ant_pos - 1)) / 2.0
        vol_3d = vol_3d / n_pairs
    else:
        vol_3d = np.abs(np.sum(back_projections, axis=0))

    # 4. Apply Coherence Factor (CF) Weighting
    if weighting_mode in ['cf', 'hybrid']:
        cf_map = compute_coherence_factor(back_projections)
        vol_3d = vol_3d * (cf_map ** cf_power)

    return vol_3d

def das_3d(td_data, probe_coordinates, ini_t, fin_t, speed,
                m_size_x=100, m_size_y=100, m_size_z=50,
                z_min_mm=70.0, z_max_mm=120.0, margin_mm=0.0, step_size=definitions.STEP_SIZE,
                sample_width=definitions.RASTER_WIDTH, sample_height=definitions.RASTER_HEIGHT,
                antenna_delay=0.0, weighting_mode='none', cf_power=1.0, dist_power=1.0, use_dmas=True):
    """3D WDAS Volumetric Reconstruction wrapper supporting Hybrid, Distance, and CF Weighting."""
    
    distances, x_vec_mm, y_vec_mm, z_vec_mm = get_probe_distances_3d(
        probe_coordinates,
        num_pixels_x=m_size_x,
        num_pixels_y=m_size_y,
        num_pixels_z=m_size_z,
        z_min_mm=z_min_mm,
        z_max_mm=z_max_mm,
        sample_width=sample_width,
        sample_height=sample_height,
        margin_mm=margin_mm,
        step_size=step_size
    )

    vol_3d = back_proj_3d(
        td_data=td_data, 
        pix_dists=distances, 
        prop_speed=speed, 
        ini_t=ini_t, 
        fin_t=fin_t, 
        antenna_delay=antenna_delay, 
        weighting_mode=weighting_mode,
        cf_power=cf_power,
        dist_power=dist_power,
        use_dmas=use_dmas
    )

    return vol_3d, x_vec_mm, y_vec_mm, z_vec_mm

def plot_img(img, num_rows, num_cols, tum_x=0.0, tum_y=0.0, tum_rad=0.0, adi_rad=0.0, ant_rad=0.0,
             save_str='', save_fig=False, cmap='inferno', max_val=1.0, min_val=0.0,
             title='', normalise=False, crop_img=False, cbar_fmt='%.1f',
             norm_cbar=False, cbar_max=1.0, transparent=False, dpi=300,
             save_close=True, roi_x = 0, roi_y = 0, roi_r = 5, annul_r_out = 10, annul_r_in = 5):
    """Displays a reconstruction, making a publication-ready figure

    Parameters
    ----------
    img : array_like
        The 2D reconstruction that will be displayed
    tum_x : float
        The x-position of the tumor in the scan, in meters
    tum_y : float
        The y-position of the tumor in the scan, in meters
    tum_rad : float
        The radius of the tumor in the scan, in meters
    adi_rad : float
        The approximate radius of the breast in the scan, in meters
    ant_rad : float
        The corrected antenna radius during the scan (corrected for
        antenna time-delay)
    save_str : str, optional
        The complete path for saving the figure as a .png - only used if
        save_fig is True
    save_fig : bool
        If True, will save the displayed image to the location specified
        in save_str
    cmap : str
        Specifies the colormap that will be used when displaying the
        image
    max_val : float
        The maximum intensity to display in the image
    title : str
        The title for the plot
    normalise : bool
        If True, will normalise the image to have the maximum value
        max_val
    crop_img : bool
        If True, will set the values in the image to NaN outside of the
        inner antenna trajectory
    cbar_fmt : str
        The format for the tick labels on the colorbar
    norm_cbar : bool
        If True, will normalise the colorbar to have min value 0,
        max value of cbar_max
    cbar_max : float
        If norm_cbar, this will be the maximum value of the cbar
    transparent : bool
        If True, will save the image with a transparent background
        (i.e., whitespace will be transparent)
    dpi : int
        The DPI to be used if saving the image
    save_close : bool
        If True, closes the figure after saving
    """

    ant_rad *= 100  # Convert from m to cm to facilitate plot

    # Define angles for plot the tissue geometry
    draw_angs = np.linspace(0, 2 * np.pi, 1000)

    if normalise:  # If normalizing the image

        # If the max value is 1.0, scale so that the intensity is
        # between 0 and 1.0
        if max_val == 1.0:
            img = img - np.min(img[np.logical_not(np.isnan(img))])
            img = img / np.max(img[np.logical_not(np.isnan(img))])
            # TODO: what is the difference between these two max values?
            # img = img / np.max(img)

        # If the max value isn't set to unity, then scale the image so
        # that intensity is between 0 and max_val
        else:
            img = img - min_val
            img = img / (max_val - min_val)

    epsilon = 1e-12
    img_safe = img + epsilon

    img_to_plot = wiener(img_safe, (5,5), 0.1)

    # Set the font to times new roman
    plt.rc('font', family='Times New Roman')
    # plt.figure()  # Make the figure window
    fig, ax = plt.subplots()  # <-- replace plt.figure()

    x_min = 0
    y_min = 0

    x_max = (num_cols - 1) * definitions.STEP_SIZE / 10  # convert mm to cm
    y_max = (num_rows - 1) * definitions.STEP_SIZE / 10  # convert mm to cm

    # If normalizing the image, ensure the cbar axis extends from 0 to 1
    if normalise:
        ax.imshow(20*np.log10(img_to_plot), extent=[x_min, x_max, y_min, y_max], cmap=cmap, aspect='equal', vmin=20*np.log10(min_val), vmax=20*np.log10(max_val), origin='lower')
        # plt.imshow(20*np.log10(img_to_plot), cmap=cmap,
        #            aspect='equal', vmin=20*np.log10(min_val), vmax=20*np.log10(max_val))
        # plt.imshow(img_to_plot, cmap=cmap,
        #            aspect='equal', vmin=min_val, vmax=max_val)
    elif norm_cbar:
        ax.imshow(img_to_plot, cmap=cmap, extent=[x_min, x_max, y_min, y_max],
                    aspect='equal', vmin=0, vmax=cbar_max, origin='lower')
    # else:  # If nor normalizing the image
    #     plt.imshow(img_to_plot, cmap=cmap, extent=tick_bounds, aspect='equal')
    else:  # If nor normalizing the image
        ax.imshow(20*np.log10(img_to_plot), cmap=cmap, extent=[x_min, x_max, y_min, y_max], aspect='equal', origin='lower')

    # Set the size of the axis tick labels
    # plt.tick_params(labelsize=14)
    ax.tick_params(labelsize=14)

    # Set the x/y-ticks at multiples of 5 cm
    # plt.gca().set_xticks([-10, -5, 0, 5, 10])
    # plt.gca().set_yticks([-10, -5, 0, 5, 10])

    # Specify the colorbar tick format and size

    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))  # switch to sci-notation outside [1e-3, 1e3]
    # plt.colorbar(format=fmt, label ="Reflection magnitude [dB]").ax.tick_params(labelsize=14)

    # cbar = plt.colorbar(format=fmt)          # create colorbar
    cbar = fig.colorbar(ax.images[0], ax=ax, format=fmt)
    cbar.set_label("Reflection magnitude [dB]", fontsize=16)
    cbar.ax.tick_params(labelsize=14)            # ticks stay separate

    # Optional sanity: ensure view matches your extent
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # then draw using *_cm
    #ax.add_patch(Circle((roi_x, roi_y), roi_r, fill=False, linewidth=2, edgecolor="red", zorder=10))
    #ax.add_patch(Circle((roi_x, roi_y), annul_r_in,  fill=False, linewidth=2, linestyle="--", edgecolor="red", zorder=10))
    #ax.add_patch(Circle((roi_x, roi_y), annul_r_out, fill=False, linewidth=2, linestyle="--", edgecolor="red", zorder=10))

    plt.title(title, fontsize=20)  # Make the plot title
    plt.xlabel('x-axis [cm]', fontsize=16)  # Make the x-axis label
    plt.ylabel('y-axis [cm]', fontsize=16)  # Make the y-axis label
    plt.tight_layout()  # Remove excess whitespace in the figure

    # plt.show()

    # If saving the image, save it to the save_str path and close it
    if save_fig:
        p = Path(save_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_str, transparent=transparent, dpi=dpi,
                    bbox_inches='tight')

        if save_close:  # If wanting to close the fig after saving
            plt.close() 

def save_3d_volume(vol_3d, x_vec_mm, y_vec_mm, z_vec_mm, save_dir, step_size_mm=None):
    """Saves reconstructed 3D volume and spatial grid metadata to .npz file."""
    filename = f"{save_dir}.npz"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    save_kwargs = {
        'vol_3d': vol_3d,
        'x_vec_mm': x_vec_mm,
        'y_vec_mm': y_vec_mm,
        'z_vec_mm': z_vec_mm,
    }
    
    # Store step_size_mm inside the archive if provided
    if step_size_mm is not None:
        save_kwargs['step_size_mm'] = np.float32(step_size_mm)
        
    np.savez_compressed(filename, **save_kwargs)
    print(f"Saved volume archive to {filename}")

def build_raster_probe_coordinates(rows, cols):
    """Builds probe index coordinates matching the rig's serpentine
    (boustrophedon) raster pattern: odd rows traversed in reverse.

    Must match the actual order .s1p files were saved in during the scan.

    Returns
    -------
    probe_coordinates : array_like
        Shape (2, n_positions): [x_indices, y_indices]
    """
    x_idx = []
    y_idx = []
    for row in range(rows):
        col_order = range(cols) if row % 2 == 0 else reversed(range(cols))
        for col in col_order:
            x_idx.append(col)
            y_idx.append(row)
    return np.array([x_idx, y_idx])

def subtract_saved_volumes(vol1_name, vol2_name, output_name):
    """
    Loads two 3D volumes saved by img_constr2_3.py, subtracts vol2 from vol1,
    and saves the resulting volume with its original coordinate vectors.
    
    Parameters
    ----------
    vol1_path : str
        Name of .npz file (e.g., target volume).
    vol2_path : str
        Name of the second .npz file (e.g., reference volume).
    output_path : str
        Name to save the subtracted .npz file.
    """
    # Load the compressed .npz archives
    # Note: .npz extension is automatically appended if missing by np.savez, 
    # but we should ensure we are loading the correct path.
    vol1_path = f"{definitions.PROCESS_SAVE_FILE}/volumes/{vol1_name}"
    vol2_path = f"{definitions.PROCESS_SAVE_FILE}/volumes/{vol2_name}"
    output_path = f"{definitions.PROCESS_SAVE_FILE}/volumes/{output_name}"


    if not str(vol1_path).endswith('.npz'): vol1_path = f"{vol1_path}.npz"
    if not str(vol2_path).endswith('.npz'): vol2_path = f"{vol2_path}.npz"
        
    data1 = np.load(vol1_path)
    data2 = np.load(vol2_path)
    
    # Extract the 3D volume matrices
    vol1 = data1['vol_3d']
    vol2 = data2['vol_3d']
    
    # Ensure dimensions match before subtracting
    if vol1.shape != vol2.shape:
        raise ValueError(f"Volume shapes do not match: {vol1.shape} vs {vol2.shape}")
        
    # Subtract the volumes
    diff_vol = vol1 - vol2
    
    # Ensure the output directory exists
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the result keeping the coordinate vectors intact
    np.savez_compressed(
        output_path, 
        vol_3d=diff_vol, 
        x_vec_mm=data1['x_vec_mm'], 
        y_vec_mm=data1['y_vec_mm'], 
        z_vec_mm=data1['z_vec_mm']
    )
    print(f"Saved subtracted 3D volume to {output_path}.npz")

def process_scan(tar_name, ref_name=None, vox_x=50, vox_y=50, vox_z=50, margin_mm = 0.0, 
                 z_min=30.0, z_max=200.0, prop_speed=3e8, antenna_min_freq = definitions.ANTENNA_MIN_FREQ,
                 antenna_max_freq = definitions.ANTENNA_MAX_FREQ, antenna_delay=0.5e-9, start_time=None, end_time=None, 
                 use_ref=True, cf_power=1.0, dist_power=1.0, weighting_mode='hybrid', use_svd=True, use_auto_time_gate=False,
                 pulse_duration_ns=0.8e-9, save_name=None, use_spatial_mean_subtraction=False):
    if ref_name is None and use_ref:
        ref_name = f"{tar_name}_ref"
    C = 3e8
    if start_time is None:
        start_time = max((z_min / 1000 / C) + antenna_delay - 0.2e-9, 0.1e-9)
    if end_time is None:
        end_time = min(z_max / 1000 / C + antenna_delay + 0.5e-9, 5e-9)

    if save_name is None:
        save_name = tar_name

    if use_ref:
        print(f"Reconstructing {tar_name} using {ref_name} as reference")
        ref_dir = f"{definitions.VNA_SAVE_FILE}/{ref_name}"
    else:
        print(f"Reconstructing {tar_name} using no reference")
        ref_dir = None

    tar_dir = f"{definitions.VNA_SAVE_FILE}/{tar_name}"
    save_dir = f"{definitions.PROCESS_SAVE_FILE}/volumes/{save_name}"

    # --- Read Step Size automatically from scan metadata ---
    meta = load_scan_metadata(tar_dir, fallback_step_size=definitions.STEP_SIZE)
    step_size_mm = meta.get("step_size_mm", definitions.STEP_SIZE)
    print(f"Loaded step size: {step_size_mm} mm")

    # 1. Calibrate target against empty scene
    freqs, cal_data, probe_coordinates = calibrate_sinograms(tar_dir, ref_dir)
    
    cols = int(np.max(probe_coordinates[0])) + 1
    rows = int(np.max(probe_coordinates[1])) + 1

    valid_idx = (freqs >= antenna_min_freq) & (freqs <= antenna_max_freq)

    freqs_clean = freqs[valid_idx]
    cal_data_clean = cal_data[valid_idx, :]

    if use_spatial_mean_subtraction:
        spatial_mean = np.mean(cal_data_clean, axis=1, keepdims=True)
        cal_data_clean = cal_data_clean - spatial_mean
    
    if use_svd:
        # Remove clutter
        cal_data_clean = svd_clutter_removal(cal_data_clean, num_components_to_remove=1)

    print("Computing ICZT time domain signal...")
    td_data = iczt(cal_data_clean, ini_t=start_time, fin_t=end_time,
                    n_time_pts=definitions.N_TIME_POINTS, ini_f=freqs_clean[0], fin_f=freqs_clean[-1],
                    axis=0)
    if use_auto_time_gate:
        scan_times = np.linspace(start_time, end_time, definitions.N_TIME_POINTS)
        td_data = auto_time_gate(td_data, scan_times, pulse_duration_ns=pulse_duration_ns)


    plot_sample_time_signals(td_data, ini_t=start_time, fin_t=end_time, probe_coordinates=probe_coordinates, filename=tar_name)

    # 2. Perform 3D Coherent DAS / DMAS Reconstruction
    print("Reconstructing 3D volume via 3D Coherent DMAS...")
    vol_3d, x_vec_mm, y_vec_mm, z_vec_mm = das_3d(
        td_data=td_data,
        probe_coordinates=probe_coordinates,
        ini_t=start_time,
        fin_t=end_time,
        speed=prop_speed,
        m_size_x=vox_x,
        m_size_y=vox_y,
        m_size_z=vox_z,
        z_min_mm=z_min,
        z_max_mm=z_max,
        sample_width=(cols - 1) * step_size_mm,
        sample_height=(rows - 1) * step_size_mm,
        antenna_delay=antenna_delay,
        use_dmas=True,
        margin_mm=margin_mm,
        cf_power=cf_power,
        dist_power=dist_power,
        weighting_mode=weighting_mode,
        step_size=step_size_mm
    )

    # 3. Save 3D raw volume to disk (.npz format)
    save_3d_volume(vol_3d, x_vec_mm, y_vec_mm, z_vec_mm, save_dir, step_size_mm=step_size_mm)

def main():
    z_min = 80.0
    z_max = 220.0
    time_gate_ini = 0.1e-9
    margin_mm = 20
    antenna_delay = 0.55e-9
    target = "PhantomFoil_2"
    reference = "PhantomFoil_2_ref"
    use_ref = True
    weighting_mode = 'none'
    cf_power=1.0
    dist_power=1.0
    use_svd = False
    use_auto_time_gate = False
    pulse_duration_ns = 0.7e-9
    process_scan(tar_name=target, ref_name=reference, z_min=z_min, z_max=z_max, 
                 start_time=time_gate_ini, margin_mm=margin_mm, antenna_delay=antenna_delay,
                 use_ref=use_ref, weighting_mode=weighting_mode, cf_power=cf_power, dist_power=dist_power,
                 use_svd=use_svd, use_auto_time_gate=use_auto_time_gate, pulse_duration_ns=pulse_duration_ns)
    
    # tar_dir = 'ScanOutputs/RollAlu_2'       # Target scan (with aluminum cylinder)
    # ref_dir = 'ScanOutputs/RollAlu_2_ref'       # Reference scan (free space)
    # vol_str = 'Outputs/volumes/RollAlu_2'

    # m_size_x = 50       # Lateral X resolution
    # m_size_y = 50       # Lateral Y resolution
    # m_size_z = 50        # Depth Z resolution
    # z_min_mm = 50.0      # Minimum depth (mm)
    # z_max_mm = 200.0     # Maximum depth (mm)
    # speed = 3e8          # Speed of light in free space (m/s)

    # # Internal antenna delay in seconds (adjust if target depth appears shifted in Z)
    # # e.g., 0.3e-9 = 0.3 ns internal delay
    # antenna_delay = 0.4e-9  

    # # 1. Calibrate target against empty scene
    # freqs, cal_data, probe_coordinates = calibrate_sinograms(tar_dir, ref_dir)

    # cols = int(np.max(probe_coordinates[0])) + 1
    # rows = int(np.max(probe_coordinates[1])) + 1

    # start_time = 0.1e-9   # 0.1 ns
    # end_time = 1.7e-9     # 1.5 ns

    # # Truncate frequency array and sinogram data to >= 2.0 GHz
    # valid_idx = freqs >= 2.0e9

    # freqs_clean = freqs[valid_idx]
    # cal_data_clean = cal_data[valid_idx, :]

    # print("Computing ICZT time domain signal...")
    # td_data = iczt(cal_data_clean, ini_t=start_time, fin_t=end_time,
    #                n_time_pts=definitions.N_TIME_POINTS, ini_f=freqs_clean[0], fin_f=freqs_clean[-1],
    #                axis=0)


    # # 2. Perform 3D Coherent DAS / DMAS Reconstruction
    # print("Reconstructing 3D volume via 3D Coherent DMAS...")
    # vol_3d, x_vec_mm, y_vec_mm, z_vec_mm = das_3d(
    #     td_data=td_data,
    #     probe_coordinates=probe_coordinates,
    #     ini_t=start_time,
    #     fin_t=end_time,
    #     speed=speed,
    #     m_size_x=m_size_x,
    #     m_size_y=m_size_y,
    #     m_size_z=m_size_z,
    #     z_min_mm=z_min_mm,
    #     z_max_mm=z_max_mm,
    #     sample_width=(cols - 1) * definitions.STEP_SIZE,
    #     sample_height=(rows - 1) * definitions.STEP_SIZE,
    #     antenna_delay=antenna_delay,
    #     use_dmas=True
    # )

    # # 3. Save 3D raw volume to disk (.npz format)
    # save_3d_volume(vol_3d, x_vec_mm, y_vec_mm, z_vec_mm, vol_str)

# def main():
#     # --- Config: adjust paths and physical params for the current setup ---
#     tar_dir = 'ScanOutputs/mmTest6'       # .s1p files with tumour/object present
#     ref_dir = 'ScanOutputs/mmTest5'    # .s1p files, empty scene
#     save_str = 'Outputs/figures/Test6Recon'
#     vol_str = 'Outputs/volumes/Test6Recon'

#     #standoff_dist = 100      # in mm
#     #m_size = 100          # reconstruction grid resolution (pixels/side)
#     #ant_rad = 0.15         # TODO: set to your actual antenna trajectory radius, in m
#     m_size_x = 100       # Lateral X resolution
#     m_size_y = 100       # Lateral Y resolution
#     m_size_z = 50        # Depth Z resolution
#     z_min_mm = 70.0      # Minimum depth (mm)
#     z_max_mm = 120.0     # Maximum depth (mm)
#     speed = 3e8          # Medium propagation speed (m/s)[cite: 22]

#     # 1. Calibrate: subtract reference from target sinogram
#     freqs, cal_data, probe_coordinates = calibrate_sinograms(tar_dir, ref_dir)
#     #freqs, cal_data = load_s1p_sinogram(tar_dir)

#     cols = int(np.max(probe_coordinates[0])) + 1
#     rows = int(np.max(probe_coordinates[1])) + 1

#     # 2. Build probe coordinates matching scan order
#     #probe_coordinates = build_raster_probe_coordinates(rows, cols)
#     print("Built coordinates...")

#     start_time = 0.1e-9   # 0.1 ns
#     end_time = 1.5e-9     # 1.5 ns (adjust based on max target depth)

#     print("Computing ICZT time domain signal...")
#     td_data = iczt(cal_data, ini_t=start_time, fin_t=end_time,
#                    n_time_pts=N_TIME_POINTS, ini_f=freqs[0], fin_f=freqs[-1],
#                    axis=0)

#     # 3. Perform 3D DAS / DMAS Reconstruction
#     print("Reconstructing 3D volume via 3D DAS...")
#     vol_3d, z_vec_mm = das_3d(
#         td_data=td_data,
#         probe_coordinates=probe_coordinates,
#         ini_t=start_time,
#         fin_t=end_time,
#         speed=speed,
#         m_size_x=m_size_x,
#         m_size_y=m_size_y,
#         m_size_z=m_size_z,
#         z_min_mm=z_min_mm,
#         z_max_mm=z_max_mm,
#         sample_width=(cols - 1) * STEP_SIZE,
#         sample_height=(rows - 1) * STEP_SIZE,
#         use_dmas=True
#     )

#     # 4. Save 3D raw volume to disk (.npz format)
#     save_3d_volume(vol_3d, z_vec_mm, vol_str)

#     # SWEEP_STEP = 1e-10
#     # NUM_STEPS = 1
#     # SWEEP_START = 2e-10
#     # SWEEP_SPAN = 10e-10
    
#     # for i in range(NUM_STEPS):
#     #     print(f"Timegate {i}...")
#     #     start_time = i * SWEEP_STEP + SWEEP_START
#     #     end_time = start_time + SWEEP_SPAN
#     #     string_save = f"{save_str}{i}.png"
#     #     # 3. Time-gate: convert to time domain via ICZT
#     #     td_data = iczt(cal_data, ini_t=start_time, fin_t=end_time,
#     #                 n_time_pts=N_TIME_POINTS, ini_f=freqs[0], fin_f=freqs[-1],
#     #                 axis=0)
        
#     #     time_vector = np.linspace(start_time, end_time, N_TIME_POINTS)

#     #     plt.figure(figsize=(10, 6))

#     #     # Select a few positions to plot (e.g., first, middle, and last)
#     #     num_positions = td_data.shape[1]
#     #     positions_to_plot = [0, num_positions // 2, num_positions - 1]

#     #     for pos in positions_to_plot:
#     #         # Plotting the absolute value (envelope) of the complex time-domain data
#     #         # as used in the back_proj function
#     #         plt.plot(time_vector * 1e9, np.abs(td_data[:, pos]), label=f'Position Index {pos}')

#     #     plt.title("Time Domain Envelope for Selected Probe Positions")
#     #     plt.xlabel("Time (ns)")
#     #     plt.ylabel("Signal Amplitude")
#     #     plt.legend()
#     #     plt.grid(True, linestyle='--', alpha=0.7)
#     #     plt.tight_layout()
#     #     plt.savefig(f"Outputs/figures/td{i}.png", dpi=300, bbox_inches='tight')

#     #     # 4. Reconstruct image via DAS beamformer
#     #     img = das(td_data, probe_coordinates, ini_t=start_time, fin_t=end_time,
#     #             ant_rad=ant_rad, speed=speed, m_size=m_size, sample_width=(cols-1)*STEP_SIZE,
#     #               sample_height=(rows-1)*STEP_SIZE, standoff_z_mm=standoff_dist, use_dmas=True)

#     #     # 5. Plot
#     #     plot_img(img, rows, cols, ant_rad=ant_rad, save_str=string_save, save_fig=True,
#     #             title='Reconstruction', norm_cbar=False)

if __name__ == "__main__":
    main()