import os, definitions, webbrowser
import numpy as np
import plotly.graph_objects as go
from skimage.filters import threshold_otsu

import kaleido
from pathlib import Path

def prepare_volume_data(filename, use_Otsu=False, dynamic_range_db=15.0):
    """Loads npz data and converts to independent absolute/relative dB values."""
    npz_path = f'{definitions.PROCESS_SAVE_FILE}/volumes/{filename}.npz'
    data = np.load(npz_path)
    
    vol_3d = data['vol_3d']
    x_vec_mm = data['x_vec_mm']
    y_vec_mm = data['y_vec_mm']
    z_vec_mm = data['z_vec_mm']
    
    step_size_mm = float(data['step_size_mm']) if 'step_size_mm' in data else definitions.STEP_SIZE

    # Convert to dB scale
    peak_val = np.max(vol_3d)
    vol_db_abs = 10.0 * np.log10(vol_3d + 1e-12)
    vol_db_rel = 10.0 * np.log10((vol_3d / (peak_val + 1e-12)) + 1e-12)
    vmax = np.max(vol_db_abs)

    if use_Otsu:
        vol_clipped = np.maximum(vol_db_rel, -dynamic_range_db)
        otsu_thresh_rel = threshold_otsu(vol_clipped)
        vmin = vmax + otsu_thresh_rel
        vol_masked_rel = np.where(vol_db_rel >= otsu_thresh_rel, vol_db_rel, -dynamic_range_db)
        plot_values = vmax + vol_masked_rel
        effective_dr = abs(otsu_thresh_rel)
    else:
        effective_dr = dynamic_range_db
        vmin = vmax - dynamic_range_db
        plot_values = np.maximum(vol_db_abs, vmin)

    X, Y, Z = np.meshgrid(x_vec_mm, y_vec_mm, z_vec_mm, indexing='xy')
    
    return {
        'X': X.flatten(), 'Y': Y.flatten(), 'Z': Z.flatten(),
        'values': plot_values.flatten(),
        'vmin': vmin, 'vmax': vmax,
        'dr': effective_dr, 'step_size': step_size_mm
    }

def plot_overlay_volumes(filename1, filename2, dr_db1=10.0, dr_db2=6.0, 
                         use_Otsu1=False, use_Otsu2=False,
                         opacity1=0.125, opacity2=0.25):
    """Overlays two 3D volumes in a single scene with independent scaling and colorbars.

    Parameters
    ----------
    filename1 : str
        First volume dataset name (e.g., Target / Tumour Scan)
    filename2 : str
        Second volume dataset name (e.g., Reference / Background / Steel Scan)
    """
    print(f"Loading datasets for overlay: '{filename1}' and '{filename2}'...")
    d1 = prepare_volume_data(filename1, use_Otsu=use_Otsu1, dynamic_range_db=dr_db1)
    d2 = prepare_volume_data(filename2, use_Otsu=use_Otsu2, dynamic_range_db=dr_db2)

    # Trace 1: Target Scan (e.g., Warm Inferno colorscale)
    trace1 = go.Volume(
        x=d1['X'], y=d1['Y'], z=d1['Z'],
        value=d1['values'],
        isomin=d1['vmin'], isomax=d1['vmax'],
        cmin=d1['vmin'], cmax=d1['vmax'],
        opacity=opacity1,
        surface_count=12,
        colorscale='Inferno',
        name=f"Scan 1: {filename1}",
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(
            title=f"Unaltered",
            x=1.00,        # Positioned at right edge
            len=0.75
        )
    )

    # Trace 2: Overlay Scan (e.g., Cool Cyan/Viridis colorscale)
    trace2 = go.Volume(
        x=d2['X'], y=d2['Y'], z=d2['Z'],
        value=d2['values'],
        isomin=d2['vmin'], isomax=d2['vmax'],
        cmin=d2['vmin'], cmax=d2['vmax'],
        opacity=opacity2,
        surface_count=10,
        colorscale='Ice',  # Alternative high-contrast palettes: 'Viridis', 'Cyan', 'Electric'
        name=f"Scan 2: {filename2}",
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(
            title=f"Weighted & SVD [dB, a.u]",
            x=1.15,        # Offset further right to avoid colorbar overlap
            len=0.75
        )
    )

    fig = go.Figure(data=[trace1, trace2])

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=-2.0)
    )

    fig.update_layout(
        title=f"3D Dual Overlay: {filename1} (Top {d1['dr']:.1f}dB) vs {filename2} (Top {d2['dr']:.1f}dB)",
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z Depth [mm]',
            aspectmode='data',
            camera=camera
        ),
        margin=dict(l=0, r=150, b=0, t=50) # Added right margin for dual colorbars
    )

    out_dir = f"{definitions.PROCESS_SAVE_FILE}/figures/overlays"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{filename1}_VS_{filename2}.html")
    
    fig.write_html(out_path)
    print(f"Saved dual overlay plot to {out_path}")
    webbrowser.open(os.path.abspath(out_path))

    out_path_png = f"{Path(__file__).resolve().parent.parent.parent.parent}/Report/Figures/{filename1}_detection"
    fig.write_image(f"{out_path_png}.png", scale=1, width=900, height=500)
    print(f"Saved png to {out_path_png}.png")
    fig.update_layout(
        scene=dict(
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-2.5, y=0, z=0)  # Inverted X position to swing axes to bottom/left
            ),
            zaxis=dict(autorange="reversed")  # Flips depth direction if needed for bottom alignment
        )
    )
    fig.write_image(f"{out_path_png}YZ.png", scale=1, width=500, height=470)
    print(f"Saved png to {out_path_png}YZ.png")

def auto_scale_volume_dB(vol_db_rel, max_dynamic_range_db=15.0):
    """Calculates an automated statistical threshold (Otsu) on a relative dB volume.

    Parameters
    ----------
    vol_db_rel : ndarray
        Volume data in relative dB scale (0 dB = peak signal).
    max_dynamic_range_db : float
        Maximum dynamic range window to search for Otsu threshold.

    Returns
    -------
    vol_display : ndarray
        Volume with values below threshold masked to floor dB.
    otsu_thresh_rel : float
        Computed Otsu threshold in relative dB (negative float, e.g. -8.5 dB).
    """
    # Clip data to specified dynamic range window for clean histogram computation
    vol_clipped = np.maximum(vol_db_rel, -max_dynamic_range_db)
    
    # Compute threshold on relative dB data
    otsu_thresh_rel = threshold_otsu(vol_clipped)
    
    # Apply threshold mask
    vol_display = np.where(vol_db_rel >= otsu_thresh_rel, vol_db_rel, -max_dynamic_range_db)
    
    return vol_display, otsu_thresh_rel

def plot_interactive_volume(filename, dynamic_range_db=6.0, use_Otsu=False, save_name=None, use_flat_db_max=False, flat_db_max=-40.0):
    # 1. Load the data
    data = np.load(f'{definitions.PROCESS_SAVE_FILE}/volumes/{filename}.npz')
    vol_3d = data['vol_3d']        # shape: (ny, nx, nz)
    x_vec_mm = data['x_vec_mm']
    y_vec_mm = data['y_vec_mm']
    z_vec_mm = data['z_vec_mm']
    if save_name is None:
        save_name=filename

    # 2. Convert to dB scale ONCE
    # Using 10*log10 for power/intensity output (DMAS/WDAS)
    peak_val = np.max(vol_3d)
    vol_db_abs = 10.0 * np.log10(vol_3d + 1e-12)                           # Absolute dBFS
    vol_db_rel = 10.0 * np.log10((vol_3d / (peak_val + 1e-12)) + 1e-12)    # Relative dB (Peak = 0 dB)

    if use_flat_db_max:
        vmax = flat_db_max
    else:
        vmax = np.max(vol_db_abs)  # Peak reflection in true absolute dBFS

    # 3. Determine display thresholding
    if use_Otsu:
        vol_db_rel_masked, otsu_thresh_rel = auto_scale_volume_dB(
            vol_db_rel, max_dynamic_range_db=dynamic_range_db
        )
        # vmin is offset by the relative Otsu threshold
        vmin = vmax + otsu_thresh_rel
        effective_dr = abs(otsu_thresh_rel)
        # Reconstruct absolute dB array with Otsu thresholding
        plot_values = vmax + vol_db_rel_masked
        print(f"Otsu threshold applied: cutoff at {otsu_thresh_rel:.2f} dB relative to peak. "
              f"Displaying top {effective_dr:.2f} dB range (vmin={vmin:.2f} dBFS, vmax={vmax:.2f} dBFS)")
    else:
        effective_dr = dynamic_range_db
        vmin = vmax - dynamic_range_db
        plot_values = np.maximum(vol_db_abs, vmin)

    # 4. Create spatial grid mapping
    X, Y, Z = np.meshgrid(x_vec_mm, y_vec_mm, z_vec_mm, indexing='xy')

    print(f"Generating interactive 3D plot for {filename}... (this may take a moment)")

    # 5. Create the volumetric plot using isosurfaces
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=plot_values.flatten(),
        isomin=vmin,               # Lower bound for isosurface cutoff
        isomax=vmax,               # Upper bound is the actual peak dBFS value
        opacity=0.15,
        surface_count=15,
        colorscale='inferno',
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title='Intensity\n(dB, a.u.)')
    ))

    # 6. Define Camera View (X horizontal right, Y vertical up, Z depth into screen)
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=-2.0)
    )

    # 7. Format the layout
    fig.update_layout(
        title=f"3D Radar Reconstruction: Top {effective_dr:.2f} dB of {filename}",
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z Depth [mm]',
            aspectmode='data',
            camera=camera
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 8. Save and display
    out_path_html = f"{definitions.PROCESS_SAVE_FILE}/figures/{filename}/{save_name}_{effective_dr:.1f}dB.html"
    os.makedirs(os.path.dirname(out_path_html), exist_ok=True)

    fig.write_html(out_path_html)
    print(f"Saved interactive plot to {out_path_html}")
    webbrowser.open(os.path.abspath(out_path_html))

    # Was used for report image generation
    # out_path_png = f"{Path(__file__).resolve().parent.parent.parent.parent}/Report/Figures/{save_name}_{effective_dr:.1f}dB"
    # fig.write_image(f"{out_path_png}.png", scale=1, width=700, height=500)
    # print(f"Saved png to {out_path_png}.png")
    # fig.update_layout(
    #     scene=dict(
    #         camera=dict(
    #             up=dict(x=0, y=1, z=0),
    #             center=dict(x=0, y=0, z=0),
    #             eye=dict(x=2.5, y=0, z=0)  # Inverted X position to swing axes to bottom/left
    #         ),
    #         zaxis=dict(autorange="reversed")  # Flips depth direction if needed for bottom alignment
    #     )
    # )
    # fig.write_image(f"{out_path_png}YZ.png", scale=1, width=500, height=470)
    # print(f"Saved png to {out_path_png}YZ.png")

    

if __name__ == "__main__":
    main_filename = "PhantomFoil_2"
    plot_interactive_volume(main_filename, dynamic_range_db=25.0, use_Otsu=False)
    #plot_overlay_volumes("PhantomSteel_1", "PhantomSteel_1_svd_hybrid")