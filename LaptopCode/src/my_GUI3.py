"""
=============================================================================
                          GUI CONTROL SYSTEM (my_GUI3.py)
=============================================================================
A graphical interface for VNA measurement, hardware rig control, 
calibration, raster scanning, microwave 3D reconstruction, and anomaly detection.

Standalone implementation completely independent of my_CLI.py.
=============================================================================
"""

import sys, os, re, time, queue, threading, socket, traceback, webbrowser, tkinter as tk
from tkinter import ttk, scrolledtext
import serial.tools.list_ports
import skrf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Project module imports
import definitions
import my_vna
import my_jetson
import my_oecp
import plot3d
import img_constr2 as img


# =====================================================================
# SYSTEM STATE MANAGEMENT (Replaces cli.state)
# =====================================================================
class SystemState:
    """Manages active hardware connection handles, task threads, and parameters."""
    def __init__(self):
        self.cal_f_start = getattr(definitions, "F_START", 100.0e6)
        self.cal_f_stop = getattr(definitions, "F_STOP", 6.3e9)
        self.step_size = getattr(definitions, "STEP_SIZE", 4.0)
        self.f_points = getattr(definitions, "F_POINTS", 101)
        self.vna = None
        self.vna_port = getattr(definitions, "VNA_PORT", "COM10")
        self.jetson_sock = None
        self.jetson_ip = getattr(definitions, "JETSON_IP", "192.168.180.2")
        self.jetson_port = getattr(definitions, "PORT", 5005)
        self.active_task = None
        self.cal_stage = None  # None, "open", "short", "load"
        self.mode = "antenna"  # "antenna" or "probe"


# =====================================================================
# FORM SPECS & HELPERS
# =====================================================================
def F(key, label, default=None, kind="str", req=False, choices=None):
    """Helper function to define an input field specification."""
    return dict(key=key, label=label, default=default, kind=kind, req=req, choices=choices)

def _ls(path, ext=None, dirs=False):
    """Lists files or directories safely."""
    if not os.path.isdir(path): return []
    if dirs: return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
    return sorted(f[:-len(ext)] for f in os.listdir(path) if f.endswith(ext))

# Dynamic option providers for GUI dropdowns
LISTERS = {
    "scans": lambda: _ls(definitions.VNA_SAVE_FILE, dirs=True),
    "volumes": lambda: _ls(f"{definitions.PROCESS_SAVE_FILE}/volumes", ".npz"),
    "measures": lambda: _ls(definitions.VNA_MEASURE_FILE, ".s1p"),
    "refs": lambda: _ls(definitions.OECP_REF_FILE, ".s1p"),
}

SCAN_RE = re.compile(r"Scanned position: R:(\d+), C:(\d+)")
CASTS = {"int": lambda x: int(float(x)), "float": float, "bool": bool, "str": str}


class QueueWriter:
    """Thread-safe stdout/stderr stream wrapper feeding a queue."""
    def __init__(self, q): 
        self.q = q
    def write(self, t):
        if t: 
            self.q.put(str(t))
    def flush(self): 
        pass


class RigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rig & VNA Control Center (Independent)")
        self.root.geometry("1180x880")
        self.root.minsize(950, 700)

        # Standalone System State Instance
        self.state = SystemState()

        # Thread-safe log queue and stdout/stderr redirection
        self.q = queue.Queue()
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        sys.stdout = sys.stderr = QueueWriter(self.q)

        # State tracking variables
        self.cells, self.last_cell, self.n_done, self.n_total = {}, None, 0, 0
        self.auto_scroll_var = tk.BooleanVar(value=True)

        # Build GUI Layout with PanedWindow for resizable terminal
        self.build_layout()

        # Keyboard shortcuts
        root.bind_all("<Escape>", lambda e: self.estop())
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Periodic background polling timers
        root.after(100, self.poll_output)
        root.after(300, self.refresh_status)

    def check_connections(self, require_vna=False, require_rig=False):
        """Validates required hardware connections before command execution.
        Updates status badges and logs error if requirements are not met."""
        valid = True

        if require_vna:
            if self.state.vna is None:
                self.log("[Connection Error] VNA is not connected. Please connect to VNA port first.")
                valid = False

        if require_rig:
            if self.state.jetson_sock is None:
                self.log("[Connection Error] Motion Rig is not connected. Please connect to Jetson IP/Port first.")
                valid = False

        self.refresh_status()
        return valid

    def build_tabs_schema(self):
        """Constructs dynamic TABS schema referencing native class methods."""
        cur_start = lambda: self.state.cal_f_start / 1e6
        cur_stop = lambda: self.state.cal_f_stop / 1e6

        return {
            "Measure": [
                ("Measure S11", self.measure_s11_mhz_wrapper, "args", [
                    F("name", "Name", req=True, choices="measures"),
                    F("f_start", "F start (MHz)", cur_start, "float"),
                    F("f_stop", "F stop (MHz)", cur_stop, "float"),
                    F("f_points", "Points", lambda: self.state.f_points, "int")],
                 lambda gui, vals, t0: gui.show_measurement(vals["out_dir"], vals["name"], t0)),
                ("Plot S11", lambda args: self.cmd_plot(args), "args", [
                    F("name", "Measurement name", req=True, choices="measures")],
                 lambda gui, vals, t0: gui.show_measurement(definitions.VNA_MEASURE_FILE, vals["name"], 0)),
                ("Plot Reference", lambda args: self.cmd_plot_ref(args), "args", [
                    F("name", "Reference name", req=True, choices="refs")],
                 lambda gui, vals, t0: gui.show_measurement(definitions.OECP_REF_FILE, vals["name"], 0)),
                ("New OECP Reference", lambda args: self.cmd_new_ref(args), "args", [
                    F("name", "Reference name", req=True, choices="refs")]),
                ("Test Material (OECP)", lambda args: self.cmd_test_mut(args), "args", [
                    F("name", "MUT name", req=True, choices="measures"),
                    F("overwrite", "Re-measure / Overwrite", False, "bool")]),
            ],
            "Reconstruction": [
                ("Anomaly Detection (Full Recon)", lambda args: self.cmd_recon(args), "args", [
                    F("tar_name", "Target scan", req=True, choices="scans"),
                    F("ref_name", "Ref scan (blank=auto)", choices="scans"),
                    F("z_min", "Z min (mm)", 30.0, "float"),
                    F("z_max", "Z max (mm)", 180.0, "float"),
                    F("margin_mm", "Margin (mm)", 0.0, "float")]),
                ("Reconstruct 3D Volume", img.process_scan, "kw", [
                    F("tar_name", "Target scan", req=True, choices="scans"),
                    F("ref_name", "Reference scan (blank=default)", choices="scans"),
                    F("z_min", "Z min (mm)", 80.0, "float"),
                    F("z_max", "Z max (mm)", 230.0, "float"),
                    F("margin_mm", "Margin (mm)", 0.0, "float"),
                    F("weighting_mode", "Weighting mode", "none"),
                    F("use_svd", "SVD clutter removal", False, "bool"),
                    F("antenna_delay", "Antenna Delay (s)", 0.6e-9, "float"),
                    F("save_name", "Save name (blank=default)")]),
                ("Overlay Two Volumes", lambda vol_a, vol_b: plot3d.plot_overlay_volumes(vol_a, vol_b), "kw", [
                    F("vol_a", "Volume A", req=True, choices="volumes"), 
                    F("vol_b", "Volume B", req=True, choices="volumes")]),
                ("Plot 3D Volume", lambda args: self.cmd_plot3d(args), "args", [
                    F("filename", "Volume name", req=True, choices="volumes"),
                    F("db", "Dynamic range (dB)", 6.0, "float")]),
            ],
        }

    def build_layout(self):
        """Creates the primary vertical resizable layout with PanedWindow."""
        self.paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.paned.pack(fill="both", expand=True, padx=4, pady=4)

        # Top Pane: Control header, settings, and main tabbed notebook
        self.top_pane = ttk.Frame(self.paned)
        self.paned.add(self.top_pane, weight=3)

        # Bottom Pane: Terminal / Output Console
        self.bottom_pane = ttk.Frame(self.paned)
        self.paned.add(self.bottom_pane, weight=1)

        # ---- TOP PANE CONTENT ----
        self.build_header(self.top_pane)

        self.nb = ttk.Notebook(self.top_pane)
        self.nb.pack(fill="both", expand=True, padx=4, pady=4)

        self.build_control_tab(self.add_tab(self.nb, "Control"))
        self.build_cal_tab(self.add_tab(self.nb, "Calibration"))
        self.build_raster_tab(self.add_tab(self.nb, "Raster Scan"))

        # Build dynamic forms from TABS schema
        tabs_schema = self.build_tabs_schema()
        for name, specs in tabs_schema.items():
            tab = self.add_tab(self.nb, name)
            
            if name == "Measure":
                paned_m = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
                paned_m.pack(fill="both", expand=True, padx=4, pady=4)

                left_frame = ttk.Frame(paned_m)
                right_frame = ttk.Frame(paned_m)

                paned_m.add(left_frame, weight=1)
                paned_m.add(right_frame, weight=1)

                for spec in specs:
                    self.build_form(left_frame, *spec).pack(fill="x", padx=4, pady=3)

                self.build_measure_plot_area(right_frame)

            elif name == "Reconstruction":
                for spec in specs:
                    self.build_form(tab, *spec).pack(fill="x", padx=6, pady=4)
                
                self.build_3d_viewer_panel(tab)
            else:
                for spec in specs:
                    self.build_form(tab, *spec).pack(fill="x", padx=6, pady=4)

        # ---- BOTTOM PANE CONTENT (TERMINAL) ----
        self.build_terminal(self.bottom_pane)

    def build_header(self, parent):
        """Top bar containing connection badges, settings parameters, and E-STOP."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill="x", padx=6, pady=4)

        # --- Status Badges (Left) ---
        sf = ttk.LabelFrame(header_frame, text="System Status")
        sf.pack(side="left", fill="y", padx=(0, 6))

        self.lbl_vna = tk.Label(sf, text="VNA: ...", anchor="w", font=("Consolas", 9, "bold"))
        self.lbl_vna.pack(fill="x", padx=6, pady=1)

        self.lbl_jet = tk.Label(sf, text="Rig: ...", anchor="w", font=("Consolas", 9, "bold"))
        self.lbl_jet.pack(fill="x", padx=6, pady=1)

        self.lbl_cal = tk.Label(sf, text="Calibration: ...", anchor="w", font=("Consolas", 9, "bold"))
        self.lbl_cal.pack(fill="x", padx=6, pady=1)

        self.lbl_task = tk.Label(sf, text="Task: ...", anchor="w", font=("Consolas", 9, "bold"))
        self.lbl_task.pack(fill="x", padx=6, pady=1)

        # --- Quick Settings Bar (Middle) ---
        self.build_settings(header_frame)

        # --- E-STOP Button (Right) ---
        estop_btn = tk.Button(
            header_frame, text="E-STOP\n(Esc)", bg="#d32f2f", fg="white", 
            activebackground="#8b0000", activeforeground="white",
            font=("Arial", 14, "bold"), width=9, command=self.estop, relief="raised", bd=3
        )
        estop_btn.pack(side="right", fill="y", padx=4, pady=2)

    def build_settings(self, parent):
        """Settings panel with start/stop frequencies in MHz on the same line, plus VNA/Antenna limits."""
        cfg = ttk.LabelFrame(parent, text="Global Settings & Freq Limits")
        cfg.pack(side="left", fill="both", expand=True, padx=4)

        self.setting_vars = {}

        # 1. Hardware Frequency Limits Banner
        vna_min = getattr(definitions, "VNA_MIN_FREQ", 50e4) / 1e6
        vna_max = getattr(definitions, "VNA_MAX_FREQ", 3e9) / 1e6
        ant_min = getattr(definitions, "ANTENNA_MIN_FREQ", 1e9) / 1e6
        ant_max = getattr(definitions, "ANTENNA_MAX_FREQ", 6e9) / 1e6

        lim_lbl = ttk.Label(
            cfg, 
            text=f"Hardware Limits (MHz) ➔ VNA: {vna_min:.1f}–{vna_max:.1f} MHz  |  Antenna: {ant_min:.1f}–{ant_max:.1f} MHz",
            font=("Segoe UI", 8, "bold"),
            foreground="#2e7d32"
        )
        lim_lbl.grid(row=0, column=0, columnspan=6, sticky="w", padx=6, pady=(1, 3))

        # 2. Row 1: Start and Stop Frequencies on the SAME LINE (in MHz)
        f_start_mhz = self.state.cal_f_start / 1e6
        f_stop_mhz = self.state.cal_f_stop / 1e6

        self.setting_vars["f_start"] = tk.StringVar(value=f"{f_start_mhz:.2f}".rstrip('0').rstrip('.'))
        self.setting_vars["f_stop"] = tk.StringVar(value=f"{f_stop_mhz:.2f}".rstrip('0').rstrip('.'))

        ttk.Label(cfg, text="F start (MHz):", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="e", padx=(6, 2), pady=2)
        ttk.Entry(cfg, textvariable=self.setting_vars["f_start"], width=9).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Button(
            cfg, text="Set", width=4,
            command=lambda: self.apply_setting("f_start", self.setting_vars["f_start"].get())
        ).grid(row=1, column=2, padx=(2, 10), pady=2)

        ttk.Label(cfg, text="F stop (MHz):", font=("Segoe UI", 9)).grid(row=1, column=3, sticky="e", padx=(6, 2), pady=2)
        ttk.Entry(cfg, textvariable=self.setting_vars["f_stop"], width=9).grid(row=1, column=4, sticky="w", pady=2)
        ttk.Button(
            cfg, text="Set", width=4,
            command=lambda: self.apply_setting("f_stop", self.setting_vars["f_stop"].get())
        ).grid(row=1, column=5, padx=(2, 6), pady=2)

        # 3. Row 2: Step Size and Points
        self.setting_vars["step_size"] = tk.StringVar(value=str(self.state.step_size))
        self.setting_vars["f_points"] = tk.StringVar(value=str(self.state.f_points))

        ttk.Label(cfg, text="Step (mm):", font=("Segoe UI", 9)).grid(row=2, column=0, sticky="e", padx=(6, 2), pady=2)
        ttk.Entry(cfg, textvariable=self.setting_vars["step_size"], width=9).grid(row=2, column=1, sticky="w", pady=2)
        ttk.Button(
            cfg, text="Set", width=4,
            command=lambda: self.apply_setting("step_size", self.setting_vars["step_size"].get())
        ).grid(row=2, column=2, padx=(2, 10), pady=2)

        ttk.Label(cfg, text="Points:", font=("Segoe UI", 9)).grid(row=2, column=3, sticky="e", padx=(6, 2), pady=2)
        ttk.Entry(cfg, textvariable=self.setting_vars["f_points"], width=9).grid(row=2, column=4, sticky="w", pady=2)
        ttk.Button(
            cfg, text="Set", width=4,
            command=lambda: self.apply_setting("f_points", self.setting_vars["f_points"].get())
        ).grid(row=2, column=5, padx=(2, 6), pady=2)

    def apply_setting(self, param_key, val_str):
        """Applies configuration updates (converting MHz to Hz for frequency keys)."""
        val_to_pass = val_str
        if param_key in ("f_start", "f_stop"):
            try:
                val_to_pass = str(float(val_str) * 1e6)
            except ValueError:
                pass
        
        self.run_task(
            f"config {param_key} {val_str} MHz" if param_key in ("f_start", "f_stop") else f"config {param_key} {val_str}",
            lambda: self.cmd_config(param_key, val_to_pass),
            guard=False
        )

    def build_terminal(self, parent):
        """Constructs the resizable output terminal panel inside the lower pane."""
        tf = ttk.LabelFrame(parent, text="Console & Output Log")
        tf.pack(fill="both", expand=True, padx=6, pady=4)

        tb = ttk.Frame(tf)
        tb.pack(fill="x", padx=4, pady=2)

        ttk.Label(tb, text="● Live Stream", font=("Consolas", 9, "bold"), foreground="#2e7d32").pack(side="left")
        
        ttk.Checkbutton(tb, text="Auto-Scroll", variable=self.auto_scroll_var).pack(side="right", padx=6)
        ttk.Button(tb, text="Clear Console", command=self.clear_terminal).pack(side="right", padx=4)

        self.term = scrolledtext.ScrolledText(
            tf, height=14, state="disabled", bg="#121212", fg="#00ff66",
            insertbackground="white", font=("Consolas", 10), wrap="word"
        )
        self.term.pack(fill="both", expand=True, padx=4, pady=4)

    def clear_terminal(self):
        """Clears console text."""
        self.term.config(state="normal")
        self.term.delete("1.0", "end")
        self.term.config(state="disabled")

    def build_measure_plot_area(self, parent):
        """Creates the S11 measurement plot container on the right side of the Measure tab."""
        plot_frame = ttk.LabelFrame(parent, text="S11 Measurement Plot")
        plot_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("No measurement data loaded")
        self.ax.set_xlabel("Frequency (GHz)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.grid(True, linestyle="--", alpha=0.6)

        self.mcanvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.mcanvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def build_3d_viewer_panel(self, parent):
        """Builds an HTML 3D Plotly Volume launcher panel in the Reconstruction tab."""
        vf = ttk.LabelFrame(parent, text="Interactive 3D Volume Viewer (WebGL)")
        vf.pack(fill="x", padx=6, pady=6)

        self.view_vol_var = tk.StringVar()
        ttk.Label(vf, text="Select Volume:").pack(side="left", padx=(10, 4), pady=6)

        cb = self.dropdown(vf, self.view_vol_var, "volumes", 24)
        cb.pack(side="left", padx=4, pady=6)

        ttk.Button(
            vf, text="🌐 Launch Interactive 3D Plot", 
            command=lambda: self.open_volume_html(self.view_vol_var.get())
        ).pack(side="left", padx=12, pady=6)

        ttk.Label(
            vf, text="Renders full 3D interactive isosurfaces with orbit, zoom, and opacity controls.",
            font=("Segoe UI", 9, "italic"), foreground="#555555"
        ).pack(side="left", padx=8)

    def open_volume_html(self, vol_name, is_overlay=False):
        """Locates and opens the generated Plotly 3D HTML plot in the system web browser."""
        if not vol_name:
            self.log("[Viewer Error] No volume selected.")
            return

        vol_name = vol_name.strip()
        possible_paths = [
            os.path.join(definitions.PROCESS_SAVE_FILE, "figures", "overlays", f"{vol_name}.html"),
            os.path.join(definitions.PROCESS_SAVE_FILE, "figures", f"{vol_name}.html"),
            os.path.join(definitions.PROCESS_SAVE_FILE, "volumes", f"{vol_name}.html")
        ]

        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break

        if found_path:
            self.log(f"[3D Viewer] Opening HTML viewer for {vol_name}...")
            webbrowser.open(f"file://{os.path.abspath(found_path)}")
        else:
            self.log(f"[3D Viewer] Generating HTML plot for {vol_name}...")
            def gen_and_open():
                self.cmd_plot3d([vol_name])
            self.run_task(f"render-3d {vol_name}", gen_and_open, guard=False)

    # ------------------------------------------------------------ Helpers
    def add_tab(self, nb, name):
        f = ttk.Frame(nb)
        nb.add(f, text=f"  {name}  ")
        return f

    def log(self, msg):
        self.q.put(msg + "\n")

    def run_task(self, desc, fn, guard=True, on_done=None):
        """Executes long-running operations in a background thread."""
        t = self.state.active_task
        if guard and t and t.is_alive():
            self.log("[Busy] A task is already running! (Press E-STOP to abort).")
            return
        
        self.log(f"> {desc}")

        def work():
            ok = False
            try:
                fn()
                ok = True
            except Exception as e:
                print(f"[Error] {e}")
                traceback.print_exc()
            finally:
                print(f"[Done] {desc}")
                if ok and on_done:
                    self.root.after(0, on_done)

        th = threading.Thread(target=work, daemon=True)
        if guard:
            self.state.active_task = th
        th.start()

    def estop(self):
        self.log("> *** E-STOP TRIGGERED ***")
        self.cmd_estop()

    def on_close(self):
        """Cleans up background threads and restores stdout/stderr on exit."""
        t = self.state.active_task
        if t and t.is_alive():
            self.cmd_estop()
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr
        self.root.destroy()

    # ------------------------------------------------------------ Generic Form Builder
    def build_form(self, parent, title, func, mode, fields, on_done=None):
        box = ttk.LabelFrame(parent, text=title)
        vs = {}
        max_cols = 2

        for i, f in enumerate(fields):
            r, c = divmod(i, max_cols)
            lbl_txt = f["label"] + (" *" if f["req"] else "")
            ttk.Label(box, text=lbl_txt).grid(row=r, column=c * 2, sticky="e", padx=(8, 2), pady=3)

            if f["kind"] == "bool":
                vs[f["key"]] = tk.BooleanVar(value=bool(f["default"]))
                w = ttk.Checkbutton(box, variable=vs[f["key"]])
            else:
                d = f["default"]
                default_val = str(d() if callable(d) else (d if d is not None else ""))
                vs[f["key"]] = tk.StringVar(value=default_val)
                
                if f["choices"]:
                    w = self.dropdown(box, vs[f["key"]], f["choices"], 16)
                else:
                    w = ttk.Entry(box, textvariable=vs[f["key"]], width=18)
            
            w.grid(row=r, column=c * 2 + 1, sticky="w", pady=3, padx=(0, 8))

        btn = ttk.Button(
            box, text="Run Command", 
            command=lambda: self.run_form(title, func, mode, fields, vs, on_done)
        )
        row_count = (len(fields) - 1) // max_cols + 1 if fields else 1
        btn.grid(row=0, column=max_cols * 2, rowspan=max(1, row_count), padx=12, pady=4)

        return box

    def run_form(self, title, func, mode, fields, vs, on_done=None):
        """Parses input values from a form and calls the target function."""
        try:
            vals = {}
            for f in fields:
                v = vs[f["key"]].get()
                if f["kind"] != "bool":
                    v = v.strip()
                    if v == "":
                        d = f["default"]
                        v = d() if callable(d) else d
                        if v is None or v == "":
                            if f["req"]: 
                                raise ValueError(f"'{f['label']}' is required")
                            v = None
                vals[f["key"]] = None if v is None else CASTS[f["kind"]](v)

            if mode == "args":
                # Maintain exact positional array length; pass None for any blank optional field anywhere in the list
                argv = [None if vals[f["key"]] is None else str(vals[f["key"]]) for f in fields]
                desc_args = [str(x) if x is not None else "''" for x in argv]
                call, desc = (lambda: func(argv)), f"{title} {' '.join(desc_args)}"
            else:
                kw = {k: v for k, v in vals.items() if v is not None}
                call, desc = (lambda: func(**kw)), f"{title} {kw}"
        except ValueError as e:
            self.log(f"[Input Error] {e}")
            return

        t0 = time.time()
        self.run_task(desc, call, on_done=(lambda: on_done(self, vals, t0)) if on_done else None)

    def dropdown(self, parent, var, choices, width):
        """Combobox whose option list dynamically refreshes upon opening."""
        get = (lambda: LISTERS[choices]()) if isinstance(choices, str) else (lambda: list(choices))
        cb = ttk.Combobox(parent, textvariable=var, values=get(), width=width)
        cb.config(postcommand=lambda: cb.config(values=get()))
        return cb

    # ------------------------------------------------------------ Control Tab
    def build_control_tab(self, tab):
        top_frame = ttk.Frame(tab)
        top_frame.pack(fill="x", padx=6, pady=4)

        # Connection Controls
        conn = ttk.LabelFrame(top_frame, text="Hardware Connections")
        conn.pack(side="left", fill="both", expand=True, padx=(0, 4))

        self.port_var = tk.StringVar(value=self.state.vna_port)
        ttk.Label(conn, text="VNA Port").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        cb = ttk.Combobox(
            conn, textvariable=self.port_var, width=12,
            postcommand=lambda: cb.config(values=[p.device for p in serial.tools.list_ports.comports()])
        )
        cb.grid(row=0, column=1, pady=4)
        ttk.Button(
            conn, text="Connect VNA", 
            command=lambda: self.run_task(
                "connect-vna", lambda: self.cmd_connect_vna(self.port_var.get()), guard=False
            )
        ).grid(row=0, column=2, padx=6, pady=4)

        self.ip_var = tk.StringVar(value=self.state.jetson_ip)
        self.jport_var = tk.StringVar(value=str(self.state.jetson_port))
        ttk.Label(conn, text="Jetson IP / Port").grid(row=1, column=0, padx=4, pady=4, sticky="e")
        
        ip_frame = ttk.Frame(conn)
        ip_frame.grid(row=1, column=1, pady=4, sticky="w")
        ttk.Entry(ip_frame, textvariable=self.ip_var, width=12).pack(side="left", padx=(0, 2))
        ttk.Entry(ip_frame, textvariable=self.jport_var, width=5).pack(side="left")

        ttk.Button(
            conn, text="Connect Rig", 
            command=lambda: self.run_task(
                "connect-jetson", lambda: self.cmd_connect_jetson(self.ip_var.get(), self.jport_var.get()),
                guard=False
            )
        ).grid(row=1, column=2, padx=6, pady=4)

        # Motion & Jog Controls
        jog = ttk.LabelFrame(tab, text="Rig Motion & Manual Jog Control")
        jog.pack(fill="x", padx=6, pady=4)

        pad = ttk.Frame(jog)
        pad.pack(side="left", padx=12, pady=6)

        for txt, d, r, c in (("▲ Up", "up", 0, 1), ("◀ Left", "left", 1, 0), ("▶ Right", "right", 1, 2), ("▼ Down", "down", 2, 1)):
            tk.Button(
                pad, text=txt, width=8, height=1, font=("Arial", 10, "bold"),
                command=lambda d=d: self.jog(d)
            ).grid(row=r, column=c, padx=2, pady=2)

        tk.Button(
            pad, text="■ STOP", width=8, height=1, bg="#ff9800", fg="black", font=("Arial", 10, "bold"),
            command=lambda: self.run_task("stop", lambda: self.cmd_stop(), guard=False)
        ).grid(row=1, column=1, padx=2, pady=2)

        ctrl_right = ttk.Frame(jog)
        ctrl_right.pack(side="left", fill="y", padx=16, pady=6)

        df = ttk.Frame(ctrl_right)
        df.pack(anchor="w", pady=4)
        ttk.Label(df, text="Jog Distance (mm):").pack(side="left")
        self.jog_dist = tk.StringVar(value=str(self.state.step_size))
        ttk.Entry(df, textvariable=self.jog_dist, width=8).pack(side="left", padx=6)

        hf = ttk.Frame(ctrl_right)
        hf.pack(anchor="w", pady=6)
        for txt, ax in (("Home All", ""), ("Home X", "x"), ("Home Y", "y")):
            ttk.Button(
                hf, text=txt, 
                command=lambda ax=ax: self.run_task(
                    f"home {ax}".strip(), lambda ax=ax: self.cmd_home(ax)
                )
            ).pack(side="left", padx=3)

    def jog(self, direction):
        try: 
            dist = float(self.jog_dist.get())
        except ValueError: 
            self.log("[Input Error] Distance must be a valid number.")
            return
        self.run_task(f"move {dist} {direction}", lambda: self.cmd_move(dist, direction))

    def show_measurement(self, out_dir, name, t0=0):
        """Displays newly captured or selected S11 measurement on the plot canvas."""
        path = os.path.join(out_dir, f"{name}.s1p")
        if not os.path.exists(path):
            self.log(f"[Measure] No measurement file found at {path}")
            return
        if t0 > 0 and os.path.getmtime(path) < t0 - 1:
            self.log(f"[Measure] No new measurement file found at {path}")
            return
        try:
            net = skrf.Network(path)
            self.ax.clear()
            self.ax.plot(net.f / 1e9, net.s_db[:, 0, 0], linewidth=1.5, color="#0066cc")
            self.ax.set_title(f"S11 Measurement: {name}")
            self.ax.set_xlabel("Frequency (GHz)")
            self.ax.set_ylabel("Magnitude (dB)")
            self.ax.grid(True, linestyle="--", alpha=0.6)
            self.fig.tight_layout()
            self.mcanvas.draw_idle()
        except Exception as e:
            self.log(f"[Plot Error] {e}")

    # ------------------------------------------------------------ Calibration Tab
    def build_cal_tab(self, tab):
        box = ttk.LabelFrame(tab, text="VNA Calibration Wizard")
        box.pack(fill="x", padx=6, pady=6)

        self.cal_mode = tk.StringVar(value=self.state.mode)
        ttk.Label(box, text="Calibration Mode:").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        ttk.Combobox(
            box, textvariable=self.cal_mode, values=["antenna", "probe"], width=12, state="readonly"
        ).grid(row=0, column=1, padx=4, pady=6)

        self.btn_cal = ttk.Button(box, text="Start Calibration", command=self.cal_step)
        self.btn_cal.grid(row=0, column=2, padx=8, pady=6)

        ttk.Button(
            box, text="Load Saved", 
            command=lambda: self.run_task("cal-load", lambda: self.cmd_cal_load([self.cal_mode.get()]))
        ).grid(row=0, column=3, padx=4, pady=6)

        ttk.Button(
            box, text="Reset", 
            command=lambda: self.run_task("cal-reset", lambda: self.cmd_cal_reset([]))
        ).grid(row=0, column=4, padx=4, pady=6)

        self.lbl_cal_hint = ttk.Label(box, text="", font=("Segoe UI", 9, "italic"), foreground="#0066cc")
        self.lbl_cal_hint.grid(row=1, column=0, columnspan=5, pady=4, padx=6)

    def cal_step(self):
        args = [] if self.state.cal_stage else [self.cal_mode.get()]
        self.run_task("calibrate " + " ".join(args), lambda: self.cmd_calibrate(args))

    # ------------------------------------------------------------ Raster Tab
    def build_raster_tab(self, tab):
        box = ttk.LabelFrame(tab, text="Raster Scan Control (Homes automatically first)")
        box.pack(fill="x", padx=6, pady=4)

        self.r_vars = {
            k: tk.StringVar(value=str(v)) for k, v in
            (("Output Folder", "scan1"), 
             ("Width (mm)", definitions.RASTER_WIDTH),
             ("Height (mm)", definitions.RASTER_HEIGHT))
        }

        for i, (k, v) in enumerate(self.r_vars.items()):
            ttk.Label(box, text=k).grid(row=0, column=i * 2, padx=(8, 2), pady=6)
            w = self.dropdown(box, v, "scans", 14) if k == "Output Folder" else ttk.Entry(box, textvariable=v, width=12)
            w.grid(row=0, column=i * 2 + 1, pady=6)

        ttk.Button(box, text="Start Scan", command=self.start_raster).grid(row=0, column=6, padx=12, pady=6)

        prog_frame = ttk.Frame(tab)
        prog_frame.pack(fill="x", padx=6, pady=2)
        
        self.lbl_prog = ttk.Label(prog_frame, text="Scan status: Idle", font=("Segoe UI", 10, "bold"))
        self.lbl_prog.pack(side="left", padx=6)

        # Canvas for interactive grid map
        canvas_frame = ttk.LabelFrame(tab, text="Live Scanning Grid Map")
        canvas_frame.pack(fill="both", expand=True, padx=6, pady=4)

        self.canvas = tk.Canvas(canvas_frame, bg="#1e1e1e", height=320)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)

    def start_raster(self):
        try:
            name = self.r_vars["Output Folder"].get().strip()
            w = int(float(self.r_vars["Width (mm)"].get()))
            h = int(float(self.r_vars["Height (mm)"].get()))
            step = float(self.state.step_size)
            if not name: 
                raise ValueError("Output folder is required.")
        except ValueError as e:
            self.log(f"[Input Error] {e}")
            return

        self.init_grid(int(h // step) + 1, int(w // step) + 1)
        
        def job():
            self.cmd_raster_scan([name, str(w), str(h)])
            
        self.run_task(f"raster-scan {name} {w}x{h} step {step}", job)

    def init_grid(self, rows, cols):
        c = self.canvas
        c.update_idletasks()
        c.delete("all")
        
        s = min((c.winfo_width() - 10) / cols, (c.winfo_height() - 10) / rows)
        if s <= 0: 
            s = 10
            
        self.cells, self.last_cell, self.n_done, self.n_total = {}, None, 0, rows * cols
        self.scan_start_time = None  # None indicates homing is in progress
        
        for r in range(rows):
            for k in range(cols):
                x, y = 5 + k * s, 5 + (rows - 1 - r) * s
                self.cells[(r, k)] = c.create_rectangle(x, y, x + s, y + s, fill="#333333", outline="#555555")
                
        self.lbl_prog.config(text=f"Progress: 0 / {self.n_total} (0%) | Est. remaining: Homing...")

    def mark_cell(self, r, k):
        if (r, k) not in self.cells: 
            return

        # Start the timer when the first point is received (after homing completes)
        if self.scan_start_time is None:
            self.scan_start_time = time.time()

        self.canvas.itemconfig(self.cells[(r, k)], fill="#2ecc71")
        self.last_cell, self.n_done = (r, k), self.n_done + 1

        # Rounded percentage for display
        pct = round((self.n_done / self.n_total) * 100) if self.n_total else 0

        # Accurate ETA calculated purely from active scanning pace
        elapsed = time.time() - self.scan_start_time
        scanned_intervals = self.n_done - 1  # Completed intervals since scanning started

        if scanned_intervals > 0 and elapsed > 0:
            avg_sec_per_point = elapsed / scanned_intervals
            remaining_sec = avg_sec_per_point * (self.n_total - self.n_done)
            mins, secs = divmod(int(remaining_sec), 60)
            hrs, mins = divmod(mins, 60)
            time_str = f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs > 0 else f"{mins:02d}:{secs:02d}"
        else:
            time_str = "Calculating..."

        self.lbl_prog.config(
            text=f"Progress: {self.n_done} / {self.n_total} ({pct}%) | Est. remaining: {time_str}"
        )

    # ------------------------------------------------------------ Polling & Status
    def poll_output(self):
        """Polls queue for stdout/stderr messages and updates terminal."""
        try:
            while True:
                t = self.q.get_nowait()
                self.term.config(state="normal")
                self.term.insert("end", t)
                if self.auto_scroll_var.get():
                    self.term.see("end")
                self.term.config(state="disabled")

                for m in SCAN_RE.finditer(t):
                    self.mark_cell(int(m.group(1)), int(m.group(2)))

                if "[Done] raster-scan" in t and self.last_cell:
                    self.canvas.itemconfig(self.cells[self.last_cell], fill="#2ecc71")
        except queue.Empty:
            pass
        self.root.after(100, self.poll_output)

    def refresh_status(self):
        """Refreshes status indicators and calibration status in real-time."""
        s, G, R, O = self.state, "#1b5e20", "#b71c1c", "#e65100"
        
        v = s.vna is not None
        self.lbl_vna.config(text=f"VNA:    {'Connected (' + str(s.vna_port or '') + ')' if v else 'Not Connected'}", fg=G if v else R)

        j = s.jetson_sock is not None
        self.lbl_jet.config(text=f"Rig:    {'Connected' if j else 'Not Connected'}", fg=G if j else R)

        cal_active = getattr(my_vna, "_calibration", None) is not None
        if cal_active:
            self.lbl_cal.config(text=f"Calibration: ACTIVE ({s.mode})", fg=G)
        elif s.cal_stage:
            self.lbl_cal.config(text=f"Calibration: In Progress ({s.cal_stage.upper()})", fg=O)
        else:
            self.lbl_cal.config(text="Calibration: NONE", fg=R)

        busy = bool(s.active_task and s.active_task.is_alive())
        self.lbl_task.config(text=f"Task:   {'Running...' if busy else 'Idle'}", fg=O if busy else G)

        if s.cal_stage:
            self.btn_cal.config(text=f"Measure {s.cal_stage.upper()} Standard")
            self.lbl_cal_hint.config(text=f"Connect the {s.cal_stage.upper()} standard, then click the button above.")
        else:
            self.btn_cal.config(text="Start Calibration")
            self.lbl_cal_hint.config(text="")

        self.root.after(300, self.refresh_status)

    # =====================================================================
    # NATIVE COMMAND IMPLEMENTATIONS (Replacing my_CLI calls)
    # =====================================================================
    def cmd_connect_vna(self, port):
        """Connects to the NanoVNA on the given serial COM port."""
        try:
            print(f"Connecting to VNA on port {port}...")
            self.state.vna = my_vna.vna_connect(port)
            self.state.vna_port = port
            print(f"Successfully connected to VNA on {port}.")
        except Exception as e:
            self.state.vna = None
            print(f"Failed to connect to VNA on {port}: {e}")
        finally:
            self.refresh_status()

    def cmd_connect_jetson(self, ip, port):
        """Establishes TCP socket connection to the motion rig."""
        try:
            port_int = int(port)
            print(f"Connecting to Rig at {ip}:{port_int}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((ip, port_int))
            sock.settimeout(None)
            self.state.jetson_sock = sock
            self.state.jetson_ip = ip
            self.state.jetson_port = port_int
            print(f"Successfully connected to Rig at {ip}:{port_int}.")
        except Exception as e:
            self.state.jetson_sock = None
            print(f"Failed to connect to Rig at {ip}:{port}: {e}")
        finally:
            self.refresh_status()

    def cmd_move(self, dist, direction):
        """Sends manual motion commands to the rig."""
        if not self.check_connections(require_rig=True):
            return
        try:
            my_jetson.jetson_send(self.state.jetson_sock, f"move {dist} {direction}")
            if not my_jetson.jetson_read(self.state.jetson_sock, f"moved {float(dist)}mm {direction}", timeout=5):
                print(f"[Error] Failed to move {dist}mm {direction}.")
        except (socket.error, OSError) as e:
            print(f"[Connection Error] Lost connection to Motion Rig: {e}")
            self.state.jetson_sock = None
            self.refresh_status()

    def cmd_stop(self, axis=""):
        """Stops motion on specified axis or all axes."""
        if not self.check_connections(require_rig=True):
            return
        try:
            my_jetson.stop(self.state.jetson_sock, axis=axis)
            print(f"Sent stop command {axis if axis else 'all'}.")
        except (socket.error, OSError) as e:
            print(f"[Connection Error] Motion Rig disconnected: {e}")
            self.state.jetson_sock = None
            self.refresh_status()

    def cmd_home(self, axis=""):
        """Homes specified axis or both axes."""
        if not self.check_connections(require_rig=True):
            return
        try:
            print(f"Homing axis: {axis if axis else 'all'}...")
            my_jetson.home(self.state.jetson_sock, axis=axis)
        except (socket.error, OSError) as e:
            print(f"[Connection Error] Lost connection to Motion Rig during home: {e}")
            self.state.jetson_sock = None
            self.refresh_status()

    def cmd_estop(self):
        """Aborts active thread tasks and stops hardware motion immediately."""
        print("*** EMERGENCY STOP TRIGGERED ***")
        if self.state.jetson_sock:
            try:
                my_jetson.estop(self.state.jetson_sock)
            except Exception as e:
                print(f"Error stopping rig: {e}")
                self.state.jetson_sock = None
        self.state.active_task = None
        self.refresh_status()

    def cmd_config(self, param_key, val_str):
        """Updates global frequency, step size, or points settings."""
        try:
            val = float(val_str)
            if param_key == "f_start":
                self.state.cal_f_start = val
                print(f"F start updated to {val / 1e6:.2f} MHz ({val} Hz).")
            elif param_key == "f_stop":
                self.state.cal_f_stop = val
                print(f"F stop updated to {val / 1e6:.2f} MHz ({val} Hz).")
            elif param_key == "step_size":
                self.state.step_size = val
                definitions.STEP_SIZE = val
                print(f"Step size updated to {val} mm.")
            elif param_key == "f_points":
                val_int = int(val)
                self.state.f_points = val_int
                definitions.F_POINTS = val_int
                print(f"Frequency points updated to {val_int}.")
            else:
                print(f"Unknown config key: {param_key}")
        except ValueError:
            print(f"Invalid value for config parameter {param_key}: {val_str}")

    def measure_s11_mhz_wrapper(self, args):
        """Converts f_start and f_stop arguments from MHz to Hz before calling cmd_measure."""
        args_conv = list(args)
        if len(args_conv) >= 2 and args_conv[1] not in (None, ""):
            try:
                args_conv[1] = str(float(args_conv[1]) * 1e6)
            except ValueError:
                pass
        if len(args_conv) >= 3 and args_conv[2] not in (None, ""):
            try:
                args_conv[2] = str(float(args_conv[2]) * 1e6)
            except ValueError:
                pass
        self.cmd_measure(args_conv)

    def cmd_measure(self, args):
        """Captures S11 measurement using connected VNA."""
        if not self.check_connections(require_vna=True):
            return

        if not args or args[0] in (None, ""):
            print("[Error] Measurement name required.")
            return

        name = args[0]
        f_start = float(args[1]) if len(args) > 1 and args[1] not in (None, "") else self.state.cal_f_start
        f_stop = float(args[2]) if len(args) > 2 and args[2] not in (None, "") else self.state.cal_f_stop
        f_points = int(args[3]) if len(args) > 3 and args[3] not in (None, "") else self.state.f_points
        out_dir = args[4] if len(args) > 4 and args[4] not in (None, "") else definitions.VNA_MEASURE_FILE

        try:
            s11 = my_vna.vna_measure(self.state.vna, f_start, f_stop, f_points, verbose=True)
            if s11 is not None:
                my_vna.vna_save(s11, out_dir, name)
                print(f"Saved S11 measurement '{name}' to {out_dir}.")
        except Exception as e:
            print(f"[VNA Error] Measurement failed: {e}")
            self.state.vna = None
            self.refresh_status()

    def cmd_plot(self, args):
        """Plots single S11 measurement spectrum."""
        if not args or args[0] in (None, ""):
            print("[Error] Measurement name required.")
            return
        name = args[0]
        my_vna.plot_s11_db(name, s11_dir=definitions.VNA_MEASURE_FILE, plot_dir=definitions.VNA_PLOT_FILE)

    def cmd_plot_ref(self, args):
        """Plots OECP reference S11 spectrum."""
        if not args or args[0] in (None, ""):
            print("[Error] Reference name required.")
            return
        name = args[0]
        my_vna.plot_s11_db(name, s11_dir=definitions.OECP_REF_FILE, plot_dir=definitions.VNA_PLOT_FILE, title=f"Reference S11: {name}")

    def cmd_new_ref(self, args):
        """Measures and records a new OECP standard reference."""
        if not self.check_connections(require_vna=True):
            return

        if not args or args[0] in (None, ""):
            print("[Error] Reference name required.")
            return
        name = args[0]
        try:
            my_oecp.get_new_ref(self.state.vna, name)
        except Exception as e:
            print(f"[VNA Error] Reference measurement failed: {e}")
            self.state.vna = None
            self.refresh_status()

    def cmd_test_mut(self, args):
        """Measures or processes Material Under Test dielectric properties."""
        if not args or args[0] in (None, ""):
            print("[Error] MUT name required.")
            return
        mut_name = args[0]
        overwrite = bool(args[1]) if len(args) > 1 and args[1] not in (None, "", "False", "false") else False
        
        if overwrite and not self.check_connections(require_vna=True):
            return

        try:
            my_oecp.test_material(
                mut_name, self.state.vna, overwrite=overwrite, 
                f_start=self.state.cal_f_start, f_stop=self.state.cal_f_stop, f_points=self.state.f_points
            )
        except Exception as e:
            print(f"[Error] Material test failed: {e}")

    def cmd_recon(self, args):
        """Performs automated anomaly detection reconstruction (Normal DAS + SVD/Hybrid)."""
        if not args or args[0] in (None, ""):
            print("[Error] Target scan name required.")
            return
        tar_name = args[0]
        margin_mm = float(args[1]) if len(args) > 1 and args[1] not in (None, "") else 0.0
        ref_name = args[2] if len(args) > 2 and args[2] not in (None, "") else None
        z_min = float(args[3]) if len(args) > 3 and args[3] not in (None, "") else 30.0
        z_max = float(args[4]) if len(args) > 4 and args[4] not in (None, "") else 180.0
        
        use_ref = True if (ref_name is not None and ref_name != "") else False

        # Step 1: Standard / Normal Reconstruction
        print(f"[Anomaly Detection] Step 1: Creating normal image for '{tar_name}'...")
        img.process_scan(
            tar_name=tar_name, ref_name=ref_name, margin_mm=margin_mm,
            z_min=z_min, z_max=z_max, use_ref=use_ref,
            use_svd=False, weighting_mode="none", save_name=f"{tar_name}"
        )

        # Step 2: Enhanced Reconstruction (SVD Clutter Removal + Hybrid WDAS)
        print(f"[Anomaly Detection] Step 2: Creating SVD + Hybrid image for '{tar_name}'...")
        img.process_scan(
            tar_name=tar_name, ref_name=ref_name, margin_mm=margin_mm,
            z_min=z_min, z_max=z_max, use_ref=use_ref,
            use_svd=True, weighting_mode="hybrid", save_name=f"{tar_name}_svd"
        )

        plot3d.plot_overlay_volumes(tar_name, f"{tar_name}_svd")


    def cmd_plot3d(self, args):
        """Renders 3D volume isosurfaces using Plotly."""
        if not args or args[0] in (None, ""):
            print("[Error] Volume filename required.")
            return
        filename = args[0]
        db = float(args[1]) if len(args) > 1 and args[1] not in (None, "") else 6.0
        plot3d.plot_interactive_volume(filename, dynamic_range_db=db)

    def cmd_calibrate(self, args):
        """Steps through 1-port Open-Short-Load calibration sequence."""
        if not self.check_connections(require_vna=True):
            return

        try:
            if self.state.cal_stage is None:
                mode = args[0] if args and args[0] not in (None, "") else "antenna"
                self.state.mode = mode
                self.state.cal_stage = "open"
                print(f"Starting {mode} calibration sequence. Attach OPEN standard and click button.")
            elif self.state.cal_stage == "open":
                my_vna.cal_measure_standard(self.state.vna, "open", self.state.cal_f_start, self.state.cal_f_stop, self.state.f_points)
                self.state.cal_stage = "short"
                print("OPEN standard measured. Attach SHORT standard and click calibration button.")
            elif self.state.cal_stage == "short":
                my_vna.cal_measure_standard(self.state.vna, "short", self.state.cal_f_start, self.state.cal_f_stop, self.state.f_points)
                self.state.cal_stage = "load"
                print("SHORT standard measured. Attach LOAD standard and click calibration button.")
            elif self.state.cal_stage == "load":
                my_vna.cal_measure_standard(self.state.vna, "load", self.state.cal_f_start, self.state.cal_f_stop, self.state.f_points)
                cal_path = os.path.join(definitions.CAL_SAVE_PATH, f"{self.state.mode}_cal.pkl")
                if my_vna.cal_run(save_path=cal_path):
                    print(f"Calibration successful! Saved to {cal_path}")
                else:
                    print("Calibration calculation failed.")
                self.state.cal_stage = None
        except Exception as e:
            print(f"[VNA Error] Calibration step failed: {e}")
            self.state.vna = None
            self.refresh_status()

    def cmd_cal_load(self, args):
        """Loads saved calibration file from disk."""
        mode = args[0] if args and args[0] not in (None, "") else "antenna"
        cal_path = os.path.join(definitions.CAL_SAVE_PATH, f"{mode}_cal.pkl")
        if my_vna.cal_load(cal_path):
            self.state.mode = mode
            print(f"Loaded {mode} calibration from {cal_path}.")
        else:
            print(f"Failed to load calibration from {cal_path}.")

    def cmd_cal_reset(self, args=None):
        """Resets active calibration."""
        my_vna.cal_reset()
        self.state.cal_stage = None
        print("Calibration reset.")

    def cmd_raster_scan(self, args):
        """Executes complete automated 2D raster scan routine."""
        if not self.check_connections(require_vna=True, require_rig=True):
            return

        if not args or args[0] in (None, ""):
            print("[Error] Output scan name required.")
            return
        name = args[0]
        w = float(args[1]) if len(args) > 1 and args[1] not in (None, "") else definitions.RASTER_WIDTH
        h = float(args[2]) if len(args) > 2 and args[2] not in (None, "") else definitions.RASTER_HEIGHT
        
        try:
            print(f"Homing rig before starting raster scan '{name}' ({w}x{h} mm)...")
            my_jetson.go_home(self.state.jetson_sock)
            
            out_dir = f"{definitions.VNA_SAVE_FILE}/{name}"
            print(f"Starting raster scan output to {out_dir}...")
            
            ok = my_jetson.raster(
                self.state.jetson_sock, scan_or_not=True, vna=self.state.vna,
                output_dir=out_dir, scan_width=w, scan_height=h,
                freq_start=self.state.cal_f_start, freq_end=self.state.cal_f_stop,
                freq_points=self.state.f_points
            )
            
            if ok:
                cols = int(w // self.state.step_size) + 1
                rows = int(h // self.state.step_size) + 1
                my_jetson.save_scan_metadata(name, step_size_mm=self.state.step_size, grid_cols=cols, grid_rows=rows)
                print(f"Raster scan '{name}' completed successfully. Returning home...")
                my_jetson.go_home(self.state.jetson_sock)
            else:
                print(f"Raster scan '{name}' encountered errors.")
        except (socket.error, OSError) as e:
            print(f"[Connection Error] Lost connection during raster scan: {e}")
            self.state.jetson_sock = None
            self.refresh_status()
        except Exception as e:
            print(f"[Error] Raster scan failed: {e}")


def main():
    root = tk.Tk()
    RigGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()