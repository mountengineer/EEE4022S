import definitions, my_vna, time, socket, json, os

def jetson_send(sock, cmd, wait=0.0):
    sock.sendall((cmd + "\n").encode())
    if wait > 0:
        time.sleep(wait)

def jetson_read(sock, expected, timeout=1.5):
    """Listen for response"""
    try:
        start_time = time.perf_counter()
        sock.settimeout(0.5)
        while (time.perf_counter() - start_time) < timeout:
            try:
                data = sock.recv(1024).decode('utf-8').strip().lower()
                if not data:
                    print("[Server disconnected]")
                    return False
                data = data.split("\n")
                for i in range(len(data)):
                    print(f"[Jetson] {data[i].strip()}")
                    if "reached limit" in data[i] or "e-stop triggered" in data[i]:
                        return False
                    if expected.lower() in data[i]:
                        return True
            except socket.timeout:
                continue
            except Exception:
                return False
        print(f"Timed out. No response after {timeout} seconds.")
        return False
    finally:
        sock.settimeout(None)

def save_scan_metadata(scan_name, step_size_mm, grid_cols=None, grid_rows=None, extra_meta=None):
    """Saves scan metadata to a metadata.json file in the target directory."""
    scan_dir = f"{definitions.VNA_SAVE_FILE}/{scan_name}"
    os.makedirs(scan_dir, exist_ok=True)
    metadata = {
        "step_size_mm": float(step_size_mm),
        "grid_cols": int(grid_cols) if grid_cols else None,
        "grid_rows": int(grid_rows) if grid_rows else None,
    }
    if extra_meta:
        metadata.update(extra_meta)

    meta_path = os.path.join(scan_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {meta_path}")

def row(sock, cols, row_index, scan_or_not, vna=None, output_dir=definitions.VNA_SAVE_FILE, 
        freq_start=None, freq_end=None, freq_points=None):
    if freq_start is None:
        freq_start = max(definitions.ANTENNA_MIN_FREQ, definitions.VNA_MIN_FREQ, definitions.F_START)
    if freq_end is None:
        freq_end = min(definitions.ANTENNA_MAX_FREQ, definitions.VNA_MAX_FREQ, definitions.F_STOP)
    if freq_points is None:
        freq_points = definitions.F_POINTS
    
    direction = None
    if (row_index % 2) == 1:
        direction = "left"
    else:
        direction = "right"
    for i in range(cols):
        #time.sleep(SCAN_TIME)
        if scan_or_not and vna != None:
            if (row_index % 2) == 0:
                col_index = i
            else:
                col_index = cols-1-i
            my_vna.vna_scan(vna, f"Scan_{append_zeros(row_index)}_{append_zeros(col_index)}", output_dir=output_dir, 
                            f_start=freq_start, f_stop=freq_end, f_points=freq_points, verbose=False)
            print(f"Scanned position: R:{row_index}, C:{col_index}")
        if i != cols-1:
            jetson_send(sock, f"move {definitions.STEP_SIZE} {direction}")
            if not jetson_read(sock, f"moved {float(definitions.STEP_SIZE)}mm {direction}", timeout=5):
                print(f"Column {col_index} move error.")
                return False
    return True

def raster(sock, scan_or_not, vna=None, output_dir=definitions.VNA_SAVE_FILE,
            scan_width=definitions.RASTER_WIDTH, scan_height=definitions.RASTER_HEIGHT, freq_start=None, 
            freq_end=None, freq_points=None):
    # Read current values into these as defaults
    if freq_start is None:
        freq_start = max(definitions.ANTENNA_MIN_FREQ, definitions.VNA_MIN_FREQ, definitions.F_START)
    if freq_end is None:
        freq_end = min(definitions.ANTENNA_MAX_FREQ, definitions.VNA_MAX_FREQ, definitions.F_STOP)
    if freq_points is None:
        freq_points = definitions.F_POINTS

    print(f"Scan height: {scan_height}, step size: {definitions.STEP_SIZE}")

    rows = int(scan_height//definitions.STEP_SIZE) + 1
    cols = int(scan_width//definitions.STEP_SIZE) + 1
    print(f"Performing raster scan with {rows} rows and {cols} columns.")
    print(f"Sweeps will be from {freq_start}Hz to {freq_end}Hz across {freq_points} points")
    error = False
    for i in range(rows):
        print(f"Row {i+1}...")
        # time.sleep(SCAN_TIME)
        if not row(sock, cols, i, scan_or_not=scan_or_not, vna=vna, output_dir=output_dir, 
                   freq_start=freq_start, freq_end=freq_end, freq_points=freq_points):
            print(f"Error in Row {i+1}.")
            error = True
            break
        if (i != rows-1):
            jetson_send(sock, f"move {definitions.STEP_SIZE} up")
            if not jetson_read(sock, f"moved {float(definitions.STEP_SIZE)}mm up", timeout=5):
                print(f"Row {i+1} move error.")
                error = True
                break
    if error:
        print("Failed to perform raster scan.")
        exit(1)
    return True

def stop(sock, axis=""):
    if not axis:
        jetson_send(sock, "stop x", wait=0.05)
        jetson_send(sock, "stop y")
    else:
        jetson_send(sock, f"stop {axis}")

def estop(sock):
    jetson_send(sock, "estop")

def home(sock, axis="", timeout = 45):
    if not axis:
        go_home(sock, timeout=timeout)
        return
    jetson_send(sock, f"home {axis}")
    if not jetson_read(sock, f"home {axis} reached.", timeout):
        print(f"Failed to reach home {axis}.")

def go_home(sock, timeout=45):
    jetson_send(sock, "home")
    if not jetson_read(sock, "home x and y reached.", timeout):
        print("Failed to reach home")
        exit(1)

def append_zeros(some_int, max_digits=2):
    num_zeros = None
    for i in range(max_digits):
        if (some_int // (10 ** i)) < 10:
            # print(i)
            num_zeros = max_digits - (i + 1)
            break
    if num_zeros == None:
        return some_int
    return f"{num_zeros*"0"}{some_int}"