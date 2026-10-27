import matplotlib
from pathlib import Path

# Base directory targeting Proj/out/ relative to this file's location (Proj/src/)
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "out"
REF_DIR = BASE_DIR / "ref"

# Optional: ensure the output directory actually exists on disk
OUT_DIR.mkdir(parents=True, exist_ok=True)

VNA_SAVE_FILE = str(OUT_DIR / "RasterScanOutputs")
OECP_SAVE_FILE = str(OUT_DIR / "DielectricResults")
OECP_PLOT_FILE = str(OUT_DIR / "DielectricPlots")
VNA_MEASURE_FILE = str(OUT_DIR / "S11Outputs")
VNA_PLOT_FILE = str(OUT_DIR / "S11Plots")
PROCESS_SAVE_FILE = str(OUT_DIR / "Outputs")
OECP_REF_FILE = str(REF_DIR / "OECPReferences")
CAL_SAVE_PATH = str(REF_DIR / "Calibration")

matplotlib.use("Agg")

JETSON_IP = "192.168.180.2"
PORT = 5005
VNA_PORT = "COM10" # Need to figure out which port it will be on/set this dynamically
BAUD = 115200

#OECP Reference materials
R1 = "Air"
R2 = "DeionisedWater"
R3 = "Propanol2"
R4 = "Ethanol"
REFERENCES = [R1, R2, R3, R4]
# Corresponding in PyOECP: NB: MUST BE CHANGED WITH ABOVE
M1 = "Open"
M2 = "Water_Kaatze"
M3 = "Propanol2_Barthel"
M4 = "Ethanol_Barthel"


SCAN_TIME = 0.1
STEP_SIZE = 4 # in mm
RASTER_WIDTH = 400 # in mm
RASTER_HEIGHT = 250 # in mm
ANTENNA_MIN_FREQ = 2.0e9
ANTENNA_MAX_FREQ = 18.0e9
VNA_MIN_FREQ = 100.0e6
VNA_MAX_FREQ = 6.3e9
F_START = 100.0e6
F_STOP = 6.3e9
F_POINTS = 101

INI_T = 5e-10
FIN_T = 1e-9
GATE_STEP = 2e-11
NUM_GATE_STEPS = 10
N_TIME_POINTS = 700

CAL_ORDER = ["open", "short", "load"]