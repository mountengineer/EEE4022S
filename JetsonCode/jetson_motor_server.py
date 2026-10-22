import socket
import threading
import time
import queue
import sys

try:
    import Jetson.GPIO as GPIO
except ImportError:
    class MockGPIO:
        BOARD = OUT = IN = HIGH = LOW = PUD_UP = None
        def setmode(*a, **kw): pass
        def setup(*a, **kw): pass
        def output(*a, **kw): pass
        def input(*a, **kw): return 1
        def cleanup(*a, **kw): pass
        def PWM(*a, **kw):
            class _PWM:
                def start(self, duty): pass
                def ChangeDutyCycle(self, duty): pass
                def stop(self): pass
            return _PWM()
    GPIO = MockGPIO()

# ---------------- GPIO Pins ----------------
MOTOR_X_PIN = 32
MOTOR_Y_PIN = 33
DIRECTION_X = 37
DIRECTION_Y = 35
LIMIT_PIN = 40
# SYNC_IN = 31 # seems not to be used anymore

estop_event = threading.Event()

GPIO.setmode(GPIO.BOARD)
GPIO.setup(MOTOR_X_PIN, GPIO.OUT)
GPIO.setup(MOTOR_Y_PIN, GPIO.OUT)
GPIO.setup(DIRECTION_X, GPIO.OUT)
GPIO.setup(DIRECTION_Y, GPIO.OUT)
GPIO.setup(LIMIT_PIN, GPIO.IN)

# ---------------- PWM setup ----------------
FIXED_DUTY = 50  # Fixed speed

frequency = 7500
x_pwm = GPIO.PWM(MOTOR_X_PIN, frequency)
y_pwm = GPIO.PWM(MOTOR_Y_PIN, frequency)
x_pwm.start(0)
y_pwm.start(0)

# ---------------- Motor control state ----------------
running = True
limited = False # Flag for when the limit interrupts trigger
current_axis = "x"
current_dir = GPIO.LOW
BOUNCE_DIST = 2 # mm

# ---------------- Interrupt / Event Callback ----------------

def reached_limit(channel):
    global limited
    limited = True

GPIO.add_event_detect(
    LIMIT_PIN,
    GPIO.RISING,
    callback=reached_limit,
    bouncetime=50 # Need to allow time for limit pin to be disabled again
)

def emergency_stop():
    set_motor("x", False)
    set_motor("y", False)
    estop_event.set()

# ---------------- Helper functions ----------------
def setDirection(direction):
    global current_axis, current_dir
    if direction in ["u", "up"]:
        GPIO.output(DIRECTION_Y, GPIO.LOW)
        current_axis = "y"
        current_dir = GPIO.LOW
    elif direction in ["d", "down"]:
        GPIO.output(DIRECTION_Y, GPIO.HIGH)
        current_axis = "y"
        current_dir = GPIO.HIGH
    elif direction in ["r", "right"]:
        GPIO.output(DIRECTION_X, GPIO.LOW)
        current_axis = "x"
        current_dir = GPIO.LOW
    elif direction in ["l", "left"]:
        GPIO.output(DIRECTION_X, GPIO.HIGH)
        current_axis = "x"
        current_dir = GPIO.HIGH

def set_motor(axis, on_off):
    """Turn motor PWM on or off."""
    pwm = None
    if axis == "x":
        pwm = x_pwm
    elif axis == "y":
        pwm = y_pwm
    else:
        return
    global current_axis
    if on_off:
        pwm.ChangeDutyCycle(FIXED_DUTY)
        current_axis = axis
    else:
        pwm.ChangeDutyCycle(0)

def is_number(s):
    """Checks if string is a number"""
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_opposite_dir(direction):
    if direction in ["d", "down"]:
        return "u"
    if direction in ["u", "up"]:
        return "d"
    if direction in ["l", "left"]:
        return "r"
    if direction in ["r", "right"]:
        return "l"

# ---------------- Homing ----------------
def home_system(send_fn=None):
    if home("x", send_fn=send_fn):
        time.sleep(0.1)
        if home("y", send_fn=send_fn):
            msg = "Homing complete"
            print(msg)
            if send_fn:
                send_fn(msg)

def home(axis, send_fn=None):
    global limited
    limited = False
    if axis == "x":
        setDirection("l")
    elif axis == "y":
        setDirection("d")
    else:
        if (send_fn):
            send_fn(f"Invalid axis: {axis}. Must be in {{\"x\", \"y\"}})")
        return

    # Move until limit reached
    set_motor(axis, True)
    while not limited:
        if estop_event.is_set():
            set_motor(axis, False)
            return False
        time.sleep(0.001)

    set_motor(axis, False)

    if limited:
        time.sleep(0.5)
        limited = False
        if GPIO.input(LIMIT_PIN) == 1:
            if axis == "x":
                move_distance(BOUNCE_DIST, "r")
            elif axis == "y":
                move_distance(BOUNCE_DIST, "u")
            else:
                print("Impossible!")
        
    
    msg = f"Homing {axis} complete"
    print(msg)
    if send_fn:
        send_fn(msg)
    return True

# ---------------- Strip map SAR movement ----------------
def move_distance(distance_mm, direction, freq_hz=frequency, send_fn=None):
    global limited
    limited = False
    steps_needed = ((distance_mm * 200 * 16) / 8.0)  # 400 STEPS_PER_MM
    sleep_duration = steps_needed/freq_hz
    
    # delay = 1.0 / (2 * freq_hz) # Does not seem to be used anymore
    axis = None
    if direction in ["r", "right", "l", "left"]:
        axis = "x"
    elif direction in ["u", "up", "d", "down"]:
        axis = "y"
    else:
        if send_fn: # Send error message to explain issue
            send_fn(f"Invalid direction: {direction}. Must be in {{\"u\"\, \"up\", \"d\", \"down\", \"r\", \"right\", \"l\", \"left\"}} (case not sensitive)")
        return

    elapsed_time = sleep_duration
    setDirection(direction)
    set_motor(axis, True)

    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) < sleep_duration:
        if limited or estop_event.is_set():
            elapsed_time = time.perf_counter() - start_time
            break
        time.sleep(0.001)

    set_motor(axis, False)

    if limited:
        time.sleep(0.5)
        limited = False
        actual_steps = freq_hz * elapsed_time
        actual_distance = actual_steps * 8.0 / (16.0 * 200.0)
        msg = f"Reached limit. Moved {actual_distance}mm {direction}"
        if GPIO.input(LIMIT_PIN) == 1:
            move_distance(BOUNCE_DIST, get_opposite_dir(direction=direction))
    else:
        msg = f"Moved {distance_mm}mm {direction}"
        
    if estop_event.is_set():
        msg = "E-STOP: move aborted"
    print(msg)
    if send_fn:
        send_fn(msg)
        
# ---------------- TCP Server ----------------
TCP_IP = "192.168.180.2"  # Jetson IP
TCP_PORT = 5005

def command_worker(cmd_queue, send_fn):
    while True:
        item = cmd_queue.get()
        if item is None:
            return
        if estop_event.is_set():
            with cmd_queue.mutex:
                cmd_queue.queue.clear()
            estop_event.clear()   # fresh command clears any previous e-stop
        tokens = item
        if tokens[0] in ["start", "stop"]:
            if len(tokens) == 1: # Ensure array can be indexed to [1]
                tokens.append("None")
            if tokens[1] not in ["x", "y"]:
                send_fn(f"Invalid axis: {tokens[1]}. Must be in {{\"x\", \"y\"}}")
                continue
            axis = tokens[1]
            if tokens[0] == "start":
                set_motor(axis, True)
                send_fn(f"Motor {axis} on.")
            else:
                set_motor(axis, False)
                send_fn(f"Motor {axis} off.")
        elif tokens[0] == "home":
            if len(tokens) == 1:
                home_system(send_fn=send_fn)
                send_fn(f"Home x and y reached.")
            else:
                if tokens[1] not in ["x", "y"]:
                    send_fn(f"Invalid axis: {tokens[1]}. Must be in {{\"x\", \"y\"}}")
                    continue
                axis = tokens[1]
                if home(axis, send_fn=send_fn):
                    send_fn(f"Home {axis} reached.")
        elif tokens[0] == "move":
            if len(tokens) != 3:
                send_fn(f"Invalid command: {tokens}. Must be in form: move {{distance}} {{direction}}")
                continue
            if not is_number(tokens[1]):
                send_fn(f"Invalid distance: {tokens[1]}. Must be a number.")
                continue
            if tokens[2] not in ["u", "up", "d", "down", "r", "right", "l", "left"]:
                send_fn(f"Invalid direction: {tokens[2]}. Must be in {{\"u\"\, \"up\", \"d\", \"down\", \"r\", \"right\", \"l\", \"left\"}} (case not sensitive)")
                continue
            move_distance(float(tokens[1]), tokens[2], send_fn=send_fn)
        elif tokens[0] == "exit":
            send_fn("Exiting...")
            return
        else:
            send_fn(f"Invalid command: {tokens[0]}. Must be in {{\"start\", \"stop\", \"home\", \"move\"}}")
            continue

def client_thread(conn, addr):
    print(f"Connected by {addr}")

    def send_msg(msg):
        try:
            conn.sendall((msg + "\n").encode())
        except:
            pass  # client may have disconnected

    cmd_queue = queue.Queue()
    worker = threading.Thread(target=command_worker, args=(cmd_queue, send_msg), daemon=True)
    worker.start()
    
    try:
        while True:
            data = conn.recv(1024).decode('utf-8').strip().lower()
            if not data:
                break
            print(f"Received: {data}")
            send_msg(f"Received: {data}")
            tokens = data.split()

            if tokens[0] == "estop":
                emergency_stop()
                send_msg("E-STOP triggered")
                continue

            cmd_queue.put(tokens)
    except Exception as e:
        print(f"Client error: {e}")
    finally:
        cmd_queue.put(None)
        conn.close()

def start_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    print(f"Server listening on {TCP_IP}:{TCP_PORT}")
    while True:
        conn, addr = s.accept()
        threading.Thread(target=client_thread, args=(conn, addr), daemon=True).start()

# ---------------- Main ----------------
try:
    start_server()
except KeyboardInterrupt:
    running = False
    GPIO.cleanup()
