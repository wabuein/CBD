import time
import re
import cv2
import numpy as np
import serial
import RPi.GPIO as GPIO
from ultralytics import YOLO

# -----------------------------
# YOUR RPi PIN LIST (BCM)
# -----------------------------
PIN_SERVO = 18   # Pin 12 (GPIO18)  -> SC90 Servo signal

PIN_TRIG1 = 17   # Pin 13 (GPIO17)  -> US1 TRIG  (choose which is width/thickness below)
PIN_ECHO1 = 22   # Pin 15 (GPIO22)  -> US1 ECHO  (through divider)
PIN_TRIG2 = 23   # Pin 16 (GPIO23)  -> US2 TRIG
PIN_ECHO2 = 24   # Pin 18 (GPIO24)  -> US2 ECHO

PIN_IR1 = 25     # Pin 22 (GPIO25)  -> IR1 (stop + length gate)
PIN_IR2 = 5      # Pin 29 (GPIO5)   -> IR2 (servo timing)

# -----------------------------
# CONFIG YOU MUST SET
# -----------------------------
# Belt speed when running at your chosen signal voltage (e.g. 24V moderate).
# Start with manufacturer: 50 mm/s, then calibrate.
BELT_SPEED_MM_S = 50.0

# IR polarity: many IR modules output LOW when triggered.
IR_ACTIVE_LOW = True

# YOLO config
MODEL_PATH = "yolo11n.pt"
CAM_SOURCE = 1
YOLO_FRAMES = 600
YOLO_IMGSZ = 512
YOLO_CONF  = 0.25

# Pico USB serial
PICO_PORT = "/dev/ttyACM0"  # might be /dev/ttyACM1
PICO_BAUD = 115200

# Script file
SCRIPT_FILE = "script.txt"

# Servo positions (duty %) - tune for your hardware
SERVO_NEUTRAL = 7.5
SERVO_LEFT    = 10.5
SERVO_RIGHT   = 4.5

# Debounce / stability times
IR_OFF_STABLE_S = 0.12   # IR1 must be OFF continuously for this long to count as cleared
LOOP_DT = 0.005

# Ultrasonic
SPEED_OF_SOUND = 343.0
ULTRA_TIMEOUT_S = 0.03

# Which ultrasonic does what (based on your placement):
# - Top ultrasonic = thickness
# - Side ultrasonic = width
# Choose which sensor (1 or 2) is TOP/SIDE.
US_TOP = 2     # set to 1 or 2
US_SIDE = 1    # set to 1 or 2

def gpio_setup():
    GPIO.setmode(GPIO.BCM)

    # IR inputs: use pull-ups if active-low
    pud = GPIO.PUD_UP if IR_ACTIVE_LOW else GPIO.PUD_DOWN
    GPIO.setup(PIN_IR1, GPIO.IN, pull_up_down=pud)
    GPIO.setup(PIN_IR2, GPIO.IN, pull_up_down=pud)

    # Ultrasonics
    for trig in (PIN_TRIG1, PIN_TRIG2):
        GPIO.setup(trig, GPIO.OUT)
        GPIO.output(trig, 0)
    for echo in (PIN_ECHO1, PIN_ECHO2):
        GPIO.setup(echo, GPIO.IN)

    # Servo
    GPIO.setup(PIN_SERVO, GPIO.OUT)

def ir_triggered(pin: int) -> bool:
    v = GPIO.input(pin)
    return (v == 0) if IR_ACTIVE_LOW else (v == 1)

def pico_send(ser: serial.Serial, cmd: str, timeout=1.0) -> str:
    ser.write((cmd.strip() + "\n").encode("utf-8"))
    ser.flush()
    t0 = time.time()
    while time.time() - t0 < timeout:
        if ser.in_waiting:
            return ser.readline().decode("utf-8", errors="ignore").strip()
        time.sleep(0.01)
    return ""

def servo_init():
    pwm = GPIO.PWM(PIN_SERVO, 50)  # 50Hz
    pwm.start(0)
    return pwm

def servo_set(pwm, duty: float):
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.35)
    pwm.ChangeDutyCycle(0)

def measure_distance_cm(trig_pin: int, echo_pin: int) -> float | None:
    GPIO.output(trig_pin, 0)
    time.sleep(0.000002)
    GPIO.output(trig_pin, 1)
    time.sleep(0.00001)
    GPIO.output(trig_pin, 0)

    t0 = time.time()
    while GPIO.input(echo_pin) == 0:
        if time.time() - t0 > ULTRA_TIMEOUT_S:
            return None
    start = time.time()

    while GPIO.input(echo_pin) == 1:
        if time.time() - start > ULTRA_TIMEOUT_S:
            return None
    end = time.time()

    dt = end - start
    dist_m = (SPEED_OF_SOUND * dt) / 2.0
    return dist_m * 100.0

def us_pins(which: int):
    if which == 1:
        return PIN_TRIG1, PIN_ECHO1
    return PIN_TRIG2, PIN_ECHO2

def read_ultrasound_averaged(which: int, samples: int = 7, delay: float = 0.02):
    trig, echo = us_pins(which)
    vals = []
    for _ in range(samples):
        d = measure_distance_cm(trig, echo)
        if d is not None:
            vals.append(d)
        time.sleep(delay)
    if not vals:
        return None
    vals.sort()
    # median is robust
    return vals[len(vals)//2]

def parse_user_script(path: str) -> dict:
    """
    Accepts:
    Object: apple, optional: color:red, optional: width:50, length:80, thickness:20, min_conf:0.4, path:C5
    """
    out = {}
    try:
        txt = open(path, "r", encoding="utf-8").read().strip()
    except FileNotFoundError:
        return out

    def grab(key):
        m = re.search(rf"{key}\s*:\s*([^,\n]+)", txt, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None

    out["object"] = grab("object")
    out["color"] = grab("color")
    out["width"] = float(grab("width")) if grab("width") else None
    out["length"] = float(grab("length")) if grab("length") else None
    out["thickness"] = float(grab("thickness")) if grab("thickness") else None

    mc = grab("min_conf") or grab("minimum confidence")
    out["min_conf"] = float(mc) if mc else None

    out["path"] = grab("path")
    return out

def decide_direction(rule: dict) -> str:
    p = (rule.get("path") or "").strip().upper()
    if re.fullmatch(r"C\d+", p):
        return "LEFT"
    return "RIGHT"

def run_yolo_600(model: YOLO, cap: cv2.VideoCapture,
                frames=YOLO_FRAMES, imgsz=YOLO_IMGSZ, conf=YOLO_CONF):
    confs = []
    class_counts = {}
    t0 = time.time()

    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)
        for r in results:
            names = r.names
            boxes = r.boxes
            if boxes is None:
                continue
            for cls, c in zip(boxes.cls, boxes.conf):
                c = float(c)
                confs.append(c)
                ci = int(cls)
                cname = names.get(ci, str(ci)) if isinstance(names, dict) else names[ci]
                class_counts[cname] = class_counts.get(cname, 0) + 1

    t1 = time.time()
    top = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "frames_done": len(confs),
        "mean_conf": float(np.mean(confs)) if confs else 0.0,
        "top_classes": top,
        "yolo_seconds": t1 - t0,
    }

def wait_ir_off_stable(pin: int, stable_s: float) -> float:
    """
    Wait until IR is NOT triggered continuously for stable_s.
    Returns the timestamp when stable OFF is achieved.
    """
    off_start = None
    while True:
        if not ir_triggered(pin):
            if off_start is None:
                off_start = time.time()
            if time.time() - off_start >= stable_s:
                return time.time()
        else:
            off_start = None
        time.sleep(LOOP_DT)

def main():
    gpio_setup()

    servo_pwm = servo_init()
    servo_set(servo_pwm, SERVO_NEUTRAL)

    # Camera + model
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAM_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Pico serial
    ser = serial.Serial(PICO_PORT, PICO_BAUD, timeout=0.1)
    time.sleep(1.0)
    pico_send(ser, "PING")

    # Start conveyor
    pico_send(ser, "RUN")
    print("RUNNING. Waiting for IR1 to detect an object...")

    try:
        while True:
            # ---------- WAIT FOR IR1 ON ----------
            while not ir_triggered(PIN_IR1):
                time.sleep(LOOP_DT)

            # Timer1 starts: object is detected at IR1
            t_block_start = time.time()
            print(f"IR1 ON -> t_block_start={t_block_start:.3f}")

            # Stop conveyor + start Timer2
            pico_send(ser, "STOP")
            t_stop_start = time.time()
            print(f"STOP -> t_stop_start={t_stop_start:.3f}")

            # IMPORTANT: your method assumes IR1 stays ON during stop
            # We'll warn if it drops.
            if not ir_triggered(PIN_IR1):
                print("WARNING: IR1 went OFF immediately after stopping. Your length math may be wrong. "
                      "Move IR1 slightly upstream so the object remains detected while stopped.")

            # ---------- MEASURE + YOLO WHILE STOPPED ----------
            # Ultrasound (averaged)
            top_cm  = read_ultrasound_averaged(US_TOP)
            side_cm = read_ultrasound_averaged(US_SIDE)
            print(f"Ultrasound: top={top_cm} cm, side={side_cm} cm")

            # YOLO 600 frames
            y = run_yolo_600(model, cap)
            print(f"YOLO: {y}")

            # Read script and decide direction
            rule = parse_user_script(SCRIPT_FILE)
            direction = decide_direction(rule)
            print(f"Rule: {rule} => direction={direction}")

            # Resume conveyor + stop Timer2
            pico_send(ser, "RUN")
            t_stop_end = time.time()
            print(f"RUN -> t_stop_end={t_stop_end:.3f}")

            # ---------- Timer1 ends when IR1 is OFF (stable) ----------
            t_block_end = wait_ir_off_stable(PIN_IR1, IR_OFF_STABLE_S)
            print(f"IR1 OFF stable -> t_block_end={t_block_end:.3f}")

            # Compute length with YOUR method
            t_block = t_block_end - t_block_start
            t_stop  = t_stop_end - t_stop_start
            t_move  = t_block - t_stop
            length_mm = max(0.0, t_move * BELT_SPEED_MM_S)

            print(f"t_block={t_block:.3f}s, t_stop={t_stop:.3f}s, t_move={t_move:.3f}s")
            print(f"Estimated length = {length_mm:.1f} mm (speed={BELT_SPEED_MM_S} mm/s)")

            # ---------- SERVO TIMING USING IR2 ----------
            # Wait for object to reach diverter zone, then swing
            print("Waiting for IR2 (diverter cue)...")
            # IR2 trigger
            while not ir_triggered(PIN_IR2):
                time.sleep(LOOP_DT)

            print("IR2 triggered -> swinging servo")
            if direction == "LEFT":
                servo_set(servo_pwm, SERVO_LEFT)
            else:
                servo_set(servo_pwm, SERVO_RIGHT)

            # Hold briefly then reset neutral
            time.sleep(0.35)
            servo_set(servo_pwm, SERVO_NEUTRAL)

            print("Cycle complete. Ready for next object.\n")
            time.sleep(0.15)

    finally:
        try:
            pico_send(ser, "STOP")
        except Exception:
            pass
        cap.release()
        servo_pwm.stop()
        GPIO.cleanup()
        ser.close()

if __name__ == "__main__":
    main()