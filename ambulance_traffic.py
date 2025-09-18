import time
import numpy as np
import sounddevice as sd
import RPi.GPIO as GPIO
from road import Road
import database

# ================= CONFIG =================
SAMPLE_RATE = 48000
DURATION = 0.5
SIREN_FREQ_RANGE = (650, 1700)
SIREN_THRESHOLD = 50.0

# GPIO setup
GPIO.setmode(GPIO.BCM)
LED_PINS = {
    "Road 1": {"green": 17, "red": 27},
    "Road 2": {"green": 22, "red": 10},
    "Road 3": {"green": 9, "red": 11},
    "Road 4": {"green": 5, "red": 6}
}

for road_pins in LED_PINS.values():
    GPIO.setup(road_pins["green"], GPIO.OUT)
    GPIO.setup(road_pins["red"], GPIO.OUT)
    GPIO.output(road_pins["green"], GPIO.LOW)
    GPIO.output(road_pins["red"], GPIO.HIGH)

# Database and roads
database.create_database()
road1 = Road("Road 1", 40, 1000, 300, 1)
road2 = Road("Road 2", 60, 800, 300, 2)
road3 = Road("Road 3", 70, 1100, 300, 1.7)
road4 = Road("Road 4", 30, 700, 300, 1.2)
road1.next, road2.next, road3.next, road4.next = road2, road3, road4, road1
roads = [road1, road2, road3, road4]

active_road = road1
active_road.turn_green()
GPIO.output(LED_PINS[active_road.get_name()]["green"], GPIO.HIGH)
GPIO.output(LED_PINS[active_road.get_name()]["red"], GPIO.LOW)
start_time = time.time()
road_timestamp, camera_timestamp = None, None

print("System started. Monitoring for ambulance sirens and traffic cycles...")

# ================= HELPERS =================
def capture_audio():
    """Capture audio sample from microphone."""
    try:
        audio = sd.rec(int(DURATION*SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype='int16')
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print("Audio capture error:", e)
        return None

def detect_siren(audio):
    """Return True if ambulance siren detected."""
    if audio is None:
        return False
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/SAMPLE_RATE)
    power = np.abs(spectrum)
    siren_power = power[(freqs >= SIREN_FREQ_RANGE[0]) & (freqs <= SIREN_FREQ_RANGE[1])].sum()
    return siren_power > SIREN_THRESHOLD

def set_road_lights(active):
    for road in roads:
        if road == active:
            GPIO.output(LED_PINS[road.get_name()]["green"], GPIO.HIGH)
            GPIO.output(LED_PINS[road.get_name()]["red"], GPIO.LOW)
        else:
            GPIO.output(LED_PINS[road.get_name()]["green"], GPIO.LOW)
            GPIO.output(LED_PINS[road.get_name()]["red"], GPIO.HIGH)

# ================= MAIN LOOP =================
try:
    while True:
        curr_time = time.time()
        if road_timestamp is None:
            road_timestamp = curr_time
        if camera_timestamp is None:
            camera_timestamp = curr_time

        # ---- AUDIO ----
        audio = capture_audio()
        if detect_siren(audio):
            print("🚨 Emergency detected!")
            road1.set_hasEmergencyVehicle(True)
        else:
            road1.set_hasEmergencyVehicle(False)

        # ---- NORMAL TRAFFIC CYCLE ----
        if curr_time - start_time > active_road.get_green_time():
            active_road.turn_red()
            active_road = active_road.next
            active_road.turn_green()
            start_time = curr_time
            set_road_lights(active_road)
            print(f"[Normal Cycle] Green light: {active_road.get_name()}")

        # ---- EMERGENCY PRIORITY ----
        for road in roads:
            if road.get_hasEmergencyVehicle():
                print(f"[Emergency Priority] Green light: {road.get_name()}")
                for r in roads:
                    r.turn_red()
                road.turn_green()
                active_road = road
                start_time = curr_time
                set_road_lights(active_road)
                break

        # ---- VEHICLE COUNT UPDATE ----
        if curr_time - road_timestamp > 1:
            print("\n[Vehicle Counts Update]")
            for road in roads:
                road.update()
                print(f"{road.get_name()} - Vehicle count: {road.get_vehicle_count()}")
            road_timestamp = curr_time

        # ---- CAMERA UPDATE SIMULATION ----
        if curr_time - camera_timestamp > 10:
            print("\n[Camera Update] All roads updated.")
            camera_timestamp = curr_time

except KeyboardInterrupt:
    print("\nExiting program...")
    GPIO.cleanup()

