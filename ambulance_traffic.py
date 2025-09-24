import numpy as np
import sounddevice as sd
import RPi.GPIO as GPIO
import time

# ---------------- CONFIGURATION ----------------
GREEN_PIN = 17       # BCM 17 -> Physical pin 11
RED_PIN = 27         # BCM 27 -> Physical pin 13

SAMPLE_RATE = 48000  # Supported by USB mic hw:2,0
CHUNK       = 2048   # Audio block size

SIREN_RANGE = (650, 1700)  # Hz range for ambulance siren
THRESHOLD   = 0.15          # Detection sensitivity (0–1 normalized)

REQUIRED_FRAMES = 1          # Consecutive frames needed to trigger red LED

# ---------------- GPIO SETUP -------------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(RED_PIN, GPIO.OUT)

def set_leds(green_on: bool, red_on: bool):
    GPIO.output(GREEN_PIN, green_on)
    GPIO.output(RED_PIN, red_on)

def detect_siren(block):
    """Return True if siren-like signal is detected in audio block."""
    windowed = block * np.hanning(len(block))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(block), 1 / SAMPLE_RATE)
    magnitude = np.abs(spectrum)
    magnitude = magnitude / np.max(magnitude + 1e-9)  # normalize

    # Average magnitude in siren band
    siren_band = magnitude[(freqs >= SIREN_RANGE[0]) & (freqs <= SIREN_RANGE[1])]
    avg_power = np.mean(siren_band) if len(siren_band) > 0 else 0
    return avg_power > THRESHOLD

print("Traffic signal active. Listening for ambulance siren... Press Ctrl+C to stop.")

consecutive = 0

try:
    with sd.InputStream(device="hw:2,0", channels=1,
                        samplerate=SAMPLE_RATE, blocksize=CHUNK) as stream:
        while True:
            audio, _ = stream.read(CHUNK)
            audio = audio.flatten()
            if detect_siren(audio):
                consecutive += 1
            else:
                consecutive = 0

            if consecutive >= REQUIRED_FRAMES:
                # Siren detected: Red on, Green off
                set_leds(False, True)
                print("Ambulance siren detected! Red LED ON")
            else:
                # Normal traffic: Green on, Red off
                set_leds(True, False)

            time.sleep(0.1)  # adjust speed of loop

except KeyboardInterrupt:
    print("\nExiting and cleaning up GPIO...")
finally:
    GPIO.cleanup()
