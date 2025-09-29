import RPi.GPIO as GPIO
import sounddevice as sd
import numpy as np
import time
import signal
import sys

# ==============================
# CONFIGURATION
# ==============================
SAMPLE_RATE = 44100   # Sample rate in Hz (standard for audio)
DURATION = 2          # Duration of each recording in seconds
THRESHOLD = 8.0       # Siren detection threshold (tune this if false triggers occur)
DEVICE_INDEX = 10     # Input device index (from `python3 -m sounddevice`)

# GPIO pin numbers (BCM mode)
GREEN_LED = 17
RED_LED = 27

# ==============================
# SETUP
# ==============================
GPIO.setmode(GPIO.BCM)
GPIO.setup(GREEN_LED, GPIO.OUT)
GPIO.setup(RED_LED, GPIO.OUT)

print("Starting ambulance siren detection. Press Ctrl+C to stop.")

# ==============================
# CLEAN EXIT HANDLER
# ==============================
def cleanup_and_exit(sig=None, frame=None):
    print("\nInterrupted by user - cleaning up GPIO and exiting.")
    GPIO.cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)

# ==============================
# DETECTION FUNCTION
# ==============================
def detect_ambulance(audio):
    """
    Detects ambulance siren using frequency-domain analysis (FFT).
    Returns score value (higher = stronger siren-like signal).
    """
    fft = np.fft.fft(audio.flatten())
    magnitude = np.abs(fft)
    score = np.mean(magnitude[1000:2000]) / np.mean(magnitude)  # Example detection logic
    return score

# ==============================
# MAIN LOOP
# ==============================
try:
    while True:
        # Record audio sample
        audio = sd.rec(int(DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype='float32',
                       device=DEVICE_INDEX)
        sd.wait()

        # Detect siren
        score = detect_ambulance(audio)

        if score > THRESHOLD:
            print(f"{time.strftime('%H:%M:%S')} - Ambulance detected (score={score:.2f}) - GREEN ON")
            GPIO.output(GREEN_LED, GPIO.HIGH)
            GPIO.output(RED_LED, GPIO.LOW)
        else:
            print(f"{time.strftime('%H:%M:%S')} - No ambulance (score={score:.2f}) - RED ON")
            GPIO.output(GREEN_LED, GPIO.LOW)
            GPIO.output(RED_LED, GPIO.HIGH)

        time.sleep(0.5)

except KeyboardInterrupt:
    cleanup_and_exit()
