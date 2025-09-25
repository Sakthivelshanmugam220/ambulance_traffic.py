import sounddevice as sd
import numpy as np
import RPi.GPIO as GPIO
import time
import datetime

# === Parameters ===
SAMPLE_RATE = 44100       # Hz
DURATION = 2.0            # seconds per recording
THRESHOLD = 15.0          # stricter threshold to reduce false positives
SIREN_MIN = 600           # Hz lower bound of ambulance siren
SIREN_MAX = 1500          # Hz upper bound of ambulance siren

# === GPIO setup ===
RED_PIN = 17
GREEN_PIN = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)

# Start with RED ON (normal traffic)
GPIO.output(RED_PIN, GPIO.HIGH)
GPIO.output(GREEN_PIN, GPIO.LOW)


def detect_ambulance(audio, samplerate):
    """Return True if ambulance siren is detected in audio"""
    # FFT
    fft_spectrum = np.fft.rfft(audio[:, 0])
    freqs = np.fft.rfftfreq(len(audio), 1/samplerate)
    magnitude = np.abs(fft_spectrum)

    # Focus only on siren range
    siren_band = (freqs >= SIREN_MIN) & (freqs <= SIREN_MAX)
    score = np.mean(magnitude[siren_band]) / (np.mean(magnitude) + 1e-6)

    now = datetime.datetime.now().strftime("%H:%M:%S")
    if score > THRESHOLD:
        print(f"{now} - Ambulance detected (score={score:.2f})")
        return True
    else:
        print(f"{now} - No ambulance (score={score:.2f})")
        return False


try:
    print("Starting ambulance siren detection. Press Ctrl+C to stop.")
    while True:
        audio = sd.rec(int(DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype='float64')
        sd.wait()

        ambulance = detect_ambulance(audio, SAMPLE_RATE)

        if ambulance:
            # ✅ GREEN ON when siren detected
            GPIO.output(GREEN_PIN, GPIO.HIGH)
            GPIO.output(RED_PIN, GPIO.LOW)
        else:
            # ✅ RED ON otherwise
            GPIO.output(GREEN_PIN, GPIO.LOW)
            GPIO.output(RED_PIN, GPIO.HIGH)

        time.sleep(0.5)

except KeyboardInterrupt:
    print("Interrupted by user - cleaning up GPIO and exiting.")
finally:
    GPIO.cleanup()
