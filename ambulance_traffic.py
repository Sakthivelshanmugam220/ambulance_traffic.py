#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
import RPi.GPIO as GPIO
from scipy.signal import butter, lfilter

# -------- CONFIGURATION --------
SAMPLE_RATE = 16000         # Hz supported by your USB mic
CHUNK = 2048                # Samples per block (~0.13 s)
SIREN_BAND = (600, 1800)    # Hz – adjust if your local sirens differ
POWER_THRESH = 0.25         # Relative spectral power threshold
FRAMES_REQUIRED = 5         # Number of consecutive detections
DEVICE = 'hw:2,0'           # From `arecord -l`

GREEN = 17   # BCM for pin 11
RED = 27     # BCM for pin 13

# -------- SETUP GPIO --------
GPIO.setmode(GPIO.BCM)
GPIO.setup(GREEN, GPIO.OUT)
GPIO.setup(RED, GPIO.OUT)
GPIO.output(GREEN, True)
GPIO.output(RED, False)

# -------- FILTER HELPERS --------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

# -------- SIREN DETECTION --------
def detect_siren(block):
    # Band-pass filter
    filtered = bandpass_filter(block, SIREN_BAND[0], SIREN_BAND[1], SAMPLE_RATE)

    # FFT magnitude
    spectrum = np.abs(np.fft.rfft(filtered))
    band_power = np.sum(spectrum)
    total_power = np.sum(np.abs(np.fft.rfft(block))) + 1e-8

    # Relative energy in band
    ratio = band_power / total_power
    return ratio > POWER_THRESH

# -------- MAIN LOOP --------
print("Traffic signal active. Listening for ambulance siren... Press Ctrl+C to stop.")
detection_count = 0

try:
    with sd.InputStream(device=DEVICE, channels=1, samplerate=SAMPLE_RATE,
                        blocksize=CHUNK, dtype='float32') as stream:
        while True:
            block, _ = stream.read(CHUNK)
            block = block[:, 0]

            if detect_siren(block):
                detection_count += 1
            else:
                detection_count = max(detection_count-1, 0)

            if detection_count >= FRAMES_REQUIRED:
                # Switch to RED for ambulance priority
                GPIO.output(GREEN, False)
                GPIO.output(RED, True)
                print("Ambulance siren detected – priority given.")
            else:
                # Normal GREEN signal
                GPIO.output(GREEN, True)
                GPIO.output(RED, False)
except KeyboardInterrupt:
    print("Exiting and cleaning up GPIO...")
finally:
    GPIO.cleanup()

