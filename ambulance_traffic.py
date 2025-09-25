import sounddevice as sd
import numpy as np
import RPi.GPIO as GPIO
import time
import datetime
from scipy.signal import butter, lfilter

# === Parameters ===
SAMPLE_RATE = 48000       # Supported USB mic rate
DURATION = 2.0            # Seconds per audio frame
THRESHOLD = 15.0          # Energy threshold for detection
SIREN_MIN = 600           # Min freq of ambulance siren (Hz)
SIREN_MAX = 1500          # Max freq of ambulance siren (Hz)

# === GPIO Setup ===
RED_LED = 17
GREEN_LED = 27
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_LED, GPIO.OUT)
GPIO.setup(GREEN_LED, GPIO.OUT)

# Default: Green ON, Red OFF
GPIO.output(RED_LED, GPIO.LOW)
GPIO.output(GREEN_LED, GPIO.HIGH)

# === Bandpass Filter (Butterworth) ===
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# === Siren Detection ===
def detect_siren(audio, fs):
    # Apply bandpass filter
    filtered = bandpass_filter(audio, SIREN_MIN, SIREN_MAX, fs)

    # FFT
    spectrum = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), 1/fs)

    # Energy in siren band
    mask = (freqs >= SIREN_MIN) & (freqs <= SIREN_MAX)
    energy = np.sum(spectrum[mask])

    return energy

# === Main Loop ===
try:
    print("🚦 Starting ambulance siren detection. Press Ctrl+C to stop.")

    while True:
        # Record from mic
        audio = sd.rec(int(DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1, dtype='float32')
        sd.wait()

        audio = np.squeeze(audio)  # 1D array

        score = detect_siren(audio, SAMPLE_RATE)

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        if score > THRESHOLD:
            print(f"{timestamp} - 🚨 Ambulance detected (score={score:.2f}) - RED ON")
            GPIO.output(RED_LED, GPIO.HIGH)
            GPIO.output(GREEN_LED, GPIO.LOW)
        else:
            print(f"{timestamp} - Normal traffic (score={score:.2f}) - GREEN ON")
            GPIO.output(RED_LED, GPIO.LOW)
            GPIO.output(GREEN_LED, GPIO.HIGH)

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user - cleaning up GPIO and exiting.")
    GPIO.cleanup()
