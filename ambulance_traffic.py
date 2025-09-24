import numpy as np
import sounddevice as sd
import RPi.GPIO as GPIO
import time
from scipy.signal import get_window

# ---------------- CONFIGURATION ----------------
GREEN_PIN = 17   # BCM 17 -> physical pin 11
RED_PIN   = 27   # BCM 27 -> physical pin 13
SAMPLE_RATE = 16000       # Match your MATLAB resample rate
BLOCK_SIZE  = 2048        # Audio block size
OVERLAP     = 1024        # For STFT
SIREN_RANGE = (600, 1600) # Ambulance harmonic band (Hz)
POWER_THRESH = 0.4        # Relative band power threshold
FLUX_THRESH  = 0.08       # Spectral flux threshold for sweep pattern
HOLD_TIME    = 4.0        # Seconds to keep GREEN on after detection

# ---------------- GPIO SETUP -------------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.output(GREEN_PIN, GPIO.LOW)
GPIO.output(RED_PIN, GPIO.HIGH)

def spectral_flux(prev_mag, curr_mag):
    diff = curr_mag - prev_mag
    diff[diff < 0] = 0
    return np.sum(diff)

def detect_siren(block, prev_mag):
    window = get_window('hamming', BLOCK_SIZE)
    spectrum = np.fft.rfft(block * window)
    freqs = np.fft.rfftfreq(BLOCK_SIZE, 1/SAMPLE_RATE)
    mag = np.abs(spectrum)

    # Normalize to max magnitude
    mag /= np.max(mag) if np.max(mag) > 0 else 1

    # Band power in siren range
    band = (freqs >= SIREN_RANGE[0]) & (freqs <= SIREN_RANGE[1])
    band_power = np.mean(mag[band]) if np.any(band) else 0

    # Spectral flux for sweeping pattern
    flux = spectral_flux(prev_mag, mag)

    # Decision: strong band power + noticeable sweep
    siren_detected = band_power > POWER_THRESH and flux > FLUX_THRESH
    return siren_detected, mag

print("Listening for ambulance siren... Ctrl+C to exit.")

try:
    prev_mag = np.zeros(BLOCK_SIZE//2+1)
    last_detect = 0
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                        blocksize=BLOCK_SIZE, dtype='float32') as stream:
        while True:
            audio, _ = stream.read(BLOCK_SIZE)
            audio = audio.flatten()
            detected, prev_mag = detect_siren(audio, prev_mag)

            now = time.time()
            if detected:
                last_detect = now
                print("Ambulance siren detected! GREEN ON.")
            
            # Control LEDs: GREEN when siren detected recently
            if now - last_detect < HOLD_TIME:
                GPIO.output(RED_PIN, GPIO.LOW)
                GPIO.output(GREEN_PIN, GPIO.HIGH)
            else:
                GPIO.output(GREEN_PIN, GPIO.LOW)
                GPIO.output(RED_PIN, GPIO.HIGH)
            
            time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping detection.")
finally:
    GPIO.cleanup()
