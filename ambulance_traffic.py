
#!/usr/bin/env python3
"""
ambulance_traffic.py

Live ambulance-siren detection for Raspberry Pi using a USB mic (mono).
Green LED ON normally; Red LED ON when ambulance siren detected.

Dependencies (install in your venv):
pip install numpy scipy sounddevice RPi.GPIO
"""

import time
import collections
import numpy as np
import sounddevice as sd
from scipy.signal import wiener, stft
import RPi.GPIO as GPIO

# ---------------- CONFIGURATION ----------------
GREEN_PIN = 17              # BCM pin for GREEN (physical pin 11)
RED_PIN = 27                # BCM pin for RED   (physical pin 13)

MIC_DEVICE = "hw:2,0"       # Change if arecord -l shows different card/device
SAMPLE_RATE = 48000         # USB mic sample rate (use what arecord reported)
CHANNELS = 1                # mono
FRAME_SECONDS = 0.5         # process every 0.5 s (latency)
FRAME_LEN = int(SAMPLE_RATE * FRAME_SECONDS)

# STFT parameters (for spectral analysis)
NPERSEG = 2048
NOVERLAP = NPERSEG // 2
NFFT = 2048

SIREN_MIN_HZ = 600          # lower bound of siren band
SIREN_MAX_HZ = 1700         # upper bound of siren band

# Detection tuning
RATIO_THRESHOLD = 6.0       # siren_band_power / background_power ratio required
SMOOTH_WINDOW = 4           # number of recent decisions to smooth (majority voting)
HOLD_SECONDS = 3.0          # when detected, hold red for at least this long

# Safety / small constants
EPS = 1e-9

# ---------------- GPIO SETUP ----------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(RED_PIN, GPIO.OUT)

def set_normal():
    GPIO.output(GREEN_PIN, GPIO.HIGH)
    GPIO.output(RED_PIN, GPIO.LOW)

def set_emergency():
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(RED_PIN, GPIO.HIGH)

# initialize normal state
set_normal()

# ---------------- Helper functions ----------------
def process_frame(audio_frame):
    """
    audio_frame: 1D numpy array, length FRAME_LEN
    Returns: detection_score (float) — higher means more likely siren
    We compute STFT magnitude, sum energy in siren band and in background,
    then return ratio = siren_power / background_power.
    """
    # 1) Denoise a bit with Wiener filter (1D)
    clean = wiener(audio_frame, mysize=5)

    # 2) STFT
    f, t, Zxx = stft(clean, fs=SAMPLE_RATE, window='hamming',
                     nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, boundary=None)
    mag = np.abs(Zxx)

    # 3) Frequency bins for siren band
    siren_idx = np.where((f >= SIREN_MIN_HZ) & (f <= SIREN_MAX_HZ))[0]
    if siren_idx.size == 0:
        return 0.0

    # 4) Compute band powers; average over time frames
    siren_power = mag[siren_idx, :].sum()
    total_power = mag.sum()
    background_power = total_power - siren_power

    # normalize by band width to avoid bias
    siren_power_per_bin = siren_power / (len(siren_idx) + EPS)
    bg_power_per_bin = background_power / ( (mag.shape[0] - len(siren_idx)) + EPS )

    # final ratio
    ratio = (siren_power_per_bin + EPS) / (bg_power_per_bin + EPS)
    return ratio

def majority_vote(queue):
    """Return True if majority of last SMOOTH_WINDOW entries indicate detection."""
    if len(queue) < SMOOTH_WINDOW:
        # require at least some history
        return sum(queue) > (len(queue) / 2.0)
    return sum(queue) >= (SMOOTH_WINDOW // 2 + 1)

# ---------------- Main loop ----------------
def main():
    print("Starting ambulance siren detection. Press Ctrl+C to stop.")
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = CHANNELS

    # ring buffer to collect audio frames
    buf = np.zeros(0, dtype='float32')

    # history of recent boolean decisions for smoothing
    recent = collections.deque(maxlen=SMOOTH_WINDOW)

    last_emergency_time = 0.0

    try:
        with sd.InputStream(device=MIC_DEVICE, channels=CHANNELS, samplerate=SAMPLE_RATE, dtype='float32') as stream:
            while True:
                # read however many samples to fill one FRAME_LEN chunk
                needed = FRAME_LEN - buf.size
                if needed > 0:
                    data, overflow = stream.read(needed)
                    if CHANNELS > 1:
                        data = np.mean(data, axis=1)
                    else:
                        data = data.flatten()
                    buf = np.concatenate((buf, data))
                if buf.size >= FRAME_LEN:
                    frame = buf[:FRAME_LEN]
                    buf = buf[FRAME_LEN:]  # drop processed part

                    # normalize frame energy (prevent level variations from breaking ratio)
                    frame = frame - np.mean(frame)
                    max_abs = np.max(np.abs(frame)) + EPS
                    frame = frame / max_abs

                    # process
                    score = process_frame(frame)
                    is_siren = (score >= RATIO_THRESHOLD)

                    recent.append(1 if is_siren else 0)
                    smoothed = majority_vote(recent)

                    now = time.time()
                    if smoothed:
                        # set emergency, update timer
                        last_emergency_time = now
                        set_emergency()
                        print(f"{time.strftime('%H:%M:%S')} - Ambulance detected (score={score:.2f}) - RED ON")
                    else:
                        # if we are within hold period after last detection, keep emergency
                        if now - last_emergency_time < HOLD_SECONDS:
                            set_emergency()
                            # do not print repeatedly
                        else:
                            set_normal()
                    # small sleep to avoid busy-loop
                    time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nInterrupted by user - cleaning up GPIO and exiting.")
    except Exception as e:
        print("Error:", e)
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
