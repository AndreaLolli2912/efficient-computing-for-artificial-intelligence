"""
src\ec4a1\sensors\mic_stream.py
"""
import os
from time import time

import argparse
import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd

store_audio = 0
audio_buffer = []
current_filename = None


def callback(indata, frames, callback_time, status):
    global store_audio, audio_buffer
    if store_audio: audio_buffer.append(indata.copy())

def save_audio(args: argparse.Namespace, config: dict):
    global audio_buffer, current_filename
    if len(audio_buffer) == 0: return -1

    audio_data = np.concatenate(audio_buffer, axis=0)

    output_dir = config["audio"]["base_dir"]
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, current_filename)
    write(audio_path, args.sampling_rate, audio_data)

    filesize = os.path.getsize(audio_path) / 1024
    print("Saved '%s' :%.2f KB\n"%(audio_path, filesize))

    audio_buffer = []
    return audio_path

def start_recording(args: argparse.Namespace, config: dict):

    global store_audio, current_filename
    device   = config["audio"]["device"]
    channels = config["audio"]["channels"]

    print("Recording configuration:\n- Bit depth: %s\n- Sampling rate: %d\n- Duration: %d seconds\n" \
    %(args.bit_depth, args.sampling_rate, args.duration))

    print("**** Press `P` to start/stop recording, `Q` to close application. ****\n")

    with sd.InputStream(
        device=device,
        channels=channels,
        dtype=args.bit_depth,
        samplerate=args.sampling_rate,
        blocksize=args.sampling_rate, # NOTE: is this the same?
        callback=callback
    ):
        while True:
            key = input().strip().lower()
            if key == "p":
                store_audio = not store_audio
                current_filename = "%f.wav"%time()
                sd.sleep(int(args.duration * 1000))
                audio_path = save_audio(args, config)
                store_audio = not store_audio
                return audio_path