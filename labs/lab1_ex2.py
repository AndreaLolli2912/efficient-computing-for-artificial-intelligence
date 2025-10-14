import os
import sounddevice as sd
import numpy as np
from time import time
from scipy.io.wavfile import write

SAMPLERATE = 48_000  # 48 kHz sample rate
store_audio = False
audio_buffer = []  # we'll collect audio chunks here
current_filename = None


def callback(indata, frames, callback_time, status):
    """This is called automatically for each audio block."""
    global store_audio, audio_buffer

    if store_audio:
        # Copy the block (sounddevice reuses buffers, so we need to copy it)
        audio_buffer.append(indata.copy())


def save_audio():
    """Write all buffered audio to a single WAV file."""
    global audio_buffer, current_filename

    if len(audio_buffer) == 0:
        print("No audio data to save.")
        return

    # Stack all chunks into one continuous array
    audio_data = np.concatenate(audio_buffer, axis=0)

    # Save to disk
    write(current_filename, SAMPLERATE, audio_data)
    filesize = os.path.getsize(current_filename) / 1024
    print(f"Saved '{current_filename}' ({filesize:.2f} KB)")

    # Clear the buffer for next session
    audio_buffer = []


with sd.InputStream(
    device=1, channels=1, dtype='int16', samplerate=SAMPLERATE,
    blocksize=48000, callback=callback
):
    print("Press P to start/stop saving audio, Q to quit.")
    while True:
        key = input().strip().lower()

        if key == 'q':
            # If currently recording, save before quitting
            if store_audio:
                print("Stopping and saving before exit...")
                store_audio = False
                save_audio()
            print("Bye!")
            break

        elif key == 'p':
            # Toggle storage
            store_audio = not store_audio
            if store_audio:
                current_filename = f"{int(time())}.wav"
                print(f"Recording started → saving to {current_filename}")
            else:
                print("Recording stopped → saving to file...")
                save_audio()
