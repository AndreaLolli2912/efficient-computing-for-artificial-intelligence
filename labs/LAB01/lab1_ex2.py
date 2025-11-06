"""
c.	Modify the script to let the user to disable/enable the audio storage by pressing P key.
"""


import os
import sounddevice as sd
from time import time
from scipy.io.wavfile import write


def callback(indata, frames, callback_time, status):
    """This is called (from a separate thread) for each audio block."""
    global store_audio

    if store_audio is True:
        timestamp = time()
        write(f'{timestamp}.wav', 48000, indata)
        filesize_in_bytes = os.path.getsize(f'{timestamp}.wav')
        filesize_in_kb = filesize_in_bytes / 1024
        print(f'Size: {filesize_in_kb:.2f}KB')


store_audio = True

with sd.InputStream(device=1, channels=1, dtype='int32', samplerate=48000, blocksize=48000, callback=callback):
    while True:
        key = input()
        if key in ('q', 'Q'):
            print('Stop recording.')
            break
        if key in ('p', 'P'):
            store_audio = not store_audio