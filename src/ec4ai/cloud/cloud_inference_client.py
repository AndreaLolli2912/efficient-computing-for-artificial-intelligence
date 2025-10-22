from src.ec4ai.utils.audio_utils import load_audio

import base64
from subprocess import Popen
from time import sleep, time

import argparse
import numpy as np
import requests
import torchaudio
import torchaudio.transforms as T

MAX_SAMPLING_RATE = 16_000

def cloud_inference_client(audio_path: str, api_url:str, n_trials:int = 20):
    # Fix the CPU frequency to its maximum value (1.5 GHz)
    Popen(
        'sudo sh -c "echo performance >'
        '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
        shell=True,
    ).wait()
    
    audio_data, sampling_rate = load_audio(audio_path, MAX_SAMPLING_RATE)
    encoded = base64.b64encode(audio_data.numpy().tobytes()).decode('utf-8')

    times = []

    for i in range(n_trials):
        start = time()
        try:
            response = requests.post(
            api_url, json={'input': encoded, 'sampling_rate': sampling_rate}
            )
            prediction = response.json()#['output']
            print(f'Prediction: {prediction}')
            times.append(time()- start)
        except requests.exceptions.ConnectionError as e:
            print(f"Connection failed: {e}")
        except requests.exceptions.HTTPError as e:
           print(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
           print(f"An error occurred: {e}")

        sleep(2)

    median_time = np.median(times)
    std_time = np.std(times)

    print(f'Latency: {median_time:.2f}+/-{std_time:.2f}s')