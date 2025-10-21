import base64
from subprocess import Popen
from time import sleep, time

import numpy as np
import requests
import torchaudio
import torchaudio.transforms as T

def cloud_inference_client(filename: str, n_trials=20):
    # Fix the CPU frequency to its maximum value (1.5 GHz)
    Popen(
        'sudo sh -c "echo performance >'
        '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
        shell=True,
    ).wait()

    API_URL = (
        'https://72be983c-f0af-4fa7-846a-0f079e293e88.deepnoteproject.com/predict'
    )
    x, sampling_rate = torchaudio.load(filename)
    
    if sampling_rate > 16_000:
        transform = T.Resample(sampling_rate, 16_000)
        x = transform(x).copy()
        sampling_rate = 16_000
    
    encoded = base64.b64encode(x.numpy().tobytes()).decode('utf-8')

    times = []

    for i in range(n_trials):
        start = time()
        try:
            response = requests.post(
            API_URL, json={'input': encoded, 'sampling_rate': sampling_rate}
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